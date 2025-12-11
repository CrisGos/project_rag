# rag_app/ui/chat.py
import chainlit as cl
from pathlib import Path
import shutil
import re

from rag_app.config.settings import PDF_DIR, ALLOW_OCR
from rag_app.ingestion.build_index import build_vectorstore, verify_index_for_pdf
from rag_app.rag.chain import make_rag_chain

# Ensure PDF_DIR exists
Path(PDF_DIR).mkdir(parents=True, exist_ok=True)

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the user session:
    1. Show welcome message.
    2. Ask to select an existing PDF or upload a new one.
    """
    cl.user_session.set("pdf_name", None)
    cl.user_session.set("rag_chain", None)

    await cl.Message(content="Welcome to the RAG Aeronautical Manual Assistant! ✈️").send()

    # List existing PDFs
    existing_pdfs = sorted([p.name for p in Path(PDF_DIR).glob("*.pdf")])
    actions = []
    
    if existing_pdfs:
        for p in existing_pdfs:
            # Use 'select_...' prefix to identify this action type
            actions.append(cl.Action(name="select_pdf", value=p, payload={"value": p}, label=f"📚 {p}"))
    
    actions.append(cl.Action(name="global_search", value="global", payload={"value": "global"}, label="🌍 Global Search (All Documents)"))
    actions.append(cl.Action(name="upload_pdf", value="new", payload={"value": "new"}, label="➕ Upload new PDF"))

    res = await cl.AskActionMessage(
        content="Please select a manual to chat with, or upload a new one:",
        actions=actions,
        timeout=600  # 10 minutes wait
    ).send()

    if res and res.get("name") == "select_pdf":
        pdf_name = res.get("payload", {}).get("value")
        await setup_pdf(pdf_name)
    
    elif res and res.get("name") == "global_search":
        await setup_pdf(None)

    elif res and res.get("name") == "upload_pdf":
        await handle_file_upload()

async def handle_file_upload():
    files = await cl.AskFileMessage(
        content="Please upload a PDF file:",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=600
    ).send()

    if files:
        file = files[0]
        dest_path = Path(PDF_DIR) / file.name
        
        # Save file
        with open(dest_path, "wb") as f:
            with open(file.path, "rb") as source:
                shutil.copyfileobj(source, f)
        
        # Ask for OCR if enabled
        use_ocr = False
        if ALLOW_OCR:
            res_ocr = await cl.AskActionMessage(
                content="Enable OCR for this document? (Choose 'Yes' if it is a scanned document)",
                actions=[
                    cl.Action(name="ocr", value="yes", payload={"value": "yes"}, label="Yes"),
                    cl.Action(name="ocr", value="no", payload={"value": "no"}, label="No")
                ]
            ).send()
            if res_ocr and res_ocr.get("payload", {}).get("value") == "yes":
                use_ocr = True

        msg = cl.Message(content=f"Building index for `{file.name}`... (OCR={use_ocr})")
        await msg.send()
        
        # Build index (this might be slow, so we wrap it or just run it)
        # Assuming build_vectorstore is synchronous
        try:
            await cl.make_async(build_vectorstore)(str(dest_path), use_ocr=use_ocr)
            msg.content = f"✅ Index built successfully for `{file.name}`!"
            await msg.update()
            await setup_pdf(file.name)
        except Exception as e:
            msg.content = f"❌ Error building index: {e}"
            await msg.update()

async def setup_pdf(pdf_name: str | None):
    """
    Verify index and setup session.
    """
    if pdf_name:
        pdf_path = Path(PDF_DIR) / pdf_name
        cl.user_session.set("pdf_name", pdf_name)
        
        # Verify index
        msg = cl.Message(content=f"Verifying index for `{pdf_name}`...")
        await msg.send()
        
        try:
            is_indexed = await cl.make_async(verify_index_for_pdf)(str(pdf_path))
            if is_indexed:
                msg.content = f"✅ Ready to chat with `{pdf_name}`!"
                await msg.update()
                
                # Init RAG chain (reusable)
                rag = make_rag_chain()
                cl.user_session.set("rag_chain", rag)
            else:
                msg.content = f"⚠️ Index not found for `{pdf_name}`. Please trigger a rebuild via re-upload (or implement a rebuild action)."
                await msg.update()
        except Exception as e:
            msg.content = f"❌ Error verifying index: {e}"
            await msg.update()
    else:
        # Global search setup
        cl.user_session.set("pdf_name", "GLOBAL_SEARCH_SENTINEL") # Use a sentinel or just handle None? 
        # Requirement says: "if flow requires pdf_name... ask explicitly". 
        # But here we are EXPLICITLY choosing global.
        # Let's treat None as Global in `main` loop logic.
        cl.user_session.set("pdf_name", None)
        rag = make_rag_chain()
        cl.user_session.set("rag_chain", rag)
        await cl.Message(content="✅ Ready to search across ALL documents!").send()


def _detect_aircraft_name(content: str) -> bool:
    """
    Simple heuristic to check if the user message contains a known aircraft name.
    """
    try:
        # 1. Get all PDF names (stems)
        pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
        # 2. Extract potential aircraft tokens.
        tokens = set()
        for p in pdf_files:
            stem = p.stem
            # Improved tokenization: split by any non-alphanumeric char
            parts = re.split(r'[^a-zA-Z0-9]', stem)
            for part in parts:
                # Filter noise (e.g. "AC", "II", "POH" might be kept if length >= 3)
                # "A320" -> length 4 -> kept
                # "0624" -> length 4 -> kept (maybe specific enough)
                if len(part) >= 3:
                     tokens.add(part.lower())
            
        # 3. Check if any token is present in the user content
        content_lower = content.lower()
        for token in tokens:
            if token in content_lower:
                return True
        
        # 4. Fallback: Check for explicit "Name: Query" format for unknown aircraft
        # Regex: Start of line, Word-like (>=2 chars), Colon
        if re.match(r'^\s*[\w\-.]{2,}\s*:', content):
            return True

        return False
        
    except Exception:
        # Fail safe
        return False



@cl.on_message
async def main(message: cl.Message):
    rag = cl.user_session.get("rag_chain")
    # check if we have a set pdf_name (could be None for global, or a string)
    # But wait, user_session.get("key") returning None could mean "not set" OR "set to None".
    # Chainlit user_session behavior: if key doesn't exist, returns None. 
    # Valid states for "pdf_name":
    # 1. String "foo.pdf" -> Filtered
    # 2. None (if we explicitly set it to None for global) -> Global
    # 3. Key missing or never set? -> Should prompt.
    
    # Let's inspect how initialized. on_chat_start sets it to None initially.
    # So we need a way to distinguish "Not selected yet" vs "Global Mode".
    # I will use a string "ALL" or similar for Global in session, but map to None for backend?
    # Or just check if rag_chain is initialized. rag_chain is only set AFTER setup_pdf.
    
    if not rag:
         # Initial setup handling if user bypassed on_chat_start or something weird, but standard flow assumes prompt there.
         # However, requirements say: "If the UI needs a document... and user hasn't specified... ask explicitly"
         # If rag is None, it means we haven't finished setup.
         
         # List existing PDFs for the prompt
         existing_pdfs = sorted([p.name for p in Path(PDF_DIR).glob("*.pdf")])
         actions = [cl.Action(name="select_pdf", value=p, payload={"value": p}, label=f"📚 {p}") for p in existing_pdfs]
         actions.append(cl.Action(name="global_search", value="global", payload={"value": "global"}, label="🌍 Global Search (All Documents)"))
         
         res = await cl.AskActionMessage(
            content="⚠️ Document context not set. Which document do you want to query?",
            actions=actions,
            timeout=120
         ).send()
         
         if res and res.get("name") == "select_pdf":
             pdf_name = res.get("payload", {}).get("value")
             await setup_pdf(pdf_name)
             # Recursively call main? Or just wait for next message?
             # Better to just run the query now that we have context.
             # We need to re-fetch rag and pdf_name
             rag = cl.user_session.get("rag_chain")
             # pdf_name will be fetched below
         elif res and res.get("name") == "global_search":
             await setup_pdf(None)
             rag = cl.user_session.get("rag_chain")
         else:
             await cl.Message("❌ No selection made. Please try again.").send()
             return

    # Now we should have rag ready.
    # pdf_name in session: None (Global) or String (Filtered)
    pdf_name = cl.user_session.get("pdf_name")

    # GUARDRAIL: If Global Search (pdf_name is None), require aircraft name in query.
    if pdf_name is None:
        if not _detect_aircraft_name(message.content):
             await cl.Message(
                 content="✈️ **Global Search Guardrail**\n\n"
                         "To use Global Search (searching across ALL manuals), you must explicitly mention the aircraft name in your question.\n\n"
                         "**Example:** 'A320: What is the maximum operating altitude?'"
             ).send()
             return

    # Call RAG
    msg = cl.Message(content="")
    await msg.send()
    
    # Retrieve & Generate
    # make_rag_chain returns a sync callable: run(question, pdf_name, k)
    # We'll run it in a thread
    try:
        resp = await cl.make_async(rag)(message.content, pdf_name=pdf_name, k=4)
        
        answer = resp.get("answer", "No answer generated.")
        is_external = resp.get("is_external", False)
        external_sources = resp.get("external_sources", [])
        source_docs = resp.get("source_docs", [])
        
        msg.content = answer
        
        # Append sources
        if is_external and external_sources:
             elements = []
             # Create simple text list or links for external sources
             source_list = "\n".join([f"- {s}" for s in external_sources])
             msg.content += f"\n\n**External Sources:**\n{source_list}"
             
        elif source_docs:
            elements = []
            for i, doc in enumerate(source_docs, 1):
                page = doc.get('page_number', '?')
                text = doc.get('text', '')[:500] + "..."
                doc_name = doc.get('pdf_name', pdf_name or "Unknown")
                source_name = f"{doc_name} (p.{page})"
                elements.append(
                    cl.Text(name=source_name, content=text, display="inline")
                )
            msg.elements = elements
            msg.content += "\n\n**Sources:**"  # Label for UI clarity
        
        await msg.update()
        
    except Exception as e:
        await cl.Message(content=f"❌ An error occurred: {e}").send()
