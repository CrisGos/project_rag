# rag_app/ui/chat.py
import chainlit as cl
from pathlib import Path
import shutil

from rag_app.config.settings import PDF_DIR, ALLOW_OCR, ENABLE_WEB_SEARCH
from rag_app.ingestion.build_index import build_vectorstore, verify_index_for_pdf
from rag_app.rag.chain import make_rag_chain
from rag_app.agents.search_agent import SearchAgent

# Ensure PDF_DIR exists
Path(PDF_DIR).mkdir(parents=True, exist_ok=True)

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the user session:
    1. Show welcome message.
    2. Ask to select an existing PDF or upload a new one.
    3. Initialize search agent if enabled.
    """
    cl.user_session.set("pdf_name", None)
    cl.user_session.set("rag_chain", None)
    
    # Initialize search agent
    if ENABLE_WEB_SEARCH:
        try:
            search_agent = SearchAgent()
            cl.user_session.set("search_agent", search_agent)
        except Exception as e:
            print(f"Warning: Could not initialize search agent: {e}")
            cl.user_session.set("search_agent", None)
    else:
        cl.user_session.set("search_agent", None)

    welcome_msg = "Welcome to the RAG Aeronautical Manual Assistant! ✈️"
    if ENABLE_WEB_SEARCH:
        welcome_msg += "\n\n🌐 **Web Search Enabled**: I can also search the web for current information!"
    
    await cl.Message(content=welcome_msg).send()

    # List existing PDFs
    existing_pdfs = sorted([p.name for p in Path(PDF_DIR).glob("*.pdf")])
    actions = []
    
    if existing_pdfs:
        for p in existing_pdfs:
            actions.append(cl.Action(name="select_pdf", value=p, payload={"value": p}, label=f"📚 {p}"))
    
    actions.append(cl.Action(name="upload_pdf", value="new", payload={"value": "new"}, label="➕ Upload new PDF"))

    res = await cl.AskActionMessage(
        content="Please select a manual to chat with, or upload a new one:",
        actions=actions,
        timeout=600
    ).send()

    if res and res.get("name") == "select_pdf":
        pdf_name = res.get("payload", {}).get("value")
        await setup_pdf(pdf_name)
    
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
        
        try:
            await cl.make_async(build_vectorstore)(str(dest_path), use_ocr=use_ocr)
            msg.content = f"✅ Index built successfully for `{file.name}`!"
            await msg.update()
            await setup_pdf(file.name)
        except Exception as e:
            msg.content = f"❌ Error building index: {e}"
            await msg.update()

async def setup_pdf(pdf_name: str):
    """Verify index and setup session."""
    pdf_path = Path(PDF_DIR) / pdf_name
    cl.user_session.set("pdf_name", pdf_name)
    
    msg = cl.Message(content=f"Verifying index for `{pdf_name}`...")
    await msg.send()
    
    try:
        is_indexed = await cl.make_async(verify_index_for_pdf)(str(pdf_path))
        if is_indexed:
            msg.content = f"✅ Ready to chat with `{pdf_name}`!"
            await msg.update()
            
            # Init RAG chain
            rag = make_rag_chain()
            cl.user_session.set("rag_chain", rag)
        else:
            msg.content = f"⚠️ Index not found for `{pdf_name}`. Please trigger a rebuild via re-upload."
            await msg.update()
    except Exception as e:
        msg.content = f"❌ Error verifying index: {e}"
        await msg.update()


@cl.on_message
async def main(message: cl.Message):
    rag = cl.user_session.get("rag_chain")
    pdf_name = cl.user_session.get("pdf_name")
    search_agent = cl.user_session.get("search_agent")

    if not rag or not pdf_name:
        await cl.Message("⚠️ Please select a PDF first by restarting the chat (Refresh page).").send()
        return

    # Show thinking message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Get RAG response first
        resp = await cl.make_async(rag)(message.content, pdf_name=pdf_name, k=4)
        
        answer = resp.get("answer", "No answer generated.")
        source_docs = resp.get("source_docs", [])
        
        # Enhance with web search if agent is available
        if search_agent:
            enhanced = search_agent.enhance_with_web_search(
                query=message.content,
                rag_answer=answer,
                rag_sources=source_docs
            )
            
            answer = enhanced["answer"]
            used_web = enhanced["used_web_search"]
            web_results = enhanced["web_results"]
            
            # Add web search indicator
            if used_web and web_results:
                answer = f"🌐 *Enhanced with web search*\n\n{answer}"
        
        msg.content = answer
        
        # Append document sources
        if source_docs:
            elements = []
            for i, doc in enumerate(source_docs, 1):
                page = doc.get('page_number', '?')
                text = doc.get('text', '')[:500] + "..."
                source_name = f"📄 Page {page}"
                elements.append(
                    cl.Text(name=source_name, content=text, display="inline")
                )
            msg.elements = elements
            msg.content += "\n\n**📚 Document Sources:**"
        
        await msg.update()
        
    except Exception as e:
        await cl.Message(content=f"❌ An error occurred: {e}").send()


# Optional: Add manual web search action
@cl.action_callback("manual_web_search")
async def on_manual_search(action):
    """Manual web search trigger"""
    search_agent = cl.user_session.get("search_agent")
    
    if not search_agent:
        await cl.Message("⚠️ Web search is not enabled.").send()
        return
    
    query = action.value
    web_context = search_agent.web_search.get_context(query, max_results=5)
    
    await cl.Message(content=web_context).send()