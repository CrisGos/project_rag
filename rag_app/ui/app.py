from __future__ import annotations
from pathlib import Path

import streamlit as st

from rag_app.config.settings import (
    PDF_DIR,
    logger,
    audit_event,
    ALLOW_OCR,
)
from rag_app.ui.auth import login
from rag_app.ingestion.build_index import build_vectorstore, verify_index_for_pdf
from rag_app.rag.chain import make_rag_chain

st.set_page_config(page_title="RAG – Aeronautical Manual (Weaviate + Ollama)", layout="wide")
st.title("RAG – Aeronautical Manual (Weaviate + Ollama)")

# --- Auth ---
ok, name, username = login()
if not ok:
    st.stop()
st.sidebar.success(f"Signed in as: {name}")
if st.sidebar.button("Logout"):
    for k in list(st.session_state.keys()):
        if k.startswith("auth"):
            del st.session_state[k]
    st.rerun()

# --- Sidebar: PDF selection ---
st.sidebar.header("Manual")
uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"], key="upload_pdf")
existing_pdfs = sorted([p.name for p in Path(PDF_DIR).glob("*.pdf")])
choice = st.sidebar.selectbox("Choose existing PDF", ["(none)"] + existing_pdfs, key="pdf_choice")

pdf_path: str | None = None
if uploaded is not None:
    dest = Path(PDF_DIR) / uploaded.name
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    pdf_path = str(dest)
    st.sidebar.success(f"Saved: {dest.name}")
elif choice != "(none)":
    pdf_path = str(Path(PDF_DIR) / choice)

# OCR toggle
ocr_enabled = False
if ALLOW_OCR:
    content_type = st.sidebar.radio(
        "Content type",
        options=["Selectable text", "Scanned (OCR)"],
        index=0,
        key="pdf_content_type",
    )
    ocr_enabled = content_type == "Scanned (OCR)"
else:
    st.sidebar.caption("OCR disabled by configuration.")

col_b1, col_b2 = st.sidebar.columns(2)
do_build = col_b1.button("Build/Update index", type="primary", use_container_width=True)
do_verify = col_b2.button("Verify index", use_container_width=True)

status = st.empty()

active_pdf_name = Path(pdf_path).name if pdf_path else None
session_key = f"chat_{active_pdf_name}" if active_pdf_name else "chat_(none)"
if session_key not in st.session_state:
    st.session_state[session_key] = []

if do_build:
    if not pdf_path:
        st.warning("Select or upload a PDF before building the index.")
    else:
        try:
            idx_id = build_vectorstore(pdf_path, use_ocr=ocr_enabled)
            status.success(f"Index built for: {idx_id}")
        except Exception as e:
            status.error(f"Index build error: {e}")

is_indexed = False
if pdf_path:
    try:
        is_indexed = verify_index_for_pdf(pdf_path)
        if do_verify:
            if is_indexed:
                status.success("Index verified: vectors found for this PDF.")
            else:
                status.warning("No vectors found for this PDF.")
    except Exception as e:
        if do_verify:
            status.error(f"Index verification error: {e}")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Manual status")
    if not pdf_path:
        st.info("Upload or choose a PDF to start.")
    else:
        st.markdown(f"**Selected PDF:** `{Path(pdf_path).name}`")
        st.markdown(f"**OCR:** `{ocr_enabled}`")
        st.markdown(f"**Indexed:** `{is_indexed}`")

    st.subheader("Retrieval parameters")
    k = st.slider("Top-k chunks", 1, 8, 4, key="topk_slider")
    show_sources = st.checkbox("Show retrieved chunks", value=False, key="show_src_chk")

with right:
    st.subheader("Chat")
    rag = make_rag_chain()

    question = st.text_input(
        "Ask something that should be answered from the manual:",
        "",
        key="chat_q_input",
        placeholder="e.g., What is the maximum crosswind component for landing?",
    )
    ask = st.button("Ask", type="secondary", key="chat_ask_btn")

    if ask:
        if not pdf_path:
            st.warning("Select or upload a PDF first.")
        else:
            with st.spinner("Retrieving + generating…"):
                try:
                    out = rag(question, pdf_name=Path(pdf_path).name, k=k)
                    st.markdown("### Answer")
                    st.write(out["answer"])

                    st.session_state[session_key].append(
                        {"q": question, "a": out["answer"], "src": out["source_docs"]}
                    )

                    if show_sources:
                        st.markdown("### Retrieved chunks")
                        for i, d in enumerate(out["source_docs"], 1):
                            st.markdown(
                                f"**[{i}] page {d.get('page_number')}** (chunk {d.get('chunk_id')})"
                            )
                            st.code((d.get("text") or "")[:2000])

                    audit_event(
                        "qa",
                        {
                            "user": username,
                            "pdf": Path(pdf_path).name,
                            "q": question,
                            "k": k,
                            "indexed": is_indexed,
                        },
                    )
                except Exception as e:
                    st.error(f"RAG error: {e}")

    with st.expander("Chat history"):
        for turn in st.session_state[session_key]:
            st.markdown(f"**Q:** {turn['q']}")
            st.markdown(f"**A:** {turn['a']}")
