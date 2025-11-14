from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

from rag_app.config.settings import (
    PDF_DIR,
    logger,
    ensure_weaviate_schema_http,
    weaviate_ready,
    weaviate_graphql,
    WEAVIATE_CLASS,
    OLLAMA_EMBED_MODEL,
)


# ---------- PDF loading ----------
def load_pdf_text(pdf_path: str, use_ocr: bool) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number, text)
    """
    pages: List[Tuple[int, str]] = []
    if not use_ocr:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i, text))
    else:
        # OCR path; requires poppler and tesseract installed on the system
        images = convert_from_path(pdf_path, dpi=200)
        for i, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img) or ""
            text = text.strip()
            if text:
                pages.append((i, text))
    return pages


def split_pages_to_chunks(pages: List[Tuple[int, str]]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=160)
    chunks: List[dict] = []
    for page_no, text in pages:
        for idx, chunk in enumerate(splitter.split_text(text)):
            chunks.append(
                {
                    "page_number": int(page_no),
                    "chunk_id": f"{page_no}-{idx}",
                    "text": chunk,
                }
            )
    return chunks


# ---------- Weaviate helpers (HTTP-only, manual vectors) ----------
def _batch_upsert_http(pdf_name: str, chunks: List[dict]) -> None:
    """
    Upsert using /v1/batch/objects via settings.weaviate_batch_upsert() logic.
    Here we expand inline to avoid circular imports.
    """
    from rag_app.config.settings import weaviate_batch_upsert  # local import

    embed = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    vectors = embed.embed_documents([c["text"] for c in chunks])

    objects = []
    for c, v in zip(chunks, vectors):
        objects.append(
            {
                "class": WEAVIATE_CLASS,
                "properties": {
                    "pdf_name": pdf_name,
                    "page_number": int(c["page_number"]),
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                },
                "vector": v,
            }
        )
    weaviate_batch_upsert(objects)


def verify_index_for_pdf(pdf_path: str) -> bool:
    """
    Existence check via GraphQL Aggregate.meta.count. Inline the where block
    (no variables) to avoid input-type mismatches.
    """
    if not weaviate_ready():
        raise RuntimeError("Weaviate not ready on /v1/.well-known/ready")

    ensure_weaviate_schema_http()

    pdf_name = Path(pdf_path).name
    q = f"""
{{
  Aggregate {{
    {WEAVIATE_CLASS}(
      where: {{
        operator: Equal
        path: ["pdf_name"]
        valueText: {json.dumps(pdf_name)}
      }}
    ) {{
      meta {{ count }}
    }}
  }}
}}
"""
    resp = weaviate_graphql(q, {})
    count = resp["data"]["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
    return int(count) > 0


# ---------- Orchestration ----------
def build_vectorstore(pdf_path: str, use_ocr: bool = False) -> str:
    if not weaviate_ready():
        raise RuntimeError("Weaviate not ready on /v1/.well-known/ready")

    ensure_weaviate_schema_http()

    logger.info("Building index for %s (OCR=%s)", pdf_path, use_ocr)
    pages = load_pdf_text(pdf_path, use_ocr=use_ocr)
    if not pages:
        raise RuntimeError("No text extracted from PDF (check OCR toggle and dependencies).")

    chunks = split_pages_to_chunks(pages)
    pdf_name = Path(pdf_path).name
    _batch_upsert_http(pdf_name, chunks)
    logger.info("Index built for %s", pdf_name)
    return pdf_name
