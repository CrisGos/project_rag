# build_index.py
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
    ensure_weaviate_schema,
    get_weaviate_client,
    WEAVIATE_CLASS,
    OLLAMA_EMBED_MODEL,
)
from weaviate.classes.query import Filter


# ---------- PDF loading ----------
def load_pdf_text(pdf_path: str, use_ocr: bool) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number, text)
    """
    pages: List[Tuple[int, str]] = []
    if not use_ocr:
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append((i, text))
        except Exception as e:
            logger.error(f"PDF reading failed: {e}")
            raise
    else:
        # OCR path; requires poppler and tesseract installed on the system
        try:
            images = convert_from_path(pdf_path, dpi=200)
            for i, img in enumerate(images, start=1):
                text = pytesseract.image_to_string(img) or ""
                text = text.strip()
                if text:
                    pages.append((i, text))
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise
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


# ---------- Weaviate helpers (v4 Client) ----------
def _batch_upsert(pdf_name: str, chunks: List[dict]) -> None:
    """
    Upsert using client.batch.dynamic().
    """
    embed = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    try:
        vectors = embed.embed_documents([c["text"] for c in chunks])
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

    try:
        with get_weaviate_client() as client:
            collection = client.collections.get(WEAVIATE_CLASS)
            logger.info("Starting batch upload of %d chunks...", len(chunks))
            
            with collection.batch.dynamic() as batch:
                for c, v in zip(chunks, vectors):
                    batch.add_object(
                        properties={
                            "pdf_name": pdf_name,
                            "page_number": int(c["page_number"]),
                            "chunk_id": c["chunk_id"],
                            "text": c["text"],
                        },
                        vector=v
                    )
            
            # Check for failed objects
            if len(client.batch.failed_objects) > 0:
                logger.error(f"Batch upload had {len(client.batch.failed_objects)} failures.")
                for fail in client.batch.failed_objects[:5]:
                    logger.error(f"Failure: {fail.message}")
                raise RuntimeError("Weaviate batch upsert had failures.")
                
            logger.info("Batch upload completed.")
            
    except Exception as e:
        logger.error(f"Batch upsert failed: {e}")
        raise


def verify_index_for_pdf(pdf_path: str) -> bool:
    """
    Check if chunks exist for this PDF using aggregation.
    """
    ensure_weaviate_schema()
    pdf_name = Path(pdf_path).name
    
    try:
        with get_weaviate_client() as client:
            if not client.collections.exists(WEAVIATE_CLASS):
                return False
                
            collection = client.collections.get(WEAVIATE_CLASS)
            count_res = collection.aggregate.over_all(
                filters=Filter.by_property("pdf_name").equal(pdf_name),
                total_count=True
            )
            return count_res.total_count > 0
    except Exception as e:
        logger.warning(f"Index verification failed: {e}")
        return False


# ---------- Orchestration ----------
def build_vectorstore(pdf_path: str, use_ocr: bool = False) -> str:
    ensure_weaviate_schema()

    logger.info("Building index for %s (OCR=%s)", pdf_path, use_ocr)
    pages = load_pdf_text(pdf_path, use_ocr=use_ocr)
    if not pages:
        raise RuntimeError("No text extracted from PDF (check OCR toggle and dependencies).")

    chunks = split_pages_to_chunks(pages)
    pdf_name = Path(pdf_path).name
    _batch_upsert(pdf_name, chunks)
    logger.info("Index built for %s", pdf_name)
    return pdf_name

