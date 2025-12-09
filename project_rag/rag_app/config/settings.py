# Settings.py
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import requests

# ========= Project paths =========
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
LOG_DIR: Path = DATA_DIR / "logs"
PDF_DIR: Path = DATA_DIR / "pdfs"

for p in (DATA_DIR, LOG_DIR, PDF_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ========= Logging =========
LOG_FILE = LOG_DIR / "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("RAG-App")


def audit_event(event: str, payload: dict | None = None) -> None:
    try:
        line = {"event": event, "payload": payload or {}}
        with (LOG_DIR / "interactions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    except Exception as e:
        logger.warning("audit_event failed: %s", e)

# ========= LLM & Embeddings (Ollama local) =========
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

# ========= Reply Language =========
REPLY_LANGUAGE = st.secrets.get("REPLY_LANGUAGE", os.getenv("REPLY_LANGUAGE", "en"))

# ========= Weaviate (HTTP only) =========
WEAVIATE_URL: str = st.secrets.get("WEAVIATE_URL", os.getenv("WEAVIATE_URL", "http://127.0.0.1:8080")).rstrip("/")
WEAVIATE_CLASS: str = os.getenv("WEAVIATE_CLASS", "RagChunk")

def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")

ALLOW_OCR = _bool_env("ALLOW_OCR", True)

# ========== Langfuse Observability ==========
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
LANGFUSE_ENABLED: bool = _bool_env("LANGFUSE_ENABLED", False) 

# ---------- HTTP helpers ----------
def weaviate_ready() -> bool:
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/.well-known/ready", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ensure_weaviate_schema_http() -> None:
    """
    Create class if not exists using REST /v1/schema.
    Class is configured with manual vectors (no vectorizer).
    """
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=10)
        r.raise_for_status()
        schema = r.json()
        classes = [c.get("class") for c in schema.get("classes", [])]
    except Exception as e:
        logger.warning("Failed to read schema: %s", e)
        classes = []

    if WEAVIATE_CLASS in classes:
        logger.info("Schema class %s already exists", WEAVIATE_CLASS)
        return

    payload = {
        "class": WEAVIATE_CLASS,
        "description": "RAG chunks (manual vectors)",
        "vectorizer": "none",
        "properties": [
            {"name": "pdf_name", "dataType": ["text"]},
            {"name": "page_number", "dataType": ["int"]},
            {"name": "chunk_id", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"]},
        ],
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    try:
        r = requests.post(
            f"{WEAVIATE_URL}/v1/schema", 
            json=payload, 
            headers=headers,
            timeout=20
        )
        r.raise_for_status()
        logger.info("Created Weaviate class %s", WEAVIATE_CLASS)
    except requests.exceptions.HTTPError as e:
        logger.error("Failed creating class: HTTP %s - %s", e.response.status_code, e.response.text)
        raise
    except Exception as e:
        logger.error("Failed creating class: %s", e)
        raise


def weaviate_batch_upsert(objects: List[Dict[str, Any]]) -> None:
    """
    REST batch upsert to /v1/batch/objects with manual vectors.
    Each object: {"class": ..., "properties": {...}, "vector": [...]}.
    """
    payload = {"objects": objects}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    r = requests.post(
        f"{WEAVIATE_URL}/v1/batch/objects", 
        json=payload, 
        headers=headers,
        timeout=60
    )
    if r.status_code >= 300:
        raise RuntimeError(f"Weaviate batch upsert failed: {r.status_code} {r.text}")


def weaviate_graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /v1/graphql
    """
    body = {"query": query, "variables": variables}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    r = requests.post(
        f"{WEAVIATE_URL}/v1/graphql", 
        json=body, 
        headers=headers,
        timeout=30
    )
    if r.status_code >= 300:
        raise RuntimeError(f"GraphQL HTTP error {r.status_code}: {r.text}")
    out = r.json()
    if "errors" in out:
        raise RuntimeError(f"GraphQL errors: {out['errors']}")
    return out