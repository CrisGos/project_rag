# Settings.py
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st


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
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace("localhost", "127.0.0.1")
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

# ========= Reply Language =========
REPLY_LANGUAGE = st.secrets.get("REPLY_LANGUAGE", os.getenv("REPLY_LANGUAGE", "en"))

# ========= Weaviate (v4 Client) =========
import weaviate
from weaviate.classes.config import Property, DataType, Configure

# Helper to parse URL, e.g. "http://weaviate:8080" -> host="weaviate", port=8080
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

# ========== Tavily Search ==========
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")


def get_weaviate_client():
    from urllib.parse import urlparse
    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    
    # Smart fallback: If we can't resolve 'weaviate', we are likely running locally outside the container.
    # We fallback to localhost.
    if host == "weaviate":
        import socket
        try:
            socket.gethostbyname(host)
        except socket.gaierror:
            logger.warning("Could not resolve 'weaviate' hostname. Falling back to '127.0.0.1' for local execution.")
            host = "127.0.0.1"
    
    # Attempt to connect via gRPC if possible, assuming 50051 is standard.
    # If running in docker compose, "weaviate" host should respond on 50051.
    # If running locally, localhost:50051 should work if exposed.
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=parsed.scheme == "https",
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=parsed.scheme == "https",
    )

def ensure_weaviate_schema() -> None:
    """
    Ensure the Weaviate class exists with the correct schema using v4 client.
    """
    try:
        with get_weaviate_client() as client:
            if client.collections.exists(WEAVIATE_CLASS):
                return
            
            logger.info(f"Creating Weaviate class {WEAVIATE_CLASS}...")
            client.collections.create(
                name=WEAVIATE_CLASS,
                description="RAG chunks (manual vectors)",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="pdf_name", data_type=DataType.TEXT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                ]
            )
            logger.info(f"Created Weaviate class {WEAVIATE_CLASS}")
    except Exception as e:
        logger.error(f"Failed to ensure schema: {e}")
        raise
