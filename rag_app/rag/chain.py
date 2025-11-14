from __future__ import annotations

import json
from typing import Dict, List

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from rag_app.config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    REPLY_LANGUAGE,
    weaviate_graphql,
    WEAVIATE_CLASS,
)


def _system_prompt() -> str:
    return (
        f"You are a concise assistant. Answer ONLY using the manual content provided "
        f"in the retrieved chunks. If the answer is not present, say you don't know. "
        f"Reply language: {REPLY_LANGUAGE}. Keep answers short and cite page number(s) "
        f"in parentheses."
    )


def retrieve_weaviate(query: str, pdf_name: str, k: int = 4) -> List[dict]:
    """
    Use Weaviate GraphQL Get with inline where + nearVector, no variables.
    """
    embed = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    qvec = embed.embed_query(query)

    gql = f"""
{{
  Get {{
    {WEAVIATE_CLASS}(
      where: {{
        operator: Equal
        path: ["pdf_name"]
        valueText: {json.dumps(pdf_name)}
      }}
      nearVector: {{
        vector: {json.dumps(qvec)}
      }}
      limit: {int(k)}
    ) {{
      text
      page_number
      chunk_id
      _additional {{ distance }}
    }}
  }}
}}
"""
    resp = weaviate_graphql(gql, {})
    objs = resp["data"]["Get"][WEAVIATE_CLASS]

    out: List[dict] = []
    for o in objs:
        out.append(
            {
                "text": o.get("text", ""),
                "page_number": o.get("page_number"),
                "chunk_id": o.get("chunk_id"),
                "distance": o.get("_additional", {}).get("distance"),
            }
        )
    return out


def make_rag_chain():
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_CHAT_MODEL,
        temperature=0.2,
    )

    def run(question: str, pdf_name: str, k: int = 4) -> Dict[str, any]:
        docs = retrieve_weaviate(question, pdf_name=pdf_name, k=k)
        context = "\n\n".join([f"[p.{d['page_number']}] {d['text']}" for d in docs])
        msgs = [
            ("system", _system_prompt()),
            ("human", f"Question: {question}\n\nContext:\n{context}"),
        ]
        resp = llm.invoke(msgs)
        return {"answer": resp.content, "source_docs": docs}

    return run
