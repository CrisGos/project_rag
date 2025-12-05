# chain.py - Migrado a LangGraph
from __future__ import annotations
import json
from typing import Dict, List, TypedDict, Annotated
from operator import add

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END

from rag_app.config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    REPLY_LANGUAGE,
    weaviate_graphql,
    WEAVIATE_CLASS,
)


# ========== Estado del Grafo ==========
class RAGState(TypedDict):
    """Estado que se pasa entre nodos del grafo LangGraph"""
    question: str
    pdf_name: str
    k: int
    retrieved_docs: List[dict]
    context: str
    answer: str
    source_docs: List[dict]


# ========== Funciones auxiliares ==========
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


# ========== Nodos del Grafo LangGraph ==========
def retrieve_node(state: RAGState) -> RAGState:
    """
    Nodo 1: Recuperar documentos relevantes de Weaviate
    """
    question = state["question"]
    pdf_name = state["pdf_name"]
    k = state["k"]
    
    # Recuperar documentos usando búsqueda vectorial
    docs = retrieve_weaviate(question, pdf_name=pdf_name, k=k)
    
    # Construir contexto concatenando los chunks recuperados
    context = "\n\n".join([f"[p.{d['page_number']}] {d['text']}" for d in docs])
    
    return {
        **state,
        "retrieved_docs": docs,
        "context": context,
        "source_docs": docs,
    }


def generate_node(state: RAGState) -> RAGState:
    """
    Nodo 2: Generar respuesta usando el LLM con el contexto recuperado
    """
    question = state["question"]
    context = state["context"]
    
    # Inicializar el modelo LLM
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_CHAT_MODEL,
        temperature=0.2,
    )
    
    # Construir mensajes para el LLM
    msgs = [
        ("system", _system_prompt()),
        ("human", f"Question: {question}\n\nContext:\n{context}"),
    ]
    
    # Generar respuesta
    resp = llm.invoke(msgs)
    
    return {
        **state,
        "answer": resp.content,
    }


# ========== Construcción del Grafo LangGraph ==========
def build_rag_graph():
    """
    Construye el grafo RAG usando LangGraph.
    
    Flujo: START -> retrieve -> generate -> END
    """
    # Crear el grafo con el estado definido
    workflow = StateGraph(RAGState)
    
    # Agregar nodos al grafo
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # Definir el flujo de ejecución
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compilar el grafo
    return workflow.compile()


# ========== Función principal compatible con el código existente ==========
def make_rag_chain():
    """
    Crea y retorna una función callable que ejecuta el grafo RAG.
    Mantiene la misma interfaz que la versión original para compatibilidad.
    """
    graph = build_rag_graph()
    
    def run(question: str, pdf_name: str, k: int = 4) -> Dict[str, any]:
        """
        Ejecuta el grafo RAG con la pregunta y parámetros dados.
        
        Args:
            question: Pregunta del usuario
            pdf_name: Nombre del PDF a consultar
            k: Número de chunks a recuperar
            
        Returns:
            Dict con 'answer' y 'source_docs'
        """
        # Preparar el estado inicial
        initial_state = {
            "question": question,
            "pdf_name": pdf_name,
            "k": k,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "source_docs": [],
        }
        
        # Ejecutar el grafo
        final_state = graph.invoke(initial_state)
        
        # Retornar en el formato esperado
        return {
            "answer": final_state["answer"],
            "source_docs": final_state["source_docs"],
        }
    
    return run
