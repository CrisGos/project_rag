# chain.py - Migrado a LangGraph
from __future__ import annotations

from typing import Dict, List, TypedDict, Annotated, Optional
from operator import add

from rag_app.services.tavily import query_tavily

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END

from rag_app.config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    REPLY_LANGUAGE,
    get_weaviate_client,
    WEAVIATE_CLASS,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_HOST,
    LANGFUSE_ENABLED,
)
from weaviate.classes.query import Filter

# Langfuse imports
LANGFUSE_AVAILABE = False
langfuse_client = None

try:
    from langfuse import Langfuse    
    LANGFUSE_AVAILABLE = True
except ImportError:
    pass

# Initialize Langfuse client
if LANGFUSE_AVAILABLE and LANGFUSE_ENABLED and LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
    try:
        langfuse_client = Langfuse(
            secret_key = LANGFUSE_SECRET_KEY,
            public_key = LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
    except Exception:
        langfuse_client = None


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
    # New fields for Tavily fallback
    is_external: bool
    external_sources: List[str]


# ========== Funciones auxiliares ==========
def _system_prompt() -> str:
    return (
        f"You are a concise assistant. Answer ONLY using the manual content provided "
        f"in the retrieved chunks. If the answer is not present, say you don't know. "
        f"Reply language: {REPLY_LANGUAGE}. Keep answers short and cite page number(s) "
        f"in parentheses."
    )


def retrieve_weaviate(query: str, pdf_name: str | None, k: int = 4) -> List[dict]:
    """
    Use Weaviate v4 client for retrieval.
    """
    embed = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    qvec = embed.embed_query(query)
    
    out: List[dict] = []
    
    try:
        with get_weaviate_client() as client:
            collection = client.collections.get(WEAVIATE_CLASS)
            if pdf_name:
                results = collection.query.near_vector(
                    near_vector=qvec,
                    limit=int(k),
                    filters=Filter.by_property("pdf_name").equal(pdf_name),
                    return_metadata=["distance"]
                )
            else:
                # Global search without filter
                results = collection.query.near_vector(
                    near_vector=qvec,
                    limit=int(k),
                    return_metadata=["distance"]
                )
            
            for obj in results.objects:
                out.append({
                    "text": obj.properties.get("text", ""),
                    "page_number": obj.properties.get("page_number"),
                    "chunk_id": obj.properties.get("chunk_id"),
                    "pdf_name": obj.properties.get("pdf_name"),
                    "distance": obj.metadata.distance,
                })
    except Exception as e:
        # Fallback or log if retrieval fails
        print(f"Retrieval error: {e}")
        
    return out


def check_exists_logic(state: RAGState) -> str:
    """
    Conditional logic to determine next step.
    """
    pdf_name = state.get("pdf_name")
    if not pdf_name:
         # If no pdf_name, maybe go to retrieve? Or just Tavily?
         # Assuming intent is RAG on a doc. If global search (no pdf_name), 
         # we probably stick to retrieval?
         # Prompt says: "documento solicitado... no existe... Si false -> tavily".
         # If no doc requested, standard RAG? 
         # Let's assume pdf_name is required for this check.
         # If pdf_name is None, let's just retrieve (global search).
         return "retrieve"

    # Check Weaviate
    exists = False
    try:
        with get_weaviate_client() as client:
            collection = client.collections.get(WEAVIATE_CLASS)
            response = collection.query.fetch_objects(
                limit=1,
                filters=Filter.by_property("pdf_name").equal(pdf_name),
                return_properties=["pdf_name"]
            )
            if len(response.objects) > 0:
                exists = True
    except Exception:
        exists = False
        
    if exists:
        return "retrieve"
    else:
        return "tavily_agent"


def tavily_agent_node(state: RAGState) -> RAGState:
    """
    Node: Fallback to Tavily search when document is missing.
    """
    question = state["question"]
    pdf_name = state.get("pdf_name", "")
    
    # Optional: Include pdf_name in query as hint
    query = f"{question} (Context: {pdf_name})" if pdf_name else question
    
    result = query_tavily(query)
    
    # Format answer to indicate external source
    answer = f"[External Search] {result['answer']}"
    
    return {
        **state,
        "answer": answer,
        "is_external": True,
        "external_sources": result["sources"]
    }


def check_answer_logic(state: RAGState) -> str:
    """
    Check if the generated answer is useful or if we need to fallback to Tavily.
    """
    answer = state.get("answer", "").lower()
    # "don't know" logic as per prompt requirements
    if "don't know" in answer or "do not know" in answer or "no connection" in answer:
         return "tavily_agent"
    return "end"


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

    # If global search, we might want to include the source PDF name in the context for the LLM
    context_parts = []
    for d in docs:
        source_info = f"[DOC: {d.get('pdf_name', 'Unknown')}] " if not pdf_name else ""
        context_parts.append(f"{source_info}[p.{d['page_number']}] {d['text']}")
    
    context = "\n\n".join(context_parts)
    
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

    # Log to Langfuse if available
    if langfuse_client:
        try:
            trace = langfuse_client.trace(
                name="rag-generation",
                metadata={"pdf_name": state.get("pdf_name")},
            )
            trace.generation(
                name="llm-call",
                model=OLLAMA_CHAT_MODEL,
                input={"question": question, "context": context[:500]},
                output=resp.content,
            )
        except Exception:
            pass

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
    workflow.add_node("tavily_agent", tavily_agent_node)
    
    # Definir el flujo de ejecución
    # Entry point checks existence via conditional edge
    workflow.set_conditional_entry_point(
        check_exists_logic,
        {
            "retrieve": "retrieve",
            "tavily_agent": "tavily_agent"
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    
    # Conditional edge after generate to check if answer is valid
    workflow.add_conditional_edges(
        "generate",
        check_answer_logic,
        {
            "end": END,
            "tavily_agent": "tavily_agent"
        }
    )
    
    workflow.add_edge("tavily_agent", END)
    
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
            "is_external": False,
            "external_sources": [],
        }
        
        # Ejecutar el grafo
        final_state = graph.invoke(initial_state)

        # Log trace end to Langfuse
        trace = None
        if trace:
            try:
                trace.update(
                    output={"answer": final_state["answer"]},
                )
            except Exception:
                pass
        
        # Retornar en el formato esperado
        return {
            "answer": final_state["answer"],
            "source_docs": final_state["source_docs"],
            "is_external": final_state.get("is_external", False),
            "external_sources": final_state.get("external_sources", []),
        }
    
    return run
