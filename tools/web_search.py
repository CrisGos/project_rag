from tavily import TavilyClient
from typing import List, Dict, Optional
from rag_app.config.settings import TAVILY_API_KEY

class WebSearchTool:
    """Web search tool using Tavily API"""
    
    def __init__(self):
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        self.client = TavilyClient(api_key=TAVILY_API_KEY)
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform web search"""
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True
            )
            return response.get('results', [])
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    def get_context(self, query: str, max_results: int = 3) -> str:
        """Get formatted context for RAG"""
        results = self.search(query, max_results)
        
        if not results:
            return ""
        
        context = "🌐 Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Unknown')
            content = result.get('content', '')
            url = result.get('url', '')
            context += f"{i}. **{title}**\n{content}\n[Source: {url}]\n\n"
        
        return context