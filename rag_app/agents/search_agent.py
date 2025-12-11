from typing import Dict, List
import re
from tools.web_search import WebSearchTool

class SearchAgent:
    """Agent that decides when to use web search vs RAG"""
    
    def __init__(self):
        self.web_search = WebSearchTool()
        
        # Keywords that trigger web search
        self.web_keywords = [
            'latest', 'current', 'recent', 'today', 'news',
            'update', 'now', 'search web', 'google', 'find online',
            'what is happening', 'web search', 'internet'
        ]
    
    def should_use_web_search(self, query: str) -> bool:
        """Determine if web search should be used"""
        query_lower = query.lower()
        
        # Check for explicit web search keywords
        if any(keyword in query_lower for keyword in self.web_keywords):
            return True
        
        # Check for current year references
        if re.search(r'\b202[3-9]\b', query_lower):
            return True
        
        return False
    
    def enhance_with_web_search(self, query: str, rag_answer: str, rag_sources: List) -> Dict:
        """Enhance RAG response with web search if needed"""
        
        use_web = self.should_use_web_search(query)
        web_results = []
        enhanced_answer = rag_answer
        
        if use_web:
            web_results = self.web_search.search(query, max_results=3)
            
            if web_results:
                web_context = self.web_search.get_context(query, max_results=3)
                enhanced_answer = f"{rag_answer}\n\n---\n\n{web_context}"
        
        return {
            "answer": enhanced_answer,
            "used_web_search": use_web,
            "web_results": web_results,
            "rag_sources": rag_sources
        }