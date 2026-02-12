from tavily import TavilyClient
from ..core.config import settings
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TavilyService:
    """
    Service to interact with Tavily API for real-time news search.
    """
    
    def __init__(self):
        """Initialize Tavily client"""
        self.api_key = settings.TAVILY_API_KEY
        if not self.api_key:
            logger.warning("TAVILY_API_KEY is not set. News search will be disabled.")
            self.client = None
        else:
            try:
                self.client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
                self.client = None

    def search_news(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for news articles related to the query.
        
        Args:
            query: Search query (e.g., "LeBron James injury update")
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing 'title', 'url', 'content', 'published_date'
        """
        if not self.client:
            logger.warning("Tavily client not initialized. returning mock results.")
            return self._mock_results(query)
            
        try:
            # specialized search for news
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
                include_images=False,
                # topic="news" # Tavily python client might not support topic param directly in all versions, check docs. 
                # Assuming basic search for now.
            )
            
            # The response structure depends on the client version, assuming standard 'results' list
            results = response.get('results', [])
            
            cleaned_results = []
            for res in results:
                cleaned_results.append({
                    'title': res.get('title'),
                    'url': res.get('url'),
                    'content': res.get('content'),
                    'published_date': res.get('published_date', 'Unknown')
                })
                
            return cleaned_results
            
        except Exception as e:
            logger.error(f"Error searching Tavily: {e}")
            return self._mock_results(query)

    def _mock_results(self, query: str) -> List[Dict]:
        """Return mock results for testing or when API key is missing"""
        return [
            {
                'title': f"Mock News: {query}",
                'url': "http://example.com/news",
                'content': "This is a mock news article content for testing purposes.",
                'published_date': "2024-02-12"
            }
        ]
