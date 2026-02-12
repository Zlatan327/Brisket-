from app.services.tavily_service import TavilyService
from app.services.gemini_service import GeminiService
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tavily_service():
    logger.info("Testing TavilyService...")
    service = TavilyService()
    
    # Test with mock query
    results = service.search_news("LeBron James injury", max_results=2)
    logger.info(f"Results Count: {len(results)}")
    if results:
        logger.info(f"First Result Title: {results[0].get('title')}")
    
    # If API key is missing, it should return mock results
    if not settings.TAVILY_API_KEY:
        assert len(results) > 0, "Should return mock results when no API key"
        assert "Mock News" in results[0]['title'], "Should be mock news"

def test_gemini_service():
    logger.info("Testing GeminiService...")
    service = GeminiService()
    
    # Test sentiment analysis
    result = service.analyze_sentiment("Lakers win championship!")
    logger.info(f"Sentiment Analysis: {result}")
    
    if not settings.GEMINI_API_KEY:
        assert result['analysis'] == 'Gemini not configured', "Should return 'Gemini not configured' when no API key"

if __name__ == "__main__":
    logger.info("Starting Integration Tests...")
    test_tavily_service()
    test_gemini_service()
    logger.info("Integration Tests Completed Successfully!")
