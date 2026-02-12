"""
Sentiment Analysis Pipeline
GenAI-powered sentiment analysis for team news, trade rumors, and coaching stability
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
from ..services.tavily_service import TavilyService
from ..services.gemini_service import GeminiService
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes news sentiment using Tavily (search) and Gemini (analysis).
    Falls back to keyword-based approach if AI unavailable.
    """
    
    # Sentiment keywords (fallback)
    POSITIVE_KEYWORDS = [
        'win', 'victory', 'success', 'dominant', 'impressive', 'strong',
        'confident', 'healthy', 'chemistry', 'momentum', 'playoff',
        'championship', 'elite', 'excellent', 'outstanding', 'stellar'
    ]
    
    NEGATIVE_KEYWORDS = [
        'loss', 'defeat', 'struggle', 'injury', 'injured', 'questionable',
        'doubtful', 'fire', 'fired', 'tension', 'conflict', 'trade',
        'rebuild', 'tank', 'tanking', 'disappointing', 'poor', 'weak'
    ]
    
    TRADE_KEYWORDS = [
        'trade', 'deal', 'move', 'acquire', 'ship', 'send', 'package',
        'swap', 'exchange', 'deadline', 'rumor', 'speculation'
    ]
    
    COACHING_INSTABILITY_KEYWORDS = [
        'fire', 'fired', 'firing', 'hot seat', 'pressure', 'replace',
        'interim', 'search', 'candidate', 'tension', 'conflict'
    ]
    
    def __init__(self):
        """Initialize with AI services"""
        self.tavily = TavilyService()
        self.gemini = GeminiService()
    
    def analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using Gemini if available, else keywords
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        # Try Gemini first
        if self.gemini.model:
            result = self.gemini.analyze_sentiment(text)
            if result['analysis'] not in ['Gemini not configured', 'Analysis failed']:
                if result['score'] != 0.0:
                    return result['score']
        
        # Fallback to keyword matching
        text_lower = text.lower()
        
        # Count positive and negative keywords
        positive_count = sum(1 for keyword in self.POSITIVE_KEYWORDS if keyword in text_lower)
        negative_count = sum(1 for keyword in self.NEGATIVE_KEYWORDS if keyword in text_lower)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return 0.0  # Neutral
        
        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / total_keywords
        
        return sentiment
    
    def calculate_trade_rumor_intensity(self, articles: List[str]) -> float:
        """
        Calculate trade rumor intensity from news articles
        
        Args:
            articles: List of article texts
            
        Returns:
            Trade rumor intensity (0.0 - 1.0)
        """
        if len(articles) == 0:
            return 0.0
        
        total_trade_mentions = 0
        
        for article in articles:
            article_lower = article.lower()
            trade_mentions = sum(1 for keyword in self.TRADE_KEYWORDS if keyword in article_lower)
            total_trade_mentions += trade_mentions
        
        # Normalize by number of articles
        avg_mentions = total_trade_mentions / len(articles)
        
        # Cap at 1.0
        return min(1.0, avg_mentions / 5.0)  # 5+ mentions = max intensity
    
    def calculate_coaching_stability(self, articles: List[str]) -> float:
        """
        Calculate coaching stability index
        
        Args:
            articles: List of article texts
            
        Returns:
            Stability index (0.0 - 1.0, where 1.0 = stable)
        """
        if len(articles) == 0:
            return 1.0  # Assume stable if no news
        
        instability_count = 0
        
        for article in articles:
            article_lower = article.lower()
            for keyword in self.COACHING_INSTABILITY_KEYWORDS:
                if keyword in article_lower:
                    instability_count += 1
                    break  # Count once per article
        
        # Calculate stability (inverse of instability)
        instability_rate = instability_count / len(articles)
        stability = 1.0 - instability_rate
        
        return stability
    
    def analyze_team_sentiment(
        self,
        team_name: str,
        articles: List[str],
        lookback_days: int = 7
    ) -> Dict:
        """
        Comprehensive team sentiment analysis
        
        Args:
            team_name: Team name
            articles: List of recent article texts
            lookback_days: Days to look back
            
        Returns:
            Dictionary with sentiment metrics
        """
        if len(articles) == 0:
            return {
                'sentiment_score': 0.0,
                'trade_rumor_intensity': 0.0,
                'coaching_stability': 1.0,
                'articles_analyzed': 0,
                'confidence': 0.0
            }
        
        # Calculate sentiment for each article
        sentiments = [self.analyze_text_sentiment(article) for article in articles]
        
        # Weight recent articles more heavily
        weighted_sentiments = []
        for i, sentiment in enumerate(sentiments):
            # More recent = higher weight (1.0 to 0.5)
            weight = 1.0 - (i * 0.5 / len(sentiments))
            weighted_sentiments.append(sentiment * weight)
        
        avg_sentiment = sum(weighted_sentiments) / len(weighted_sentiments) if weighted_sentiments else 0.0
        
        # Calculate other metrics
        trade_intensity = self.calculate_trade_rumor_intensity(articles)
        coaching_stability = self.calculate_coaching_stability(articles)
        
        # Confidence based on number of articles
        confidence = min(1.0, len(articles) / 10.0)  # 10+ articles = max confidence
        
        return {
            'sentiment_score': avg_sentiment,
            'trade_rumor_intensity': trade_intensity,
            'coaching_stability': coaching_stability,
            'articles_analyzed': len(articles),
            'confidence': confidence
        }
    
    def get_sentiment_impact_on_prediction(self, sentiment_data: Dict) -> float:
        """
        Convert sentiment to win probability adjustment
        
        Args:
            sentiment_data: Dictionary from analyze_team_sentiment
            
        Returns:
            Win probability adjustment (-0.05 to +0.05)
        """
        sentiment_score = sentiment_data['sentiment_score']
        trade_intensity = sentiment_data['trade_rumor_intensity']
        coaching_stability = sentiment_data['coaching_stability']
        confidence = sentiment_data['confidence']
        
        # Base adjustment from sentiment
        adjustment = sentiment_score * 0.03  # ±3% max from sentiment
        
        # Penalty for high trade rumors
        adjustment -= trade_intensity * 0.02  # -2% max
        
        # Penalty for coaching instability
        adjustment -= (1.0 - coaching_stability) * 0.02  # -2% max
        
        # Scale by confidence
        adjustment *= confidence
        
        return max(-0.05, min(0.05, adjustment))


# Mock news fetcher (would be replaced with actual API calls)
class NewsFetcher:
    """Fetches news articles for teams using Tavily"""
    
    def __init__(self):
        self.tavily = TavilyService()
    
    def fetch_team_news(self, team_name: str, days: int = 3) -> List[str]:
        """
        Fetch recent news articles for a team via Tavily API
        
        Args:
            team_name: Team name
            days: Days to look back (used in query mainly)
            
        Returns:
            List of article texts
        """
        query = f"{team_name} basketball team news rumors injuries"
        
        # Search Tavily
        results = self.tavily.search_news(query, max_results=5)
        
        articles = []
        for res in results:
            # Combine title and content snippet for analysis
            text = f"{res.get('title', '')}. {res.get('content', '')}"
            articles.append(text)
            
        # Fallback to mock if API returns nothing (e.g. key missing)
        if not articles:
             # Placeholder - would integrate with ESPN, The Athletic, Reddit APIs
            # For now, return mock articles
            
            mock_articles = {
                'Lakers': [
                    "Lakers win impressive victory over Celtics with dominant performance",
                    "LeBron James injury concerns as he sits out practice",
                    "Trade rumors swirl around Lakers roster ahead of deadline",
                ],
                'Celtics': [
                    "Celtics maintain strong chemistry despite recent loss",
                    "Jayson Tatum excellent in clutch situations this season",
                    "Boston looking healthy heading into playoff push",
                ]
            }
            return mock_articles.get(team_name, [])
            
        return articles


# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    fetcher = NewsFetcher()
    
    print("Sentiment Analysis Pipeline Test")
    print("=" * 50)
    
    # Test Lakers sentiment
    print("\nLakers Sentiment Analysis:")
    print("-" * 50)
    
    lakers_articles = fetcher.fetch_team_news('Lakers')
    lakers_sentiment = analyzer.analyze_team_sentiment('Lakers', lakers_articles)
    
    print(f"Articles Analyzed: {lakers_sentiment['articles_analyzed']}")
    print(f"Sentiment Score: {lakers_sentiment['sentiment_score']:.2f} (-1 to +1)")
    print(f"Trade Rumor Intensity: {lakers_sentiment['trade_rumor_intensity']:.2f}")
    print(f"Coaching Stability: {lakers_sentiment['coaching_stability']:.2f}")
    print(f"Confidence: {lakers_sentiment['confidence']:.2f}")
    
    impact = analyzer.get_sentiment_impact_on_prediction(lakers_sentiment)
    print(f"\nWin Probability Impact: {impact:+.1%}")
    
    # Test Celtics sentiment
    print("\n\nCeltics Sentiment Analysis:")
    print("-" * 50)
    
    celtics_articles = fetcher.fetch_team_news('Celtics')
    celtics_sentiment = analyzer.analyze_team_sentiment('Celtics', celtics_articles)
    
    print(f"Articles Analyzed: {celtics_sentiment['articles_analyzed']}")
    print(f"Sentiment Score: {celtics_sentiment['sentiment_score']:.2f}")
    print(f"Trade Rumor Intensity: {celtics_sentiment['trade_rumor_intensity']:.2f}")
    print(f"Coaching Stability: {celtics_sentiment['coaching_stability']:.2f}")
    print(f"Confidence: {celtics_sentiment['confidence']:.2f}")
    
    impact2 = analyzer.get_sentiment_impact_on_prediction(celtics_sentiment)
    print(f"\nWin Probability Impact: {impact2:+.1%}")
    
    print("\n" + "=" * 50)
    print("Note: This uses keyword-based analysis.")
    print("Can be upgraded to transformers (BERT, GPT) for better accuracy.")
