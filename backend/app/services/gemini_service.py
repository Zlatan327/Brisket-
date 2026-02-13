import google.generativeai as genai
from ..core.config import settings
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    """
    Service to interact with Google Gemini API for sentiment analysis and prediction context.
    """
    
    def __init__(self):
        """Initialize Gemini client"""
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. AI analysis will be disabled.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                # Use a widely available model
                self.model = genai.GenerativeModel('gemini-2.0-flash')
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.model = None

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a text (e.g., news article).
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary with sentiment score (-1.0 to 1.0) and analysis
        """
        if not self.model:
            return {'score': 0.0, 'analysis': 'Gemini not configured'}
            
        prompt = f"""
        Analyze the sentiment of the following sports news text regarding team morale and winning chances.
        Return a JSON object with:
        - "score": float between -1.0 (very negative impact) and 1.0 (very positive impact)
        - "reasoning": brief explanation
        
        Text: {text[:2000]}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Basic parsing, ideally use stricter format or json mode if available
            # For now, let's just return raw text or mock parsing if simple
            return {'score': 0.0, 'analysis': response.text}
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {'score': 0.0, 'analysis': 'Analysis failed'}

    def analyze_prediction_context(self, game_info: Dict, statistical_pred: float) -> str:
        """
        Provide a "gut check" analysis on the statistical prediction.
        
        Args:
            game_info: Dict with game details (teams, key players, recent news)
            statistical_pred: Model's win probability (0.0-1.0)
            
        Returns:
            Text analysis of whether the model might be missing context
        """
        if not self.model:
            return "AI analysis unavailable."
            
        prompt = f"""
        You are a basketball expert. The statistical model predicts a {statistical_pred:.1%} chance of {game_info.get('home_team')} winning against {game_info.get('away_team')}.
        
        Consider this recent news/context:
        {game_info.get('news_summary', 'No specific news.')}
        
        Key injuries:
        {game_info.get('injuries', 'None reported.')}
        
        Does this context support or contradict the statistical model? Provide a brief "Expert Insight".
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return "AI analysis failed."
