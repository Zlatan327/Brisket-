"""
Prediction Service
Orchestrates data gathering (stats + sentiment) and model inference for real-time predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import logging

from ..core.config import settings
from ..ml.ensemble_predictor import EnsemblePredictor
from ..ml.feature_engineer import FeatureEngineer
from ..data.sentiment_analyzer import SentimentAnalyzer, NewsFetcher
from ..ml.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Service to handle full prediction flow:
    1. Identify teams and date
    2. Fetch historical stats (rolling averages)
    3. Fetch and analyze real-time news (Tavily + Gemini)
    4. Construct feature vector
    5. Run inference (Ensemble)
    6. Return detailed response
    """
    
    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.feature_engineer = FeatureEngineer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher()
        self.shap_explainer = SHAPExplainer()
        
        # Load historical data for feature engineering
        # In a real production app, this might query a DB. 
        # For now, we'll load the CSV or usage a database connector if available.
        # We'll assume the feature engineer can access necessary history or we pass it.
        try:
            self.games_df = pd.read_csv("historical_games_2023_24.csv")
            self.games_df['date'] = pd.to_datetime(self.games_df['date'])
            logger.info(f"Loaded {len(self.games_df)} historical games for context.")
        except Exception as e:
            logger.error(f"Could not load historical games: {e}")
            self.games_df = pd.DataFrame()

    async def predict_matchup(self, home_team: str, away_team: str, date: str) -> Dict:
        """
        Predict outcome for a specific matchup using real-time insights
        
        Args:
            home_team: Home team abbreviation (e.g., "LAL")
            away_team: Away team abbreviation (e.g., "BOS")
            date: Game date (YYYY-MM-DD)
            
        Returns:
            Dictionary with prediction details (probability, confidence, explanation)
        """
        logger.info(f"Predicting matchup: {home_team} vs {away_team} on {date}")
        
        # 1. Calculate Base Statistical Features
        # We need to reconstruct the features based on the team's history up to this date.
        features = self._build_features(home_team, away_team, date)
        
        # 2. Add Real-time Sentiment Analysis
        features = await self._enrich_with_sentiment(features, home_team, away_team)
        
        # 3. Format for Model
        feature_array = self._prepare_feature_array(features)
        
        # 4. Run Prediction
        prediction_details = self.ensemble.predict_with_details(feature_array)
        home_win_prob = prediction_details['ensemble_predictions'][0]
        
        # 5. Get Explanation
        explanation = self.shap_explainer.explain_prediction(feature_array[0])
        
        # 6. Construct Response
        predicted_winner = home_team if home_win_prob > 0.5 else away_team
        confidence = self.ensemble.get_confidence_level(home_win_prob)
        
        return {
            "game_id": f"{home_team}_{away_team}_{date}",
            "home_team": home_team,
            "away_team": away_team,
            "date": date,
            "home_win_probability": float(home_win_prob),
            "away_win_probability": 1.0 - float(home_win_prob),
            "predicted_winner": predicted_winner,
            "confidence_level": confidence,
            "top_factors": explanation['top_5_features'],
            "sentiment_analysis": {
                "home_score": features.get('home_sentiment_score', 0),
                "away_score": features.get('away_sentiment_score', 0),
                "news_impact": features.get('sentiment_impact', 0) # Derived metic
            }
        }

    def _build_features(self, home_team: str, away_team: str, date: str) -> Dict:
        """Construct base features from historical data"""
        # This uses the FeatureEngineer logic but applied to a specific point in time
        # For simplicity in this iteration, we will use the most recent known stats 
        # for each team from self.games_df or defaults.
        
        # TODO: Refactor FeatureEngineer to expose precise "get_team_stats_at_date" methods.
        # For now, we will calculate rolling stats manually using the helpers in FeatureEngineer
        
        # Helper to find team ID (assuming we have a mapping, or use string logic if DF uses names)
        # Using placeholder IDs or names based on DF structure
        # Loading logic...
        
        # Simplified: Use FeatureEngineer's calculation methods
        # Handle case where history is missing (e.g. fresh install / scratch env)
        if hasattr(self, 'games_df') and not self.games_df.empty:
            try:
                home_rolling = self.feature_engineer.calculate_rolling_stats(
                    self.games_df, 
                    self._get_team_id(home_team), 
                    date
                )
                away_rolling = self.feature_engineer.calculate_rolling_stats(
                    self.games_df, 
                    self._get_team_id(away_team), 
                    date
                )
                
                home_rest = self.feature_engineer.calculate_rest_days(self.games_df, self._get_team_id(home_team), date)
                away_rest = self.feature_engineer.calculate_rest_days(self.games_df, self._get_team_id(away_team), date)
            except Exception as e:
                logger.warning(f"Error calculating rolling stats (using defaults): {e}")
                home_rolling = {'avg_score': 110, 'avg_allowed': 110, 'last_5_wins': 2, 'last_10_wins': 5}
                away_rolling = {'avg_score': 110, 'avg_allowed': 110, 'last_5_wins': 2, 'last_10_wins': 5}
                home_rest = 3
                away_rest = 3
        else:
            logger.warning("Historical data not available. Using default average stats.")
            home_rolling = {'avg_score': 110.0, 'avg_allowed': 110.0, 'last_5_wins': 2, 'last_10_wins': 5}
            away_rolling = {'avg_score': 110.0, 'avg_allowed': 110.0, 'last_5_wins': 2, 'last_10_wins': 5}
            home_rest = 3
            away_rest = 3
        
        # Base features dictionary with defaults for static/unknowns
        features = {
            'home_efg_pct': 0.54, # League Avg placeholders if history missing
            'away_efg_pct': 0.54,
            'home_tov_rate': 0.13,
            'away_tov_rate': 0.13,
            'home_drb_pct': 0.77,
            'away_drb_pct': 0.77,
            'home_ft_rate': 0.20,
            'away_ft_rate': 0.20,
            
            'home_net_rating': home_rolling['avg_score'] - home_rolling['avg_allowed'],
            'away_net_rating': away_rolling['avg_score'] - away_rolling['avg_allowed'],
            'home_pace': 99.0,
            'away_pace': 99.0,
            
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'travel_distance': 0.0, # Placeholder
            'timezone_shift': 0,
            'elevation_change': 0,
            
            'home_fatigue_score': max(0, 50 - (home_rest * 10)),
            'away_fatigue_score': max(0, 50 - (away_rest * 10)),
            'home_cumulative_load': 0.0,
            'away_cumulative_load': 0.0,
            
            'home_last_5_wins': home_rolling['last_5_wins'],
            'away_last_5_wins': away_rolling['last_5_wins'],
            'home_last_10_wins': home_rolling['last_10_wins'],
            'away_last_10_wins': away_rolling['last_10_wins'],
            
            'home_injury_impact': 0.0,
            'away_injury_impact': 0.0,
            
            # Sentiment placeholders (filled later)
            'home_sentiment_score': 0.0, 
            'away_sentiment_score': 0.0,
            'home_trade_rumors': 0.0, 
            'away_trade_rumors': 0.0,
            'home_coaching_stability': 1.0, 
            'away_coaching_stability': 1.0,
            'home_tanking_score': 0.0, 
            'away_tanking_score': 0.0,
            
            'is_back_to_back_home': 1 if home_rest == 0 else 0,
            'is_back_to_back_away': 1 if away_rest == 0 else 0,
            'home_court_advantage': 1.0
        }
        return features

    async def _enrich_with_sentiment(self, features: Dict, home_team: str, away_team: str) -> Dict:
        """Fetch news and update sentiment features"""
        
        # 1. Fetch News
        home_news = self.news_fetcher.fetch_team_news(home_team)
        away_news = self.news_fetcher.fetch_team_news(away_team)
        
        # 2. Analyze Sentiment
        home_sentiment = self.sentiment_analyzer.analyze_team_sentiment(home_team, home_news)
        away_sentiment = self.sentiment_analyzer.analyze_team_sentiment(away_team, away_news)
        
        # 3. Update Features
        features['home_sentiment_score'] = home_sentiment['sentiment_score']
        features['home_trade_rumors'] = home_sentiment['trade_rumor_intensity']
        features['home_coaching_stability'] = home_sentiment['coaching_stability']
        
        features['away_sentiment_score'] = away_sentiment['sentiment_score']
        features['away_trade_rumors'] = away_sentiment['trade_rumor_intensity']
        features['away_coaching_stability'] = away_sentiment['coaching_stability']
        
        return features

    def _prepare_feature_array(self, features: Dict) -> np.ndarray:
        """Convert features dict to the specific array order expected by the model"""
        # Order MUST match training data exactly
        ordered_values = [
            features['home_efg_pct'], features['away_efg_pct'],
            features['home_tov_rate'], features['away_tov_rate'],
            features['home_drb_pct'], features['away_drb_pct'],
            features['home_ft_rate'], features['away_ft_rate'],
            features['home_net_rating'], features['away_net_rating'],
            features['home_pace'], features['away_pace'],
            features['home_rest_days'], features['away_rest_days'],
            features['travel_distance'], features['timezone_shift'], features['elevation_change'],
            features['home_fatigue_score'], features['away_fatigue_score'],
            features['home_cumulative_load'], features['away_cumulative_load'],
            features['home_last_5_wins'], features['away_last_5_wins'],
            features['home_last_10_wins'], features['away_last_10_wins'],
            features['home_injury_impact'], features['away_injury_impact'],
            features['home_sentiment_score'], features['away_sentiment_score'],
            features['home_trade_rumors'], features['away_trade_rumors'],
            features['home_coaching_stability'], features['away_coaching_stability'],
            features['home_tanking_score'], features['away_tanking_score'],
            features['is_back_to_back_home'], features['is_back_to_back_away'],
            features['home_court_advantage']
        ]
        return np.array(ordered_values).reshape(1, -1)
        
    def _get_team_id(self, team_name: str) -> int:
        """Map team name to ID - simplified for prototype"""
        # In real app, query Team model or use a mapping dict
        # This is a stub to make the code run with the existing CSV structure if it uses IDs
        # If CSV uses names, we might need to adjust logic.
        # Assuming CSV has IDs.
        team_map = {
            'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751, 'CHA': 1610612766,
            'CHI': 1610612741, 'CLE': 1610612739, 'DAL': 1610612742, 'DEN': 1610612743,
            'DET': 1610612765, 'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
            'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763, 'MIA': 1610612748,
            'MIL': 1610612749, 'MIN': 1610612750, 'NOP': 1610612740, 'NYK': 1610612752,
            'OKC': 1610612760, 'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
            'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'TOR': 1610612761,
            'UTA': 1610612762, 'WAS': 1610612764
        }
        return team_map.get(team_name, 0)
