"""
Q4 Feature Engineering
Extracts features specific to 4th quarter performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class Q4FeatureEngineer:
    """
    Feature engineer for Q4 micro-predictions.
    Focuses on game state entering Q4 and clutch factors.
    """
    
    def __init__(self):
        """Initialize Q4 feature engineer"""
        pass
        
    def extract_features(self, game_state: Dict) -> Dict:
        """
        Extract features from current game state
        
        Args:
            game_state: Dictionary containing:
                - home_score_q3: Score at end of Q3
                - away_score_q3: Score at end of Q3
                - home_team_id: ID of home team
                - away_team_id: ID of away team
                - home_clutch_stats: Dict of clutch stats (season avg)
                - away_clutch_stats: Dict of clutch stats (season avg)
                - pre_game_prob: Pre-game win probability
                
        Returns:
            Dictionary of features for Q4 model
        """
        # 1. Score Differential
        score_diff = game_state['home_score_q3'] - game_state['away_score_q3']
        
        # 2. Clutch Stats (Season Avg)
        # Defaults if not provided
        home_clutch = game_state.get('home_clutch_stats', {})
        away_clutch = game_state.get('away_clutch_stats', {})
        
        home_clutch_ortg = home_clutch.get('ortg', 110.0)
        away_clutch_ortg = away_clutch.get('ortg', 110.0)
        home_clutch_drtg = home_clutch.get('drtg', 110.0)
        away_clutch_drtg = away_clutch.get('drtg', 110.0)
        
        home_clutch_net = home_clutch_ortg - home_clutch_drtg
        away_clutch_net = away_clutch_ortg - away_clutch_drtg
        
        # 3. Star Power (simplified)
        # Usage rate of top 2 players available
        home_star_usage = home_clutch.get('star_usage', 0.30)
        away_star_usage = away_clutch.get('star_usage', 0.30)
        
        # 4. Pace Factor
        # Q4 pace is usually slower. We can use season clutch pace or just Q4 pace.
        pace = (home_clutch.get('pace', 98.0) + away_clutch.get('pace', 98.0)) / 2
        
        # 5. Pre-game context
        pre_game_prob = game_state.get('pre_game_prob', 0.5)
        
        return {
            'score_diff': score_diff,
            'home_clutch_net_rating': home_clutch_net,
            'away_clutch_net_rating': away_clutch_net,
            'home_star_usage': home_star_usage,
            'away_star_usage': away_star_usage,
            'pace': pace,
            'pre_game_win_prob': pre_game_prob
        }
    
    def prepare_for_inference(self, features: Dict) -> np.ndarray:
        """
        Convert feature dictionary to numpy array for model inference
        
        Order must match training!
        1. score_diff
        2. home_clutch_net_rating
        3. away_clutch_net_rating
        4. home_star_usage
        5. away_star_usage
        6. pace
        7. pre_game_win_prob
        """
        return np.array([
            features['score_diff'],
            features['home_clutch_net_rating'],
            features['away_clutch_net_rating'],
            features['home_star_usage'],
            features['away_star_usage'],
            features['pace'],
            features['pre_game_win_prob']
        ]).reshape(1, -1)
