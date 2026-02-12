"""
Q4 Prediction Model
Specialized XGBoost model for 4th quarter outcomes
"""

import numpy as np
import xgboost as xgb
from typing import Dict, List, Optional
import os
import pickle

class Q4PredictionModel:
    """
    Predicts game winner entering Q4 features.
    optimized for "micro-predictions" (fast, sensitive to current score).
    """
    
    def __init__(self, model_path: str = "models/q4_model_v1.json"):
        """
        Initialize Q4 model
        
        Args:
            model_path: Path to load/save model
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = [
            'score_diff', 
            'home_clutch_net_rating', 
            'away_clutch_net_rating',
            'home_star_usage',
            'away_star_usage',
            'pace', 
            'pre_game_win_prob'
        ]
        
    def build_model(self):
        """Build and configure the model"""
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=4,         # Shallower than main model
            learning_rate=0.05,
            n_estimators=100,    # Fewer trees for micro-model
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Save after training
        self.save_model()
        
        return {
            'best_score': self.model.best_score,
            'feature_importances': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict(self, X: np.ndarray) -> float:
        """
        Predict Home Win Probability
        
        Args:
            X: Feature array (shape 1, n_features)
            
        Returns:
            Probability of Home Win (0.0 - 1.0)
        """
        if self.model is None:
            # Fallback simple logic if model not loaded
            # Logistic function on score diff + pre_game_prob
            score_diff = X[0][0]
            pre_game = X[0][6]
            
            # Simple sigmoid approximation
            # If score diff is 0, prob is close to pre_game
            # +10 points -> ~90% win prob?
            
            z = score_diff * 0.15 + (pre_game - 0.5) * 2
            prob = 1 / (1 + np.exp(-z))
            return float(prob)
            
        return float(self.model.predict_proba(X)[:, 1][0])
    
    def save_model(self):
        """Save model to disk"""
        if self.model:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_model(self.model_path)
    
    def load_model(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            print(f"Q4 Model loaded from {self.model_path}")
            return True
        return False
