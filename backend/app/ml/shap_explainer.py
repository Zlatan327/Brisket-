"""
SHAP Explainability Wrapper
Provides model interpretability using SHAP (SHapley Additive exPlanations)
"""

from typing import Dict, List, Optional, Any
import numpy as np


class SHAPExplainer:
    """
    Wrapper for SHAP explainability
    Explains model predictions by showing feature contributions
    """
    
    # Feature names for basketball predictions
    FEATURE_NAMES = [
        # Four Factors
        'home_efg_pct', 'away_efg_pct',
        'home_tov_rate', 'away_tov_rate',
        'home_drb_pct', 'away_drb_pct',
        'home_ft_rate', 'away_ft_rate',
        
        # Advanced metrics
        'home_net_rating', 'away_net_rating',
        'home_pace', 'away_pace',
        
        # Rest and travel
        'home_rest_days', 'away_rest_days',
        'travel_distance', 'timezone_shift',
        'elevation_change',
        
        # Fatigue
        'home_fatigue_score', 'away_fatigue_score',
        'home_cumulative_load', 'away_cumulative_load',
        
        # Form
        'home_last_5_wins', 'away_last_5_wins',
        'home_last_10_wins', 'away_last_10_wins',
        
        # Injuries
        'home_injury_impact', 'away_injury_impact',
        
        # Sentiment
        'home_sentiment_score', 'away_sentiment_score',
        'home_trade_rumors', 'away_trade_rumors',
        'home_coaching_stability', 'away_coaching_stability',
        
        # Tanking (NBA only)
        'home_tanking_score', 'away_tanking_score',
        
        # Context
        'is_back_to_back_home', 'is_back_to_back_away',
        'home_court_advantage',
    ]
    
    def __init__(self):
        """Initialize SHAP explainer"""
        # Will be set when model is loaded
        self.explainer = None
        self.model = None
    
    def initialize_explainer(self, model: Any, background_data: np.ndarray):
        """
        Initialize SHAP explainer with model and background data
        
        Args:
            model: Trained ML model (XGBoost, DNN, etc.)
            background_data: Sample of training data for baseline
        """
        try:
            import shap
            
            # Choose explainer based on model type
            if hasattr(model, 'predict_proba'):
                # Tree-based models (XGBoost, RandomForest)
                self.explainer = shap.TreeExplainer(model)
            else:
                # Deep learning models
                self.explainer = shap.DeepExplainer(model, background_data)
            
            self.model = model
            
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            self.explainer = None
    
    def explain_prediction(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Explain a single prediction using SHAP values
        
        Args:
            features: Feature vector for the game
            feature_names: Optional custom feature names
            
        Returns:
            Dictionary with SHAP explanation
        """
        if self.explainer is None:
            return self._mock_explanation(features, feature_names)
        
        try:
            import shap
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Get base value (expected value)
            base_value = self.explainer.expected_value
            
            # Create explanation dictionary
            if feature_names is None:
                feature_names = self.FEATURE_NAMES[:len(features)]
            
            # Sort features by absolute SHAP value
            feature_contributions = []
            for i, (name, value, shap_val) in enumerate(zip(feature_names, features, shap_values)):
                feature_contributions.append({
                    'feature': name,
                    'value': float(value),
                    'shap_value': float(shap_val),
                    'contribution_pct': abs(float(shap_val)) / sum(abs(shap_values)) * 100
                })
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            return {
                'base_value': float(base_value),
                'prediction': float(base_value + sum(shap_values)),
                'feature_contributions': feature_contributions,
                'top_5_features': feature_contributions[:5]
            }
        
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return self._mock_explanation(features, feature_names)
    
    def _mock_explanation(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Create mock SHAP explanation for testing
        (Used when SHAP is not installed)
        
        Args:
            features: Feature vector
            feature_names: Feature names
            
        Returns:
            Mock explanation dictionary
        """
        if feature_names is None:
            feature_names = self.FEATURE_NAMES[:len(features)]
        
        # Generate mock SHAP values (random but realistic)
        np.random.seed(42)
        mock_shap_values = np.random.randn(len(features)) * 0.05
        
        base_value = 0.5  # 50% baseline win probability
        
        feature_contributions = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, features, mock_shap_values)):
            feature_contributions.append({
                'feature': name,
                'value': float(value),
                'shap_value': float(shap_val),
                'contribution_pct': abs(float(shap_val)) / sum(abs(mock_shap_values)) * 100
            })
        
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'base_value': base_value,
            'prediction': base_value + sum(mock_shap_values),
            'feature_contributions': feature_contributions,
            'top_5_features': feature_contributions[:5],
            'note': 'Mock explanation - SHAP not installed'
        }
    
    def format_explanation_for_ui(self, explanation: Dict) -> str:
        """
        Format SHAP explanation for user-friendly display
        
        Args:
            explanation: SHAP explanation dictionary
            
        Returns:
            Formatted string
        """
        output = []
        output.append("=" * 60)
        output.append("PREDICTION EXPLANATION (SHAP)")
        output.append("=" * 60)
        
        output.append(f"\nBase Win Probability: {explanation['base_value']:.1%}")
        output.append(f"Final Prediction: {explanation['prediction']:.1%}")
        output.append(f"Total Adjustment: {(explanation['prediction'] - explanation['base_value']):.1%}")
        
        output.append("\n" + "-" * 60)
        output.append("TOP 5 CONTRIBUTING FACTORS:")
        output.append("-" * 60)
        
        for i, feature in enumerate(explanation['top_5_features'], 1):
            impact = "+" if feature['shap_value'] > 0 else ""
            bar_length = int(abs(feature['contribution_pct']) / 2)
            bar = "█" * bar_length
            
            output.append(f"\n{i}. {feature['feature']}")
            output.append(f"   Value: {feature['value']:.3f}")
            output.append(f"   Impact: {impact}{feature['shap_value']:.1%}  {bar}")
            output.append(f"   Contribution: {feature['contribution_pct']:.1f}%")
        
        output.append("\n" + "=" * 60)
        
        return "\n".join(output)
    
    def get_feature_importance(self, shap_values: np.ndarray) -> Dict:
        """
        Calculate overall feature importance from SHAP values
        
        Args:
            shap_values: SHAP values for multiple predictions
            
        Returns:
            Dictionary with feature importance
        """
        # Mean absolute SHAP value for each feature
        importance = np.abs(shap_values).mean(axis=0)
        
        feature_importance = {}
        for i, name in enumerate(self.FEATURE_NAMES[:len(importance)]):
            feature_importance[name] = float(importance[i])
        
        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance


# Example usage
if __name__ == "__main__":
    explainer = SHAPExplainer()
    
    print("SHAP Explainability Test")
    print("=" * 60)
    
    # Mock game features (Lakers vs Celtics)
    game_features = np.array([
        # Four Factors (home, away)
        0.541, 0.489,  # eFG%
        0.113, 0.135,  # TOV%
        0.750, 0.720,  # DRB%
        0.235, 0.205,  # FT Rate
        
        # Advanced metrics
        7.0, -7.0,     # Net Rating
        100.0, 100.0,  # Pace
        
        # Rest and travel
        2, 0,          # Rest days (Celtics on back-to-back)
        2591, 3, 280,  # Travel distance, timezone, elevation
        
        # Fatigue
        30.0, 85.0,    # Fatigue scores
        40.0, 100.0,   # Cumulative load
        
        # Form
        4, 2,          # Last 5 wins
        7, 5,          # Last 10 wins
        
        # Injuries
        15.0, 25.0,    # Injury impact
        
        # Sentiment
        0.61, -0.17,   # Sentiment scores
        0.0, 0.20,     # Trade rumors
        1.0, 1.0,      # Coaching stability
        
        # Tanking
        0.0, 0.0,      # Tanking scores
        
        # Context
        0, 1,          # Back-to-back flags
        1.0            # Home court advantage
    ])
    
    # Get explanation
    explanation = explainer.explain_prediction(game_features)
    
    # Format and print
    formatted = explainer.format_explanation_for_ui(explanation)
    print(formatted)
    
    print("\n\nAll Feature Contributions:")
    print("-" * 60)
    for feature in explanation['feature_contributions'][:10]:
        print(f"{feature['feature']:30s} {feature['shap_value']:+.3f}")
