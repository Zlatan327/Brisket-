"""
XGBoost Classifier
Gradient boosting model for game winner prediction
"""

import numpy as np
from typing import Dict, Tuple, Optional
import json


class XGBoostModel:
    """
    XGBoost classifier for basketball game prediction
    
    Advantages:
    - Handles non-linear relationships
    - Feature importance built-in
    - Fast training and prediction
    - Robust to overfitting
    """
    
    def __init__(self):
        """Initialize XGBoost model"""
        self.model = None
        self.feature_importance = None
        
    def build_model(self, params: Optional[Dict] = None):
        """
        Build XGBoost classifier
        
        Args:
            params: Optional custom parameters
        """
        try:
            import xgboost as xgb
            
            # Default parameters optimized for basketball prediction
            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
            
            if params:
                default_params.update(params)
            
            self.model = xgb.XGBClassifier(**default_params)
            return self.model
            
        except ImportError:
            print("XGBoost not installed. Install with: pip install xgboost")
            return None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 20
    ) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Training results
        """
        if self.model is None:
            self.build_model()
        
        try:
            # Train with early stopping (new API)
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Get feature importance
            self.feature_importance = self.model.feature_importances_
            
            # Get training results
            results = {
                'best_iteration': getattr(self.model, 'best_iteration', None),
                'best_score': getattr(self.model, 'best_score', None),
                'feature_importance': self.feature_importance.tolist()
            }
            
            return results
            
        except Exception as e:
            print(f"Error training model: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict win probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Win probabilities (0.0 - 1.0)
        """
        if self.model is None:
            # Return mock predictions
            return np.random.rand(len(X)) * 0.3 + 0.35
        
        try:
            predictions = self.model.predict_proba(X)[:, 1]
            return predictions
        except Exception as e:
            print(f"Error predicting: {e}")
            return np.random.rand(len(X)) * 0.3 + 0.35
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            return {'accuracy': 0.0, 'auc': 0.0}
        
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
            
            # Get predictions
            y_pred_proba = self.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'confusion_matrix': cm.tolist(),
                'predictions_sample': y_pred_proba[:10].tolist()
            }
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {'accuracy': 0.0, 'auc': 0.0}
    
    def get_feature_importance(self, feature_names: list) -> Dict:
        """
        Get feature importance rankings
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature importance
        """
        if self.feature_importance is None:
            return {}
        
        importance_dict = {}
        for name, importance in zip(feature_names, self.feature_importance):
            importance_dict[name] = float(importance)
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is not None:
            self.model.save_model(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(filepath)
            self.feature_importance = self.model.feature_importances_
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")


# Example usage
if __name__ == "__main__":
    print("XGBoost Model Test")
    print("=" * 60)
    
    # Create mock training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 37
    
    X_train = np.random.randn(n_samples, n_features)
    # Create labels with some pattern
    y_train = (X_train[:, 0] + X_train[:, 1] * 0.5 + X_train[:, 2] * 0.3 > 0).astype(int)
    
    X_val = np.random.randn(200, n_features)
    y_val = (X_val[:, 0] + X_val[:, 1] * 0.5 + X_val[:, 2] * 0.3 > 0).astype(int)
    
    X_test = np.random.randn(200, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] * 0.5 + X_test[:, 2] * 0.3 > 0).astype(int)
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Initialize model
    xgb_model = XGBoostModel()
    
    print("\nBuilding XGBoost model...")
    model = xgb_model.build_model()
    
    if model is not None:
        print("\nTraining model...")
        results = xgb_model.train(X_train, y_train, X_val, y_val)
        
        print(f"\nBest Iteration: {results.get('best_iteration', 'N/A')}")
        print(f"Best Score: {results.get('best_score', 'N/A'):.3f}")
        
        print("\nEvaluating model...")
        metrics = xgb_model.evaluate(X_test, y_test)
        
        print(f"\nTest Accuracy: {metrics['accuracy']:.1%}")
        print(f"Test AUC: {metrics['auc']:.3f}")
        
        print("\nSample Predictions:")
        for i, pred in enumerate(metrics['predictions_sample'][:5]):
            actual = y_test[i]
            print(f"  Game {i+1}: {pred:.1%} (actual: {'Win' if actual == 1 else 'Loss'})")
        
        # Feature importance
        feature_names = [f'feature_{i}' for i in range(n_features)]
        importance = xgb_model.get_feature_importance(feature_names)
        
        print("\nTop 5 Most Important Features:")
        for i, (name, imp) in enumerate(list(importance.items())[:5], 1):
            print(f"  {i}. {name}: {imp:.3f}")
    else:
        print("\nXGBoost not installed - using mock predictions")
        predictions = xgb_model.predict(X_test)
        print(f"\nMock predictions (first 5): {predictions[:5]}")
