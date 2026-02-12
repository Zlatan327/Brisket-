"""
Ensemble Voting System
Combines DNN and XGBoost predictions for improved accuracy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .dnn_model import DNNModel
from .xgboost_model import XGBoostModel


class EnsemblePredictor:
    """
    Ensemble model combining DNN and XGBoost
    
    Voting Strategy:
    - Weighted average of model probabilities
    - DNN weight: 0.5
    - XGBoost weight: 0.5
    - Can be adjusted based on validation performance
    """
    
    def __init__(self):
        """Initialize ensemble predictor"""
        self.dnn_model = DNNModel()
        self.xgb_model = XGBoostModel()
        
        # Default weights (can be optimized)
        self.weights = {
            'dnn': 0.5,
            'xgb': 0.5
        }
        
        self.is_trained = False
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """
        Train both models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training results for both models
        """
        print("Training DNN model...")
        dnn_results = self.dnn_model.train(
            X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32
        )
        
        print("\nTraining XGBoost model...")
        xgb_results = self.xgb_model.train(
            X_train, y_train, X_val, y_val,
            early_stopping_rounds=20
        )
        
        # Optimize weights based on validation performance
        self._optimize_weights(X_val, y_val)
        
        self.is_trained = True
        
        return {
            'dnn': dnn_results,
            'xgb': xgb_results,
            'optimized_weights': self.weights
        }
    
    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Optimize ensemble weights based on validation performance
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        try:
            from sklearn.metrics import roc_auc_score
            
            # Get individual predictions
            dnn_pred = self.dnn_model.predict(X_val)
            xgb_pred = self.xgb_model.predict(X_val)
            
            # Calculate individual AUC scores
            dnn_auc = roc_auc_score(y_val, dnn_pred)
            xgb_auc = roc_auc_score(y_val, xgb_pred)
            
            # Weight by relative performance
            total_auc = dnn_auc + xgb_auc
            self.weights['dnn'] = dnn_auc / total_auc
            self.weights['xgb'] = xgb_auc / total_auc
            
            print(f"\nOptimized Weights:")
            print(f"  DNN: {self.weights['dnn']:.3f} (AUC: {dnn_auc:.3f})")
            print(f"  XGBoost: {self.weights['xgb']:.3f} (AUC: {xgb_auc:.3f})")
            
        except Exception as e:
            print(f"Error optimizing weights: {e}")
            # Keep default 50/50 weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using ensemble
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble win probabilities
        """
        # Get predictions from both models
        dnn_pred = self.dnn_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        
        # Weighted average
        ensemble_pred = (
            self.weights['dnn'] * dnn_pred +
            self.weights['xgb'] * xgb_pred
        )
        
        return ensemble_pred
    
    def predict_with_details(self, X: np.ndarray) -> Dict:
        """
        Predict with detailed breakdown
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with all predictions
        """
        dnn_pred = self.dnn_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        ensemble_pred = self.predict(X)
        
        return {
            'dnn_predictions': dnn_pred.tolist(),
            'xgb_predictions': xgb_pred.tolist(),
            'ensemble_predictions': ensemble_pred.tolist(),
            'weights': self.weights
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate ensemble on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics for all models
        """
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
            
            # Get predictions
            dnn_pred = self.dnn_model.predict(X_test)
            xgb_pred = self.xgb_model.predict(X_test)
            ensemble_pred = self.predict(X_test)
            
            # Convert to binary
            dnn_binary = (dnn_pred > 0.5).astype(int)
            xgb_binary = (xgb_pred > 0.5).astype(int)
            ensemble_binary = (ensemble_pred > 0.5).astype(int)
            
            # Calculate metrics
            results = {
                'dnn': {
                    'accuracy': float(accuracy_score(y_test, dnn_binary)),
                    'auc': float(roc_auc_score(y_test, dnn_pred))
                },
                'xgb': {
                    'accuracy': float(accuracy_score(y_test, xgb_binary)),
                    'auc': float(roc_auc_score(y_test, xgb_pred))
                },
                'ensemble': {
                    'accuracy': float(accuracy_score(y_test, ensemble_binary)),
                    'auc': float(roc_auc_score(y_test, ensemble_pred)),
                    'confusion_matrix': confusion_matrix(y_test, ensemble_binary).tolist()
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error evaluating ensemble: {e}")
            return {}
    
    def get_confidence_level(self, probability: float) -> str:
        """
        Convert probability to confidence level
        
        Args:
            probability: Win probability (0.0 - 1.0)
            
        Returns:
            Confidence level string
        """
        if probability >= 0.70 or probability <= 0.30:
            return "HIGH"
        elif probability >= 0.60 or probability <= 0.40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def save_models(self, dnn_path: str, xgb_path: str):
        """Save both models"""
        self.dnn_model.save_model(dnn_path)
        self.xgb_model.save_model(xgb_path)
        print(f"Models saved to {dnn_path} and {xgb_path}")
    
    def load_models(self, dnn_path: str, xgb_path: str):
        """Load both models"""
        self.dnn_model.load_model(dnn_path)
        self.xgb_model.load_model(xgb_path)
        self.is_trained = True
        print(f"Models loaded from {dnn_path} and {xgb_path}")


# Example usage
if __name__ == "__main__":
    print("Ensemble Predictor Test")
    print("=" * 60)
    
    # Create mock data
    np.random.seed(42)
    n_samples = 1000
    n_features = 37
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] * 0.5 > 0).astype(int)
    
    X_val = np.random.randn(200, n_features)
    y_val = (X_val[:, 0] + X_val[:, 1] * 0.5 > 0).astype(int)
    
    X_test = np.random.randn(200, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] * 0.5 > 0).astype(int)
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Initialize ensemble
    ensemble = EnsemblePredictor()
    
    print("\nTraining ensemble...")
    print("=" * 60)
    
    # Note: This will use mock predictions if TensorFlow/XGBoost not installed
    results = ensemble.train(X_train, y_train, X_val, y_val)
    
    print("\n\nEvaluating ensemble...")
    print("=" * 60)
    
    metrics = ensemble.evaluate(X_test, y_test)
    
    if metrics:
        print("\nDNN Performance:")
        print(f"  Accuracy: {metrics['dnn']['accuracy']:.1%}")
        print(f"  AUC: {metrics['dnn']['auc']:.3f}")
        
        print("\nXGBoost Performance:")
        print(f"  Accuracy: {metrics['xgb']['accuracy']:.1%}")
        print(f"  AUC: {metrics['xgb']['auc']:.3f}")
        
        print("\nEnsemble Performance:")
        print(f"  Accuracy: {metrics['ensemble']['accuracy']:.1%}")
        print(f"  AUC: {metrics['ensemble']['auc']:.3f}")
        
        # Sample predictions
        print("\n\nSample Predictions:")
        print("=" * 60)
        sample_pred = ensemble.predict_with_details(X_test[:5])
        
        for i in range(5):
            dnn = sample_pred['dnn_predictions'][i]
            xgb = sample_pred['xgb_predictions'][i]
            ens = sample_pred['ensemble_predictions'][i]
            actual = y_test[i]
            confidence = ensemble.get_confidence_level(ens)
            
            print(f"\nGame {i+1}:")
            print(f"  DNN: {dnn:.1%}")
            print(f"  XGBoost: {xgb:.1%}")
            print(f"  Ensemble: {ens:.1%} ({confidence} confidence)")
            print(f"  Actual: {'Win' if actual == 1 else 'Loss'}")
