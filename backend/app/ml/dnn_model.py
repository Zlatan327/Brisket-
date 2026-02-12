"""
Deep Neural Network (DNN) Model
Binary classification for game winner prediction
"""

import numpy as np
from typing import Dict, Tuple, Optional
import json


class DNNModel:
    """
    Deep Neural Network for basketball game prediction
    
    Architecture:
    - Input layer: 37 features
    - Hidden layer 1: 128 neurons (ReLU)
    - Dropout: 0.3
    - Hidden layer 2: 64 neurons (ReLU)
    - Dropout: 0.3
    - Hidden layer 3: 32 neurons (ReLU)
    - Output layer: 1 neuron (Sigmoid) - win probability
    """
    
    def __init__(self, input_dim: int = 37):
        """
        Initialize DNN model
        
        Args:
            input_dim: Number of input features
        """
        self.input_dim = input_dim
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the DNN architecture"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            model = keras.Sequential([
                # Input layer
                layers.Input(shape=(self.input_dim,)),
                
                # Hidden layer 1
                layers.Dense(128, activation='relu', name='hidden_1'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # Hidden layer 2
                layers.Dense(64, activation='relu', name='hidden_2'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # Hidden layer 3
                layers.Dense(32, activation='relu', name='hidden_3'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                # Output layer
                layers.Dense(1, activation='sigmoid', name='output')
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
            
            self.model = model
            return model
            
        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow")
            return None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict:
        """
        Train the DNN model
        
        Args:
            X_train: Training features
            y_train: Training labels (0 or 1)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        try:
            from tensorflow import keras
            
            # Early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Learning rate reduction
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            self.history = history.history
            return self.history
            
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
            return np.random.rand(len(X)) * 0.3 + 0.35  # 0.35-0.65 range
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
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
            return {'accuracy': 0.0, 'auc': 0.0, 'loss': 0.0}
        
        try:
            loss, accuracy, auc = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Get predictions for additional metrics
            y_pred_proba = self.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix, classification_report
            
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                'loss': float(loss),
                'accuracy': float(accuracy),
                'auc': float(auc),
                'confusion_matrix': cm.tolist(),
                'predictions_sample': y_pred_proba[:10].tolist()
            }
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {'accuracy': 0.0, 'auc': 0.0, 'loss': 0.0}
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")


# Example usage
if __name__ == "__main__":
    print("DNN Model Test")
    print("=" * 60)
    
    # Create mock training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 37
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)  # Simple rule
    
    X_val = np.random.randn(200, n_features)
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
    
    X_test = np.random.randn(200, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Initialize model
    dnn = DNNModel(input_dim=n_features)
    
    print("\nBuilding DNN model...")
    model = dnn.build_model()
    
    if model is not None:
        print("\nModel Architecture:")
        model.summary()
        
        print("\nTraining model...")
        history = dnn.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
        
        print("\nEvaluating model...")
        metrics = dnn.evaluate(X_test, y_test)
        
        print(f"\nTest Accuracy: {metrics['accuracy']:.1%}")
        print(f"Test AUC: {metrics['auc']:.3f}")
        print(f"Test Loss: {metrics['loss']:.3f}")
        
        print("\nSample Predictions:")
        for i, pred in enumerate(metrics['predictions_sample'][:5]):
            print(f"  Game {i+1}: {pred:.1%} win probability")
    else:
        print("\nTensorFlow not installed - using mock predictions")
        predictions = dnn.predict(X_test)
        print(f"\nMock predictions (first 5): {predictions[:5]}")
