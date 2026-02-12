"""
Improved Model with Hyperparameter Tuning
Optimizes XGBoost parameters for better accuracy
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json


def train_improved_model():
    print("Improved Model Training with Hyperparameter Tuning")
    print("=" * 60)
    
    # Load combined dataset
    print("\nLoading data...")
    features_2024_25 = pd.read_csv("features_2024_25_enhanced.csv")
    features_2025_26 = pd.read_csv("features_2025_26_enhanced.csv")
    
    combined_df = pd.concat([features_2024_25, features_2025_26], ignore_index=True)
    print(f"✓ Combined dataset: {len(combined_df)} games")
    
    # Feature columns
    feature_cols = [
        'home_efg_pct', 'away_efg_pct', 'home_tov_rate', 'away_tov_rate',
        'home_drb_pct', 'away_drb_pct', 'home_ft_rate', 'away_ft_rate',
        'home_net_rating', 'away_net_rating', 'home_pace', 'away_pace',
        'home_rest_days', 'away_rest_days', 'travel_distance', 'timezone_shift', 
        'elevation_change', 'home_fatigue_score', 'away_fatigue_score',
        'home_cumulative_load', 'away_cumulative_load', 'home_last_5_wins', 
        'away_last_5_wins', 'home_last_10_wins', 'away_last_10_wins',
        'home_injury_impact', 'away_injury_impact', 'home_sentiment_score', 
        'away_sentiment_score', 'home_trade_rumors', 'away_trade_rumors',
        'home_coaching_stability', 'away_coaching_stability', 'home_tanking_score', 
        'away_tanking_score', 'is_back_to_back_home', 'is_back_to_back_away',
        'home_court_advantage'
    ]
    
    X = combined_df[feature_cols].values
    y = combined_df['home_win'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Hyperparameter tuning
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    print("\nSearching for best parameters...")
    print("(This may take a few minutes)")
    
    grid_search = GridSearchCV(
        xgb, param_grid, cv=3, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV accuracy: {grid_search.best_score_:.1%}")
    
    # Train final model with best parameters
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Accuracy: {accuracy:.1%}")
    print(f"Test AUC: {auc:.3f}")
    
    # High-confidence predictions
    print("\n" + "=" * 60)
    print("HIGH-CONFIDENCE PERFORMANCE")
    print("=" * 60)
    
    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
        high_conf_mask = (y_pred_proba > threshold) | (y_pred_proba < (1 - threshold))
        
        if high_conf_mask.sum() > 0:
            y_high_conf = y_test[high_conf_mask]
            y_pred_high_conf = (y_pred_proba[high_conf_mask] > 0.5).astype(int)
            
            acc = accuracy_score(y_high_conf, y_pred_high_conf)
            coverage = high_conf_mask.sum() / len(y_test) * 100
            
            print(f"{int(threshold*100)}% confidence: {acc:.1%} accuracy ({coverage:.1f}% coverage)")
    
    # Save model
    best_model.save_model("xgboost_improved_2024_26.json")
    
    # Save parameters
    with open("model_config.json", "w") as f:
        json.dump({
            "best_params": grid_search.best_params_,
            "cv_accuracy": float(grid_search.best_score_),
            "test_accuracy": float(accuracy),
            "test_auc": float(auc),
            "training_games": len(combined_df),
            "feature_count": len(feature_cols)
        }, f, indent=2)
    
    print(f"\n✓ Model saved to xgboost_improved_2024_26.json")
    print(f"✓ Config saved to model_config.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"Previous model: 64.6%")
    print(f"Improved model: {accuracy:.1%}")
    print(f"Gain: +{(accuracy - 0.646)*100:.1f}%")
    
    return best_model, accuracy


if __name__ == "__main__":
    model, accuracy = train_improved_model()
