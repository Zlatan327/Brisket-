"""
Fast Improved Model Training
Uses RandomizedSearchCV for faster hyperparameter optimization
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import time


def train_fast_improved_model():
    print("Fast Improved Model Training (RandomizedSearchCV)")
    print("=" * 60)
    start_time = time.time()
    
    # Load combined dataset
    try:
        features_2024_25 = pd.read_csv("features_2024_25_enhanced.csv")
        features_2025_26 = pd.read_csv("features_2025_26_enhanced.csv")
        combined_df = pd.concat([features_2024_25, features_2025_26], ignore_index=True)
        print(f"✓ Loaded {len(combined_df)} games")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, 0

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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Hyperparameter tuning (RandomizedSearchCV)
    print("\nOptimizing hyperparameters (20 iterations)...")
    
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=20,
        cv=3, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
        
    # Evaluate
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Accuracy: {accuracy:.1%}")
    print(f"Test AUC: {auc:.3f}")
    
    # Save model
    best_model.save_model("xgboost_improved_2024_26.json")
    
    print(f"\n✓ Model saved")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    return best_model, accuracy

if __name__ == "__main__":
    train_fast_improved_model()
