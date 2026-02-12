"""
Combined Multi-Season Training
Train on 2024-25 + 2025-26 seasons for better accuracy
"""

import pandas as pd
import numpy as np
from xgboost_model import XGBoostModel
from sklearn.model_selection import train_test_split


def train_combined_seasons():
    print("Combined Multi-Season Training")
    print("=" * 60)
    print("Training on 2024-25 + 2025-26 seasons")
    print("=" * 60)
    
    # Load both seasons
    print("\nLoading season data...")
    
    try:
        features_2024_25 = pd.read_csv("features_2024_25_enhanced.csv")
        print(f"✓ 2024-25 season: {len(features_2024_25)} games")
    except:
        print("✗ 2024-25 season not found")
        features_2024_25 = pd.DataFrame()
    
    try:
        features_2025_26 = pd.read_csv("features_2025_26_enhanced.csv")
        print(f"✓ 2025-26 season: {len(features_2025_26)} games")
    except:
        print("✗ 2025-26 season not found")
        features_2025_26 = pd.DataFrame()
    
    # Combine datasets
    if not features_2024_25.empty and not features_2025_26.empty:
        combined_df = pd.concat([features_2024_25, features_2025_26], ignore_index=True)
        print(f"\n✓ Combined dataset: {len(combined_df)} games")
    elif not features_2024_25.empty:
        combined_df = features_2024_25
        print(f"\n✓ Using 2024-25 only: {len(combined_df)} games")
    elif not features_2025_26.empty:
        combined_df = features_2025_26
        print(f"\n✓ Using 2025-26 only: {len(combined_df)} games")
    else:
        print("\n✗ No data available")
        return None
    
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
    
    # Prepare data
    X = combined_df[feature_cols].values
    y = combined_df['home_win'].values
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"Home win rate: {y.mean():.1%}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST ON COMBINED SEASONS")
    print("=" * 60)
    
    xgb_model = XGBoostModel()
    xgb_model.build_model()
    results = xgb_model.train(X_train, y_train, X_val, y_val)
    
    print(f"\nBest iteration: {results.get('best_iteration', 'N/A')}")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    
    metrics = xgb_model.evaluate(X_test, y_test)
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.1%}")
    print(f"Test AUC: {metrics['auc']:.3f}")
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Save model
    xgb_model.save_model("xgboost_combined_2024_26.json")
    print(f"\n✓ Model saved to xgboost_combined_2024_26.json")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES (Combined Seasons)")
    print("=" * 60)
    
    importance = xgb_model.get_feature_importance(feature_cols)
    for i, (name, imp) in enumerate(list(importance.items())[:10], 1):
        print(f"{i:2d}. {name:30s} {imp:.3f}")
    
    # Per-night accuracy simulation
    print("\n" + "=" * 60)
    print("ESTIMATED PER-NIGHT ACCURACY")
    print("=" * 60)
    
    y_pred_proba = xgb_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Simulate typical nights
    typical_nights = [5, 6, 7, 10, 12]
    
    for n_games in typical_nights:
        expected_correct = n_games * metrics['accuracy']
        print(f"{n_games}-game night: {expected_correct:.1f}/{n_games} correct ({metrics['accuracy']:.1%})")
    
    # High-confidence filtering
    print("\n" + "=" * 60)
    print("HIGH-CONFIDENCE PREDICTIONS")
    print("=" * 60)
    
    confidence_thresholds = [0.60, 0.65, 0.70, 0.75]
    
    for threshold in confidence_thresholds:
        high_conf_mask = (y_pred_proba > threshold) | (y_pred_proba < (1 - threshold))
        
        if high_conf_mask.sum() > 0:
            y_high_conf = y_test[high_conf_mask]
            y_pred_high_conf = (y_pred_proba[high_conf_mask] > 0.5).astype(int)
            
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_high_conf, y_pred_high_conf)
            coverage = high_conf_mask.sum() / len(y_test) * 100
            
            print(f"{int(threshold*100)}% confidence: {acc:.1%} accuracy ({coverage:.1f}% coverage)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Trained on {len(combined_df)} games (2024-26 seasons)")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.1%}")
    print(f"✓ Test AUC: {metrics['auc']:.3f}")
    print(f"✓ Model ready for 2025-26 season predictions")
    
    return metrics


if __name__ == "__main__":
    metrics = train_combined_seasons()
