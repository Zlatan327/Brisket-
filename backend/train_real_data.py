"""
Real Data Backtesting
Train model on real NBA data and validate accuracy
"""

import pandas as pd
import numpy as np
from xgboost_model import XGBoostModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


def main():
    print("Real NBA Data Backtesting")
    print("=" * 60)
    
    # Load features
    print("\nLoading features...")
    features_df = pd.read_csv("features_2023_24.csv")
    print(f"Loaded {len(features_df)} games with {features_df.shape[1]} columns")
    
    # Define feature columns (37 features)
    feature_cols = [
        # Four Factors (8)
        'home_efg_pct', 'away_efg_pct',
        'home_tov_rate', 'away_tov_rate',
        'home_drb_pct', 'away_drb_pct',
        'home_ft_rate', 'away_ft_rate',
        
        # Advanced Metrics (4)
        'home_net_rating', 'away_net_rating',
        'home_pace', 'away_pace',
        
        # Rest & Travel (5)
        'home_rest_days', 'away_rest_days',
        'travel_distance', 'timezone_shift', 'elevation_change',
        
        # Fatigue (4)
        'home_fatigue_score', 'away_fatigue_score',
        'home_cumulative_load', 'away_cumulative_load',
        
        # Form (4)
        'home_last_5_wins', 'away_last_5_wins',
        'home_last_10_wins', 'away_last_10_wins',
        
        # Injuries (2)
        'home_injury_impact', 'away_injury_impact',
        
        # Sentiment (6)
        'home_sentiment_score', 'away_sentiment_score',
        'home_trade_rumors', 'away_trade_rumors',
        'home_coaching_stability', 'away_coaching_stability',
        
        # Tanking (2)
        'home_tanking_score', 'away_tanking_score',
        
        # Context (2)
        'is_back_to_back_home', 'is_back_to_back_away',
        'home_court_advantage'
    ]
    
    # Prepare data
    X = features_df[feature_cols].values
    y = features_df['home_win'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Home win rate: {y.mean():.1%}")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain set: {X_train.shape[0]} games")
    print(f"Val set: {X_val.shape[0]} games")
    print(f"Test set: {X_test.shape[0]} games")
    
    # Train XGBoost model
    print("\n" + "=" * 60)
    print("Training XGBoost Model on Real NBA Data")
    print("=" * 60)
    
    xgb_model = XGBoostModel()
    xgb_model.build_model()
    
    results = xgb_model.train(X_train, y_train, X_val, y_val)
    
    print(f"\nBest Iteration: {results.get('best_iteration', 'N/A')}")
    print(f"Best Score: {results.get('best_score', 'N/A')}")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    
    metrics = xgb_model.evaluate(X_test, y_test)
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.1%}")
    print(f"Test AUC: {metrics['auc']:.3f}")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Sample predictions
    print(f"\nSample Predictions:")
    for i in range(min(10, len(metrics['predictions_sample']))):
        pred = metrics['predictions_sample'][i]
        actual = y_test[i]
        print(f"  Game {i+1}: {pred:.1%} (actual: {'Home Win' if actual == 1 else 'Away Win'})")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("Top 10 Most Important Features")
    print("=" * 60)
    
    feature_names = feature_cols
    importance = xgb_model.get_feature_importance(feature_names)
    
    for i, (name, imp) in enumerate(list(importance.items())[:10], 1):
        print(f"{i:2d}. {name:30s} {imp:.3f}")
    
    # Save model
    print("\n" + "=" * 60)
    xgb_model.save_model("xgboost_nba_2023_24.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Trained on {len(X_train)} real NBA games")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.1%}")
    print(f"✓ Test AUC: {metrics['auc']:.3f}")
    print(f"✓ Model saved to xgboost_nba_2023_24.json")
    
    # Compare to target
    target_accuracy = 0.87
    if metrics['accuracy'] >= target_accuracy:
        print(f"\n🎉 EXCEEDS TARGET! ({metrics['accuracy']:.1%} >= {target_accuracy:.0%})")
    else:
        gap = target_accuracy - metrics['accuracy']
        print(f"\n⚠️  Below target by {gap:.1%} ({metrics['accuracy']:.1%} vs {target_accuracy:.0%})")


if __name__ == "__main__":
    main()
