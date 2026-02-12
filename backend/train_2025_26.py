"""
Collect 2025-26 Season Completed Games
Train on current season's completed games
"""

from historical_data_collector import HistoricalDataCollector
from enhanced_feature_engineer import EnhancedFeatureEngineer
from xgboost_model import XGBoostModel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def collect_and_train_2025_26():
    print("2025-26 NBA Season - Current Season Training")
    print("=" * 60)
    print("Collecting completed games from 2025-26 season")
    print("(Season in progress - February 2026)")
    print("=" * 60)
    
    collector = HistoricalDataCollector()
    
    # Collect 2025-26 season (current season)
    print("\nFetching 2025-26 season from NBA API...")
    games_2025_26 = collector.get_games_by_season("2025-26")
    
    if not games_2025_26.empty:
        print(f"\n✓ Collected {len(games_2025_26)} game records")
        
        # Process games
        processed_df = collector.process_games_for_backtesting(games_2025_26)
        print(f"✓ Processed {len(processed_df)} unique games")
        
        # Save raw data
        collector.save_to_csv(processed_df, "historical_games_2025_26.csv")
        
        # Engineer features
        print("\nEngineering features with real travel data...")
        engineer = EnhancedFeatureEngineer()
        features_df = engineer.engineer_features(processed_df)
        
        # Save features
        features_df.to_csv("features_2025_26_enhanced.csv", index=False)
        print(f"\n✓ Features saved to features_2025_26_enhanced.csv")
        
        # Show date range
        print(f"\nDate range: {processed_df['date'].min()} to {processed_df['date'].max()}")
        print(f"Games completed so far: {len(processed_df)}")
        
        # Train model on this data
        print("\n" + "=" * 60)
        print("TRAINING MODEL ON 2025-26 COMPLETED GAMES")
        print("=" * 60)
        
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
        
        X = features_df[feature_cols].values
        y = features_df['home_win'].values
        
        print(f"\nDataset: {X.shape[0]} games")
        print(f"Home win rate: {y.mean():.1%}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        # Train model
        print("\nTraining XGBoost on 2025-26 season...")
        xgb_model = XGBoostModel()
        xgb_model.build_model()
        results = xgb_model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        
        metrics = xgb_model.evaluate(X_test, y_test)
        
        print(f"\nTest Accuracy: {metrics['accuracy']:.1%}")
        print(f"Test AUC: {metrics['auc']:.3f}")
        
        # Save model
        xgb_model.save_model("xgboost_2025_26_current.json")
        print(f"\n✓ Model saved to xgboost_2025_26_current.json")
        
        # Feature importance
        print("\n" + "=" * 60)
        print("TOP 10 FEATURES (2025-26 Season)")
        print("=" * 60)
        
        importance = xgb_model.get_feature_importance(feature_cols)
        for i, (name, imp) in enumerate(list(importance.items())[:10], 1):
            print(f"{i:2d}. {name:30s} {imp:.3f}")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"✓ Trained on {len(processed_df)} games from 2025-26 season")
        print(f"✓ Test Accuracy: {metrics['accuracy']:.1%}")
        print(f"✓ Model ready to predict remaining 2025-26 games")
        
        return processed_df, features_df, metrics
        
    else:
        print("\n✗ No 2025-26 season data available")
        return None, None, None


if __name__ == "__main__":
    games_df, features_df, metrics = collect_and_train_2025_26()
