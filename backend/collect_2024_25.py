"""
Collect 2024-25 Season Data
Most recent complete season for training
"""

from historical_data_collector import HistoricalDataCollector
from enhanced_feature_engineer import EnhancedFeatureEngineer
import pandas as pd


def collect_2024_25_season():
    print("Collecting 2024-25 NBA Season")
    print("=" * 60)
    print("This is the most recent COMPLETE season")
    print("We'll use this to predict 2025-26 season games")
    print("=" * 60)
    
    collector = HistoricalDataCollector()
    
    # Collect 2024-25 season
    print("\nFetching 2024-25 season from NBA API...")
    games_2024_25 = collector.get_games_by_season("2024-25")
    
    if not games_2024_25.empty:
        print(f"\n✓ Collected {len(games_2024_25)} game records")
        
        # Process for backtesting
        processed_df = collector.process_games_for_backtesting(games_2024_25)
        print(f"✓ Processed {len(processed_df)} unique games")
        
        # Save raw data
        collector.save_to_csv(processed_df, "historical_games_2024_25.csv")
        
        # Engineer features with real travel data
        print("\nEngineering features with real travel data...")
        engineer = EnhancedFeatureEngineer()
        features_df = engineer.engineer_features(processed_df)
        
        # Save features
        features_df.to_csv("features_2024_25_enhanced.csv", index=False)
        print(f"\n✓ Features saved to features_2024_25_enhanced.csv")
        
        # Show sample
        if len(processed_df) > 0:
            sample = processed_df.iloc[0]
            print(f"\nSample game from 2024-25:")
            print(f"  {sample['away_team_name']} @ {sample['home_team_name']}")
            print(f"  Score: {sample['away_score']} - {sample['home_score']}")
            print(f"  Date: {sample['date']}")
            
            # Show feature sample
            feat_sample = features_df.iloc[0]
            print(f"\n  Travel: {feat_sample['travel_distance']:.0f} miles")
            print(f"  Timezone: {feat_sample['timezone_shift']} hours")
            print(f"  Elevation: {feat_sample['elevation_change']:,.0f} feet")
        
        return processed_df, features_df
    else:
        print("\n✗ No data collected - API may not have 2024-25 data yet")
        return None, None


if __name__ == "__main__":
    games_df, features_df = collect_2024_25_season()
    
    if games_df is not None:
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total games: {len(games_df)}")
        print(f"Features: {features_df.shape[1]}")
        print(f"Date range: {games_df['date'].min()} to {games_df['date'].max()}")
        print("\n✓ Ready to train model on 2024-25 season!")
    else:
        print("\n⚠️  2024-25 season data not available")
        print("Will use 2022-24 seasons for training")
