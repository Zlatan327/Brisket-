"""
Multi-Season Data Collector
Collects NBA data from 2020-2023 seasons
"""

from historical_data_collector import HistoricalDataCollector
import time

def collect_multi_season_data():
    print("Multi-Season NBA Data Collection")
    print("=" * 60)
    
    collector = HistoricalDataCollector()
    
    # Collect 2022-23 season (most recent complete season before 2023-24)
    print("\nCollecting 2022-23 season...")
    print("-" * 60)
    
    games_2022_23 = collector.get_games_by_season("2022-23")
    
    if not games_2022_23.empty:
        print(f"\n✓ Collected {len(games_2022_23)} game records")
        
        # Process for backtesting
        processed_df = collector.process_games_for_backtesting(games_2022_23)
        
        print(f"✓ Processed {len(processed_df)} unique games")
        
        # Save to CSV
        collector.save_to_csv(processed_df, "historical_games_2022_23.csv")
        
        # Show sample
        if len(processed_df) > 0:
            sample = processed_df.iloc[0]
            print(f"\nSample game:")
            print(f"  {sample['away_team_name']} @ {sample['home_team_name']}")
            print(f"  Score: {sample['away_score']} - {sample['home_score']}")
            print(f"  Date: {sample['date']}")
        
        return processed_df
    else:
        print("\n✗ No data collected")
        return None


if __name__ == "__main__":
    collect_multi_season_data()
