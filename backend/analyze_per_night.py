"""
Per-Game-Night Accuracy Analysis
Calculates accuracy for each game night (date)
"""

import pandas as pd
import numpy as np
from xgboost_model import XGBoostModel


def analyze_per_night_accuracy():
    print("Per-Game-Night Accuracy Analysis")
    print("=" * 60)
    
    # Load features
    features_df = pd.read_csv("features_2023_24.csv")
    
    # Define feature columns
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
    
    # Load trained model
    print("\nLoading trained model...")
    xgb_model = XGBoostModel()
    xgb_model.load_model("xgboost_nba_2023_24.json")
    
    # Prepare data
    X = features_df[feature_cols].values
    y = features_df['home_win'].values
    
    # Get predictions
    print("Generating predictions...")
    y_pred_proba = xgb_model.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Add predictions to dataframe
    features_df['predicted_home_win'] = y_pred
    features_df['prediction_correct'] = (y_pred == y).astype(int)
    features_df['confidence'] = np.abs(y_pred_proba - 0.5) * 2  # 0-1 scale
    
    # Group by date
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    nightly_stats = features_df.groupby('date').agg({
        'prediction_correct': ['sum', 'count', 'mean'],
        'confidence': 'mean'
    }).reset_index()
    
    nightly_stats.columns = ['date', 'correct', 'total', 'accuracy', 'avg_confidence']
    nightly_stats = nightly_stats.sort_values('date')
    
    # Overall stats
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Total game nights: {len(nightly_stats)}")
    print(f"Total games: {len(features_df)}")
    print(f"Avg games per night: {len(features_df) / len(nightly_stats):.1f}")
    print(f"Overall accuracy: {features_df['prediction_correct'].mean():.1%}")
    print(f"Avg confidence: {features_df['confidence'].mean():.1%}")
    
    # Per-night stats
    print("\n" + "=" * 60)
    print("PER-NIGHT ACCURACY DISTRIBUTION")
    print("=" * 60)
    
    accuracy_bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_labels = ['0-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    nightly_stats['accuracy_bin'] = pd.cut(nightly_stats['accuracy'], bins=accuracy_bins, labels=accuracy_labels)
    
    distribution = nightly_stats['accuracy_bin'].value_counts().sort_index()
    
    for bin_label, count in distribution.items():
        pct = count / len(nightly_stats) * 100
        print(f"{bin_label:10s}: {count:3d} nights ({pct:5.1f}%)")
    
    # Best and worst nights
    print("\n" + "=" * 60)
    print("BEST NIGHTS (100% Accuracy)")
    print("=" * 60)
    
    perfect_nights = nightly_stats[nightly_stats['accuracy'] == 1.0].sort_values('total', ascending=False)
    
    if len(perfect_nights) > 0:
        print(f"Total perfect nights: {len(perfect_nights)}")
        print(f"\nTop 5 perfect nights by game count:")
        for idx, row in perfect_nights.head(5).iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: {int(row['correct'])}/{int(row['total'])} games ({row['avg_confidence']:.1%} avg confidence)")
    else:
        print("No perfect nights")
    
    print("\n" + "=" * 60)
    print("WORST NIGHTS (Lowest Accuracy)")
    print("=" * 60)
    
    worst_nights = nightly_stats.sort_values('accuracy').head(10)
    
    for idx, row in worst_nights.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {int(row['correct'])}/{int(row['total'])} games ({row['accuracy']:.1%})")
    
    # Monthly breakdown
    print("\n" + "=" * 60)
    print("MONTHLY ACCURACY")
    print("=" * 60)
    
    features_df['month'] = features_df['date'].dt.to_period('M')
    monthly_stats = features_df.groupby('month').agg({
        'prediction_correct': ['sum', 'count', 'mean']
    }).reset_index()
    
    monthly_stats.columns = ['month', 'correct', 'total', 'accuracy']
    
    for idx, row in monthly_stats.iterrows():
        print(f"{row['month']}: {int(row['correct'])}/{int(row['total'])} games ({row['accuracy']:.1%})")
    
    # Save detailed results
    nightly_stats.to_csv("per_night_accuracy.csv", index=False)
    print(f"\n✓ Detailed results saved to per_night_accuracy.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    median_accuracy = nightly_stats['accuracy'].median()
    std_accuracy = nightly_stats['accuracy'].std()
    
    print(f"Median nightly accuracy: {median_accuracy:.1%}")
    print(f"Std deviation: {std_accuracy:.1%}")
    print(f"Best night: {nightly_stats['accuracy'].max():.1%}")
    print(f"Worst night: {nightly_stats['accuracy'].min():.1%}")
    
    nights_above_60 = (nightly_stats['accuracy'] >= 0.6).sum()
    print(f"\nNights with 60%+ accuracy: {nights_above_60}/{len(nightly_stats)} ({nights_above_60/len(nightly_stats):.1%})")
    
    nights_above_70 = (nightly_stats['accuracy'] >= 0.7).sum()
    print(f"Nights with 70%+ accuracy: {nights_above_70}/{len(nightly_stats)} ({nights_above_70/len(nightly_stats):.1%})")


if __name__ == "__main__":
    analyze_per_night_accuracy()
