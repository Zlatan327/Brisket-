"""
Enhanced Feature Engineer with Real Travel Data
Uses actual NBA city locations for travel calculations
"""

import pandas as pd
import numpy as np
from typing import Dict
from four_factors import FourFactorsCalculator
from nba_locations import NBALocationCalculator


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering with real travel data
    """
    
    def __init__(self):
        """Initialize enhanced feature engineer"""
        self.four_factors_calc = FourFactorsCalculator()
        self.location_calc = NBALocationCalculator()
        
    def calculate_four_factors(self, game: Dict) -> Dict:
        """Calculate Four Factors for a game"""
        home_stats = {
            'fgm': game['home_fgm'], 'fga': game['home_fga'],
            'fg3m': game['home_fg3m'], 'ftm': game['home_ftm'],
            'fta': game['home_fta'], 'oreb': game['home_oreb'],
            'dreb': game['home_dreb'], 'tov': game['home_tov'],
            'opp_dreb': game['away_dreb']
        }
        
        away_stats = {
            'fgm': game['away_fgm'], 'fga': game['away_fga'],
            'fg3m': game['away_fg3m'], 'ftm': game['away_ftm'],
            'fta': game['away_fta'], 'oreb': game['away_oreb'],
            'dreb': game['away_dreb'], 'tov': game['away_tov'],
            'opp_dreb': game['home_dreb']
        }
        
        home_ff = self.four_factors_calc.calculate_team_four_factors(home_stats)
        away_ff = self.four_factors_calc.calculate_team_four_factors(away_stats)
        
        return {
            'home_efg_pct': home_ff['efg_pct'],
            'away_efg_pct': away_ff['efg_pct'],
            'home_tov_rate': home_ff['tov_rate'],
            'away_tov_rate': away_ff['tov_rate'],
            'home_drb_pct': home_ff['drb_pct'],
            'away_drb_pct': away_ff['drb_pct'],
            'home_ft_rate': home_ff['ft_rate'],
            'away_ft_rate': away_ff['ft_rate']
        }
    
    def calculate_rolling_stats(
        self, games_df: pd.DataFrame, team_id: int,
        current_date: str, window: int = 10
    ) -> Dict:
        """Calculate rolling statistics for a team"""
        team_games = games_df[
            ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False).head(window)
        
        if len(team_games) == 0:
            return {'last_5_wins': 0, 'last_10_wins': 0, 'avg_score': 100.0, 'avg_allowed': 100.0}
        
        wins, scores, allowed = [], [], []
        
        for _, game in team_games.iterrows():
            if game['home_team_id'] == team_id:
                wins.append(1 if game['home_win'] == 1 else 0)
                scores.append(game['home_score'])
                allowed.append(game['away_score'])
            else:
                wins.append(1 if game['home_win'] == 0 else 0)
                scores.append(game['away_score'])
                allowed.append(game['home_score'])
        
        return {
            'last_5_wins': sum(wins[:5]),
            'last_10_wins': sum(wins[:10]),
            'avg_score': np.mean(scores) if scores else 100.0,
            'avg_allowed': np.mean(allowed) if allowed else 100.0
        }
    
    def calculate_rest_days(
        self, games_df: pd.DataFrame, team_id: int, current_date: str
    ) -> int:
        """Calculate days of rest for a team"""
        prev_games = games_df[
            ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False)
        
        if len(prev_games) == 0:
            return 3
        
        prev_date = pd.to_datetime(prev_games.iloc[0]['date'])
        curr_date = pd.to_datetime(current_date)
        rest_days = (curr_date - prev_date).days - 1
        return max(0, rest_days)
    
    def engineer_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all 37 features with REAL travel data"""
        print("Engineering features with REAL travel data...")
        print(f"Total games: {len(games_df)}")
        
        games_df['date'] = pd.to_datetime(games_df['date'])
        games_df = games_df.sort_values('date')
        
        features_list = []
        
        for idx, game in games_df.iterrows():
            # Four Factors
            four_factors = self.calculate_four_factors(game)
            
            # Rolling stats
            home_rolling = self.calculate_rolling_stats(games_df, game['home_team_id'], game['date'])
            away_rolling = self.calculate_rolling_stats(games_df, game['away_team_id'], game['date'])
            
            # Rest days
            home_rest = self.calculate_rest_days(games_df, game['home_team_id'], game['date'])
            away_rest = self.calculate_rest_days(games_df, game['away_team_id'], game['date'])
            
            # REAL TRAVEL DATA using NBA locations
            travel_metrics = self.location_calc.calculate_travel_metrics(
                game['away_team_name'],
                game['home_team_name']
            )
            
            # Net rating
            home_net_rating = home_rolling['avg_score'] - home_rolling['avg_allowed']
            away_net_rating = away_rolling['avg_score'] - away_rolling['avg_allowed']
            
            # Pace (simplified)
            home_pace = 100.0
            away_pace = 100.0
            
            # Fatigue
            home_fatigue = max(0, 50 - (home_rest * 10))
            away_fatigue = max(0, 50 - (away_rest * 10))
            
            # Back-to-back
            is_b2b_home = 1 if home_rest == 0 else 0
            is_b2b_away = 1 if away_rest == 0 else 0
            
            # Compile features
            features = {
                'game_id': game['game_id'],
                'date': game['date'],
                'home_team_name': game['home_team_name'],
                'away_team_name': game['away_team_name'],
                'home_win': game['home_win'],
                
                **four_factors,
                
                'home_net_rating': home_net_rating,
                'away_net_rating': away_net_rating,
                'home_pace': home_pace,
                'away_pace': away_pace,
                
                # REAL TRAVEL DATA
                'home_rest_days': home_rest,
                'away_rest_days': away_rest,
                'travel_distance': travel_metrics['distance'],
                'timezone_shift': travel_metrics['timezone_shift'],
                'elevation_change': travel_metrics['elevation_change'],
                
                'home_fatigue_score': home_fatigue,
                'away_fatigue_score': away_fatigue,
                'home_cumulative_load': 0.0,
                'away_cumulative_load': 0.0,
                
                'home_last_5_wins': home_rolling['last_5_wins'],
                'away_last_5_wins': away_rolling['last_5_wins'],
                'home_last_10_wins': home_rolling['last_10_wins'],
                'away_last_10_wins': away_rolling['last_10_wins'],
                
                'home_injury_impact': 0.0,
                'away_injury_impact': 0.0,
                
                'home_sentiment_score': 0.0,
                'away_sentiment_score': 0.0,
                'home_trade_rumors': 0.0,
                'away_trade_rumors': 0.0,
                'home_coaching_stability': 1.0,
                'away_coaching_stability': 1.0,
                
                'home_tanking_score': 0.0,
                'away_tanking_score': 0.0,
                
                'is_back_to_back_home': is_b2b_home,
                'is_back_to_back_away': is_b2b_away,
                'home_court_advantage': 1.0
            }
            
            features_list.append(features)
            
            if (idx + 1) % 200 == 0:
                print(f"Processed {idx + 1}/{len(games_df)} games...")
        
        features_df = pd.DataFrame(features_list)
        print(f"\n✓ Feature engineering complete with REAL travel data!")
        print(f"Features shape: {features_df.shape}")
        
        return features_df


if __name__ == "__main__":
    print("Enhanced Feature Engineering with Real Travel Data")
    print("=" * 60)
    
    # Load 2022-23 season
    print("\nProcessing 2022-23 season...")
    games_2022 = pd.read_csv("historical_games_2022_23.csv")
    
    engineer = EnhancedFeatureEngineer()
    features_2022 = engineer.engineer_features(games_2022)
    
    features_2022.to_csv("features_2022_23_enhanced.csv", index=False)
    print(f"\n✓ Saved to features_2022_23_enhanced.csv")
    
    # Show sample with real travel data
    print("\nSample with REAL travel data:")
    sample = features_2022.iloc[100]
    print(f"  {sample['away_team_name']} @ {sample['home_team_name']}")
    print(f"  Travel distance: {sample['travel_distance']:.0f} miles")
    print(f"  Timezone shift: {sample['timezone_shift']} hours")
    print(f"  Elevation change: {sample['elevation_change']:,.0f} feet")
