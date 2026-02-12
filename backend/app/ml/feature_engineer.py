"""
Feature Engineering Pipeline for Historical Data
Calculates all 37 features from raw game data
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..utils.four_factors import FourFactorsCalculator
from ..utils.travel_fatigue import TravelFatigueCalculator


class FeatureEngineer:
    """
    Feature engineering pipeline for historical games
    
    Calculates all 37 features:
    - Four Factors (8)
    - Advanced Metrics (4)
    - Rest & Travel (5)
    - Fatigue (4)
    - Form (4)
    - Injuries (2)
    - Sentiment (6)
    - Tanking (2)
    - Context (2)
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.four_factors_calc = FourFactorsCalculator()
        self.travel_calc = TravelFatigueCalculator()
        
    def calculate_four_factors(self, game: Dict) -> Dict:
        """
        Calculate Four Factors for a game
        
        Args:
            game: Game dictionary with stats
            
        Returns:
            Dictionary with Four Factors features
        """
        # Home team
        home_stats = {
            'fgm': game['home_fgm'],
            'fga': game['home_fga'],
            'fg3m': game['home_fg3m'],
            'ftm': game['home_ftm'],
            'fta': game['home_fta'],
            'oreb': game['home_oreb'],
            'dreb': game['home_dreb'],
            'tov': game['home_tov'],
            'opp_dreb': game['away_dreb']
        }
        
        # Away team
        away_stats = {
            'fgm': game['away_fgm'],
            'fga': game['away_fga'],
            'fg3m': game['away_fg3m'],
            'ftm': game['away_ftm'],
            'fta': game['away_fta'],
            'oreb': game['away_oreb'],
            'dreb': game['away_dreb'],
            'tov': game['away_tov'],
            'opp_dreb': game['home_dreb']
        }
        
        # Calculate Four Factors
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
        self,
        games_df: pd.DataFrame,
        team_id: int,
        current_date: str,
        window: int = 10
    ) -> Dict:
        """
        Calculate rolling statistics for a team
        
        Args:
            games_df: All games DataFrame
            team_id: Team ID
            current_date: Current game date
            window: Rolling window size
            
        Returns:
            Dictionary with rolling stats
        """
        # Get team's previous games
        team_games = games_df[
            ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False).head(window)
        
        if len(team_games) == 0:
            return {
                'last_5_wins': 0,
                'last_10_wins': 0,
                'avg_score': 100.0,
                'avg_allowed': 100.0
            }
        
        # Calculate wins
        wins = []
        scores = []
        allowed = []
        
        for _, game in team_games.iterrows():
            if game['home_team_id'] == team_id:
                # Team was home
                wins.append(1 if game['home_win'] == 1 else 0)
                scores.append(game['home_score'])
                allowed.append(game['away_score'])
            else:
                # Team was away
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
        self,
        games_df: pd.DataFrame,
        team_id: int,
        current_date: str
    ) -> int:
        """
        Calculate days of rest for a team
        
        Args:
            games_df: All games DataFrame
            team_id: Team ID
            current_date: Current game date
            
        Returns:
            Number of rest days
        """
        # Get team's previous game
        prev_games = games_df[
            ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False)
        
        if len(prev_games) == 0:
            return 3  # Default
        
        prev_date = pd.to_datetime(prev_games.iloc[0]['date'])
        curr_date = pd.to_datetime(current_date)
        
        rest_days = (curr_date - prev_date).days - 1
        return max(0, rest_days)
    
    def engineer_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all 37 features for each game
        
        Args:
            games_df: Raw games DataFrame
            
        Returns:
            DataFrame with all features
        """
        print("Engineering features for all games...")
        print(f"Total games: {len(games_df)}")
        
        # Convert date to datetime
        games_df['date'] = pd.to_datetime(games_df['date'])
        
        # Sort by date
        games_df = games_df.sort_values('date')
        
        features_list = []
        
        for idx, game in games_df.iterrows():
            # Four Factors (8 features)
            four_factors = self.calculate_four_factors(game)
            
            # Rolling stats for home team
            home_rolling = self.calculate_rolling_stats(
                games_df, game['home_team_id'], game['date']
            )
            
            # Rolling stats for away team
            away_rolling = self.calculate_rolling_stats(
                games_df, game['away_team_id'], game['date']
            )
            
            # Rest days
            home_rest = self.calculate_rest_days(
                games_df, game['home_team_id'], game['date']
            )
            away_rest = self.calculate_rest_days(
                games_df, game['away_team_id'], game['date']
            )
            
            # Travel distance (simplified - would use actual city locations)
            travel_distance = 500.0  # Placeholder
            
            # Net rating (simplified)
            home_net_rating = home_rolling['avg_score'] - home_rolling['avg_allowed']
            away_net_rating = away_rolling['avg_score'] - away_rolling['avg_allowed']
            
            # Pace (simplified - possessions per game)
            home_pace = 100.0  # Placeholder
            away_pace = 100.0  # Placeholder
            
            # Fatigue (simplified)
            home_fatigue = max(0, 50 - (home_rest * 10))
            away_fatigue = max(0, 50 - (away_rest * 10))
            
            # Back-to-back
            is_b2b_home = 1 if home_rest == 0 else 0
            is_b2b_away = 1 if away_rest == 0 else 0
            
            # Compile all features
            features = {
                'game_id': game['game_id'],
                'date': game['date'],
                'home_team_name': game['home_team_name'],
                'away_team_name': game['away_team_name'],
                'home_win': game['home_win'],
                
                # Four Factors (8)
                **four_factors,
                
                # Advanced Metrics (4)
                'home_net_rating': home_net_rating,
                'away_net_rating': away_net_rating,
                'home_pace': home_pace,
                'away_pace': away_pace,
                
                # Rest & Travel (5)
                'home_rest_days': home_rest,
                'away_rest_days': away_rest,
                'travel_distance': travel_distance,
                'timezone_shift': 0,  # Placeholder
                'elevation_change': 0,  # Placeholder
                
                # Fatigue (4)
                'home_fatigue_score': home_fatigue,
                'away_fatigue_score': away_fatigue,
                'home_cumulative_load': 0.0,  # Placeholder
                'away_cumulative_load': 0.0,  # Placeholder
                
                # Form (4)
                'home_last_5_wins': home_rolling['last_5_wins'],
                'away_last_5_wins': away_rolling['last_5_wins'],
                'home_last_10_wins': home_rolling['last_10_wins'],
                'away_last_10_wins': away_rolling['last_10_wins'],
                
                # Injuries (2) - Placeholder
                'home_injury_impact': 0.0,
                'away_injury_impact': 0.0,
                
                # Sentiment (6) - Placeholder
                'home_sentiment_score': 0.0,
                'away_sentiment_score': 0.0,
                'home_trade_rumors': 0.0,
                'away_trade_rumors': 0.0,
                'home_coaching_stability': 1.0,
                'away_coaching_stability': 1.0,
                
                # Tanking (2) - Placeholder
                'home_tanking_score': 0.0,
                'away_tanking_score': 0.0,
                
                # Context (2)
                'is_back_to_back_home': is_b2b_home,
                'is_back_to_back_away': is_b2b_away,
                'home_court_advantage': 1.0
            }
            
            features_list.append(features)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(games_df)} games...")
        
        features_df = pd.DataFrame(features_list)
        print(f"\n✓ Feature engineering complete!")
        print(f"Features shape: {features_df.shape}")
        
        return features_df


# Example usage
if __name__ == "__main__":
    print("Feature Engineering Pipeline")
    print("=" * 60)
    
    # Load historical games
    print("\nLoading historical games...")
    games_df = pd.read_csv("historical_games_2023_24.csv")
    print(f"Loaded {len(games_df)} games")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Engineer features
    features_df = engineer.engineer_features(games_df)
    
    # Save features
    features_df.to_csv("features_2023_24.csv", index=False)
    print(f"\n✓ Features saved to features_2023_24.csv")
    
    # Show sample
    print("\nSample features:")
    print(features_df.iloc[100][['home_team_name', 'away_team_name', 'home_win', 
                                   'home_efg_pct', 'away_efg_pct', 'home_rest_days', 'away_rest_days']])
