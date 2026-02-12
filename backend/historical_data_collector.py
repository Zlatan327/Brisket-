"""
Historical Data Collector
Collects real NBA game data from 2020-2026 for backtesting
"""

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, teamgamelog
from nba_api.stats.static import teams
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional


class HistoricalDataCollector:
    """
    Collects historical NBA game data
    
    Features:
    - Game results (2020-2026)
    - Team stats per game
    - Advanced metrics
    - Box scores
    """
    
    def __init__(self):
        """Initialize data collector"""
        self.all_teams = teams.get_teams()
        self.rate_limit_delay = 0.6  # 600ms between requests
        
    def get_games_by_season(self, season: str) -> pd.DataFrame:
        """
        Get all games for a season
        
        Args:
            season: Season string (e.g., "2023-24")
            
        Returns:
            DataFrame with game data
        """
        print(f"Fetching games for {season} season...")
        
        try:
            # Use LeagueGameFinder to get all games
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00'  # NBA
            )
            
            games = gamefinder.get_data_frames()[0]
            
            # Rate limit
            time.sleep(self.rate_limit_delay)
            
            print(f"Found {len(games)} game records for {season}")
            return games
            
        except Exception as e:
            print(f"Error fetching games for {season}: {e}")
            return pd.DataFrame()
    
    def get_team_game_log(self, team_id: int, season: str) -> pd.DataFrame:
        """
        Get game log for a specific team
        
        Args:
            team_id: NBA team ID
            season: Season string
            
        Returns:
            DataFrame with team game log
        """
        try:
            gamelog = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season
            )
            
            df = gamelog.get_data_frames()[0]
            
            # Rate limit
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except Exception as e:
            print(f"Error fetching game log for team {team_id}: {e}")
            return pd.DataFrame()
    
    def collect_historical_games(
        self,
        start_season: str = "2020-21",
        end_season: str = "2023-24"
    ) -> pd.DataFrame:
        """
        Collect all historical games
        
        Args:
            start_season: Starting season
            end_season: Ending season
            
        Returns:
            DataFrame with all games
        """
        print(f"Collecting historical games from {start_season} to {end_season}")
        print("=" * 60)
        
        all_games = []
        
        # Generate season list
        start_year = int(start_season.split("-")[0])
        end_year = int(end_season.split("-")[0])
        
        seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year + 1)]
        
        for season in seasons:
            games_df = self.get_games_by_season(season)
            
            if not games_df.empty:
                games_df['SEASON'] = season
                all_games.append(games_df)
                print(f"✓ {season}: {len(games_df)} records")
            else:
                print(f"✗ {season}: No data")
            
            # Longer delay between seasons
            time.sleep(1.0)
        
        if all_games:
            combined_df = pd.concat(all_games, ignore_index=True)
            print(f"\nTotal records collected: {len(combined_df)}")
            return combined_df
        else:
            print("No games collected")
            return pd.DataFrame()
    
    def process_games_for_backtesting(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process games into format for backtesting
        
        Args:
            games_df: Raw games DataFrame
            
        Returns:
            Processed DataFrame with home/away pairs
        """
        print("\nProcessing games for backtesting...")
        
        # Each game appears twice (once for each team)
        # We need to pair them into single rows
        
        # Sort by game ID and date
        games_df = games_df.sort_values(['GAME_ID', 'GAME_DATE'])
        
        # Group by game ID
        game_pairs = []
        
        for game_id, group in games_df.groupby('GAME_ID'):
            if len(group) != 2:
                continue  # Skip if not exactly 2 teams
            
            # Determine home and away
            teams_list = group.to_dict('records')
            
            # Usually the second team listed is home (but check MATCHUP)
            team1, team2 = teams_list[0], teams_list[1]
            
            # Check MATCHUP to determine home/away
            if '@' in team1.get('MATCHUP', ''):
                # team1 is away
                away_team = team1
                home_team = team2
            else:
                # team1 is home
                home_team = team1
                away_team = team2
            
            game_pair = {
                'game_id': game_id,
                'date': home_team['GAME_DATE'],
                'season': home_team.get('SEASON', ''),
                
                # Teams
                'home_team_id': home_team['TEAM_ID'],
                'home_team_name': home_team['TEAM_NAME'],
                'away_team_id': away_team['TEAM_ID'],
                'away_team_name': away_team['TEAM_NAME'],
                
                # Scores
                'home_score': home_team['PTS'],
                'away_score': away_team['PTS'],
                'home_win': 1 if home_team['WL'] == 'W' else 0,
                
                # Home team stats
                'home_fgm': home_team.get('FGM', 0),
                'home_fga': home_team.get('FGA', 0),
                'home_fg3m': home_team.get('FG3M', 0),
                'home_fg3a': home_team.get('FG3A', 0),
                'home_ftm': home_team.get('FTM', 0),
                'home_fta': home_team.get('FTA', 0),
                'home_oreb': home_team.get('OREB', 0),
                'home_dreb': home_team.get('DREB', 0),
                'home_reb': home_team.get('REB', 0),
                'home_ast': home_team.get('AST', 0),
                'home_tov': home_team.get('TOV', 0),
                'home_stl': home_team.get('STL', 0),
                'home_blk': home_team.get('BLK', 0),
                'home_pf': home_team.get('PF', 0),
                
                # Away team stats
                'away_fgm': away_team.get('FGM', 0),
                'away_fga': away_team.get('FGA', 0),
                'away_fg3m': away_team.get('FG3M', 0),
                'away_fg3a': away_team.get('FG3A', 0),
                'away_ftm': away_team.get('FTM', 0),
                'away_fta': away_team.get('FTA', 0),
                'away_oreb': away_team.get('OREB', 0),
                'away_dreb': away_team.get('DREB', 0),
                'away_reb': away_team.get('REB', 0),
                'away_ast': away_team.get('AST', 0),
                'away_tov': away_team.get('TOV', 0),
                'away_stl': away_team.get('STL', 0),
                'away_blk': away_team.get('BLK', 0),
                'away_pf': away_team.get('PF', 0),
            }
            
            game_pairs.append(game_pair)
        
        processed_df = pd.DataFrame(game_pairs)
        print(f"Processed {len(processed_df)} unique games")
        
        return processed_df
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str):
        """Save DataFrame to CSV"""
        df.to_csv(filepath, index=False)
        print(f"\nData saved to {filepath}")
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load DataFrame from CSV"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} games from {filepath}")
        return df


# Example usage
if __name__ == "__main__":
    print("Historical Data Collector")
    print("=" * 60)
    
    collector = HistoricalDataCollector()
    
    # Collect games from 2023-24 season only (for testing)
    print("\nCollecting 2023-24 season data...")
    games_df = collector.get_games_by_season("2023-24")
    
    if not games_df.empty:
        print(f"\nRaw data shape: {games_df.shape}")
        print(f"Columns: {list(games_df.columns)[:10]}...")
        
        # Process for backtesting
        processed_df = collector.process_games_for_backtesting(games_df)
        
        print(f"\nProcessed data shape: {processed_df.shape}")
        print(f"\nSample game:")
        if len(processed_df) > 0:
            sample = processed_df.iloc[0]
            print(f"  {sample['away_team_name']} @ {sample['home_team_name']}")
            print(f"  Score: {sample['away_score']} - {sample['home_score']}")
            print(f"  Winner: {'Home' if sample['home_win'] == 1 else 'Away'}")
            print(f"  Date: {sample['date']}")
        
        # Save to CSV
        collector.save_to_csv(processed_df, "historical_games_2023_24.csv")
        
        print("\n✓ Data collection successful!")
    else:
        print("\n✗ No data collected")
