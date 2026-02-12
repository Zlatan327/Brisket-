"""
NBA API Client - Wrapper for swar/nba_api
Fetches NBA game data, team stats, player stats, and injury reports
"""

from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    playergamelog,
    leaguestandings,
    teamdashboardbygeneralsplits,
    commonteamroster
)
from nba_api.stats.static import teams as nba_teams
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import time


class NBAClient:
    """Client for fetching NBA data via nba_api"""
    
    def __init__(self):
        self.teams = nba_teams.get_teams()
        self.rate_limit_delay = 0.6  # 600ms between requests to avoid rate limiting
    
    def _rate_limit(self):
        """Sleep to avoid hitting API rate limits"""
        time.sleep(self.rate_limit_delay)
    
    def get_all_teams(self) -> List[Dict]:
        """Get all NBA teams"""
        return self.teams
    
    def get_team_by_name(self, team_name: str) -> Optional[Dict]:
        """Get team by name"""
        return nba_teams.find_team_by_abbreviation(team_name)
    
    def get_games_by_date(self, date: datetime) -> pd.DataFrame:
        """
        Get all NBA games for a specific date
        
        Args:
            date: datetime object for the target date
            
        Returns:
            DataFrame with game data
        """
        self._rate_limit()
        
        date_str = date.strftime("%Y-%m-%d")
        
        # Use LeagueGameFinder to get games
        gamefinder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_str,
            date_to_nullable=date_str,
            league_id_nullable='00'  # NBA
        )
        
        games = gamefinder.get_data_frames()[0]
        return games
    
    def get_team_game_log(self, team_id: int, season: str = "2025-26") -> pd.DataFrame:
        """
        Get game log for a specific team
        
        Args:
            team_id: NBA team ID
            season: Season string (e.g., "2025-26")
            
        Returns:
            DataFrame with team game log
        """
        self._rate_limit()
        
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season
        )
        
        return gamelog.get_data_frames()[0]
    
    def get_team_stats(self, team_id: int, season: str = "2025-26") -> Dict:
        """
        Get advanced team statistics
        
        Args:
            team_id: NBA team ID
            season: Season string
            
        Returns:
            Dictionary with team stats including Four Factors
        """
        self._rate_limit()
        
        dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id,
            season=season,
            measure_type_detailed_defense='Advanced'
        )
        
        overall_stats = dashboard.get_data_frames()[0]
        
        if len(overall_stats) == 0:
            return {}
        
        stats = overall_stats.iloc[0]
        
        return {
            'efg_pct': stats.get('EFG_PCT', 0),
            'ts_pct': stats.get('TS_PCT', 0),
            'orb_pct': stats.get('OREB_PCT', 0),
            'drb_pct': stats.get('DREB_PCT', 0),
            'tov_rate': stats.get('TM_TOV_PCT', 0),
            'ft_rate': stats.get('FTA_RATE', 0),
            'net_rating': stats.get('NET_RATING', 0),
            'ortg': stats.get('OFF_RATING', 0),
            'drtg': stats.get('DEF_RATING', 0),
            'pace': stats.get('PACE', 0)
        }
    
    def get_team_roster(self, team_id: int, season: str = "2025-26") -> pd.DataFrame:
        """
        Get current roster for a team
        
        Args:
            team_id: NBA team ID
            season: Season string
            
        Returns:
            DataFrame with roster data
        """
        self._rate_limit()
        
        roster = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season
        )
        
        return roster.get_data_frames()[0]
    
    def get_season_games(self, season: str = "2025-26") -> pd.DataFrame:
        """
        Get all games for a season
        
        Args:
            season: Season string (e.g., "2025-26")
            
        Returns:
            DataFrame with all season games
        """
        self._rate_limit()
        
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00'
        )
        
        games = gamefinder.get_data_frames()[0]
        return games
    
    def calculate_rest_days(self, team_id: int, game_date: datetime, season: str = "2025-26") -> int:
        """
        Calculate rest days for a team before a specific game
        
        Args:
            team_id: NBA team ID
            game_date: Date of the game
            season: Season string
            
        Returns:
            Number of rest days
        """
        game_log = self.get_team_game_log(team_id, season)
        
        if len(game_log) == 0:
            return 7  # Default if no previous games
        
        # Convert GAME_DATE to datetime
        game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])
        
        # Filter games before the target date
        previous_games = game_log[game_log['GAME_DATE'] < game_date]
        
        if len(previous_games) == 0:
            return 7  # First game of season
        
        # Get most recent game
        last_game_date = previous_games['GAME_DATE'].max()
        
        # Calculate rest days
        rest_days = (game_date - last_game_date).days - 1
        
        return max(0, rest_days)


# Example usage
if __name__ == "__main__":
    client = NBAClient()
    
    # Get all teams
    teams = client.get_all_teams()
    print(f"Found {len(teams)} NBA teams")
    
    # Get Lakers stats
    lakers = client.get_team_by_name("LAL")
    if lakers:
        print(f"\nLakers ID: {lakers['id']}")
        stats = client.get_team_stats(lakers['id'])
        print(f"Lakers Stats: {stats}")
