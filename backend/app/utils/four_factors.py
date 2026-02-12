"""
Feature Engineering - Four Factors Calculation
Implements the Four Factors of Basketball Success with 2026-specific weighting
"""

from typing import Dict, Optional
import pandas as pd


class FourFactorsCalculator:
    """
    Calculates the Four Factors of Basketball Success
    
    Weighted importance (2026):
    - Shooting Efficiency: 40%
    - Turnover Rate: 25%
    - Rebounding: 20%
    - Free Throw Rate: 15%
    """
    
    WEIGHTS = {
        'shooting': 0.40,
        'turnovers': 0.25,
        'rebounding': 0.20,
        'free_throws': 0.15
    }
    
    def __init__(self):
        pass
    
    def calculate_efg(self, fg: int, fga: int, fg3: int) -> float:
        """
        Calculate Effective Field Goal Percentage
        eFG% = (FG + 0.5 * 3P) / FGA
        
        Args:
            fg: Field goals made
            fga: Field goal attempts
            fg3: Three-pointers made
            
        Returns:
            eFG% (0.0 - 1.0)
        """
        if fga == 0:
            return 0.0
        
        return (fg + 0.5 * fg3) / fga
    
    def calculate_ts(self, pts: int, fga: int, fta: int) -> float:
        """
        Calculate True Shooting Percentage
        TS% = PTS / (2 * (FGA + 0.44 * FTA))
        
        Args:
            pts: Points scored
            fga: Field goal attempts
            fta: Free throw attempts
            
        Returns:
            TS% (0.0 - 1.0)
        """
        denominator = 2 * (fga + 0.44 * fta)
        
        if denominator == 0:
            return 0.0
        
        return pts / denominator
    
    def calculate_tov_rate(self, tov: int, fga: int, fta: int) -> float:
        """
        Calculate Turnover Rate
        TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        
        Args:
            tov: Turnovers
            fga: Field goal attempts
            fta: Free throw attempts
            
        Returns:
            TOV% (0.0 - 1.0)
        """
        denominator = fga + 0.44 * fta + tov
        
        if denominator == 0:
            return 0.0
        
        return tov / denominator
    
    def calculate_orb_pct(self, orb: int, team_orb: int, opp_drb: int) -> float:
        """
        Calculate Offensive Rebound Percentage
        ORB% = ORB / (ORB + Opp DRB)
        
        Args:
            orb: Offensive rebounds
            team_orb: Team offensive rebounds
            opp_drb: Opponent defensive rebounds
            
        Returns:
            ORB% (0.0 - 1.0)
        """
        denominator = team_orb + opp_drb
        
        if denominator == 0:
            return 0.0
        
        return orb / denominator
    
    def calculate_drb_pct(self, drb: int, team_drb: int, opp_orb: int) -> float:
        """
        Calculate Defensive Rebound Percentage
        DRB% = DRB / (DRB + Opp ORB)
        
        Args:
            drb: Defensive rebounds
            team_drb: Team defensive rebounds
            opp_orb: Opponent offensive rebounds
            
        Returns:
            DRB% (0.0 - 1.0)
        """
        denominator = team_drb + opp_orb
        
        if denominator == 0:
            return 0.0
        
        return drb / denominator
    
    def calculate_ft_rate(self, fta: int, fga: int) -> float:
        """
        Calculate Free Throw Rate
        FT Rate = FTA / FGA
        
        Args:
            fta: Free throw attempts
            fga: Field goal attempts
            
        Returns:
            FT Rate (0.0+)
        """
        if fga == 0:
            return 0.0
        
        return fta / fga
    
    def calculate_four_factors_score(self, team_stats: Dict, opp_stats: Dict) -> float:
        """
        Calculate weighted Four Factors score
        Higher score = better performance
        
        Args:
            team_stats: Dictionary with team stats
            opp_stats: Dictionary with opponent stats
            
        Returns:
            Four Factors score (0-100)
        """
        # 1. Shooting Efficiency (40%)
        team_efg = team_stats.get('efg_pct', 0.5)
        opp_efg = opp_stats.get('efg_pct', 0.5)
        shooting_score = (team_efg - opp_efg + 0.2) * 2.5  # Normalize to 0-1
        
        # 2. Turnover Rate (25%) - lower is better for team, higher is better for opponent
        team_tov = team_stats.get('tov_rate', 0.15)
        opp_tov = opp_stats.get('tov_rate', 0.15)
        tov_score = (opp_tov - team_tov + 0.1) * 5  # Normalize to 0-1
        
        # 3. Rebounding (20%)
        team_drb = team_stats.get('drb_pct', 0.5)
        opp_drb = opp_stats.get('drb_pct', 0.5)
        reb_score = (team_drb - opp_drb + 0.2) * 2.5  # Normalize to 0-1
        
        # 4. Free Throw Rate (15%)
        team_ft = team_stats.get('ft_rate', 0.25)
        opp_ft = opp_stats.get('ft_rate', 0.25)
        ft_score = (team_ft - opp_ft + 0.2) * 2.5  # Normalize to 0-1
        
        # Weighted sum
        total_score = (
            shooting_score * self.WEIGHTS['shooting'] +
            tov_score * self.WEIGHTS['turnovers'] +
            reb_score * self.WEIGHTS['rebounding'] +
            ft_score * self.WEIGHTS['free_throws']
        )
        
        # Convert to 0-100 scale
        return max(0, min(100, total_score * 100))
    
    def calculate_net_rating(self, ortg: float, drtg: float) -> float:
        """
        Calculate Net Rating
        Net Rating = Offensive Rating - Defensive Rating
        
        Args:
            ortg: Offensive rating (points per 100 possessions)
            drtg: Defensive rating (points allowed per 100 possessions)
            
        Returns:
            Net Rating
        """
        return ortg - drtg
    
    def calculate_pace(self, possessions: int, minutes: int) -> float:
        """
        Calculate Pace (possessions per 48 minutes)
        
        Args:
            possessions: Total possessions
            minutes: Total minutes played
            
        Returns:
            Pace (possessions per 48 minutes)
        """
        if minutes == 0:
            return 0.0
        
        return (possessions / minutes) * 48
    
    def calculate_team_four_factors(self, game_stats: Dict) -> Dict:
        """
        Calculate all Four Factors for a team from game stats
        
        Args:
            game_stats: Dictionary with raw game statistics
            
        Returns:
            Dictionary with calculated Four Factors
        """
        return {
            'efg_pct': self.calculate_efg(
                game_stats.get('fg', 0),
                game_stats.get('fga', 0),
                game_stats.get('fg3', 0)
            ),
            'ts_pct': self.calculate_ts(
                game_stats.get('pts', 0),
                game_stats.get('fga', 0),
                game_stats.get('fta', 0)
            ),
            'tov_rate': self.calculate_tov_rate(
                game_stats.get('tov', 0),
                game_stats.get('fga', 0),
                game_stats.get('fta', 0)
            ),
            'orb_pct': game_stats.get('orb_pct', 0.0),  # Usually pre-calculated
            'drb_pct': game_stats.get('drb_pct', 0.0),  # Usually pre-calculated
            'ft_rate': self.calculate_ft_rate(
                game_stats.get('fta', 0),
                game_stats.get('fga', 0)
            ),
            'net_rating': self.calculate_net_rating(
                game_stats.get('ortg', 110.0),
                game_stats.get('drtg', 110.0)
            ),
            'pace': self.calculate_pace(
                game_stats.get('possessions', 100),
                game_stats.get('minutes', 48)
            )
        }


# Example usage
if __name__ == "__main__":
    calc = FourFactorsCalculator()
    
    # Example team stats
    team_stats = {
        'fg': 40,
        'fga': 85,
        'fg3': 12,
        'pts': 110,
        'fta': 20,
        'tov': 12,
        'orb_pct': 0.28,
        'drb_pct': 0.75,
        'ortg': 115.0,
        'drtg': 108.0,
        'possessions': 100,
        'minutes': 48
    }
    
    opp_stats = {
        'fg': 38,
        'fga': 88,
        'fg3': 10,
        'pts': 105,
        'fta': 18,
        'tov': 15,
        'orb_pct': 0.25,
        'drb_pct': 0.72,
        'ortg': 108.0,
        'drtg': 115.0,
        'possessions': 100,
        'minutes': 48
    }
    
    print("Four Factors Calculator Test")
    print("=" * 50)
    
    team_factors = calc.calculate_team_four_factors(team_stats)
    print("\nTeam Four Factors:")
    for key, value in team_factors.items():
        print(f"  {key}: {value:.3f}")
    
    opp_factors = calc.calculate_team_four_factors(opp_stats)
    print("\nOpponent Four Factors:")
    for key, value in opp_factors.items():
        print(f"  {key}: {value:.3f}")
    
    score = calc.calculate_four_factors_score(team_factors, opp_factors)
    print(f"\nFour Factors Score: {score:.1f}/100")
    print("(Higher = Team has advantage)")
