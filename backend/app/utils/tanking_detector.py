"""
Tanking Detection Algorithm (NBA Only)
Detects strategic tanking signals for 2026 draft prospects
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd


class TankingDetector:
    """
    Detects tanking signals in NBA teams
    
    Signals:
    1. Star rest on back-to-backs
    2. Young player minutes (>60% threshold)
    3. Veteran trade activity
    4. Cluster classification (Lottery Bound)
    """
    
    # Thresholds
    YOUNG_MINUTES_THRESHOLD = 0.60  # 60% of minutes to young players
    VETERAN_AGE_THRESHOLD = 28      # Players 28+ considered veterans
    YOUNG_AGE_THRESHOLD = 24        # Players <24 considered young
    
    def __init__(self):
        pass
    
    def calculate_star_rest_probability(
        self,
        team_id: int,
        game_date: datetime,
        is_back_to_back: bool,
        star_players: List[Dict],
        recent_rest_games: int
    ) -> float:
        """
        Calculate probability that stars will rest
        
        Args:
            team_id: NBA team ID
            game_date: Date of upcoming game
            is_back_to_back: Whether game is on back-to-back
            star_players: List of star player dictionaries
            recent_rest_games: Number of games stars rested in last 10
            
        Returns:
            Probability (0.0 - 1.0)
        """
        base_probability = 0.1  # 10% baseline
        
        # Back-to-back multiplier
        if is_back_to_back:
            base_probability += 0.3  # +30% for back-to-back
        
        # Recent rest pattern
        rest_rate = recent_rest_games / 10.0
        base_probability += rest_rate * 0.2  # Up to +20% based on pattern
        
        # Late season (after All-Star break)
        if game_date.month >= 2:  # February onwards
            base_probability += 0.15  # +15% late season
        
        return min(1.0, base_probability)
    
    def calculate_young_lineup_minutes_pct(
        self,
        game_log: pd.DataFrame,
        roster: pd.DataFrame
    ) -> float:
        """
        Calculate percentage of minutes given to young players
        
        Args:
            game_log: Recent game log with player minutes
            roster: Team roster with player ages
            
        Returns:
            Percentage (0.0 - 1.0)
        """
        if len(game_log) == 0 or len(roster) == 0:
            return 0.0
        
        # Merge game log with roster to get ages
        merged = game_log.merge(roster, on='player_id', how='left')
        
        # Calculate total minutes
        total_minutes = merged['minutes'].sum()
        
        if total_minutes == 0:
            return 0.0
        
        # Calculate young player minutes
        young_minutes = merged[
            merged['age'] < self.YOUNG_AGE_THRESHOLD
        ]['minutes'].sum()
        
        return young_minutes / total_minutes
    
    def count_veteran_trades(
        self,
        team_id: int,
        lookback_days: int = 30
    ) -> int:
        """
        Count veteran trades in the last N days
        
        Args:
            team_id: NBA team ID
            lookback_days: Number of days to look back
            
        Returns:
            Count of veteran trades
        """
        # Placeholder - would query trade database
        # For now, return 0
        return 0
    
    def classify_team_cluster(
        self,
        win_pct: float,
        net_rating: float,
        playoff_odds: float
    ) -> str:
        """
        Classify team into cluster
        
        Args:
            win_pct: Win percentage (0.0 - 1.0)
            net_rating: Net rating
            playoff_odds: Playoff probability (0.0 - 1.0)
            
        Returns:
            Cluster name
        """
        if win_pct >= 0.600 and net_rating > 3.0:
            return "Championship Contender"
        elif win_pct >= 0.450 and playoff_odds > 0.5:
            return "Play-In Team"
        elif win_pct < 0.350:
            return "Lottery Bound"
        else:
            return "Young Core"
    
    def calculate_tanking_score(
        self,
        star_rest_prob: float,
        young_minutes_pct: float,
        veteran_trades: int,
        cluster: str
    ) -> float:
        """
        Calculate overall tanking score (0-100)
        
        Args:
            star_rest_prob: Star rest probability
            young_minutes_pct: Young player minutes percentage
            veteran_trades: Number of veteran trades
            cluster: Team cluster classification
            
        Returns:
            Tanking score (0-100, higher = more tanking)
        """
        score = 0.0
        
        # 1. Star rest (30% weight)
        score += star_rest_prob * 30
        
        # 2. Young minutes (40% weight)
        if young_minutes_pct > self.YOUNG_MINUTES_THRESHOLD:
            excess = young_minutes_pct - self.YOUNG_MINUTES_THRESHOLD
            score += min(40, excess * 100)
        
        # 3. Veteran trades (20% weight)
        score += min(20, veteran_trades * 5)
        
        # 4. Cluster classification (10% weight)
        cluster_scores = {
            "Championship Contender": 0,
            "Play-In Team": 2,
            "Young Core": 5,
            "Lottery Bound": 10
        }
        score += cluster_scores.get(cluster, 0)
        
        return min(100.0, score)
    
    def detect_soft_tanking_signals(
        self,
        team_stats: Dict,
        recent_games: List[Dict]
    ) -> Dict:
        """
        Detect "soft tanking" signals
        
        Args:
            team_stats: Team statistics
            recent_games: List of recent game dictionaries
            
        Returns:
            Dictionary with soft tanking signals
        """
        signals = {
            'q4_unforced_turnovers_spike': False,
            'star_ft_rate_drop': False,
            'defensive_intensity_drop': False,
            'close_game_losses': 0
        }
        
        if len(recent_games) == 0:
            return signals
        
        # Analyze Q4 turnovers
        q4_tov_avg = sum(g.get('q4_tov', 0) for g in recent_games) / len(recent_games)
        if q4_tov_avg > 4.0:  # More than 4 Q4 turnovers per game
            signals['q4_unforced_turnovers_spike'] = True
        
        # Analyze star FT rate
        star_ft_rate = team_stats.get('star_ft_rate', 0.25)
        season_avg_ft_rate = team_stats.get('season_avg_ft_rate', 0.25)
        if star_ft_rate < season_avg_ft_rate * 0.8:  # 20% drop
            signals['star_ft_rate_drop'] = True
        
        # Count close game losses (within 5 points)
        close_losses = sum(
            1 for g in recent_games
            if g.get('result') == 'loss' and abs(g.get('margin', 100)) <= 5
        )
        signals['close_game_losses'] = close_losses
        
        return signals
    
    def get_tanking_recommendation(self, tanking_score: float) -> str:
        """
        Get betting recommendation based on tanking score
        
        Args:
            tanking_score: Tanking score (0-100)
            
        Returns:
            Recommendation string
        """
        if tanking_score >= 70:
            return "HIGH RISK - Strong tanking signals, bet against"
        elif tanking_score >= 50:
            return "MODERATE RISK - Some tanking signals, reduce confidence"
        elif tanking_score >= 30:
            return "LOW RISK - Minor tanking signals, monitor"
        else:
            return "NO RISK - No significant tanking signals"


# Example usage
if __name__ == "__main__":
    detector = TankingDetector()
    
    print("Tanking Detection Algorithm Test")
    print("=" * 50)
    
    # Example: Lottery-bound team
    print("\nScenario: Lottery-Bound Team")
    print("-" * 50)
    
    star_rest_prob = detector.calculate_star_rest_probability(
        team_id=1,
        game_date=datetime(2026, 3, 15),  # Late season
        is_back_to_back=True,
        star_players=[],
        recent_rest_games=3  # Rested 3 of last 10
    )
    print(f"Star Rest Probability: {star_rest_prob:.1%}")
    
    young_minutes_pct = 0.68  # 68% minutes to young players
    print(f"Young Player Minutes: {young_minutes_pct:.1%}")
    
    veteran_trades = 2
    print(f"Veteran Trades (last 30 days): {veteran_trades}")
    
    cluster = detector.classify_team_cluster(
        win_pct=0.280,
        net_rating=-8.5,
        playoff_odds=0.01
    )
    print(f"Team Cluster: {cluster}")
    
    tanking_score = detector.calculate_tanking_score(
        star_rest_prob=star_rest_prob,
        young_minutes_pct=young_minutes_pct,
        veteran_trades=veteran_trades,
        cluster=cluster
    )
    print(f"\nTanking Score: {tanking_score:.1f}/100")
    
    recommendation = detector.get_tanking_recommendation(tanking_score)
    print(f"Recommendation: {recommendation}")
    
    # Soft tanking signals
    recent_games = [
        {'q4_tov': 5, 'result': 'loss', 'margin': 3},
        {'q4_tov': 6, 'result': 'loss', 'margin': 2},
        {'q4_tov': 4, 'result': 'loss', 'margin': 7},
    ]
    
    signals = detector.detect_soft_tanking_signals(
        team_stats={'star_ft_rate': 0.18, 'season_avg_ft_rate': 0.25},
        recent_games=recent_games
    )
    
    print(f"\nSoft Tanking Signals:")
    for signal, value in signals.items():
        print(f"  {signal}: {value}")
