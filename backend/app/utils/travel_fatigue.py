"""
Travel and Fatigue Calculator
Calculates rest days, travel distance, and fatigue impact on performance
"""

from typing import Dict, Tuple
from datetime import datetime, timedelta
import math


class TravelFatigueCalculator:
    """
    Calculates travel and fatigue metrics for NBA and EuroLeague
    
    Factors:
    - Rest days between games
    - Travel distance
    - Timezone shifts
    - Cumulative load (5-day and 10-day)
    - Elevation changes
    """
    
    # City coordinates (latitude, longitude, elevation in feet)
    NBA_CITIES = {
        'ATL': (33.7490, -84.3880, 1050),   # Atlanta
        'BOS': (42.3601, -71.0589, 20),     # Boston
        'BKN': (40.6782, -73.9442, 20),     # Brooklyn
        'CHA': (35.2271, -80.8431, 750),    # Charlotte
        'CHI': (41.8781, -87.6298, 600),    # Chicago
        'CLE': (41.4993, -81.6944, 650),    # Cleveland
        'DAL': (32.7767, -96.7970, 430),    # Dallas
        'DEN': (39.7392, -104.9903, 5280),  # Denver (Mile High!)
        'DET': (42.3314, -83.0458, 600),    # Detroit
        'GSW': (37.7749, -122.4194, 50),    # Golden State
        'HOU': (29.7604, -95.3698, 50),     # Houston
        'IND': (39.7684, -86.1581, 715),    # Indiana
        'LAC': (34.0522, -118.2437, 300),   # LA Clippers
        'LAL': (34.0522, -118.2437, 300),   # LA Lakers
        'MEM': (35.1495, -90.0490, 330),    # Memphis
        'MIA': (25.7617, -80.1918, 10),     # Miami
        'MIL': (43.0389, -87.9065, 600),    # Milwaukee
        'MIN': (44.9778, -93.2650, 830),    # Minnesota
        'NOP': (29.9511, -90.0715, 10),     # New Orleans
        'NYK': (40.7128, -74.0060, 30),     # New York
        'OKC': (35.4676, -97.5164, 1200),   # Oklahoma City
        'ORL': (28.5383, -81.3792, 80),     # Orlando
        'PHI': (39.9526, -75.1652, 40),     # Philadelphia
        'PHX': (33.4484, -112.0740, 1100),  # Phoenix
        'POR': (45.5152, -122.6784, 50),    # Portland
        'SAC': (38.5816, -121.4944, 30),    # Sacramento
        'SAS': (29.4241, -98.4936, 650),    # San Antonio
        'TOR': (43.6532, -79.3832, 250),    # Toronto
        'UTA': (40.7608, -111.8910, 4200),  # Utah
        'WAS': (38.9072, -77.0369, 10),     # Washington
    }
    
    def __init__(self):
        pass
    
    def calculate_distance(self, city1: str, city2: str, league: str = "NBA") -> float:
        """
        Calculate great circle distance between two cities
        
        Args:
            city1: First city code
            city2: Second city code
            league: "NBA" or "EuroLeague"
            
        Returns:
            Distance in miles
        """
        if league == "NBA":
            if city1 not in self.NBA_CITIES or city2 not in self.NBA_CITIES:
                return 500.0  # Default
            
            lat1, lon1, _ = self.NBA_CITIES[city1]
            lat2, lon2, _ = self.NBA_CITIES[city2]
        else:
            # EuroLeague - would need separate city database
            return 1000.0  # Default for EuroLeague
        
        # Haversine formula
        R = 3959  # Earth radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def calculate_timezone_shift(self, city1: str, city2: str) -> int:
        """
        Calculate timezone difference between cities
        
        Args:
            city1: First city code
            city2: Second city code
            
        Returns:
            Timezone shift in hours (absolute value)
        """
        # Simplified timezone mapping for NBA cities
        timezones = {
            'ATL': -5, 'BOS': -5, 'BKN': -5, 'CHA': -5, 'CLE': -5,
            'DET': -5, 'IND': -5, 'MIA': -5, 'NYK': -5, 'ORL': -5,
            'PHI': -5, 'TOR': -5, 'WAS': -5,  # Eastern
            'CHI': -6, 'DAL': -6, 'HOU': -6, 'MEM': -6, 'MIN': -6,
            'MIL': -6, 'NOP': -6, 'OKC': -6, 'SAS': -6,  # Central
            'DEN': -7, 'PHX': -7, 'POR': -7, 'UTA': -7,  # Mountain
            'GSW': -8, 'LAC': -8, 'LAL': -8, 'SAC': -8,  # Pacific
        }
        
        tz1 = timezones.get(city1, -5)
        tz2 = timezones.get(city2, -5)
        
        return abs(tz2 - tz1)
    
    def calculate_elevation_change(self, city1: str, city2: str) -> int:
        """
        Calculate elevation change between cities
        
        Args:
            city1: First city code
            city2: Second city code
            
        Returns:
            Elevation change in feet (absolute value)
        """
        if city1 not in self.NBA_CITIES or city2 not in self.NBA_CITIES:
            return 0
        
        _, _, elev1 = self.NBA_CITIES[city1]
        _, _, elev2 = self.NBA_CITIES[city2]
        
        return abs(elev2 - elev1)
    
    def calculate_fatigue_impact(
        self,
        rest_days: int,
        travel_distance: float,
        timezone_shift: int,
        elevation_change: int,
        is_back_to_back: bool
    ) -> float:
        """
        Calculate fatigue impact score (0-100)
        Higher = more fatigue = worse performance expected
        
        Args:
            rest_days: Days of rest before game
            travel_distance: Distance traveled in miles
            timezone_shift: Timezone difference in hours
            elevation_change: Elevation change in feet
            is_back_to_back: Whether game is on back-to-back
            
        Returns:
            Fatigue impact score (0-100)
        """
        fatigue_score = 0.0
        
        # 1. Rest days (40% weight)
        if is_back_to_back:
            fatigue_score += 40
        elif rest_days == 1:
            fatigue_score += 25
        elif rest_days == 2:
            fatigue_score += 10
        # 3+ days rest = minimal fatigue
        
        # 2. Travel distance (30% weight)
        if travel_distance > 2000:  # Cross-country
            fatigue_score += 30
        elif travel_distance > 1000:
            fatigue_score += 20
        elif travel_distance > 500:
            fatigue_score += 10
        
        # 3. Timezone shift (20% weight)
        fatigue_score += timezone_shift * 5  # 5 points per hour
        
        # 4. Elevation change (10% weight)
        if elevation_change > 4000:  # Going to/from Denver
            fatigue_score += 10
        elif elevation_change > 2000:
            fatigue_score += 5
        
        return min(100.0, fatigue_score)
    
    def calculate_cumulative_load(
        self,
        games_last_5_days: int,
        games_last_10_days: int,
        total_miles_last_10_days: float
    ) -> float:
        """
        Calculate cumulative load score
        
        Args:
            games_last_5_days: Number of games in last 5 days
            games_last_10_days: Number of games in last 10 days
            total_miles_last_10_days: Total travel miles in last 10 days
            
        Returns:
            Cumulative load score (0-100)
        """
        load_score = 0.0
        
        # Games in last 5 days (50% weight)
        if games_last_5_days >= 4:  # "4 in 5 nights"
            load_score += 50
        elif games_last_5_days == 3:
            load_score += 35
        elif games_last_5_days == 2:
            load_score += 20
        
        # Games in last 10 days (30% weight)
        if games_last_10_days >= 6:
            load_score += 30
        elif games_last_10_days >= 5:
            load_score += 20
        elif games_last_10_days >= 4:
            load_score += 10
        
        # Travel miles (20% weight)
        if total_miles_last_10_days > 5000:
            load_score += 20
        elif total_miles_last_10_days > 3000:
            load_score += 10
        
        return min(100.0, load_score)
    
    def get_performance_adjustment(self, fatigue_score: float) -> float:
        """
        Convert fatigue score to win probability adjustment
        
        Args:
            fatigue_score: Fatigue score (0-100)
            
        Returns:
            Win probability adjustment (-0.15 to 0.0)
        """
        # High fatigue = lower win probability
        # Max penalty: -15% win probability
        return -(fatigue_score / 100) * 0.15


# Example usage
if __name__ == "__main__":
    calc = TravelFatigueCalculator()
    
    print("Travel & Fatigue Calculator Test")
    print("=" * 50)
    
    # Scenario: Lakers traveling to Boston on back-to-back
    print("\nScenario: Lakers @ Celtics (back-to-back)")
    print("-" * 50)
    
    distance = calc.calculate_distance('LAL', 'BOS')
    print(f"Travel Distance: {distance:.0f} miles")
    
    tz_shift = calc.calculate_timezone_shift('LAL', 'BOS')
    print(f"Timezone Shift: {tz_shift} hours")
    
    elev_change = calc.calculate_elevation_change('LAL', 'BOS')
    print(f"Elevation Change: {elev_change} feet")
    
    fatigue = calc.calculate_fatigue_impact(
        rest_days=0,
        travel_distance=distance,
        timezone_shift=tz_shift,
        elevation_change=elev_change,
        is_back_to_back=True
    )
    print(f"\nFatigue Impact: {fatigue:.1f}/100")
    
    # Cumulative load
    cumulative = calc.calculate_cumulative_load(
        games_last_5_days=4,  # 4 in 5 nights!
        games_last_10_days=6,
        total_miles_last_10_days=5500
    )
    print(f"Cumulative Load: {cumulative:.1f}/100")
    
    # Performance adjustment
    adjustment = calc.get_performance_adjustment(fatigue)
    print(f"\nWin Probability Adjustment: {adjustment:.1%}")
    print(f"(Lakers' win probability reduced by {abs(adjustment):.1%})")
    
    # Scenario 2: Denver home game (elevation advantage)
    print("\n\nScenario: Heat @ Nuggets (elevation)")
    print("-" * 50)
    
    distance2 = calc.calculate_distance('MIA', 'DEN')
    elev_change2 = calc.calculate_elevation_change('MIA', 'DEN')
    
    print(f"Travel Distance: {distance2:.0f} miles")
    print(f"Elevation Change: {elev_change2} feet (Mile High!)")
    
    fatigue2 = calc.calculate_fatigue_impact(
        rest_days=2,
        travel_distance=distance2,
        timezone_shift=2,
        elevation_change=elev_change2,
        is_back_to_back=False
    )
    print(f"Fatigue Impact: {fatigue2:.1f}/100")
