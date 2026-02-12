"""
Complete Travel Distance Calculator
Real NBA city locations and distances
"""

from typing import Dict, Tuple
import math


class NBALocationCalculator:
    """
    NBA team locations and travel calculator
    Uses actual city coordinates
    """
    
    # NBA team locations (latitude, longitude, elevation in feet)
    NBA_CITIES = {
        # Eastern Conference
        'Atlanta Hawks': (33.7573, -84.3963, 1050),
        'Boston Celtics': (42.3662, -71.0621, 20),
        'Brooklyn Nets': (40.6826, -73.9754, 20),
        'Charlotte Hornets': (35.2251, -80.8392, 750),
        'Chicago Bulls': (41.8807, -87.6742, 600),
        'Cleveland Cavaliers': (41.4965, -81.6882, 650),
        'Detroit Pistons': (42.3410, -83.0550, 600),
        'Indiana Pacers': (39.7640, -86.1555, 720),
        'Miami Heat': (25.7814, -80.1870, 10),
        'Milwaukee Bucks': (43.0436, -87.9170, 600),
        'New York Knicks': (40.7505, -73.9934, 35),
        'Orlando Magic': (28.5392, -81.3839, 100),
        'Philadelphia 76ers': (39.9012, -75.1720, 40),
        'Toronto Raptors': (43.6435, -79.3791, 250),
        'Washington Wizards': (38.8981, -77.0209, 25),
        
        # Western Conference
        'Dallas Mavericks': (32.7905, -96.8103, 430),
        'Denver Nuggets': (39.7487, -105.0077, 5280),  # Mile High
        'Golden State Warriors': (37.7680, -122.3877, 10),
        'Houston Rockets': (29.7508, -95.3621, 50),
        'Los Angeles Clippers': (34.0430, -118.2673, 300),
        'Los Angeles Lakers': (34.0430, -118.2673, 300),
        'Memphis Grizzlies': (35.1382, -90.0505, 280),
        'Minnesota Timberwolves': (44.9795, -93.2760, 830),
        'New Orleans Pelicans': (29.9490, -90.0821, 10),
        'Oklahoma City Thunder': (35.4634, -97.5151, 1200),
        'Phoenix Suns': (33.4457, -112.0712, 1100),
        'Portland Trail Blazers': (45.5316, -122.6668, 50),
        'Sacramento Kings': (38.5802, -121.4997, 30),
        'San Antonio Spurs': (29.4270, -98.4375, 650),
        'Utah Jazz': (40.7683, -111.9011, 4200)
    }
    
    # Timezone offsets from ET (Eastern Time)
    TIMEZONES = {
        'Atlanta Hawks': 0,
        'Boston Celtics': 0,
        'Brooklyn Nets': 0,
        'Charlotte Hornets': 0,
        'Cleveland Cavaliers': 0,
        'Detroit Pistons': 0,
        'Indiana Pacers': 0,
        'Miami Heat': 0,
        'New York Knicks': 0,
        'Orlando Magic': 0,
        'Philadelphia 76ers': 0,
        'Toronto Raptors': 0,
        'Washington Wizards': 0,
        
        'Chicago Bulls': -1,  # CT
        'Dallas Mavericks': -1,
        'Houston Rockets': -1,
        'Memphis Grizzlies': -1,
        'Milwaukee Bucks': -1,
        'Minnesota Timberwolves': -1,
        'New Orleans Pelicans': -1,
        'Oklahoma City Thunder': -1,
        'San Antonio Spurs': -1,
        
        'Denver Nuggets': -2,  # MT
        'Phoenix Suns': -2,
        'Utah Jazz': -2,
        
        'Golden State Warriors': -3,  # PT
        'Los Angeles Clippers': -3,
        'Los Angeles Lakers': -3,
        'Portland Trail Blazers': -3,
        'Sacramento Kings': -3
    }
    
    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in miles
        """
        R = 3959  # Earth's radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def calculate_travel_metrics(
        self,
        from_team: str,
        to_team: str
    ) -> Dict:
        """
        Calculate all travel metrics between two teams
        
        Args:
            from_team: Origin team name
            to_team: Destination team name
            
        Returns:
            Dictionary with travel metrics
        """
        if from_team not in self.NBA_CITIES or to_team not in self.NBA_CITIES:
            return {
                'distance': 0.0,
                'timezone_shift': 0,
                'elevation_change': 0
            }
        
        # Get coordinates
        from_lat, from_lon, from_elev = self.NBA_CITIES[from_team]
        to_lat, to_lon, to_elev = self.NBA_CITIES[to_team]
        
        # Calculate distance
        distance = self.haversine_distance(from_lat, from_lon, to_lat, to_lon)
        
        # Timezone shift
        from_tz = self.TIMEZONES.get(from_team, 0)
        to_tz = self.TIMEZONES.get(to_team, 0)
        timezone_shift = abs(to_tz - from_tz)
        
        # Elevation change
        elevation_change = abs(to_elev - from_elev)
        
        return {
            'distance': round(distance, 1),
            'timezone_shift': timezone_shift,
            'elevation_change': elevation_change
        }
    
    def get_longest_trips(self) -> list:
        """Get the 10 longest trips in the NBA"""
        trips = []
        
        teams = list(self.NBA_CITIES.keys())
        
        for i, team1 in enumerate(teams):
            for team2 in teams[i+1:]:
                metrics = self.calculate_travel_metrics(team1, team2)
                trips.append({
                    'from': team1,
                    'to': team2,
                    'distance': metrics['distance'],
                    'timezone_shift': metrics['timezone_shift'],
                    'elevation_change': metrics['elevation_change']
                })
        
        trips.sort(key=lambda x: x['distance'], reverse=True)
        return trips[:10]


# Example usage
if __name__ == "__main__":
    print("NBA Location Calculator")
    print("=" * 60)
    
    calc = NBALocationCalculator()
    
    # Test some common trips
    print("\nCommon NBA Trips:")
    print("-" * 60)
    
    trips = [
        ('Miami Heat', 'Portland Trail Blazers'),
        ('Boston Celtics', 'Los Angeles Lakers'),
        ('New York Knicks', 'Golden State Warriors'),
        ('Atlanta Hawks', 'Denver Nuggets'),
        ('Toronto Raptors', 'Phoenix Suns')
    ]
    
    for from_team, to_team in trips:
        metrics = calc.calculate_travel_metrics(from_team, to_team)
        print(f"\n{from_team} → {to_team}")
        print(f"  Distance: {metrics['distance']:.0f} miles")
        print(f"  Timezone shift: {metrics['timezone_shift']} hours")
        print(f"  Elevation change: {metrics['elevation_change']:,} feet")
    
    # Longest trips
    print("\n" + "=" * 60)
    print("10 LONGEST NBA TRIPS")
    print("=" * 60)
    
    longest = calc.get_longest_trips()
    
    for i, trip in enumerate(longest, 1):
        print(f"{i:2d}. {trip['from']:25s} → {trip['to']:25s}")
        print(f"    {trip['distance']:.0f} miles, {trip['timezone_shift']}hr TZ, {trip['elevation_change']:,}ft elev")
    
    # Denver altitude impact
    print("\n" + "=" * 60)
    print("DENVER ALTITUDE IMPACT (Mile High City)")
    print("=" * 60)
    
    sea_level_teams = ['Miami Heat', 'Boston Celtics', 'Brooklyn Nets', 
                       'Golden State Warriors', 'Portland Trail Blazers']
    
    for team in sea_level_teams:
        metrics = calc.calculate_travel_metrics(team, 'Denver Nuggets')
        print(f"{team:30s} → Denver: {metrics['elevation_change']:,} ft elevation gain")
