"""
Injury Report Aggregator
Fetches and aggregates injury reports from multiple sources
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import time


class InjuryReportAggregator:
    """Aggregates injury reports from multiple sources"""
    
    def __init__(self):
        self.sources = {
            'nba_official': 'https://official.nba.com/nba-injury-report-2025-26-season/',
            'espn': 'https://www.espn.com/nba/injuries',
            'rotowire': 'https://www.rotowire.com/basketball/nba-lineups.php'
        }
        self.rate_limit_delay = 1.0  # 1 second between requests
    
    def _rate_limit(self):
        """Sleep to avoid overwhelming sources"""
        time.sleep(self.rate_limit_delay)
    
    def fetch_nba_official_injuries(self) -> List[Dict]:
        """
        Fetch injuries from official NBA injury report
        
        Returns:
            List of injury dictionaries
        """
        self._rate_limit()
        
        injuries = []
        
        try:
            # Note: This is a placeholder - actual implementation will depend on
            # the structure of the official NBA injury report page
            response = requests.get(self.sources['nba_official'], timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse injury report table (structure varies)
                # This is a simplified example
                injury_rows = soup.find_all('tr', class_='injury-row')
                
                for row in injury_rows:
                    try:
                        player_name = row.find('td', class_='player-name').text.strip()
                        team = row.find('td', class_='team').text.strip()
                        status = row.find('td', class_='status').text.strip()
                        injury_type = row.find('td', class_='injury').text.strip()
                        
                        injuries.append({
                            'player_name': player_name,
                            'team': team,
                            'status': status.lower(),  # out, doubtful, questionable, probable
                            'injury_type': injury_type,
                            'source': 'nba_official',
                            'last_updated': datetime.utcnow()
                        })
                    except Exception as e:
                        print(f"Error parsing injury row: {e}")
                        continue
        
        except Exception as e:
            print(f"Error fetching NBA official injuries: {e}")
        
        return injuries
    
    def fetch_espn_injuries(self) -> List[Dict]:
        """
        Fetch injuries from ESPN
        
        Returns:
            List of injury dictionaries
        """
        self._rate_limit()
        
        injuries = []
        
        try:
            response = requests.get(self.sources['espn'], timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse ESPN injury page (structure varies)
                # Placeholder implementation
                pass
        
        except Exception as e:
            print(f"Error fetching ESPN injuries: {e}")
        
        return injuries
    
    def fetch_rotowire_injuries(self) -> List[Dict]:
        """
        Fetch injuries from RotoWire
        
        Returns:
            List of injury dictionaries
        """
        self._rate_limit()
        
        injuries = []
        
        try:
            response = requests.get(self.sources['rotowire'], timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse RotoWire lineups page
                # Placeholder implementation
                pass
        
        except Exception as e:
            print(f"Error fetching RotoWire injuries: {e}")
        
        return injuries
    
    def aggregate_all_injuries(self) -> List[Dict]:
        """
        Aggregate injuries from all sources
        
        Returns:
            Deduplicated list of injuries
        """
        all_injuries = []
        
        # Fetch from all sources
        all_injuries.extend(self.fetch_nba_official_injuries())
        all_injuries.extend(self.fetch_espn_injuries())
        all_injuries.extend(self.fetch_rotowire_injuries())
        
        # Deduplicate by player name (keep most recent)
        unique_injuries = {}
        
        for injury in all_injuries:
            player_name = injury['player_name']
            
            if player_name not in unique_injuries:
                unique_injuries[player_name] = injury
            else:
                # Keep the most recent update
                if injury['last_updated'] > unique_injuries[player_name]['last_updated']:
                    unique_injuries[player_name] = injury
        
        return list(unique_injuries.values())
    
    def get_team_injuries(self, team_name: str) -> List[Dict]:
        """
        Get injuries for a specific team
        
        Args:
            team_name: Team name or abbreviation
            
        Returns:
            List of injuries for that team
        """
        all_injuries = self.aggregate_all_injuries()
        
        team_injuries = [
            injury for injury in all_injuries
            if team_name.lower() in injury['team'].lower()
        ]
        
        return team_injuries
    
    def get_player_injury_status(self, player_name: str) -> Optional[Dict]:
        """
        Get current injury status for a specific player
        
        Args:
            player_name: Player's full name
            
        Returns:
            Injury dictionary if player is injured, None otherwise
        """
        all_injuries = self.aggregate_all_injuries()
        
        for injury in all_injuries:
            if player_name.lower() in injury['player_name'].lower():
                return injury
        
        return None
    
    def calculate_team_injury_impact(self, team_name: str, player_values: Dict[str, float]) -> float:
        """
        Calculate total injury impact for a team
        
        Args:
            team_name: Team name
            player_values: Dictionary mapping player names to their impact scores
            
        Returns:
            Total injury impact (0-100)
        """
        team_injuries = self.get_team_injuries(team_name)
        
        total_impact = 0.0
        
        severity_multipliers = {
            'out': 1.0,
            'doubtful': 0.7,
            'questionable': 0.3,
            'probable': 0.1
        }
        
        for injury in team_injuries:
            player_name = injury['player_name']
            status = injury['status']
            
            # Get player value (default to 10 if not found)
            player_value = player_values.get(player_name, 10.0)
            
            # Apply severity multiplier
            multiplier = severity_multipliers.get(status, 0.5)
            
            total_impact += player_value * multiplier
        
        return min(100.0, total_impact)


# Example usage
if __name__ == "__main__":
    aggregator = InjuryReportAggregator()
    
    print("Fetching injury reports...")
    injuries = aggregator.aggregate_all_injuries()
    
    print(f"\nFound {len(injuries)} total injuries")
    
    if len(injuries) > 0:
        print(f"\nSample injury: {injuries[0]}")
