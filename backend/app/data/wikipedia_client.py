"""
Wikipedia API Client - For player injury history and context
Uses Wikipedia API to fetch player biographies and injury information
"""

import wikipedia
import requests
from typing import Dict, Optional, List
from datetime import datetime
import re


class WikipediaClient:
    """Client for fetching player information from Wikipedia"""
    
    def __init__(self):
        wikipedia.set_lang("en")
        self.api_url = "https://en.wikipedia.org/w/api.php"
    
    def search_player(self, player_name: str) -> Optional[str]:
        """
        Search for a player's Wikipedia page
        
        Args:
            player_name: Player's full name
            
        Returns:
            Wikipedia page title if found, None otherwise
        """
        try:
            # Search for the player
            search_results = wikipedia.search(f"{player_name} basketball")
            
            if len(search_results) == 0:
                return None
            
            # Return the first result (most relevant)
            return search_results[0]
        
        except Exception as e:
            print(f"Error searching for {player_name}: {e}")
            return None
    
    def get_player_page_url(self, player_name: str) -> Optional[str]:
        """
        Get the URL of a player's Wikipedia page
        
        Args:
            player_name: Player's full name
            
        Returns:
            Wikipedia page URL
        """
        page_title = self.search_player(player_name)
        
        if not page_title:
            return None
        
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            return page.url
        except Exception as e:
            print(f"Error getting page URL for {player_name}: {e}")
            return None
    
    def get_player_summary(self, player_name: str) -> Optional[str]:
        """
        Get a summary of the player's Wikipedia page
        
        Args:
            player_name: Player's full name
            
        Returns:
            Summary text
        """
        page_title = self.search_player(player_name)
        
        if not page_title:
            return None
        
        try:
            summary = wikipedia.summary(page_title, sentences=5, auto_suggest=False)
            return summary
        except Exception as e:
            print(f"Error getting summary for {player_name}: {e}")
            return None
    
    def extract_injury_history(self, player_name: str) -> List[Dict]:
        """
        Extract injury history from player's Wikipedia page
        
        Args:
            player_name: Player's full name
            
        Returns:
            List of injury dictionaries with type, year, and severity
        """
        page_title = self.search_player(player_name)
        
        if not page_title:
            return []
        
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            content = page.content.lower()
            
            injuries = []
            
            # Common injury keywords
            injury_keywords = {
                'acl': {'type': 'ACL tear', 'severity': 'severe', 'recovery_months': 9},
                'mcl': {'type': 'MCL tear', 'severity': 'moderate', 'recovery_months': 3},
                'achilles': {'type': 'Achilles tear', 'severity': 'severe', 'recovery_months': 12},
                'meniscus': {'type': 'Meniscus tear', 'severity': 'moderate', 'recovery_months': 4},
                'ankle sprain': {'type': 'Ankle sprain', 'severity': 'mild', 'recovery_months': 1},
                'hamstring': {'type': 'Hamstring strain', 'severity': 'mild', 'recovery_months': 2},
                'concussion': {'type': 'Concussion', 'severity': 'moderate', 'recovery_months': 1},
                'broken': {'type': 'Fracture', 'severity': 'moderate', 'recovery_months': 3},
                'fractured': {'type': 'Fracture', 'severity': 'moderate', 'recovery_months': 3},
            }
            
            # Search for injury mentions
            for keyword, injury_info in injury_keywords.items():
                if keyword in content:
                    # Try to extract year
                    pattern = rf'{keyword}.*?(\d{{4}})'
                    matches = re.findall(pattern, content)
                    
                    if matches:
                        for year in matches:
                            injuries.append({
                                'type': injury_info['type'],
                                'year': int(year),
                                'severity': injury_info['severity'],
                                'recovery_months': injury_info['recovery_months']
                            })
                    else:
                        # Add without year if found
                        injuries.append({
                            'type': injury_info['type'],
                            'year': None,
                            'severity': injury_info['severity'],
                            'recovery_months': injury_info['recovery_months']
                        })
            
            return injuries
        
        except Exception as e:
            print(f"Error extracting injury history for {player_name}: {e}")
            return []
    
    def calculate_injury_risk_score(self, player_name: str) -> float:
        """
        Calculate injury risk score based on Wikipedia history
        
        Args:
            player_name: Player's full name
            
        Returns:
            Risk score (0.0 - 1.0, where 1.0 is highest risk)
        """
        injuries = self.extract_injury_history(player_name)
        
        if len(injuries) == 0:
            return 0.0  # No documented injury history
        
        current_year = datetime.now().year
        risk_score = 0.0
        
        for injury in injuries:
            # Base severity score
            severity_scores = {
                'severe': 0.4,
                'moderate': 0.2,
                'mild': 0.1
            }
            
            base_score = severity_scores.get(injury['severity'], 0.1)
            
            # Recency multiplier (more recent = higher risk)
            if injury['year']:
                years_ago = current_year - injury['year']
                recency_multiplier = max(0.1, 1.0 - (years_ago * 0.1))
            else:
                recency_multiplier = 0.5  # Unknown year
            
            risk_score += base_score * recency_multiplier
        
        # Cap at 1.0
        return min(1.0, risk_score)
    
    def get_player_context(self, player_name: str) -> Dict:
        """
        Get comprehensive player context from Wikipedia
        
        Args:
            player_name: Player's full name
            
        Returns:
            Dictionary with URL, summary, injuries, and risk score
        """
        return {
            'wikipedia_url': self.get_player_page_url(player_name),
            'summary': self.get_player_summary(player_name),
            'injury_history': self.extract_injury_history(player_name),
            'injury_risk_score': self.calculate_injury_risk_score(player_name)
        }


# Example usage
if __name__ == "__main__":
    client = WikipediaClient()
    
    # Test with a well-known player
    player = "LeBron James"
    context = client.get_player_context(player)
    
    print(f"\n{player} Context:")
    print(f"URL: {context['wikipedia_url']}")
    print(f"Summary: {context['summary'][:200]}...")
    print(f"Injury History: {context['injury_history']}")
    print(f"Injury Risk Score: {context['injury_risk_score']:.2f}")
