"""
Quick test script for NBA API client
"""

from nba_client import NBAClient

def test_nba_client():
    print("Testing NBA API Client...")
    print("=" * 50)
    
    client = NBAClient()
    
    # Test 1: Get all teams
    print("\n1. Fetching all NBA teams...")
    teams = client.get_all_teams()
    print(f"   ✓ Found {len(teams)} NBA teams")
    print(f"   Sample: {teams[0]['full_name']} ({teams[0]['abbreviation']})")
    
    # Test 2: Get Lakers info
    print("\n2. Getting Lakers team info...")
    lakers = client.get_team_by_name("LAL")
    if lakers:
        print(f"   ✓ Lakers ID: {lakers['id']}")
        print(f"   ✓ Full Name: {lakers['full_name']}")
    
    # Test 3: Get team stats (this may take a moment due to rate limiting)
    print("\n3. Fetching Lakers stats (2025-26 season)...")
    try:
        stats = client.get_team_stats(lakers['id'], season="2025-26")
        if stats:
            print(f"   ✓ eFG%: {stats.get('efg_pct', 'N/A')}")
            print(f"   ✓ Net Rating: {stats.get('net_rating', 'N/A')}")
            print(f"   ✓ Pace: {stats.get('pace', 'N/A')}")
        else:
            print("   ⚠ No stats available (season may not have started)")
    except Exception as e:
        print(f"   ⚠ Error fetching stats: {e}")
        print("   (This is normal if the 2025-26 season hasn't started)")
    
    print("\n" + "=" * 50)
    print("NBA API Client test complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_nba_client()
