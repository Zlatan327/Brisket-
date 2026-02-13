"""
Quick test script for Wikipedia client
"""

from app.data.wikipedia_client import WikipediaClient

def test_wikipedia_client():
    print("Testing Wikipedia Client...")
    print("=" * 50)
    
    client = WikipediaClient()
    
    # Test with LeBron James
    player = "LeBron James"
    
    print(f"\n1. Searching for {player}...")
    page_title = client.search_player(player)
    if page_title:
        print(f"   [OK] Found: {page_title}")
    else:
        print(f"   [FAIL] Not found")
        return
    
    print(f"\n2. Getting Wikipedia URL...")
    url = client.get_player_page_url(player)
    if url:
        print(f"   [OK] URL: {url[:60]}...")
    
    print(f"\n3. Getting player summary...")
    summary = client.get_player_summary(player)
    if summary:
        print(f"   [OK] Summary: {summary[:150]}...")
    
    print(f"\n4. Extracting injury history...")
    injuries = client.extract_injury_history(player)
    print(f"   [OK] Found {len(injuries)} injury mentions")
    if len(injuries) > 0:
        print(f"   Sample: {injuries[0]}")
    
    print(f"\n5. Calculating injury risk score...")
    risk_score = client.calculate_injury_risk_score(player)
    print(f"   [OK] Risk Score: {risk_score:.2f} (0.0 = low, 1.0 = high)")
    
    print("\n" + "=" * 50)
    print("Wikipedia Client test complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_wikipedia_client()
