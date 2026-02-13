from app.utils.four_factors import FourFactorsCalculator
from app.utils.tanking_detector import TankingDetector
from app.utils.travel_fatigue import TravelFatigueCalculator
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_analyzers():
    logger.info("Verifying Data Analyzers...")
    
    # 1. Four Factors
    logger.info("\n1. Testing Four Factors Calculator...")
    ff_calc = FourFactorsCalculator()
    mock_stats = {
        'fgm': 40, 'fga': 85, 'fg3m': 12, 'ftm': 18, 'fta': 22,
        'oreb': 10, 'dreb': 35, 'tov': 14, 'opp_dreb': 32
    }
    ff = ff_calc.calculate_team_four_factors(mock_stats)
    logger.info(f"Four Factors Results: {ff}")
    assert 'efg_pct' in ff
    
    # 2. Tanking Detector
    logger.info("\n2. Testing Tanking Detector...")
    tank_detector = TankingDetector()
    
    # Calculate components first
    star_rest = tank_detector.calculate_star_rest_probability(
        team_id=1, game_date=datetime.now(), is_back_to_back=True, star_players=[], recent_rest_games=0
    )
    cluster = tank_detector.classify_team_cluster(0.25, -8.0, 0.0) # Bad team
    tank_score = tank_detector.calculate_tanking_score(
        star_rest_prob=star_rest, 
        young_minutes_pct=0.7, 
        veteran_trades=1, 
        cluster=cluster
    )
    recommendation = tank_detector.get_tanking_recommendation(tank_score)
    
    logger.info(f"Tanking Score (Simulated Bad Team): {tank_score}")
    logger.info(f"Recommendation: {recommendation}")
    assert tank_score > 0
    
    # 3. Travel Fatigue
    logger.info("\n3. Testing Travel Fatigue Calculator...")
    travel_calc = TravelFatigueCalculator()
    
    city1 = "LAL"
    city2 = "BOS"
    dist = travel_calc.calculate_distance(city1, city2)
    tz = travel_calc.calculate_timezone_shift(city1, city2)
    elev = travel_calc.calculate_elevation_change(city1, city2)
    
    fatigue = travel_calc.calculate_fatigue_impact(
        rest_days=0,
        travel_distance=dist,
        timezone_shift=tz,
        elevation_change=elev,
        is_back_to_back=True
    )
    
    logger.info(f"Travel: {city1} -> {city2}")
    logger.info(f"Distance: {dist:.0f}, TZ: {tz}, Elev: {elev}")
    logger.info(f"Fatigue Score: {fatigue}")
    assert fatigue > 0

if __name__ == "__main__":
    verify_analyzers()
