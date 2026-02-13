import asyncio
from app.services.prediction_service import PredictionService

async def test_matchup():
    print("Testing PredictionService.predict_matchup()...")
    service = PredictionService()
    
    # Test a known matchup
    try:
        result = await service.predict_matchup("LAL", "BOS", "2024-02-14")
        print("\nPrediction Result:")
        print(f"Matchup: {result['home_team']} vs {result['away_team']}")
        print(f"Winner: {result['predicted_winner']} ({result['home_win_probability']:.1%})")
        print(f"Confidence: {result['confidence_level']}")
        print("\nSentiment Analysis:")
        print(f"Home Sentiment: {result['sentiment_analysis']['home_score']}")
        print(f"Away Sentiment: {result['sentiment_analysis']['away_score']}")
        print("\nTop Factors:")
        for factor in result['top_factors']:
            print(f"- {factor}")
            
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(test_matchup())
