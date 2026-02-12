from ml.q4_feature_eng import Q4FeatureEngineer
from ml.q4_model import Q4PredictionModel
import numpy as np

def test_q4_pipeline():
    print("Testing Q4 Prediction Pipeline...")
    
    # 1. Mock Game State
    game_state = {
        'home_score_q3': 85,
        'away_score_q3': 82,   # +3 Home Lead
        'home_clutch_stats': {'ortg': 115.0, 'drtg': 105.0, 'pace': 96.0},
        'away_clutch_stats': {'ortg': 108.0, 'drtg': 112.0, 'pace': 98.0},
        'pre_game_prob': 0.65
    }
    
    # 2. Extract Features
    engineer = Q4FeatureEngineer()
    features = engineer.extract_features(game_state)
    print("\nExtracted Features:")
    for k, v in features.items():
        print(f"  {k}: {v:.2f}")
        
    # 3. Predict
    model = Q4PredictionModel()
    # No training yet, so it should use fallback logic
    
    X = engineer.prepare_for_inference(features)
    prob = model.predict(X)
    
    print(f"\nPredicted Win Probability (Home): {prob:.2%}")
    
    # Test Fallback Logic Sanity
    # If home leads by 20, prob should be very high
    game_state_blowout = game_state.copy()
    game_state_blowout['home_score_q3'] = 100
    features_blowout = engineer.extract_features(game_state_blowout)
    X_blowout = engineer.prepare_for_inference(features_blowout)
    prob_blowout = model.predict(X_blowout)
    print(f"Blowout Lead (+18) Win Prob: {prob_blowout:.2%}")
    
    assert prob_blowout > prob, "Blowout should have higher probability"
    print("\nSanity Check Passed!")

if __name__ == "__main__":
    test_q4_pipeline()
