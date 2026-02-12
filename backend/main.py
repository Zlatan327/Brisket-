"""
FastAPI Main Application
NBA + EuroLeague Basketball Prediction API
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np

# Import our modules
from ensemble_predictor import EnsemblePredictor
from four_factors import FourFactorsCalculator
from tanking_detector import TankingDetector
from travel_fatigue import TravelFatigueCalculator
from sentiment_analyzer import SentimentAnalyzer
from shap_explainer import SHAPExplainer

# Initialize FastAPI app
app = FastAPI(
    title="NBA + EuroLeague Prediction API",
    description="AI-driven basketball game prediction with 87% target accuracy",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (will be loaded from disk in production)
ensemble = EnsemblePredictor()
four_factors_calc = FourFactorsCalculator()
tanking_detector = TankingDetector()
travel_calc = TravelFatigueCalculator()
sentiment_analyzer = SentimentAnalyzer()
shap_explainer = SHAPExplainer()


# Pydantic models for request/response
class GameFeatures(BaseModel):
    """Features for a single game prediction"""
    home_team: str
    away_team: str
    game_date: str
    league: str  # "NBA" or "EuroLeague"
    
    # Four Factors
    home_efg_pct: float
    away_efg_pct: float
    home_tov_rate: float
    away_tov_rate: float
    home_drb_pct: float
    away_drb_pct: float
    home_ft_rate: float
    away_ft_rate: float
    
    # Advanced metrics
    home_net_rating: float
    away_net_rating: float
    home_pace: float
    away_pace: float
    
    # Rest and travel
    home_rest_days: int
    away_rest_days: int
    travel_distance: float
    timezone_shift: int
    elevation_change: int
    
    # Fatigue
    home_fatigue_score: float
    away_fatigue_score: float
    home_cumulative_load: float
    away_cumulative_load: float
    
    # Form
    home_last_5_wins: int
    away_last_5_wins: int
    home_last_10_wins: int
    away_last_10_wins: int
    
    # Injuries
    home_injury_impact: float
    away_injury_impact: float
    
    # Sentiment
    home_sentiment_score: float
    away_sentiment_score: float
    home_trade_rumors: float
    away_trade_rumors: float
    home_coaching_stability: float
    away_coaching_stability: float
    
    # Tanking
    home_tanking_score: float
    away_tanking_score: float
    
    # Context
    is_back_to_back_home: int
    is_back_to_back_away: int
    home_court_advantage: float


class PredictionResponse(BaseModel):
    """Prediction response"""
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    league: str
    
    # Predictions
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str
    confidence_level: str  # HIGH, MEDIUM, LOW
    
    # Model breakdown
    dnn_prediction: Optional[float] = None
    xgb_prediction: Optional[float] = None
    ensemble_weights: Dict[str, float]
    
    # Explanation
    top_factors: List[Dict]
    
    # Metadata
    prediction_timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool


# API Endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "NBA + EuroLeague Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=ensemble.is_trained
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_game(features: GameFeatures):
    """
    Predict game winner
    
    Args:
        features: Game features
        
    Returns:
        Prediction with explanation
    """
    try:
        # Convert features to numpy array
        feature_array = np.array([
            features.home_efg_pct, features.away_efg_pct,
            features.home_tov_rate, features.away_tov_rate,
            features.home_drb_pct, features.away_drb_pct,
            features.home_ft_rate, features.away_ft_rate,
            features.home_net_rating, features.away_net_rating,
            features.home_pace, features.away_pace,
            features.home_rest_days, features.away_rest_days,
            features.travel_distance, features.timezone_shift, features.elevation_change,
            features.home_fatigue_score, features.away_fatigue_score,
            features.home_cumulative_load, features.away_cumulative_load,
            features.home_last_5_wins, features.away_last_5_wins,
            features.home_last_10_wins, features.away_last_10_wins,
            features.home_injury_impact, features.away_injury_impact,
            features.home_sentiment_score, features.away_sentiment_score,
            features.home_trade_rumors, features.away_trade_rumors,
            features.home_coaching_stability, features.away_coaching_stability,
            features.home_tanking_score, features.away_tanking_score,
            features.is_back_to_back_home, features.is_back_to_back_away,
            features.home_court_advantage
        ]).reshape(1, -1)
        
        # Get predictions
        prediction_details = ensemble.predict_with_details(feature_array)
        home_win_prob = prediction_details['ensemble_predictions'][0]
        
        # Get SHAP explanation
        explanation = shap_explainer.explain_prediction(feature_array[0])
        
        # Determine winner and confidence
        predicted_winner = features.home_team if home_win_prob > 0.5 else features.away_team
        confidence = ensemble.get_confidence_level(home_win_prob)
        
        return PredictionResponse(
            game_id=f"{features.home_team}_{features.away_team}_{features.game_date}",
            home_team=features.home_team,
            away_team=features.away_team,
            game_date=features.game_date,
            league=features.league,
            home_win_probability=home_win_prob,
            away_win_probability=1.0 - home_win_prob,
            predicted_winner=predicted_winner,
            confidence_level=confidence,
            dnn_prediction=prediction_details['dnn_predictions'][0],
            xgb_prediction=prediction_details['xgb_predictions'][0],
            ensemble_weights=ensemble.weights,
            top_factors=explanation['top_5_features'],
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/teams/{league}", response_model=List[str])
async def get_teams(league: str):
    """Get all teams for a league"""
    if league.upper() == "NBA":
        # Return NBA teams
        return ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
                "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
                "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
    elif league.upper() == "EUROLEAGUE":
        # Return EuroLeague teams (placeholder)
        return ["Barcelona", "Real Madrid", "Olympiacos", "Panathinaikos", "Fenerbahce"]
    else:
        raise HTTPException(status_code=400, detail="Invalid league. Use 'NBA' or 'EuroLeague'")


@app.get("/models/status", response_model=Dict)
async def model_status():
    """Get model status and performance metrics"""
    return {
        "ensemble_trained": ensemble.is_trained,
        "ensemble_weights": ensemble.weights,
        "feature_count": 37,
        "target_accuracy": 0.87,
        "update_schedule": "Daily at 6 AM ET, Pre-game at 2 hours before tip-off"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
