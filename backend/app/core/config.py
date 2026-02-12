from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "NBA + EuroLeague Prediction API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./nba_predictions.db"
    
    # Security
    SECRET_KEY: str = "YOUR_SECRET_KEY_HERE"
    
    # APIs
    NBA_API_TIMEOUT: int = 30
    TAVILY_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # ML Models
    MODEL_PATH: str = "models/"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
