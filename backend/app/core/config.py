from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import os

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
        # Find .env file in backend root (3 levels up from this file)
        base_path = Path(__file__).resolve().parent.parent.parent
        env_file = os.path.join(base_path, ".env")
        env_file_encoding = 'utf-8'

settings = Settings()
