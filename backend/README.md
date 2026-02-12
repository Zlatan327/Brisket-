# NBA + EuroLeague Basketball Prediction Tool - Backend

## Overview
Backend API for the NBA + EuroLeague basketball prediction tool. Built with FastAPI, PostgreSQL, and machine learning models (DNN + XGBoost).

## Project Structure
```
backend/
├── models.py                 # SQLAlchemy ORM models
├── database.py               # Database configuration
├── nba_client.py            # NBA API wrapper
├── euroleague_client.py     # EuroLeague API wrapper
├── wikipedia_client.py      # Wikipedia scraper
├── injury_aggregator.py     # Injury report aggregator
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt

# Install euroleaguer from GitHub
pip install git+https://github.com/FlavioLeccese92/euroleaguer.git
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your database credentials and API keys
```

### 3. Initialize Database
```bash
python database.py
```

This will create all tables in your PostgreSQL database.

## Database Schema

### Tables
- **teams**: NBA and EuroLeague teams with cluster classification
- **players**: Player roster with injury tracking and advanced metrics
- **games**: Game schedule and results with rest/travel data
- **advanced_metrics**: Four Factors and team ratings per game
- **predictions**: Model predictions with SHAP explanations
- **tanking_indicators**: NBA tanking signals (star rest, young lineups, trades)
- **sentiment_data**: GenAI news sentiment analysis

## Data Sources

### NBA Data (nba_api)
```python
from nba_client import NBAClient

client = NBAClient()
teams = client.get_all_teams()
stats = client.get_team_stats(team_id=1610612747)  # Lakers
```

### EuroLeague Data (euroleaguer)
```python
from euroleague_client import EuroLeagueClient

client = EuroLeagueClient()
teams = client.get_all_teams(season=2025)
```

### Player Injuries
```python
from injury_aggregator import InjuryReportAggregator

aggregator = InjuryReportAggregator()
injuries = aggregator.aggregate_all_injuries()
```

### Wikipedia Context
```python
from wikipedia_client import WikipediaClient

wiki = WikipediaClient()
context = wiki.get_player_context("LeBron James")
print(context['injury_risk_score'])
```

## Next Steps

### Phase 2: Feature Engineering
- Implement Four Factors calculation
- Build Second Apron tracking
- Create tanking detection algorithm
- Develop travel/fatigue calculator
- Build sentiment analysis pipeline

### Phase 3: ML Models
- Implement DNN model
- Implement XGBoost model
- Create ensemble voting system
- Build backtesting framework

### Phase 4: API Development
- Set up FastAPI routes
- Implement prediction endpoints
- Add background jobs (Celery)
- Set up caching (Redis)

## Development

### Run Database Initialization
```bash
python database.py
```

### Test NBA Client
```bash
python nba_client.py
```

### Test Wikipedia Client
```bash
python wikipedia_client.py
```

## Notes

- All API clients include rate limiting to avoid overwhelming data sources
- Wikipedia scraper extracts injury history for risk assessment
- Injury aggregator combines multiple sources for comprehensive coverage
- Database schema supports both NBA and EuroLeague with league-specific fields
