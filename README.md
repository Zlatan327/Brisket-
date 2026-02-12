# NBA + EuroLeague Basketball Prediction Tool

🏀 Machine learning system for predicting NBA game winners with 64.6% accuracy (80.6% on high-confidence picks).

## 🎯 Current Performance

- **Overall Accuracy**: 64.6%
- **High-Confidence (75%+)**: 80.6% accuracy
- **Training Data**: 2,286 games (2024-26 seasons)
- **Features**: 37 with real travel data

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| Test Accuracy | 64.6% |
| Test AUC | 0.717 |
| Training Games | 2,286 |
| Current Season | 2025-26 |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nba-prediction-tool.git
cd nba-prediction-tool

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Train Model

```bash
# Collect latest season data
python collect_2025_26.py

# Train combined model
python train_combined.py

# Or train improved model with hyperparameter tuning
python train_improved.py
```

### Run API

```bash
# Start FastAPI server
uvicorn main:app --reload

# API will be available at http://localhost:8000
```

### Make Predictions

```bash
# Using API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "game_date": "2026-02-15"
  }'
```

## 📁 Project Structure

```
nba-prediction-tool/
├── backend/
│   ├── models.py                    # Database schema
│   ├── nba_client.py               # NBA API wrapper
│   ├── nba_locations.py            # Real NBA city coordinates
│   ├── enhanced_feature_engineer.py # Feature engineering
│   ├── xgboost_model.py            # XGBoost classifier
│   ├── main.py                     # FastAPI application
│   ├── train_combined.py           # Multi-season training
│   ├── train_improved.py           # Hyperparameter tuning
│   └── requirements.txt            # Dependencies
├── frontend/                        # (Coming soon)
├── .github/
│   └── workflows/
│       └── retrain.yml             # Auto-retrain workflow
└── README.md
```

## 🎯 Features

### Engineered Features (37 total)
- **Four Factors**: eFG%, TOV%, DRB%, FT%
- **Advanced Metrics**: Net rating, pace
- **Travel Data**: Real distances, timezones, elevation
- **Rest & Fatigue**: Days off, back-to-backs
- **Form**: Last 5/10 wins
- **Context**: Home court advantage

### Real Travel Data
- Actual NBA city coordinates
- Haversine distance calculation
- Timezone shifts (up to 3 hours)
- Elevation changes (Denver: 5,280 ft)

## 📊 Model Performance

### By Confidence Level
| Confidence | Accuracy | Coverage | Use Case |
|------------|----------|----------|----------|
| All games | 64.6% | 100% | General predictions |
| 70%+ | 75.8% | 36% | Reliable picks |
| 75%+ | 80.6% | 22.5% | Best bets ⭐ |

### Top Predictors
1. Away net rating (0.075)
2. Away turnover rate (0.066)
3. Home turnover rate (0.065)
4. Home net rating (0.064)
5. Away free throw rate (0.063)

## 🔄 Continuous Updates

### Automated Retraining
The model automatically retrains weekly with new games:

```bash
# Manual retrain
python collect_2025_26.py  # Collect new games
python train_combined.py    # Retrain model
```

### GitHub Actions
Automated workflow runs weekly to:
1. Collect new 2025-26 season games
2. Retrain model with updated data
3. Commit new model to repository

## 📈 API Endpoints

### Predict Game Winner
```http
POST /predict
Content-Type: application/json

{
  "home_team": "Los Angeles Lakers",
  "away_team": "Boston Celtics",
  "game_date": "2026-02-15"
}
```

Response:
```json
{
  "prediction": "home_win",
  "confidence": 0.78,
  "home_win_probability": 0.78,
  "top_factors": [
    {"feature": "away_net_rating", "contribution": 0.15},
    {"feature": "home_rest_days", "contribution": 0.12}
  ]
}
```

### Other Endpoints
- `GET /health` - Health check
- `GET /teams/{league}` - List teams (NBA/EuroLeague)
- `GET /models/status` - Model metrics

## 🛠️ Technology Stack

- **ML**: XGBoost, scikit-learn
- **API**: FastAPI, Uvicorn
- **Data**: nba-api, pandas, numpy
- **Database**: SQLAlchemy, PostgreSQL (optional)

## 📝 Dependencies

```txt
xgboost>=3.2.0
scikit-learn>=1.8.0
fastapi>=0.128.0
uvicorn>=0.40.0
pandas>=3.0.0
numpy>=2.4.0
nba-api>=1.11.0
```

## 🎯 Roadmap

- [x] Phase 1-5: Core system
- [x] Real data integration
- [x] Multi-season training
- [ ] Phase 6: Frontend dashboard
- [ ] Phase 7: Real-time updates
- [ ] Phase 8: Production deployment
- [ ] Injury data integration
- [ ] Sentiment analysis
- [ ] Point spread predictions

## 📊 Performance History

| Model | Games | Accuracy | Notes |
|-------|-------|----------|-------|
| 2023-24 only | 1,396 | 80.4% | Best historical |
| 2025-26 only | 885 | 61.0% | Limited data |
| Combined 2024-26 | 2,286 | 64.6% | Current production |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- NBA API for game data
- XGBoost team for the ML framework
- FastAPI for the web framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for basketball analytics**
*Last updated: February 2026*
