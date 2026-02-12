"""
Database schema for NBA + EuroLeague Basketball Prediction Tool
SQLAlchemy ORM models
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class LeagueEnum(enum.Enum):
    NBA = "NBA"
    EUROLEAGUE = "EuroLeague"


class TeamClusterEnum(enum.Enum):
    # NBA Clusters
    CHAMPIONSHIP_CONTENDER = "Championship Contender"
    PLAY_IN_TEAM = "Play-In Team"
    LOTTERY_BOUND = "Lottery Bound"
    YOUNG_CORE = "Young Core"
    
    # EuroLeague Clusters
    TITLE_FAVORITE = "Title Favorite"
    PLAYOFF_LOCK = "Playoff Lock"
    BUBBLE_TEAM = "Bubble Team"
    REBUILDING = "Rebuilding"


class Team(Base):
    __tablename__ = 'teams'
    
    team_id = Column(Integer, primary_key=True)
    league = Column(Enum(LeagueEnum), nullable=False)
    name = Column(String(100), nullable=False)
    city = Column(String(100))
    conference = Column(String(50))  # NBA: East/West, EuroLeague: null
    
    # 2026-specific fields
    second_apron_status = Column(Boolean, default=False)  # NBA only
    roster_continuity_score = Column(Float, default=100.0)  # 0-100
    cluster_classification = Column(Enum(TeamClusterEnum))
    
    # Relationships
    home_games = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")
    players = relationship("Player", back_populates="team")
    tanking_indicators = relationship("TankingIndicator", back_populates="team")
    sentiment_data = relationship("SentimentData", back_populates="team")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Player(Base):
    __tablename__ = 'players'
    
    player_id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    name = Column(String(100), nullable=False)
    position = Column(String(10))
    
    # Injury tracking
    injury_status = Column(String(50))  # out, doubtful, questionable, probable, healthy
    injury_impact_score = Column(Float, default=0.0)  # 0-100
    
    # Advanced metrics
    ts_added = Column(Float, default=0.0)  # True Shooting Added
    fga_added = Column(Float, default=0.0)  # Field Goals Added
    
    # Wikipedia integration
    wikipedia_bio_url = Column(String(500))
    
    # Relationships
    team = relationship("Team", back_populates="players")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Game(Base):
    __tablename__ = 'games'
    
    game_id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey('teams.team_id'))
    away_team_id = Column(Integer, ForeignKey('teams.team_id'))
    
    date = Column(DateTime, nullable=False)
    league = Column(Enum(LeagueEnum), nullable=False)
    season = Column(String(20))  # e.g., "2025-2026"
    
    # Game metrics
    pace = Column(Float)
    possessions = Column(Integer)
    
    # Rest and travel
    home_rest_days = Column(Integer)
    away_rest_days = Column(Integer)
    travel_distance = Column(Float)  # miles
    timezone_shift = Column(Integer)  # hours
    
    # Results
    result = Column(String(10))  # "home" or "away"
    home_score = Column(Integer)
    away_score = Column(Integer)
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    advanced_metrics = relationship("AdvancedMetrics", back_populates="game", uselist=False)
    predictions = relationship("Prediction", back_populates="game")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdvancedMetrics(Base):
    __tablename__ = 'advanced_metrics'
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.game_id'), unique=True)
    
    # Four Factors (home team)
    home_efg_pct = Column(Float)  # Effective FG%
    home_ts_pct = Column(Float)   # True Shooting%
    home_orb_pct = Column(Float)  # Offensive Rebound%
    home_drb_pct = Column(Float)  # Defensive Rebound%
    home_tov_rate = Column(Float) # Turnover Rate
    home_ft_rate = Column(Float)  # Free Throw Rate
    
    # Four Factors (away team)
    away_efg_pct = Column(Float)
    away_ts_pct = Column(Float)
    away_orb_pct = Column(Float)
    away_drb_pct = Column(Float)
    away_tov_rate = Column(Float)
    away_ft_rate = Column(Float)
    
    # Team ratings
    home_net_rating = Column(Float)
    home_ortg = Column(Float)  # Offensive Rating
    home_drtg = Column(Float)  # Defensive Rating
    
    away_net_rating = Column(Float)
    away_ortg = Column(Float)
    away_drtg = Column(Float)
    
    # Four Factors weighted score
    four_factors_score = Column(Float)
    
    # Relationships
    game = relationship("Game", back_populates="advanced_metrics")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = 'predictions'
    
    prediction_id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.game_id'))
    
    # Prediction results
    predicted_winner = Column(String(10))  # "home" or "away"
    win_probability = Column(Float)  # 0.0 - 1.0
    confidence_score = Column(Float)  # 0-100
    
    # Explainability
    shap_explanation = Column(JSON)  # SHAP values for each feature
    
    # Model tracking
    model_version = Column(String(50))
    prediction_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Validation
    actual_result = Column(String(10))  # "home" or "away" (filled after game)
    is_correct = Column(Boolean)
    
    # Relationships
    game = relationship("Game", back_populates="predictions")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class TankingIndicator(Base):
    """NBA-only: Tracks tanking signals"""
    __tablename__ = 'tanking_indicators'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    date = Column(DateTime, nullable=False)
    
    # Tanking signals
    star_rest_probability = Column(Float, default=0.0)  # 0.0 - 1.0
    young_lineup_minutes_pct = Column(Float, default=0.0)  # % of minutes to young players
    veteran_trade_activity = Column(Integer, default=0)  # count of trades in last 30 days
    tanking_cluster_score = Column(Float, default=0.0)  # 0-100
    
    # Relationships
    team = relationship("Team", back_populates="tanking_indicators")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class SentimentData(Base):
    """GenAI-powered sentiment analysis from news"""
    __tablename__ = 'sentiment_data'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    date = Column(DateTime, nullable=False)
    
    # Sentiment scores
    news_sentiment_score = Column(Float, default=0.0)  # -1.0 to 1.0
    trade_rumor_intensity = Column(Float, default=0.0)  # 0.0 - 1.0
    coaching_stability_index = Column(Float, default=1.0)  # 0.0 - 1.0
    
    # Source tracking
    articles_analyzed = Column(Integer, default=0)
    
    # Relationships
    team = relationship("Team", back_populates="sentiment_data")
    
    created_at = Column(DateTime, default=datetime.utcnow)
