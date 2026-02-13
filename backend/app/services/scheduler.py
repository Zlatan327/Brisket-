import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import asyncio

from app.data.nba_client import NBAClient
from app.services.tavily_service import TavilyService
from app.data.sentiment_analyzer import SentimentAnalyzer
from app.core.database import SessionLocal
from app.models.base import Game, LeagueEnum

logger = logging.getLogger(__name__)

class SchedulerService:
    """
    Handles background scheduled tasks:
    1. Daily NBA stats update
    2. Daily News & Sentiment analysis
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.nba_client = NBAClient()
        self.tavily = TavilyService()
        self.sentiment = SentimentAnalyzer()
        
    def start(self):
        """Start the scheduler"""
        # Schedule daily update at 4:00 AM UTC (approx 11 PM ET / 8 PM PT depending on DST, good for post-game)
        self.scheduler.add_job(
            self.update_daily_data,
            CronTrigger(hour=4, minute=0),
            id="daily_update",
            replace_existing=True
        )
        self.scheduler.start()
        logger.info("Scheduler started. Daily update scheduled for 04:00 UTC.")

    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
        logger.info("Scheduler shut down.")

    async def update_daily_data(self):
        """
        Main daily job:
        1. Get results for yesterday's games
        2. Get schedule for today's games
        3. Analyze sentiment for teams playing today
        """
        logger.info("Starting daily data update...")
        
        try:
            # 1. Update Yesterday's Results (Mock logic for now as we don't have a full live DB updater yet)
            # In a real app, this would query NBA API for yesterday's scores and update 'games' table
            yesterday = datetime.now() - timedelta(days=1)
            logger.info(f"Fetching results for {yesterday.date()}...")
            # self.nba_client.update_scores(yesterday) # TODO: Implement this method in NBAClient
            
            # 2. Get Today's Schedule
            today = datetime.now()
            logger.info(f"Fetching schedule for {today.date()}...")
            games_df = self.nba_client.get_games_by_date(today)
            
            if not games_df.empty:
                logger.info(f"Found {len(games_df)} games for today.")
                
                # 3. Analyze Sentiment for Teams Playing Today
                teams_to_analyze = set()
                # Assuming games_df has columns like 'TEAM_ID', 'MATCHUP', etc.
                # Simplified: Just grab all team IDs from the DF
                if 'TEAM_ID' in games_df.columns:
                    teams_to_analyze.update(games_df['TEAM_ID'].unique())
                
                logger.info(f"Analyzing sentiment for {len(teams_to_analyze)} teams...")
                for team_id in teams_to_analyze:
                    # We need team name/abbr for Tavily
                    # self.sentiment.analyze_team_sentiment(team_name) 
                    # TODO: specific mapping team_id -> team_name needed here
                    pass 
                
            else:
                logger.info("No games scheduled for today.")
                
            logger.info("Daily update completed successfully.")
            
        except Exception as e:
            logger.error(f"Daily update failed: {e}")

# Global instance
scheduler_service = SchedulerService()
