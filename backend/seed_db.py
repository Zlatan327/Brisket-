from app.core.database import SessionLocal, init_db
from app.models.base import Team, LeagueEnum, TeamClusterEnum
from app.data.nba_client import NBAClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_teams():
    db = SessionLocal()
    client = NBAClient()
    
    logger.info("Fetching teams from NBA Client...")
    nba_teams = client.get_all_teams()
    
    count = 0
    for team_data in nba_teams:
        # Check if exists
        exists = db.query(Team).filter_by(team_id=team_data['id']).first()
        if exists:
            continue
            
        new_team = Team(
            team_id=team_data['id'],
            league=LeagueEnum.NBA,
            name=team_data['full_name'],
            city=team_data['city'],
            conference=None, # NBA API doesn't always give this simply, leaving null for now
            cluster_classification=TeamClusterEnum.YOUNG_CORE # Default
        )
        db.add(new_team)
        count += 1
    
    db.commit()
    logger.info(f"Seeded {count} new teams.")
    db.close()

if __name__ == "__main__":
    init_db()
    seed_teams()
