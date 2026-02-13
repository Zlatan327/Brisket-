import asyncio
import logging
from app.services.scheduler import scheduler_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scheduler_job():
    logger.info("Testing Daily Data Update Job...")
    
    # Manually trigger the job function
    await scheduler_service.update_daily_data()
    
    logger.info("Job execution finished.")

if __name__ == "__main__":
    asyncio.run(test_scheduler_job())
