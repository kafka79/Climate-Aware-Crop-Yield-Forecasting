import sys
from loguru import logger

def setup_logger(log_level: str = "INFO"):
    """
    Configure loguru logger for the project.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    logger.add("logs/pipeline.log", rotation="10 MB", retention="10 days", level="DEBUG")

# Create logs directory if it doesn't exist
import os
if not os.path.exists("logs"):
    os.makedirs("logs")

setup_logger()
