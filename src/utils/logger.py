import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Configures and returns a centralized logger for the QFedX framework.
    Provides strict formatting to simplify Slurm HPC log scraping and debugging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Consistent format: [Timestamp] [Level] [Module] Message
        formatter = logging.Formatter(
            '%(asctime)s | [%(levelname)s] | %(name)s : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
