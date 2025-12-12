import logging
import sys
from pathlib import Path
from src.config import LoggingConfig

def setup_logger(config: LoggingConfig):
    """Configures the root logger based on the provided configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.save_to_file:
        log_path = Path(config.log_file)
        if log_path.parent != Path('.'):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.log_file))

    logging.basicConfig(
        level=config.level.upper(),
        format=config.format,
        handlers=handlers,
        force=True
    )
    # Silence verbose libraries
    logging.getLogger("tensorflow").setLevel(logging.ERROR)