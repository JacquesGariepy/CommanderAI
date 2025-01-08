import logging
from .config import Config
def setup_logging():
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL, logging.DEBUG),
        format="%(asctime)s - %(levelname)s - %(message)s - [%(pathname)s:%(lineno)d]",
        filename=Config.LOG_FILE,
        filemode="a",
    )
