import os
import yaml
import logging.config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

def setup_logging():
    logging_path = os.path.join(CONFIG_DIR, "logging.yaml")
    with open(logging_path, "r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

# Initialize once
setup_logging()
logger = logging.getLogger("chat-model")
