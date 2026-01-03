import os
import json
from src.core.logging import logger

def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON {path}: {e}")
        return {}

def save_json(path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON {path}: {e}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
