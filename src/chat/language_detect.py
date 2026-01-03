from langdetect import detect
from src.core.logging import logger

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
        return lang
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return "unknown"
