import re
from langdetect import detect, detect_langs
from src.core.logging import logger


ROMAN_URDU_WORDS = {
    "hai", "hain", "ho", "kya", "ka", "ki", "ke", "tum", "aap",
    "mein", "mera", "meri", "hum", "kaise", "kyun",
    "nahi", "haan", "tha", "thi", "raha", "rahi",
    "wale", "walon", "kon", "kaun", "iman", "imaan", "tha", "gai"
}


def _has_arabic_script(text: str) -> bool:
    """Check for Urdu/Arabic script (ا ب پ ت ث ج etc)"""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _has_devanagari_script(text: str) -> bool:
    """Check for Hindi/Devanagari script (अ आ इ ई उ etc)"""
    return bool(re.search(r"[\u0900-\u097F]", text))


def _is_roman_urdu(text: str) -> bool:
    """Detect Roman Urdu (Hindustani written in Latin letters)"""
    words = text.lower().split()
    # Need at least 2 roman urdu indicator words
    matches = sum(1 for w in words if w in ROMAN_URDU_WORDS)
    return matches >= 2


def detect_language(text: str) -> str:
    """Detect language: 'ur' (Urdu), 'hi' (Hindi), 'en' (English), 'roman' (Roman Urdu).
    
    Priority order:
    1. Script detection (Arabic = Urdu, Devanagari = Hindi)
    2. Roman-Urdu indicators
    3. langdetect library
    4. Default to English
    """
    if not text or not text.strip():
        logger.info("Empty text, defaulting to 'en'")
        return "en"

    # 1. Script-based detection (fastest)
    if _has_devanagari_script(text):
        logger.info(f"Language: Hindi (Devanagari script detected)")
        return "hi"
    
    if _has_arabic_script(text):
        logger.info(f"Language: Urdu (Arabic script detected)")
        return "ur"

    # 2. Roman-Urdu heuristics
    if _is_roman_urdu(text):
        logger.info(f"Language: Roman Urdu (roman words detected)")
        return "roman"

    # 3. langdetect library
    try:
        detected = detect(text)
        logger.info(f"langdetect result: {detected}")
        
        if detected.startswith("en"):
            return "en"
        elif detected.startswith("hi"):
            return "hi"
        elif detected.startswith("ur"):
            return "ur"
        else:
            # Default to English for other languages
            return "en"
    except Exception as e:
        logger.warning(f"langdetect failed: {e}, defaulting to 'en'")
        return "en"
