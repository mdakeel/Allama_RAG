import os
import yaml
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Load .env file
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

def load_settings():
    settings_path = os.path.join(CONFIG_DIR, "settings.yaml")
    with open(settings_path, "r") as f:
        raw = yaml.safe_load(f)

    # Resolve ${VAR} placeholders with environment values
    def resolve(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            key = value[2:-1]
            return os.getenv(key, "")
        return value

    def walk(d):
        if isinstance(d, dict):
            return {k: walk(resolve(v)) for k, v in d.items()}
        return resolve(d)

    return walk(raw)

# Export SETTINGS globally
SETTINGS = load_settings()
