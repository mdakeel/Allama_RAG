import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def data_path(*parts):
    return os.path.join(BASE_DIR, "data", *parts)

def log_path(*parts):
    return os.path.join(BASE_DIR, "logs", *parts)
