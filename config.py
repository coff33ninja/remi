import os

class Config:
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", 10))
    LOG_FILE = os.getenv("LOG_FILE", "assistant.log")
