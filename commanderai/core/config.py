import os
from dotenv import load_dotenv

class Config:
    load_dotenv() 
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    LOG_FILE = os.environ.get("LOG_FILE", "logs/commanderai.log")
    
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    # LLM default model
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    MAX_RETRIES = os.environ.get("MAX_RETRIES", 2)
    MAX_TOKENS = os.environ.get("MAX_TOKENS", None)
    TEMPERATURE = os.environ.get("TEMPERATURE", 0.5)
    TIMEOUT = os.environ.get("TIMEOUT", None)
