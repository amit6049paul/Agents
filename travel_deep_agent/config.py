"""
config.py
---------
Loads settings from a local .env file. This is the ONLY place that touches
os.environ, so every other module just imports from here.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads a ".env" file in the project root, if present

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "travel_memory.db")


def require_api_key() -> None:
    """Fail fast with a friendly message instead of a confusing SDK error."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "No GEMINI_API_KEY found.\n"
            "1) Copy .env.example to .env\n"
            "2) Paste your key from https://aistudio.google.com/app/apikey\n"
            "3) Re-run: python main.py"
        )
