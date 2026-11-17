import os
from dotenv import load_dotenv

# Ensure environment variables from .env override system environment
load_dotenv(override=True)
if os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

def rotate_api_key() -> str:
    """Helper to rotate API keys if a comma-separated list GEMINI_API_KEYS is provided in .env."""
    keys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    current = os.environ.get("GEMINI_API_KEY")
    try:
        next_idx = (keys.index(current) + 1) % len(keys)
    except ValueError:
        next_idx = 0
    next_key = keys[next_idx]
    os.environ["GEMINI_API_KEY"] = next_key
    os.environ["GOOGLE_API_KEY"] = next_key
    return next_key

# Primary model for ADK agents
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
