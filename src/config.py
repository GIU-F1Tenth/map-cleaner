"""
config.py
Loads the .env file and exposes all project paths as resolved absolute Paths.
ROOT is always the project root (one level above src/).
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ROOT is src/ — where all the code and subdirectories live
ROOT = Path(__file__).parent.resolve()

# Load .env from src/
load_dotenv(ROOT / ".env")

# ── resolved absolute paths ───────────────────────────────────────────────────

MODELS_DIR: Path = (ROOT / os.getenv("MODELS_DIR", "models")).resolve()
SAM_WEIGHTS: Path = MODELS_DIR / os.getenv("SAM_WEIGHTS", "mobile_sam.pt")
MAPS_DIR: Path = (ROOT / os.getenv("MAPS_DIR", "maps")).resolve()
OUTPUT_DIR: Path = (ROOT / os.getenv("OUTPUT_DIR", "maps")).resolve()

# Create directories if they don't exist yet
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
