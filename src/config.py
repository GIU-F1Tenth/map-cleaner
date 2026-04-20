"""
config.py
Exposes all project paths as resolved absolute Paths.
"""

from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# ── resolved absolute paths ───────────────────────────────────────────────────

MODELS_DIR: Path = (ROOT / "models").resolve()
SAM_WEIGHTS: Path = MODELS_DIR / "mobile_sam.pt"
MAPS_DIR: Path = (ROOT / "maps").resolve()
OUTPUT_DIR: Path = (ROOT / "maps").resolve()
SAM_WEIGHTS_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)
# Create directories if they don't exist yet
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
