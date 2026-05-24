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

# Startup default. True: show Original | SAM Mask | Cleaned. False: show only cleaned.
SHOW_COMPARISON_PREVIEW: bool = False

# Manual painting brush range in pixels.
BRUSH_SIZE_MIN: int = 1
BRUSH_SIZE_MAX: int = 15
BRUSH_SIZE_DEFAULT: int = 3

# Generated wall thickness range in pixels.
WALL_THICKNESS_MIN: int = 1
WALL_THICKNESS_MAX: int = 8
WALL_THICKNESS_DEFAULT: int = 2

# Generated wall smoothing range. Higher values produce smoother wall outlines.
WALL_SMOOTHING_MIN: float = 0.0
WALL_SMOOTHING_MAX: float = 8.0
WALL_SMOOTHING_STEPS: int = 15
WALL_SMOOTHING_DEFAULT: float = 0.0

# Create directories if they don't exist yet
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
