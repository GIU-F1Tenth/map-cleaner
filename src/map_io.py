"""
map_io.py
Handles loading and saving ROS occupancy grid files (.pgm + .yaml).
"""

from pathlib import Path

import cv2
import numpy as np
import yaml


def load_map(yaml_path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load a ROS map from a .yaml file.
    Returns (greyscale image, metadata dict).
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    pgm_path = yaml_path.parent / meta["image"]
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {pgm_path}")

    return img, meta


def save_map(cleaned: np.ndarray, meta: dict, out_pgm: Path, out_yaml: Path) -> None:
    """
    Save a cleaned occupancy grid as .pgm + .yaml.
    The .yaml is a copy of the original metadata with the image filename updated.
    """
    cv2.imwrite(str(out_pgm), cleaned)

    new_meta = dict(meta)
    new_meta["image"] = out_pgm.name
    with open(out_yaml, "w") as f:
        yaml.dump(new_meta, f, default_flow_style=False)


def pgm_to_rgb(pgm: np.ndarray) -> np.ndarray:
    """Convert greyscale PGM to RGB array for SAM input."""
    return cv2.cvtColor(pgm, cv2.COLOR_GRAY2RGB)


def build_comparison_image(original: np.ndarray,
                            cleaned: np.ndarray,
                            track_mask: np.ndarray) -> np.ndarray:
    """
    Build a side-by-side BGR comparison: Original | Cleaned.
    The track interior is tinted green on the cleaned side.
    """
    orig_bgr    = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cleaned_bgr = cv2.cvtColor(cleaned,  cv2.COLOR_GRAY2BGR)

    if track_mask is not None:
        cleaned_bgr[track_mask == 1] = (180, 230, 180)

    h, w = original.shape
    gap   = np.full((h, 20, 3), 160, dtype=np.uint8)
    panel = np.hstack([orig_bgr, gap, cleaned_bgr])

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 800)
    thick = max(1, int(scale * 2))
    pad   = int(10 * scale)
    cv2.putText(panel, "Original", (pad, pad + 20),           font, scale, (0, 80, 220), thick)
    cv2.putText(panel, "Cleaned",  (w + 20 + pad, pad + 20),  font, scale, (0, 160, 60), thick)
    return panel