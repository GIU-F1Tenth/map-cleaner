"""
map_io.py
Handles loading and saving map image files plus .yaml metadata.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml


def load_map(yaml_path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load a map from a .yaml file.
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


def save_map(cleaned: np.ndarray, out_image: Path) -> None:
    """
    Save the cleaned grayscale occupancy grid image.
    """
    cv2.imwrite(str(out_image), cleaned)


def pgm_to_rgb(pgm: np.ndarray) -> np.ndarray:
    """Convert greyscale PGM to RGB array for SAM input."""
    if pgm.ndim == 2:
        rgb = cv2.cvtColor(pgm, cv2.COLOR_GRAY2RGB)
    else:
        rgb = pgm
    return np.ascontiguousarray(rgb)


def build_comparison_image(
    original: np.ndarray,
    cleaned: np.ndarray,
    track_mask: np.ndarray,
    show_comparison: bool = True,
) -> np.ndarray:
    """
    Build a side-by-side BGR comparison: Original | SAM Mask | Cleaned.
    Free cells are tinted green on the cleaned side while occupied and unknown
    cells stay visible.
    """
    # ensure both are grayscale before converting to BGR
    if original.ndim == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if cleaned.ndim == 3:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    mask_bgr = np.full_like(orig_bgr, 205)

    cleaned_bgr[cleaned == 255] = (180, 230, 180)
    h, w = original.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 800)
    thick = max(1, int(scale * 2))
    pad = int(10 * scale)

    if not show_comparison:
        cv2.putText(
            cleaned_bgr,
            "Cleaned",
            (pad, pad + 20),
            font,
            scale,
            (0, 160, 60),
            thick,
        )
        return cleaned_bgr

    if track_mask is not None:
        if track_mask.shape != original.shape:
            track_mask = cv2.resize(
                track_mask,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        track_mask = (track_mask > 0).astype(np.uint8)
        mask_bgr[track_mask == 1] = (255, 255, 255)
        contours, _ = cv2.findContours(
            track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(mask_bgr, contours, -1, (0, 0, 0), 1)

    gap = np.full((h, 20, 3), 160, dtype=np.uint8)
    panel = np.hstack([orig_bgr, gap, mask_bgr, gap, cleaned_bgr])

    cv2.putText(panel, "Original", (pad, pad + 20), font, scale, (0, 80, 220), thick)
    cv2.putText(
        panel,
        "SAM Mask",
        (w + 20 + pad, pad + 20),
        font,
        scale,
        (220, 80, 0),
        thick,
    )
    cv2.putText(
        panel,
        "Cleaned",
        (2 * (w + 20) + pad, pad + 20),
        font,
        scale,
        (0, 160, 60),
        thick,
    )
    return panel
