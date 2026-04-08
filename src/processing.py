"""
processing.py
Derives a clean grid from a binary SAM mask.
"""

import cv2
import numpy as np


FREE = 255
OCCUPIED = 0
UNKNOWN = 205


def derive_occupancy_grid(
    track_mask: np.ndarray,
    original: np.ndarray,
    wall_thickness: int = 2,
    smooth_sigma: float = 3.0,
    min_hole_area: int = 500,
) -> np.ndarray:
    """
    Convert a binary SAM track mask into a 3-value occupancy grid:

      - track interior  → FREE     (255)
      - boundary ring   → OCCUPIED (0)
      - everything else → UNKNOWN  (205)
    """
    track_mask = np.ascontiguousarray(track_mask.astype(np.uint8))

    # remove scattered noise dots inside the mask
    inverted = (1 - track_mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    cleaned_mask = track_mask.copy()
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_hole_area:
            cleaned_mask[labels == label] = 1

    # keep only the largest free region
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        cleaned_mask, connectivity=8
    )
    if num_labels2 > 1:
        largest = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        cleaned_mask = (labels2 == largest).astype(np.uint8)

    # smooth the boundary
    blurred = cv2.GaussianBlur(
        cleaned_mask.astype(np.float32),
        (0, 0),
        sigmaX=smooth_sigma,
        sigmaY=smooth_sigma,
    )
    smooth_mask = (blurred > 0.5).astype(np.uint8)

    # wall boundary ring
    k_size = wall_thickness * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(smooth_mask, kernel, iterations=1)
    wall = ((dilated == 1) & (smooth_mask == 0)).astype(np.uint8)

    # build grid
    h, w = smooth_mask.shape
    out = np.full((h, w), UNKNOWN, dtype=np.uint8)
    out[smooth_mask == 1] = FREE
    out[wall == 1] = OCCUPIED

    return out


def grid_stats(grid: np.ndarray) -> dict:
    total = grid.size
    return {
        "free": round(100 * np.sum(grid == FREE) / total, 1),
        "occupied": round(100 * np.sum(grid == OCCUPIED) / total, 1),
        "unknown": round(100 * np.sum(grid == UNKNOWN) / total, 1),
    }
