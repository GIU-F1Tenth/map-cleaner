"""
processing.py
Derives a clean ROS occupancy grid from a binary SAM track mask.
"""

import cv2
import numpy as np


FREE     = 255
OCCUPIED = 0
UNKNOWN  = 205


def derive_occupancy_grid(track_mask: np.ndarray,
                          wall_thickness: int = 2) -> np.ndarray:
    """
    Convert a binary SAM track mask into a 3-value ROS occupancy grid:

      track interior  → FREE     (255)
      boundary ring   → OCCUPIED (0)    solid connected wall
      everything else → UNKNOWN  (205)

    The wall is built by dilating the track mask outward by wall_thickness px
    and subtracting the original, giving a clean boundary that exactly follows
    SAM's segmentation with no spikes or noise.
    """
    h, w = track_mask.shape
    out = np.full((h, w), UNKNOWN, dtype=np.uint8)
    out[track_mask == 1] = FREE

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (wall_thickness * 2 + 1, wall_thickness * 2 + 1)
    )
    dilated   = cv2.dilate(track_mask, kernel, iterations=1)
    wall_ring = (dilated == 1) & (track_mask == 0)
    out[wall_ring] = OCCUPIED

    return out


def grid_stats(grid: np.ndarray) -> dict:
    """Return percentage breakdown of free / occupied / unknown pixels."""
    total = grid.size
    return {
        "free":     round(100 * np.sum(grid == FREE)     / total, 1),
        "occupied": round(100 * np.sum(grid == OCCUPIED) / total, 1),
        "unknown":  round(100 * np.sum(grid == UNKNOWN)  / total, 1),
    }