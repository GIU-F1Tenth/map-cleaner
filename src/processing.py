"""
processing.py
Derives a clean grid from a binary SAM mask.
"""

from collections.abc import Sequence

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
    manual_walls: np.ndarray | None = None,
    manual_unknown: np.ndarray | None = None,
    manual_free: np.ndarray | None = None,
    seed_points: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    grid, _ = derive_occupancy_grid_and_mask(
        track_mask,
        original,
        wall_thickness,
        smooth_sigma,
        min_hole_area,
        manual_walls,
        manual_unknown,
        manual_free,
        seed_points,
    )
    return grid


def derive_occupancy_grid_and_mask(
    track_mask: np.ndarray,
    original: np.ndarray,
    wall_thickness: int = 2,
    smooth_sigma: float = 3.0,
    min_hole_area: int = 500,
    manual_walls: np.ndarray | None = None,
    manual_unknown: np.ndarray | None = None,
    manual_free: np.ndarray | None = None,
    seed_points: Sequence[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a binary SAM track mask into a 3-value occupancy grid:

      - track interior  → FREE     (255)
      - boundary ring   → OCCUPIED (0)
      - everything else → UNKNOWN  (205)
    """
    track_mask = np.ascontiguousarray((track_mask > 0).astype(np.uint8))
    original_gray = _as_gray(original)
    manual_wall_mask = _as_mask(manual_walls, original_gray.shape)
    manual_unknown_mask = _as_mask(manual_unknown, original_gray.shape)
    manual_free_mask = _as_mask(manual_free, original_gray.shape)
    manual_helper_mask = (
        (manual_wall_mask == 1)
        & (manual_unknown_mask == 0)
        & (manual_free_mask == 0)
        & (original_gray > 64)
    ).astype(np.uint8)
    manual_remove_mask = ((manual_helper_mask == 1) | (manual_free_mask == 1)).astype(
        np.uint8
    )
    source_gray = original_gray.copy()
    source_gray[manual_unknown_mask == 1] = UNKNOWN
    source_gray[manual_free_mask == 1] = FREE
    source_gray[manual_wall_mask == 1] = OCCUPIED

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

    clip_barrier = _border_clip_barrier(
        source_gray, smooth_mask, min_hole_area, manual_wall_mask
    )
    cleanup_barrier = _temporary_cleanup_barrier(
        source_gray, smooth_mask, wall_thickness, min_hole_area, manual_wall_mask
    )
    outside = _outside_reachable_from_edges(cleanup_barrier)
    inside_mask = ((smooth_mask == 1) & (outside == 0)).astype(np.uint8)
    fitted_mask = _fit_mask_to_borders(
        smooth_mask, clip_barrier, wall_thickness, seed_points
    )
    if np.count_nonzero(fitted_mask) > 0:
        if np.count_nonzero(inside_mask) > 0:
            merged_mask = ((inside_mask == 1) & (fitted_mask == 1)).astype(np.uint8)
            merged_count = np.count_nonzero(merged_mask)
            fitted_count = np.count_nonzero(fitted_mask)
            min_merged_count = max(10, int(fitted_count * 0.25))
            inside_mask = (
                merged_mask if merged_count >= min_merged_count else fitted_mask
            )
        else:
            inside_mask = fitted_mask

    inside_mask = _keep_seeded_or_largest_component(inside_mask, seed_points)

    preserved_structures = _final_preserved_structures(
        original_gray, inside_mask, min_hole_area, manual_remove_mask
    )
    final_free = ((inside_mask == 1) & (preserved_structures == 0)).astype(np.uint8)
    final_free[manual_free_mask == 1] = 1
    final_free[manual_helper_mask == 1] = 0
    final_free = _keep_seeded_or_largest_component(final_free, seed_points)
    helper_free_mask = _helper_free_fill_mask(
        manual_helper_mask, final_free, inside_mask
    )
    helper_clearance = _helper_output_clearance(manual_helper_mask, wall_thickness)
    helper_fill_zone = (
        (helper_clearance == 1) & ((inside_mask == 1) | (helper_free_mask == 1))
    ).astype(np.uint8)

    # wall boundary ring
    k_size = wall_thickness * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(final_free, kernel, iterations=1)
    wall = ((dilated == 1) & (final_free == 0)).astype(np.uint8)
    preserved_clearance = cv2.dilate(preserved_structures, kernel, iterations=1)
    wall[(preserved_clearance == 1) & (preserved_structures == 0)] = 0
    wall[helper_clearance == 1] = 0
    preserved_structures[helper_clearance == 1] = 0

    # build grid
    h, w = final_free.shape
    out = np.full((h, w), UNKNOWN, dtype=np.uint8)
    out[final_free == 1] = FREE
    out[wall == 1] = OCCUPIED
    out[(preserved_structures == 1) & (original_gray <= 64)] = OCCUPIED
    out[helper_clearance == 1] = UNKNOWN
    out[helper_fill_zone == 1] = FREE
    out[manual_unknown_mask == 1] = UNKNOWN
    out[(manual_free_mask == 1) & (final_free == 1)] = FREE

    return out, inside_mask


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _as_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8))
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _border_clip_barrier(
    original: np.ndarray,
    candidate_mask: np.ndarray,
    min_area: int,
    manual_walls: np.ndarray,
) -> np.ndarray:
    barrier = _significant_original_structures(original, candidate_mask, min_area)
    barrier[manual_walls == 1] = 1
    return (barrier == 1).astype(np.uint8)


def _fit_mask_to_borders(
    candidate_mask: np.ndarray,
    border_mask: np.ndarray,
    wall_thickness: int,
    seed_points: Sequence[tuple[int, int]] | None,
) -> np.ndarray:
    if np.count_nonzero(candidate_mask) == 0:
        return candidate_mask.copy()

    snap_radius = max(3, min(14, wall_thickness * 3 + 4))
    snap_size = snap_radius * 2 + 1
    snap_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (snap_size, snap_size)
    )
    snapped_borders = cv2.dilate(border_mask, snap_kernel, iterations=1)
    clipped_mask = ((candidate_mask == 1) & (snapped_borders == 0)).astype(np.uint8)

    return _keep_seeded_or_largest_component(clipped_mask, seed_points)


def _keep_seeded_or_largest_component(
    mask: np.ndarray,
    seed_points: Sequence[tuple[int, int]] | None,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask.copy()

    h, w = mask.shape
    seed_labels: list[int] = []
    if seed_points:
        for x, y in seed_points:
            x = int(x)
            y = int(y)
            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            label = int(labels[y, x])
            if label > 0:
                seed_labels.append(label)
                continue

            radius = 12
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            y0 = max(0, y - radius)
            y1 = min(h, y + radius + 1)
            nearby = labels[y0:y1, x0:x1].ravel()
            nearby = nearby[nearby > 0]
            if nearby.size:
                counts = np.bincount(nearby)
                seed_labels.append(int(np.argmax(counts)))

    if seed_labels:
        counts = np.bincount(seed_labels, minlength=num_labels)
        best_count = np.max(counts[1:])
        candidates = np.flatnonzero(counts == best_count)
        candidates = candidates[candidates > 0]
        selected_label = max(
            candidates, key=lambda label: stats[label, cv2.CC_STAT_AREA]
        )
    else:
        selected_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    keep = np.zeros_like(mask, dtype=np.uint8)
    keep[labels == selected_label] = 1
    return keep


def _helper_free_fill_mask(
    helper_mask: np.ndarray,
    final_free: np.ndarray,
    inside_mask: np.ndarray,
) -> np.ndarray:
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        helper_mask, connectivity=8
    )
    keep = np.zeros_like(helper_mask, dtype=np.uint8)
    if num_labels <= 1:
        return keep

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    outside = inside_mask == 0
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        border = (cv2.dilate(component, kernel, iterations=1) == 1) & (
            component == 0
        )
        touches_free = np.any(border & (final_free == 1))
        touches_outside = np.any(border & outside)
        if touches_free and not touches_outside:
            keep[component == 1] = 1

    return keep


def _helper_output_clearance(helper_mask: np.ndarray, wall_thickness: int) -> np.ndarray:
    if np.count_nonzero(helper_mask) == 0:
        return helper_mask.copy()

    radius = max(2, min(12, wall_thickness * 2 + 3))
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(helper_mask, kernel, iterations=1).astype(np.uint8)


def _temporary_cleanup_barrier(
    original: np.ndarray,
    candidate_mask: np.ndarray,
    wall_thickness: int,
    min_area: int,
    manual_walls: np.ndarray,
) -> np.ndarray:
    real_structures = _significant_original_structures(
        original, candidate_mask, min_area
    )
    real_structures[manual_walls == 1] = 1
    close_size = max(45, wall_thickness * 10 + 1, int(np.sqrt(max(1, min_area)) * 2))
    close_size = min(close_size, 91)
    if close_size % 2 == 0:
        close_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    closed = cv2.morphologyEx(real_structures, cv2.MORPH_CLOSE, kernel)
    return (closed == 1).astype(np.uint8)


def _significant_original_structures(
    original: np.ndarray,
    candidate_mask: np.ndarray,
    min_area: int,
    free_threshold: int = 250,
) -> np.ndarray:
    preserve_area = max(80, min(500, min_area // 4))
    nearby_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    nearby_candidate = cv2.dilate(candidate_mask, nearby_kernel, iterations=1)
    non_free = ((original < free_threshold) & (nearby_candidate == 1)).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        non_free, connectivity=8
    )

    keep = np.zeros_like(candidate_mask, dtype=np.uint8)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < preserve_area:
            continue
        keep[labels == label] = 1

    return keep


def _final_preserved_structures(
    original: np.ndarray,
    candidate_mask: np.ndarray,
    min_area: int,
    manual_helper: np.ndarray,
    free_threshold: int = 250,
) -> np.ndarray:
    structures = _significant_original_structures(
        original, candidate_mask, min_area, free_threshold
    )
    structures[manual_helper == 1] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        structures, connectivity=8
    )

    keep = np.zeros_like(candidate_mask, dtype=np.uint8)
    exterior = _exterior_touch_mask(candidate_mask, radius=3)

    for label in range(1, num_labels):
        component = labels == label
        if np.any(component & (exterior == 1)):
            continue
        if stats[label, cv2.CC_STAT_AREA] < 1:
            continue
        keep[component] = 1

    return keep


def _exterior_touch_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate((mask == 0).astype(np.uint8), kernel, iterations=1)


def _outside_reachable_from_edges(cleanup_barrier: np.ndarray) -> np.ndarray:
    open_area = (cleanup_barrier == 0).astype(np.uint8)
    h, w = open_area.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for x in range(w):
        if open_area[0, x] == 1:
            cv2.floodFill(open_area, flood_mask, (x, 0), 2)
        if open_area[h - 1, x] == 1:
            cv2.floodFill(open_area, flood_mask, (x, h - 1), 2)

    for y in range(h):
        if open_area[y, 0] == 1:
            cv2.floodFill(open_area, flood_mask, (0, y), 2)
        if open_area[y, w - 1] == 1:
            cv2.floodFill(open_area, flood_mask, (w - 1, y), 2)

    return (open_area == 2).astype(np.uint8)


def grid_stats(grid: np.ndarray) -> dict:
    total = grid.size
    return {
        "free": round(100 * np.sum(grid == FREE) / total, 1),
        "occupied": round(100 * np.sum(grid == OCCUPIED) / total, 1),
        "unknown": round(100 * np.sum(grid == UNKNOWN) / total, 1),
    }
