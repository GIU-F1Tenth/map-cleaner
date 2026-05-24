"""
sam_model.py
MobileSAM segmentation using mobile_sam package directly.
"""

import cv2
import numpy as np
import urllib.request
from config import SAM_WEIGHTS, SAM_WEIGHTS_URL


def init_sam():
    try:
        from mobile_sam import sam_model_registry
    except ImportError:
        raise RuntimeError(
            "mobile_sam is not installed.\n"
            "  pip install git+https://github.com/ChaoningZhang/MobileSAM.git\n"
            "  pip install timm"
        )

    if not SAM_WEIGHTS.exists():
        print(f"[sam] Weights not found, downloading to {SAM_WEIGHTS}...")
        SAM_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAM_WEIGHTS_URL, SAM_WEIGHTS)
        print("[sam] Download complete.")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    model.eval()
    print(f"[sam] Model loaded from {SAM_WEIGHTS}")
    return model


def segment_points(
    model, image_rgb: np.ndarray, points: list[tuple[int, int]]
) -> np.ndarray:
    """
    Point-prompted segmentation with one or more foreground points.
    All points are passed together as a single prompt so SAM returns
    a mask that covers all of them.
    """
    masks, scores = segment_point_candidates(model, image_rgb, points)
    best = _select_mask_covering_points(masks, scores, points)
    return np.ascontiguousarray(best.astype(np.uint8))


def segment_point_candidates(
    model, image_rgb: np.ndarray, points: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return all SAM mask candidates for the current foreground prompt points.
    """
    from mobile_sam import SamPredictor
    import warnings

    image_rgb = np.ascontiguousarray(image_rgb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor = SamPredictor(model)
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.ones(len(points), dtype=int),
            multimask_output=True,
        )

    masks = np.ascontiguousarray((masks > 0).astype(np.uint8))
    scores = np.asarray(scores, dtype=float)
    return masks, scores


def _select_mask_covering_points(
    masks: np.ndarray,
    scores: np.ndarray,
    points: list[tuple[int, int]],
    point_radius: int = 3,
) -> np.ndarray:
    """
    Prefer SAM candidates that include every foreground prompt point.

    SAM's highest confidence mask can sometimes be a smaller partial region even
    when several positive points were provided. This selection step makes prompt
    coverage the first priority, then uses SAM's score as the tie-breaker.
    """
    masks = np.ascontiguousarray(masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    scores = np.asarray(scores, dtype=float)
    if scores.shape[0] != masks.shape[0]:
        scores = np.zeros(masks.shape[0], dtype=float)

    coverage = np.array(
        [_count_covered_points(mask, points, point_radius) for mask in masks],
        dtype=int,
    )
    required = len(points)
    covers_all = coverage == required
    if np.any(covers_all):
        candidate_indexes = np.flatnonzero(covers_all)
    else:
        max_coverage = int(np.max(coverage)) if coverage.size else 0
        candidate_indexes = np.flatnonzero(coverage == max_coverage)

    best_idx = int(candidate_indexes[np.argmax(scores[candidate_indexes])])
    best = (masks[best_idx] > 0).astype(np.uint8)
    _force_prompt_points_into_mask(best, points, point_radius)
    return best


def _count_covered_points(
    mask: np.ndarray,
    points: list[tuple[int, int]],
    radius: int,
) -> int:
    return sum(1 for point in points if _mask_contains_point(mask, point, radius))


def _mask_contains_point(
    mask: np.ndarray,
    point: tuple[int, int],
    radius: int,
) -> bool:
    h, w = mask.shape[:2]
    x, y = int(point[0]), int(point[1])
    if x < 0 or y < 0 or x >= w or y >= h:
        return False

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    return bool(np.any(mask[y0:y1, x0:x1] > 0))


def _force_prompt_points_into_mask(
    mask: np.ndarray,
    points: list[tuple[int, int]],
    radius: int,
) -> None:
    h, w = mask.shape[:2]
    for x, y in points:
        x = int(x)
        y = int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), radius, 1, thickness=-1)


def resize_mask_to(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    mask = np.ascontiguousarray(mask)
    if mask.shape == target_shape:
        return mask
    return cv2.resize(
        mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
    )
