"""
sam_model.py
MobileSAM model initialisation and segmentation via ultralytics.
"""

import numpy as np
import cv2


def init_sam(model_path: str = "mobile_sam.pt"):
    """
    Load MobileSAM via ultralytics.
    Weights are auto-downloaded on first run if not present.
    """
    try:
        import ultralytics
        model = ultralytics.SAM(model_path)
        return model
    except ImportError:
        raise RuntimeError("ultralytics is not installed.\n  pip install ultralytics")
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM model: {e}")


def segment_auto(model, image_rgb: np.ndarray,
                 min_area_frac: float = 0.05,
                 max_area_frac: float = 0.70) -> np.ndarray:
    """
    Run SAM automatic mask generation and return the mask that best represents
    the track interior — the largest mask whose area fraction is between
    min_area_frac and max_area_frac of the total image.
    """
    results = model.predict(image_rgb, retina_masks=True, imgsz=1024)

    if results[0].masks is None:
        raise ValueError("SAM returned no masks. Try using a prompt point instead.")

    masks_np = results[0].masks.data.cpu().numpy()
    total_px = image_rgb.shape[0] * image_rgb.shape[1]

    best_mask, best_area = None, 0
    for mask in masks_np:
        area = int(mask.sum())
        frac = area / total_px
        if min_area_frac <= frac <= max_area_frac and area > best_area:
            best_mask = mask
            best_area = area

    if best_mask is None:
        raise ValueError(
            "No qualifying mask found (area between "
            f"{min_area_frac*100:.0f}%–{max_area_frac*100:.0f}% of image).\n"
            "Try using a prompt point instead."
        )

    return best_mask.astype(np.uint8)


def segment_point(model, image_rgb: np.ndarray,
                  point: tuple[int, int]) -> np.ndarray:
    """
    Run SAM with a single foreground point prompt inside the track.
    Returns the largest mask returned for that point.
    """
    results = model.predict(
        image_rgb,
        points=[list(point)],
        labels=[1],
        retina_masks=True,
        imgsz=1024,
    )

    if results[0].masks is None:
        raise ValueError(f"SAM returned no masks for point {point}.")

    masks_np = results[0].masks.data.cpu().numpy()
    best_mask = max(masks_np, key=lambda m: m.sum())
    return best_mask.astype(np.uint8)


def resize_mask_to(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize mask to (h, w) using nearest-neighbour to preserve binary values."""
    if mask.shape == target_shape:
        return mask
    return cv2.resize(
        mask, (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )