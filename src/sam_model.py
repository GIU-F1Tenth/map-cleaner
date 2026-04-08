"""
sam_model.py
MobileSAM segmentation using mobile_sam package directly.
"""

from pathlib import Path
import cv2
import numpy as np
from config import SAM_WEIGHTS


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
        raise RuntimeError(
            f"Weights not found: {SAM_WEIGHTS}\n"
            "Please place mobile_sam.pt in the models/ folder."
        )
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
    best = masks[int(np.argmax(scores))]
    return np.ascontiguousarray(best.astype(np.uint8))


def resize_mask_to(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    mask = np.ascontiguousarray(mask)
    if mask.shape == target_shape:
        return mask
    return cv2.resize(
        mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
    )
