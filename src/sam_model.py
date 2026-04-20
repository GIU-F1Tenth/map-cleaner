"""
sam_model.py
MobileSAM segmentation using mobile_sam package directly.
"""

from pathlib import Path
import cv2
import numpy as np
import urllib
from config import SAM_WEIGHTS
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
