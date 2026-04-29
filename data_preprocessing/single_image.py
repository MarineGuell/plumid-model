"""
data_preprocessing/single_image.py
-----------------------------------

In-memory single-image counterpart to `datapreprocessing.py`.

The original module is folder-based and writes intermediate stages to
sub-directories under `preprocessed/`. For the FastAPI service we need
to run the *same* operations on a single uploaded image and return the
final tensor / image without touching disk. This module reproduces the
exact pipeline: segmentation → denoise → contrast (CLAHE on L) →
padding (square + 224×224 resize).
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from PIL import Image


# ------------------------------------------------------------------ #
# 1. Segmentation (mirrors call_sam3_replicate)                       #
# ------------------------------------------------------------------ #

def _segment_largest_object(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Pull the largest plausible feather-shaped object out of an image.

    Returns a cropped BGR image (object on black background) or None
    when no acceptable object is found. Logic adapted from
    `call_sam3_replicate` + the watershed/contour fallback.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    binary = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 5
    )

    # Watershed-style separation
    dist_transform = cv.distanceTransform(binary, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    _, sure_fg = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv.dilate(binary, kernel, iterations=3)
    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB), markers)

    min_area = (h * w) * 0.002
    max_area = (h * w) * 0.95

    masks = []
    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue
        mask = (markers == marker_id).astype(np.uint8)
        area = int(mask.sum())
        if min_area <= area <= max_area:
            masks.append((area, mask))

    # Fallback: plain contour detection
    if not masks:
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv.contourArea(c)
            if min_area <= area <= max_area:
                m = np.zeros((h, w), dtype=np.uint8)
                cv.drawContours(m, [c], 0, 1, -1)
                masks.append((int(area), m))

    if not masks:
        return None

    # Keep the biggest accepted blob
    masks.sort(key=lambda t: t[0], reverse=True)
    _, mask = masks[0]

    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    mask_3ch = np.stack([mask] * 3, axis=-1)
    segmented = image_bgr * mask_3ch
    cropped = segmented[y1:y2 + 1, x1:x2 + 1]
    return cropped


# ------------------------------------------------------------------ #
# 2. Denoise / 3. Contrast — taken straight from datapreprocessing    #
# ------------------------------------------------------------------ #

def _denoise(image_bgr: np.ndarray) -> np.ndarray:
    return cv.fastNlMeansDenoisingColored(image_bgr, None, 10, 10, 7, 21)


def _enhance_contrast(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv.cvtColor(cv.merge((cl, a, b)), cv.COLOR_LAB2BGR)


# ------------------------------------------------------------------ #
# 4. Padding + 224×224 resize (same as image_padding)                 #
# ------------------------------------------------------------------ #

def _square_pad_and_resize(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    x = (side - w) // 2
    y = (side - h) // 2
    canvas[y:y + h, x:x + w] = image_bgr
    return cv.resize(canvas, (size, size), interpolation=cv.INTER_LANCZOS4)


# ------------------------------------------------------------------ #
# Public entrypoint                                                   #
# ------------------------------------------------------------------ #

def preprocess_bytes(
    image_bytes: bytes,
    *,
    target_size: int = 224,
    skip_segmentation: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Run the full preprocessing chain on raw image bytes.

    Args:
        image_bytes: raw image bytes (PNG/JPEG/…).
        target_size: output side length (default 224, matches model input).
        skip_segmentation: when True, skip segmentation and feed the full
            image into the rest of the pipeline. Useful when the upload
            is already a clean crop.

    Returns:
        (rgb_array, info) where:
            - rgb_array is a uint8 ndarray of shape (target_size, target_size, 3)
              in RGB order (suitable for PIL / matplotlib / model inference).
            - info is a dict with metadata about the operation.

    Raises:
        ValueError: if the bytes can't be decoded.
    """
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv.imdecode(nparr, cv.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Cannot decode image bytes (unsupported format?)")

    info: dict = {
        "input_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
        "segmented": False,
        "target_size": target_size,
    }

    work = image_bgr
    if not skip_segmentation:
        seg = _segment_largest_object(image_bgr)
        if seg is not None and seg.size > 0:
            work = seg
            info["segmented"] = True
            info["bbox_size"] = [int(seg.shape[1]), int(seg.shape[0])]
        else:
            info["segmented"] = False
            info["fallback"] = "no-object-detected; using full image"

    work = _denoise(work)
    work = _enhance_contrast(work)
    work = _square_pad_and_resize(work, size=target_size)

    rgb = cv.cvtColor(work, cv.COLOR_BGR2RGB)
    return rgb, info


def encode_png(rgb_array: np.ndarray) -> bytes:
    """Encode an RGB ndarray to PNG bytes."""
    pil = Image.fromarray(rgb_array, mode="RGB")
    import io as _io
    buf = _io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
