"""
service.py — Plum'ID model microservice
=========================================

FastAPI wrapper around the PyTorch classifier downloaded from HuggingFace.

**No image preprocessing**. The image bytes are decoded, resized to 224×224,
and passed directly to the model. The classifier handles ImageNet
normalisation internally.

Why no preprocessing? The watershed-based segmentation in the previous
version was over-splitting single feathers into multiple candidates, which
the classifier then labelled as feathers, producing false-positive
TOO_MANY_FEATHERS warnings. The DenseNet model is robust enough to handle
raw photographs without a custom segmentation step.

Endpoints
---------
* GET  /health        — fast liveness probe.
* GET  /model/status  — introspection on the loaded classifier.
* POST /predict       — multipart upload; returns the species prediction.
* POST /augment       — batch dataset augmentation job (offline; CLI-style).

Run
---
    uvicorn service:app --host 0.0.0.0 --port 8001

Environment variables (see also `inference.classifier`)
-------------------------------------------------------
    LOG_LEVEL              INFO | DEBUG | WARNING | ERROR
    HF_REPO_ID             Required. e.g. "Azerty112/Plum_ID_V1"
    HF_MODEL_FILENAME      Optional. Specific .pth file in the repo.
    HF_REVISION            Optional. Git ref, defaults to "main".
    HF_TOKEN               Optional. Required for private repos.
    PRELOAD_MODEL          0 | 1 (default 1).
    MODEL_INPUT_SIZE       Optional. Square edge size in pixels. Default 224.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Any, Dict

import cv2 as cv
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from inference import ClassifierError, get_classifier

# --------------------------------------------------------------------- #
# Logging                                                                #
# --------------------------------------------------------------------- #

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("plumid-model")

# --------------------------------------------------------------------- #
# App                                                                    #
# --------------------------------------------------------------------- #

app = FastAPI(
    title="Plum'ID — Model service",
    version="2.0.0",
    description=(
        "Simple image classification microservice. Decodes the upload, "
        "resizes to 224×224, runs the DenseNet classifier downloaded from "
        "HuggingFace. No fancy segmentation — just the model."
    ),
)

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10_000_000))


# --------------------------------------------------------------------- #
# Startup                                                                #
# --------------------------------------------------------------------- #

def _preload_model_in_background() -> None:
    def _runner() -> None:
        try:
            log.info("Preloading classifier in background…")
            get_classifier().ensure_loaded()
            log.info("Classifier preload OK")
        except Exception as exc:  # noqa: BLE001
            log.error("Classifier preload FAILED: %s", exc)
    threading.Thread(target=_runner, name="classifier-preload", daemon=True).start()


@app.on_event("startup")
def _on_startup() -> None:
    if os.environ.get("PRELOAD_MODEL", "1") not in {"0", "false", "False"}:
        _preload_model_in_background()
    else:
        log.info("PRELOAD_MODEL disabled; model will load on first /predict")


# --------------------------------------------------------------------- #
# Upload helpers                                                         #
# --------------------------------------------------------------------- #

async def _read_upload(file: UploadFile) -> bytes:
    if file is None:
        raise HTTPException(status_code=400, detail="Missing 'file' field")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES} bytes)",
        )
    return content


def _decode_to_rgb_224(image_bytes: bytes, target_size: int = 224) -> np.ndarray:
    """
    Decode an arbitrary image (JPEG/PNG/HEIC-as-JPEG/…) and return a
    `target_size`×`target_size` RGB ndarray (uint8). Uses Pillow first
    for broad format support, falls back to OpenCV.
    """
    # First try Pillow — handles JPEG/PNG/WEBP/GIF + EXIF orientation.
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im = im.convert("RGB")
            # Honor EXIF orientation (most phone cameras embed it).
            try:
                from PIL import ImageOps
                im = ImageOps.exif_transpose(im)
            except Exception:  # noqa: BLE001
                pass
            im = im.resize((target_size, target_size), Image.LANCZOS)
            return np.array(im, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        log.debug("Pillow decode failed (%s); falling back to OpenCV", exc)

    # Fallback to OpenCV.
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv.imdecode(buf, cv.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    rgb = cv.resize(rgb, (target_size, target_size), interpolation=cv.INTER_LANCZOS4)
    return rgb


# --------------------------------------------------------------------- #
# Routes                                                                 #
# --------------------------------------------------------------------- #


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "plumid-model", "version": app.version}


@app.get("/model/status")
def model_status() -> Dict[str, Any]:
    return asdict(get_classifier().status)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict the bird species from a feather image.

    Decode → resize 224×224 → run DenseNet → return prediction.

    Returns 200 with `{ok: true, species_id, species_name, model_class,
    confidence, top_k, ...}` on success. Returns 503 if the classifier
    is still loading. Returns 400/500 for bad input or unexpected errors.
    """
    content = await _read_upload(file)
    t0 = time.perf_counter()

    # 1. Decode + resize
    try:
        clf = get_classifier()
        target_size = clf.status.input_size or int(
            os.environ.get("MODEL_INPUT_SIZE", "224")
        )
        rgb = _decode_to_rgb_224(content, target_size=target_size)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Image decoding failed")
        raise HTTPException(
            status_code=400, detail=f"Cannot decode image: {exc}"
        ) from exc

    # 2. Inference (in a worker thread; first call is cold)
    try:
        prediction = await asyncio.to_thread(clf.predict, rgb)
    except ClassifierError as exc:
        log.error("Classifier not available: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Classifier not available: {exc}. "
                "Check /model/status for details."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {exc}"
        ) from exc

    dt_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "predict: filename=%s bytes=%d species_id=%d species=%s confidence=%.2f latency_ms=%s",
        file.filename,
        len(content),
        prediction["species_id"],
        prediction["species_name"],
        prediction["confidence"],
        dt_ms,
    )

    return JSONResponse(
        {
            "ok": True,
            **prediction,
            "latency_ms": dt_ms,
        }
    )


# --------------------------------------------------------------------- #
# Offline / batch — kept for parity with the CLI                         #
# --------------------------------------------------------------------- #

@app.post("/augment")
async def augment_endpoint(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    limit: int = Form(500),
):
    """Batch dataset augmentation job (offline; volume-mounted dirs)."""
    if not os.path.isdir(input_dir):
        raise HTTPException(status_code=400, detail=f"input_dir not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    from augmentation.augmentation import DatasetGenerator
    from augmentation_config import (
        DEFAULT_OPERATIONS,
        DEFAULT_ROTATE_PROBABILITY,
        DEFAULT_ROTATE_MAX_LEFT_DEGREE,
        DEFAULT_ROTATE_MAX_RIGHT_DEGREE,
        DEFAULT_BLUR_PROBABILITY,
        DEFAULT_RANDOM_NOISE_PROBABILITY,
        DEFAULT_HORIZONTAL_FLIP_PROBABILITY,
        DEFAULT_VERTICAL_FLIP_PROBABILITY,
    )

    gen = DatasetGenerator(
        folder_path=input_dir,
        num_files=limit,
        save_to_disk=True,
        folder_destination=output_dir,
    )
    if "rotate" in DEFAULT_OPERATIONS:
        gen.rotate(
            probability=DEFAULT_ROTATE_PROBABILITY,
            max_left_degree=DEFAULT_ROTATE_MAX_LEFT_DEGREE,
            max_right_degree=DEFAULT_ROTATE_MAX_RIGHT_DEGREE,
        )
    if "blur" in DEFAULT_OPERATIONS:
        gen.blur(probability=DEFAULT_BLUR_PROBABILITY)
    if "random_noise" in DEFAULT_OPERATIONS:
        gen.random_noise(probability=DEFAULT_RANDOM_NOISE_PROBABILITY)
    if "horizontal_flip" in DEFAULT_OPERATIONS:
        gen.horizontal_flip(probability=DEFAULT_HORIZONTAL_FLIP_PROBABILITY)
    if "vertical_flip" in DEFAULT_OPERATIONS:
        gen.vertical_flip(probability=DEFAULT_VERTICAL_FLIP_PROBABILITY)

    t0 = time.perf_counter()
    try:
        gen.execute()
    except Exception as exc:  # noqa: BLE001
        log.exception("Augmentation failed")
        raise HTTPException(status_code=500, detail=f"Augmentation failed: {exc}") from exc
    dt_s = round(time.perf_counter() - t0, 2)
    return {
        "ok": True,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "limit": limit,
        "elapsed_seconds": dt_s,
    }
