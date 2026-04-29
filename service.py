"""
service.py — Plum'ID model microservice
=========================================

FastAPI wrapper around the preprocessing + inference pipeline. Designed
to run as a long-lived container on Railway alongside the API.

Endpoints
---------
* GET  /health        — fast liveness probe (always 200 once uvicorn is up).
* GET  /model/status  — introspection: is the classifier loaded? which
                        architecture? how many classes? cache info.
* POST /preprocess    — multipart upload of an image; returns the
                        preprocessed PNG (segmented, denoised, contrast-
                        enhanced, padded to MODEL_INPUT_SIZE × …).
* POST /predict       — multipart upload; returns a JSON prediction by
                        running the trained model downloaded from
                        HuggingFace (env: HF_REPO_ID).
* POST /augment       — kept for parity with the CLI: takes a folder of
                        already-preprocessed images mounted into the
                        container and produces N augmented variants.
                        Useful for offline dataset generation jobs; not
                        meant to be called from the public API.

Run
---
    uvicorn service:app --host 0.0.0.0 --port 8001

Environment variables (see also `inference.classifier`)
-------------------------------------------------------
    LOG_LEVEL              INFO | DEBUG | WARNING | ERROR
    HF_REPO_ID             Required. e.g. "Azerty112/Plum_ID_V1"
    HF_MODEL_FILENAME      Optional. Specific .pth file inside the repo.
    HF_REVISION            Optional. Git ref, defaults to "main".
    HF_TOKEN               Optional. Required for private repos.
    PRELOAD_MODEL          0 | 1 (default 1). Load the model on startup
                           in a background thread so the first /predict
                           is fast.
    PORT                   Honoured by the container CMD.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from data_preprocessing.single_image import encode_png, preprocess_bytes
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
    version="1.1.0",
    description=(
        "Image preprocessing + inference microservice for the Plum'ID "
        "feather-identification stack. The classifier is downloaded from "
        "HuggingFace at first use (or eagerly at startup if PRELOAD_MODEL=1)."
    ),
)

# Tuning knob: maximum upload size in bytes (10 MB by default).
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10_000_000))


# --------------------------------------------------------------------- #
# Startup: optionally preload the model in a background thread so the   #
# server can answer /health immediately while the weights download.     #
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
        log.info("PRELOAD_MODEL disabled; the model will load on first /predict")


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


async def _read_upload(file: UploadFile) -> bytes:
    """Read & validate an UploadFile, enforcing the size cap."""
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


# --------------------------------------------------------------------- #
# Routes                                                                 #
# --------------------------------------------------------------------- #


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Liveness probe used by Railway / docker compose healthchecks.

    Stays 200 even while the model is still downloading — the readiness
    of the classifier is reported separately via /model/status, so a
    slow first download doesn't cause the container to be killed.
    """
    return {"status": "ok", "service": "plumid-model", "version": app.version}


@app.get("/model/status")
def model_status() -> Dict[str, Any]:
    """Introspect the loaded classifier (or its load error)."""
    return asdict(get_classifier().status)


@app.post("/preprocess")
async def preprocess_endpoint(
    file: UploadFile = File(...),
    skip_segmentation: bool = Form(False),
    target_size: int = Form(224),
):
    """
    Run the preprocessing pipeline on a single image.

    Returns
    -------
    image/png — the preprocessed image (target_size × target_size).
                Metadata about the run is exposed in response headers
                (`X-PlumID-Segmented`, `X-PlumID-Input-Size`, …).
    """
    content = await _read_upload(file)
    if target_size <= 0 or target_size > 1024:
        raise HTTPException(status_code=400, detail="target_size out of range")

    t0 = time.perf_counter()
    try:
        rgb, info = preprocess_bytes(
            content,
            target_size=target_size,
            skip_segmentation=skip_segmentation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}") from exc

    png = encode_png(rgb)
    dt_ms = round((time.perf_counter() - t0) * 1000, 1)

    headers = {
        "X-PlumID-Segmented": "1" if info.get("segmented") else "0",
        "X-PlumID-Input-Size": "x".join(map(str, info.get("input_size", []))),
        "X-PlumID-Target-Size": str(info.get("target_size", target_size)),
        "X-PlumID-Latency-Ms": str(dt_ms),
    }
    log.info(
        "preprocess: filename=%s bytes=%d segmented=%s latency_ms=%s",
        file.filename,
        len(content),
        info.get("segmented"),
        dt_ms,
    )
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict the bird species from a feather image.

    Pipeline:
        1. Decode the upload.
        2. Preprocess (segmentation + denoise + contrast + 224×224 pad).
        3. Run the classifier downloaded from HuggingFace.

    Returns a JSON response with the top species, the top-3 candidates,
    preprocessing metadata, and timing info. If the classifier failed
    to load, returns 503 with a clear error explaining what to do.
    """
    content = await _read_upload(file)

    t0 = time.perf_counter()
    try:
        clf = get_classifier()
        # The classifier wants to know its expected input size; we let it
        # handle that internally (`predict` resizes if needed). We use 224
        # here because that's also what the trained network expects.
        target_size = clf.status.input_size or int(
            os.environ.get("MODEL_INPUT_SIZE", "224")
        )
        rgb, info = preprocess_bytes(content, target_size=target_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Predict preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}") from exc

    # Inference can be slow (esp. first call). Run it on a worker
    # thread so we don't block the asyncio event loop.
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
        "predict: filename=%s bytes=%d species=%s confidence=%.3f latency_ms=%s",
        file.filename,
        len(content),
        prediction["species"],
        prediction["confidence"],
        dt_ms,
    )

    return JSONResponse(
        {
            **prediction,
            "preprocessing": info,
            "latency_ms": dt_ms,
        }
    )


@app.post("/augment")
async def augment_endpoint(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    limit: int = Form(500),
):
    """
    Run the dataset augmentation step on a folder of preprocessed
    images. **This is a batch / offline job** — the directories must be
    visible inside the model container (mount a volume).

    Args:
        input_dir:  path to a folder of preprocessed images.
        output_dir: where to write the augmented variants.
        limit:      number of variants to generate.
    """
    if not os.path.isdir(input_dir):
        raise HTTPException(status_code=400, detail=f"input_dir not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Lazy import — augmentation drags scipy/skimage which are heavy.
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
