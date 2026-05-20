"""
service.py — Plum'ID model microservice
=========================================

FastAPI wrapper around the preprocessing + inference pipeline.

Endpoints
---------
* GET  /health        — fast liveness probe.
* GET  /model/status  — introspection on the loaded classifier.
* POST /preprocess    — multipart upload; returns the preprocessed PNG
                        (segmentation + denoise + contrast + 224×224).
* POST /predict       — multipart upload; runs the full pipeline:
                          1. preprocess.preprocess(image_bytes)
                          2. If multiple candidates → ask the model which
                             ones are feathers and pick.
                          3. If 0 or 2+ remaining → 422 with a clear
                             warning_code + user_message.
                          4. Otherwise → run the classifier on the
                             selected image and return the prediction.
* POST /augment       — batch dataset augmentation job (offline).

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
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List

import cv2 as cv
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
import io

from inference import ClassifierError, get_classifier
from preprocess import (
    PreprocessResult,
    WARNING_MULTIPLE_CANDIDATES,
    WARNING_NO_FEATHER,
    WARNING_TOO_MANY_FEATHERS,
    preprocess,
    resolve_candidates,
)

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
    version="1.2.0",
    description=(
        "Image preprocessing + inference microservice. Handles multi-feather "
        "images by running each candidate through the classifier and "
        "filtering out the 'Non_plumes' false positives."
    ),
)

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10_000_000))


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #

def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


def _encode_png_bgr(img_bgr: np.ndarray) -> bytes:
    """Encode a BGR ndarray to PNG bytes (for the /preprocess response)."""
    rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _warning_body(result: PreprocessResult) -> Dict[str, Any]:
    """Build the JSON body for a preprocessing warning response."""
    return {
        "ok": False,
        "warning_code": result.warning_code,
        "message": result.user_message,
    }


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


# --------------------------------------------------------------------- #
# Routes                                                                 #
# --------------------------------------------------------------------- #


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "plumid-model", "version": app.version}


@app.get("/model/status")
def model_status() -> Dict[str, Any]:
    return asdict(get_classifier().status)


@app.post("/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...)):
    """
    Run only the preprocessing pipeline on a single image (no inference).

    Returns
    -------
    image/png — the preprocessed image when exactly one feather is found.
    application/json — when 0 or N candidates are found, with the
                       appropriate warning_code and user_message.
    """
    content = await _read_upload(file)

    t0 = time.perf_counter()
    try:
        result = preprocess(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}") from exc

    dt_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not result.ok:
        # Cas 0 plume ou N plumes (sans résolution par le modèle).
        log.info(
            "preprocess: filename=%s bytes=%d warning=%s latency_ms=%s",
            file.filename, len(content), result.warning_code, dt_ms,
        )
        return JSONResponse(
            status_code=result.status_code,
            content={
                **_warning_body(result),
                "candidates_count": (
                    len(result.candidates) if result.candidates else 0
                ),
                "latency_ms": dt_ms,
            },
        )

    png = _encode_png_bgr(result.image)
    headers = {"X-PlumID-Latency-Ms": str(dt_ms)}
    log.info(
        "preprocess: filename=%s bytes=%d ok latency_ms=%s",
        file.filename, len(content), dt_ms,
    )
    return Response(content=png, media_type="image/png", headers=headers)


def _resolve_with_classifier(result: PreprocessResult, clf) -> PreprocessResult:
    """
    Glue between preprocess.resolve_candidates and the classifier.

    The `is_feather_fn` callback runs the model on each BGR candidate
    image (224×224) and returns True for candidates the model recognises
    as a feather (i.e. not 'Non_plumes').
    """
    def _is_feather(img_bgr: np.ndarray) -> bool:
        rgb = _bgr_to_rgb(img_bgr)
        try:
            return clf.is_feather(rgb)
        except Exception as exc:  # noqa: BLE001
            log.warning("is_feather check failed (%s); treating as non-feather", exc)
            return False

    return resolve_candidates(result, is_feather_fn=_is_feather)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """
    Full pipeline: preprocess → (resolve multi-candidates) → predict.

    Possible outcomes
    -----------------
    200 — preprocessing OK and inference succeeded.
    422 — preprocessing failed with one of:
            WARNING_NO_FEATHER       (no recognisable feather in the image)
            WARNING_TOO_MANY_FEATHERS (multiple feathers detected by the model)
    400 — invalid image bytes.
    503 — classifier not loaded yet / failed to load.
    500 — unexpected failure.
    """
    content = await _read_upload(file)
    t0 = time.perf_counter()

    # ------------- 1. Preprocess --------------------------------------- #
    try:
        result = preprocess(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Predict preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}") from exc

    # ------------- 2. Multi-candidate resolution (if needed) ----------- #
    if result.warning_code == WARNING_MULTIPLE_CANDIDATES:
        try:
            clf = get_classifier()
            # Ensure model is loaded so the candidate filter has something
            # to work with. Done in a worker thread to keep the event loop free.
            await asyncio.to_thread(clf.ensure_loaded)
            result = await asyncio.to_thread(
                _resolve_with_classifier, result, clf
            )
        except ClassifierError as exc:
            log.error("Classifier not available for candidate resolution: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Classifier not available: {exc}. "
                    "Check /model/status for details."
                ),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("Candidate resolution failed")
            raise HTTPException(
                status_code=500,
                detail=f"Candidate resolution failed: {exc}",
            ) from exc

    # ------------- 3. Preprocessing-level failures --------------------- #
    if not result.ok:
        dt_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info(
            "predict: filename=%s bytes=%d warning=%s latency_ms=%s",
            file.filename, len(content), result.warning_code, dt_ms,
        )
        return JSONResponse(
            status_code=result.status_code,
            content={
                **_warning_body(result),
                "latency_ms": dt_ms,
            },
        )

    # ------------- 4. Run the classifier on the chosen image ----------- #
    try:
        clf = get_classifier()
        rgb = _bgr_to_rgb(result.image)
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
