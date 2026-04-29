"""
service.py — Plum'ID model microservice
=========================================

FastAPI wrapper around the preprocessing / inference pipeline. Designed
to run as a long-lived container on Railway alongside the API.

Endpoints
---------
* GET  /health        — liveness probe.
* POST /preprocess    — multipart upload of an image; returns the
                        preprocessed PNG (segmented, denoised, contrast-
                        enhanced, padded to 224×224).
* POST /predict       — multipart upload; returns a JSON prediction. As
                        no trained classifier is bundled in this repo
                        yet, the response is a clearly-labelled stub
                        (random species pick) so the API contract can be
                        wired end-to-end. Replace the stub block once a
                        real model is trained.
* POST /augment       — kept for parity with the CLI: takes a folder of
                        already-preprocessed images mounted into the
                        container and produces N augmented variants.
                        Useful for offline dataset generation jobs; not
                        meant to be called from the public API.

Run
---
    uvicorn service:app --host 0.0.0.0 --port 8001

Environment variables
---------------------
    LOG_LEVEL              INFO | DEBUG | WARNING | ERROR
    MODEL_PREDICT_TIMEOUT  Reserved for future use (real inference).
    PORT                   Honoured by the container CMD (Railway sets it).
"""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from data_preprocessing.single_image import encode_png, preprocess_bytes

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
    version="1.0.0",
    description=(
        "Image preprocessing + (stub) inference microservice for the "
        "Plum'ID feather-identification stack."
    ),
)

# Tuning knob: maximum upload size in bytes (10 MB by default).
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10_000_000))


# Stub list — replaced by a trained classifier once available.
# Mirrors the species seeded in db/initdb/01-schema.sql.
STUB_SPECIES = [
    "Pie bavarde (Pica pica)",
    "Pic épeiche (Dendrocopos major)",
    "Perruche à collier (Psittacula krameri)",
    "Geai des chênes (Garrulus glandarius)",
    "Corneille noire (Corvus corone)",
    "Canard colvert (Anas platyrhynchos)",
]


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
    """Liveness probe used by Railway / docker compose healthchecks."""
    return {"status": "ok", "service": "plumid-model", "version": app.version}


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

    The request flow is:
        1. Decode the upload.
        2. Run the preprocessing pipeline (segmentation + denoise +
           contrast + padding/resize to 224×224).
        3. Hand the tensor to a classifier.

    Steps 1–2 are real. Step 3 is a **placeholder** that returns a
    random pick from the seeded species list, plus a clear
    `model: "stub"` flag in the response. Wire a real model in here
    once one is trained — keep the response shape stable.
    """
    content = await _read_upload(file)

    t0 = time.perf_counter()
    try:
        _, info = preprocess_bytes(content, target_size=224)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Predict preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}") from exc

    # ----------------------------------------------------------------- #
    # >>> Replace this block with a real inference call. <<<            #
    # e.g. `probs = model.predict(tensor)` ; `idx = probs.argmax()`     #
    # then map idx -> species name and copy `probs.tolist()` into       #
    # the `confidences` field.                                          #
    # ----------------------------------------------------------------- #
    rng = random.Random()
    rng.seed(hash(content) & 0xFFFFFFFF)
    pick = rng.choice(STUB_SPECIES)
    confidences = sorted(
        ((sp, rng.random()) for sp in STUB_SPECIES),
        key=lambda x: x[1],
        reverse=True,
    )
    # Force the picked species on top
    confidences = [(pick, max(c for _, c in confidences) + 0.05)] + [
        c for c in confidences if c[0] != pick
    ]
    total = sum(c for _, c in confidences)
    confidences = [(name, round(c / total, 4)) for name, c in confidences]

    dt_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "predict: filename=%s bytes=%d species=%s latency_ms=%s",
        file.filename,
        len(content),
        pick,
        dt_ms,
    )

    return JSONResponse(
        {
            "model": "stub",
            "warning": (
                "No trained classifier is bundled in this build. "
                "Returning a deterministic stub prediction so the API "
                "contract can be exercised end-to-end."
            ),
            "species": pick,
            "confidence": confidences[0][1],
            "top_k": [
                {"species": name, "confidence": conf} for name, conf in confidences[:3]
            ],
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
