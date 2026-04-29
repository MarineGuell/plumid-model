# Plum'ID — Model service

Image preprocessing + (stub) inference microservice for the Plum'ID
feather-identification stack. Runs as a long-lived FastAPI container
and is called by `plumid-api` over the internal network.

The repository keeps both modes available:

* **CLI** (`pipeline.py`) — original batch tool that walks a folder,
  preprocesses every image, and runs data augmentation. Useful for
  building a training set.
* **HTTP service** (`service.py`) — the new entry point used in
  production. Same preprocessing chain, single-image, in-memory.

---

## Pipeline stages (shared by CLI and service)

| Stage              | What it does                                                            |
| ------------------ | ----------------------------------------------------------------------- |
| Segmentation       | Pulls the largest plausible feather-shaped object out of the image. Uses adaptive threshold + watershed, with a contour-detection fallback. |
| Denoising          | `cv2.fastNlMeansDenoisingColored`                                        |
| Contrast           | CLAHE on the L channel (LAB color space)                                |
| Padding & resize   | Square pad with black background, then resize to 224×224                |
| Augmentation (CLI) | Rotate / blur / random noise / horizontal flip — see `augmentation_config.py` |

---

## HTTP service

```bash
# Build & run
docker build -t plumid-model .
docker run --rm -p 8001:8001 plumid-model

# Or directly with Python
pip install -r requirements.txt
uvicorn service:app --host 0.0.0.0 --port 8001
```

### Endpoints

| Method | Path           | Purpose |
| ------ | -------------- | ------- |
| GET    | `/health`      | Liveness probe (used by Railway / docker compose). |
| POST   | `/preprocess`  | Multipart upload (`file`); returns the preprocessed PNG. Optional form fields: `skip_segmentation` (bool), `target_size` (int, default 224). Metadata exposed via `X-PlumID-*` headers. |
| POST   | `/predict`     | Multipart upload (`file`); returns a JSON prediction (currently a clearly-labelled stub — see below). |
| POST   | `/augment`     | Batch job. Form fields: `input_dir`, `output_dir`, `limit`. Requires the directories to be visible inside the container (mount a volume). |

### Why `/predict` returns a stub

No trained classifier is bundled in this build. The endpoint runs the
**real** preprocessing chain, then picks a species deterministically
from the seeded list and tags the response with `model: "stub"` and a
warning string. This lets the API contract be wired end-to-end so the
mobile/web client and the rest of the stack can be tested without
waiting for a model. To go live, replace the marked block in
`service.py` with a real inference call — keep the response shape
stable.

### Example call

```bash
curl -F "file=@feather.jpg" http://localhost:8001/predict | jq
```

```json
{
  "model": "stub",
  "warning": "No trained classifier is bundled ...",
  "species": "Pic épeiche (Dendrocopos major)",
  "confidence": 0.2118,
  "top_k": [
    {"species": "Pic épeiche (Dendrocopos major)", "confidence": 0.2118},
    {"species": "Geai des chênes (Garrulus glandarius)", "confidence": 0.1903},
    {"species": "Corneille noire (Corvus corone)", "confidence": 0.1740}
  ],
  "preprocessing": {
    "input_size": [1024, 768],
    "segmented": true,
    "target_size": 224,
    "bbox_size": [410, 612]
  },
  "latency_ms": 412.7
}
```

---

## Railway deployment

Deploy this repository as its own Railway service:

1. **New service → Deploy from GitHub repo →** select `plumid-model`.
2. Railway picks up `Dockerfile` automatically. The provided
   `railway.json` sets `/health` as the healthcheck path.
3. The service uses Railway's private networking — no public domain
   is needed. The API reaches it at
   `http://${{plumid-model.RAILWAY_PRIVATE_DOMAIN}}:$PORT`.
4. (Optional) Set environment variables — see `.env.example`. The
   defaults are safe for production.

---

## CLI mode (training-set generation)

The original `pipeline.py` is still there:

```bash
# Full pipeline (preprocess + augment)
python pipeline.py -input=raw_images -output=augmented -limit=1000

# Preprocessing only
python pipeline.py -input=raw_images --preprocess-only

# Augmentation only (assumes input is already preprocessed)
python pipeline.py -input=preprocessed --skip-preprocessing -output=augmented -limit=2000
```

Configurable knobs live in `augmentation_config.py`.

---

## Repository layout

```
plumid-model/
├── service.py                       # FastAPI service (HTTP entrypoint)
├── pipeline.py                      # CLI batch tool (training-set generation)
├── Dockerfile                       # service container image
├── railway.json                     # Railway config
├── requirements.txt
├── augmentation/
│   ├── augmentation.py              # DatasetGenerator class
│   └── operations.py                # Rotate / Blur / Flip / Noise
├── augmentation_config.py
├── data_preprocessing/
│   ├── datapreprocessing.py         # disk-based pipeline (CLI)
│   └── single_image.py              # in-memory pipeline (service)
├── utils/
│   └── utils.py                     # helpers (file I/O, progress bar)
└── tests/
```

---

## Tests

```bash
pip install -r requirements.txt
pytest -q
```
