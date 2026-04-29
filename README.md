# Plum'ID — Model service

Image preprocessing **+ inference** microservice for the Plum'ID
feather-identification stack. Runs as a long-lived FastAPI container
and is called by `plumid-api` over the internal network.

The trained classifier is **downloaded from HuggingFace** at first use
(repo configured by `HF_REPO_ID`). Default repo:
[`Azerty112/Plum_ID_V1`](https://huggingface.co/Azerty112/Plum_ID_V1).

This repo keeps both modes available:

* **HTTP service** (`service.py`) — production entry point. Single-image
  in-memory pipeline, real model inference.
* **CLI** (`pipeline.py`) — the original batch tool that walks a folder,
  preprocesses every image, and runs data augmentation. Useful for
  building a training set.

---

## Pipeline stages

| Stage              | What it does                                                            |
| ------------------ | ----------------------------------------------------------------------- |
| Segmentation       | Pulls the largest plausible feather-shaped object out of the image.      |
| Denoising          | `cv2.fastNlMeansDenoisingColored`                                        |
| Contrast           | CLAHE on the L channel (LAB color space)                                |
| Padding & resize   | Square pad, resize to 224×224 (matches model input)                     |
| **Inference**      | PyTorch classifier downloaded from HuggingFace                          |
| Augmentation (CLI) | Rotate / blur / random noise / horizontal flip                          |

---

## HTTP service

```bash
# Build & run
docker build -t plumid-model .
docker run --rm -p 8001:8001 \
  -e HF_REPO_ID=Azerty112/Plum_ID_V1 \
  plumid-model

# Or directly with Python
pip install -r requirements.txt
HF_REPO_ID=Azerty112/Plum_ID_V1 uvicorn service:app --host 0.0.0.0 --port 8001
```

### Endpoints

| Method | Path             | Purpose |
| ------ | ---------------- | ------- |
| GET    | `/health`        | Liveness probe (returns 200 even while the model is downloading). |
| GET    | `/model/status`  | Readiness + introspection: architecture, num classes, weights file, load duration, last error. |
| POST   | `/preprocess`    | Multipart upload (`file`); returns the preprocessed PNG. Optional form fields: `skip_segmentation`, `target_size`. |
| POST   | `/predict`       | Multipart upload (`file`); returns a JSON prediction. |
| POST   | `/augment`       | Batch job (offline dataset generation). |

### Example call

```bash
curl -F "file=@feather.jpg" http://localhost:8001/predict | jq
```

```json
{
  "model": "real",
  "architecture": "resnet50",
  "num_classes": 6,
  "species": "Pic épeiche (Dendrocopos major)",
  "confidence": 0.842,
  "top_k": [
    {"species": "Pic épeiche (Dendrocopos major)", "confidence": 0.842},
    {"species": "Geai des chênes (Garrulus glandarius)", "confidence": 0.107},
    {"species": "Corneille noire (Corvus corone)", "confidence": 0.031}
  ],
  "preprocessing": {
    "input_size": [1024, 768],
    "segmented": true,
    "target_size": 224,
    "bbox_size": [410, 612]
  },
  "latency_ms": 312.7
}
```

### Checking that the model loaded

```bash
curl http://localhost:8001/model/status | jq
```

Right after startup you might see `"ready": false` (the weights are
still downloading). Once the background loader finishes you get:

```json
{
  "ready": true,
  "architecture": "resnet50",
  "num_classes": 6,
  "class_names": ["Pie bavarde (Pica pica)", "..."],
  "weights_file": "model_V2_55%_Plum_ID.pth",
  "repo_id": "Azerty112/Plum_ID_V1",
  "revision": "main",
  "input_size": 224,
  "device": "cpu",
  "last_load_seconds": 4.31,
  "last_error": null
}
```

If `ready: false` persists, look at `last_error` — common causes are a
private repo without `HF_TOKEN`, a wrong filename, or an architecture
the auto-detector can't recognise.

---

## How it works

1. On startup, a background thread calls `Classifier.ensure_loaded()`.
   `/health` keeps responding 200 throughout, so the container doesn't
   get killed by Railway / Kubernetes during the cold-boot download.
2. The loader calls `huggingface_hub.hf_hub_download` to fetch the
   `.pth` file, with the result cached at `$HF_HOME`.
3. The state_dict is inspected to **auto-detect the architecture** —
   ResNet (any depth), EfficientNet B0–B3, MobileNet V2. The number of
   classes is read from the last linear layer's weight shape.
4. The matching torchvision model is instantiated, the weights are
   loaded, and the model is set to `eval()` on CPU.
5. `/predict` runs preprocessing + inference, returns the top-3 species
   with confidence scores.

If you retrain with a different backbone, **no code change is needed**
provided the architecture is one of the supported ones. If it's a
custom CNN, set `MODEL_ARCHITECTURE` and (if needed) extend
`_detect_architecture` in `inference/classifier.py`.

---

## Configuration reference

See [`.env.example`](.env.example) for the full list. The two essentials:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_REPO_ID` | *(required)* | HuggingFace repo containing the `.pth`. |
| `HF_HOME` | `/home/appuser/.cache/huggingface` | Cache directory. Mount a volume here on Railway to avoid re-downloading on every deploy. |

Optional:

| Variable | Use case |
| --- | --- |
| `HF_MODEL_FILENAME` | Pin a specific weights file. |
| `HF_REVISION` | Pin a git ref (default `main`). |
| `HF_TOKEN` | Required for private repos. |
| `MODEL_NUM_CLASSES` | Force the number of classes (auto-inferred otherwise). |
| `MODEL_CLASS_NAMES` | CSV of species names in model output order. |
| `MODEL_ARCHITECTURE` | Force the architecture if auto-detection fails. |
| `MODEL_INPUT_SIZE` / `MODEL_NORM_MEAN` / `MODEL_NORM_STD` | Override preprocessing for non-standard models. |
| `PRELOAD_MODEL` | `0` to defer loading until the first `/predict` (default `1`). |

---

## Railway deployment

1. **New service → Deploy from GitHub repo →** select `plumid-model`.
2. Railway picks up `Dockerfile` automatically.
3. **Variables**:
   * `HF_REPO_ID=Azerty112/Plum_ID_V1`
   * (optional) `HF_TOKEN=hf_…` if the repo becomes private.
4. **Volumes**: attach a volume mounted at
   `/home/appuser/.cache/huggingface` to keep the weights across deploys.
5. The service uses Railway's private networking — no public domain
   required. The API reaches it at
   `http://${{plumid-model.RAILWAY_PRIVATE_DOMAIN}}:8001`.

The first deploy takes ~3 minutes (Docker build + first download).
Subsequent deploys reuse the cached image layers and the cached weights.

---

## CLI mode (training-set generation)

The original `pipeline.py` is still there:

```bash
python pipeline.py -input=raw_images -output=augmented -limit=1000
python pipeline.py -input=raw_images --preprocess-only
python pipeline.py -input=preprocessed --skip-preprocessing -output=augmented -limit=2000
```

Configurable knobs live in `augmentation_config.py`.

---

## Repository layout

```
plumid-model/
├── service.py                       # FastAPI service (HTTP entrypoint)
├── pipeline.py                      # CLI batch tool (training-set generation)
├── inference/
│   ├── __init__.py
│   └── classifier.py                # HF download + arch detection + predict
├── data_preprocessing/
│   ├── datapreprocessing.py         # disk-based pipeline (CLI)
│   └── single_image.py              # in-memory pipeline (service)
├── augmentation/
│   ├── augmentation.py              # DatasetGenerator class
│   └── operations.py                # Rotate / Blur / Flip / Noise
├── augmentation_config.py
├── utils/
│   └── utils.py
├── tests/
├── Dockerfile
├── entrypoint.sh
├── railway.json
├── requirements.txt
└── .env.example
```

---

## Tests

```bash
pip install -r requirements.txt
pytest -q
```
