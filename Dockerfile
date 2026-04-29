# =====================================================================
# PlumID — Model microservice
# ---------------------------------------------------------------------
# Wraps the preprocessing + inference pipeline behind a FastAPI server.
# The PyTorch classifier is downloaded from HuggingFace at first use
# (or eagerly at startup if PRELOAD_MODEL=1).
#
# Build:
#   docker build -t plumid-model:latest .
#
# Run:
#   docker run --rm -p 8001:8001 \
#     -e HF_REPO_ID=Azerty112/Plum_ID_V1 \
#     plumid-model:latest
#
# The first run downloads ~30 MB of weights from HuggingFace to
# /home/appuser/.cache/huggingface — mount a volume there in production
# to avoid re-downloading on every cold boot.
# =====================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# System deps:
#   - curl                 → healthcheck
#   - libgomp1             → OpenMP (cv2.fastNlMeansDenoisingColored,
#                            and torch on some kernels)
#   - libglib2.0-0         → opencv runtime
#   - build-essential      → wheels for scipy / scikit-image on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# pip respects the `--extra-index-url` line at the top of requirements.txt,
# so torch CPU comes from download.pytorch.org and everything else from
# PyPI, with a single `pip install`.
RUN pip install -r requirements.txt

COPY . /app
RUN chmod +x ./entrypoint.sh

# Drop privileges and pre-create the HF cache dir.
RUN useradd -m appuser \
 && mkdir -p /home/appuser/.cache/huggingface \
 && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

EXPOSE 8001

# Generous start-period: the first boot may need to download ~30 MB of
# model weights. /health stays 200 throughout (model load happens in a
# background thread), so 30s is plenty.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["sh", "-c", "uvicorn service:app --host 0.0.0.0 --port ${PORT:-8001}"]
