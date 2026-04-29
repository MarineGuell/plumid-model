# =====================================================================
# PlumID — Model microservice
# ---------------------------------------------------------------------
# Wraps the preprocessing + (stub) inference pipeline behind a FastAPI
# server. Listens on $PORT (default 8001).
#
# Build:
#   docker build -t plumid-model:latest .
#
# Run:
#   docker run --rm -p 8001:8001 plumid-model:latest
#
# Health: GET /health -> {"status":"ok",...}
# =====================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8001

WORKDIR /app

# OpenCV headless still needs libgomp + a couple of GL/X stubs at runtime
# (fastNlMeansDenoisingColored uses OpenMP). build-essential is needed
# by scikit-image / scipy wheels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching.
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source.
COPY . /app

# Make sure the entrypoint is executable.
RUN chmod +x /app/entrypoint.sh

# Drop privileges.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

ENTRYPOINT ["/app/entrypoint.sh"]
