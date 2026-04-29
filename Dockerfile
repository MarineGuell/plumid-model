FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps:
#   - curl                 → healthcheck
#   - libgomp1             → OpenMP (cv2.fastNlMeansDenoisingColored)
#   - libglib2.0-0         → opencv runtime
#   - build-essential      → wheels for scipy / scikit-image on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
RUN chmod +x ./entrypoint.sh

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["sh", "-c", "uvicorn service:app --host 0.0.0.0 --port ${PORT:-8001}"]