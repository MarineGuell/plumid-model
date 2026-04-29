#!/bin/sh
# =====================================================================
# PlumID model — runtime entrypoint
# ---------------------------------------------------------------------
# Resolves the listening port from $PORT (Railway / Heroku style),
# falling back to 8001.
# =====================================================================
set -e

PORT="${PORT:-8001}"

echo "[plumid-model] starting uvicorn on 0.0.0.0:${PORT}"
exec python -m uvicorn service:app --host 0.0.0.0 --port "${PORT}"
