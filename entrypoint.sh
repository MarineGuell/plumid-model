#!/bin/sh
# PlumID model entrypoint
# ------------------------------------------------------------------
# No DB, no migrations. Just announces the port and execs the
# requested command. Mirrors the Poireaut entrypoint shape.
set -e

echo "⟡ PlumID model starting…"

if [ -n "$PORT" ]; then
    echo "⟡ Binding to PORT=$PORT"
fi

echo "⟡ Handing off to: $*"
exec "$@"