#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/dreamshaper}"
MODELS_DIR="${LCM_MODELS_DIR:-/models}"
PORT="${PORT:-8765}"
HOST="${HOST:-0.0.0.0}"

export LCM_MODELS_DIR="${MODELS_DIR}"

if [[ ! -d "${MODELS_DIR}" ]]; then
    echo "Warning: models directory ${MODELS_DIR} does not exist"
fi

cd "${APP_DIR}"
exec ./dreamshaper-cli serve --host "${HOST}" --port "${PORT}"
