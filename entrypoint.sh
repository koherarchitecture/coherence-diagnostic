#!/bin/bash
set -e

MODEL_DIR="/app/models/deberta-coherence"
DATA_DIR="/app/data"

# Weights ship as a GitHub Release tarball, NOT in the git repo (GitHub's 100MB
# file limit; LFS bandwidth is too small for a public model). The tarball extracts
# to model.safetensors + the small config/tokenizer files. Override MODEL_URL to
# pin a different version (e.g. .../v1.1/model.tar.gz) or host elsewhere.
MODEL_URL="${MODEL_URL:-https://github.com/koherarchitecture/coherence-diagnostic/releases/download/v1.0/model.tar.gz}"

mkdir -p "$DATA_DIR" "$MODEL_DIR"

# Download + extract on first run (persistent volume starts empty), or if the
# weights are missing / only a stub.
WEIGHTS="$MODEL_DIR/model.safetensors"
SIZE=$(stat -c%s "$WEIGHTS" 2>/dev/null || echo 0)
if [ ! -f "$WEIGHTS" ] || [ "$SIZE" -lt 1000000 ]; then
    echo "Downloading model from $MODEL_URL ..."
    curl -fL --retry 3 -o /tmp/model.tar.gz "$MODEL_URL"
    tar -xzf /tmp/model.tar.gz -C "$MODEL_DIR"
    rm -f /tmp/model.tar.gz
    echo "Model ready ($(stat -c%s "$WEIGHTS") bytes)."
else
    echo "Model present at $WEIGHTS ($SIZE bytes)."
fi

# Start application
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
