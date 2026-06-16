#!/bin/bash
# Download the DeBERTa model (~564MB tarball, ~704MB extracted) for local use.
#
# The weights are hosted as a GitHub Release tarball, not in the git repo
# (GitHub's 100MB file limit). The tarball extracts to model.safetensors plus
# the small config/tokenizer files, completing models/deberta-coherence/.
#
# Usage:  ./download_model.sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)/models/deberta-coherence"
URL="${MODEL_URL:-https://github.com/koherarchitecture/coherence-diagnostic/releases/download/v1.0/model.tar.gz}"

mkdir -p "$DIR"
DST="$DIR/model.safetensors"

if [ -f "$DST" ] && [ "$(wc -c < "$DST")" -gt 1000000 ]; then
    echo "Model weights already present at $DST"
    exit 0
fi

echo "Downloading model tarball from:"
echo "  $URL"
curl -fL --retry 3 -o /tmp/coherence-model.tar.gz "$URL"
tar -xzf /tmp/coherence-model.tar.gz -C "$DIR"
rm -f /tmp/coherence-model.tar.gz
echo "Done -> $DST ($(wc -c < "$DST") bytes)"
