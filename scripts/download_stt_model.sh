#!/usr/bin/env bash
# Download sherpa-onnx Nemotron 3.5 multilingual streaming ONNX pack.
# Usage: ./scripts/download_stt_model.sh [80|160|320|560|1120]
set -euo pipefail

CHUNK="${1:-560}"
case "$CHUNK" in
  80|160|320|560|1120) ;;
  *)
    echo "chunk must be 80, 160, 320, 560, or 1120 (ms)" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
NAME="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-${CHUNK}ms-int8-2026-06-11"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NAME}.tar.bz2"
DEST="$MODELS/$NAME"

mkdir -p "$MODELS"
if [[ -f "$DEST/encoder.int8.onnx" && -f "$DEST/tokens.txt" ]]; then
  echo "Already present: $DEST"
  ln -sfn "$NAME" "$MODELS/nemotron-current"
  echo "Linked models/nemotron-current -> $NAME"
  exit 0
fi

TMP="$MODELS/${NAME}.tar.bz2"
echo "Downloading $URL"
curl -L --fail --progress-bar -o "$TMP" "$URL"
echo "Extracting..."
tar -xjf "$TMP" -C "$MODELS"
rm -f "$TMP"
ln -sfn "$NAME" "$MODELS/nemotron-current"
echo "Done. Active model: models/nemotron-current ($CHUNK ms chunk)"
ls -lh "$DEST" | head -20
