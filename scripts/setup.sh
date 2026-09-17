#!/usr/bin/env bash
# One-shot setup for the interview PoC demo (same as http://127.0.0.1:8765).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Python venv + deps"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt

echo "==> Realtime TTS sidecar venv (supertonic-realtime — same as live demo)"
if [[ ! -x .venv-realtime/bin/python ]]; then
  python3 -m venv .venv-realtime
fi
.venv-realtime/bin/pip install -q -U pip
.venv-realtime/bin/pip install -q 'supertonic-realtime[serve]' fastapi uvicorn soundfile numpy python-multipart

echo "==> Nemotron STT model (~650MB, skipped if present)"
./scripts/download_stt_model.sh

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> Created .env — put your OPENAI_API_KEY in it, then re-run."
  echo "    Edit: $ROOT/.env"
  exit 0
fi

if ! grep -qE '^OPENAI_API_KEY=.+' .env; then
  echo "WARN: OPENAI_API_KEY empty in .env — fill it before LLM→TTS works."
fi

echo ""
echo "Setup done. Start the demo:"
echo "  ./scripts/demo_up.sh"
echo "  → http://127.0.0.1:8765"
