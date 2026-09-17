#!/usr/bin/env bash
# Start the same demo as http://127.0.0.1:8765
# Lab :8765 + supertonic-realtime sidecar :8766
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" || ! -x "$ROOT/.venv-realtime/bin/python" ]]; then
  echo "Run setup first: ./scripts/setup.sh"
  exit 1
fi

# Load OPENAI_* from ./.env if not already exported (never print values).
ENV_FILE="$ROOT/.env"
if [[ -z "${OPENAI_API_KEY:-}" && -f "$ENV_FILE" ]]; then
  eval "$(
    python3 - "$ENV_FILE" <<'PY'
import shlex, sys
from pathlib import Path
wanted = {"OPENAI_API_KEY", "OPENAI_INTERVIEW_MODEL", "OPENAI_INTERVIEW_TEMPERATURE"}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    k = k.strip()
    if k not in wanted:
        continue
    print(f"export {k}={shlex.quote(v.strip().strip(chr(34)).strip(chr(39)))}")
PY
  )"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARN: OPENAI_API_KEY not set — STT works; LLM→TTS will fail. Put it in .env"
else
  echo "OPENAI_API_KEY: configured"
fi

# English sample wav (generated once; gitignored)
if [[ ! -f "$ROOT/samples/en_interview.wav" ]]; then
  echo "Generating samples/en_interview.wav …"
  mkdir -p "$ROOT/samples"
  "$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path
from open_stt_tts.tts_supertonic import synthesize
out = Path("samples/en_interview.wav")
synthesize(
    "I designed a streaming data pipeline that validated quality with automated checks in production.",
    voice="M1", lang="en", speed=1.05, out=out, stream_sentences=False,
)
print("wrote", out)
PY
fi

for port in 8765 8766; do
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "Stopping :$port"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 0.5
  fi
done

echo "Starting realtime sidecar :8766"
"$ROOT/.venv-realtime/bin/python" "$ROOT/scripts/realtime_sidecar.py" &
SIDECAR_PID=$!
sleep 1.5

echo "Starting lab UI :8765"
"$ROOT/.venv/bin/python" -m open_stt_tts serve --port 8765 &
LAB_PID=$!
sleep 1.5

if ! curl -sf "http://127.0.0.1:8765/api/status" >/dev/null; then
  echo "ERROR: lab failed to start"
  kill $LAB_PID ${SIDECAR_PID:-} 2>/dev/null || true
  exit 1
fi

echo ""
echo "Demo → http://127.0.0.1:8765"
echo "Ctrl+C to stop."
trap 'kill $LAB_PID $SIDECAR_PID 2>/dev/null; exit 0' INT TERM
wait $LAB_PID
