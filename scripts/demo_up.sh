#!/usr/bin/env bash
# Start PoC lab (:8765) + optional realtime sidecar (:8766).
# Does not touch learning-agent / main app.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer already-exported OPENAI_*; else load from .env (never print values).
# Standalone: ./ .env   Legacy nested: ../../.env (ai-interview)
ENV_FILE=""
for candidate in "$ROOT/.env" "$(cd "$ROOT/../.." 2>/dev/null && pwd)/.env"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    ENV_FILE="$candidate"
    break
  fi
done
if [[ -z "${OPENAI_API_KEY:-}" && -n "$ENV_FILE" ]]; then
  eval "$(
    python3 - "$ENV_FILE" <<'PY'
import shlex, sys
from pathlib import Path
wanted = {
    "OPENAI_API_KEY",
    "OPENAI_INTERVIEW_MODEL",
    "OPENAI_INTERVIEW_TEMPERATURE",
}
path = Path(sys.argv[1])
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    k = k.strip()
    if k not in wanted:
        continue
    v = v.strip().strip('"').strip("'")
    print(f"export {k}={shlex.quote(v)}")
PY
  )"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARN: OPENAI_API_KEY not set — STT still works; LLM→TTS turn will 400."
else
  echo "OPENAI_API_KEY: configured"
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing .venv — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# Ensure English demo sample exists (gitignored *.wav)
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

# Free ports if leftover
for port in 8765 8766; do
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "Stopping process(es) on :$port"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 0.8
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
      sleep 0.3
    fi
  fi
done

if [[ -x "$ROOT/.venv-realtime/bin/python" ]]; then
  echo "Starting realtime sidecar on :8766"
  "$ROOT/.venv-realtime/bin/python" "$ROOT/scripts/realtime_sidecar.py" &
  SIDECAR_PID=$!
  sleep 1.5
else
  echo "No .venv-realtime — interview turn falls back to official Supertonic"
  SIDECAR_PID=""
fi

echo "Starting lab UI on :8765"
"$ROOT/.venv/bin/python" -m open_stt_tts serve --port 8765 &
LAB_PID=$!
sleep 1.5

if ! curl -sf "http://127.0.0.1:8765/api/status" >/dev/null; then
  echo "ERROR: lab UI failed to start on :8765"
  kill ${SIDECAR_PID:-} 2>/dev/null || true
  exit 1
fi

echo ""
echo "Lab UI:     http://127.0.0.1:8765"
echo "Sidecar:    http://127.0.0.1:8766/health  (if started)"
echo "Pids: lab=$LAB_PID sidecar=${SIDECAR_PID:-none}"
echo "Press Ctrl+C to stop."
trap 'kill $LAB_PID ${SIDECAR_PID:-} 2>/dev/null; exit 0' INT TERM
wait $LAB_PID
