# Open STT / TTS PoC — interview pipeline

Standalone **mini interview loop** that mirrors production shape:

| Production (LiveKit agents) | This PoC |
|-----------------------------|----------|
| Chunked mic / Deepgram streaming partials | Nemotron 3.5 ASR via sherpa-onnx (`/ws/stt`) |
| OpenAI chat token stream → Inworld sentence TTS | OpenAI stream → sentence split → Supertonic |

**Zero changes** to learning-agent / main app. Everything lives under `poc/open-stt-tts/`.

| Role | Model | Runtime |
|------|--------|---------|
| STT | NVIDIA **Nemotron 3.5 ASR Streaming 0.6B** | [`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) |
| LLM | OpenAI chat (same key as main app) | `openai` Python SDK stream |
| TTS | **Supertonic 3** (+ optional `supertonic-realtime` sidecar) | Official package / side venv |

## Boss demo — one command

```bash
cd open-stt-tts   # or poc/open-stt-tts if still nested
pip install -r requirements.txt          # once
./scripts/download_stt_model.sh          # once (~650MB)
# optional snappy TTS:
python3 -m venv .venv-realtime && .venv-realtime/bin/pip install 'supertonic-realtime[serve]' fastapi uvicorn

cp .env.example .env                     # set OPENAI_API_KEY
chmod +x scripts/demo_up.sh
./scripts/demo_up.sh
# → http://127.0.0.1:8765
```

In the UI:

1. **STT** — pick a sample → **Stream sample (chunked)**, or **Start mic** → live partials  
2. **Live LLM → TTS** — edit prompt (or **Use STT text**) → **Run live LLM → TTS**  
3. **Full turn** — mic → STT final → auto LLM → speak  

Pass criteria: STT partials before EOF; LLM tokens drip live; first TTS clause plays while LLM still streaming.

## Setup (manual)

```bash
cd poc/open-stt-tts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/download_stt_model.sh          # default 560ms pack
# ./scripts/download_stt_model.sh 160    # lower latency

export OPENAI_API_KEY=...                # or rely on demo_up.sh sourcing ../.env
```

### Realtime TTS sidecar (optional)

```bash
python3 -m venv .venv-realtime
.venv-realtime/bin/pip install 'supertonic-realtime[serve]' fastapi uvicorn
.venv-realtime/bin/python scripts/realtime_sidecar.py   # :8766
```

Interview turn prefers the sidecar (`first_chunk_steps=4`); falls back to official Supertonic in-process.

## API surface

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Mini interview UI |
| `GET /api/status` | Models / OpenAI / sidecar flags |
| `GET /api/sample-wav?id=` | Bundled test wav for chunked WS feed |
| `WS /ws/stt?language=` | Binary PCM s16le @ 16 kHz → JSON partials |
| `POST /api/interview/turn` | NDJSON: `llm_delta` → `tts_sending` → `tts_audio` → `done` |
| `POST /api/stt` / `/api/tts` / … | Older click-lab endpoints still available |

## Layout

```
poc/open-stt-tts/
  README.md
  requirements.txt
  scripts/
    demo_up.sh
    download_stt_model.sh
    realtime_sidecar.py
    realtime_tts_worker.py
  open_stt_tts/
    web.py              # FastAPI lab
    live_stt.py         # WS session
    llm.py              # OpenAI stream
    sentence_buf.py     # LLM → sentence split
    realtime_bridge.py  # sidecar / official TTS
    stt_nemotron.py
    tts_supertonic.py
    static/index.html
  models/               # gitignored
  out/                  # gitignored
```

## Library notes (Sep 2026)

### STT — Nemotron via sherpa-onnx

Streaming is model-native (cache-aware FastConformer-RNNT). Chunk size is baked into each ONNX pack (80 / 160 / 320 / 560 / 1120 ms). Set `language` per stream (`en`, `auto`, …).

### TTS — Supertonic

Official API synthesizes per clause (not AR token audio). The unofficial `supertonic-realtime` fork adds `synthesize_stream()` with `first_chunk_steps` for lower TTFB — still clause-level, not word audio.

## Not in this PoC

- LiveKit agent plugin / room join  
- Spatius / avatar  
- Production Docker / K8s  
- Deepgram / Inworld A/B harness  
- Any edits under learning-agent or main app paths  
