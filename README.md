# Open STT / TTS — interview demo

Same lab as **http://127.0.0.1:8765**: mic/sample → Nemotron STT → OpenAI live tokens → Supertonic speech.

TTS uses the **unofficial `supertonic-realtime` sidecar** (what the live demo runs). Official Supertonic is only a fallback.

## Run

```bash
git clone https://github.com/avinash-kumar-dev/open-stt-tts.git
cd open-stt-tts

chmod +x scripts/*.sh
./scripts/setup.sh          # venvs, deps, STT model (~650MB once)
# edit .env → set OPENAI_API_KEY=

./scripts/demo_up.sh        # → http://127.0.0.1:8765
```

In the UI:

1. **STT** — sample or mic → live partials  
2. **Live LLM → TTS** — real OpenAI stream, sentence clips spoken  
3. **Full turn** — mic → transcript → LLM → speak (stage lights + log)

Stop with Ctrl+C.

## Needs

- Python 3.10+
- `OPENAI_API_KEY` in `.env` (see `.env.example`)
- ~1GB disk for STT ONNX + Supertonic weights (auto-download)

## Not included

LiveKit agents, Deepgram, Inworld, main interview app wiring.
