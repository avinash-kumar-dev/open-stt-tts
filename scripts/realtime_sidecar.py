"""Warm sidecar for unofficial supertonic-realtime (run with .venv-realtime only)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "web"
OUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Supertonic realtime sidecar")
app.mount("/out", StaticFiles(directory=str(OUT)), name="out")

_tts = None


def get_tts():
    global _tts
    if _tts is None:
        from supertonic import TTS

        _tts = TTS(auto_download=True)
    return _tts


def _ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def word_delta_iter(text: str):
    parts = text.split(" ")
    for i, w in enumerate(parts):
        yield w if i == len(parts) - 1 else w + " "


@app.get("/health")
def health() -> dict:
    from supertonic import TTS

    return {
        "ok": True,
        "engine": "supertonic-realtime",
        "has_synthesize_stream": hasattr(TTS, "synthesize_stream"),
        "model_loaded": _tts is not None,
    }


@app.post("/stream")
def stream(
    text: str = Form(...),
    voice: str = Form("M1"),
    lang: str = Form("en"),
    speed: float = Form(1.05),
    total_steps: int = Form(8),
    first_chunk_steps: int = Form(4),
    first_chunk_chars: int = Form(60),
    max_chunk_chars: int = Form(180),
    min_chunk_chars: int = Form(24),
    feed: str = Form("full"),
) -> StreamingResponse:
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    if feed not in {"full", "words"}:
        raise HTTPException(400, "feed must be full|words")

    def event_stream() -> Iterator[str]:
        t0 = time.perf_counter()
        tts = get_tts()
        style = tts.get_voice_style(voice_name=voice)
        options = {
            "lang": lang,
            "speed": speed,
            "total_steps": total_steps,
            "first_chunk_steps": first_chunk_steps,
            "first_chunk_chars": first_chunk_chars,
            "max_chunk_chars": max_chunk_chars,
            "min_chunk_chars": min_chunk_chars,
            "feed": feed,
            "voice": voice,
        }
        yield _ndjson(
            {
                "type": "plan",
                "engine": "supertonic-realtime sidecar (warm)",
                "full_text": text,
                "options": options,
            }
        )
        yield _ndjson(
            {
                "type": "sending",
                "text": text if feed == "full" else "(word deltas…)",
                "feed": feed,
            }
        )
        source = word_delta_iter(text) if feed == "words" else text
        first_audio_ms = None
        total_audio_ms = 0.0
        count = 0
        try:
            for chunk in tts.synthesize_stream(
                source,
                style,
                lang=lang,
                speed=speed,
                total_steps=total_steps,
                first_chunk_steps=first_chunk_steps,
                first_chunk_chars=first_chunk_chars,
                max_chunk_chars=max_chunk_chars,
                min_chunk_chars=min_chunk_chars,
            ):
                elapsed = (time.perf_counter() - t0) * 1000
                if first_audio_ms is None:
                    first_audio_ms = elapsed
                arr = np.asarray(chunk.wav, dtype=np.float32).reshape(-1)
                out = OUT / f"rtfork_{chunk.index}_{uuid.uuid4().hex[:8]}.wav"
                sf.write(str(out), arr, int(chunk.sample_rate))
                audio_ms = float(chunk.duration_s) * 1000.0
                total_audio_ms += audio_ms
                count += 1
                yield _ndjson(
                    {
                        "type": "audio",
                        "index": chunk.index,
                        "text": chunk.text,
                        "audio_url": f"/out/{out.name}",
                        "sample_rate": int(chunk.sample_rate),
                        "audio_ms": round(audio_ms, 1),
                        "steps_used": int(chunk.total_steps),
                        "elapsed_ms": round(elapsed, 1),
                        "ttfb_ms": round(first_audio_ms, 1),
                    }
                )
        except Exception as e:  # noqa: BLE001
            yield _ndjson({"type": "error", "error": str(e)})
            return

        wall = (time.perf_counter() - t0) * 1000
        yield _ndjson(
            {
                "type": "done",
                "clause_count": count,
                "ttfb_ms": None if first_audio_ms is None else round(first_audio_ms, 1),
                "wall_ms": round(wall, 1),
                "total_audio_ms": round(total_audio_ms, 1),
                "rtf": round(wall / total_audio_ms, 3) if total_audio_ms > 0 else 0.0,
                "options": options,
            }
        )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main(host: str = "127.0.0.1", port: int = 8766) -> None:
    import uvicorn

    print(f"Realtime sidecar → http://{host}:{port}  (use .venv-realtime Python)")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Warm supertonic-realtime sidecar")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ns = ap.parse_args()
    main(host=ns.host, port=ns.port)
