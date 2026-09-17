"""Simple local web UI to click-test STT / TTS / roundtrip."""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .audio_io import split_sentences, write_wav
from .llm import openai_configured, stream_interview_reply
from .live_stt import LiveSttSession
from .realtime_bridge import iter_realtime_stream, iter_synth_audio_parts, realtime_available, synth_one_clause
from .sentence_buf import pop_complete_sentences
from .stt_nemotron import DEFAULT_MODEL_DIR, stream_wav
from .tts_supertonic import synthesize

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
OUT = ROOT / "out" / "web"
OUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Open STT/TTS PoC", docs_url="/docs")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/out", StaticFiles(directory=str(OUT)), name="out")


def _audio_url(path: Path) -> str:
    return f"/out/{path.name}"


def _list_samples() -> list[dict]:
    items = []
    # Prefer local PoC samples (includes English demo clip).
    local = ROOT / "samples"
    if local.is_dir():
        for p in sorted(local.glob("*.wav")):
            hint = p.stem.split("_")[0] if "_" in p.stem else p.stem
            items.append({"id": p.stem, "name": p.name, "path": str(p), "hint_lang": hint})
    wav_dir = DEFAULT_MODEL_DIR / "test_wavs"
    if wav_dir.is_dir():
        for p in sorted(wav_dir.glob("*.wav")):
            items.append({"id": p.stem, "name": p.name, "path": str(p), "hint_lang": p.stem})
    return items


def _ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/status")
def status() -> dict:
    model_ok = (DEFAULT_MODEL_DIR / "encoder.int8.onnx").is_file()
    return {
        "stt_model_ready": model_ok,
        "stt_model_dir": str(DEFAULT_MODEL_DIR) if model_ok else None,
        "samples": _list_samples(),
        "voices": ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"],
        "realtime_fork": realtime_available(),
        "openai_configured": openai_configured(),
        "pipeline": "mic/sample → Nemotron STT | OpenAI stream → Supertonic TTS",
    }


@app.post("/api/tts")
def api_tts(
    text: str = Form(...),
    voice: str = Form("M1"),
    lang: str = Form("en"),
    speed: float = Form(1.0),
    steps: int = Form(8),
    stream_sentences: str = Form("false"),
) -> JSONResponse:
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    stream = str(stream_sentences).lower() in {"1", "true", "yes", "on"}
    out = OUT / f"tts_{uuid.uuid4().hex[:10]}.wav"
    try:
        _, stats = synthesize(
            text,
            voice=voice,
            lang=lang,
            speed=speed,
            steps=steps,
            out=out,
            stream_sentences=stream,
        )
    except Exception as e:  # noqa: BLE001 — surface to UI
        raise HTTPException(500, f"TTS failed: {e}") from e
    return JSONResponse(
        {
            "ok": True,
            "text": text,
            "audio_url": _audio_url(out),
            "ttfb_ms": None if stats.ttfb_ms is None else round(stats.ttfb_ms, 1),
            "wall_ms": round(stats.wall_ms, 1),
            "total_audio_ms": round(stats.total_audio_ms, 1),
            "rtf": round(stats.rtf, 3),
            "sample_rate": stats.sample_rate,
            "chunks": [
                {
                    "index": c.index,
                    "text": c.text,
                    "synth_ms": round(c.synth_ms, 1),
                    "audio_ms": round(c.audio_ms, 1),
                    "rtf": round(c.rtf, 3),
                }
                for c in stats.chunks
            ],
            "chunk_count": len(stats.chunks),
        }
    )


@app.post("/api/tts/stream")
def api_tts_stream(
    text: str = Form(...),
    voice: str = Form("M1"),
    lang: str = Form("en"),
    speed: float = Form(1.0),
    steps: int = Form(8),
) -> StreamingResponse:
    """Sentence-level streaming: emit text-as-sent, then audio URL per sentence."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text required")

    sentences = split_sentences(text) or [text]

    def event_stream() -> Iterator[str]:
        t0 = time.perf_counter()
        yield _ndjson(
            {
                "type": "plan",
                "full_text": text,
                "sentence_count": len(sentences),
                "sentences": sentences,
                "voice": voice,
                "lang": lang,
            }
        )
        first_audio_ms: float | None = None
        total_audio_ms = 0.0
        for i, sentence in enumerate(sentences):
            yield _ndjson(
                {
                    "type": "sending",
                    "index": i,
                    "total": len(sentences),
                    "text": sentence,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            )
            out = OUT / f"tts_s{i}_{uuid.uuid4().hex[:8]}.wav"
            try:
                _, stats = synthesize(
                    sentence,
                    voice=voice,
                    lang=lang,
                    speed=speed,
                    steps=steps,
                    out=out,
                    stream_sentences=False,
                )
            except Exception as e:  # noqa: BLE001
                yield _ndjson({"type": "error", "index": i, "text": sentence, "error": str(e)})
                return

            elapsed = (time.perf_counter() - t0) * 1000
            if first_audio_ms is None:
                first_audio_ms = elapsed
            chunk = stats.chunks[0] if stats.chunks else None
            audio_ms = chunk.audio_ms if chunk else stats.total_audio_ms
            total_audio_ms += audio_ms
            yield _ndjson(
                {
                    "type": "audio",
                    "index": i,
                    "total": len(sentences),
                    "text": sentence,
                    "audio_url": _audio_url(out),
                    "synth_ms": round(chunk.synth_ms, 1) if chunk else round(stats.wall_ms, 1),
                    "audio_ms": round(audio_ms, 1),
                    "rtf": round(chunk.rtf, 3) if chunk else round(stats.rtf, 3),
                    "elapsed_ms": round(elapsed, 1),
                    "ttfb_ms": round(first_audio_ms, 1),
                }
            )

        wall = (time.perf_counter() - t0) * 1000
        yield _ndjson(
            {
                "type": "done",
                "sentence_count": len(sentences),
                "ttfb_ms": None if first_audio_ms is None else round(first_audio_ms, 1),
                "wall_ms": round(wall, 1),
                "total_audio_ms": round(total_audio_ms, 1),
                "rtf": round(wall / total_audio_ms, 3) if total_audio_ms > 0 else 0.0,
            }
        )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/tts/realtime-stream")
def api_tts_realtime_stream(
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
    """Unofficial supertonic-realtime clause stream (side venv; does not replace official)."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    if feed not in {"full", "words"}:
        raise HTTPException(400, "feed must be full|words")

    def event_stream() -> Iterator[str]:
        yield from iter_realtime_stream(
            text,
            voice=voice,
            lang=lang,
            speed=speed,
            total_steps=total_steps,
            first_chunk_steps=first_chunk_steps,
            first_chunk_chars=first_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
            feed=feed,
        )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stt")
async def api_stt(
    language: str = Form("auto"),
    chunk_ms: int = Form(100),
    sample_id: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> JSONResponse:
    tmp_path: Path | None = None
    try:
        if sample_id:
            samples = {s["id"]: s for s in _list_samples()}
            if sample_id not in samples:
                raise HTTPException(404, f"unknown sample {sample_id}")
            wav_path = Path(samples[sample_id]["path"])
        elif file is not None:
            suffix = Path(file.filename or "upload.wav").suffix or ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=OUT)
            tmp_path = Path(tmp.name)
            tmp.write(await file.read())
            tmp.close()
            wav_path = tmp_path
        else:
            raise HTTPException(400, "provide sample_id or file")

        partials: list[dict] = []

        def on_partial(p):
            partials.append(
                {
                    "text": p.text,
                    "is_endpoint": p.is_endpoint,
                    "elapsed_ms": round(p.elapsed_ms, 1),
                    "audio_ms_fed": round(p.audio_ms_fed, 1),
                }
            )

        try:
            text, stats = stream_wav(
                wav_path,
                language=language,
                chunk_ms=chunk_ms,
                on_partial=on_partial,
            )
        except FileNotFoundError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"STT failed: {e}") from e

        return JSONResponse(
            {
                "ok": True,
                "text": text,
                "partials": partials,
                "partial_count": stats.partials,
                "first_partial_ms": stats.first_partial_ms,
                "wall_ms": round(stats.wall_ms, 1),
                "audio_ms": round(stats.audio_ms, 1),
                "rtf": round(stats.rtf, 3),
                "streaming_ok": bool(
                    stats.partials >= 1
                    and stats.first_partial_ms is not None
                    and stats.first_partial_ms < stats.audio_ms
                ),
            }
        )
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/api/roundtrip")
def api_roundtrip(
    text: str = Form(...),
    voice: str = Form("M1"),
    lang: str = Form("en"),
    language: str = Form("en"),
    chunk_ms: int = Form(100),
) -> JSONResponse:
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text required")

    out_tts = OUT / f"rt_{uuid.uuid4().hex[:10]}.wav"
    try:
        samples, tts_stats = synthesize(
            text, voice=voice, lang=lang, out=out_tts, stream_sentences=False
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"TTS failed: {e}") from e

    out_16k = out_tts.with_name(out_tts.stem + "_16k.wav")
    sr = tts_stats.sample_rate
    if sr != 16000:
        duration = len(samples) / sr
        n = int(duration * 16000)
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=max(n, 1), endpoint=False)
        mono = np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.float32)
        write_wav(out_16k, mono, 16000)
    else:
        out_16k = out_tts

    partials: list[dict] = []

    def on_partial(p):
        partials.append({"text": p.text, "is_endpoint": p.is_endpoint})

    try:
        stt_text, stt_stats = stream_wav(
            out_16k, language=language, chunk_ms=chunk_ms, on_partial=on_partial
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"STT failed: {e}") from e

    return JSONResponse(
        {
            "ok": True,
            "source_text": text,
            "stt_text": stt_text,
            "audio_url": _audio_url(out_tts),
            "tts_ttfb_ms": round(tts_stats.ttfb_ms or 0, 1),
            "tts_rtf": round(tts_stats.rtf, 3),
            "stt_first_partial_ms": stt_stats.first_partial_ms,
            "stt_rtf": round(stt_stats.rtf, 3),
            "stt_partials": stt_stats.partials,
            "partials": partials,
            "match": stt_text.strip().lower().rstrip(".") == text.strip().lower().rstrip("."),
        }
    )


@app.get("/favicon.ico", response_model=None)
def favicon():
    return JSONResponse({}, status_code=204)


@app.get("/api/sample-wav")
def api_sample_wav(id: str) -> FileResponse:
    """Serve a bundled Nemotron test wav for chunked browser streaming."""
    samples = {s["id"]: s for s in _list_samples()}
    if id not in samples:
        raise HTTPException(404, f"unknown sample {id}")
    path = Path(samples[id]["path"])
    if not path.is_file():
        raise HTTPException(404, "sample file missing")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.websocket("/ws/stt")
async def ws_stt(websocket: WebSocket, language: str = "en") -> None:
    """Live STT: client sends binary int16 LE PCM @ 16kHz mono; server returns JSON partials."""
    await websocket.accept()
    try:
        session = LiveSttSession(language=language)
    except Exception as e:  # noqa: BLE001
        await websocket.send_json({"type": "error", "error": str(e)})
        await websocket.close()
        return

    await websocket.send_json(
        {
            "type": "ready",
            "language": language,
            "sample_rate": 16000,
            "format": "pcm_s16le_mono",
        }
    )
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                for ev in session.accept_pcm_i16_bytes(message["bytes"]):
                    await websocket.send_json(
                        {
                            "type": "partial" if not ev.is_final else "final",
                            "text": ev.text,
                            "elapsed_ms": round(ev.elapsed_ms, 1),
                            "audio_ms_fed": round(ev.audio_ms_fed, 1),
                            "first_partial_ms": session.first_partial_ms,
                            "partial_count": session.partials,
                        }
                    )
            elif "text" in message and message["text"] is not None:
                raw = message["text"]
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    msg = {"type": raw}
                if msg.get("type") in {"end", "finish", "stop"}:
                    for ev in session.finish():
                        await websocket.send_json(
                            {
                                "type": "partial" if not ev.is_final else "final",
                                "text": ev.text,
                                "elapsed_ms": round(ev.elapsed_ms, 1),
                                "audio_ms_fed": round(ev.audio_ms_fed, 1),
                                "first_partial_ms": session.first_partial_ms,
                                "partial_count": session.partials,
                            }
                        )
                    await websocket.send_json(
                        {
                            "type": "done",
                            "finals": session.finals,
                            "text": " ".join(session.finals).strip(),
                            "first_partial_ms": session.first_partial_ms,
                            "partial_count": session.partials,
                            "audio_ms": round(session.audio_ms, 1),
                        }
                    )
                    break
    except WebSocketDisconnect:
        return


@app.post("/api/interview/turn")
def api_interview_turn(
    prompt: str = Form(...),
    voice: str = Form("M1"),
    lang: str = Form("en"),
    speed: float = Form(1.05),
) -> StreamingResponse:
    """
    Real interview-shaped turn:
    OpenAI token stream → sentence boundaries → Supertonic clause audio.
    Same shape as learning-agent: LLM deltas → TTS.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt required")
    if not openai_configured():
        raise HTTPException(
            400,
            "OPENAI_API_KEY not set. Export it in the shell before starting the PoC.",
        )

    def event_stream() -> Iterator[str]:
        t0 = time.perf_counter()
        buf = ""
        full_llm = ""
        clause_i = 0
        first_token_ms = None
        first_audio_ms = None
        yield _ndjson(
            {
                "type": "plan",
                "prompt": prompt,
                "voice": voice,
                "lang": lang,
                "flow": "openai_stream → sentence_split → supertonic",
            }
        )
        try:
            for token in stream_interview_reply(prompt):
                elapsed = (time.perf_counter() - t0) * 1000
                if first_token_ms is None:
                    first_token_ms = elapsed
                full_llm += token
                buf += token
                yield _ndjson(
                    {
                        "type": "llm_delta",
                        "text": token,
                        "elapsed_ms": round(elapsed, 1),
                        "first_token_ms": round(first_token_ms, 1),
                    }
                )
                sentences, buf = pop_complete_sentences(buf)
                for sentence in sentences:
                    yield _ndjson(
                        {
                            "type": "tts_sending",
                            "index": clause_i,
                            "text": sentence,
                            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                        }
                    )
                    try:
                        # Stream EVERY realtime part (fork may split one sentence into
                        # multiple WAVs). Playing only the last part dropped words.
                        for part in iter_synth_audio_parts(
                            sentence, voice=voice, lang=lang, speed=speed, first_chunk_steps=4
                        ):
                            elapsed = (time.perf_counter() - t0) * 1000
                            if first_audio_ms is None:
                                first_audio_ms = elapsed
                            yield _ndjson(
                                {
                                    "type": "tts_audio",
                                    "index": clause_i,
                                    "text": part["text"],
                                    "audio_url": part["audio_url"],
                                    "engine": part["engine"],
                                    "audio_ms": part.get("audio_ms"),
                                    "elapsed_ms": round(elapsed, 1),
                                    "first_audio_ms": round(first_audio_ms, 1),
                                    "first_token_ms": round(first_token_ms or 0, 1),
                                }
                            )
                            clause_i += 1
                    except Exception as e:  # noqa: BLE001
                        yield _ndjson({"type": "error", "error": f"TTS failed: {e}"})
                        return
        except Exception as e:  # noqa: BLE001
            yield _ndjson({"type": "error", "error": f"LLM failed: {e}"})
            return

        tail = buf.strip()
        if tail:
            yield _ndjson({"type": "tts_sending", "index": clause_i, "text": tail})
            try:
                for part in iter_synth_audio_parts(
                    tail, voice=voice, lang=lang, speed=speed, first_chunk_steps=4
                ):
                    elapsed = (time.perf_counter() - t0) * 1000
                    if first_audio_ms is None:
                        first_audio_ms = elapsed
                    yield _ndjson(
                        {
                            "type": "tts_audio",
                            "index": clause_i,
                            "text": part["text"],
                            "audio_url": part["audio_url"],
                            "engine": part["engine"],
                            "audio_ms": part.get("audio_ms"),
                            "elapsed_ms": round(elapsed, 1),
                            "first_audio_ms": round(first_audio_ms, 1),
                            "first_token_ms": round(first_token_ms or 0, 1),
                        }
                    )
                    clause_i += 1
            except Exception as e:  # noqa: BLE001
                yield _ndjson({"type": "error", "error": f"TTS failed: {e}"})
                return

        yield _ndjson(
            {
                "type": "done",
                "llm_text": full_llm.strip(),
                "clauses": clause_i,
                "first_token_ms": None if first_token_ms is None else round(first_token_ms, 1),
                "first_audio_ms": None if first_audio_ms is None else round(first_audio_ms, 1),
                "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    print(f"Open STT/TTS UI → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
