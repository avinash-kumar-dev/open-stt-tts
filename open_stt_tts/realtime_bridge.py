"""Bridge to unofficial supertonic-realtime via a side venv (no import conflict)."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "realtime_tts_worker.py"
SIDECAR = ROOT / "scripts" / "realtime_sidecar.py"
REALTIME_PYTHON = ROOT / ".venv-realtime" / "bin" / "python"
SIDECAR_URL = os.environ.get("REALTIME_SIDECAR_URL", "http://127.0.0.1:8766")


def realtime_available() -> dict:
    py = REALTIME_PYTHON
    ok = py.is_file() and WORKER.is_file()
    sidecar_up = False
    if ok:
        try:
            with urllib.request.urlopen(f"{SIDECAR_URL}/health", timeout=0.4) as r:
                sidecar_up = r.status == 200
        except Exception:  # noqa: BLE001
            sidecar_up = False
    return {
        "available": ok,
        "sidecar_up": sidecar_up,
        "sidecar_url": SIDECAR_URL,
        "python": str(py) if py.is_file() else None,
        "worker": str(WORKER),
        "install_hint": (
            "python3 -m venv .venv-realtime && "
            ".venv-realtime/bin/pip install 'supertonic-realtime[serve]' fastapi uvicorn"
        ),
        "start_sidecar": (
            f".venv-realtime/bin/python {SIDECAR}   # warm model on :8766"
        ),
        "official": False,
        "upstream": "https://github.com/datmieu204/supertonic-realtime",
        "options": {
            "lang": "ISO code / na",
            "speed": "default 1.05 (0.7–2.0)",
            "total_steps": "diffusion steps for clauses after first (default 8)",
            "first_chunk_steps": "fewer steps on first clause for lower TTFB (default 4)",
            "first_chunk_chars": "char target for first clause (default 60)",
            "max_chunk_chars": "char target for later clauses (default 180)",
            "min_chunk_chars": "min clause length at sentence split (default 24)",
            "feed": "full | words (simulate LLM deltas)",
            "cancel": "threading.Event barge-in (not wired in UI yet)",
            "server": "optional: .venv-realtime/bin/supertonic serve → WS /v1/realtime",
        },
    }


def _iter_sidecar(fields: dict) -> Iterator[str] | None:
    try:
        boundary = "----pocboundary"
        body = b""
        for k, v in fields.items():
            body += (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
            ).encode()
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{SIDECAR_URL}/stream",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=600)

        def gen() -> Iterator[str]:
            with resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    yield line.decode("utf-8", errors="replace")

        return gen()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _normalize_tts_text(text: str) -> str:
    """Normalize punctuation that confuses clause splitters / TTS."""
    return (
        (text or "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .strip()
    )


def iter_synth_audio_parts(
    text: str,
    *,
    voice: str = "M1",
    lang: str = "en",
    speed: float = 1.05,
    first_chunk_steps: int = 4,
) -> Iterator[dict]:
    """
    Yield every audio part for one sentence as it becomes ready.

    Realtime fork often splits one sentence into multiple WAVs (e.g. "First," + rest).
    Callers must play ALL parts — keeping only the last drops words.
    """
    text = _normalize_tts_text(text)
    if not text:
        raise ValueError("empty text")

    # Keep each LLM sentence as ONE clip. The realtime ClauseBuffer splits on
    # commas once len(buf) >= first_chunk_chars — so the target must be
    # STRICTLY greater than the sentence length (len+1). Tiny first clauses
    # like "First," caused gaps; old code also dropped all but the last part.
    got_audio = False
    err: str | None = None
    n = max(len(text) + 1, 64)
    for line in iter_realtime_stream(
        text,
        voice=voice,
        lang=lang,
        speed=speed,
        total_steps=8,
        first_chunk_steps=first_chunk_steps,
        first_chunk_chars=n,
        max_chunk_chars=n,
        min_chunk_chars=min(24, n),
        feed="full",
    ):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "error":
            err = str(ev.get("error") or "realtime error")
            break
        if ev.get("type") != "audio":
            continue
        got_audio = True
        yield {
            "audio_url": ev["audio_url"],
            "text": ev.get("text") or text,
            "ttfb_ms": ev.get("ttfb_ms"),
            "audio_ms": ev.get("audio_ms"),
            "engine": "supertonic-realtime",
            "steps_used": ev.get("steps_used"),
            "part_index": ev.get("index"),
        }

    if got_audio:
        return
    if err:
        # Fall through to official rather than hard-fail when sidecar glitches.
        pass

    from .tts_supertonic import synthesize

    out = ROOT / "out" / "web" / f"clause_{uuid.uuid4().hex[:10]}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    _, stats = synthesize(text, voice=voice, lang=lang, speed=speed, out=out, stream_sentences=False)
    yield {
        "audio_url": f"/out/{out.name}",
        "text": text,
        "ttfb_ms": stats.ttfb_ms,
        "audio_ms": stats.total_audio_ms,
        "engine": "supertonic-official",
        "steps_used": None,
        "part_index": 0,
    }


def synth_one_clause(
    text: str,
    *,
    voice: str = "M1",
    lang: str = "en",
    speed: float = 1.05,
    first_chunk_steps: int = 4,
) -> dict:
    """Synthesize one sentence; concatenate all realtime parts into a single WAV."""
    import numpy as np
    import soundfile as sf

    parts = list(
        iter_synth_audio_parts(
            text, voice=voice, lang=lang, speed=speed, first_chunk_steps=first_chunk_steps
        )
    )
    if not parts:
        raise RuntimeError("TTS produced no audio")
    if len(parts) == 1:
        return {k: v for k, v in parts[0].items() if k != "part_index"}

    # Stitch parts in order so no leading clause audio is dropped.
    arrays: list[np.ndarray] = []
    sr = None
    for p in parts:
        path = ROOT / "out" / "web" / Path(p["audio_url"]).name
        data, file_sr = sf.read(str(path), dtype="float32")
        if sr is None:
            sr = int(file_sr)
        elif int(file_sr) != sr:
            raise RuntimeError(f"sample rate mismatch {file_sr} vs {sr}")
        arrays.append(np.asarray(data, dtype=np.float32).reshape(-1))
    joined = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
    out = ROOT / "out" / "web" / f"clause_join_{uuid.uuid4().hex[:10]}.wav"
    sf.write(str(out), joined, sr or 44100)
    return {
        "audio_url": f"/out/{out.name}",
        "text": _normalize_tts_text(text),
        "ttfb_ms": parts[0].get("ttfb_ms"),
        "audio_ms": sum(float(p.get("audio_ms") or 0) for p in parts),
        "engine": parts[0].get("engine"),
        "steps_used": parts[0].get("steps_used"),
    }


def iter_realtime_stream(
    text: str,
    *,
    voice: str = "M1",
    lang: str = "en",
    speed: float = 1.05,
    total_steps: int = 8,
    first_chunk_steps: int = 4,
    first_chunk_chars: int = 60,
    max_chunk_chars: int = 180,
    min_chunk_chars: int = 24,
    feed: str = "full",
    custom_style: str | None = None,
) -> Iterator[str]:
    """Prefer warm sidecar; fall back to one-shot worker subprocess."""
    info = realtime_available()
    if not info["available"]:
        yield json.dumps({"type": "error", "error": "realtime venv missing", **info}) + "\n"
        return

    fields = {
        "text": text,
        "voice": voice,
        "lang": lang,
        "speed": str(speed),
        "total_steps": str(total_steps),
        "first_chunk_steps": str(first_chunk_steps),
        "first_chunk_chars": str(first_chunk_chars),
        "max_chunk_chars": str(max_chunk_chars),
        "min_chunk_chars": str(min_chunk_chars),
        "feed": feed,
    }

    sidecar = _iter_sidecar(fields)
    if sidecar is not None:
        yield from sidecar
        return

    cmd = [
        str(REALTIME_PYTHON),
        str(WORKER),
        "--text",
        text,
        "--voice",
        voice,
        "--lang",
        lang,
        "--speed",
        str(speed),
        "--total-steps",
        str(total_steps),
        "--first-chunk-steps",
        str(first_chunk_steps),
        "--first-chunk-chars",
        str(first_chunk_chars),
        "--max-chunk-chars",
        str(max_chunk_chars),
        "--min-chunk-chars",
        str(min_chunk_chars),
        "--feed",
        feed,
    ]
    if custom_style:
        cmd.extend(["--custom-style", custom_style])

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    yield json.dumps(
        {
            "type": "plan",
            "engine": "supertonic-realtime worker (cold each request — start sidecar for speed)",
            "hint": info["start_sidecar"],
        }
    ) + "\n"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(ROOT),
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            # skip duplicate plan from worker if we already emitted a hint plan
            if '"type": "plan"' in line or '"type":"plan"' in line:
                continue
            yield line
    finally:
        proc.wait(timeout=30)
        if proc.returncode not in (0, None) and proc.stderr:
            err = proc.stderr.read().strip()
            if err:
                yield json.dumps({"type": "error", "error": err[-2000:]}) + "\n"
