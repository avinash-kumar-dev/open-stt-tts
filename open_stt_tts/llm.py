"""Streaming OpenAI chat — same shape as learning-agent LLM → TTS."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path


INTERVIEW_SYSTEM = """You are a professional technical interview coach speaking out loud.
Give a clear, conversational spoken answer in 3–6 short sentences.
Do not use markdown, bullet lists, or code fences — plain speech only.
Keep each sentence under ~25 words so speech can start quickly."""

_ENV_LOADED = False


def _ensure_openai_env() -> None:
    """Load OPENAI_* from ai-interview/.env when not already exported."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return
    # Prefer local .env (standalone repo); also accept nested ai-interview/.env.
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / ".env",  # repo root (open-stt-tts/.env)
        here.parents[3] / ".env",  # legacy: ai-interview/.env when nested
    ]
    wanted = {
        "OPENAI_API_KEY",
        "OPENAI_INTERVIEW_MODEL",
        "OPENAI_INTERVIEW_TEMPERATURE",
    }
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                if k not in wanted or os.environ.get(k, "").strip():
                    continue
                os.environ[k] = v.strip().strip('"').strip("'")
        except OSError:
            continue
        break


def openai_configured() -> bool:
    _ensure_openai_env()
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def stream_interview_reply(
    user_text: str,
    *,
    model: str | None = None,
    system: str = INTERVIEW_SYSTEM,
) -> Iterator[str]:
    """Yield content token deltas from OpenAI chat completions stream."""
    _ensure_openai_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model_name = model or os.environ.get("OPENAI_INTERVIEW_MODEL", "gpt-4.1-mini")
    stream = client.chat.completions.create(
        model=model_name,
        temperature=float(os.environ.get("OPENAI_INTERVIEW_TEMPERATURE", "0.7")),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text.strip()},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue
        piece = getattr(delta, "content", None) or ""
        if piece:
            yield piece
