"""Supertonic 3 TTS via official Python package."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import split_sentences, write_wav


@dataclass
class TtsChunkResult:
    index: int
    text: str
    samples: np.ndarray
    sample_rate: int
    synth_ms: float
    audio_ms: float
    ttfb_ms: float  # from session start to this chunk's first sample ready

    @property
    def rtf(self) -> float:
        if self.audio_ms <= 0:
            return 0.0
        return self.synth_ms / self.audio_ms


@dataclass
class TtsSessionStats:
    chunks: list[TtsChunkResult] = field(default_factory=list)
    sample_rate: int = 44100
    wall_ms: float = 0.0

    @property
    def ttfb_ms(self) -> float | None:
        return self.chunks[0].ttfb_ms if self.chunks else None

    @property
    def total_audio_ms(self) -> float:
        return sum(c.audio_ms for c in self.chunks)

    @property
    def rtf(self) -> float:
        if self.total_audio_ms <= 0:
            return 0.0
        return self.wall_ms / self.total_audio_ms


def _load_engine():
    from supertonic import TTS

    return TTS(auto_download=True)


def synthesize(
    text: str,
    *,
    voice: str = "M1",
    lang: str = "en",
    speed: float = 1.0,
    steps: int = 8,
    custom_style: str | Path | None = None,
    out: str | Path | None = None,
    stream_sentences: bool = False,
    on_chunk: Callable[[TtsChunkResult], None] | None = None,
) -> tuple[np.ndarray, TtsSessionStats]:
    """
    Synthesize text.

    - stream_sentences=False: one synthesize() call (basic capability).
    - stream_sentences=True: split sentences and synth each; reports TTFB on
      first sentence — demonstrates app/serving-layer streaming used by agents.
    - custom_style: optional Voice Builder JSON (overrides --voice).
    """
    tts = _load_engine()
    if custom_style:
        style = tts.get_voice_style_from_path(str(custom_style))
    else:
        style = tts.get_voice_style(voice_name=voice)
    stats = TtsSessionStats()
    t0 = time.perf_counter()

    pieces = split_sentences(text) if stream_sentences else [text.strip()]
    pieces = [p for p in pieces if p]
    if not pieces:
        raise ValueError("empty text")

    all_audio: list[np.ndarray] = []
    sample_rate = 44100

    for i, piece in enumerate(pieces):
        t_chunk = time.perf_counter()
        wav, duration = tts.synthesize(
            text=piece,
            voice_style=style,
            total_steps=steps,
            speed=speed,
            lang=lang,
            verbose=False,
        )
        synth_ms = (time.perf_counter() - t_chunk) * 1000.0
        samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        sample_rate = int(getattr(tts, "sample_rate", 44100) or 44100)
        stats.sample_rate = sample_rate

        # Official API returns duration as np.ndarray of seconds (per batch item).
        dur = np.asarray(duration).reshape(-1)
        dur_s = float(dur[0]) if dur.size and float(dur[0]) > 0 else 0.0
        audio_ms = dur_s * 1000.0 if dur_s > 0 else len(samples) / sample_rate * 1000.0

        chunk = TtsChunkResult(
            index=i,
            text=piece,
            samples=samples,
            sample_rate=sample_rate,
            synth_ms=synth_ms,
            audio_ms=audio_ms,
            # Session TTFB only meaningful on first chunk; later = this chunk's synth cost
            ttfb_ms=(time.perf_counter() - t0) * 1000.0 if i == 0 else synth_ms,
        )
        stats.chunks.append(chunk)
        all_audio.append(samples)
        if on_chunk:
            on_chunk(chunk)

        # Small silence between sentences when streaming
        if stream_sentences and i < len(pieces) - 1:
            all_audio.append(np.zeros(int(sample_rate * 0.25), dtype=np.float32))

    stats.wall_ms = (time.perf_counter() - t0) * 1000.0
    combined = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype=np.float32)

    if out:
        write_wav(out, combined, sample_rate)
        # Also write per-sentence files when streaming
        if stream_sentences:
            out_path = Path(out)
            for c in stats.chunks:
                part = out_path.with_name(f"{out_path.stem}.s{c.index}{out_path.suffix}")
                write_wav(part, c.samples, c.sample_rate)

    return combined, stats
