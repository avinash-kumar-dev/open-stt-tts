"""Audio helpers for the open STT/TTS PoC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav_mono_f32(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono, resampled to target_sr if needed."""
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if sr == target_sr:
        return mono, sr
    # Linear resample (good enough for PoC; prefer ffmpeg for prod)
    duration = len(mono) / sr
    n = int(duration * target_sr)
    x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32), target_sr


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio, sample_rate)


def split_sentences(text: str) -> list[str]:
    """Naive sentence split for streaming TTS demos."""
    import re

    parts = re.split(r"(?<=[.!?؟。！？])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
