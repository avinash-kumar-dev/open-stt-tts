"""Nemotron 3.5 streaming STT via sherpa-onnx."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "nemotron-current"


@dataclass
class SttPartial:
    text: str
    is_endpoint: bool
    elapsed_ms: float
    audio_ms_fed: float


@dataclass
class SttSessionStats:
    partials: int = 0
    endpoints: int = 0
    first_partial_ms: float | None = None
    wall_ms: float = 0.0
    audio_ms: float = 0.0
    finals: list[str] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        if self.audio_ms <= 0:
            return 0.0
        return self.wall_ms / self.audio_ms


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not path.is_dir():
        raise FileNotFoundError(
            f"STT model not found at {path}. Run: ./scripts/download_stt_model.sh"
        )
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing {path / name}")
    return path


def create_recognizer(
    model_dir: str | Path | None = None,
    *,
    num_threads: int = 2,
    provider: str = "cpu",
    enable_endpoint: bool = True,
):
    import sherpa_onnx

    md = resolve_model_dir(model_dir)
    # Nemotron packs use feature_dim=128 (not the Zipformer default 80).
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(md / "tokens.txt"),
        encoder=str(md / "encoder.int8.onnx"),
        decoder=str(md / "decoder.int8.onnx"),
        joiner=str(md / "joiner.int8.onnx"),
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=128,
        model_type="nemotron",
        decoding_method="greedy_search",
        enable_endpoint_detection=enable_endpoint,
        provider=provider,
    )


def _set_language(stream, language: str) -> None:
    lang = (language or "auto").strip()
    # Python binding: set_option; some builds expose SetOption.
    if hasattr(stream, "set_option"):
        stream.set_option("language", lang)
    elif hasattr(stream, "SetOption"):
        stream.SetOption("language", lang)


def stream_wav(
    wav_path: str | Path,
    *,
    language: str = "auto",
    chunk_ms: int = 100,
    model_dir: str | Path | None = None,
    on_partial: Callable[[SttPartial], None] | None = None,
) -> tuple[str, SttSessionStats]:
    """Feed a WAV in small chunks to prove streaming partials."""
    from .audio_io import load_wav_mono_f32

    samples, sr = load_wav_mono_f32(wav_path, target_sr=16000)
    recognizer = create_recognizer(model_dir)
    stream = recognizer.create_stream()
    _set_language(stream, language)

    stats = SttSessionStats()
    t0 = time.perf_counter()
    chunk = max(1, int(sr * chunk_ms / 1000))
    last_text = ""
    finals: list[str] = []

    for i in range(0, len(samples), chunk):
        piece = samples[i : i + chunk]
        stream.accept_waveform(sr, piece)
        stats.audio_ms = (i + len(piece)) / sr * 1000.0

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        text = recognizer.get_result(stream)
        # get_result may return str or object with .text depending on version
        if not isinstance(text, str):
            text = getattr(text, "text", str(text))
        text = (text or "").strip()
        is_ep = bool(recognizer.is_endpoint(stream))
        elapsed = (time.perf_counter() - t0) * 1000.0

        if text and text != last_text:
            stats.partials += 1
            if stats.first_partial_ms is None:
                stats.first_partial_ms = elapsed
            last_text = text
            event = SttPartial(text=text, is_endpoint=is_ep, elapsed_ms=elapsed, audio_ms_fed=stats.audio_ms)
            if on_partial:
                on_partial(event)

        if is_ep:
            stats.endpoints += 1
            if text:
                finals.append(text)
            recognizer.reset(stream)
            last_text = ""
            _set_language(stream, language)

    # Flush tail
    tail = np.zeros(int(sr * 0.5), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = recognizer.get_result(stream)
    if not isinstance(text, str):
        text = getattr(text, "text", str(text))
    text = (text or "").strip()
    if text:
        finals.append(text)
        if on_partial:
            on_partial(
                SttPartial(
                    text=text,
                    is_endpoint=True,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    audio_ms_fed=stats.audio_ms,
                )
            )

    stats.wall_ms = (time.perf_counter() - t0) * 1000.0
    stats.finals = finals
    final_text = " ".join(finals).strip() or last_text
    return final_text, stats


def stream_mic(
    *,
    language: str = "auto",
    model_dir: str | Path | None = None,
    seconds: float = 0.0,
    on_partial: Callable[[SttPartial], None] | None = None,
) -> tuple[str, SttSessionStats]:
    """Live mic → streaming STT. Ctrl+C or --seconds to stop."""
    import sounddevice as sd

    recognizer = create_recognizer(model_dir)
    stream = recognizer.create_stream()
    _set_language(stream, language)

    stats = SttSessionStats()
    finals: list[str] = []
    last_text = ""
    t0 = time.perf_counter()
    sample_rate = 16000
    block = int(sample_rate * 0.1)

    print("Listening… (Ctrl+C to stop)" if seconds <= 0 else f"Listening {seconds}s…")

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        nonlocal last_text
        samples = indata[:, 0].copy()
        stream.accept_waveform(sample_rate, samples)
        stats.audio_ms += frames / sample_rate * 1000.0
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        text = recognizer.get_result(stream)
        if not isinstance(text, str):
            text = getattr(text, "text", str(text))
        text = (text or "").strip()
        is_ep = bool(recognizer.is_endpoint(stream))
        elapsed = (time.perf_counter() - t0) * 1000.0
        if text and text != last_text:
            stats.partials += 1
            if stats.first_partial_ms is None:
                stats.first_partial_ms = elapsed
            last_text = text
            if on_partial:
                on_partial(SttPartial(text=text, is_endpoint=is_ep, elapsed_ms=elapsed, audio_ms_fed=stats.audio_ms))
        if is_ep:
            stats.endpoints += 1
            if text:
                finals.append(text)
            recognizer.reset(stream)
            last_text = ""
            _set_language(stream, language)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block,
            callback=callback,
        ):
            if seconds > 0:
                sd.sleep(int(seconds * 1000))
            else:
                while True:
                    sd.sleep(200)
    except KeyboardInterrupt:
        pass

    stats.wall_ms = (time.perf_counter() - t0) * 1000.0
    stats.finals = finals
    return " ".join(finals).strip() or last_text, stats
