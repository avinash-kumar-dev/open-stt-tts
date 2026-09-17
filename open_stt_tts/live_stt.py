"""Live Nemotron STT session for WebSocket / chunked audio (mirrors LiveKit frames)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .stt_nemotron import _set_language, create_recognizer


@dataclass
class LivePartial:
    text: str
    is_final: bool
    elapsed_ms: float
    audio_ms_fed: float


class LiveSttSession:
    """Feed 16 kHz float32 mono chunks; poll for partial/final transcripts."""

    def __init__(self, *, language: str = "en", model_dir=None) -> None:
        self.language = language or "en"
        self.recognizer = create_recognizer(model_dir)
        self.stream = self.recognizer.create_stream()
        _set_language(self.stream, self.language)
        self.t0 = time.perf_counter()
        self.audio_ms = 0.0
        self.last_text = ""
        self.partials = 0
        self.first_partial_ms: float | None = None
        self.finals: list[str] = []
        self.sample_rate = 16000

    def accept_pcm_f32(self, samples: np.ndarray) -> list[LivePartial]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return []
        self.stream.accept_waveform(self.sample_rate, samples)
        self.audio_ms += len(samples) / self.sample_rate * 1000.0
        return self._decode()

    def accept_pcm_i16_bytes(self, data: bytes) -> list[LivePartial]:
        if not data:
            return []
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return self.accept_pcm_f32(arr)

    def _decode(self) -> list[LivePartial]:
        events: list[LivePartial] = []
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream)
        if not isinstance(text, str):
            text = getattr(text, "text", str(text))
        text = (text or "").strip()
        is_ep = bool(self.recognizer.is_endpoint(self.stream))
        elapsed = (time.perf_counter() - self.t0) * 1000.0

        if text and text != self.last_text:
            self.partials += 1
            if self.first_partial_ms is None:
                self.first_partial_ms = elapsed
            self.last_text = text
            events.append(
                LivePartial(
                    text=text,
                    is_final=False,
                    elapsed_ms=elapsed,
                    audio_ms_fed=self.audio_ms,
                )
            )

        if is_ep:
            if text:
                self.finals.append(text)
                events.append(
                    LivePartial(
                        text=text,
                        is_final=True,
                        elapsed_ms=elapsed,
                        audio_ms_fed=self.audio_ms,
                    )
                )
            self.recognizer.reset(self.stream)
            self.last_text = ""
            _set_language(self.stream, self.language)
        return events

    def finish(self) -> list[LivePartial]:
        """Flush with short silence + input_finished."""
        tail = np.zeros(int(self.sample_rate * 0.4), dtype=np.float32)
        events = self.accept_pcm_f32(tail)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream)
        if not isinstance(text, str):
            text = getattr(text, "text", str(text))
        text = (text or "").strip()
        elapsed = (time.perf_counter() - self.t0) * 1000.0
        if text:
            self.finals.append(text)
            events.append(
                LivePartial(
                    text=text,
                    is_final=True,
                    elapsed_ms=elapsed,
                    audio_ms_fed=self.audio_ms,
                )
            )
        return events
