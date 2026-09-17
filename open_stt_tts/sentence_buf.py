"""Helpers to pull complete sentences from a growing LLM buffer."""

from __future__ import annotations

import re


_SENT_END = re.compile(r"(?<=[.!?؟。！？])\s+")


def pop_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Split off complete sentences from buffer.
    Returns (sentences, remainder). Remainder keeps unfinished tail.
    """
    if not buffer:
        return [], ""
    # Need trailing whitespace or EOS after punctuation to commit a sentence
    # while streaming — otherwise "Dr." mid-stream is ambiguous.
    parts = _SENT_END.split(buffer)
    if len(parts) == 1:
        # No split yet — if buffer ends with punctuation and we got a long pause
        # caller may flush; while streaming keep in remainder.
        return [], buffer

    # re.split with capturing isn't used; split removes delimiters' following space
    # Rebuild: find spans with finditer on original
    sentences: list[str] = []
    last = 0
    for m in re.finditer(r"[.!?؟。！？]+(?:\s+|$)", buffer):
        end = m.end()
        piece = buffer[last:end].strip()
        if piece:
            sentences.append(piece)
        last = end
    remainder = buffer[last:]
    return sentences, remainder
