"""Lightweight language detection utilities.

Currently supports English and German with a simple heuristic to avoid
extra dependencies. Returns "unknown" when confidence is low.
"""

from __future__ import annotations

import re
from typing import Iterable


_GERMAN_CHARS = set("äöüßÄÖÜ")
_GERMAN_STOPWORDS = {
    "der", "die", "das", "und", "ist", "nicht", "mit", "für", "auf", "im", "den",
    "eine", "ein", "zu", "von", "als", "auch", "sich", "dem", "des", "bei", "oder",
    "wir", "sie", "er", "es", "dass", "wie", "wird", "werden", "kann", "können",
    "über", "nach", "vor", "aus", "an", "am", "um", "doch", "nur", "noch"
}
_ENGLISH_STOPWORDS = {
    "the", "and", "is", "are", "not", "with", "for", "on", "in", "to", "of",
    "a", "an", "as", "also", "this", "that", "these", "those", "be", "was", "were",
    "can", "could", "should", "would", "will", "have", "has", "had", "from", "by",
    "about", "into", "over", "after", "before", "only", "more", "than"
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß]+", text.lower())


def _count_hits(tokens: Iterable[str], vocab: set[str]) -> int:
    return sum(1 for t in tokens if t in vocab)


def detect_language(text: str) -> str:
    """Detect language from text. Returns "en", "de", or "unknown"."""
    if not text:
        return "unknown"

    if any(ch in text for ch in _GERMAN_CHARS):
        return "de"

    tokens = _tokenize(text)
    if len(tokens) < 5:
        return "unknown"

    de_hits = _count_hits(tokens, _GERMAN_STOPWORDS)
    en_hits = _count_hits(tokens, _ENGLISH_STOPWORDS)

    if de_hits == 0 and en_hits == 0:
        return "unknown"

    if de_hits >= max(3, en_hits * 1.2):
        return "de"
    if en_hits >= max(3, de_hits * 1.2):
        return "en"

    return "unknown"
