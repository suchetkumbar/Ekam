"""Phonetic keys for blocking: metaphone over romanisation-folded tokens."""

from __future__ import annotations

from functools import lru_cache

import jellyfish

_SKIP = {"&", "AND", "OF", "THE", "SRI", "SHRI", "NEW", "M", "S"}


@lru_cache(maxsize=500_000)
def metaphone(token: str) -> str:
    return jellyfish.metaphone(token) if token.isalpha() else token


def phonetic_key(folded_tokens: tuple[str, ...] | list[str], n_tokens: int = 2) -> str:
    """Metaphone of the first ``n_tokens`` significant folded tokens, joined by '|'.

    Folding first means SHREE/SRI and LAKSHMI/LAXMI already agree before metaphone runs.
    """
    sig = [t for t in folded_tokens if t not in _SKIP and len(t) > 1]
    return "|".join(metaphone(t) for t in sig[:n_tokens])


def soundex(token: str) -> str:
    return jellyfish.soundex(token) if token.isalpha() else token
