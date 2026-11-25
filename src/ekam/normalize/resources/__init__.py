"""Loaders for the curated lookup tables in this directory (cached, read-only)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

_DIR = Path(__file__).parent


def _load(name: str) -> dict:
    with (_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def legal_suffixes() -> dict:
    return _load("legal_suffixes.yaml")


@cache
def abbreviations() -> dict:
    return _load("abbreviations.yaml")


@cache
def states() -> dict:
    return _load("states.yaml")


@cache
def cities() -> dict:
    return _load("cities.yaml")


@cache
def hindi_lexicon() -> dict:
    return {str(k): v for k, v in _load("hindi_lexicon.yaml").items()}


@cache
def hindi_lexicon_heldout() -> dict:
    """Generator-only spellings (never used by the normaliser)."""
    return {str(k): v for k, v in _load("hindi_lexicon_heldout.yaml").items()}


@cache
def hindi_lexicon_generator() -> dict:
    """Everything the noise generator may emit: shared core + held-out words."""
    return {**hindi_lexicon(), **hindi_lexicon_heldout()}
