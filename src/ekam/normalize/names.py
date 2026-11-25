"""Business-name normalisation: honorific prefixes, legal suffixes, abbreviations."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from ekam.normalize import resources
from ekam.normalize.text import basic_clean
from ekam.normalize.translit import fold, to_latin

# "M/s", "Messrs" are trade prefixes that carry no identity; they become "M S" after cleaning.
_PREFIXES = [("M", "S"), ("MS",), ("MESSRS",), ("MESSERS",), ("THE",)]
_STOP = {"&", "AND", "OF", "THE"}


@lru_cache(maxsize=1)
def _suffix_patterns() -> list[tuple[tuple[str, ...], str]]:
    pats = []
    for code, spec in resources.legal_suffixes().items():
        for variant in spec["match"]:
            toks = tuple(basic_clean(to_latin(variant)).split())
            if toks:
                pats.append((toks, code))
    # longest first so "OPC PRIVATE LIMITED" wins over "PRIVATE LIMITED"
    pats.sort(key=lambda p: -len(p[0]))
    return pats


@lru_cache(maxsize=1)
def _abbrev_map() -> dict[str, str]:
    m = {}
    for full, spec in resources.abbreviations().items():
        for v in spec.get("known", []):
            m[basic_clean(v) or v] = full
    return m


def strip_prefixes(tokens: list[str]) -> list[str]:
    for p in _PREFIXES:
        if len(tokens) > len(p) and tuple(tokens[: len(p)]) == p:
            return tokens[len(p) :]
    return tokens


def split_legal_suffix(tokens: list[str]) -> tuple[list[str], str | None]:
    """Remove the trailing legal-form suffix, returning (base_tokens, canonical_code).

    >>> split_legal_suffix("SHARMA TRADERS PVT LTD".split())
    (['SHARMA', 'TRADERS'], 'PVT_LTD')
    """
    for pat, code in _suffix_patterns():
        n = len(pat)
        if len(tokens) > n and tuple(tokens[-n:]) == pat:
            return tokens[:-n], code
    return tokens, None


def expand_abbreviations(tokens: list[str]) -> list[str]:
    m = _abbrev_map()
    return [m.get(t, t) for t in tokens]


@dataclass(frozen=True)
class NormName:
    raw: str
    clean: str                      # punctuation/case normalised, transliterated
    base_tokens: tuple[str, ...]    # suffix and prefixes removed, abbreviations expanded
    suffix: str | None              # canonical legal form, e.g. PVT_LTD
    folded: tuple[str, ...] = field(default=())  # romanisation-folded base tokens

    @property
    def base(self) -> str:
        return " ".join(self.base_tokens)

    @property
    def key(self) -> str:
        """Order-preserving folded key without stop-words (used for exact-key blocking)."""
        return " ".join(t for t in self.folded if t not in _STOP)


def normalize_name(raw: str | None) -> NormName:
    raw = raw or ""
    clean = basic_clean(to_latin(raw))
    tokens = strip_prefixes(clean.split())
    tokens = expand_abbreviations(tokens)
    base, suffix = split_legal_suffix(tokens)
    if not base:  # the whole name was a suffix-looking string; keep it rather than lose it
        base, suffix = tokens, None
    return NormName(
        raw=raw, clean=clean, base_tokens=tuple(base), suffix=suffix, folded=tuple(fold(base))
    )
