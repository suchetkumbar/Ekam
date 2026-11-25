"""Unicode and punctuation normalisation shared by every field normaliser."""

from __future__ import annotations

import re
import unicodedata

# Indic script blocks (Devanagari .. Malayalam). Used to decide whether to transliterate.
_INDIC_RE = re.compile(r"[ऀ-ൿ]")
_PUNCT_RE = re.compile(r"[.,;:()\[\]{}/\\|'\"`~!?*_+=<>#@$%^-]")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\u0000-\u001f\u007f​-‍﻿]")


def nfkc(s: str) -> str:
    """NFKC-normalise and remove control / zero-width characters (except ZWJ in Indic)."""
    return _CTRL_RE.sub(" ", unicodedata.normalize("NFKC", s))


def has_indic(s: str) -> bool:
    return bool(_INDIC_RE.search(s))


def strip_accents(s: str) -> str:
    """Remove Latin diacritics (é -> e). Leaves Indic combining marks untouched."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.combining(ch) and not ("ऀ" <= ch <= "ൿ"):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def basic_clean(s: str | None) -> str:
    """Upper-case, punctuation -> space, '&' kept as its own token, whitespace collapsed.

    >>> basic_clean("  Sharma Traders (P.) Ltd.,  ")
    'SHARMA TRADERS P LTD'
    """
    if not s:
        return ""
    s = strip_accents(nfkc(s)).upper()
    s = s.replace("&", " & ")
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()
