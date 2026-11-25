"""CIN and GSTIN format/checksum validation.

GSTIN layout (15 chars): 2-digit state code + 10-char PAN + entity number [1-9A-Z]
+ 'Z' (default) + check character (mod-36 Luhn-style; see :func:`gstin_check_char`).
Ekam only ever *generates* clearly synthetic GSTINs (see ekam.synth.gstin).
"""

from __future__ import annotations

import re

_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_CIN_RE = re.compile(r"^[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}$")
_STRIP = re.compile(r"[^0-9A-Za-z]")


def gstin_check_char(first14: str) -> str:
    """Check character for the first 14 GSTIN characters.

    Each char's value (0-9, A=10 .. Z=35) is multiplied by 1 or 2 alternately; the product
    is split into quotient and remainder by 36 and both are summed; the check value is
    ``(36 - sum % 36) % 36``.
    """
    if len(first14) != 14:
        raise ValueError("need exactly 14 characters")
    total = 0
    for i, ch in enumerate(first14.upper()):
        v = _ALNUM.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return _ALNUM[(36 - total % 36) % 36]


def is_valid_gstin(value: str | None) -> bool:
    if not value:
        return False
    v = value.upper()
    return bool(_GSTIN_RE.match(v)) and gstin_check_char(v[:14]) == v[14]


def normalize_gstin(value: str | None) -> str | None:
    """Upper-case, strip separators; return it only if format AND checksum are valid."""
    if not value:
        return None
    v = _STRIP.sub("", str(value)).upper()
    return v if is_valid_gstin(v) else None


def is_valid_cin(value: str | None) -> bool:
    return bool(value) and bool(_CIN_RE.match(value.upper()))


def normalize_cin(value: str | None) -> str | None:
    if not value:
        return None
    v = _STRIP.sub("", str(value)).upper()
    return v if _CIN_RE.match(v) else None
