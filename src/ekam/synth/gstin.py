"""Clearly-synthetic GSTINs.

Format and checksum are valid (so validators and features behave realistically), but:

* the 10-char PAN block always starts with the reserved prefix (default ``ZZZ``),
* the 4th PAN char is 'C' (company), the 5th the name's initial; the remaining PAN
  chars and the entity number encode a scrambled per-entity serial, so two different
  entities never share a GSTIN (as in reality) - never derived from any real registry.

:func:`is_synthetic_gstin` lets tests assert that no generated value escapes the scheme.
"""

from __future__ import annotations

import random
import string

from ekam.normalize import resources
from ekam.normalize.identifiers import gstin_check_char, is_valid_gstin


def state_gst_code(state: str | None) -> str:
    spec = resources.states().get(state or "")
    return spec["gst"] if spec else "97"  # 97 = "Other Territory" in the GST scheme


# Serial space: PAN digits (10^4) x PAN check letter (26) x GSTIN entity number (35).
_ENTITY_NO = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SERIAL_SPACE = 10_000 * 26 * len(_ENTITY_NO)  # 9.1M distinct values
_MULT = 7_919  # prime, coprime with SERIAL_SPACE -> idx -> serial is a bijection


def entity_serial(entity_index: int, seed: int) -> int:
    """Bijective scramble of an entity index into [0, SERIAL_SPACE). Distinct entities get
    distinct serials (as real GSTINs are unique per PAN), but serials are not sequential."""
    if not 0 <= entity_index < SERIAL_SPACE:
        raise ValueError("entity index outside the synthetic GSTIN serial space")
    return (entity_index * _MULT + seed * 104_729) % SERIAL_SPACE


def make_synthetic_gstin(state: str | None, name: str | None, serial: int,
                         prefix: str = "ZZZ") -> str:
    """15-char GSTIN with valid checksum, reserved PAN prefix, unique per ``serial``."""
    initial = next((c for c in (name or "").upper() if c in string.ascii_uppercase), "X")
    serial, digits = divmod(serial, 10_000)
    serial, letter = divmod(serial, 26)
    entity_no = _ENTITY_NO[serial % len(_ENTITY_NO)]
    pan = f"{prefix}C{initial}{digits:04d}{string.ascii_uppercase[letter]}"
    first14 = state_gst_code(state) + pan + entity_no + "Z"
    return first14 + gstin_check_char(first14)


def is_synthetic_gstin(value: str, prefix: str = "ZZZ") -> bool:
    return is_valid_gstin(value) and value[2 : 2 + len(prefix)] == prefix


def corrupt_identifier(value: str, rng: random.Random) -> str:
    """One-character keying error (substitution or adjacent transposition)."""
    s = list(value)
    i = rng.randrange(len(s))
    if rng.random() < 0.6:
        pool = string.digits if s[i].isdigit() else string.ascii_uppercase
        s[i] = rng.choice([c for c in pool if c != s[i]])
    elif i + 1 < len(s) and s[i] != s[i + 1]:
        s[i], s[i + 1] = s[i + 1], s[i]
    else:
        s[i] = "0" if s[i] != "0" else "O"
    return "".join(s)
