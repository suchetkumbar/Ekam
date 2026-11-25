"""Indian address parsing: PIN code, state, city/district and the remaining street part."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ekam.normalize import resources
from ekam.normalize.text import basic_clean
from ekam.normalize.translit import to_latin

_PIN_RE = re.compile(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)")

# Street-type abbreviations: expanded so "MG RD" and "M.G. ROAD" agree.
STREET_ABBREV = {
    "RD": "ROAD", "ST": "STREET", "MKT": "MARKET", "NGR": "NAGAR", "NR": "NEAR",
    "OPP": "OPPOSITE", "BLDG": "BUILDING", "FLR": "FLOOR", "SEC": "SECTOR", "SECT": "SECTOR",
    "INDL": "INDUSTRIAL", "IND": "INDUSTRIAL", "ESTT": "ESTATE", "EST": "ESTATE",
    "APT": "APARTMENT", "APTS": "APARTMENTS", "CHS": "CHS", "COMPLX": "COMPLEX",
    "CMPLX": "COMPLEX", "CROSS": "CROSS", "MN": "MAIN", "NO": "NO", "PO": "POST",
    "DIST": "DISTRICT", "DT": "DISTRICT", "TAL": "TALUKA", "TQ": "TALUKA", "VILL": "VILLAGE",
    "GIDC": "GIDC", "MIDC": "MIDC", "KIADB": "KIADB", "PH": "PHASE", "PHS": "PHASE",
}


@lru_cache(maxsize=1)
def _state_lookup() -> tuple[list[tuple[tuple[str, ...], str]], dict[str, str]]:
    """(multi-token patterns longest-first, two-letter codes) -> canonical state."""
    pats, codes = [], {}
    for name, spec in resources.states().items():
        for v in [name, *spec.get("alias", [])]:
            toks = tuple(basic_clean(v).split())
            if not toks:
                continue
            if len(toks) == 1 and len(toks[0]) <= 2:
                codes[toks[0]] = name
            elif len(toks) >= 2 and all(len(t) == 1 for t in toks) and len(toks) <= 2:
                codes["".join(toks)] = name  # "U P" -> "UP"
                pats.append((toks, name))
            else:
                pats.append((toks, name))
    pats.sort(key=lambda p: -len(p[0]))
    return pats, codes


@lru_cache(maxsize=1)
def _city_lookup() -> list[tuple[tuple[str, ...], str]]:
    pats = []
    for canon, spec in resources.cities().items():
        for v in [canon, *spec.get("alias", [])]:
            toks = tuple(basic_clean(v).split())
            if toks:
                pats.append((toks, canon))
    pats.sort(key=lambda p: -len(p[0]))
    return pats


@lru_cache(maxsize=1)
def _pin_prefixes() -> list[tuple[str, str]]:
    out = [(p, name) for name, spec in resources.states().items() for p in spec.get("pin", [])]
    out.sort(key=lambda x: -len(x[0]))
    return out


def normalize_pin(value: str | int | None) -> str | None:
    """Return a valid 6-digit PIN or None. Accepts '560 001', 560001, '560001.0'."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value).split(".")[0])
    return digits if len(digits) == 6 and digits[0] != "0" else None


def states_for_pin(pin: str | None) -> set[str]:
    if not pin:
        return set()
    hits = {name for p, name in _pin_prefixes() if pin.startswith(p)}
    return hits


def pin_state_consistent(pin: str | None, state: str | None) -> bool | None:
    """True/False if both known; None if either is missing or the prefix table is silent."""
    if not pin or not state:
        return None
    cands = states_for_pin(pin)
    return None if not cands else state in cands


def canonical_state(text: str | None) -> str | None:
    if not text:
        return None
    toks = basic_clean(to_latin(text)).split()
    found = _find(toks, _state_lookup()[0])
    if found:
        return found[1]
    joined = "".join(toks)
    return _state_lookup()[1].get(joined)


def canonical_city(text: str | None) -> str | None:
    if not text:
        return None
    toks = basic_clean(to_latin(text)).split()
    found = _find(toks, _city_lookup())
    return found[1] if found else " ".join(toks).title() or None


def _find(tokens: list[str], pats) -> tuple[int, str, int] | None:
    """Last (rightmost) occurrence of any pattern: (start, canonical, length)."""
    best = None
    for pat, canon in pats:
        n = len(pat)
        for i in range(len(tokens) - n, -1, -1):
            if tuple(tokens[i : i + n]) == pat:
                if best is None or i > best[0] or (i == best[0] and n > best[2]):
                    best = (i, canon, n)
                break
    return best


@dataclass(frozen=True)
class ParsedAddress:
    raw: str
    clean: str
    street: str
    city: str | None
    state: str | None
    pin: str | None

    @property
    def pin_state_ok(self) -> bool | None:
        return pin_state_consistent(self.pin, self.state)


def expand_street(tokens: list[str]) -> list[str]:
    return [STREET_ABBREV.get(t, t) for t in tokens]


def parse_address(raw: str | None, state_hint: str | None = None) -> ParsedAddress:
    """Parse a free-text Indian address.

    Strategy: take the right-most valid PIN; drop 'INDIA'; find the right-most state
    mention (full names and multi-letter aliases anywhere, two-letter codes only as the
    final token); find a gazetteer city, else fall back to the token just before the state
    (the dominant MCA pattern ``... <CITY> <State> INDIA <PIN>``).
    """
    raw = raw or ""
    text = to_latin(raw)
    pins = list(_PIN_RE.finditer(text))
    pin = None
    if pins:
        m = pins[-1]
        pin = m.group(1) + m.group(2)
        text = text[: m.start()] + " " + text[m.end() :]
    toks = [t for t in basic_clean(text).split() if t != "INDIA"]

    state, state_pos = None, None
    hit = _find(toks, _state_lookup()[0])
    if hit:
        state, state_pos = hit[1], hit[0]
        toks = toks[: hit[0]] + toks[hit[0] + hit[2] :]
        # MCA often repeats the state ("ALLAHABAD UTTAR PRADESH Uttar Pradesh"): drop every copy
        for pat in (p for p, canon in _state_lookup()[0] if canon == state):
            while (dup := _find(toks, [(pat, state)])) is not None:
                toks = toks[: dup[0]] + toks[dup[0] + dup[2] :]
                if dup[0] < state_pos:
                    state_pos -= dup[2]
    elif toks and toks[-1] in _state_lookup()[1]:
        state, state_pos = _state_lookup()[1][toks[-1]], len(toks) - 1
        toks = toks[:-1]
    if state is None and state_hint:
        state = canonical_state(state_hint)

    city = None
    chit = _find(toks, _city_lookup())
    if chit:
        city = chit[1]
        for pat in [chit_pat for chit_pat, canon in _city_lookup() if canon == city]:
            while (dup := _find(toks, [(pat, city)])) is not None:
                toks = toks[: dup[0]] + toks[dup[0] + dup[2] :]
    elif state_pos is not None and 0 < state_pos <= len(toks):
        cand = toks[state_pos - 1]
        if cand.isalpha() and len(cand) >= 3 and cand not in STREET_ABBREV.values():
            city = cand.title()
            toks = toks[: state_pos - 1] + toks[state_pos:]

    street = " ".join(expand_street(toks))
    return ParsedAddress(
        raw=raw, clean=basic_clean(text), street=street, city=city, state=state, pin=pin
    )
