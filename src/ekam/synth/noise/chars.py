"""Character-level noise: keyboard typos and OCR confusions."""

from __future__ import annotations

import random

_ROWS = ["1234567890", "QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
_NEIGH: dict[str, str] = {}
for r, row in enumerate(_ROWS):
    for c, ch in enumerate(row):
        n = []
        for dr in (-1, 0, 1):
            rr = r + dr
            if 0 <= rr < len(_ROWS):
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if (dr, dc) != (0, 0) and 0 <= cc < len(_ROWS[rr]):
                        n.append(_ROWS[rr][cc])
        _NEIGH[ch] = "".join(n)

# OCR confusions in both directions (upper-case text dominates Indian vendor data).
OCR_SUBS: list[tuple[str, str]] = [
    ("O", "0"), ("0", "O"), ("I", "1"), ("1", "I"), ("L", "1"), ("I", "L"), ("S", "5"),
    ("5", "S"), ("B", "8"), ("8", "B"), ("G", "6"), ("Z", "2"), ("E", "F"), ("RN", "M"),
    ("M", "RN"), ("CL", "D"), ("D", "CL"), ("VV", "W"), ("W", "VV"), ("H", "N"), ("U", "V"),
    ("a", "o"), ("e", "c"), ("l", "1"), ("rn", "m"), ("m", "rn"), ("o", "0"),
]


def _case_like(src: str, ch: str) -> str:
    return ch.lower() if src.islower() else ch


def typo(value: str, rng: random.Random, max_edits: int = 1) -> str:
    """1..max_edits keyboard edits (substitute / delete / insert / transpose) on letters."""
    s = list(value)
    letters = [i for i, ch in enumerate(s) if ch.isalnum()]
    if len(letters) < 4:
        return value
    for _ in range(rng.randint(1, max_edits)):
        letters = [i for i, ch in enumerate(s) if ch.isalnum()]
        if len(letters) < 3:
            break
        i = rng.choice(letters[1:])  # leave the first character alone (people rarely miss it)
        op = rng.random()
        up = s[i].upper()
        if op < 0.4 and up in _NEIGH:
            s[i] = _case_like(s[i], rng.choice(_NEIGH[up]))
        elif op < 0.65:
            del s[i]
        elif op < 0.85 and up in _NEIGH:
            s.insert(i, _case_like(s[i], rng.choice(_NEIGH[up])))
        elif i + 1 < len(s) and s[i + 1].isalnum():
            s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)


def ocr(value: str, rng: random.Random, n_max: int = 3) -> str:
    """Apply 1..n_max OCR confusions at random applicable positions."""
    s = value
    for _ in range(rng.randint(1, n_max)):
        cands = [(a, b) for a, b in OCR_SUBS if a in s]
        if not cands:
            break
        a, b = rng.choice(cands)
        positions = [i for i in range(len(s)) if s.startswith(a, i)]
        i = rng.choice(positions)
        s = s[:i] + b + s[i + len(a) :]
    return s
