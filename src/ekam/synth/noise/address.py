"""Address, city, state and PIN-code noise."""

from __future__ import annotations

import random
from functools import lru_cache

from ekam.normalize import resources
from ekam.normalize.address import STREET_ABBREV
from ekam.synth.noise.chars import ocr, typo

# full word -> abbreviations actually seen (inverse of the normaliser's expansion table)
_ABBREV_OUT: dict[str, list[str]] = {}
for short, full in STREET_ABBREV.items():
    if short != full:
        _ABBREV_OUT.setdefault(full, []).append(short)
_ABBREV_OUT.update({"ROAD": ["RD", "RD.", "Rd"], "NEAR": ["NR", "NR.", "Nr."],
                    "OPPOSITE": ["OPP", "OPP.", "Opp."]})


def street_abbrev(street: str, rng: random.Random) -> str:
    toks = street.split()
    idx = [i for i, t in enumerate(toks) if t.upper() in _ABBREV_OUT]
    if not idx:
        return street
    for i in idx:
        if rng.random() < 0.7:
            toks[i] = rng.choice(_ABBREV_OUT[toks[i].upper()])
    return " ".join(toks)


def street_reorder(street: str, rng: random.Random) -> str:
    """Move a block of tokens (e.g. 'NO 12' or the locality) to the other end."""
    toks = street.split()
    if len(toks) < 4:
        return street
    cut = rng.randrange(1, len(toks) - 1)
    parts = [" ".join(toks[cut:]), " ".join(toks[:cut])]
    return ", ".join(parts)


def street_truncate(street: str, rng: random.Random) -> str:
    toks = street.split()
    if len(toks) < 3:
        return street
    keep = max(1, int(len(toks) * rng.uniform(0.4, 0.75)))
    return " ".join(toks[:keep])


def street_typo(street: str, rng: random.Random, max_edits: int = 1) -> str:
    return typo(street, rng, max_edits=max_edits)


def street_ocr(street: str, rng: random.Random) -> str:
    return ocr(street, rng)


def _city_entry(city: str | None):
    return resources.cities().get(city) if city else None


def city_alias(city: str, rng: random.Random) -> str:
    e = _city_entry(city)
    return rng.choice(e["alias"]) if e and e.get("alias") else city


def city_alias_heldout(city: str, rng: random.Random) -> str:
    e = _city_entry(city)
    return rng.choice(e["heldout"]) if e and e.get("heldout") else city


def state_variant(state: str, rng: random.Random) -> str:
    spec = resources.states().get(state)
    if not spec or not spec.get("alias"):
        return state
    return rng.choice(spec["alias"] + [state.upper(), state.lower()])


@lru_cache(maxsize=1)
def _same_state_prefixes() -> dict[str, list[str]]:
    return {name: spec.get("pin", []) for name, spec in resources.states().items()}


def pin_wrong(pin: str, rng: random.Random, state: str | None = None) -> str:
    """Corrupt a PIN realistically: last-digit slip, digit transposition, or another
    PIN from the same state's postal circle."""
    op = rng.random()
    d = list(pin)
    if op < 0.45:
        i = rng.choice([4, 5])
        d[i] = str((int(d[i]) + rng.choice([1, 2, 3, 7, 8, 9])) % 10)
    elif op < 0.75:
        i = rng.randrange(1, 5)
        d[i], d[i + 1] = d[i + 1], d[i]
        if d == list(pin):
            d[5] = str((int(d[5]) + 1) % 10)
    else:
        prefixes = _same_state_prefixes().get(state or "", []) or [pin[:2]]
        p = rng.choice(prefixes)
        d = list(p + "".join(str(rng.randrange(10)) for _ in range(6 - len(p))))
    new = "".join(d)
    if new == pin or new[0] == "0":
        new = pin[:5] + str((int(pin[5]) + 5) % 10)
    return new
