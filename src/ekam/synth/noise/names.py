"""Business-name noise: legal suffixes, abbreviations, romanisation, script, token ops."""

from __future__ import annotations

import random
import re
from functools import lru_cache

from ekam.normalize import resources
from ekam.normalize.names import split_legal_suffix
from ekam.normalize.text import basic_clean
from ekam.normalize.translit import latin_token_to_devanagari
from ekam.synth.noise.chars import ocr, typo


@lru_cache(maxsize=1)
def _abbrev_forward(heldout: bool) -> dict[str, list[str]]:
    key = "heldout" if heldout else "known"
    out = {}
    for full, spec in resources.abbreviations().items():
        variants = [v for v in spec.get(key) or [] if v and v != "&"]
        if full == "AND":
            variants = ["&"]
        if variants:
            out[full] = variants
    return out


def _split(name: str) -> tuple[list[str], str | None, list[str]]:
    """(base tokens, suffix code, original suffix tokens) using the normaliser's matcher."""
    toks = basic_clean(name).split()
    base, code = split_legal_suffix(toks)
    return base, code, toks[len(base) :]


def suffix_variant(name: str, rng: random.Random) -> str:
    base, code, _ = _split(name)
    if code is None:
        return name
    variants = resources.legal_suffixes()[code]["render"]
    return " ".join(base) + " " + rng.choice(variants)


def suffix_drop(name: str, rng: random.Random) -> str:
    base, code, _ = _split(name)
    return " ".join(base) if code is not None else name


def _abbrev(name: str, rng: random.Random, heldout: bool) -> str:
    fwd = _abbrev_forward(heldout)
    toks = name.split()
    idx = [i for i, t in enumerate(toks) if basic_clean(t) in fwd]
    if not idx:
        return name
    for i in rng.sample(idx, k=rng.randint(1, len(idx))):
        toks[i] = rng.choice(fwd[basic_clean(toks[i])])
    return " ".join(toks)


def abbrev(name: str, rng: random.Random) -> str:
    return _abbrev(name, rng, heldout=False)


def abbrev_heldout(name: str, rng: random.Random) -> str:
    return _abbrev(name, rng, heldout=True)


def case(name: str, rng: random.Random) -> str:
    style = rng.choice(["title", "lower", "title", "mixed"])
    if style == "title":
        return name.title()
    if style == "lower":
        return name.lower()
    return " ".join(t.title() if rng.random() < 0.5 else t for t in name.split())


def prefix_ms(name: str, rng: random.Random) -> str:
    return rng.choice(["M/s ", "M/S. ", "M/s. ", "Messrs "]) + name


def token_drop(name: str, rng: random.Random) -> str:
    base, _, suffix = _split(name)
    if len(base) < 3:
        return name
    i = rng.randrange(1, len(base))  # never drop the first, most identifying token
    return " ".join(base[:i] + base[i + 1 :] + suffix)


def token_swap(name: str, rng: random.Random) -> str:
    base, _, suffix = _split(name)
    if len(base) < 2:
        return name
    i = rng.randrange(len(base) - 1)
    base[i], base[i + 1] = base[i + 1], base[i]
    return " ".join(base + suffix)


def spacing(name: str, rng: random.Random) -> str:
    """Glue two tokens together or split one (common in hand-keyed / OCR data)."""
    toks = name.split()
    if len(toks) >= 2 and rng.random() < 0.6:
        i = rng.randrange(len(toks) - 1)
        return " ".join(toks[:i] + [toks[i] + toks[i + 1]] + toks[i + 2 :])
    long_ = [i for i, t in enumerate(toks) if len(t) >= 7]
    if not long_:
        return name
    i = rng.choice(long_)
    cut = rng.randrange(3, len(toks[i]) - 2)
    return " ".join(toks[:i] + [toks[i][:cut], toks[i][cut:]] + toks[i + 1 :])


# Variant rewrites for Indian romanisation; each is (pattern, replacement) on lower case.
_ROMAN_VARIANTS = [
    (r"^sh", "s"), (r"^s(?!h)", "sh"), (r"ee", "i"), (r"(?<=[^aeiou])i$", "ee"), (r"aa", "a"),
    (r"(?<=[^aeiou])a(?=[^aeiou])", "aa"), (r"w", "v"), (r"v", "w"), (r"ksh", "x"), (r"x", "ksh"),
    (r"ph", "f"), (r"oo", "u"), (r"u", "oo"), (r"(?<=[^aeiou])a$", ""), (r"(?<=[^aeiou])$", "a"),
    (r"([^aeiou])\1", r"\1"), (r"(?<=[aeiou])([gklmnprst])(?=[aeiou])", r"\1\1"),
    (r"^shri$", "shree"), (r"^shri$", "sri"), (r"^sri$", "shree"), (r"^shree$", "sri"),
    (r"th", "t"), (r"(?<=[aeiou])t(?=[aeiou])", "th"), (r"dh", "d"), (r"bh", "b"),
    (r"au", "ou"), (r"ou", "au"), (r"y", "i"),
]


@lru_cache(maxsize=1)
def _english_words() -> set[str]:
    words = set(resources.hindi_lexicon_generator()) | set(resources.abbreviations())
    for spec in resources.legal_suffixes().values():
        for v in spec["match"]:
            words.update(basic_clean(v).split())
    return words


def translit_variant(name: str, rng: random.Random) -> str:
    """Re-spell 1-2 Indian (non-English-business-word) tokens with romanisation variants."""
    toks = name.split()
    eng = _english_words()
    idx = [i for i, t in enumerate(toks) if t.isalpha() and len(t) >= 3 and t.upper() not in eng]
    if not idx:
        return name
    for i in rng.sample(idx, k=min(len(idx), rng.randint(1, 2))):
        tok = toks[i]
        low = tok.lower()
        rules = [(p, r) for p, r in _ROMAN_VARIANTS if re.search(p, low)]
        rng.shuffle(rules)
        for p, r in rules[: rng.randint(1, 2)]:
            new = re.sub(p, r, low, count=1)
            if len(new) >= 2:
                low = new
        toks[i] = low.upper() if tok.isupper() else low.title() if tok.istitle() else low
    return " ".join(toks)


def devanagari(name: str, rng: random.Random) -> str:
    """Render the whole name in Devanagari (lexicon for business words, heuristic otherwise)."""
    base, code, _ = _split(name)
    lex = resources.hindi_lexicon_generator()
    out = []
    for t in base:
        out.append(rng.choice(lex[t]) if t in lex else latin_token_to_devanagari(t))
    if code == "PVT_LTD":
        out += rng.choice([["प्राइवेट", "लिमिटेड"], ["प्रा.", "लि."], ["प्रायवेट", "लिमिटेड"]])
    elif code == "OPC_PVT_LTD":
        out += ["ओपीसी", "प्राइवेट", "लिमिटेड"]
    elif code == "LTD":
        out += [rng.choice(["लिमिटेड", "लि."])]
    elif code == "LLP":
        out += ["एलएलपी"]
    elif code == "CO":
        out += ["एंड", "कंपनी"]
    return " ".join(out)


def name_typo(name: str, rng: random.Random, max_edits: int = 1) -> str:
    return typo(name, rng, max_edits=max_edits)


def name_ocr(name: str, rng: random.Random) -> str:
    return ocr(name, rng)
