"""Transliteration (Devanagari <-> Latin) and Indian-romanisation folding.

Two different jobs live here:

* :func:`to_latin` turns Devanagari text into upper-case Latin. Common English loanwords
  written in Devanagari (प्राइवेट, लिमिटेड, ट्रेडर्स ...) are mapped back through the curated
  lexicon; everything else goes through ITRANS with Hindi schwa deletion.
* :func:`fold` collapses spelling variation that Indian romanisation produces
  (Shree/Shri/Sri, Lakshmi/Laxmi, Agarwal/Agrawal/Aggarwal...) into one key. Fold output
  is for *matching keys only*; it is never shown to a user.
"""

from __future__ import annotations

import re
from functools import lru_cache

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from ekam.normalize import resources
from ekam.normalize.text import has_indic

_DEVANAGARI_TOKEN = re.compile(r"[ऀ-ॿ]+\.?")
_LOAN_VOWELS = str.maketrans({"ॉ": "ो", "ऑ": "ओ",   # ॉ->ो, ऑ->ओ
                              "ॅ": "े", "ऍ": "ए"})  # ॅ->े, ऍ->ए


@lru_cache(maxsize=1)
def _reverse_lexicon() -> dict[str, str]:
    rev = {}
    for english, spellings in resources.hindi_lexicon().items():
        for sp in spellings:
            rev.setdefault(sp, english)
            rev.setdefault(sp.rstrip("."), english)
    return rev


def _itrans_to_plain(itrans: str) -> str:
    # Hindi schwa deletion: a word-final short 'a' after a consonant is not pronounced
    # (राम = rAma -> "ram"); a long 'A' is kept (शर्मा = sharmA -> "sharma").
    s = re.sub(r"(?<=[^aeiouAEIOU])a$", "", itrans)
    s = s.replace("RRi", "ri").replace("Ri", "ri").replace("~N", "n").replace("~n", "n")
    s = s.replace(".N", "n").replace(".n", "n").replace(".h", "").replace("M", "n")
    s = s.replace("H", "h")
    return s.lower()


@lru_cache(maxsize=200_000)
def devanagari_token_to_latin(token: str) -> str:
    rev = _reverse_lexicon()
    if token in rev:
        return rev[token]
    if token.rstrip(".") in rev:
        return rev[token.rstrip(".")]
    # Signs for English loan vowels (ऑ/ॉ, ऍ/ॅ) have no ITRANS mapping: approximate first.
    token = token.translate(_LOAN_VOWELS)
    itrans = transliterate(token.rstrip("."), sanscript.DEVANAGARI, sanscript.ITRANS)
    itrans = re.sub(r"[^\x00-\x7f]", "", itrans)  # drop anything still unmapped
    return _itrans_to_plain(itrans).upper()


def to_latin(s: str) -> str:
    """Transliterate any Devanagari tokens in ``s`` to Latin, leaving other text as is."""
    if not s or not has_indic(s):
        return s
    out = _DEVANAGARI_TOKEN.sub(lambda m: " " + devanagari_token_to_latin(m.group(0)) + " ", s)
    return re.sub(r"\s+", " ", out).strip()


# ---- Latin -> Devanagari (used by the noise generator) ---------------------------------

_LAT2ITRANS = [
    ("ksh", "kSh"), ("x", "kSh"), ("aa", "A"), ("ee", "I"), ("ii", "I"), ("oo", "U"),
    ("uu", "U"), ("w", "v"), ("q", "k"), ("ck", "k"),
]


def latin_token_to_devanagari(token: str) -> str:
    """Heuristic English-spelled Indian word -> Devanagari (for synthetic noise only).

    The heuristic is deliberately simple (ITRANS with a few spelling rules); its output is
    plausible but not always the standard Hindi spelling. That's acceptable for noise
    generation and is stated in docs/NOISE.md.
    """
    lex = resources.hindi_lexicon_generator()
    up = token.upper()
    if up in lex:
        return lex[up][0]
    t = token.lower()
    if not t.isalpha():
        return token
    for a, b in _LAT2ITRANS:
        t = t.replace(a, b)
    t = re.sub(r"c(?!h)", "k", t)
    # Romanised Hindi writes both अ and आ as "a", and ई as "i" at word end. Heuristics:
    # a closed final syllable usually has a long vowel (Ram, Kumar, Agarwal), and a final
    # "i" is usually long (Laxmi, Devi).
    t = re.sub(r"(?<=[^aeiouAIU])a(?=[^aeiouAIU]+$)", "A", t) if len(t) > 2 else t
    t = re.sub(r"i$", "I", t)
    if re.search(r"[bcdfghjklmnpqrstvyzSh]$", t) and not t.endswith(("A", "I", "U")):
        t = t + "a"  # inherent vowel, avoids a dangling virama
    elif t.endswith("a") and len(t) > 2:
        t = t[:-1] + "A"  # Sharma, Gupta, Verma -> long final A
    return transliterate(t, sanscript.ITRANS, sanscript.DEVANAGARI)


# ---- Folding -------------------------------------------------------------------------

_FOLD_RULES = [
    ("ksh", "x"), ("ks", "x"), ("sh", "s"), ("ph", "f"), ("w", "v"), ("z", "j"), ("q", "k"),
    ("ck", "k"), ("chh", "ch"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
    ("gh", "g"), ("jh", "j"), ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"),
    ("uu", "u"), ("ou", "o"), ("au", "o"), ("ey", "e"), ("y", "i"),
]
_DOUBLE = re.compile(r"(.)\1+")


@lru_cache(maxsize=500_000)
def fold_token(token: str) -> str:
    """Fold one Latin token to a romanisation-invariant key.

    >>> [fold_token(t) for t in ["SHREE", "SHRI", "SRI"]]
    ['SRI', 'SRI', 'SRI']
    >>> fold_token("LAKSHMI") == fold_token("LAXMI")
    True
    """
    t = token.lower()
    if not t.isalpha():
        return token.upper()
    t = re.sub(r"c(?!h)", "k", t)
    for a, b in _FOLD_RULES:
        t = t.replace(a, b)
    t = _DOUBLE.sub(r"\1", t)
    if len(t) > 3 and t.endswith("a"):
        t = t[:-1]
    return t.upper()


def fold(tokens: list[str]) -> list[str]:
    return [fold_token(t) for t in tokens]
