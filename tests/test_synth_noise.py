"""Each noise type does what docs/NOISE.md says, and known variants round-trip through the
normaliser (held-out ones are allowed not to)."""

import random

import pytest

from ekam.normalize import normalize_name, normalize_pin, parse_address
from ekam.synth.noise import address as an
from ekam.synth.noise import names as nn
from ekam.synth.noise.chars import ocr, typo

NAMES = ["SHARMA TRADERS PRIVATE LIMITED", "SHREE LAXMI ENTERPRISES LIMITED",
         "GLOBAL TECHNOLOGIES PVT LTD", "PATEL INDUSTRIES (INDIA) LIMITED", "KUMAR BROTHERS"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", range(20))
def test_suffix_variant_preserves_identity(name, seed):
    out = nn.suffix_variant(name, random.Random(seed))
    a, b = normalize_name(name), normalize_name(out)
    assert (a.base, a.suffix) == (b.base, b.suffix), out


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", range(20))
def test_known_abbreviations_round_trip(name, seed):
    out = nn.abbrev(name, random.Random(seed))
    assert normalize_name(out).base == normalize_name(name).base, out


def test_heldout_abbreviations_are_not_known_to_normaliser():
    out = nn.abbrev_heldout("SHARMA TRADERS PRIVATE LIMITED", random.Random(0))
    assert out != "SHARMA TRADERS PRIVATE LIMITED"
    assert normalize_name(out).base != "SHARMA TRADERS"


@pytest.mark.parametrize("seed", range(30))
def test_devanagari_round_trip_of_suffix_and_business_words(seed):
    out = nn.devanagari("SHARMA TRADERS PRIVATE LIMITED", random.Random(seed))
    assert any("ऀ" <= ch <= "ॿ" for ch in out)
    n = normalize_name(out)
    assert n.suffix == "PVT_LTD"
    assert n.base == "SHARMA TRADERS"


def test_suffix_drop_and_noop_without_suffix():
    assert nn.suffix_drop("SHARMA TRADERS PVT LTD", random.Random(0)) == "SHARMA TRADERS"
    assert nn.suffix_drop("KUMAR BROTHERS", random.Random(0)) == "KUMAR BROTHERS"


@pytest.mark.parametrize("seed", range(50))
def test_typo_and_ocr_small_edits(seed):
    rng = random.Random(seed)
    s = "SHARMA TRADERS PRIVATE LIMITED"
    t = typo(s, rng, max_edits=1)
    assert abs(len(t) - len(s)) <= 1
    o = ocr(s, rng)
    assert o != s or not any(a in s for a in "OISBGZ")


@pytest.mark.parametrize("seed", range(30))
def test_translit_variant_leaves_business_words(seed):
    out = nn.translit_variant("SHARMA TRADERS PRIVATE LIMITED", random.Random(seed))
    assert out.endswith("TRADERS PRIVATE LIMITED")


@pytest.mark.parametrize("seed", range(50))
def test_pin_wrong_is_valid_but_different(seed):
    out = an.pin_wrong("560010", random.Random(seed), state="Karnataka")
    assert out != "560010" and normalize_pin(out) == out


@pytest.mark.parametrize("seed", range(10))
def test_city_alias_maps_back(seed):
    alias = an.city_alias("Bengaluru", random.Random(seed))
    assert parse_address(f"12 Road {alias}").city == "Bengaluru"


def test_noise_functions_are_pure_given_rng():
    for fn in [nn.suffix_variant, nn.abbrev, nn.translit_variant, nn.devanagari, nn.case,
               nn.token_drop, nn.spacing]:
        assert fn(NAMES[0], random.Random(5)) == fn(NAMES[0], random.Random(5))
