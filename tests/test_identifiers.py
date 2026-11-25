import random
import string

from hypothesis import given, settings
from hypothesis import strategies as st

from ekam.normalize.identifiers import (
    gstin_check_char,
    is_valid_gstin,
    normalize_cin,
    normalize_gstin,
)
from ekam.synth.gstin import (
    SERIAL_SPACE,
    corrupt_identifier,
    entity_serial,
    is_synthetic_gstin,
    make_synthetic_gstin,
)

STATES = ["Goa", "Karnataka", "Gujarat", "Rajasthan", "Uttar Pradesh", None]


@given(seed=st.integers(0, 10**9), state=st.sampled_from(STATES),
       name=st.text(alphabet=string.ascii_letters + " ", max_size=30))
@settings(max_examples=300, deadline=None)
def test_synthetic_gstin_is_valid_and_marked_synthetic(seed, state, name):
    g = make_synthetic_gstin(state, name, seed % SERIAL_SPACE)
    assert len(g) == 15
    assert is_valid_gstin(g)
    assert is_synthetic_gstin(g)
    assert g[2:5] == "ZZZ" and g[5] == "C"


def test_gstin_state_code():
    assert make_synthetic_gstin("Karnataka", "Acme", 1).startswith("29")
    assert make_synthetic_gstin("Goa", "Acme", 1).startswith("30")


@given(seed=st.integers(0, 10**9))
@settings(max_examples=300, deadline=None)
def test_single_char_substitution_breaks_checksum(seed):
    rng = random.Random(seed)
    g = make_synthetic_gstin("Gujarat", "Patel", rng.randrange(SERIAL_SPACE))
    i = rng.randrange(14)
    pool = string.digits if g[i].isdigit() else string.ascii_uppercase
    bad = g[:i] + rng.choice([c for c in pool if c != g[i]]) + g[i + 1 :]
    assert gstin_check_char(bad[:14]) != g[14]


@given(seed=st.integers(0, 10**9))
@settings(max_examples=200, deadline=None)
def test_corruption_changes_identifier(seed):
    rng = random.Random(seed)
    g = make_synthetic_gstin("Goa", "X", rng.randrange(SERIAL_SPACE))
    assert corrupt_identifier(g, rng) != g


def test_normalize_gstin_strips_and_validates():
    g = make_synthetic_gstin("Goa", "Acme", 3)
    assert normalize_gstin(f" {g[:5]}-{g[5:].lower()} ") == g
    assert normalize_gstin(g[:-1] + ("0" if g[-1] != "0" else "1")) is None
    assert normalize_gstin(None) is None


def test_normalize_cin():
    assert normalize_cin("u72200ka2010ptc052345") == "U72200KA2010PTC052345"
    assert normalize_cin("U0801.KA1994PTC016106") is None
    assert normalize_cin("CORP") is None


def test_gstins_unique_across_entities():
    """Same state + same initial must still never collide (real GSTINs are unique)."""
    seen = {make_synthetic_gstin("Karnataka", "Sharma", entity_serial(i, 13))
            for i in range(300_000)}
    assert len(seen) == 300_000
