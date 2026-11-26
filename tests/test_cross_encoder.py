"""M3: serialisation, augmentation, cascade threshold, active-learning loop, CE smoke test."""

import random

import numpy as np
import polars as pl
import pytest

from ekam.match.active import run_curve, select
from ekam.match.augment import OPS, augment_pair, augment_record, parse, render
from ekam.match.cascade import recall_threshold
from ekam.match.serialize import serialize_records

DF = pl.DataFrame({
    "name_clean": ["SHARMA TRADERS PVT LTD", "LAXMI ENTERPRISES"],
    "name_raw": ["शर्मा ट्रेडर्स प्रा. लि.", "Laxmi Ent."],
    "street": ["12 MG ROAD", None], "city_norm": ["Jaipur", "Panaji"],
    "state_norm": ["Rajasthan", "Goa"], "pin_norm": ["302001", None],
    "cin_norm": [None, None], "gstin_norm": ["08ZZZCS1234A1Z5", None],
})


def test_serialize_ditto_format_and_missing_values():
    s = serialize_records(DF, ["name", "street", "city", "pin"])
    assert s[0] == "COL name VAL sharma traders pvt ltd COL street VAL 12 mg road " \
                   "COL city VAL jaipur COL pin VAL 302001"
    assert "COL street VAL  COL city VAL panaji" in s[1]      # tag kept, value empty
    assert s[1].endswith("COL pin VAL")


def test_serialize_field_order_and_name_column():
    s = serialize_records(DF, ["city", "name"], name_column="name_raw")
    assert s[0].startswith("COL city VAL jaipur COL name VAL शर्मा")


def test_serialize_never_contains_ground_truth():
    df = DF.with_columns(pl.lit("E0000001").alias("entity_id"))
    assert not any("e0000001" in x for x in serialize_records(df, ["name", "street", "city"]))


def test_parse_render_round_trip():
    s = serialize_records(DF, ["name", "street", "city", "state", "pin"])
    for x in s:
        assert parse(render(parse(x))) == parse(x)
        assert [f for f, _ in parse(x)] == ["name", "street", "city", "state", "pin"]


@pytest.mark.parametrize("op", [o for o in OPS if o != "side_swap"])
def test_augment_ops_keep_fields(op):
    s = serialize_records(DF, ["name", "street", "city", "state", "pin"])[0]
    rng = random.Random(0)
    for _ in range(50):
        out = augment_record(s, op, rng)
        assert sorted(f for f, _ in parse(out)) == sorted(f for f, _ in parse(s))


def test_side_swap_and_determinism():
    a, b = "COL name VAL a b c", "COL name VAL x y z"
    assert augment_pair(a, b, random.Random(1), ("side_swap",)) == (b, a)
    r1 = [augment_pair(a, b, random.Random(7)) for _ in range(5)]
    r2 = [augment_pair(a, b, random.Random(7)) for _ in range(5)]
    assert r1 == r2


def test_recall_threshold_keeps_requested_share():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 10_000)
    s = rng.random(10_000) + y * 0.5
    for keep in [0.9, 0.995, 1.0]:
        thr = recall_threshold(s, y, keep)
        assert (s[y == 1] >= thr).mean() >= keep


def test_select_uncertainty_picks_closest_to_half():
    scores = np.array([0.01, 0.49, 0.99, 0.55, 0.2])
    got = select("uncertainty", scores, np.arange(5), 2, np.random.default_rng(0))
    assert sorted(got) == [1, 3]
    got = select("random", scores, np.array([0, 2, 4]), 2, np.random.default_rng(0))
    assert set(got) <= {0, 2, 4} and len(set(got)) == 2


def test_run_curve_budgets_and_shared_seed_set():
    rng = np.random.default_rng(0)
    y_pool = (rng.random(2000) < 0.2).astype(np.int8)
    y_eval = (rng.random(500) < 0.2).astype(np.int8)
    seen = []

    def learner(idx, y):  # a fake learner that remembers what it was trained on
        seen.append(set(idx.tolist()))
        p = float(y.mean())
        return lambda which: np.full(len(y_pool) if which == "pool" else len(y_eval), p)

    for strat in ["uncertainty", "random"]:
        rows = run_curve(learner, y_pool, y_eval, strat, [50, 100, 200], seed=3,
                         seed_labels=50)
        assert [r["labels"] for r in rows] == [50, 100, 200]
    # both strategies start from the same seed set; labels only grow within a curve
    assert seen[0] == seen[3]
    assert seen[0] <= seen[1] <= seen[2]


@pytest.mark.slow
def test_cross_encoder_learns_toy_task():
    """Smoke test with the cached encoder: learns 'same name' on 256 toy pairs."""
    from ekam.match.cross_encoder import CEConfig, CrossEncoder

    rng = random.Random(0)
    words = ["sharma", "gupta", "laxmi", "balaji", "krishna", "patel", "om", "sai", "star"]
    left, right, y = [], [], []
    for _ in range(256):
        a = " ".join(rng.sample(words, 2))
        same = rng.random() < 0.5
        b = a if same else " ".join(rng.sample(words, 2))
        left.append(f"COL name VAL {a} traders")
        right.append(f"COL name VAL {b} traders")
        y.append(float(a == b))
    ce = CrossEncoder(CEConfig(epochs=3, batch_size=16, lr=1e-4, max_len=32), seed=0)
    ce.fit(left, right, np.array(y, np.float32))
    p = ce.score(left, right)
    assert p.shape == (256,)
    assert ((p >= 0.5) == np.array(y, bool)).mean() > 0.8
