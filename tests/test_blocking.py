import numpy as np
import polars as pl
import pytest

from ekam.blocking.metrics import (
    best_threshold,
    blocking_metrics,
    encode_pairs,
    label_pairs,
    prf,
    true_pairs,
)
from ekam.blocking.rules import rule_pairs, union_pairs


@pytest.fixture
def recs():
    return pl.DataFrame({
        "idx": [0, 1, 2, 3, 4, 5],
        "entity_id": ["A", "A", "B", "B", "B", "C"],
        "pin_norm": ["560001", "560001", "302001", None, "302001", "560001"],
        "first_token": ["SHARMA", "SHARMA", "GUPTA", "GUPTA", "GUPTA", "SHARMA"],
        "name_phonetic": ["SRM|TRTRS", "SRM|TRTRS", "KPT", "KPT", "KPT", "SRM|ENT"],
    })


def test_true_pairs(recs):
    t = {tuple(p) for p in true_pairs(recs)}
    assert t == {(0, 1), (2, 3), (2, 4), (3, 4)}


def test_rule_pairs_conjunction_and_nulls(recs):
    out = rule_pairs(recs, {"pin_first": ["pin_norm", "first_token"]}, max_block_size=10)
    got = {tuple(p) for p in out["pin_first"]}
    # (0,1) and (0,5),(1,5) share pin+first token; record 3 has no PIN -> never keyed
    assert got == {(0, 1), (0, 5), (1, 5), (2, 4)}


def test_rule_block_size_cap(recs):
    out = rule_pairs(recs, {"first": ["first_token"]}, max_block_size=2)
    assert len(out["first"]) == 0  # both blocks have 3 records > cap


def test_union_dedups_and_orders():
    a = np.array([[1, 0], [2, 3]])
    b = np.array([[0, 1], [3, 3], [4, 2]])
    u = union_pairs(a, b)
    assert u.tolist() == [[0, 1], [2, 3], [2, 4]]


def test_blocking_metrics(recs):
    truth = true_pairs(recs)
    cand = np.array([[0, 1], [2, 4], [0, 5]])
    m = blocking_metrics(cand, truth, n=6)
    assert m["pair_completeness"] == pytest.approx(2 / 4)
    assert m["reduction_ratio"] == pytest.approx(1 - 3 / 15)
    assert m["pair_quality"] == pytest.approx(2 / 3)


def test_label_pairs_and_prf(recs):
    truth = true_pairs(recs)
    cand = np.array([[0, 1], [2, 4], [0, 5]])
    y = label_pairs(cand, truth, 6)
    assert y.tolist() == [1, 1, 0]
    r = prf(y, np.array([1, 0, 1]), n_true_total=4)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["recall_e2e"] == pytest.approx(1 / 4)


def test_encode_pairs_symmetric():
    assert encode_pairs(np.array([[3, 1], [1, 3]]), 10).tolist() == [13]


def test_best_threshold_separable():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    t = best_threshold(y, s)
    assert 0.4 < t <= 0.6


def test_knn_pairs_excludes_self_and_respects_k():
    pytest.importorskip("faiss")
    from ekam.blocking.knn import build_index, knn_pairs, knn_search

    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 16)).astype(np.float32)
    x = np.concatenate([base, base + 0.01 * rng.normal(size=base.shape)]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    idx = build_index(x, kind="flat")
    sims, nbrs = knn_search(idx, x, 3)
    p1 = knn_pairs(sims, nbrs, 1)
    assert (p1[:, 0] < p1[:, 1]).all()
    assert {(i, i + 50) for i in range(50)} <= {tuple(p) for p in p1}
    assert len(knn_pairs(sims, nbrs, 3)) >= len(p1)
