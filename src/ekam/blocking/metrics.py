"""Blocking and pairwise-matching metrics.

Pairs are encoded as int64 ``l * n + r`` (l < r) so set operations are vectorised.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def encode_pairs(p: np.ndarray, n: int) -> np.ndarray:
    if len(p) == 0:
        return np.empty(0, np.int64)
    lo, hi = np.minimum(p[:, 0], p[:, 1]), np.maximum(p[:, 0], p[:, 1])
    return np.unique(lo.astype(np.int64) * n + hi)


def true_pairs(df: pl.DataFrame) -> np.ndarray:
    """All (l<r) index pairs sharing an entity_id. ``df`` needs ``idx`` and ``entity_id``."""
    g = df.group_by("entity_id").agg(pl.col("idx").sort()).filter(pl.col("idx").list.len() > 1)
    out = []
    for ids in g["idx"].to_list():
        a = np.asarray(ids, dtype=np.int64)
        i, j = np.triu_indices(len(a), k=1)
        out.append(np.stack([a[i], a[j]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 2), np.int64)


def blocking_metrics(cand: np.ndarray, truth: np.ndarray, n: int) -> dict:
    c, t = encode_pairs(cand, n), encode_pairs(truth, n)
    hit = np.intersect1d(c, t, assume_unique=True).size
    total = n * (n - 1) / 2
    return {
        "pairs": int(c.size),
        "pairs_per_record": c.size / n if n else 0.0,
        "pair_completeness": hit / t.size if t.size else 1.0,
        "reduction_ratio": 1 - c.size / total if total else 1.0,
        "pair_quality": hit / c.size if c.size else 0.0,
    }


def label_pairs(cand: np.ndarray, truth: np.ndarray, n: int) -> np.ndarray:
    """1 if the candidate pair is a true match, else 0 (aligned with ``cand`` rows)."""
    c = cand[:, 0].astype(np.int64) * n + cand[:, 1]
    return np.isin(c, encode_pairs(truth, n)).astype(np.int8)


def prf(y_true: np.ndarray, y_pred: np.ndarray, n_true_total: int | None = None) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    out = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
           "tp": tp, "fp": fp, "fn": fn}
    if n_true_total is not None:  # end-to-end: misses from blocking count as false negatives
        r_all = tp / n_true_total if n_true_total else 0.0
        out["recall_e2e"] = r_all
        out["f1_e2e"] = 2 * p * r_all / (p + r_all) if p + r_all else 0.0
    return out


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 (computed exactly over sorted scores)."""
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    pos = y.sum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(pos, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(scores[order][i])
