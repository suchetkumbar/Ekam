"""Stage 1 of the M3 cascade: a LightGBM pre-filter over blocked candidates.

The pre-filter is the M2 ``no_ids`` LightGBM (string similarities, no identifiers) trained
on a uniform sample of train-split candidates. Its threshold is set on the valid split so
that ``keep_recall`` of the true candidate pairs survive; the cross-encoder only sees the
survivors. Train-split survivors are chosen with **out-of-fold** scores, so the training
pairs of the cross-encoder look like the pairs it will be tested on (no self-scoring).

Cached at ``data/processed/<level>/m3_prefilter.npz``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from ekam.blocking.metrics import best_threshold, prf
from ekam.features.pairs import TfidfName, feature_names, pair_features
from ekam.match.baselines import make_matcher

log = logging.getLogger(__name__)


def prefilter_path(cfg: DictConfig, level: str) -> Path:
    return Path(cfg.paths.processed) / level / "m3_prefilter.npz"


def load_prefilter(cfg: DictConfig, level: str) -> dict[str, np.ndarray]:
    with np.load(prefilter_path(cfg, level)) as z:
        return {k: z[k] for k in z.files}


def recall_threshold(scores: np.ndarray, y: np.ndarray, keep: float) -> float:
    """Largest threshold that keeps at least ``keep`` of the positives (scores >= thr)."""
    pos = np.sort(scores[y == 1])
    if len(pos) == 0:
        return float("-inf")
    k = int(np.floor((1 - keep) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def build_prefilter(cfg: DictConfig, level: str, force: bool = False) -> dict[str, np.ndarray]:
    from ekam.match.synthetic import _load

    path = prefilter_path(cfg, level)
    if path.exists() and not force:
        return load_prefilter(cfg, level)
    t0 = time.perf_counter()
    pc, m, seed = cfg.ce.prefilter, cfg.matcher, int(cfg.seed)
    data = {s: _load(cfg, level, s) for s in ["train", "valid", "test"]}
    df_tr, cand_tr, y_tr, _ = data["train"]
    rng = np.random.default_rng(seed)
    n_pool = min(int(m.train_max_pairs), len(cand_tr))
    pool = np.sort(rng.choice(len(cand_tr), size=n_pool, replace=False))  # natural prevalence

    tfidf = TfidfName(m.tfidf.analyzer, m.tfidf.ngram_range, int(m.tfidf.min_df),
                      int(m.tfidf.max_features)).fit(df_tr["name_key"].to_list())
    groups = list(pc.feature_groups)
    cols = feature_names(groups)

    def feats(split, rows):
        df, cand = data[split][0], data[split][1]
        return pair_features(df, cand[rows], tfidf, groups).select(cols).to_numpy()

    x_pool, y_pool = feats("train", pool), y_tr[pool]
    x_val, y_val = feats("valid", np.arange(len(data["valid"][1]))), data["valid"][2]
    x_te, y_te = feats("test", np.arange(len(data["test"][1]))), data["test"][2]

    k = int(pc.oof_folds)
    fold = rng.integers(0, k, size=len(pool))
    oof = np.empty(len(pool), dtype=np.float32)
    for f in range(k):
        tr = fold != f
        mod = make_matcher("lightgbm", cfg, seed + f)
        mod.fit(x_pool[tr], y_pool[tr], x_val, y_val)
        oof[~tr] = mod.score(x_pool[~tr])
    final = make_matcher("lightgbm", cfg, seed)
    final.fit(x_pool, y_pool, x_val, y_val)
    s_val, s_te = final.score(x_val), final.score(x_te)

    keep = float(pc.keep_recall)
    thr = recall_threshold(s_val, y_val, keep)
    thr_tr = recall_threshold(oof, y_pool, keep)
    lgbm_thr = best_threshold(y_val, s_val)
    lgbm = prf(y_te, (s_te >= lgbm_thr).astype(np.int8))
    out = {
        "train_rows": pool[oof >= thr_tr], "valid_rows": np.flatnonzero(s_val >= thr),
        "test_rows": np.flatnonzero(s_te >= thr),
        "valid_scores": s_val.astype(np.float32), "test_scores": s_te.astype(np.float32),
        "threshold": np.float64(thr), "lgbm_threshold": np.float64(lgbm_thr),
        "lgbm_test_f1": np.float64(lgbm["f1"]),
        "test_kept_recall": np.float64(y_te[s_te >= thr].sum() / max(y_te.sum(), 1)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    log.info("%s prefilter: thr %.4f keeps %.1f%% of test candidates, %.4f of true test "
             "candidates; train survivors %d / %d; LightGBM alone test F1 %.4f (%.0fs)", level,
             thr, 100 * len(out["test_rows"]) / len(y_te), out["test_kept_recall"],
             len(out["train_rows"]), len(pool), lgbm["f1"], time.perf_counter() - t0)
    return out
