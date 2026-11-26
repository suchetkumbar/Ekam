"""Baseline experiments on the synthetic multi-source data (one per level × ID variant).

Protocol (identical for every matcher):
* train on blocked candidate pairs of the *train* split (all positives + capped negatives),
* choose the decision threshold that maximises F1 on the *valid* split's candidates,
* report on the *test* split's candidates; ``recall_e2e`` / ``f1_e2e`` also count true
  pairs that blocking missed, so they measure the whole blocking+matching pipeline.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl
from omegaconf import DictConfig

from ekam.blocking.metrics import best_threshold, label_pairs, prf, true_pairs
from ekam.features.pairs import TfidfName, feature_names, pair_features
from ekam.features.prepare import load_split
from ekam.match.baselines import make_matcher, subsample_training

log = logging.getLogger(__name__)


def _load(cfg: DictConfig, level: str, split: str):
    from ekam.pipeline.block import candidates_path

    df = load_split(cfg, level, split)
    cand = np.load(candidates_path(cfg, level, split))
    truth = true_pairs(df)
    y = label_pairs(cand, truth, df.height)
    return df, cand, y, len(truth)


def run_level(cfg: DictConfig, level: str) -> list[dict]:
    m = cfg.matcher
    seed = int(cfg.seed)
    data = {s: _load(cfg, level, s) for s in ["train", "valid", "test"]}
    df_tr, cand_tr, y_tr, _ = data["train"]
    keep = subsample_training(y_tr, int(m.train_max_pairs), seed)
    data["train"] = (df_tr, cand_tr[keep], y_tr[keep], data["train"][3])

    tfidf = TfidfName(m.tfidf.analyzer, m.tfidf.ngram_range, int(m.tfidf.min_df),
                      int(m.tfidf.max_features)).fit(df_tr["name_key"].to_list())
    all_groups = ["name", "street", "geo", "ids"]
    t0 = time.perf_counter()
    feats = {s: pair_features(d[0], d[1], tfidf, all_groups) for s, d in data.items()}
    log.info("%s: features for %s pairs in %.1fs", level,
             {s: len(d[1]) for s, d in data.items()}, time.perf_counter() - t0)

    rows = []
    for variant, groups in m.feature_sets.items():
        cols = feature_names(list(groups))
        x = {s: feats[s].select(cols).to_numpy() for s in feats}
        for kind in ["logreg", "lightgbm"]:
            t0 = time.perf_counter()
            model = make_matcher(kind, cfg, seed)
            model.fit(x["train"], data["train"][2], x["valid"], data["valid"][2])
            thr = best_threshold(data["valid"][2], model.score(x["valid"]))
            s_test = model.score(x["test"])
            res = prf(data["test"][2], (s_test >= thr).astype(np.int8), data["test"][3])
            row = {"dataset": "synthetic", "level": level, "variant": variant, "model": kind,
                   "threshold": thr, "train_pairs": len(data["train"][1]),
                   "test_pairs": len(data["test"][1]), "test_true_pairs": data["test"][3],
                   "seconds": time.perf_counter() - t0, **res}
            if kind == "lightgbm":
                imp = model.importances(cols)
                row["top_features"] = ", ".join(
                    f"{k}:{v:.2f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:5])
            rows.append(row)
            log.info("%s %s %s: P=%.3f R=%.3f F1=%.3f | e2e R=%.3f F1=%.3f", level, variant,
                     kind, res["precision"], res["recall"], res["f1"], res["recall_e2e"],
                     res["f1_e2e"])
        if m.splink.enabled and variant in m.splink.variants:
            rows.append(_splink_row(cfg, level, variant, data))
    return rows


def _splink_row(cfg: DictConfig, level: str, variant: str, data: dict) -> dict:
    from ekam.match.splink_model import score_pairs, train_splink

    t0 = time.perf_counter()
    model = train_splink(data["train"][0], cfg.matcher.splink, variant == "with_ids",
                         int(cfg.seed))
    s_val = score_pairs(model, data["valid"][0], data["valid"][1])
    thr = best_threshold(data["valid"][2], s_val)
    s_test = score_pairs(model, data["test"][0], data["test"][1])
    res = prf(data["test"][2], (s_test >= thr).astype(np.int8), data["test"][3])
    log.info("%s %s splink: P=%.3f R=%.3f F1=%.3f | e2e R=%.3f F1=%.3f", level, variant,
             res["precision"], res["recall"], res["f1"], res["recall_e2e"], res["f1_e2e"])
    return {"dataset": "synthetic", "level": level, "variant": variant, "model": "splink",
            "threshold": thr, "train_pairs": None, "test_pairs": len(data["test"][1]),
            "test_true_pairs": data["test"][3], "seconds": time.perf_counter() - t0, **res}


def blocking_recall(cfg: DictConfig, level: str) -> float:
    df, cand, y, n_true = _load(cfg, level, "test")
    return float(y.sum() / n_true) if n_true else 1.0


def results_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)
