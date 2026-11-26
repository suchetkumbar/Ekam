"""String-similarity baselines on the public benchmarks (published pair splits, no blocking).

DeepMatcher layout: tableA/tableB + {train,valid,test}.csv (ltable_id, rtable_id, label).
WDC Products: pair files with ``<attr>_left`` / ``<attr>_right``. Columns that identify the
answer (``cluster_id_*``, ids) are dropped *before* features — see ``LEAKY``.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

from ekam.blocking.metrics import best_threshold, prf
from ekam.data.benchmarks import load_deepmatcher, load_wdc
from ekam.match.baselines import make_matcher
from ekam.normalize.text import basic_clean

log = logging.getLogger(__name__)
LEAKY = re.compile(r"^(cluster_id|id|pair_id|ltable_id|rtable_id|label)(_left|_right)?$")


def deepmatcher_pairs(root: Path) -> tuple[dict[str, pl.DataFrame], list[str]]:
    t = load_deepmatcher(root)
    a, b = t["tableA"], t["tableB"]
    attrs = [c for c in a.columns if c != "id"]
    out = {}
    for split in ["train", "valid", "test"]:
        p = t[split]
        j = (p.join(a.rename({c: f"{c}_left" for c in a.columns}), left_on="ltable_id",
                    right_on="id_left", how="left")
             .join(b.rename({c: f"{c}_right" for c in b.columns}), left_on="rtable_id",
                   right_on="id_right", how="left"))
        out[split] = j.with_columns(pl.col("label").cast(pl.Int8))
    return out, attrs


def wdc_pairs(root: Path, files: dict) -> tuple[dict[str, pl.DataFrame], list[str]]:
    parts = load_wdc(root, files)
    attrs = sorted({c.rsplit("_", 1)[0] for c in parts["train"].columns
                    if c.endswith("_left") and not LEAKY.match(c)})
    out = {s: d.with_columns(pl.col("label").cast(pl.Int8)) for s, d in parts.items()}
    return out, attrs


def _num(s: list) -> np.ndarray:
    out = np.full(len(s), np.nan, np.float64)
    for i, v in enumerate(s):
        if v is None:
            continue
        m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
        if m:
            out[i] = float(m.group(0))
    return out


def attribute_features(splits: dict[str, pl.DataFrame], attrs: list[str], max_chars: int
                       ) -> dict[str, pl.DataFrame]:
    feats = {s: {} for s in splits}
    for a in attrs:
        l_tr = splits["train"][f"{a}_left"].cast(pl.Utf8).to_list()
        numeric = np.mean(~np.isnan(_num(l_tr))) > 0.8 and a in {"price", "year"}
        vec = None
        long_text = float(np.mean([len(x or "") for x in l_tr])) > 200
        if not numeric:
            corpus = [basic_clean(x)[:max_chars] for x in
                      l_tr + splits["train"][f"{a}_right"].cast(pl.Utf8).to_list() if x]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                  sublinear_tf=True, dtype=np.float32).fit(corpus or ["x"])
        for s, d in splits.items():
            left = d[f"{a}_left"].cast(pl.Utf8).to_list()
            right = d[f"{a}_right"].cast(pl.Utf8).to_list()
            miss = np.array([not x or not y for x, y in zip(left, right, strict=True)])
            feats[s][f"{a}_missing"] = miss.astype(np.float32)
            if numeric:
                x, y = _num(left), _num(right)
                feats[s][f"{a}_reldiff"] = np.abs(x - y) / np.maximum(np.maximum(abs(x), abs(y)),
                                                                     1e-9)
                continue
            lc = [basic_clean(x)[:max_chars] if x else "" for x in left]
            rc = [basic_clean(x)[:max_chars] if x else "" for x in right]
            scorers = [("tset", fuzz.token_set_ratio, 100.0)]
            if not long_text:  # character-level scorers are quadratic in length: short fields only
                scorers += [("jw", JaroWinkler.normalized_similarity, 1.0),
                            ("tsort", fuzz.token_sort_ratio, 100.0)]
            for fname, scorer, scale in scorers:
                v = process.cpdist(lc, rc, scorer=scorer, workers=-1, dtype=np.float32) / scale
                v[miss] = np.nan
                feats[s][f"{a}_{fname}"] = v
            xl, xr = vec.transform(lc), vec.transform(rc)
            cos = np.asarray(xl.multiply(xr).sum(axis=1)).ravel().astype(np.float32)
            cos[miss] = np.nan
            feats[s][f"{a}_tfidf"] = cos
    return {s: pl.DataFrame(f) for s, f in feats.items()}


def run_benchmark(cfg: DictConfig, name: str, spec) -> list[dict]:
    root = Path(cfg.benchmarks.out_dir) / name
    if spec.family == "deepmatcher":
        splits, attrs = deepmatcher_pairs(root)
    else:
        splits, attrs = wdc_pairs(root, dict(spec.files))
    assert not any(LEAKY.match(f"{a}_left") for a in attrs), "leaky attribute in features"
    t0 = time.perf_counter()
    feats = attribute_features(splits, attrs, int(cfg.matcher.benchmarks.text_max_chars))
    log.info("%s: %d attrs, features in %.1fs", name, len(attrs), time.perf_counter() - t0)
    x = {s: f.to_numpy() for s, f in feats.items()}
    y = {s: splits[s]["label"].to_numpy() for s in splits}
    rows = []
    for kind in ["logreg", "lightgbm"]:
        model = make_matcher(kind, cfg, int(cfg.seed))
        model.fit(x["train"], y["train"], x["valid"], y["valid"])
        thr = best_threshold(y["valid"], model.score(x["valid"]))
        for test_split in [s for s in splits if s.startswith("test")]:
            res = prf(y[test_split], (model.score(x[test_split]) >= thr).astype(np.int8))
            rows.append({"dataset": name, "level": test_split, "variant": "attributes",
                         "model": kind, "threshold": thr, "train_pairs": len(y["train"]),
                         "test_pairs": len(y[test_split]), **res})
            log.info("%s %s %s: P=%.3f R=%.3f F1=%.3f", name, test_split, kind,
                     res["precision"], res["recall"], res["f1"])
    return rows


def run_all(cfg: DictConfig) -> list[dict]:
    rows = []
    for name, spec in cfg.benchmarks.datasets.items():
        rows += run_benchmark(cfg, name, spec)
    return rows
