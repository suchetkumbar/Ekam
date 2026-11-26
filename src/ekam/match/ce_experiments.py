"""M3 experiments on the synthetic levels: cascade (LightGBM pre-filter -> cross-encoder),
ablations, and the data shared with active learning.

Evaluation protocol (same as M2, so numbers are comparable with the M2 baselines):
* metrics are over **all** blocked test candidates: pairs removed by the pre-filter count as
  predicted non-matches, so the cascade pays for every true pair the pre-filter drops;
* ``recall_e2e`` / ``f1_e2e`` additionally count true pairs lost in blocking;
* the decision threshold is chosen on the valid split (its pre-filter survivors).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

from ekam.blocking.metrics import best_threshold, prf
from ekam.match.cascade import build_prefilter
from ekam.match.cross_encoder import CEConfig, CrossEncoder
from ekam.match.serialize import serialize_records

log = logging.getLogger(__name__)


@dataclass
class Split:
    df: pl.DataFrame
    cand: np.ndarray
    y: np.ndarray
    n_true: int


def ce_config(cfg: DictConfig, **over) -> CEConfig:
    mc = OmegaConf.to_container(cfg.ce.model, resolve=True)
    kw = {"model": mc.pop("name"), **mc, "augment_p": float(cfg.ce.augment.p),
          "augment_ops": tuple(cfg.ce.augment.ops)}
    kw.update({k: v for k, v in over.items() if k in {f.name for f in dc_fields(CEConfig)}})
    return CEConfig(**kw)


def load_splits(cfg: DictConfig, level: str) -> dict[str, Split]:
    from ekam.match.synthetic import _load

    return {s: Split(*_load(cfg, level, s)) for s in ["train", "valid", "test"]}


def _with_raw_names(cfg: DictConfig, level: str, df: pl.DataFrame) -> pl.DataFrame:
    raw = pl.read_parquet(Path(cfg.synth.out_dir) / level / "records.parquet",
                          columns=["record_id", "name"]).rename({"name": "name_raw"})
    out = df.join(raw, on="record_id", how="left")
    return out.sort("idx")


class Texts:
    """Serialised records per split, built lazily for a (fields, name column) setting."""

    def __init__(self, cfg: DictConfig, level: str, splits: dict[str, Split]):
        self.cfg, self.level, self.splits = cfg, level, splits
        self._cache: dict[tuple, dict[str, list[str]]] = {}

    def get(self, fields: list[str], name_column: str) -> dict[str, list[str]]:
        key = (tuple(fields), name_column)
        if key not in self._cache:
            out = {}
            for s, sp in self.splits.items():
                df = sp.df
                col = name_column
                if name_column == "raw":
                    df, col = _with_raw_names(self.cfg, self.level, df), "name_raw"
                out[s] = serialize_records(df, fields, name_column=col)
            self._cache[key] = out
        return self._cache[key]


def pair_texts(texts: list[str], cand: np.ndarray, rows: np.ndarray) -> tuple[list, list]:
    return [texts[i] for i in cand[rows, 0]], [texts[i] for i in cand[rows, 1]]


def run_cascade(cfg: DictConfig, level: str, splits: dict[str, Split], texts: Texts, *,
                tag: str, fields: list[str], name_column: str, train_pairs: int,
                save_model: bool = False, eval_fraction: float = 1.0, **over) -> dict:
    """Train one cross-encoder on pre-filter survivors and evaluate the whole cascade.

    ``eval_fraction`` < 1 (ablations) evaluates on a uniform sample of the valid and test
    candidates: P/R/F1 stay unbiased estimates, end-to-end metrics are not reported.
    """
    seed = int(cfg.seed)
    pf = build_prefilter(cfg, level)
    tx = texts.get(fields, name_column)
    rng = np.random.default_rng(seed)
    tr_rows = pf["train_rows"]
    tr_rows = np.sort(rng.choice(tr_rows, size=min(train_pairs, len(tr_rows)), replace=False))
    va_rows = pf["valid_rows"]
    te_rows = pf["test_rows"]
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    te_eval = np.arange(len(te.y))
    if eval_fraction < 1.0:
        srng = np.random.default_rng(seed + 1)
        va_rows = va_rows[srng.random(len(va_rows)) < eval_fraction]
        te_eval = np.flatnonzero(srng.random(len(te.y)) < eval_fraction)
        te_rows = np.intersect1d(te_rows, te_eval)
    sel = np.sort(rng.choice(va_rows, size=min(int(cfg.ce.select_pairs), len(va_rows)),
                             replace=False))

    t0 = time.perf_counter()
    ce = CrossEncoder(ce_config(cfg, **over), seed)
    ce.fit(*pair_texts(tx["train"], tr.cand, tr_rows), tr.y[tr_rows].astype(np.float32),
           val=(*pair_texts(tx["valid"], va.cand, sel), va.y[sel]))
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_va = ce.score(*pair_texts(tx["valid"], va.cand, va_rows))
    thr = best_threshold(va.y[va_rows], s_va)
    s_te = ce.score(*pair_texts(tx["test"], te.cand, te_rows))
    t_score = time.perf_counter() - t0

    pred = np.zeros(len(te.y), dtype=np.int8)
    pred[te_rows] = (s_te >= thr).astype(np.int8)
    if eval_fraction < 1.0:
        res = prf(te.y[te_eval], pred[te_eval])
        res.update(recall_e2e=None, f1_e2e=None)
    else:
        res = prf(te.y, pred, te.n_true)
    if save_model:
        out = Path(cfg.ce.models_dir) / f"{level}_{tag}"
        ce.save(out)
        np.savez(Path(cfg.paths.processed) / level / f"m3_scores_{tag}.npz", test_rows=te_rows,
                 test_scores=s_te, valid_rows=va_rows, valid_scores=s_va, threshold=thr)
    row = {"level": level, "run": tag, "model": ce.cfg.model.split("/")[-1],
           "fields": ",".join(fields), "name_column": name_column,
           "augment_p": ce.cfg.augment_p, "train_pairs": len(tr_rows),
           "train_pos_share": float(tr.y[tr_rows].mean()), "threshold": thr,
           "prefilter_kept": len(te_rows) / len(te_eval), "eval_fraction": eval_fraction,
           "prefilter_recall": float(pf["test_kept_recall"]),
           "lgbm_f1": float(pf["lgbm_test_f1"]),
           "best_epoch": int(np.argmax([h.get("val_f1", 0) for h in ce.history]) + 1),
           "train_seconds": t_train, "score_seconds": t_score, **res}
    log.info("%s %s: F1 %.4f (P %.4f R %.4f) | e2e F1 %s | LightGBM %.4f | train %.0fs "
             "score %.0fs", level, tag, res["f1"], res["precision"], res["recall"],
             res["f1_e2e"], row["lgbm_f1"], t_train, t_score)
    del ce
    _free_gpu()
    return row


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_main(cfg: DictConfig) -> list[dict]:
    rows = []
    for level in cfg.ce.levels:
        splits = load_splits(cfg, level)
        texts = Texts(cfg, level, splits)
        for variant in cfg.ce.main_variants[level]:
            rows.append(run_cascade(cfg, level, splits, texts, tag=variant,
                                    fields=list(cfg.ce.variants[variant]),
                                    name_column=cfg.ce.name_column,
                                    train_pairs=int(cfg.ce.train_pairs), save_model=True))
    return rows


def run_ablations(cfg: DictConfig) -> list[dict]:
    a = cfg.ce.ablations
    splits = load_splits(cfg, a.level)
    texts = Texts(cfg, a.level, splits)
    rows = []
    for tag, over in a.runs.items():
        over = OmegaConf.to_container(over, resolve=True) if over else {}
        fields = over.pop("fields", list(cfg.ce.variants[a.variant]))
        name_column = over.pop("name_column", cfg.ce.name_column)
        rows.append(run_cascade(cfg, a.level, splits, texts, tag=f"ablation_{tag}",
                                fields=list(fields), name_column=name_column,
                                train_pairs=int(a.train_pairs),
                                eval_fraction=float(a.eval_fraction), **over))
    return rows


# ---- active learning --------------------------------------------------------------------
def _lgbm_learner(params: dict, x_pool: np.ndarray, x_eval: np.ndarray, seed: int):
    import lightgbm as lgb

    def fit(idx, y):
        mod = lgb.LGBMClassifier(**params, random_state=seed, verbose=-1, n_jobs=-1)
        mod.fit(x_pool[idx], y)
        cache: dict[str, np.ndarray] = {}

        def score(which):
            if which not in cache:
                x = x_pool if which == "pool" else x_eval
                cache[which] = mod.predict_proba(x)[:, 1]
            return cache[which]
        return score
    return fit


def _ce_learner(cfg: DictConfig, epochs: int, pool_lr, eval_lr, seed: int):
    def fit(idx, y):
        ce = CrossEncoder(ce_config(cfg, epochs=epochs), seed)
        ce.fit([pool_lr[0][i] for i in idx], [pool_lr[1][i] for i in idx],
               y.astype(np.float32))

        def score(which):
            lr = pool_lr if which == "pool" else eval_lr
            return ce.score(*lr)
        return score
    return fit


def run_active(cfg: DictConfig) -> list[dict]:
    from ekam.features.pairs import TfidfName, feature_names, pair_features
    from ekam.match.active import run_curve

    al, m = cfg.al, cfg.matcher
    level, seed = al.level, int(cfg.seed)
    splits = load_splits(cfg, level)
    tr, te = splits["train"], splits["test"]
    rng = np.random.default_rng(seed)
    eval_rows = np.sort(rng.choice(len(te.y), size=min(int(al.eval_pairs), len(te.y)),
                                   replace=False))
    y_eval = te.y[eval_rows]
    rows = []
    if al.lightgbm.enabled:
        lc = al.lightgbm
        pool = np.sort(rng.choice(len(tr.y), size=min(int(lc.pool_pairs), len(tr.y)),
                                  replace=False))
        tfidf = TfidfName(m.tfidf.analyzer, m.tfidf.ngram_range, int(m.tfidf.min_df),
                          int(m.tfidf.max_features)).fit(tr.df["name_key"].to_list())
        groups = list(cfg.ce.prefilter.feature_groups)
        cols = feature_names(groups)
        x_pool = pair_features(tr.df, tr.cand[pool], tfidf, groups).select(cols).to_numpy()
        x_eval = pair_features(te.df, te.cand[eval_rows], tfidf, groups).select(cols).to_numpy()
        params = OmegaConf.to_container(lc.params, resolve=True)
        for s in lc.seeds:
            learner = _lgbm_learner(params, x_pool, x_eval, int(s))
            for strat in al.strategies:
                for r in run_curve(learner, tr.y[pool], y_eval, strat, list(lc.budgets),
                                   int(s), int(lc.seed_labels)):
                    rows.append({"learner": "lightgbm", **r})
    if al.cross_encoder.enabled:
        cc = al.cross_encoder
        pool = np.sort(rng.choice(len(tr.y), size=min(int(cc.pool_pairs), len(tr.y)),
                                  replace=False))
        tx = Texts(cfg, level, splits).get(list(cfg.ce.variants[al.variant]),
                                           cfg.ce.name_column)
        ce_eval = np.sort(rng.choice(len(eval_rows), size=min(int(cc.eval_pairs),
                                                                 len(eval_rows)), replace=False))
        pool_lr = pair_texts(tx["train"], tr.cand, pool)
        eval_lr = pair_texts(tx["test"], te.cand, eval_rows[ce_eval])
        for s in cc.seeds:
            learner = _ce_learner(cfg, int(cc.epochs), pool_lr, eval_lr, int(s))
            for strat in al.strategies:
                for r in run_curve(learner, tr.y[pool], y_eval[ce_eval], strat, list(cc.budgets),
                                   int(s), int(cc.seed_labels)):
                    rows.append({"learner": "cross_encoder", **r})
                _free_gpu()
    return rows
