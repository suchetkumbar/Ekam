"""Supervised pairwise baselines on string-similarity features: LR and LightGBM."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LRMatcher:
    name = "logreg"

    def __init__(self, cfg: DictConfig, seed: int):
        self.model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(),
            LogisticRegression(C=float(cfg.C), max_iter=int(cfg.max_iter), random_state=seed))

    def fit(self, x, y, x_val=None, y_val=None):
        self.model.fit(x, y)
        return self

    def score(self, x) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


class LGBMMatcher:
    name = "lightgbm"

    def __init__(self, cfg: DictConfig, seed: int, n_jobs: int = -1):
        import lightgbm as lgb

        self._lgb = lgb
        self.cfg = cfg
        self.model = lgb.LGBMClassifier(
            n_estimators=int(cfg.n_estimators), learning_rate=float(cfg.learning_rate),
            num_leaves=int(cfg.num_leaves), min_child_samples=int(cfg.min_child_samples),
            subsample=float(cfg.subsample), subsample_freq=int(cfg.subsample_freq),
            colsample_bytree=float(cfg.colsample_bytree), random_state=seed, n_jobs=n_jobs,
            verbose=-1)

    def fit(self, x, y, x_val=None, y_val=None):
        kw = {}
        if x_val is not None:
            kw = {"eval_X": (x_val,), "eval_y": (y_val,),
                  "callbacks": [self._lgb.early_stopping(int(self.cfg.early_stopping_rounds),
                                                         verbose=False)]}
        self.model.fit(x, y, **kw)
        return self

    def score(self, x) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]

    def importances(self, names: list[str]) -> dict[str, float]:
        gain = self.model.booster_.feature_importance(importance_type="gain")
        tot = gain.sum() or 1.0
        return {n: float(g / tot) for n, g in zip(names, gain, strict=True)}


def make_matcher(kind: str, cfg: DictConfig, seed: int):
    if kind == "logreg":
        return LRMatcher(cfg.matcher.lr, seed)
    if kind == "lightgbm":
        return LGBMMatcher(cfg.matcher.lgbm, seed)
    raise ValueError(kind)


def subsample_training(y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Indices: all positives + random negatives up to ``cap`` rows in total."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    room = max(cap - len(pos), 0)
    if len(neg) > room:
        neg = np.random.default_rng(seed).choice(neg, size=room, replace=False)
    return np.sort(np.concatenate([pos, neg]))
