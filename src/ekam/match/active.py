"""Active learning for the matcher: how many labels does each strategy need?

Simulated reviewer: the ground truth answers every query. Each round the learner is retrained
from scratch on all labels so far, scores the unlabelled pool, and the strategy picks the
next batch:

* ``uncertainty`` - pool pairs whose match probability is closest to 0.5 (least confident);
* ``random``      - uniform from the pool (the baseline every AL claim must beat).

Both strategies start from the same random seed set. Evaluation uses a fixed test sample and
a **fixed 0.5 threshold** (a team starting from scratch has no large labelled valid set to
tune one), plus average precision, which needs no threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score

from ekam.blocking.metrics import prf

log = logging.getLogger(__name__)

# fit(labelled pool indices, their labels) -> score(which: "pool" | "eval") -> probabilities
Learner = Callable[[np.ndarray, np.ndarray], Callable[[str], np.ndarray]]


def select(strategy: str, scores: np.ndarray | None, unlabeled: np.ndarray, k: int,
           rng: np.random.Generator) -> np.ndarray:
    k = min(k, len(unlabeled))
    if strategy == "random" or scores is None:
        return rng.choice(unlabeled, size=k, replace=False)
    if strategy == "uncertainty":
        u = np.abs(scores[unlabeled] - 0.5) + rng.random(len(unlabeled)) * 1e-9  # random ties
        return unlabeled[np.argpartition(u, k - 1)[:k]] if k < len(unlabeled) else unlabeled
    raise ValueError(strategy)


def run_curve(learner: Learner, y_pool: np.ndarray, y_eval: np.ndarray, strategy: str,
              budgets: list[int], seed: int, seed_labels: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(y_pool)
    start = np.random.default_rng(seed + 10_000).choice(n, size=seed_labels, replace=False)
    labeled = np.zeros(n, dtype=bool)
    labeled[start] = True
    rows, pool_scores = [], None
    for b in sorted(budgets):
        need = b - int(labeled.sum())
        if need > 0:
            new = select(strategy, pool_scores, np.flatnonzero(~labeled), need, rng)
            labeled[new] = True
        idx = np.flatnonzero(labeled)
        if len(np.unique(y_pool[idx])) < 2:  # cannot train a classifier on one class
            rows.append({"strategy": strategy, "seed": seed, "labels": int(len(idx)),
                         "f1": 0.0, "precision": 0.0, "recall": 0.0, "avg_precision": None,
                         "pos_share": float(y_pool[idx].mean())})
            continue
        score = learner(idx, y_pool[idx])
        s_eval = score("eval")
        res = prf(y_eval, (s_eval >= 0.5).astype(np.int8))
        rows.append({"strategy": strategy, "seed": seed, "labels": int(len(idx)),
                     "f1": res["f1"], "precision": res["precision"], "recall": res["recall"],
                     "avg_precision": float(average_precision_score(y_eval, s_eval)),
                     "pos_share": float(y_pool[idx].mean())})
        log.info("AL %s seed %d labels %d: F1@0.5 %.4f AP %.4f (pos share %.2f)", strategy,
                 seed, len(idx), res["f1"], rows[-1]["avg_precision"], rows[-1]["pos_share"])
        if b != max(budgets):
            pool_scores = score("pool")
    return rows
