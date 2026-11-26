"""`make train` (M2): string-similarity baselines (LR, LightGBM) and Splink on the synthetic
levels (with / without identifier features), plus LR / LightGBM on the public benchmarks.

Requires `make block` first (candidate pairs). Writes ``docs/reports/M2_baselines.{md,json}``
and logs every row to MLflow experiment ``ekam-m2-baselines``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from ekam.utils.config import CONF_DIR
from ekam.utils.logging import setup_logging
from ekam.utils.tracking import mlflow_run

log = logging.getLogger("ekam.train")


def _table(df: pl.DataFrame, cols: list[str]) -> list[str]:
    d = df.select([c for c in cols if c in df.columns])
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    for r in d.iter_rows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                       for v in r) + " |")
    return lines


def report(cfg: DictConfig, df: pl.DataFrame, blocking_pc: dict[str, float]) -> Path:
    out = Path(cfg.paths.reports)
    out.mkdir(parents=True, exist_ok=True)
    (out / "M2_baselines.json").write_text(json.dumps(df.to_dicts(), indent=2, default=str),
                                           encoding="utf-8")
    syn = df.filter(pl.col("dataset") == "synthetic").sort("level", "variant", "model")
    bench = df.filter(pl.col("dataset") != "synthetic").sort("dataset", "level", "model")
    lines = [
        "# Milestone 2 — baselines", "",
        "## Synthetic sources (test split, entity-disjoint)", "",
        "Matchers score the same blocked candidates (rules ∪ kNN). `precision/recall/f1` are "
        "over candidate pairs; `recall_e2e`/`f1_e2e` also count true pairs lost in blocking. "
        "Thresholds chosen on the valid split. `no_ids` removes CIN/GSTIN features; "
        "`no_street` (diagnostic) also removes street-line features.", "",
        "Blocking pair completeness on test: " +
        ", ".join(f"{k} {v:.3f}" for k, v in blocking_pc.items()) + ".", "",
        *_table(syn, ["level", "variant", "model", "precision", "recall", "f1", "recall_e2e",
                      "f1_e2e", "test_pairs", "seconds"]), "",
        "LightGBM top features (gain share):", "",
        *[f"- {r['level']} / {r['variant']}: {r['top_features']}"
          for r in syn.filter(pl.col("model") == "lightgbm").iter_rows(named=True)], "",
        "## Public benchmarks (published train/valid/test pair splits)", "",
        "Attribute-wise Jaro-Winkler, token-set/sort ratios, char TF-IDF cosine and numeric "
        "differences; same LR / LightGBM protocol. DeepMatcher splits are pair-level (not "
        "entity-disjoint); WDC `test_unseen*` contain products unseen in training.", "",
        *_table(bench, ["dataset", "level", "model", "precision", "recall", "f1", "test_pairs"]),
        "",
    ]
    (out / "M2_baselines.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "M2_baselines.md"


def run(cfg: DictConfig) -> pl.DataFrame:
    from ekam.match import benchmarks, synthetic

    rows, pcs = [], {}
    with mlflow_run(cfg, "ekam-m2-baselines", "baselines",
                    {"train_max_pairs": cfg.matcher.train_max_pairs,
                     "levels": list(cfg.matcher.levels)}) as (mlflow, _):
        for lv in cfg.matcher.levels:
            pcs[lv] = synthetic.blocking_recall(cfg, lv)
            rows += synthetic.run_level(cfg, lv)
        if cfg.matcher.benchmarks.enabled:
            rows += benchmarks.run_all(cfg)
        df = pl.DataFrame(rows, infer_schema_length=None)
        for r in df.iter_rows(named=True):
            key = f"{r['dataset']}.{r['level']}.{r['variant']}.{r['model']}"
            mlflow.log_metrics({f"{key}.{m}": float(r[m]) for m in
                                ["precision", "recall", "f1", "recall_e2e", "f1_e2e"]
                                if r.get(m) is not None})
        path = report(cfg, df, pcs)
        mlflow.log_artifact(str(path))
    log.info("baselines report -> %s", path)
    return df


@hydra.main(config_path=str(CONF_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    run(cfg)


if __name__ == "__main__":
    main()
