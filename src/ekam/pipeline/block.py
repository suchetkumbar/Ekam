"""`make block`: normalise records, generate candidates (rules / FAISS kNN / union) for every
split, and report the blocking trade-off on the held-out split.

Outputs per level under ``data/processed/<level>/``:
  ``cand_<split>.npy`` — union candidate pairs (split-local idx, l < r) used by `make train`.
Reports: ``docs/reports/M2_blocking.md`` + ``M2_blocking.png`` + ``M2_blocking.json``; MLflow
experiment ``ekam-m2-blocking``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from ekam.blocking.embed import _load_model, cached_embeddings
from ekam.blocking.knn import build_index, knn_pairs, knn_search
from ekam.blocking.metrics import blocking_metrics, true_pairs
from ekam.blocking.rules import rule_pairs, union_pairs
from ekam.features.prepare import load_split, prepare_level
from ekam.utils.config import CONF_DIR
from ekam.utils.logging import setup_logging
from ekam.utils.tracking import mlflow_run

log = logging.getLogger("ekam.block")
SPLITS = ["train", "valid", "test"]


def candidates_path(cfg: DictConfig, level: str, split: str) -> Path:
    return Path(cfg.paths.processed) / level / f"cand_{split}.npy"


def _knn(cfg, vecs, k_max):
    b = cfg.blocking.knn
    index = build_index(vecs, b.index, b.auto_flat_max, b.hnsw_m, b.hnsw_ef_search,
                        cfg.hardware.polars_max_threads)
    return knn_search(index, vecs, k_max)


def block_level(cfg: DictConfig, level: str, model) -> list[dict]:
    """Candidates for every split; curve rows for the evaluation split."""
    b = cfg.blocking
    rules = {k: list(v) for k, v in b.rules.blocks.items()}
    rows: list[dict] = []
    for split in SPLITS:
        df = load_split(cfg, level, split)
        n = df.height
        t0 = time.perf_counter()
        rp = rule_pairs(df, rules, int(b.rules.max_block_size))
        t_rules = time.perf_counter() - t0
        is_eval = split == b.eval_split
        use_knn = bool(b.knn.enabled)
        t0 = time.perf_counter()
        if use_knn:
            vecs = cached_embeddings(cfg, level, split, df, model)
            k_max = max(b.curve.knn_k) if is_eval else int(b.knn.k)
            sims, nbrs = _knn(cfg, vecs, k_max)
            kp = knn_pairs(sims, nbrs, int(b.knn.k), float(b.knn.min_similarity))
        else:
            kp = np.empty((0, 2), np.int64)
        t_knn = time.perf_counter() - t0
        rules_all = union_pairs(*rp.values())
        cand = union_pairs(rules_all, kp)
        np.save(candidates_path(cfg, level, split), cand)
        log.info("%s/%s: n=%d rules=%d knn=%d union=%d (rules %.1fs, knn %.1fs)", level, split,
                 n, len(rules_all), len(kp), len(cand), t_rules, t_knn)
        if not is_eval:
            continue

        truth = true_pairs(df)
        base = {"level": level, "split": split, "n_records": n, "true_pairs": len(truth)}

        def add(method, param, pairs, seconds=None, _base=base, _truth=truth, _n=n):
            m = blocking_metrics(pairs, _truth, _n)
            rows.append({**_base, "method": method, "param": str(param), **m,
                         "seconds": seconds})

        for name, p in rp.items():
            add(f"rule:{name}", b.rules.max_block_size, p)
        add("rules", b.rules.max_block_size, rules_all, t_rules)
        if use_knn:
            add("knn", b.knn.k, kp, t_knn)
            add("union", f"rules+knn{b.knn.k}", cand)
            for k in b.curve.knn_k:  # trade-off curves
                kp_k = knn_pairs(sims, nbrs, int(k), float(b.knn.min_similarity))
                add("curve:knn", k, kp_k)
                add("curve:union", k, union_pairs(rules_all, kp_k))
        for mbs in b.curve.rule_block_sizes:
            add("curve:rules", mbs, union_pairs(*rule_pairs(df, rules, int(mbs)).values()))
    return rows


def _plot(df: pl.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = df["level"].unique(maintain_order=True).to_list()
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4), sharey=True)
    axes = np.atleast_1d(axes)
    styles = {"curve:knn": ("kNN (k)", "o-"), "curve:rules": ("rules (max block)", "s-"),
              "curve:union": ("rules ∪ kNN (k)", "^-")}
    for ax, lv in zip(axes, levels, strict=True):
        for method, (label, st) in styles.items():
            d = df.filter((pl.col("level") == lv) & (pl.col("method") == method)) \
                .sort("pairs_per_record")
            ax.plot(d["pairs_per_record"], d["pair_completeness"], st, label=label, ms=4)
            for x, y, p in zip(d["pairs_per_record"], d["pair_completeness"], d["param"],
                               strict=True):
                ax.annotate(p, (x, y), fontsize=7, xytext=(3, -9), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_title(f"{lv} (test split)")
        ax.set_xlabel("candidate pairs per record (log)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pair completeness (recall of true pairs)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _report(cfg: DictConfig, df: pl.DataFrame) -> Path:
    out_dir = Path(cfg.paths.reports)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "M2_blocking.json").write_text(json.dumps(df.to_dicts(), indent=2), encoding="utf-8")
    _plot(df, out_dir / "M2_blocking.png")
    main = (df.filter(~pl.col("method").str.starts_with("curve"))
            .select("level", "method", "param", "pairs", "pairs_per_record",
                    "pair_completeness", "reduction_ratio", "pair_quality")
            .with_columns(pl.col(pl.Float64).exclude("reduction_ratio").round(4),
                          pl.col("reduction_ratio").map_elements(lambda x: f"{x:.6f}", pl.Utf8)))
    lines = ["# Milestone 2 — blocking", "",
             f"Encoder `{cfg.encoder.name}` (text: {', '.join(cfg.encoder.text_fields)}); rules "
             f"{dict(cfg.blocking.rules.blocks)} with max block size "
             f"{cfg.blocking.rules.max_block_size}; kNN k={cfg.blocking.knn.k}. "
             f"Evaluated on the **{cfg.blocking.eval_split}** split (entity-disjoint).", "",
             "![blocking trade-off](M2_blocking.png)", "",
             "| " + " | ".join(main.columns) + " |", "|" + "---|" * len(main.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in main.iter_rows()]
    (out_dir / "M2_blocking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "M2_blocking.md"


def run(cfg: DictConfig) -> pl.DataFrame:
    levels = list(cfg.pipeline.levels)
    for lv in levels:
        prepare_level(cfg, lv)
    model = _load_model(cfg.encoder)[0] if cfg.blocking.knn.enabled else None
    rows = []
    with mlflow_run(cfg, "ekam-m2-blocking", "blocking",
                    {"encoder": cfg.encoder.name, "rules": dict(cfg.blocking.rules.blocks),
                     "max_block_size": cfg.blocking.rules.max_block_size,
                     "k": cfg.blocking.knn.k}) as (mlflow, _):
        for lv in levels:
            rows += block_level(cfg, lv, model)
        df = pl.DataFrame(rows, infer_schema_length=None)
        for r in df.filter(pl.col("method").is_in(["rules", "knn", "union"])).iter_rows(named=True):
            key = f"{r['level']}.{r['method']}"
            mlflow.log_metrics({f"{key}.pc": r["pair_completeness"],
                                f"{key}.rr": r["reduction_ratio"],
                                f"{key}.ppr": r["pairs_per_record"]})
        report = _report(cfg, df)
        mlflow.log_artifact(str(report))
        mlflow.log_artifact(str(report.with_suffix(".png")))
    log.info("blocking report -> %s", report)
    return df


@hydra.main(config_path=str(CONF_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    run(cfg)


if __name__ == "__main__":
    main()
