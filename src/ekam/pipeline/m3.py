"""`make m3` (M3): cascade cross-encoder, ablations and active learning on the synthetic data.

Stages (``pipeline.m3_stages``, default all): ``main`` (cascade per level / variant),
``ablations`` (hard level, smaller budget), ``al`` (labels vs F1). Each stage writes
``docs/reports/M3_<stage>.json``; the report ``docs/reports/M3_results.md`` (+ ``M3_al.png``)
is rebuilt from whichever JSON files exist, so stages can be run separately:
``make m3 ARGS="pipeline.m3_stages=[al]"``. Requires `make block` (and uses the M2
baselines report for the comparison column). MLflow experiment ``ekam-m3``.
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

log = logging.getLogger("ekam.m3")

STAGES = ("main", "ablations", "al")

# light-mode categorical slots 1-2 (validated: CVD ΔE 26.6, normal 35.2); text inks
_C = {"uncertainty": "#257de0", "random": "#f47230"}
_INK, _INK2 = "#191919", "#666666"


def _json(cfg: DictConfig, stage: str) -> Path:
    return Path(cfg.paths.reports) / f"M3_{stage}.json"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.0f}"
    return "" if v is None else str(v)


def _table(df: pl.DataFrame, cols: list[str]) -> list[str]:
    d = df.select([c for c in cols if c in df.columns])
    return (["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
            + ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in d.iter_rows()])


def _m2_baselines(cfg: DictConfig) -> pl.DataFrame | None:
    p = Path(cfg.paths.reports) / "M2_baselines.json"
    if not p.exists():
        return None
    df = pl.DataFrame(json.loads(p.read_text(encoding="utf-8")), infer_schema_length=None)
    return df.filter(pl.col("dataset") == "synthetic")


def plot_al(df: pl.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    learners = [x for x in ["lightgbm", "cross_encoder"] if x in df["learner"].unique()]
    fig, axes = plt.subplots(1, len(learners), figsize=(5.2 * len(learners), 3.8),
                             sharey=True, squeeze=False)
    for ax, lr in zip(axes[0], learners, strict=True):
        sub = df.filter(pl.col("learner") == lr)
        for strat, ls, mk in [("uncertainty", "-", "o"), ("random", "--", "s")]:
            g = (sub.filter(pl.col("strategy") == strat).group_by("labels")
                 .agg(pl.col("f1").mean().alias("f1"), pl.col("f1").min().alias("lo"),
                      pl.col("f1").max().alias("hi")).sort("labels"))
            if g.is_empty():
                continue
            x, y = g["labels"].to_list(), g["f1"].to_list()
            ax.plot(x, y, ls, color=_C[strat], lw=2, marker=mk, ms=8,
                    markeredgecolor="white", markeredgewidth=2, label=strat)
            if sub["seed"].n_unique() > 1:
                ax.fill_between(x, g["lo"].to_list(), g["hi"].to_list(), color=_C[strat],
                                alpha=0.12, lw=0)
            ax.annotate(f"{strat} {y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                        xytext=(-6, 10 if strat == "uncertainty" else -16), ha="right",
                        color=_INK, fontsize=9)
        ax.set_xscale("log")
        ax.set_title({"lightgbm": "LightGBM (string features)",
                      "cross_encoder": "Cross-encoder (e5-small)"}[lr], color=_INK,
                     fontsize=11, loc="left")
        ax.set_xlabel("labelled pairs (log scale)", color=_INK2)
        ax.grid(True, color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbbbbb")
        ax.tick_params(colors=_INK2)
    axes[0][0].set_ylabel("test F1 at threshold 0.5", color=_INK2)
    axes[0][0].legend(frameon=False, loc="lower right", labelcolor=_INK)
    fig.suptitle("Active learning, hard level: labels vs F1", x=0.01, ha="left", color=_INK,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def report(cfg: DictConfig) -> Path:
    out = Path(cfg.paths.reports)
    lines = ["# Milestone 3 — cross-encoder, ablations, active learning (generated)", "",
             "Cascade = blocked candidates → LightGBM pre-filter (keeps "
             f"{cfg.ce.prefilter.keep_recall:.1%} of valid true candidates) → Ditto-style "
             f"cross-encoder (`{cfg.ce.model.name}`). Metrics over **all** blocked test "
             "candidates (pre-filtered pairs count as non-matches); `f1_e2e` also counts true "
             "pairs lost in blocking. Thresholds chosen on valid.", ""]
    m2 = _m2_baselines(cfg)
    if _json(cfg, "main").exists():
        df = pl.DataFrame(json.loads(_json(cfg, "main").read_text(encoding="utf-8")))
        if m2 is not None:
            best = (m2.filter(pl.col("model") != "splink").sort("f1", descending=True)
                    .group_by("level", "variant", maintain_order=True).first()
                    .select("level", pl.col("variant").alias("run"),
                            pl.col("model").alias("best_m2_model"),
                            pl.col("f1").alias("best_m2_f1"),
                            pl.col("f1_e2e").alias("best_m2_f1_e2e")))
            df = df.join(best, on=["level", "run"], how="left")
        lines += ["## Main results", "", *_table(df.sort("level", "run"), [
            "level", "run", "precision", "recall", "f1", "f1_e2e", "best_m2_model",
            "best_m2_f1", "best_m2_f1_e2e", "prefilter_kept", "prefilter_recall",
            "train_pairs", "best_epoch", "train_seconds", "score_seconds"]), ""]
    lv, var = cfg.ce.ablations.level, cfg.ce.ablations.variant
    if (Path(cfg.paths.processed) / lv / f"m3_scores_{var}.npz").exists():
        from ekam.match.slices import cascade_slices

        sl = cascade_slices(cfg, lv, var)
        lines += [f"## Where the gain comes from ({lv}, {var})", "",
                  "Recall of true test candidate pairs: LightGBM alone (M2 protocol) vs the "
                  "cascade, sliced by noise type on either record and by source pair. "
                  f"False positives: LightGBM {sl['fp_lgbm']}, cascade {sl['fp_cascade']}. "
                  "Leakage check: test records whose normalised name occurs in the train "
                  f"split: {sl['test_names_seen_in_train']:.2%} (entities are disjoint, so "
                  "memorising names cannot help).", "",
                  *_table(sl["table"], ["slice", "true_pairs", "lgbm_recall", "cascade_recall",
                                        "gain"]), ""]
    if _json(cfg, "ablations").exists():
        df = pl.DataFrame(json.loads(_json(cfg, "ablations").read_text(encoding="utf-8")))
        lines += [f"## Ablations ({cfg.ce.ablations.level}, {cfg.ce.ablations.variant}, "
                  f"{cfg.ce.ablations.train_pairs} training pairs each)", "",
                  *_table(df, ["run", "model", "fields", "name_column", "augment_p",
                               "precision", "recall", "f1", "f1_e2e", "best_epoch"]), ""]
    if _json(cfg, "al").exists():
        df = pl.DataFrame(json.loads(_json(cfg, "al").read_text(encoding="utf-8")),
                          infer_schema_length=None)
        plot_al(df, out / "M3_al.png")
        agg = (df.group_by("learner", "strategy", "labels")
               .agg(pl.col("f1").mean(), pl.col("f1").std().alias("f1_sd"),
                    pl.col("avg_precision").mean(), pl.col("pos_share").mean(),
                    pl.len().alias("seeds"))
               .sort("learner", "labels", "strategy"))
        lines += ["## Active learning", "",
                  f"Level {cfg.al.level}; pool = blocked train candidates, eval = "
                  f"{cfg.al.eval_pairs} blocked test candidates; F1 at a fixed 0.5 threshold, "
                  "AP = average precision. `pos_share` = share of matches among the labels.", "",
                  "![active learning](M3_al.png)", "",
                  *_table(agg, ["learner", "strategy", "labels", "f1", "f1_sd",
                                "avg_precision", "pos_share", "seeds"]), ""]
    path = out / "M3_results.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def run(cfg: DictConfig) -> Path:
    from ekam.match import ce_experiments as cx

    stages = list(cfg.pipeline.get("m3_stages") or STAGES)
    fns = {"main": cx.run_main, "ablations": cx.run_ablations, "al": cx.run_active}
    for st in stages:
        with mlflow_run(cfg, "ekam-m3", st, {"stage": st, "model": cfg.ce.model.name,
                                             "train_pairs": cfg.ce.train_pairs}) as (mlflow, _):
            rows = fns[st](cfg)
            _json(cfg, st).write_text(json.dumps(rows, indent=2, default=float),
                                      encoding="utf-8", newline="\n")
            for i, r in enumerate(rows):
                key = ".".join(str(r.get(k)) for k in ("level", "run", "learner", "strategy",
                                                        "seed", "labels") if k in r)
                mlflow.log_metrics({f"{key}.{m}": float(r[m]) for m in ("f1", "f1_e2e")
                                    if r.get(m) is not None}, step=i)
            log.info("stage %s: %d rows -> %s", st, len(rows), _json(cfg, st))
    path = report(cfg)
    log.info("M3 report -> %s", path)
    return path


@hydra.main(config_path=str(CONF_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    run(cfg)


if __name__ == "__main__":
    main()
