"""Data audit: cluster sizes, field completeness, configured vs measured noise, leakage.

Writes ``docs/reports/M1_audit.md`` (+ JSON) and logs headline numbers to MLflow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from ekam.data import leakage
from ekam.normalize import normalize_cin, normalize_pin
from ekam.synth.generator import CANONICAL, build_level, build_sources, rate_is_unconditional

log = logging.getLogger(__name__)

FIELDS = ["name", "address", "city", "state", "pin", "cin", "gstin"]


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def cluster_size_table(gt: pl.DataFrame) -> pl.DataFrame:
    sizes = gt.group_by("entity_id").len("size")
    return (sizes.group_by("size").len("entities").sort("size")
            .with_columns((pl.col("entities") / pl.col("entities").sum()).alias("share")))


def completeness(records: pl.DataFrame, sources) -> pl.DataFrame:
    rows = []
    for s in sources:
        sub = records.filter(pl.col("source") == s.name)
        row = {"source": s.name, "records": sub.height}
        for f in FIELDS:
            present_in_schema = f == "name" or bool(s.columns.get(f)) or (
                f == "address" and s.address_format == "single_line")
            if not present_in_schema or (s.address_format == "single_line"
                                         and f in ("city", "state", "pin")):
                row[f] = None
            else:
                row[f] = float(sub[f].is_not_null().mean()) if sub.height else 0.0
        rows.append(row)
    return pl.DataFrame(rows)


def noise_rates(gt: pl.DataFrame, sources, level) -> pl.DataFrame:
    rows = []
    for s in sources:
        sub = gt.filter(pl.col("source") == s.name)
        n = sub.height
        if not n:
            continue
        applied = sub.select(pl.col("noise_applied").explode()).to_series().value_counts()
        inel = sub.select(pl.col("noise_ineligible").explode()).to_series().value_counts()
        a = dict(zip(applied[:, 0].to_list(), applied["count"].to_list(), strict=True))
        b = dict(zip(inel[:, 0].to_list(), inel["count"].to_list(), strict=True))
        for t, base in sorted(s.noise.items()):
            cfg_rate = min(base * level.multiplier, level.rate_cap)
            rows.append({"source": s.name, "noise": t, "configured": cfg_rate,
                         "rolled": (a.get(t, 0) + b.get(t, 0)) / n, "applied": a.get(t, 0) / n,
                         "unconditional": rate_is_unconditional(s, t)})
    return pl.DataFrame(rows)


def realised_noise(records: pl.DataFrame, gt: pl.DataFrame, entities: pl.DataFrame) -> pl.DataFrame:
    """Data-level (not log-level) measures: how often fields still equal the clean value."""
    j = (records.join(gt.select("record_id", "entity_id"), on="record_id")
         .join(entities.select("entity_id", pl.col("name").alias("e_name"),
                               pl.col("pin").alias("e_pin"), pl.col("cin").alias("e_cin")),
               on="entity_id"))
    # single-line sources keep the PIN inside the address: measure it there
    pin_src = pl.coalesce("pin", pl.col("address").str.extract(r"(?:^|\D)([1-9]\d{5})\s*$", 1))
    j = j.with_columns(
        pin_src.map_elements(normalize_pin, pl.Utf8, skip_nulls=True).alias("pin_n"),
        pl.col("cin").map_elements(normalize_cin, pl.Utf8, skip_nulls=True).alias("cin_n"),
    )
    return (j.group_by("source").agg(
        (pl.col("name") == pl.col("e_name")).mean().alias("name_identical"),
        (pl.col("name").str.contains(r"[ऀ-ॿ]")).mean().alias("name_devanagari"),
        (pl.col("pin_n") == pl.col("e_pin")).fill_null(False).mean().alias("pin_correct"),
        (pl.col("pin_n").is_not_null() & (pl.col("pin_n") != pl.col("e_pin")))
        .fill_null(False).mean().alias("pin_wrong"),
        (pl.col("cin_n") == pl.col("e_cin")).fill_null(False).mean().alias("cin_intact"),
    ).sort("source"))


def translit_round_trip(entities: pl.DataFrame, n: int = 3000, seed: int = 0) -> dict:
    """How often the normaliser recovers the exact name key after script/romanisation noise.

    Exact-key recovery is an *upper bound* on how easy the noise is: the normaliser shares
    the core lexicon with the generator, but not the held-out words (hindi_lexicon_heldout).
    """
    import random

    from ekam.normalize import normalize_name
    from ekam.synth.noise.names import devanagari, translit_variant

    names = entities.sample(min(n, entities.height), seed=seed)["name"].to_list()
    rng = random.Random(seed)
    dev = sum(normalize_name(devanagari(x, rng)).key == normalize_name(x).key for x in names)
    tv = sum(normalize_name(translit_variant(x, rng)).key == normalize_name(x).key for x in names)
    return {"devanagari_exact_key": dev / len(names), "translit_variant_exact_key": tv / len(names),
            "sample": len(names)}


def run(cfg: DictConfig) -> dict:
    from ekam.synth.generator import load_entities

    synth_dir = Path(cfg.synth.out_dir)
    entities = load_entities(cfg)
    split = pl.read_parquet(cfg.split.out_path)
    sources = build_sources(cfg)
    leakage.check_entity_disjoint(split)

    md = ["# Milestone 1 — data audit", "",
          f"Entities: **{entities.height:,}** (MCA subset: {', '.join(cfg.data.states)}). "
          f"Seed {cfg.seed}. Generated by `make data`; regenerate to refresh.", ""]
    summary: dict = {"entities": entities.height, "levels": {}}

    split_counts = split.group_by("split").agg(pl.len().alias("entities"),
                                               pl.col("cluster_size").sum().alias("records"))
    md += ["## Entity split (shared by all levels)", "", _md_table(split_counts.sort("split")), ""]
    hom = leakage.report_cross_split_homonyms(entities, split)
    md += ["Real homonyms in the register (distinct CINs, same normalised name + PIN): "
           f"{hom['homonym_groups']} groups / {hom['entities_in_homonym_groups']} entities, "
           f"{hom['homonym_groups_cross_split']} groups span splits (allowed; they are true "
           "negatives).", ""]
    summary["homonyms"] = hom
    tq = translit_round_trip(entities)
    md += ["## Script / romanisation noise: how much does the normaliser undo?", "",
           f"On {tq['sample']:,} sampled names, the normalised key is recovered exactly after "
           f"Devanagari rendering in **{tq['devanagari_exact_key']:.1%}** of cases and after "
           f"romanisation variants in **{tq['translit_variant_exact_key']:.1%}**. The misses "
           "are held-out loanwords (e.g. लाइफ → LAIF) and re-spelled names that folding does "
           "not collapse; the matcher must handle them with fuzzy/phonetic similarity.", ""]
    summary["translit_round_trip"] = tq

    for lv_name in cfg.synth.levels:
        level = build_level(cfg, lv_name)
        d = synth_dir / lv_name
        rec = pl.read_parquet(d / "records.parquet")
        gt = pl.read_parquet(d / "ground_truth.parquet")
        assert rec.columns == CANONICAL
        leakage.check_records_follow_entity(gt, split)
        leakage.check_no_truth_in_records(rec, gt)
        order = leakage.report_order_leak(rec, gt)
        ids = leakage.report_identifier_shortcut(rec, gt)
        cs = cluster_size_table(gt)
        comp = completeness(rec, sources)
        nr = noise_rates(gt, sources, level)
        real = realised_noise(rec, gt, entities)
        unc = nr.filter(pl.col("unconditional"))
        max_dev = float((unc["rolled"] - unc["configured"]).abs().max())

        md += [f"## Level: {lv_name} (rate multiplier ×{level.multiplier})", "",
               f"Records: **{rec.height:,}**; true matching pairs: **{ids['true_pairs']:,}**.",
               "", "### Cluster size (records per entity)", "", _md_table(cs), "",
               "### Field completeness (share non-null; — = field not in that source's schema)",
               "", _md_table(comp), "",
               "### Realised noise (measured on the data, vs the clean MCA value)", "",
               _md_table(real), "",
               "### Leakage / shortcut checks", "",
               "- Entity-disjoint split, record→entity→split consistency, no entity_id in any "
               "field: **pass**",
               f"- Adjacent rows from the same entity (file-order leak): {order:.4f} "
               f"(random order ≈ 0)",
               f"- Exact-identifier shortcut (valid CIN or GSTIN equal): recall "
               f"{ids['id_shortcut_recall']:.3f}, precision {ids['id_shortcut_precision']:.4f}, "
               f"{ids['id_shortcut_false_pairs']} false pairs. **Every M2+ result is also "
               f"reported with identifier features removed.**", "",
               f"### Configured vs logged noise rates (max |rolled − configured| = "
               f"{max_dev:.3f})", "",
               "`rolled` = the noise was drawn; `applied` = it changed the value (the "
               "difference is noise that was not applicable, e.g. no PIN to corrupt). "
               "`unconditional = false` marks types drawn only when another did not fire "
               "(suffix variant vs drop, city alias vs held-out alias) or whose field is not "
               "in the source schema; they are excluded from the max-deviation check.", "",
               _md_table(nr), ""]
        summary["levels"][lv_name] = {
            "records": rec.height, **ids, "order_leak": order, "max_rate_dev": max_dev,
            "mean_cluster_size": float(gt.group_by("entity_id").len()["len"].mean()),
        }

    out_dir = Path(cfg.paths.reports)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "M1_audit.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "M1_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log_mlflow(cfg, summary, out_dir / "M1_audit.md")
    log.info("audit -> %s", out_dir / "M1_audit.md")
    return summary


def _log_mlflow(cfg: DictConfig, summary: dict, report: Path) -> None:
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        return
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    if mlflow.get_experiment_by_name(cfg.mlflow.experiment) is None:
        mlflow.create_experiment(cfg.mlflow.experiment,
                                 artifact_location=Path(cfg.mlflow.artifact_location)
                                 .resolve().as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name="m1-data-audit"):
        mlflow.log_params({"seed": cfg.seed, "entities": summary["entities"],
                           "states": ",".join(cfg.data.states)})
        for lv, m in summary["levels"].items():
            mlflow.log_metrics({f"{lv}.{k}": float(v) for k, v in m.items()})
        mlflow.log_artifact(str(report))
