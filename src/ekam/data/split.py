"""Entity-disjoint train/valid/test split, stratified by cluster size and state.

The split is assigned per *entity*; every record inherits its entity's split, so no true
entity ever has records on both sides. Because generator structure is level-independent,
one split file serves easy, medium and hard.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import polars as pl
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def bucket_label(size: int, edges: list[int]) -> str:
    edges = sorted(edges)
    for lo, hi in zip(edges, edges[1:] + [None], strict=True):
        if hi is None or size < hi:
            if hi is None:
                return f"{lo}+"
            return str(lo) if hi - lo == 1 else f"{lo}-{hi - 1}"
    return f"{edges[-1]}+"


def assign_splits(entities: pl.DataFrame, fractions: dict[str, float], seed: int) -> pl.DataFrame:
    """entities: [entity_id, stratum] -> adds `split`. Deterministic for a seed.

    Within each stratum entities are shuffled and cut by the fractions, so each split gets
    (to within one entity) the requested share of every stratum.
    """
    names = list(fractions)
    total = sum(fractions.values())
    cum, acc = [], 0.0
    for n in names:
        acc += fractions[n] / total
        cum.append(acc)
    out = []
    for (stratum,), grp in entities.sort("entity_id").group_by(["stratum"], maintain_order=True):
        ids = grp["entity_id"].to_list()
        random.Random(f"{seed}|split|{stratum}").shuffle(ids)
        n = len(ids)
        start = 0
        for name, c in zip(names, cum, strict=True):
            end = n if name == names[-1] else round(c * n)
            out.extend((eid, name) for eid in ids[start:end])
            start = end
    return entities.join(pl.DataFrame(out, schema=["entity_id", "split"], orient="row"),
                         on="entity_id")


def build_entity_split(cfg: DictConfig, ground_truth: pl.DataFrame,
                       entities: pl.DataFrame) -> pl.DataFrame:
    sizes = ground_truth.group_by("entity_id").len("cluster_size")
    df = entities.select("entity_id", "state").join(sizes, on="entity_id", how="left")
    df = df.with_columns(pl.col("cluster_size").fill_null(0))
    edges = list(cfg.split.cluster_size_buckets)
    df = df.with_columns(
        pl.col("cluster_size").map_elements(
            lambda s: "0" if s == 0 else bucket_label(s, edges), pl.Utf8
        ).alias("cluster_size_bucket")
    )
    df = df.with_columns(pl.concat_str(["cluster_size_bucket", "state"], separator="|",
                                       ignore_nulls=True).alias("stratum"))
    out = assign_splits(df, dict(cfg.split.fractions), int(cfg.seed))
    return out.select("entity_id", "split", "cluster_size", "cluster_size_bucket", "state")


def run(cfg: DictConfig) -> Path:
    synth_dir = Path(cfg.synth.out_dir)
    gt = pl.read_parquet(synth_dir / cfg.synth.levels[0] / "ground_truth.parquet")
    from ekam.synth.generator import load_entities

    ents = load_entities(cfg)  # the same (possibly sampled) entity set the generator used
    split = build_entity_split(cfg, gt, ents)
    out = Path(cfg.split.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    split.write_parquet(out)
    log.info("entity split -> %s: %s", out, dict(split.group_by("split").len().iter_rows()))
    return out
