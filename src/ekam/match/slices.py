"""Where does the cascade beat LightGBM? Recall of true test candidate pairs, sliced by the
noise types applied to either record (from the generator's ground-truth log) and by the
pair of source systems. Also the leakage check for name memorisation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig

from ekam.match.cascade import load_prefilter


def cascade_slices(cfg: DictConfig, level: str, tag: str, min_pairs: int = 500) -> dict:
    from ekam.match.synthetic import _load

    df, cand, y, _ = _load(cfg, level, "test")
    pf = load_prefilter(cfg, level)
    z = np.load(Path(cfg.paths.processed) / level / f"m3_scores_{tag}.npz")
    ce = np.zeros(len(y), np.int8)
    ce[z["test_rows"]] = z["test_scores"] >= z["threshold"]
    lg = (pf["test_scores"] >= pf["lgbm_threshold"]).astype(np.int8)

    gt = pl.read_parquet(Path(cfg.synth.out_dir) / level / "ground_truth.parquet",
                         columns=["record_id", "noise_applied"])
    rec = df.select("idx", "record_id", "source").join(gt, on="record_id").sort("idx")
    pos = np.flatnonzero(y == 1)
    pairs = pl.DataFrame({
        "l": cand[pos, 0], "r": cand[pos, 1], "lg": lg[pos], "ce": ce[pos]})
    noise = rec.select(pl.col("idx"), pl.col("noise_applied"), pl.col("source"))
    pairs = (pairs.join(noise.rename({"idx": "l", "noise_applied": "nl", "source": "sl"}), on="l")
             .join(noise.rename({"idx": "r", "noise_applied": "nr", "source": "sr"}), on="r"))
    by_noise = (pairs.with_columns(pl.concat_list("nl", "nr").list.unique().alias("t"))
                .explode("t").drop_nulls("t").group_by("t")
                .agg(pl.len().alias("true_pairs"), pl.col("lg").mean().alias("lgbm_recall"),
                     pl.col("ce").mean().alias("cascade_recall"))
                .rename({"t": "slice"}))
    by_src = (pairs.with_columns(
        pl.concat_str(pl.min_horizontal("sl", "sr"), pl.lit("+"), pl.max_horizontal("sl", "sr"))
        .alias("slice"))
        .group_by("slice").agg(pl.len().alias("true_pairs"),
                               pl.col("lg").mean().alias("lgbm_recall"),
                               pl.col("ce").mean().alias("cascade_recall"))
        .with_columns(("src:" + pl.col("slice")).alias("slice")))
    out = (pl.concat([by_noise, by_src]).filter(pl.col("true_pairs") >= min_pairs)
           .with_columns((pl.col("cascade_recall") - pl.col("lgbm_recall")).alias("gain"))
           .sort("gain", descending=True))
    train_names = (pl.read_parquet(Path(cfg.paths.processed) / level / "records_norm.parquet",
                                   columns=["name_key", "split"])
                   .filter(pl.col("split") == "train")["name_key"].drop_nulls().unique())
    neg = y == 0
    return {"table": out,
            "test_names_seen_in_train": float(df["name_key"].is_in(train_names).mean()),
            "fp_lgbm": int(lg[neg].sum()), "fp_cascade": int(ce[neg].sum())}
