"""Normalise every synthetic record once per level and cache it.

Output ``data/processed/<level>/records_norm.parquet``: one row per record with a dense
integer ``idx`` (row number, used by blocking/features), its ``split`` and normalised
fields. ``entity_id`` is joined in a *separate* column only for evaluation code; feature
code never reads it (tests assert this).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from ekam.normalize import normalize_record
from ekam.normalize.text import has_indic

log = logging.getLogger(__name__)

NORM_SCHEMA = {
    "name_clean": pl.Utf8, "name_base": pl.Utf8, "name_key": pl.Utf8, "legal_form": pl.Utf8,
    "name_phonetic": pl.Utf8, "first_token": pl.Utf8, "street": pl.Utf8, "city_norm": pl.Utf8,
    "state_norm": pl.Utf8, "pin_norm": pl.Utf8, "pin_state_ok": pl.Boolean,
    "cin_norm": pl.Utf8, "gstin_norm": pl.Utf8, "name_indic": pl.Boolean,
}


def _normalise_chunk(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        n = normalize_record(r)
        n["name_indic"] = has_indic(r.get("name") or "")
        out.append(n)
    return out


def normalise_records(records: pl.DataFrame, n_workers: int = 1,
                      chunk: int = 20_000) -> pl.DataFrame:
    rows = records.select("name", "address", "city", "state", "pin", "cin", "gstin").to_dicts()
    chunks = [rows[i : i + chunk] for i in range(0, len(rows), chunk)]
    if n_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            parts = list(ex.map(_normalise_chunk, chunks))
    else:
        parts = [_normalise_chunk(c) for c in chunks]
    norm = pl.DataFrame([r for p in parts for r in p], schema=NORM_SCHEMA)
    norm = norm.with_columns(pl.col(pl.Utf8).replace("", None))
    return pl.concat([records.select("record_id", "source"), norm], how="horizontal")


def prepared_path(cfg: DictConfig, level: str) -> Path:
    return Path(cfg.paths.processed) / level / "records_norm.parquet"


def prepare_level(cfg: DictConfig, level: str, force: bool = False) -> Path:
    out = prepared_path(cfg, level)
    if out.exists() and not force:
        return out
    d = Path(cfg.synth.out_dir) / level
    rec = pl.read_parquet(d / "records.parquet")
    gt = pl.read_parquet(d / "ground_truth.parquet").select("record_id", "entity_id")
    split = pl.read_parquet(cfg.split.out_path).select("entity_id", "split")
    norm = normalise_records(rec, n_workers=int(cfg.hardware.n_workers))
    norm = (norm.join(gt, on="record_id").join(split, on="entity_id")
            .sort("split", "record_id").with_row_index("idx"))
    out.parent.mkdir(parents=True, exist_ok=True)
    norm.write_parquet(out)
    log.info("prepared %s: %d records -> %s", level, norm.height, out)
    return out


def load_split(cfg: DictConfig, level: str, split: str) -> pl.DataFrame:
    """Records of one split with a split-local dense ``idx`` (0..n-1)."""
    df = pl.read_parquet(prepared_path(cfg, level)).filter(pl.col("split") == split)
    return df.drop("idx").with_row_index("idx")
