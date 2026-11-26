"""MCA raw CSVs -> one clean entity table (``data/interim/mca_entities.parquet``).

Each row is one real company (one CIN) = one ground-truth entity. The raw data is itself
somewhat messy (suffix variants, ~23% of addresses lack a PIN, inconsistent city
spellings); we parse it but do not "fix" it, because those values are what the noise
generator starts from.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from ekam.data.privacy import apply_allowlist
from ekam.normalize import normalize_cin, parse_address

log = logging.getLogger(__name__)


def read_mca_csv(path: Path, allow: dict[str, str]) -> pl.DataFrame:
    df = pl.read_csv(
        path, infer_schema=False, null_values=["NA", ""], encoding="utf8-lossy", quote_char='"'
    )
    return apply_allowlist(df, allow)


def _parse_addresses(addresses: list[str | None], states: list[str | None]) -> pl.DataFrame:
    rows = []
    for a, s in zip(addresses, states, strict=True):
        p = parse_address(a, state_hint=s)
        rows.append({"street": p.street, "city": p.city, "pin": p.pin, "addr_state": p.state})
    return pl.DataFrame(
        rows, schema={"street": pl.Utf8, "city": pl.Utf8, "pin": pl.Utf8, "addr_state": pl.Utf8}
    )


def build_entities(cfg: DictConfig) -> pl.DataFrame:
    raw_dir = Path(cfg.data.raw_dir)
    allow = dict(cfg.data.allow_columns)
    files = [raw_dir / f"{s}.csv" for s in cfg.data.states]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"run the download first; missing {missing}")

    df = pl.concat([read_mca_csv(f, allow) for f in files], how="diagonal_relaxed")
    df = df.with_columns(pl.col(pl.Utf8).str.strip_chars().str.replace_all(r"\s+", " "))
    n0 = df.height
    df = df.filter(pl.col("name").is_not_null() & (pl.col("name").str.len_chars() > 1))
    df = df.unique(subset=["cin"], keep="first", maintain_order=True)
    log.info("ingest: %d raw rows -> %d after dropping empty names / duplicate CINs", n0, df.height)

    parsed = _parse_addresses(df["address"].to_list(), df["state"].to_list())
    df = pl.concat([df, parsed], how="horizontal")
    df = df.with_columns(
        pl.col("cin").map_elements(lambda c: normalize_cin(c) is not None, pl.Boolean)
        .alias("cin_valid"),
        pl.col("date_of_registration").str.to_date("%d-%m-%Y", strict=False),
        pl.coalesce("addr_state", "state").alias("state"),
    ).drop("addr_state")

    # Deterministic entity ids that do not encode the CIN or any field value.
    df = df.sort("cin").with_row_index("_i").with_columns(
        pl.format("E{}", pl.col("_i").cast(pl.Utf8).str.zfill(7)).alias("entity_id")
    ).drop("_i")
    order = ["entity_id", "cin", "cin_valid", "name", "address", "street", "city", "state", "pin",
             "status", "company_class", "category", "sub_category", "activity", "roc",
             "date_of_registration"]
    return df.select([c for c in order if c in df.columns])


def run(cfg: DictConfig) -> Path:
    out = Path(cfg.data.clean_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_entities(cfg)
    df.write_parquet(out)
    log.info("wrote %s (%d entities)", out, df.height)
    return out


if __name__ == "__main__":
    import sys

    from ekam.utils.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(sys.argv[1:]))
