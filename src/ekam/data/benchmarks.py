"""Uniform view of the downloaded EM benchmarks (DeepMatcher layout and WDC Products).

DeepMatcher/Magellan layout: tableA.csv, tableB.csv, {train,valid,test}.csv with
``ltable_id, rtable_id, label``. WDC Products ships pair files with ``label`` and
``*_left`` / ``*_right`` columns. ``summarise`` reports what each one contains for DATA.md.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from omegaconf import DictConfig


def _find(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def load_deepmatcher(root: Path) -> dict[str, pl.DataFrame]:
    out = {}
    for part in ["tableA", "tableB", "train", "valid", "test"]:
        p = _find(root, f"{part}.csv")
        if p is not None:
            out[part] = pl.read_csv(p, infer_schema=False, encoding="utf8-lossy")
    return out


def load_wdc(root: Path, files: dict[str, str]) -> dict[str, pl.DataFrame]:
    """Load the configured WDC Products pair files (split name -> DataFrame)."""
    out = {}
    for split, fname in files.items():
        p = _find(root, fname)
        if p is not None:
            out[split] = pl.read_ndjson(p, infer_schema_length=10_000)
    return out


def summarise(cfg: DictConfig) -> pl.DataFrame:
    rows = []
    for name, spec in cfg.benchmarks.datasets.items():
        root = Path(cfg.benchmarks.out_dir) / name
        if not root.exists():
            rows.append({"dataset": name, "status": "missing"})
            continue
        if spec.family == "deepmatcher":
            t = load_deepmatcher(root)
            row = {"dataset": name, "status": "ok",
                   "tableA": t.get("tableA", pl.DataFrame()).height,
                   "tableB": t.get("tableB", pl.DataFrame()).height,
                   "attributes": ",".join(c for c in t.get("tableA", pl.DataFrame()).columns
                                          if c != "id")}
            for part in ["train", "valid", "test"]:
                df = t.get(part)
                row[part] = df.height if df is not None else 0
                row[f"{part}_pos"] = int((df["label"] == "1").sum()) if df is not None else 0
            rows.append(row)
        else:
            parts = load_wdc(root, dict(spec.files))
            row = {"dataset": name, "status": "ok",
                   "attributes": ",".join(sorted({c.rsplit("_", 1)[0]
                                                  for df in parts.values() for c in df.columns
                                                  if c.endswith("_left")}))}
            for part, df in parts.items():
                row[part] = df.height
                row[f"{part}_pos"] = int((df["label"].cast(pl.Utf8) == "1").sum())
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)
