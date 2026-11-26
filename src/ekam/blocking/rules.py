"""Rule-based blocking with DuckDB: equi-joins on normalised keys, capped block sizes."""

from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

from ekam.utils.duck import connect


def _key_expr(fields: list[str]) -> str:
    parts = " || '|' || ".join(f'"{f}"' for f in fields)
    not_null = " AND ".join(f'"{f}" IS NOT NULL' for f in fields)
    return f"CASE WHEN {not_null} THEN {parts} END"


def rule_pairs(df: pl.DataFrame, rules: dict[str, list[str]], max_block_size: int,
               con: duckdb.DuckDBPyConnection | None = None) -> dict[str, np.ndarray]:
    """Candidate pairs per rule as int64 arrays of shape (m, 2) with left < right.

    ``df`` must have a dense ``idx`` column plus the fields used by the rules. Keys with
    more than ``max_block_size`` records are skipped entirely (they are stop-word-like and
    would add O(n^2) pairs).
    """
    con = con or connect()
    fields = sorted({f for fs in rules.values() for f in fs})
    con.register("recs", df.select("idx", *fields).to_arrow())
    out = {}
    for name, fs in rules.items():
        q = f"""
        WITH k AS (SELECT idx, {_key_expr(list(fs))} AS key FROM recs),
             ok AS (SELECT key FROM k WHERE key IS NOT NULL GROUP BY key
                    HAVING count(*) BETWEEN 2 AND {int(max_block_size)})
        SELECT a.idx AS l, b.idx AS r
        FROM k a JOIN k b ON a.key = b.key AND a.idx < b.idx
        WHERE a.key IN (SELECT key FROM ok)
        """
        res = con.execute(q).fetchnumpy()
        out[name] = np.stack([res["l"], res["r"]], axis=1).astype(np.int64) if len(res["l"]) \
            else np.empty((0, 2), np.int64)
    con.unregister("recs")
    return out


def union_pairs(*arrays: np.ndarray) -> np.ndarray:
    arrs = [a for a in arrays if len(a)]
    if not arrs:
        return np.empty((0, 2), np.int64)
    allp = np.concatenate(arrs)
    allp = np.sort(allp, axis=1)  # enforce l < r
    allp = allp[allp[:, 0] != allp[:, 1]]
    return np.unique(allp, axis=0)
