"""Privacy guard: keep only allow-listed, non-personal columns.

An *allow*-list (not a block-list) means a future MCA snapshot that adds director names,
e-mail or phone columns cannot leak them: unknown columns are dropped by construction.
"""

from __future__ import annotations

import logging
import re

import polars as pl

log = logging.getLogger(__name__)

# Defence in depth: refuse to emit any column whose *output* name looks personal, even if
# someone adds it to the allow-list by mistake.
_PERSONAL = re.compile(r"(director|din\b|email|e_mail|phone|mobile|contact|pan\b|aadhaar)", re.I)


def apply_allowlist(df: pl.DataFrame | pl.LazyFrame, allow: dict[str, str]):
    """Select and rename allow-listed columns; drop everything else."""
    bad = [out for out in allow.values() if _PERSONAL.search(out)]
    if bad:
        raise ValueError(f"allow-list contains personal-looking columns: {bad}")
    cols = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    present = {src: out for src, out in allow.items() if src in cols}
    dropped = [c for c in cols if c not in allow]
    if dropped:
        log.info("privacy: dropping %d non-allow-listed columns: %s", len(dropped), dropped)
    return df.select([pl.col(src).alias(out) for src, out in present.items()])
