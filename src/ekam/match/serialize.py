"""Ditto-style serialisation of records for a cross-encoder.

A record becomes ``COL name VAL sharma traders pvt ltd COL street VAL ... COL city VAL ...``
(Li et al., *Deep Entity Matching with Pre-Trained Language Models*, VLDB 2021). A pair is
fed to the encoder as ``[CLS] left [SEP] right [SEP]``. Missing values keep their ``COL``
tag with an empty ``VAL`` so the model can learn that "absent" is not "different".
"""

from __future__ import annotations

import polars as pl

# logical field -> column in records_norm (``name`` can be switched to the raw source name)
FIELD_COLUMNS = {
    "name": "name_clean",
    "street": "street",
    "city": "city_norm",
    "state": "state_norm",
    "pin": "pin_norm",
    "cin": "cin_norm",
    "gstin": "gstin_norm",
}


def serialize_records(df: pl.DataFrame, fields: list[str], name_column: str = "name_clean",
                      lowercase: bool = True) -> list[str]:
    """One string per row of ``df`` (row order kept)."""
    cols = {f: (name_column if f == "name" else FIELD_COLUMNS[f]) for f in fields}
    parts = []
    for f, c in cols.items():
        v = pl.col(c).cast(pl.Utf8).fill_null("")
        if lowercase:
            v = v.str.to_lowercase()
        parts += [pl.lit(f"COL {f} VAL "), v, pl.lit(" ")]
    return df.select(pl.concat_str(parts).str.strip_chars().alias("s"))["s"].to_list()
