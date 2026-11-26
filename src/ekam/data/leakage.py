"""Leakage checks: entity-disjoint splits and ground-truth shortcuts in the generator.

``check_*`` functions raise ``LeakageError``; ``report_*`` functions only measure (used in
the audit so that "too good to be true" results can be traced to a shortcut).
"""

from __future__ import annotations

import polars as pl

from ekam.normalize import normalize_cin, normalize_gstin, normalize_name


class LeakageError(AssertionError):
    pass


def check_entity_disjoint(split: pl.DataFrame) -> None:
    dup = split.group_by("entity_id").agg(pl.col("split").n_unique().alias("k"))
    bad = dup.filter(pl.col("k") > 1)
    if bad.height:
        raise LeakageError(f"{bad.height} entities appear in more than one split")
    if split["entity_id"].n_unique() != split.height:
        raise LeakageError("duplicate entity rows in split table")


def check_records_follow_entity(ground_truth: pl.DataFrame, split: pl.DataFrame) -> pl.DataFrame:
    """Every record maps to exactly one entity that has exactly one split."""
    if ground_truth["record_id"].n_unique() != ground_truth.height:
        raise LeakageError("record_id is not unique")
    j = ground_truth.join(split.select("entity_id", "split"), on="entity_id", how="left")
    if j["split"].null_count():
        raise LeakageError(f"{j['split'].null_count()} records whose entity has no split")
    per_entity = j.group_by("entity_id").agg(pl.col("split").n_unique().alias("k"))
    if per_entity["k"].max() > 1:
        raise LeakageError("an entity's records ended up in different splits")
    return j


def check_no_truth_in_records(records: pl.DataFrame, ground_truth: pl.DataFrame) -> None:
    """No record field may contain its entity_id; record_id must not equal entity_id."""
    if "entity_id" in records.columns:
        raise LeakageError("records table contains entity_id")
    j = records.join(ground_truth.select("record_id", "entity_id"), on="record_id")
    for col in [c for c in records.columns if c not in ("record_id", "source")]:
        hit = j.filter(pl.col(col).is_not_null() & pl.col(col).str.contains(pl.col("entity_id"),
                                                                             literal=True))
        if hit.height:
            raise LeakageError(f"column {col!r} contains entity_id in {hit.height} records")


def report_order_leak(records: pl.DataFrame, ground_truth: pl.DataFrame) -> float:
    """Share of adjacent rows (file order) that belong to the same entity. Should be ~0."""
    j = records.select("record_id").join(ground_truth.select("record_id", "entity_id"),
                                         on="record_id", maintain_order="left")
    e = j["entity_id"]
    return float((e.head(-1) == e.tail(-1)).mean()) if j.height > 1 else 0.0


def report_cross_split_homonyms(entities: pl.DataFrame, split: pl.DataFrame) -> dict:
    """Distinct real entities whose normalised name key + PIN coincide, across splits.

    These are genuine homonyms in the MCA register (allowed), but we count them because
    they make test pairs look like training pairs.
    """
    keys = entities.select(
        "entity_id",
        pl.col("name").map_elements(lambda n: normalize_name(n).key, pl.Utf8).alias("k"),
        pl.col("pin").fill_null(""),
    ).join(split.select("entity_id", "split"), on="entity_id")
    grp = keys.group_by("k", "pin").agg(pl.col("split").n_unique().alias("n_splits"),
                                        pl.len().alias("n"))
    multi = grp.filter(pl.col("n") > 1)
    return {
        "homonym_groups": multi.height,
        "homonym_groups_cross_split": multi.filter(pl.col("n_splits") > 1).height,
        "entities_in_homonym_groups": int(multi["n"].sum() or 0),
    }


def report_identifier_shortcut(records: pl.DataFrame, ground_truth: pl.DataFrame) -> dict:
    """Recall/precision an exact-identifier matcher would get (valid CIN or GSTIN equal).

    If this is high, a model can 'win' by memorising IDs; M2+ therefore also reports every
    result with identifier features removed.
    """
    r = records.select(
        "record_id",
        pl.col("cin").map_elements(normalize_cin, pl.Utf8, skip_nulls=True).alias("c"),
        pl.col("gstin").map_elements(normalize_gstin, pl.Utf8, skip_nulls=True).alias("g"),
    ).join(ground_truth.select("record_id", "entity_id"), on="record_id")
    n = r.group_by("entity_id").len()
    true_pairs = int((n["len"] * (n["len"] - 1) // 2).sum())

    def pairs_by(col: str) -> pl.DataFrame:
        a = r.filter(pl.col(col).is_not_null()).select("record_id", "entity_id", col)
        return (a.join(a, on=col, suffix="_b")
                .filter(pl.col("record_id") < pl.col("record_id_b"))
                .select("record_id", "record_id_b", "entity_id", "entity_id_b"))

    both = pl.concat([pairs_by("c"), pairs_by("g")]).unique(["record_id", "record_id_b"])
    tp = both.filter(pl.col("entity_id") == pl.col("entity_id_b")).height
    fp = both.height - tp
    return {
        "true_pairs": true_pairs,
        "id_shortcut_recall": tp / true_pairs if true_pairs else 0.0,
        "id_shortcut_precision": tp / both.height if both.height else 1.0,
        "id_shortcut_false_pairs": fp,
    }
