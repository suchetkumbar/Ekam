from pathlib import Path

import polars as pl
import pytest

from ekam.data import leakage
from ekam.data.split import assign_splits, bucket_label, build_entity_split
from ekam.synth.generator import generate_for_entities

from .conftest import as_frame


def _gt(many_entities, sources, levels):
    _, truth = generate_for_entities(many_entities, sources, levels["medium"], 4)
    return pl.DataFrame(truth).drop("noise_applied", "noise_ineligible")


def test_bucket_labels():
    edges = [1, 2, 3, 5, 8]
    assert [bucket_label(s, edges) for s in [1, 2, 3, 4, 5, 7, 8, 20]] == \
        ["1", "2", "3-4", "3-4", "5-7", "5-7", "8+", "8+"]


def test_split_is_entity_disjoint_and_records_follow(cfg, many_entities, sources, levels):
    gt = _gt(many_entities, sources, levels)
    split = build_entity_split(cfg, gt, as_frame(many_entities))
    leakage.check_entity_disjoint(split)
    j = leakage.check_records_follow_entity(gt, split)
    train = set(j.filter(pl.col("split") == "train")["entity_id"])
    test = set(j.filter(pl.col("split") == "test")["entity_id"])
    assert not train & test
    shares = split.group_by("split").len().to_dict(as_series=False)
    share = dict(zip(shares["split"], shares["len"], strict=True))
    assert abs(share["train"] / split.height - 0.70) < 0.05


def test_split_is_deterministic(cfg, many_entities, sources, levels):
    gt = _gt(many_entities, sources, levels)
    a = build_entity_split(cfg, gt, as_frame(many_entities)).sort("entity_id")
    b = build_entity_split(cfg, gt, as_frame(many_entities)).sort("entity_id")
    assert a.equals(b)


def test_leaky_split_is_detected(many_entities, sources, levels):
    gt = _gt(many_entities, sources, levels)
    split = pl.DataFrame({"entity_id": ["E0000010", "E0000010"], "split": ["train", "test"]})
    with pytest.raises(leakage.LeakageError):
        leakage.check_entity_disjoint(split)
    partial = pl.DataFrame({"entity_id": [gt["entity_id"][0]], "split": ["train"]})
    with pytest.raises(leakage.LeakageError):
        leakage.check_records_follow_entity(gt, partial)


def test_stratification_balances_rare_strata():
    ents = pl.DataFrame({"entity_id": [f"E{i}" for i in range(1000)],
                         "stratum": ["big"] * 900 + ["rare"] * 100})
    out = assign_splits(ents, {"train": 0.7, "valid": 0.15, "test": 0.15}, seed=1)
    rare = out.filter(pl.col("stratum") == "rare").group_by("split").len()
    counts = dict(rare.iter_rows())
    assert counts == {"train": 70, "valid": 15, "test": 15}


# ---- the real generated data (skipped until `make data` has run) -----------------------

_SPLIT = Path("data/processed/entity_split.parquet")


@pytest.mark.slow
@pytest.mark.skipif(not _SPLIT.exists(), reason="run `make data` first")
@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_generated_data_passes_leakage_checks(level):
    split = pl.read_parquet(_SPLIT)
    leakage.check_entity_disjoint(split)
    d = Path("data/synth") / level
    gt = pl.read_parquet(d / "ground_truth.parquet")
    rec = pl.read_parquet(d / "records.parquet")
    leakage.check_records_follow_entity(gt, split)
    leakage.check_no_truth_in_records(rec.sample(min(50_000, rec.height), seed=0), gt)
    assert leakage.report_order_leak(rec, gt) < 0.01
