"""Splink (Fellegi-Sunter) baseline, DuckDB backend.

Trained **without labels** on the train split (u by random sampling, m by EM on two
blocking passes, λ from deterministic rules). It then scores exactly the same blocked
candidate pairs as the other baselines (``register_blocked_pairs_for_predict``), so blocking
is held constant across matchers. The decision threshold on match probability is chosen on
the valid split, as for the supervised models.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from omegaconf import DictConfig

from ekam.utils.duck import connect

log = logging.getLogger(__name__)

COLUMNS = ["idx", "name_key", "street", "city_norm", "state_norm", "pin_norm", "first_token",
           "name_phonetic", "cin_norm", "gstin_norm"]


def _settings(scfg: DictConfig, with_ids: bool):
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on

    comps = [
        cl.JaroWinklerAtThresholds("name_key", list(scfg.name_thresholds)),
        cl.JaroWinklerAtThresholds("street", list(scfg.street_thresholds)),
        cl.ExactMatch("city_norm"),
        cl.ExactMatch("state_norm"),
        cl.ExactMatch("pin_norm"),
    ]
    if with_ids:
        comps += [cl.ExactMatch("cin_norm"), cl.ExactMatch("gstin_norm")]
    return SettingsCreator(
        link_type="dedupe_only", unique_id_column_name="idx", comparisons=comps,
        blocking_rules_to_generate_predictions=[block_on("pin_norm", "first_token"),
                                                block_on("name_phonetic")],
        retain_intermediate_calculation_columns=False,
    )


def train_splink(df_train: pl.DataFrame, scfg: DictConfig, with_ids: bool, seed: int,
                 memory_limit: str = "4GB") -> dict:
    from splink import Linker, block_on
    from splink.internals.duckdb.database_api import DuckDBAPI

    db = DuckDBAPI(connect(memory_limit))
    linker = Linker(db.register(df_train.select(COLUMNS).to_arrow()), _settings(scfg, with_ids),
                    log_level=logging.WARNING)
    linker.training.estimate_probability_two_random_records_match(
        [block_on("name_key", "pin_norm")], recall=0.5)
    linker.training.estimate_u_using_random_sampling(max_pairs=float(scfg.u_max_pairs), seed=seed)
    for cols in scfg.em_blocking:
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on(*cols), record_sample_proportion=1.0)
    return linker.misc.save_model_to_json()


def score_pairs(model: dict, df: pl.DataFrame, pairs: np.ndarray,
                memory_limit: str = "4GB") -> np.ndarray:
    """Splink match probability for each candidate pair (aligned with ``pairs``)."""
    from splink import Linker
    from splink.internals.duckdb.database_api import DuckDBAPI

    db = DuckDBAPI(connect(memory_limit))
    linker = Linker(db.register(df.select(COLUMNS).to_arrow()), model, log_level=logging.WARNING)
    bp = pl.DataFrame({"match_key": ["0"] * len(pairs), "join_key_l": pairs[:, 0],
                       "join_key_r": pairs[:, 1]})
    linker.table_management.register_blocked_pairs_for_predict(db.register(bp.to_arrow()))
    pred = linker.inference.predict().as_duckdbpyrelation().pl()
    key = pl.DataFrame({"idx_l": pairs[:, 0], "idx_r": pairs[:, 1]}).with_row_index("row")
    scored = key.join(pred.select(pl.col("idx_l"), pl.col("idx_r"), "match_probability"),
                      on=["idx_l", "idx_r"], how="left").sort("row")
    return scored["match_probability"].fill_null(0.0).to_numpy()
