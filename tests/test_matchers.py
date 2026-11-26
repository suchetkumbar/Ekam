import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from ekam.match.baselines import make_matcher, subsample_training
from ekam.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = np.c_[y + 0.3 * rng.normal(size=n), rng.normal(size=n)]
    x[rng.random(n) < 0.1, 0] = np.nan  # missing values must be tolerated
    return x, y


@pytest.mark.parametrize("kind", ["logreg", "lightgbm"])
def test_matchers_learn_a_separable_signal(cfg, kind):
    x, y = _toy()
    xv, yv = _toy(seed=1)
    m = make_matcher(kind, cfg, 0).fit(x, y, xv, yv)
    s = m.score(xv)
    assert s.shape == (len(yv),)
    assert ((s > 0.5) == yv).mean() > 0.85


def test_subsample_keeps_all_positives():
    y = np.array([1] * 10 + [0] * 1000)
    keep = subsample_training(y, cap=110, seed=0)
    assert y[keep].sum() == 10 and len(keep) == 110
    assert (np.diff(keep) > 0).all()


def test_splink_scores_align_with_registered_pairs(cfg):
    pytest.importorskip("splink")
    from ekam.match.splink_model import score_pairs, train_splink

    rng = np.random.default_rng(0)
    names = [f"COMPANY {i} TRADERS" for i in range(300)]
    rows = []
    for i, nm in enumerate(names):
        for k in range(2):  # every entity twice -> matching pairs exist
            exact = k == 0 or i % 2 == 0
            rows.append({"name_key": nm if exact else nm.replace("TRADERS", "TRADRS"),
                         "street": f"{i} MG ROAD", "city_norm": rng.choice(["Jaipur", "Pune"]),
                         "state_norm": "Rajasthan", "pin_norm": f"30{i % 50:04d}",
                         "first_token": "COMPANY", "name_phonetic": f"K{i}",
                         "cin_norm": None, "gstin_norm": None})
    df = pl.DataFrame(rows).with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int64))
    scfg = OmegaConf.merge(cfg.matcher.splink, {"u_max_pairs": 20000,
                                                "em_blocking": [["pin_norm", "first_token"]]})
    model = train_splink(df, scfg, with_ids=False, seed=0)
    pairs = np.array([[0, 1], [0, 2], [2, 3], [4, 9]], dtype=np.int64)
    s = score_pairs(model, df, pairs)
    assert s.shape == (4,)
    assert s[0] > s[1] and s[2] > s[3]  # true duplicates score above different companies
