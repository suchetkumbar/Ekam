import polars as pl
import pytest

from ekam.data.privacy import apply_allowlist
from ekam.utils.config import load_config


def test_unknown_and_personal_columns_dropped():
    df = pl.DataFrame({"COMPANY_NAME": ["A"], "DIRECTOR_NAME": ["Some Person"],
                       "EMAIL_ADDR": ["x@example.com"], "PHONE": ["99999"]})
    out = apply_allowlist(df, {"COMPANY_NAME": "name"})
    assert out.columns == ["name"]


def test_configured_allowlist_has_no_personal_fields():
    cfg = load_config()
    allow = dict(cfg.data.allow_columns)
    out = apply_allowlist(pl.DataFrame({k: ["x"] for k in [*allow, "EMAIL_ADDR"]}), allow)
    assert "EMAIL_ADDR" not in out.columns
    assert set(out.columns) == set(allow.values())


def test_personal_column_in_allowlist_is_refused():
    with pytest.raises(ValueError):
        apply_allowlist(pl.DataFrame({"X": [1]}), {"X": "director_name"})
