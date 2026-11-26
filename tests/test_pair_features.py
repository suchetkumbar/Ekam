import numpy as np
import polars as pl
import pytest

from ekam.features.pairs import GROUPS, TfidfName, feature_names, pair_features
from ekam.features.prepare import normalise_records
from ekam.match.benchmarks import LEAKY


@pytest.fixture
def df():
    raw = pl.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4"],
        "source": ["erp", "hindi_vendor", "crm_ocr", "erp"],
        "name": ["Sharma Traders Pvt Ltd", "शर्मा ट्रेडर्स प्राइवेट लिमिटेड", "SHARMA TRDRS",
                 "Gupta Steel Limited"],
        "address": ["12 MG Road", "12 MG Rd", "12 MG ROAD, Jaipur, Rajasthan 302001", None],
        "city": ["Jaipur", "Jaipur", None, "Kanpur"],
        "state": ["Rajasthan", "RJ", None, "UP"],
        "pin": ["302001", "302001", None, None],
        "cin": ["U51909RJ2005PTC020001", None, "U51909RJ2005PTC020001", None],
        "gstin": [None, None, None, None],
    })
    return normalise_records(raw).with_row_index("idx")


def test_normalised_records_have_no_truth(df):
    assert "entity_id" not in df.columns
    assert df["state_norm"].to_list() == ["Rajasthan", "Rajasthan", "Rajasthan", "Uttar Pradesh"]
    assert df["pin_norm"].to_list()[2] == "302001"  # parsed from the one-line address


def test_feature_shapes_and_signals(df):
    tf = TfidfName(min_df=1).fit(df["name_key"].to_list())
    pairs = np.array([[0, 1], [0, 2], [0, 3]])
    f = pair_features(df, pairs, tf, ["name", "street", "geo", "ids"])
    assert f.columns == feature_names(["name", "street", "geo", "ids"])
    assert f.height == 3
    jw = f["name_jw"].to_list()
    assert jw[0] == pytest.approx(1.0)        # Devanagari variant normalises to the same key
    assert jw[1] > jw[2]                      # abbreviation closer than a different company
    assert f["cin_eq"].to_list() == [0.0, 1.0, 0.0]
    assert np.isnan(f["pin_eq"].to_list()[2])  # missing PIN -> NaN, not "different"
    assert f["any_indic"].to_list() == [1.0, 0.0, 0.0]


def test_id_group_can_be_removed(df):
    tf = TfidfName(min_df=1).fit(df["name_key"].to_list())
    f = pair_features(df, np.array([[0, 2]]), tf, ["name", "street", "geo"])
    assert not set(GROUPS["ids"]) & set(f.columns)


@pytest.mark.parametrize("col", ["cluster_id_left", "cluster_id_right", "id_left", "label",
                                 "pair_id", "ltable_id"])
def test_benchmark_leaky_columns_detected(col):
    assert LEAKY.match(col)


@pytest.mark.parametrize("col", ["title_left", "brand_right", "price_left"])
def test_benchmark_attributes_not_flagged(col):
    assert not LEAKY.match(col)
