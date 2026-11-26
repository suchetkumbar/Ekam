"""String-similarity features for candidate record pairs (vectorised with RapidFuzz).

Feature groups (named so they can be switched off, e.g. ``ids`` for the shortcut control):

* ``name``    Jaro-Winkler, token-set / token-sort / partial ratio, normalised Levenshtein,
              char-n-gram TF-IDF cosine, phonetic / first-token / legal-form agreement.
* ``street``  street-line token-set ratio and Jaro-Winkler, missingness.
* ``geo``     city / state / PIN agreement, PIN-prefix agreement, missingness indicators.
* ``ids``     CIN equal, GSTIN equal, same PAN inside GSTIN (+ "both present" flags).

Similarities involving a missing value are NaN (LightGBM handles NaN natively; the LR
pipeline imputes 0 and the ``*_missing`` indicators carry the information).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

GROUPS = {
    "name": ["name_jw", "name_token_set", "name_token_sort", "name_partial", "name_lev",
             "name_tfidf", "phonetic_eq", "first_token_eq", "legal_both", "legal_eq",
             "any_indic"],
    "street": ["street_token_set", "street_jw", "street_missing"],
    "geo": ["city_eq", "city_missing", "state_eq", "state_missing", "pin_eq", "pin3_eq",
            "pin_missing"],
    "ids": ["cin_both", "cin_eq", "gstin_both", "gstin_eq", "pan_eq"],
}


def feature_names(groups: list[str]) -> list[str]:
    return [f for g in groups for f in GROUPS[g]]


class TfidfName:
    """Character n-gram TF-IDF fitted on training-record name keys only."""

    def __init__(self, analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=200_000):
        self.vec = TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(ngram_range),
                                   min_df=min_df, max_features=max_features,
                                   sublinear_tf=True, dtype=np.float32)

    def fit(self, texts: list[str]) -> TfidfName:
        self.vec.fit([t or "" for t in texts])
        return self

    def cosine(self, texts: list[str], pairs: np.ndarray) -> np.ndarray:
        x = self.vec.transform([t or "" for t in texts])  # rows are L2-normalised
        out = np.empty(len(pairs), np.float32)
        step = 500_000
        for s in range(0, len(pairs), step):
            p = pairs[s : s + step]
            out[s : s + step] = np.asarray(x[p[:, 0]].multiply(x[p[:, 1]]).sum(axis=1)).ravel()
        return out


def _col(df: pl.DataFrame, name: str, idx: np.ndarray) -> list:
    return df[name].gather(idx).to_list()


def _sim(a: list, b: list, scorer, scale: float = 1.0) -> np.ndarray:
    aa = [x or "" for x in a]
    bb = [x or "" for x in b]
    s = process.cpdist(aa, bb, scorer=scorer, workers=-1, dtype=np.float32) / scale
    miss = np.array([not x or not y for x, y in zip(a, b, strict=True)])
    s[miss] = np.nan
    return s


def _eq(a: list, b: list) -> tuple[np.ndarray, np.ndarray]:
    """(equal as 1/0 with NaN when either missing, either-missing indicator)."""
    a_arr = np.array([x if x is not None else "" for x in a], dtype=object)
    b_arr = np.array([x if x is not None else "" for x in b], dtype=object)
    miss = (a_arr == "") | (b_arr == "")
    eq = (a_arr == b_arr).astype(np.float32)
    eq[miss] = np.nan
    return eq, miss.astype(np.float32)


def pair_features(df: pl.DataFrame, pairs: np.ndarray, tfidf: TfidfName,
                  groups: list[str]) -> pl.DataFrame:
    """Features for ``pairs`` (rows of ``df`` by dense ``idx`` position)."""
    li, ri = pairs[:, 0], pairs[:, 1]
    f: dict[str, np.ndarray] = {}
    if "name" in groups:
        a, b = _col(df, "name_key", li), _col(df, "name_key", ri)
        f["name_jw"] = _sim(a, b, JaroWinkler.normalized_similarity)
        f["name_token_set"] = _sim(a, b, fuzz.token_set_ratio, 100)
        f["name_token_sort"] = _sim(a, b, fuzz.token_sort_ratio, 100)
        f["name_partial"] = _sim(a, b, fuzz.partial_ratio, 100)
        ab, bb = _col(df, "name_base", li), _col(df, "name_base", ri)
        f["name_lev"] = _sim(ab, bb, Levenshtein.normalized_similarity)
        f["name_tfidf"] = tfidf.cosine(df["name_key"].to_list(), pairs)
        f["phonetic_eq"], _ = _eq(_col(df, "name_phonetic", li), _col(df, "name_phonetic", ri))
        f["first_token_eq"], _ = _eq(_col(df, "first_token", li), _col(df, "first_token", ri))
        eq, miss = _eq(_col(df, "legal_form", li), _col(df, "legal_form", ri))
        f["legal_both"], f["legal_eq"] = 1 - miss, np.nan_to_num(eq, nan=0.0)
        ind = df["name_indic"].to_numpy()
        f["any_indic"] = (ind[li] | ind[ri]).astype(np.float32)
    if "street" in groups:
        a, b = _col(df, "street", li), _col(df, "street", ri)
        f["street_token_set"] = _sim(a, b, fuzz.token_set_ratio, 100)
        f["street_jw"] = _sim(a, b, JaroWinkler.normalized_similarity)
        f["street_missing"] = np.isnan(f["street_jw"]).astype(np.float32)
    if "geo" in groups:
        f["city_eq"], f["city_missing"] = _eq(_col(df, "city_norm", li), _col(df, "city_norm", ri))
        f["state_eq"], f["state_missing"] = _eq(_col(df, "state_norm", li),
                                                _col(df, "state_norm", ri))
        pa, pb = _col(df, "pin_norm", li), _col(df, "pin_norm", ri)
        f["pin_eq"], f["pin_missing"] = _eq(pa, pb)
        p3a = [p[:3] if p else None for p in pa]
        p3b = [p[:3] if p else None for p in pb]
        f["pin3_eq"], _ = _eq(p3a, p3b)
    if "ids" in groups:
        eq, miss = _eq(_col(df, "cin_norm", li), _col(df, "cin_norm", ri))
        f["cin_both"], f["cin_eq"] = 1 - miss, np.nan_to_num(eq, nan=0.0)
        ga, gb = _col(df, "gstin_norm", li), _col(df, "gstin_norm", ri)
        eq, miss = _eq(ga, gb)
        f["gstin_both"], f["gstin_eq"] = 1 - miss, np.nan_to_num(eq, nan=0.0)
        pan_eq, _ = _eq([g[2:12] if g else None for g in ga], [g[2:12] if g else None for g in gb])
        f["pan_eq"] = np.nan_to_num(pan_eq, nan=0.0)
    return pl.DataFrame({k: f[k] for k in feature_names(groups)})
