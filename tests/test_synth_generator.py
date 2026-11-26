"""Ground-truth integrity of the generator (property-based where it matters)."""

import random
import string

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ekam.data import leakage
from ekam.synth.donors import DonorPool
from ekam.synth.generator import (
    CANONICAL,
    generate_for_entities,
    n_records_in_source,
    rate_is_unconditional,
)
from ekam.synth.gstin import is_synthetic_gstin

_name = st.lists(st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=10),
                 min_size=1, max_size=5).map(" ".join)
_entity = st.fixed_dictionaries({
    "name": st.one_of(_name, _name.map(lambda n: n + " PRIVATE LIMITED")),
    "street": st.one_of(st.none(), _name),
    "city": st.sampled_from([None, "Jaipur", "Bengaluru", "Panaji", "Nowhere"]),
    "state": st.sampled_from([None, "Rajasthan", "Karnataka", "Goa"]),
    "pin": st.sampled_from([None, "302001", "560010", "403001"]),
    "cin": st.sampled_from([None, "U51909RJ2005PTC020001"]),
})


@given(ents=st.lists(_entity, min_size=1, max_size=25), seed=st.integers(0, 10**6),
       level=st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_every_record_maps_to_exactly_one_input_entity(ents, seed, level, sources, levels):
    ents = [{**e, "entity_id": f"E{i:07d}"} for i, e in enumerate(ents)]
    recs, truth = generate_for_entities(ents, sources, levels[level], seed)
    ids = {e["entity_id"] for e in ents}
    assert len(recs) == len(truth)
    assert len({r["record_id"] for r in recs}) == len(recs)          # unique record ids
    assert [r["record_id"] for r in recs] == [t["record_id"] for t in truth]
    assert all(t["entity_id"] in ids for t in truth)                 # never invents entities
    expected = sum(n_records_in_source(seed, s, e["entity_id"]) for e in ents for s in sources)
    assert len(recs) == expected                                     # structure as designed
    for r in recs:
        assert set(r) == set(CANONICAL)
        assert not any(isinstance(v, str) and v.startswith("E0") for k, v in r.items()
                       if k not in ("record_id",))
        if r["gstin"] is not None and is_synthetic_gstin(r["gstin"]):
            assert r["gstin"][2:5] == "ZZZ"


def test_structure_is_level_independent(many_entities, sources, levels):
    seed = 11
    per_level = {}
    for name, lv in levels.items():
        _, truth = generate_for_entities(many_entities, sources, lv, seed)
        per_level[name] = sorted((t["entity_id"], t["source"], t["dup_index"]) for t in truth)
    assert per_level["easy"] == per_level["medium"] == per_level["hard"]


def test_deterministic_and_chunk_invariant(many_entities, sources, levels):
    lv = levels["medium"]
    pool = DonorPool.from_entities(many_entities)
    whole = generate_for_entities(many_entities, sources, lv, 3)
    parts = [generate_for_entities(many_entities[i : i + 37], sources, lv, 3, donors=pool)
             for i in range(0, len(many_entities), 37)]
    recs = [r for p in parts for r in p[0]]
    def by_id(r):
        return r["record_id"]

    assert sorted(whole[0], key=by_id) == sorted(recs, key=by_id)
    assert generate_for_entities(many_entities, sources, lv, 3) == whole
    assert generate_for_entities(many_entities, sources, lv, 4)[0] != whole[0]


def test_harder_levels_change_more(many_entities, sources, levels):
    changed = {}
    for name, lv in levels.items():
        recs, truth = generate_for_entities(many_entities, sources, lv, 5)
        changed[name] = sum(len(t["noise_applied"]) for t in truth) / len(truth)
    assert changed["easy"] < changed["medium"] < changed["hard"]


def test_schema_absent_fields_stay_absent(many_entities, sources, levels):
    recs, _ = generate_for_entities(many_entities, sources, levels["hard"], 1)
    for r in recs:
        if r["source"] in ("gst_portal", "hindi_vendor"):
            assert r["cin"] is None
        if r["source"] == "crm_ocr":
            assert r["city"] is None and r["state"] is None and r["pin"] is None


def test_no_truth_in_records_and_no_order_leak(many_entities, sources, levels):
    recs, truth = generate_for_entities(many_entities, sources, levels["medium"], 2)
    rec = pl.DataFrame(recs).select(CANONICAL).sort("record_id")
    gt = pl.DataFrame(truth).drop("noise_applied", "noise_ineligible")
    leakage.check_no_truth_in_records(rec, gt)
    assert leakage.report_order_leak(rec, gt) < 0.05


def test_noise_log_only_names_configured_types(many_entities, sources, levels):
    allowed = {s.name: set(s.noise) for s in sources}
    _, truth = generate_for_entities(many_entities, sources, levels["hard"], 9)
    for t in truth:
        assert set(t["noise_applied"]) | set(t["noise_ineligible"]) <= allowed[t["source"]]


def test_measured_rates_track_configuration(many_entities, sources, levels):
    """Rolled rate per (source, type) within sampling error of the configured rate."""
    lv = levels["medium"]
    big = [{**e, "entity_id": f"X{i:07d}"} for i, e in enumerate(many_entities * 5)]
    _, truth = generate_for_entities(big, sources, lv, 21)
    rng = random.Random(0)
    for s in sources:
        sub = [t for t in truth if t["source"] == s.name]
        testable = sorted((k, v) for k, v in s.noise.items() if rate_is_unconditional(s, k))
        for ntype, base in rng.sample(testable, k=min(6, len(testable))):
            p = min(base * lv.multiplier, lv.rate_cap)
            rolled = sum(ntype in t["noise_applied"] or ntype in t["noise_ineligible"]
                         for t in sub) / len(sub)
            sd = (p * (1 - p) / len(sub)) ** 0.5
            assert abs(rolled - p) < 5 * sd + 0.01, (s.name, ntype, rolled, p)


def test_addr_other_uses_another_companys_real_address(many_entities, sources, levels):
    by_id = {e["entity_id"]: e for e in many_entities}
    streets = {e["street"]: e["entity_id"] for e in many_entities if e.get("street")}
    recs, truth = generate_for_entities(many_entities, sources, levels["hard"], 17)
    n = 0
    for r, t in zip(recs, truth, strict=True):
        if "addr_other" not in t["noise_applied"]:
            continue
        n += 1
        own = by_id[t["entity_id"]]
        street_noise = {"addr_reorder", "addr_abbrev", "addr_typo", "addr_truncate"}
        if r["source"] == "erp" and not street_noise & set(t["noise_applied"]):
            assert r["address"] != own["street"]
            assert streets.get(r["address"]) not in (None, t["entity_id"])
    assert n > 0


def test_donor_pool_never_picks_self():
    ents = [{"entity_id": f"E{i}", "street": f"{i} MG ROAD", "city": "Jaipur",
             "state": "Rajasthan", "pin": "302001"} for i in range(6)]
    pool = DonorPool.from_entities(ents)
    rng = random.Random(0)
    for _ in range(200):
        got = pool.pick(ents[0], rng)
        assert got is None or got[0] != ents[0]["street"]
    assert DonorPool.from_entities(ents[:1]).pick(ents[0], rng) is None
