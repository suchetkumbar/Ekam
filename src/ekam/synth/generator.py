"""Clean MCA entities -> several noisy "source systems" with exact ground truth.

Design rules (each guarded by a test in tests/test_synth_*.py):

1. **Ground truth is never in the records.** Output records carry a random ``record_id``
   (UUID, derived from a seeded RNG, encoding nothing); the ``record_id -> entity_id``
   mapping is written to a separate ground-truth file.
2. **Structure is level-independent.** Whether an entity appears in a source, and how many
   in-source duplicates it has, depend only on (seed, source, entity) - so easy/medium/hard
   share the same clusters and the same entity split; only the noise differs.
3. **Every noise decision is logged** per record (applied / rolled-but-not-applicable), so
   the audit can compare measured against configured rates.
4. **Records are shuffled** (sorted by random record_id) so file order leaks nothing.
5. **Deterministic** for a fixed seed regardless of chunking or worker count: every RNG is
   seeded from a string key (Python seeds str via SHA-512, independent of PYTHONHASHSEED).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from omegaconf import DictConfig, OmegaConf

from ekam.synth import gstin as gst
from ekam.synth.donors import DonorPool
from ekam.synth.noise import address as an
from ekam.synth.noise import names as nn

log = logging.getLogger(__name__)

CANONICAL = ["record_id", "source", "name", "address", "city", "state", "pin", "cin", "gstin"]
_MAX_DUPS = 6

# Noise types that are only drawn if another one did not fire (mutually exclusive pairs).
CONDITIONAL_ON = {"name_suffix_variant": "name_suffix_drop", "city_alias": "city_alias_heldout"}


def rate_is_unconditional(src: SourceSpec, ntype: str) -> bool:
    """True if the logged draw rate of ``ntype`` should equal its configured rate."""
    if ntype in CONDITIONAL_ON:
        return False
    field = ntype.split("_", 1)[0]
    return not (field in ("cin", "gstin") and not src.columns.get(field))


@dataclass(frozen=True)
class SourceSpec:
    name: str
    coverage: float
    dup_rate: float
    address_format: str
    columns: dict[str, str | None]
    noise: dict[str, float]
    force_case: str | None = None

    @classmethod
    def from_cfg(cls, name: str, cfg) -> SourceSpec:
        d = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
        return cls(
            name=name,
            coverage=float(d["coverage"]),
            dup_rate=float(d["dup_rate"]),
            address_format=d.get("address_format", "split"),
            columns=dict(d["columns"]),
            noise={k: float(v) for k, v in (d.get("noise") or {}).items()},
            force_case=d.get("force_case"),
        )


@dataclass(frozen=True)
class Level:
    name: str
    multiplier: float
    max_typo_edits: int = 1
    rate_cap: float = 0.95


def n_records_in_source(seed: int, src: SourceSpec, entity_id: str) -> int:
    """How many records entity has in source (0 = absent). Level-independent."""
    r = random.Random(f"{seed}|struct|{src.name}|{entity_id}")
    if r.random() >= src.coverage:
        return 0
    n = 1
    while n < _MAX_DUPS and r.random() < src.dup_rate:
        n += 1
    return n


class _Noiser:
    """Applies noise types to one record and keeps the decision log."""

    def __init__(self, rng: random.Random, rates: dict[str, float]):
        self.rng, self.rates = rng, rates
        self.applied: list[str] = []
        self.ineligible: list[str] = []

    def __call__(self, ntype: str, value, fn: Callable, *args, **kw):
        p = self.rates.get(ntype, 0.0)
        if p <= 0 or self.rng.random() >= p:
            return value
        if value is None or value == "":
            self.ineligible.append(ntype)
            return value
        new = fn(value, self.rng, *args, **kw)
        (self.applied if new != value else self.ineligible).append(ntype)
        return new


def _entity_index(entity_id: str) -> int:
    digits = "".join(ch for ch in entity_id if ch.isdigit())
    if digits:
        return int(digits)
    return int(hashlib.sha256(entity_id.encode()).hexdigest(), 16) % 1_000_000


def _missing(_value, _rng):
    return None


def render_record(entity: dict, src: SourceSpec, level: Level, rng: random.Random,
                  gstin: str | None, donors: DonorPool | None = None
                  ) -> tuple[dict, list[str], list[str]]:
    rates = {k: min(v * level.multiplier, level.rate_cap) for k, v in src.noise.items()}
    nz = _Noiser(rng, rates)

    # ---- name -------------------------------------------------------------------------
    name = entity["name"]
    before = len(nz.applied)
    name = nz("name_suffix_drop", name, nn.suffix_drop)
    if len(nz.applied) == before:  # variant only if not dropped
        name = nz("name_suffix_variant", name, nn.suffix_variant)
    name = nz("name_abbrev", name, nn.abbrev)
    name = nz("name_abbrev_heldout", name, nn.abbrev_heldout)
    name = nz("name_translit_variant", name, nn.translit_variant)
    name = nz("name_token_drop", name, nn.token_drop)
    name = nz("name_token_swap", name, nn.token_swap)
    name = nz("name_spacing", name, nn.spacing)
    name = nz("name_typo", name, nn.name_typo, max_edits=level.max_typo_edits)
    name = nz("name_ocr", name, nn.name_ocr)
    name = name.upper() if src.force_case == "upper" else nz("name_case", name, nn.case)
    name = nz("name_devanagari", name, nn.devanagari)
    name = nz("name_prefix_ms", name, nn.prefix_ms)

    # ---- address ----------------------------------------------------------------------
    street = entity.get("street") or None
    moved = None  # (street, city, pin) of a different real address (branch / works / billing)

    def _other(value, r):
        nonlocal moved
        moved = donors.pick(entity, r) if donors is not None else None
        return moved[0] if moved else value

    street = nz("addr_other", street, _other)
    street = nz("addr_reorder", street, an.street_reorder)
    street = nz("addr_abbrev", street, an.street_abbrev)
    street = nz("addr_truncate", street, an.street_truncate)
    street = nz("addr_typo", street, an.street_typo, max_edits=level.max_typo_edits)
    street = nz("addr_ocr", street, an.street_ocr)

    city = (moved[1] if moved else entity.get("city")) or None
    city = nz("city_missing", city, _missing)
    before = len(nz.applied)
    city = nz("city_alias_heldout", city, an.city_alias_heldout)
    if len(nz.applied) == before:
        city = nz("city_alias", city, an.city_alias)

    state = entity.get("state") or None
    state = nz("state_missing", state, _missing)
    state = nz("state_variant", state, an.state_variant)

    pin = (moved[2] if moved else entity.get("pin")) or None
    pin = nz("pin_missing", pin, _missing)
    pin = nz("pin_wrong", pin, an.pin_wrong, state=entity.get("state"))

    # ---- identifiers --------------------------------------------------------------------
    cin = entity.get("cin") if src.columns.get("cin") else None
    if src.columns.get("cin"):
        cin = nz("cin_missing", cin, _missing)
        cin = nz("cin_typo", cin, gst.corrupt_identifier)
    g = gstin if src.columns.get("gstin") else None
    if src.columns.get("gstin"):
        g = nz("gstin_missing", g, _missing)
        g = nz("gstin_typo", g, gst.corrupt_identifier)

    if src.force_case == "upper" and street:
        street = street.upper()

    if src.address_format == "single_line":
        tail = " ".join(x for x in [state, pin] if x)
        address = ", ".join(x for x in [street, city, tail] if x) or None
        city = state = pin = None
    else:
        address = street

    rec = {"source": src.name, "name": name, "address": address, "city": city,
           "state": state, "pin": pin, "cin": cin, "gstin": g}
    return rec, nz.applied, nz.ineligible


def generate_for_entities(entities: Iterable[dict], sources: list[SourceSpec], level: Level,
                          seed: int, gstin_prefix: str = "ZZZ", donors: DonorPool | None = None
                          ) -> tuple[list[dict], list[dict]]:
    """Pure function: entities -> (records, ground_truth rows). No I/O.

    ``donors`` supplies replacement addresses for ``addr_other``; by default it is built
    from ``entities`` themselves (pass a pool built from *all* entities when chunking).
    """
    entities = list(entities)
    if donors is None:
        donors = DonorPool.from_entities(entities)
    records, truth = [], []
    for e in entities:
        eid = e["entity_id"]
        g = gst.make_synthetic_gstin(e.get("state"), e.get("name"),
                                     gst.entity_serial(_entity_index(eid), seed), gstin_prefix)
        for src in sources:
            for k in range(n_records_in_source(seed, src, eid)):
                rng = random.Random(f"{seed}|{level.name}|{src.name}|{eid}|{k}")
                rid = str(uuid.UUID(int=rng.getrandbits(128), version=4))
                rec, applied, inel = render_record(e, src, level, rng, g, donors)
                rec["record_id"] = rid
                records.append(rec)
                truth.append({"record_id": rid, "entity_id": eid, "source": src.name,
                              "dup_index": k, "noise_applied": applied,
                              "noise_ineligible": inel})
    return records, truth


_DONORS: DonorPool | None = None


def _init_worker(donors: DonorPool) -> None:
    global _DONORS
    _DONORS = donors


def _worker(args):
    chunk, sources, level, seed, prefix = args
    return generate_for_entities(chunk, sources, level, seed, prefix, donors=_DONORS)


_REC_SCHEMA = {c: pl.Utf8 for c in CANONICAL}
_GT_SCHEMA = {"record_id": pl.Utf8, "entity_id": pl.Utf8, "source": pl.Utf8,
              "dup_index": pl.Int32, "noise_applied": pl.List(pl.Utf8),
              "noise_ineligible": pl.List(pl.Utf8)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_entities(cfg: DictConfig) -> pl.DataFrame:
    df = pl.read_parquet(cfg.synth.entities_path)
    if cfg.synth.n_entities:
        df = df.sample(n=min(int(cfg.synth.n_entities), df.height), seed=cfg.seed)
        df = df.sort("entity_id")
    return df


def build_sources(cfg: DictConfig) -> list[SourceSpec]:
    return [SourceSpec.from_cfg(n, c) for n, c in cfg.synth.sources.items()]


def build_level(cfg: DictConfig, name: str) -> Level:
    from ekam.utils.config import CONF_DIR

    lv = OmegaConf.load(CONF_DIR / "noise" / f"{name}.yaml")
    return Level(name=name, multiplier=float(lv.multiplier),
                 max_typo_edits=int(lv.max_typo_edits), rate_cap=float(cfg.synth.rate_cap))


def generate_level(cfg: DictConfig, entities: pl.DataFrame, level: Level) -> Path:
    sources = build_sources(cfg)
    rows = entities.select("entity_id", "name", "street", "city", "state", "pin", "cin").to_dicts()
    cs = int(cfg.synth.chunk_size)
    tasks = [(rows[i : i + cs], sources, level, int(cfg.seed), cfg.synth.gstin_pan_prefix)
             for i in range(0, len(rows), cs)]
    recs, gts = [], []
    n_workers = int(cfg.synth.n_workers)
    donors = DonorPool.from_entities(rows)
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(donors,)) as ex:
            for r, t in ex.map(_worker, tasks):
                recs.extend(r)
                gts.extend(t)
    else:
        _init_worker(donors)
        for task in tasks:
            r, t = _worker(task)
            recs.extend(r)
            gts.extend(t)

    out = Path(cfg.synth.out_dir) / level.name
    (out / "sources").mkdir(parents=True, exist_ok=True)
    rec_df = pl.DataFrame(recs, schema=_REC_SCHEMA).select(CANONICAL).sort("record_id")
    gt_df = pl.DataFrame(gts, schema=_GT_SCHEMA).sort("record_id")
    rec_df.write_parquet(out / "records.parquet")
    gt_df.write_parquet(out / "ground_truth.parquet")

    # Per-source exports in each system's own column names (what a user would really get).
    for src in sources:
        cols = {c: n for c, n in src.columns.items() if n}
        native = (rec_df.filter(pl.col("source") == src.name)
                  .select(["record_id", *cols.keys()]).rename(cols))
        native.write_csv(out / "sources" / f"{src.name}.csv")

    manifest = {
        "level": level.name, "multiplier": level.multiplier, "seed": int(cfg.seed),
        "n_entities": entities.height, "n_records": rec_df.height,
        "records_per_source": dict(rec_df.group_by("source").len().iter_rows()),
        "sha256": {p.name: _sha256(p) for p in [out / "records.parquet",
                                                 out / "ground_truth.parquet"]},
        "sources": {s.name: OmegaConf.to_container(cfg.synth.sources[s.name], resolve=True)
                    for s in sources},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("level %s: %d records from %d entities -> %s", level.name, rec_df.height,
             entities.height, out)
    return out


def run(cfg: DictConfig) -> list[Path]:
    entities = load_entities(cfg)
    return [generate_level(cfg, entities, build_level(cfg, lv)) for lv in cfg.synth.levels]


if __name__ == "__main__":
    import sys

    from ekam.utils.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(sys.argv[1:]))
