import random

import polars as pl
import pytest

from ekam.synth.generator import Level, SourceSpec, build_sources
from ekam.utils.config import load_config

SAMPLE_ENTITIES = [
    {"entity_id": "E0000001", "name": "SHARMA TRADERS PRIVATE LIMITED",
     "street": "12 MG ROAD NEAR CITY MALL", "city": "Jaipur", "state": "Rajasthan",
     "pin": "302001", "cin": "U51909RJ2005PTC020001"},
    {"entity_id": "E0000002", "name": "SHREE LAXMI ENTERPRISES LIMITED",
     "street": "PLOT 4 GIDC ESTATE PHASE 2", "city": "Ahmedabad", "state": "Gujarat",
     "pin": "380015", "cin": "U24100GJ1999PLC035555"},
    {"entity_id": "E0000003", "name": "BANGALORE SOFT SOLUTIONS PVT LTD",
     "street": "NO 75 20TH MAIN ROAD RAJAJINAGAR", "city": "Bengaluru", "state": "Karnataka",
     "pin": "560010", "cin": "U72200KA2010PTC052345"},
    {"entity_id": "E0000004", "name": "GANESH AGRO INDUSTRIES (INDIA) LIMITED",
     "street": None, "city": None, "state": "Uttar Pradesh", "pin": None,
     "cin": "L01100UP1987PLC008888"},
    {"entity_id": "E0000005", "name": "KRISHNA & CO", "street": "SHOP 3 MAIN BAZAAR",
     "city": "Panaji", "state": "Goa", "pin": "403001", "cin": "U00000GA2001PTC000001"},
]


@pytest.fixture(scope="session")
def cfg():
    return load_config(["synth.n_workers=1"])


@pytest.fixture(scope="session")
def sources(cfg) -> list[SourceSpec]:
    return build_sources(cfg)


@pytest.fixture(scope="session")
def levels() -> dict[str, Level]:
    return {"easy": Level("easy", 0.4), "medium": Level("medium", 1.0),
            "hard": Level("hard", 1.8, max_typo_edits=2)}


@pytest.fixture
def entities() -> list[dict]:
    return [dict(e) for e in SAMPLE_ENTITIES]


@pytest.fixture
def many_entities() -> list[dict]:
    """~400 entities built by perturbing the samples (unique ids, varied names)."""
    rng = random.Random(7)
    out = []
    words = ["SHARMA", "GUPTA", "PATEL", "SINGH", "AGARWAL", "KRISHNA", "BALAJI", "OM",
             "SAI", "MAHAVIR", "NEW", "GLOBAL", "STAR", "UNIVERSAL"]
    kinds = ["TRADERS", "ENTERPRISES", "INDUSTRIES", "SERVICES", "TECHNOLOGIES", "EXPORTS"]
    sufs = ["PRIVATE LIMITED", "PVT LTD", "LIMITED", "LLP", ""]
    for i in range(400):
        base = SAMPLE_ENTITIES[i % len(SAMPLE_ENTITIES)]
        name = " ".join([rng.choice(words), rng.choice(words), rng.choice(kinds),
                         rng.choice(sufs)]).strip()
        out.append({**base, "entity_id": f"E{i + 10:07d}", "name": name})
    return out


def as_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={k: pl.Utf8 for k in SAMPLE_ENTITIES[0]})
