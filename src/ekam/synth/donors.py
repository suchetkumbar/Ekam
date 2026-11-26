"""Donor addresses for the ``addr_other`` noise type.

Real vendor masters often hold a *different* address for the same company: a works, branch,
warehouse or billing office rather than the registered office. ``addr_other`` simulates that
by replacing the record's (street, PIN) with the registered address of **another company**
in the same city (falling back to the same state, which also changes the city). Donor
addresses are real MCA street lines, so the replacement looks like any other address: a
model cannot tell a "replaced" street from an original one by its style.

Side effect, which is also realistic: the donor company and the noised record now share an
address (many Indian companies share a registered office), which creates hard negatives.

The pool is built once from *all* entities, in entity-id order, so a record's donor does
not depend on chunking or the number of workers.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field

_MIN_CITY_POOL = 5
_TRIES = 4


@dataclass
class DonorPool:
    # (state, city) -> list of (entity_id, street, city, pin); state -> same
    by_city: dict[tuple[str, str], list[tuple]] = field(default_factory=dict)
    by_state: dict[str, list[tuple]] = field(default_factory=dict)

    @classmethod
    def from_entities(cls, entities: Iterable[dict]) -> DonorPool:
        pool = cls()
        for e in sorted(entities, key=lambda e: e["entity_id"]):
            street, state = e.get("street"), e.get("state")
            if not street or not state:
                continue
            row = (e["entity_id"], street, e.get("city"), e.get("pin"))
            pool.by_state.setdefault(state, []).append(row)
            if e.get("city"):
                pool.by_city.setdefault((state, e["city"]), []).append(row)
        return pool

    def pick(self, entity: dict, rng: random.Random) -> tuple[str, str | None, str | None] | None:
        """(street, city, pin) of another company near ``entity``, or None if there is none."""
        state, city = entity.get("state"), entity.get("city")
        cands = self.by_city.get((state, city), []) if city else []
        if len(cands) < _MIN_CITY_POOL:
            cands = self.by_state.get(state, [])
        if not cands:
            return None
        for _ in range(_TRIES):
            eid, street, dcity, pin = cands[rng.randrange(len(cands))]
            if eid != entity["entity_id"] and street != entity.get("street"):
                return street, dcity, pin
        return None
