"""Normalisation for Indian business records.

Public API::

    normalize_name("M/s Sharma Trdrs Pvt. Ltd.")  -> base "SHARMA TRADERS", suffix "PVT_LTD"
    parse_address("12 MG Rd, Bangalore KA 560001") -> ParsedAddress(city='Bengaluru', ...)
    normalize_record({...})                         -> flat dict of normalised fields
"""

from __future__ import annotations

from ekam.normalize.address import (
    ParsedAddress,
    canonical_city,
    canonical_state,
    normalize_pin,
    parse_address,
    pin_state_consistent,
)
from ekam.normalize.identifiers import normalize_cin, normalize_gstin
from ekam.normalize.names import NormName, normalize_name
from ekam.normalize.phonetic import phonetic_key
from ekam.normalize.text import basic_clean
from ekam.normalize.translit import fold, fold_token, to_latin


def normalize_record(rec: dict) -> dict:
    """Normalise one canonical-schema record (name, address, city, state, pin, cin, gstin)."""
    nm = normalize_name(rec.get("name"))
    addr = parse_address(rec.get("address"), state_hint=rec.get("state"))
    pin = normalize_pin(rec.get("pin")) or addr.pin
    state = canonical_state(rec.get("state")) or addr.state
    city = canonical_city(rec.get("city")) if rec.get("city") else addr.city
    return {
        "name_clean": nm.clean,
        "name_base": nm.base,
        "name_key": nm.key,
        "legal_form": nm.suffix,
        "name_phonetic": phonetic_key(nm.folded),
        "first_token": nm.folded[0] if nm.folded else "",
        "street": addr.street,
        "city_norm": city,
        "state_norm": state,
        "pin_norm": pin,
        "pin_state_ok": pin_state_consistent(pin, state),
        "cin_norm": normalize_cin(rec.get("cin")),
        "gstin_norm": normalize_gstin(rec.get("gstin")),
    }


__all__ = [
    "NormName",
    "ParsedAddress",
    "basic_clean",
    "canonical_city",
    "canonical_state",
    "fold",
    "fold_token",
    "normalize_cin",
    "normalize_gstin",
    "normalize_name",
    "normalize_pin",
    "normalize_record",
    "parse_address",
    "phonetic_key",
    "pin_state_consistent",
    "to_latin",
]
