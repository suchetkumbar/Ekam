import pytest

from ekam.normalize import (
    canonical_city,
    canonical_state,
    normalize_pin,
    parse_address,
    pin_state_consistent,
)


@pytest.mark.parametrize(
    "raw, city, state, pin",
    [
        ("SESA GHOR 20 EDC COMPLEX PATTO PANJIM GOA Goa INDIA 403001", "Panaji", "Goa", "403001"),
        ("No. 75/7, 20th A Main Road, Rajajinagar Bangalore Karnataka INDIA 560010",
         "Bengaluru", "Karnataka", "560010"),
        ("12/13 NORTH LOWTHER ROAD ALLAHABAD UTTAR PRADESH Uttar Pradesh INDIA 211002",
         "Prayagraj", "Uttar Pradesh", "211002"),
        ("Plot 4, MG Rd, Gurgaon, HR 122001", "Gurugram", "Haryana", "122001"),
        ("DHRUVA 4, NUTAN NAGAR, KALAWAD ROAD, RAJKOT Gujarat INDIA 360 001", "Rajkot",
         "Gujarat", "360001"),
        ("Shop 3, Main Bazaar, Bombay - 400001", "Mumbai", None, "400001"),
        ("12 Station Road, Udaipur, Rajasthan", "Udaipur", "Rajasthan", None),
        ("45 Ring Road, Lucknow U.P. 226001", "Lucknow", "Uttar Pradesh", "226001"),
    ],
)
def test_parse_address(raw, city, state, pin):
    p = parse_address(raw)
    assert (p.city, p.state, p.pin) == (city, state, pin)


def test_state_and_city_removed_from_street():
    p = parse_address("12/13 NORTH LOWTHER ROAD ALLAHABAD UTTAR PRADESH Uttar Pradesh INDIA 211002")
    assert "UTTAR" not in p.street and "ALLAHABAD" not in p.street and "INDIA" not in p.street


def test_street_abbreviations_expanded():
    assert parse_address("12 MG Rd Nr Bus Stand Opp Temple").street == \
        "12 MG ROAD NEAR BUS STAND OPPOSITE TEMPLE"


def test_state_hint_used_when_text_has_none():
    assert parse_address("12 Station Road", state_hint="Goa").state == "Goa"


@pytest.mark.parametrize("value, expected", [("560 001", "560001"), (560001, "560001"),
                                             ("560001.0", "560001"), ("056000", None),
                                             ("5600", None), (None, None), ("", None)])
def test_normalize_pin(value, expected):
    assert normalize_pin(value) == expected


def test_pin_state_consistency():
    assert pin_state_consistent("560001", "Karnataka") is True
    assert pin_state_consistent("560001", "Gujarat") is False
    assert pin_state_consistent("403001", "Goa") is True
    assert pin_state_consistent(None, "Goa") is None


@pytest.mark.parametrize("text, state", [("UP", "Uttar Pradesh"), ("U.P.", "Uttar Pradesh"),
                                         ("Orissa", "Odisha"), ("KA", "Karnataka"),
                                         ("Gujrat", "Gujarat"), ("nonsense", None)])
def test_canonical_state(text, state):
    assert canonical_state(text) == state


@pytest.mark.parametrize("text, city", [("Bombay", "Mumbai"), ("BANGALORE", "Bengaluru"),
                                        ("Baroda", "Vadodara"), ("Calcutta", "Kolkata"),
                                        ("Belgaum", "Belagavi"), ("Somewhere", "Somewhere")])
def test_canonical_city(text, city):
    assert canonical_city(text) == city
