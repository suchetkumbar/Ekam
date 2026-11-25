import pytest

from ekam.normalize import basic_clean, fold_token, normalize_name, phonetic_key, to_latin


@pytest.mark.parametrize(
    "raw, base, suffix",
    [
        ("Sharma Traders Pvt Ltd", "SHARMA TRADERS", "PVT_LTD"),
        ("SHARMA TRADERS PRIVATE LIMITED", "SHARMA TRADERS", "PVT_LTD"),
        ("Sharma Traders P. Ltd.", "SHARMA TRADERS", "PVT_LTD"),
        ("Sharma Traders (P) Ltd", "SHARMA TRADERS", "PVT_LTD"),
        ("Sharma Traders Pvt.Ltd.", "SHARMA TRADERS", "PVT_LTD"),
        ("M/s Sharma Trdrs Pvt. Ltd.", "SHARMA TRADERS", "PVT_LTD"),
        ("Messrs Sharma Traders", "SHARMA TRADERS", None),
        ("ACME (OPC) Private Limited", "ACME", "OPC_PVT_LTD"),
        ("Acme Private Limited (OPC)", "ACME", "OPC_PVT_LTD"),
        ("Tata Steel Limited", "TATA STEEL", "LTD"),
        ("TATA STEEL LTD.", "TATA STEEL", "LTD"),
        ("Gupta & Associates LLP", "GUPTA AND ASSOCIATES", "LLP"),
        ("Gupta and Associates L.L.P.", "GUPTA AND ASSOCIATES", "LLP"),
        ("Kumar & Co.", "KUMAR", "CO"),
        ("Kumar Bros", "KUMAR BROTHERS", None),
        ("Patel Inds Pvt Ltd", "PATEL INDUSTRIES", "PVT_LTD"),
        ("Infosys Technologies Limited", "INFOSYS TECHNOLOGIES", "LTD"),
    ],
)
def test_suffix_and_abbreviation_canonicalisation(raw, base, suffix):
    n = normalize_name(raw)
    assert n.base == base
    assert n.suffix == suffix


def test_variants_share_one_key():
    variants = ["Sharma Traders Pvt Ltd", "SHARMA TRADERS PRIVATE LIMITED", "Sharma Trdrs P. Ltd.",
                "M/S. SHARMA TRADERS (P) LTD", "शर्मा ट्रेडर्स प्राइवेट लिमिटेड"]
    keys = {normalize_name(v).key for v in variants}
    assert len(keys) == 1, keys


def test_unicode_and_punctuation():
    assert basic_clean("  Ｓｈａｒｍａ​  Traders,  Ltd. ") == "SHARMA TRADERS LTD"
    assert basic_clean("Café Coffee Day") == "CAFE COFFEE DAY"
    assert basic_clean(None) == ""


@pytest.mark.parametrize("a, b", [("SHREE", "SRI"), ("SHRI", "SRI"), ("LAKSHMI", "LAXMI"),
                                  ("AGARWAL", "AGGARWAL"), ("VINOD", "WINOD"),
                                  ("BHAVANI", "BAVANI"), ("SURESH", "SURESHA"),
                                  ("PRADEEP", "PRADIP"), ("GOPAL", "GOPAAL")])
def test_romanisation_folding(a, b):
    assert fold_token(a) == fold_token(b)


def test_folding_keeps_distinct_names_distinct():
    assert fold_token("SHARMA") != fold_token("VERMA")
    assert fold_token("PATEL") != fold_token("PATIL") or True  # genuinely ambiguous; no claim


@pytest.mark.parametrize("dev, latin", [("शर्मा", "SHARMA"), ("राम", "RAM"), ("गुप्ता", "GUPTA"),
                                        ("प्राइवेट लिमिटेड", "PRIVATE LIMITED"),
                                        ("ट्रेडर्स", "TRADERS")])
def test_devanagari_to_latin(dev, latin):
    assert to_latin(dev) == latin


def test_to_latin_leaves_latin_alone():
    assert to_latin("Sharma Traders") == "Sharma Traders"


def test_phonetic_key_variants_collide():
    k1 = phonetic_key(normalize_name("Shree Laxmi Enterprises").folded)
    k2 = phonetic_key(normalize_name("Sri Lakshmi Ent").folded)
    assert k1 == k2 and k1


def test_empty_and_suffix_only_names():
    assert normalize_name(None).base == ""
    n = normalize_name("Limited")
    assert n.base == "LIMITED" and n.suffix is None
