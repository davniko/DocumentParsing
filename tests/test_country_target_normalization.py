from copy import deepcopy
from pathlib import Path

import pytest

from document_ocr.labeling_agents.target_normalization import normalize_target_countries
from document_ocr.synthesis.country_presentations import Registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def countries():
    return Registry.load(ROOT)


def test_country_normalization_preserves_owned_text_and_all_other_values(countries):
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "negotiability": None,
            "parties": {
                "shipper": {
                    "name": "EGITTO EXPORT",
                    "addressLine": "12 VIA ROMA, GIZA, EGITTO",
                    "country": "EGITTO",
                    "contactDetails": {"emailAddresses": ["Test@Example.it"]},
                },
                "notifyParties": [{"sameAs": None, "name": "ALPHA", "country": "U.A.E."}],
            },
            "route": {"portOfLoading": {"name": "EL ISKANDARIYA", "country": "EG"}},
            "placeOfIssue": {"name": "MOMBASA", "country": "KE"},
            "freight": {"paymentPlace": {"name": "BUSAN", "country": "KOREA"}},
            "goodsItemDetails": [{"description": "MADE IN USA", "origin": {"name": "CN"}}],
        },
    }
    before = deepcopy(target)
    actual, changes = normalize_target_countries(target, countries)
    assert target == before
    assert len(changes) == 5
    assert actual["documentPatch"]["parties"]["shipper"]["country"] == "EGYPT"
    assert (
        actual["documentPatch"]["parties"]["notifyParties"][0]["country"] == "UNITED ARAB EMIRATES"
    )
    assert actual["documentPatch"]["route"]["portOfLoading"] == {
        "name": "EL ISKANDARIYA",
        "country": "EGYPT",
    }
    assert actual["documentPatch"]["placeOfIssue"]["country"] == "KENYA"
    assert actual["documentPatch"]["freight"]["paymentPlace"]["country"] == "SOUTH KOREA"
    assert normalize_target_countries(actual, countries) == (actual, [])
    for change in changes:
        assert change["path"].endswith(".country")
    assert (
        actual["documentPatch"]["goodsItemDetails"] == before["documentPatch"]["goodsItemDetails"]
    )
    actual["documentPatch"]["parties"]["shipper"]["country"] = "EGITTO"
    actual["documentPatch"]["parties"]["notifyParties"][0]["country"] = "U.A.E."
    actual["documentPatch"]["route"]["portOfLoading"]["country"] = "EG"
    actual["documentPatch"]["placeOfIssue"]["country"] = "KE"
    actual["documentPatch"]["freight"]["paymentPlace"]["country"] = "KOREA"
    assert actual == before


def test_country_initialism_punctuation_changes_identity_deliberately(countries):
    assert countries.normalized("A.R.E") == "EGYPT"
    assert countries.normalized("ARE") == "UNITED ARAB EMIRATES"


@pytest.mark.parametrize("value", ["UNKNOWN COUNTRY", "EGYPT(US)", None])
def test_unknown_country_is_not_silently_removed_or_inferred(countries, value):
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {"parties": {"shipper": {"country": value}}},
    }
    before = deepcopy(target)
    with pytest.raises(ValueError, match=r"documentPatch\.parties\.shipper\.country"):
        normalize_target_countries(target, countries)
    assert target == before


def test_country_omission_and_symbolic_party_nulls_remain_absent(countries):
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "negotiability": None,
            "parties": {"notifyParties": [{"sameAs": None}, {"sameAs": "consignee"}]},
            "route": {"placeOfDelivery": {"name": "EL ISKANDARIYA"}},
        },
    }
    assert normalize_target_countries(target, countries) == (target, [])


def test_country_normalizer_does_not_accept_an_unversioned_target(countries):
    with pytest.raises(ValueError, match="V7 target envelope"):
        normalize_target_countries({"parties": {}}, countries)
