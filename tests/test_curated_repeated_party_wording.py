from copy import deepcopy

import pytest

from document_ocr.synthesis.curated import flat
from document_ocr.synthesis.curated_campaign import _validate_repeated_party_wording

ROLES = ("documentPatch.parties.consignee", "documentPatch.parties.notifyParties[0]")


def parties(name="SOURCE COMPANY", address="14 OLD ROAD, OLD CITY, EGYPT"):
    return {
        "documentPatch": {
            "parties": {
                "consignee": {"name": name, "addressLine": address},
                "notifyParties": [{"name": name, "addressLine": address}],
            }
        }
    }


def test_pre_presentation_identity_is_strict_even_when_printed_country_aliases_differ():
    source = flat(parties())
    generated = flat(parties("NEW COMPANY", "PLOT 14, AR RUMAYLAH, UNITED ARAB EMIRATES"))
    printed = dict(generated)
    printed[ROLES[0] + ".addressLine"] = "PLOT 14, AR RUMAYLAH, UAE"
    printed[ROLES[1] + ".addressLine"] = "PLOT 14, AR RUMAYLAH, U.A.E."
    with pytest.raises(ValueError, match="repeated source party diverged"):
        _validate_repeated_party_wording(source, printed, ROLES)
    _validate_repeated_party_wording(source, generated, ROLES)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("name", "OTHER COMPANY"),
        ("addressLine", "PLOT 15, AR RUMAYLAH, UNITED ARAB EMIRATES"),
        ("addressLine", "PLOT 14, DUBAI, UNITED ARAB EMIRATES"),
        ("addressLine", "PLOT 14, AR RUMAYLAH, EGYPT"),
        ("addressLine", "PLOT 14, AR RUMAYLAH, ARE"),
        ("addressLine", "PLOT 14, AR RUMAYLAH, A.R.E."),
        ("addressLine", None),
    ],
)
def test_genuinely_different_generated_identity_still_fails(field, replacement):
    source = flat(parties())
    generated = flat(parties("NEW COMPANY", "PLOT 14, AR RUMAYLAH, UNITED ARAB EMIRATES"))
    generated[ROLES[1] + "." + field] = replacement
    with pytest.raises(ValueError, match="repeated source party diverged"):
        _validate_repeated_party_wording(source, generated, ROLES)


def test_distinct_source_parties_are_not_forced_into_the_same_identity():
    source = parties()
    source["documentPatch"]["parties"]["notifyParties"][0]["name"] = "OTHER SOURCE COMPANY"
    generated = deepcopy(source)
    generated["documentPatch"]["parties"]["notifyParties"][0]["name"] = "OTHER NEW COMPANY"
    _validate_repeated_party_wording(flat(source), flat(generated), ROLES)


def test_missing_source_address_does_not_invent_an_identity_group():
    source = flat(parties())
    del source[ROLES[1] + ".addressLine"]
    generated = flat(parties())
    generated[ROLES[1] + ".name"] = "INDEPENDENT COMPANY"
    _validate_repeated_party_wording(source, generated, ROLES)
