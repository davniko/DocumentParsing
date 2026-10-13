"""A reviewed exclusion must reject old, internally matched source authorities."""

from copy import deepcopy

import pytest
from test_curated_ownership_omissions import omission_fixture

from document_ocr.synthesis.curated_ownership import (
    _validate_required_target_absences,
    build_owned_blueprint,
)

PATH = "documentPatch.parties.consignee.contactDetails.phoneNumbers"


def source_with_absence(value=None):
    row, history, contract, _ = omission_fixture()
    if value:
        row["target"]["documentPatch"]["parties"]["consignee"] = {"contactDetails": value}
    declarations = {"required_target_absences": {PATH: "Only occurrence belongs to goods."}}
    return row, history, contract, declarations


@pytest.mark.parametrize("value", [{"phoneNumbers": ["123456789"]}, {"phoneNumbers": "123456789"}])
def test_reviewed_absence_rejects_populated_old_source(value):
    with pytest.raises(ValueError, match="source target contradicts required absence"):
        build_owned_blueprint(*source_with_absence(value))


def test_absent_source_keeps_compilation_exact():
    args = source_with_absence()
    before = deepcopy(args)
    with_guard = build_owned_blueprint(*args)
    without_guard = build_owned_blueprint(*args[:3], {})
    assert with_guard.source == without_guard.source
    assert with_guard.contract == without_guard.contract
    assert with_guard.regions == without_guard.regions
    assert with_guard.target == without_guard.target
    assert args == before


def test_absence_checks_path_boundaries_not_string_prefixes():
    declarations = {"required_target_absences": {PATH: "Reviewed exclusion."}}
    leaves = {PATH + "Note": "not a phone"}
    _validate_required_target_absences(leaves, declarations)
    _validate_required_target_absences({PATH: None}, declarations)
    declarations = {
        "required_target_absences": {
            "documentPatch.parties.consignee.contactDetails": "All contacts omitted by review."
        }
    }
    with pytest.raises(ValueError, match="source target contradicts required absence"):
        _validate_required_target_absences(leaves, declarations)


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        [],
        {PATH: " "},
        {PATH: 7},
        {"documentPatch..parties": "reason"},
        {"documentPatch.parties[foo]": "reason"},
        {"documentPatch.parties.consignee.contactDetails.phoneNumbers[00]": "reason"},
        {"documentPatch.notAField": "reason"},
        {"schemaVersion": "reason"},
        {7: "reason"},
    ],
)
def test_invalid_absence_contract_is_an_error(declaration):
    row, history, contract, _ = source_with_absence()
    with pytest.raises(ValueError, match="required target absence"):
        build_owned_blueprint(row, history, contract, {"required_target_absences": declaration})
