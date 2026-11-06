"""Identifier-only cohorts must not edit overlapping references or coordinates."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/synthesis"))
    return importlib.import_module("refresh_r29_long_sources")


def test_seal_substring_in_bl_is_not_an_independent_occurrence(module):
    row = dict(
        joinedRawText="BL NRK7291846\nABCU1234567\nSEAL K7291846\nBL NRK7291846",
        target=dict(
            billOfLadingNumber="NRK7291846",
            containerInformation=[
                dict(equipmentIdentifier="ABCU1234567", sealNumbers=["K7291846"])
            ],
        ),
    )
    contract = module.identifier_contract(row)
    seal = next(v for v in contract.variables if v.value == "K7291846")
    assert len(seal.occurrences) == 1
    assert seal.occurrences[0].occurrence == 2
    assert len(next(v for v in contract.variables if v.value == "NRK7291846").occurrences) == 2
    row["documentId"] = "reviewed-source"
    template, _ = module.compile_identifiers(row, contract)
    assert len(template.slots) == 4


def test_coordinates_and_unpositioned_lines_are_exactly_retained(module):
    row = dict(joinedRawText="OLD\n\nTAIL", positionedText="OLD || 20,30\n\nTAIL ||")
    assert module.positioned_overlay(row, "NEW\n\nTAIL") == "NEW || 20,30\n\nTAIL ||"
    with pytest.raises(ValueError):
        module.positioned_overlay(row, "NEW\nEXTRA\n\nTAIL")


def test_missing_identifier_is_rejected(module):
    with pytest.raises(ValueError, match="not present"):
        module.identifier_contract(
            dict(joinedRawText="OTHER", target=dict(billOfLadingNumber="ABSENT"))
        )
