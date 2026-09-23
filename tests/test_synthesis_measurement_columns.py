"""OCR can place both measurement values before their ordered unit codes."""

from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.descendant import _derivation_measurement_factor


def _factor(raw, surface, field, unit):
    source = raw.encode()
    start = source.index(surface.encode())
    path = "documentPatch.cargoGroups[0]." + field + ".value"
    slot = NS(source_text=surface, byte_start=start, byte_end=start + len(surface.encode()))
    binding = NS(derivation=None, target_paths=(path,), occurrences=(slot,))
    target = {"documentPatch": {"cargoGroups": [{field: {"unit": unit, "value": 20}}]}}
    case = NS(source=source, source_target=target, target=target, template=NS(bindings=()))
    return _derivation_measurement_factor(binding, case, value_paths=(path,))


@pytest.mark.parametrize(
    ("raw", "surface", "field", "unit"),
    [
        ("16428.000 20.000 KGM MTQ", "20.000", "volume", "cubic_metre"),
        ("16428.000 20.000 KGM MTQ", "16428.000", "grossWeight", "kilogram"),
        ("é ABCU1234567 10 BOXES 12.500 2.000 KGM MTQ\n", "2.000", "volume", "cubic_metre"),
        ("Weight Measurement\n16428.000 20.000 kg cbm", "20.000", "volume", "cubic_metre"),
        ("16428.000 20.000 LBR FTQ", "20.000", "volume", "cubic_foot"),
    ],
)
def test_ordered_trailing_units_do_not_borrow_the_other_column(raw, surface, field, unit):
    assert _factor(raw, surface, field, unit) == (Decimal(1), Decimal(0))


@pytest.mark.parametrize(
    "raw",
    [
        "20.000 KGM",
        "16428.000 20.000 MTQ KGM",
        "16428.000 20.000 KGM\nMTQ",
        "16428.000 20.000 KGM unrelated MTQ",
    ],
)
def test_actual_incompatible_or_unproved_row_still_fails(raw):
    with pytest.raises(ValueError, match="dimensionally incompatible"):
        _factor(raw, "20.000", "volume", "cubic_metre")
