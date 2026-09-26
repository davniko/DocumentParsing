"""OCR can place both measurement values before their ordered unit codes."""

from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.descendant import _derivation_measurement_factor
from document_ocr.synthesis.template_compiler.host import (
    validate_compiler_measurement_unit_grounding,
)
from document_ocr.synthesis.template_compiler.tare_equations import (
    apply_equations,
    compile_equations,
    validate_equations,
)


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


def test_compiler_unit_grounding_distinguishes_wrong_label_from_dual_unit_printing():
    path = "documentPatch.cargoGroups[0].grossWeight.unit"
    target = {
        "documentPatch": {
            "cargoGroups": [{"grossWeight": {"value": 1326, "unit": "kilogram"}}]
        }
    }

    def slot(text):
        return NS(render_mode="target_binding", target_paths=(path,), source_text=text)

    with pytest.raises(ValueError, match="measurement unit binding conflicts"):
        validate_compiler_measurement_unit_grounding(
            source_target=target, drafts=(slot("MT"),)
        )
    validate_compiler_measurement_unit_grounding(
        source_target=target, drafts=(slot("KG"), slot("MT"))
    )


def test_printed_net_tare_gross_equation_is_sampled_and_validated_as_one_fact():
    raw = b"NETT WEIGHT / LOADED WEIGHT 12000 KGS TARE 3640 KGS gross weight (tare+nett) 15640 KGS"
    offset = raw.index(b"3640")
    binding = NS(
        logical_key="tare", occurrences=(
            NS(source_text="3640", byte_start=offset, byte_end=offset + 4),
        ),
    )
    source_target = {"documentPatch": {"cargoGroups": [{
        "groupId": "g1", "netWeight": {"value": 12000, "unit": "kilogram"},
        "grossWeight": {"value": 15640, "unit": "kilogram"},
    }]}}
    source = NS(source=raw, target=source_target, template=NS(bindings=(binding,)))
    contract = NS(role="tare", mode="source_fixed", target_paths=[], source_value="3640")
    equations = compile_equations(source, {"tare": contract})
    assert len(equations) == 1 and equations[0].tare_kilograms == Decimal(3640)
    target = {"documentPatch": {"cargoGroups": [{
        "groupId": "g1", "netWeight": {"value": 17000, "unit": "kilogram"},
        "grossWeight": {"value": 19100, "unit": "kilogram"},
    }]}}
    with pytest.raises(ValueError, match="differs from net plus printed tare"):
        validate_equations(target, equations)
    apply_equations(target, equations)
    assert target["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] == 20640
    validate_equations(target, equations)
    source.target["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] = 15500
    with pytest.raises(ValueError, match="source gross weight does not equal"):
        compile_equations(source, {"tare": contract})
