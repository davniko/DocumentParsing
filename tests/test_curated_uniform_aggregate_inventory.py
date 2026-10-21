from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from test_curated_equipment_inventory import H40, S20, add_owner, blueprint

from document_ocr.synthesis.curated_equipment_inventory import (
    compile_equipment_inventory,
    project_equipment_inventory,
)
from document_ocr.synthesis.curated_physical import _complete_equipment_surface
from document_ocr.synthesis.curated_templates import RenderRegion
from document_ocr.synthesis.template_compiler.descendant import _number_to_words


def grouped_blueprint(receipt="2X20GP", direct_first=False):
    bp = blueprint(scalar=False)
    rows = bp.target["documentPatch"]["containerInformation"]
    first, second = (row["equipmentIdentifier"] for row in rows)
    source = first + (" 20GP" if direct_first else "") + "\n" + second + "\n" + receipt
    regions = tuple(
        RenderRegion(
            f"identity{index}",
            0,
            source.index(number),
            source.index(number) + len(number),
            number,
            (f"documentPatch.containerInformation[{index}].equipmentIdentifier",),
            "identifier",
            f"identity{index}",
        )
        for index, number in enumerate((first, second))
    )
    bp = replace(bp, source=source, regions=regions)
    if direct_first:
        bp = add_owner(
            bp,
            "direct_first",
            "20GP",
            ["documentPatch.containers[0].typeDescription"],
            start=source.index("20GP"),
        )
    return add_owner(bp, "aggregate", receipt, ["documentPatch.containers"], "equipment_receipt")


@pytest.mark.parametrize("direct_first", [False, True])
def test_reviewed_uniform_aggregate_is_explicit_opt_in_only(direct_first):
    bp = grouped_blueprint(direct_first=direct_first)
    original = deepcopy(bp.target)
    with pytest.raises(ValueError, match="direct printed type ownership"):
        compile_equipment_inventory(bp)
    contract = compile_equipment_inventory(bp, allow_uniform_aggregate=True)
    witness = contract.uniform_aggregate
    assert witness is not None
    assert witness.owners == (("aggregate", 0),)
    assert witness.indices == (0, 1)
    assert witness.unmapped_indices == ((1,) if direct_first else (0, 1))
    assert witness.source_pair == (S20, "GENERAL_PURPOSE")
    assert contract.groups == ((0, 1),)
    assert bp.target == original
    rows = deepcopy(bp.target["documentPatch"]["containerInformation"])
    for row in rows:
        row["sizeCategory"] = H40
    surfaces, _ = project_equipment_inventory(
        contract,
        bp.target["documentPatch"]["containerInformation"],
        rows,
        format_equipment=_complete_equipment_surface,
        number_words=_number_to_words,
    )
    assert "40" in surfaces["aggregate"][0]
    rows[0]["sizeCategory"] = S20
    with pytest.raises(ValueError, match="uniform-only aggregate"):
        contract.validate_rows(rows)
    with pytest.raises(ValueError, match="uniform-only aggregate"):
        project_equipment_inventory(
            contract,
            bp.target["documentPatch"]["containerInformation"],
            rows,
            format_equipment=_complete_equipment_surface,
            number_words=_number_to_words,
        )


@pytest.mark.parametrize("receipt", ["20GP", "1X20GP", "2X40HC", "2"])
def test_missing_incomplete_count_or_contradictory_type_cannot_enable_uniform_capability(receipt):
    with pytest.raises(ValueError):
        compile_equipment_inventory(grouped_blueprint(receipt), allow_uniform_aggregate=True)


def test_mixed_source_aggregate_does_not_claim_an_unobserved_per_id_mapping():
    bp = grouped_blueprint("1X20GP + 1X40HC")
    bp.target["documentPatch"]["containerInformation"][1]["sizeCategory"] = H40
    with pytest.raises(ValueError, match="direct printed type ownership"):
        compile_equipment_inventory(bp, allow_uniform_aggregate=True)
