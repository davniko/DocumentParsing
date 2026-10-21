from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace

import pytest
from test_curated_equipment_inventory import GP, H40, S20, add_owner, blueprint

from document_ocr.synthesis.curated_equipment_aliases import validate_equipment_alias
from document_ocr.synthesis.curated_equipment_inventory import (
    compile_equipment_inventory,
    project_equipment_inventory,
    rebase_equipment_inventory_headers,
)
from document_ocr.synthesis.curated_physical import _complete_equipment_surface
from document_ocr.synthesis.curated_templates import RenderRegion
from document_ocr.synthesis.template_compiler.descendant import _number_to_words
from document_ocr.synthesis.template_compiler.equipment_receipts import validate_source_receipt


def header_blueprint(text="2*20GP", count="2"):
    bp = blueprint(extra=text)
    start = bp.source.rfind("\n") + 1
    (match,) = re.finditer(r"(?<!\d)" + re.escape(count) + r"(?!\d)", text)
    count_start = start + match.start()
    bp = add_owner(
        bp, "count", count, ["documentPatch.containers"], "container_count", start=count_start
    )
    return add_owner(bp, "type", "20GP", ["documentPatch.containers[0].typeDescription"])


@pytest.mark.parametrize(
    "text,count",
    [
        ("2*20GP", "2"),
        ("2X20GP", "2"),
        ("20GP X2", "2"),
        ("2 CONTAINERS OF 20GP", "2 CONTAINERS"),
    ],
)
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_header_formatter_receives_only_exact_type_source(text, count, alias, mixed):
    original = header_blueprint(text, count)
    bp = rebase_equipment_inventory_headers(original)
    contract = compile_equipment_inventory(bp)
    header = next(span for span in contract.spans if span.kind == "header")
    assert header.type_source == "20GP"
    (certificate,) = bp.ownership_data["inventory_headers"]
    assert original.source.encode()[certificate["typeStart"] : certificate["typeEnd"]] == b"20GP"
    rows = deepcopy(bp.target["documentPatch"]["containerInformation"])
    if mixed:
        rows[1]["sizeCategory"] = H40
    seen = []

    def formatter(row, source):
        seen.append(source)
        assert source == "20GP"
        if not alias:
            return _complete_equipment_surface(row, source)
        result = {S20: "22G1", H40: "45G1"}[row["sizeCategory"]]
        validate_equipment_alias(result, row["sizeCategory"], GP)
        return result

    surfaces, _ = project_equipment_inventory(
        contract,
        bp.target["documentPatch"]["containerInformation"],
        rows,
        format_equipment=formatter,
        number_words=_number_to_words,
    )
    assert seen
    surface = surfaces[header.key][0]
    assert "2*" not in surface and "2X" not in surface
    validate_source_receipt(surface, rows, number_words=_number_to_words)
    if mixed:
        assert " + " in surface
        with pytest.raises(ValueError):
            validate_source_receipt("2X20GP", rows, number_words=_number_to_words)
    else:
        assert " + " not in surface


@pytest.mark.parametrize(
    "amendment",
    [
        {"typeSource": "OTHER"},
        {"typeStart": -1},
        {"typeEnd": 100000},
        {"typeStart": None},
        {"typeSource": None},
    ],
)
def test_header_type_certificate_is_exact_not_an_unverified_format_hint(amendment):
    bp = rebase_equipment_inventory_headers(header_blueprint())
    bp.ownership_data["inventory_headers"][0].update(amendment)
    with pytest.raises(ValueError, match="exact type-only source span"):
        compile_equipment_inventory(bp)


def test_header_certificate_cannot_make_whole_counted_surface_a_type():
    bp = rebase_equipment_inventory_headers(header_blueprint())
    certificate = bp.ownership_data["inventory_headers"][0]
    certificate.update(
        typeSource=certificate["source"], typeStart=certificate["start"], typeEnd=certificate["end"]
    )
    with pytest.raises(ValueError, match="exact type-only source span"):
        compile_equipment_inventory(bp)


def test_header_rebase_does_not_rewrite_untouched_unrelated_realization():
    bp = header_blueprint()
    source = bp.source + "\nPallet Pallet PALLETS"
    occurrences, regions = [], list(bp.regions)
    cursor = len(bp.source)
    for occurrence, literal in enumerate(("Pallet", "Pallet", "PALLETS")):
        start = source.index(literal, cursor)
        cursor = start + len(literal)
        occurrences.append({"byte_start": start, "byte_end": cursor, "source_text": literal})
        regions.append(
            RenderRegion("unrelated_package", occurrence, start, cursor, literal, (), "package")
        )
    bindings = deepcopy(bp.historical_bindings)
    bindings["unrelated_package"] = {
        "target_paths": [
            "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].typeCategory"
        ],
        "occurrences": occurrences,
        "realization": {"slots": [{"original": "preserved"}]},
    }
    bp = replace(bp, source=source, regions=tuple(regions), historical_bindings=bindings)
    result = rebase_equipment_inventory_headers(bp)
    assert result.historical_bindings["unrelated_package"] == bindings["unrelated_package"]
    assert result.source == source and result.target == bp.target
    compile_equipment_inventory(result)
