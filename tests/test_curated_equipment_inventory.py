"""Equipment-row authority, count syntax and immutable header-rebase controls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.container_semantics import canonical_equipment_surface
from document_ocr.synthesis.curated import Occurrence, SourceContract, TargetBinding, Variable
from document_ocr.synthesis.curated_equipment_inventory import (
    compile_equipment_inventory,
    project_equipment_inventory,
    rebase_equipment_inventory_headers,
)
from document_ocr.synthesis.curated_templates import RenderRegion, SamplingBlueprint
from document_ocr.synthesis.template_compiler.descendant import _number_to_words

S20 = "TWENTY_FOOT_STANDARD_HEIGHT"
H40 = "FORTY_FOOT_HIGH_CUBE"
GP = "GENERAL_PURPOSE"


def blueprint(codes=("20GP", "20GP"), extra="", scalar=True):
    identifiers = [f"ABCU{index:07d}" for index in range(len(codes))]
    source = (
        "".join(f"{number} {code}\n" for number, code in zip(identifiers, codes, strict=True))
        + extra
    )
    regions, bindings, variables, targets, rows = [], {}, [], [], []
    cursor = 0
    for index, (number, code) in enumerate(zip(identifiers, codes, strict=True)):
        path = f"documentPatch.containerInformation[{index}].equipmentIdentifier"
        key = f"identity{index}"
        start = source.index(number, cursor)
        regions.append(
            RenderRegion(key, 0, start, start + len(number), number, (path,), "identifier", key)
        )
        variables.append(
            Variable(
                key=key,
                kind="container",
                value=number,
                meaning="Container identity",
                required_literals=[],
                occurrences=[Occurrence(text=number, occurrence=1, presentation="text")],
            )
        )
        targets.append(TargetBinding(path=path, expression="{" + key + "}"))
        start = source.index(code, start + len(number))
        cursor = start + len(code)
        if scalar:
            eq_path = f"documentPatch.containers[{index}].typeDescription"
            eq_key = f"equipment{index}"
            regions.append(
                RenderRegion(
                    eq_key,
                    0,
                    start,
                    cursor,
                    code,
                    (
                        f"documentPatch.containerInformation[{index}].sizeCategory",
                        f"documentPatch.containerInformation[{index}].typeCategory",
                    ),
                    "equipment",
                )
            )
            bindings[eq_key] = {
                "target_paths": [eq_path],
                "occurrences": [{"byte_start": start, "byte_end": cursor, "source_text": code}],
                "target_relationship": "single_target",
            }
        rows.append(
            {
                "equipmentIdentifier": number,
                "sizeCategory": H40 if "40" in code else S20,
                "typeCategory": GP,
            }
        )
    return SamplingBlueprint(
        "source-test",
        source,
        {"documentPatch": {"containerInformation": rows}},
        SourceContract(
            variables=variables, targets=targets, fixed_context="Equipment ownership test"
        ),
        tuple(regions),
        bindings,
        {},
        (),
    )


def add_owner(
    bp, key, literal, paths, derivation=None, relationship=None, occurrence=0, start=None
):
    start = bp.source.rindex(literal) if start is None else start
    region = RenderRegion(key, occurrence, start, start + len(literal), literal, (), "equipment")
    bindings = deepcopy(bp.historical_bindings)
    bindings[key] = {
        "target_paths": paths,
        "derivation": derivation,
        "target_relationship": relationship,
        "occurrences": [{"byte_start": start, "byte_end": region.end, "source_text": literal}],
    }
    return replace(
        bp,
        regions=tuple(sorted((*bp.regions, region), key=lambda item: item.start)),
        historical_bindings=bindings,
    )


def project(contract, bp, mixed=True):
    rows = deepcopy(bp.target["documentPatch"]["containerInformation"])
    for index, row in enumerate(rows):
        row["sizeCategory"] = S20 if mixed and index == 0 else H40
    return project_equipment_inventory(
        contract,
        bp.target["documentPatch"]["containerInformation"],
        rows,
        format_equipment=lambda row, source: canonical_equipment_surface(
            row["sizeCategory"], row["typeCategory"]
        ),
        number_words=_number_to_words,
    )


@pytest.mark.parametrize(
    "count",
    [
        "2 CTNR",
        "2 CTRS",
        "TWO CONTAINERS TOTAL",
        "SAY TWO CONTAINERS TOTAL",
        "TWO (2) CONTAINERS ONLY",
        "2 X",
        "02",
        "2",
    ],
)
def test_declared_count_grammar_is_exact_and_preserved(count):
    bp = add_owner(
        blueprint(extra=count), "count", count, ["documentPatch.containers"], "container_count"
    )
    contract = compile_equipment_inventory(bp)
    surfaces, _ = project(contract, bp)
    assert surfaces["count"] == [count]


@pytest.mark.parametrize(
    "count", ["3 CTNR", "TWO (3) CONTAINERS", "THREE CONTAINERS TOTAL", "2 KGS"]
)
def test_wrong_or_non_container_counts_rejected(count):
    bp = add_owner(
        blueprint(extra=count), "count", count, ["documentPatch.containers"], "container_count"
    )
    with pytest.raises(ValueError):
        compile_equipment_inventory(bp)


@pytest.mark.parametrize("surface", ["2X20GP", "20GP*2", "TWO(2X20GP) CONTAINERS ONLY"])
def test_explicit_full_receipt_recovers_missing_scope(surface):
    bp = add_owner(blueprint(extra=surface), "receipt", surface, [], "equipment_receipt")
    contract = compile_equipment_inventory(bp)
    assert contract.scope_repairs[0].indices == (0, 1)
    assert contract.groups == ((0,), (1,))
    surfaces, _ = project(contract, bp)
    assert "20" in surfaces["receipt"][0] and "40" in surfaces["receipt"][0]


def test_ambiguous_unscoped_subset_is_not_guessed():
    bp = add_owner(blueprint(extra="1X20GP"), "receipt", "1X20GP", [], "equipment_receipt")
    with pytest.raises(ValueError, match="explicit whole-inventory"):
        compile_equipment_inventory(bp)


def test_whole_receipt_does_not_create_direct_ownership():
    bp = add_owner(
        blueprint(extra="2X20GP", scalar=False),
        "receipt",
        "2X20GP",
        ["documentPatch.containers"],
        "equipment_receipt",
    )
    with pytest.raises(ValueError, match="direct printed type"):
        compile_equipment_inventory(bp)


def test_shared_scalar_equates_members_not_derived_receipts():
    bp = blueprint(extra="20GP", scalar=False)
    bp = add_owner(
        bp,
        "shared",
        "20GP",
        [
            "documentPatch.containers[0].typeDescription",
            "documentPatch.containers[1].typeDescription",
        ],
        relationship="shared_value_equality",
    )
    contract = compile_equipment_inventory(bp)
    assert contract.groups == ((0, 1),)
    with pytest.raises(ValueError, match="shared scalar"):
        project(contract, bp)
    assert project(contract, bp, mixed=False)[0]["shared"]


@pytest.mark.parametrize(
    "header,count,gap",
    [
        ("2 CONTAINERS OF 20GP", "2 CONTAINERS", " OF "),
        ("2X20GP", "2", "X"),
        ("20GP X2", "2", " X"),
    ],
)
def test_header_rebase_preserves_input_and_derives_whole_inventory(header, count, gap):
    original = add_owner(
        blueprint(extra=header),
        "header_count",
        count,
        ["documentPatch.containers"],
        "container_count",
        start=(blueprint(extra=header).source.rfind("\n") + 1)
        if header.startswith(count + gap)
        else None,
    )
    original = add_owner(
        original, "header_type", "20GP", ["documentPatch.containers[0].typeDescription"]
    )
    if gap.strip() == "X":
        original = add_owner(original, "separator", "X", [])
    old_regions = original.regions
    result = rebase_equipment_inventory_headers(original)
    assert result.source == original.source and result.target == original.target
    assert original.regions == old_regions
    assert rebase_equipment_inventory_headers(result) is result
    contract = compile_equipment_inventory(result)
    header_span = next(span for span in contract.spans if span.kind == "header")
    assert header_span.indices == (0, 1)
    rendered, _ = project(contract, result)
    assert "20" in rendered[header_span.key][0] and "40" in rendered[header_span.key][0]


def test_header_cannot_consume_non_equipment_ownership():
    bp = add_owner(
        blueprint(extra="2X20GP"),
        "count",
        "2",
        ["documentPatch.containers"],
        "container_count",
        start=blueprint(extra="2X20GP").source.rfind("\n") + 1,
    )
    bp = add_owner(bp, "typed", "20GP", ["documentPatch.containers[0].typeDescription"])
    bp = add_owner(bp, "protected", "X", [])
    protected = next(region for region in bp.regions if region.key == "protected")
    bp = replace(
        bp,
        regions=tuple(
            replace(region, target_paths=("documentPatch.goodsItemDetails[0].description",))
            if region is protected
            else region
            for region in bp.regions
        ),
    )
    with pytest.raises(ValueError, match="non-structural"):
        rebase_equipment_inventory_headers(bp)


def test_current_curated_contract_overrides_stale_historical_identity_paths():
    bp = blueprint()
    first = bp.regions[0]
    bp = replace(
        bp,
        regions=(
            replace(
                first,
                target_paths=(
                    *first.target_paths,
                    "documentPatch.containerInformation[1].equipmentIdentifier",
                ),
            ),
            *bp.regions[1:],
        ),
    )
    assert len(compile_equipment_inventory(bp).identity_spans) == 2


def test_uniform_source_types_cannot_hide_swapped_identifier_associations():
    bp = blueprint()
    bindings = deepcopy(bp.historical_bindings)
    bindings["equipment0"]["target_paths"] = ["documentPatch.containers[1].typeDescription"]
    bindings["equipment1"]["target_paths"] = ["documentPatch.containers[0].typeDescription"]
    with pytest.raises(ValueError, match="co-printed identifier row"):
        compile_equipment_inventory(replace(bp, historical_bindings=bindings))


def test_distinct_line_types_do_not_infer_an_unprinted_row_relationship():
    bp = blueprint()
    source = bp.source.replace(" 20GP", "\n20GP")
    regions = []
    bindings = deepcopy(bp.historical_bindings)
    for part in bp.regions:
        # Each substituted separator has the same byte width; no spans shift.
        regions.append(part)
    compile_equipment_inventory(
        replace(bp, source=source, regions=tuple(regions), historical_bindings=bindings)
    )


def test_measure_owner_rejected_when_source_identifier_proves_wrong_row():
    # Use an ordinary valid equipment owner plus a disjoint numeric region.
    fixed = blueprint()
    source = fixed.source.replace("20GP\n", "20GP 100 KGS\n")
    bp = blueprint(codes=("20GP", "20GP"))
    second = source.index("ABCU0000001")
    first_type = source.index("20GP")
    second_type = source.index("20GP", first_type + 1)
    shifted = []
    for region in bp.regions:
        start = (
            source.index(region.source)
            if region.key.endswith("0")
            else (second if region.curated_key else second_type)
        )
        shifted.append(replace(region, start=start, end=start + len(region.source)))
    bindings = deepcopy(bp.historical_bindings)
    for region in shifted:
        if region.key in bindings:
            bindings[region.key]["occurrences"] = [
                {"byte_start": region.start, "byte_end": region.end, "source_text": region.source}
            ]
    bp = replace(bp, source=source, regions=tuple(shifted), historical_bindings=bindings)
    bp = add_owner(bp, "weight", "100", [], start=source.index("100"))
    bp = replace(bp, ownership_data={"measure_owners": {"weight": {"row": 1}}})
    with pytest.raises(ValueError, match="row-dependent owner weight"):
        compile_equipment_inventory(bp)


def test_placement_rows_are_resolved_by_identifier_not_array_ordinal():
    bp = blueprint()
    target = deepcopy(bp.target)
    target["documentPatch"]["goodsItemDetails"] = [
        {
            "splitGoodsPlacement": [
                {"equipmentIdentifier": "ABCU0000001", "packageQuantity": 7},
                {"equipmentIdentifier": "ABCU0000000", "packageQuantity": 8},
            ]
        }
    ]
    path = "documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].equipmentIdentifier"
    contract = bp.contract.model_copy(
        update={
            "targets": [*bp.contract.targets, TargetBinding(path=path, expression="{identity0}")]
        }
    )
    # Identity0 still owns container row0. The placement reference contradicts
    # the same source ID even though its placement ordinal is also zero.
    with pytest.raises(ValueError, match="row-dependent owner identity0"):
        compile_equipment_inventory(replace(bp, target=target, contract=contract))


def test_hash_cardinality_and_corrupt_source_guards():
    bp = blueprint()
    contract = compile_equipment_inventory(bp)
    assert contract.source_target_sha256 == sha256_bytes(canonical_json_bytes(bp.target))
    assert contract.source_sha256 == sha256_bytes(bp.source.encode())
    with pytest.raises(ValueError, match="row cardinality"):
        contract.validate_rows(bp.target["documentPatch"]["containerInformation"][:1])
    with pytest.raises(ValueError, match="exact source bytes"):
        compile_equipment_inventory(replace(bp, source="!" + bp.source))


@pytest.mark.parametrize("surface", ["40'HQ FCL/FCL", "20' Dry Heavy Duty", "40SD96"])
def test_complete_reviewed_scalar_aliases_preserve_movement(surface):
    bp = blueprint(codes=(surface, surface))
    contract = compile_equipment_inventory(bp)
    result, _ = project(contract, bp)
    assert len(result) == 2
    if "FCL/FCL" in surface:
        assert all(value[0].endswith("FCL/FCL") for value in result.values())


def test_partial_length_receipt_does_not_override_direct_high_cube_evidence():
    bp = add_owner(
        blueprint(codes=("40HC", "40HC"), extra="2X40' CONTAINER"),
        "receipt",
        "2X40' CONTAINER",
        ["documentPatch.containers"],
        "equipment_receipt",
    )
    contract = compile_equipment_inventory(bp)
    assert project(contract, bp)[0]["receipt"]
