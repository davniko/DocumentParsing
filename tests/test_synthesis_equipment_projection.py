from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import host
from document_ocr.synthesis.template_compiler.equipment_projection import (
    normalize,
    normalize_owned_auxiliary_receipts,
    normalize_receipt_suffixes,
)


def auxiliary_receipt_fixture(surface="40HQ"):
    raw = "ABCU1234567 " + surface
    source = {"documentPatch": {"containers": [{"containerNumber": "ABCU1234567"}]}}
    _, _, originals = fixture()
    draft = replace(
        originals[0],
        draft_id="aux",
        logical_key="aux",
        render_mode="deterministic_auxiliary",
        target_paths=(),
        char_start=12,
        char_end=len(raw),
        source_text=surface,
    )
    return raw, source, (draft,)


def test_source_only_equipment_owner_becomes_executable_without_extra_labels():
    raw, source, drafts = auxiliary_receipt_fixture()
    result = normalize_owned_auxiliary_receipts(raw=raw, drafts=drafts, source_target=source)
    assert result[0].derivation == "equipment_receipt"
    assert result[0].dependency_paths == ("documentPatch.containers[0]",)
    assert result[0].target_paths == ()
    assert source == {"documentPatch": {"containers": [{"containerNumber": "ABCU1234567"}]}}
    host.validate_binding_realizations(raw=raw, drafts=result, source_target=source)
    assert (
        normalize_owned_auxiliary_receipts(raw=raw, drafts=result, source_target=source) == result
    )
    compiled = host.normalize_structured_row_locality(raw=raw, drafts=drafts, source_target=source)
    assert next(d for d in compiled if d.logical_key == "aux").derivation == "equipment_receipt"


@pytest.mark.parametrize("length,kind", [("20", "GP"), ("40", "HC")])
def test_split_source_only_type_modifier_joins_its_existing_physical_receipt(length, kind):
    raw, source, originals = auxiliary_receipt_fixture(f"1x{length}'{kind}")
    source["documentPatch"]["containers"][0]["typeDescription"] = length
    first = replace(
        originals[0],
        char_end=16,
        source_text=f"1x{length}",
        render_mode="deterministic_derived",
        derivation="equipment_receipt",
        dependency_paths=("documentPatch.containers[0]",),
    )
    last = replace(
        originals[0], draft_id="suffix", logical_key="suffix", char_start=17, source_text=kind
    )
    result = normalize_receipt_suffixes(raw=raw, drafts=(first, last), source_target=source)
    assert len(result) == 1
    assert result[0].source_text == f"1x{length}'{kind}"
    assert result[0].dependency_paths == first.dependency_paths
    assert source["documentPatch"]["containers"][0]["typeDescription"] == length
    host.validate_binding_realizations(raw=raw, drafts=result, source_target=source)
    assert normalize_receipt_suffixes(raw=raw, drafts=result, source_target=source) == result
    for invalid in (
        replace(last, group_key="container:1"),
        replace(last, target_paths=("documentPatch.transport.voyageNumber",)),
        replace(last, dependency_paths=("documentPatch.containers[0]",)),
    ):
        assert normalize_receipt_suffixes(
            raw=raw, drafts=(first, invalid), source_target=source
        ) == (first, invalid)


@pytest.mark.parametrize("surface", ["40 UNKNOWN", "HC", "1X20GP + 1X40HC", "2X40HQ"])
def test_incomplete_or_mixed_receipts_cannot_be_promoted_to_one_row(surface):
    raw, source, drafts = auxiliary_receipt_fixture(surface)
    assert (
        normalize_owned_auxiliary_receipts(raw=raw, drafts=drafts, source_target=source) == drafts
    )


@pytest.mark.parametrize("change", ["missing_owner", "existing_dependency", "existing_label"])
def test_receipt_promotion_neither_infers_nor_overwrites_ownership(change):
    raw, source, drafts = auxiliary_receipt_fixture()
    if change == "missing_owner":
        drafts = (replace(drafts[0], group_key="equipment:unspecified"),)
    elif change == "existing_dependency":
        drafts = (replace(drafts[0], dependency_paths=("documentPatch.containers",)),)
    else:
        source["documentPatch"]["containers"][0]["typeDescription"] = "40"
    assert (
        normalize_owned_auxiliary_receipts(raw=raw, drafts=drafts, source_target=source) == drafts
    )


def test_receipt_promotion_rejects_nonexistent_explicit_owner():
    raw, source, drafts = auxiliary_receipt_fixture()
    drafts = (replace(drafts[0], group_key="container:1"),)
    with pytest.raises(ValueError, match="absent container owner"):
        normalize_owned_auxiliary_receipts(raw=raw, drafts=drafts, source_target=source)


def test_one_bad_occurrence_prevents_partial_promotion_of_its_binding():
    raw, source, drafts = auxiliary_receipt_fixture()
    drafts += (replace(drafts[0], draft_id="other", source_text="UNKNOWN"),)
    assert (
        normalize_owned_auxiliary_receipts(raw=raw, drafts=drafts, source_target=source) == drafts
    )


def fixture():
    raw = "MTSU9625961 SEAL001 40 DRY 9'6 40 PALLETS"
    path = "documentPatch.containers[0].typeDescription"
    source = {
        "documentPatch": {
            "containers": [{"containerNumber": "MTSU9625961", "typeDescription": "40 DRY 9'6"}],
            "cargoPackages": [{"quantity": 40}],
        }
    }

    def draft(key, start, text, paths, kind):
        return host.SpanDraft(
            draft_id=key,
            logical_key=key if kind == "equipment" else "quantity",
            render_mode="target_binding",
            value_kind=kind,
            group_kind="equipment" if kind == "equipment" else "package",
            group_key="container:0" if kind == "equipment" else "package:0",
            target_paths=paths,
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Source fixture",
        )

    rows = (
        draft("type", raw.index("DRY"), "DRY 9'6", (path,), "equipment"),
        draft(
            "false-quantity",
            raw.index("40"),
            "40",
            ("documentPatch.cargoPackages[0].quantity",),
            "integer",
        ),
        draft(
            "quantity",
            raw.rindex("40"),
            "40",
            ("documentPatch.cargoPackages[0].quantity",),
            "integer",
        ),
    )
    return raw, source, rows


def test_complete_row_type_is_not_a_quantity_even_when_the_numbers_coincide():
    raw, source, rows = fixture()
    fixed = normalize(raw=raw, drafts=rows, source_target=source)
    assert {d.draft_id for d in fixed} == {"type", "quantity"}
    equipment = next(d for d in fixed if d.draft_id == "type")
    assert equipment.source_text == "40 DRY 9'6"
    host.validate_draft_source_alignment(raw=raw, drafts=fixed)
    assert normalize(raw=raw, drafts=fixed, source_target=source) == fixed
    from document_ocr.synthesis.template_compiler.descendant import (
        _render_semantic_equipment_binding,
    )

    slots = host._template_slots(raw, (equipment,))
    binding = NS(
        target_paths=equipment.target_paths,
        occurrences=slots,
        realization=host.binding_realization(draft=equipment, slots=slots, source_target=source),
    )
    output = _render_semantic_equipment_binding(
        binding,
        {
            "documentPatch": {
                "containers": [
                    {
                        "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                        "typeCategory": "GENERAL_PURPOSE",
                    }
                ]
            }
        },
    )
    assert all("20" in v for v in output.replacements.values())


@pytest.mark.parametrize("mode", ["sole_quantity", "unrelated_row", "independent_reference"])
def test_repair_requires_complete_role_and_quantity_evidence(mode):
    raw, source, rows = fixture()
    if mode == "sole_quantity":
        rows = rows[:2]
    elif mode == "unrelated_row":
        source["documentPatch"]["containers"][0]["containerNumber"] = "OTHER"
    else:
        rows = (
            rows[0],
            replace(
                rows[1],
                logical_key="reference",
                target_paths=("documentPatch.billOfLadingNumber",),
            ),
            rows[2],
        )
    assert normalize(raw=raw, drafts=rows, source_target=source) == host.merge_drafts(rows)


@pytest.mark.parametrize(
    "original,surfaces",
    [
        ("40'HQ FCL/FCL", ("40'HQ",)),
        ("1X40HC CONTAINER", ("40HC", "1X40HC CONTAINER")),
        ("40 DRY 8'6", ("40 DRY", "40 DRY 8'6")),
        ("20 DRY 8'6", ("20DRY", "20 DRY 8'6")),
        ("40GP", ("40GP", "40'")),
    ],
)
def test_owned_physical_abbreviations_are_not_missing_equipment_prefixes(original, surfaces):
    from types import SimpleNamespace as NS

    from document_ocr.synthesis.template_compiler.equipment_projection import (
        validate_source_surfaces,
    )

    binding = NS(
        realization=NS(target_values=(NS(source_value=original),)),
        occurrences=tuple(NS(source_text=s) for s in surfaces),
    )
    validate_source_surfaces(binding)


@pytest.mark.parametrize("surface", ["DRY 9'6", "HIGH CUBE", "40 CARTONS", "20GP"])
def test_partial_equipment_check_rejects_missing_or_conflicting_physical_ownership(surface):
    from types import SimpleNamespace as NS

    from document_ocr.synthesis.template_compiler.equipment_projection import (
        validate_source_surfaces,
    )

    binding = NS(
        realization=NS(target_values=(NS(source_value="40 DRY 9'6"),)),
        occurrences=(NS(source_text=surface),),
    )
    with pytest.raises(ValueError, match="physical ownership"):
        validate_source_surfaces(binding)
