"""Source-proven aggregate package noun ownership and rendering."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler import package_count_surfaces
from document_ocr.synthesis.template_compiler.descendant import (
    _derivation_numeric_values,
    _render_proven_numeric_derivation,
)
from document_ocr.synthesis.template_compiler.host import SpanDraft, validate_draft_source_alignment

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / (
    "artifacts/kie-synthesis-production/template-base/catalogs/"
    "mpci-bl-production-template-catalog1507-v34-discharge-country/cases"
)
CASES_BY_ID = {
    "boxes": (
        "doc_15b8d67061fb6bf16f0619a88fa74029897e96fc51cfd49459e963dbb797997c",
        "binding_0038",
    ),
    "bags": (
        "doc_88ec16af95fce3eaea2950bc205a2820b5b289c64c12a4d32835b1a8d3340375",
        "binding_0066",
    ),
    "pallet": (
        "doc_5c5e51a90b1e941b75e5619d9b4a56ef05149842da9709984fa9bee730c19715",
        "binding_0062",
    ),
}

OWNER_TRANSFER_CASES = (
    ("doc_05f91e32c5bc536df2c3fb58bd57d74da9d3ba3ed114a99356189b9152248bb1", 4),
    ("doc_0ba14e227c67807f66ffef2f30cbef542d221a18986f0f2a713fdf3fa613b82a", 2),
    ("doc_1149521ede78a0527c6a82ca4db69a97a2ad6c6da9a2c6ecb4db9be5d65b7a37", 4),
    ("doc_70f23755c33accfd7190d310a0a061ca629bde2cf54629f96203c43c59bdc11f", 2),
    ("doc_b39941474ea8ce76ecc2df3c98cb30f0208799b8065dc8e67d9568a18b7a34ca", 1),
)


def _case(name: str) -> tuple[str, dict, dict, SpanDraft]:
    source_id, binding_id = CASES_BY_ID[name]
    case = CASES / source_id
    raw = (case / "source.txt").read_text(encoding="utf-8")
    template = json.loads((case / "template.json").read_text(encoding="utf-8"))
    target = json.loads((case / "source-label.json").read_text(encoding="utf-8"))
    binding = next(b for b in template["bindings"] if b["binding_id"] == binding_id)
    slot = binding["occurrences"][0]
    draft = SpanDraft(
        draft_id="test_aggregate",
        logical_key=binding["logical_key"],
        render_mode=binding["render_mode"],
        value_kind=binding["value_kind"],
        group_kind=binding["group_kind"],
        group_key=binding["group_key"],
        target_paths=tuple(binding["target_paths"]),
        derivation=binding["derivation"],
        dependency_paths=tuple(binding["dependency_paths"]),
        dependency_bindings=tuple(binding["dependency_bindings"]),
        char_start=len(raw.encode()[: slot["byte_start"]].decode()),
        char_end=len(raw.encode()[: slot["byte_end"]].decode()),
        source_text=slot["source_text"],
        evidence_origin=slot["evidence_origin"],
        render_policy=slot["render_policy"],
        rationale="source-proven test",
    )
    return raw, target, binding, draft


def _source_drafts(source_id: str) -> tuple[str, dict, tuple[SpanDraft, ...]]:
    case = CASES / source_id
    raw = (case / "source.txt").read_text(encoding="utf-8")
    encoded = raw.encode()
    target = json.loads((case / "source-label.json").read_text(encoding="utf-8"))
    template = json.loads((case / "template.json").read_text(encoding="utf-8"))
    drafts = tuple(
        SpanDraft(
            draft_id=slot["slot_id"],
            logical_key=binding["logical_key"],
            render_mode=binding["render_mode"],
            value_kind=binding["value_kind"],
            group_kind=binding["group_kind"],
            group_key=binding["group_key"],
            target_paths=tuple(binding["target_paths"]),
            derivation=binding["derivation"],
            dependency_paths=tuple(binding["dependency_paths"]),
            dependency_bindings=tuple(binding["dependency_bindings"]),
            char_start=len(encoded[: slot["byte_start"]].decode()),
            char_end=len(encoded[: slot["byte_end"]].decode()),
            source_text=slot["source_text"],
            evidence_origin=slot["evidence_origin"],
            render_policy=slot["render_policy"],
            rationale=binding["rationale"],
        )
        for binding in template["bindings"]
        for slot in binding["occurrences"]
    )
    return raw, target, drafts


@pytest.mark.parametrize(
    ("name", "expected"),
    (("boxes", "4540 BOXES"), ("bags", "BAGS: 3000"), ("pallet", "PALLET: 28")),
)
def test_compiler_owns_source_proven_aggregate_noun(name: str, expected: str) -> None:
    raw, source_target, _binding, draft = _case(name)
    normalized = package_count_surfaces.normalize_sum_package_categories(
        raw=raw, drafts=(draft,), source_target=source_target
    )
    assert len(normalized) == 1
    assert normalized[0].source_text == expected
    assert (
        sum(path.endswith(".typeCategory") for path in normalized[0].dependency_paths)
        == {"boxes": 9, "bags": 3, "pallet": 1}[name]
    )
    validate_draft_source_alignment(raw=raw, drafts=normalized)


def test_render_mixed_bags_as_generic_packages() -> None:
    raw, source_target, _binding, draft = _case("bags")
    normalized = package_count_surfaces.normalize_sum_package_categories(
        raw=raw, drafts=(draft,), source_target=source_target
    )[0]
    target = copy.deepcopy(source_target)
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PACKAGE"
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 721
    target["documentPatch"]["cargoPackages"][1]["quantity"] = 721
    target["documentPatch"]["cargoPackages"][2]["quantity"] = 721
    slot = SimpleNamespace(
        slot_id="aggregate",
        source_text=normalized.source_text,
        byte_start=len(raw[: normalized.char_start].encode()),
        byte_end=len(raw[: normalized.char_end].encode()),
    )
    binding = SimpleNamespace(
        dependency_paths=normalized.dependency_paths,
        dependency_bindings=(),
        occurrences=(slot,),
    )
    rendered = package_count_surfaces.render_sum_package_nouns(
        binding=binding,
        template=None,
        source=raw.encode(),
        source_target=source_target,
        target=target,
        replacements={"aggregate": "BAGS: 2163"},
        target_quantity=2163,
    )
    assert rendered == {"aggregate": "PACKAGES: 2163"}


def test_numeric_sum_ignores_added_type_dependencies() -> None:
    raw, source_target, _binding, draft = _case("bags")
    normalized = package_count_surfaces.normalize_sum_package_categories(
        raw=raw, drafts=(draft,), source_target=source_target
    )[0]
    target = copy.deepcopy(source_target)
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PACKAGE"
    for package in target["documentPatch"]["cargoPackages"]:
        package["quantity"] = 721
    binding = SimpleNamespace(
        logical_key=normalized.logical_key,
        derivation="sum_package_quantity",
        dependency_paths=normalized.dependency_paths,
        dependency_bindings=(),
        occurrences=(SimpleNamespace(slot_id="aggregate", source_text=normalized.source_text),),
    )
    source_sum, target_sum = _derivation_numeric_values(
        binding=binding,
        source_target=source_target,
        target=target,
        bindings={},
        outputs={},
    )
    assert (source_sum, target_sum) == (3000, 2163)
    output = _render_proven_numeric_derivation(
        binding, source_value=source_sum, target_value=target_sum
    )
    assert output.replacements["aggregate"] == "BAGS: 2163"


def test_unowned_stale_noun_fails_closed_on_old_catalog() -> None:
    raw, source_target, binding, draft = _case("pallet")
    target = copy.deepcopy(source_target)
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_CASE"
    slot = SimpleNamespace(
        slot_id="aggregate",
        source_text=draft.source_text,
        byte_start=len(raw[: draft.char_start].encode()),
        byte_end=len(raw[: draft.char_end].encode()),
    )
    old_binding = SimpleNamespace(
        dependency_paths=binding["dependency_paths"],
        dependency_bindings=(),
        occurrences=(slot,),
    )
    with pytest.raises(ValueError, match="outside its derived slot"):
        package_count_surfaces.render_sum_package_nouns(
            binding=old_binding,
            template=None,
            source=raw.encode(),
            source_target=source_target,
            target=target,
            replacements={"aggregate": "26"},
            target_quantity=26,
        )


def test_separately_owned_type_noun_does_not_compete_with_sum() -> None:
    source_id = "doc_01d86535b4e4e74dfe15a62cfb50d6f1c4389dd741cb37ac8ee9c8064b705228"
    case = CASES / source_id
    raw = (case / "source.txt").read_text(encoding="utf-8")
    source_target = json.loads((case / "source-label.json").read_text(encoding="utf-8"))
    bindings = json.loads((case / "template.json").read_text(encoding="utf-8"))["bindings"]
    sum_binding = next(b for b in bindings if b["binding_id"] == "binding_0043")
    category_binding = next(b for b in bindings if b["binding_id"] == "binding_0031")
    number_slot = sum_binding["occurrences"][0]
    category_slot = next(
        slot for slot in category_binding["occurrences"] if slot["byte_start"] == 1553
    )
    numeric = SpanDraft(
        draft_id="number",
        logical_key=sum_binding["logical_key"],
        render_mode=sum_binding["render_mode"],
        value_kind=sum_binding["value_kind"],
        group_kind=sum_binding["group_kind"],
        group_key=sum_binding["group_key"],
        target_paths=(),
        derivation="sum_package_quantity",
        dependency_paths=tuple(sum_binding["dependency_paths"]),
        dependency_bindings=(),
        char_start=len(raw.encode()[: number_slot["byte_start"]].decode()),
        char_end=len(raw.encode()[: number_slot["byte_end"]].decode()),
        source_text=number_slot["source_text"],
        evidence_origin=number_slot["evidence_origin"],
        render_policy=number_slot["render_policy"],
        rationale="source-proven test",
    )
    category = replace(
        numeric,
        draft_id="category",
        logical_key=category_binding["logical_key"],
        render_mode=category_binding["render_mode"],
        value_kind=category_binding["value_kind"],
        group_kind=category_binding["group_kind"],
        group_key=category_binding["group_key"],
        target_paths=tuple(category_binding["target_paths"]),
        derivation=None,
        dependency_paths=(),
        char_start=len(raw.encode()[: category_slot["byte_start"]].decode()),
        char_end=len(raw.encode()[: category_slot["byte_end"]].decode()),
        source_text=category_slot["source_text"],
        evidence_origin=category_slot["evidence_origin"],
        render_policy=category_slot["render_policy"],
    )
    normalized = package_count_surfaces.normalize_sum_package_categories(
        raw=raw, drafts=(numeric, category), source_target=source_target
    )
    assert normalized == (numeric, category)
    validate_draft_source_alignment(raw=raw, drafts=normalized)


@pytest.mark.parametrize(("source_id", "absorbed_count"), OWNER_TRANSFER_CASES)
def test_source_proven_auxiliary_noun_transfers_whole_binding(
    source_id: str, absorbed_count: int
) -> None:
    raw, source_target, drafts = _source_drafts(source_id)
    normalized = package_count_surfaces.normalize_sum_package_categories(
        raw=raw, drafts=drafts, source_target=source_target
    )
    validate_draft_source_alignment(raw=raw, drafts=normalized)
    assert len(drafts) - len(normalized) == absorbed_count
    remaining_ids = {draft.draft_id for draft in normalized}
    removed = [draft for draft in drafts if draft.draft_id not in remaining_ids]
    assert len(removed) == absorbed_count
    assert all(
        draft.render_mode in {"deterministic_auxiliary", "literal_static"}
        and not draft.target_paths
        and not draft.dependency_paths
        for draft in removed
    )
    removed_keys = {removed_draft.logical_key for removed_draft in removed}
    assert all(set(draft.dependency_bindings).isdisjoint(removed_keys) for draft in normalized)


def test_partial_auxiliary_noun_transfer_fails_closed() -> None:
    source_id, _ = OWNER_TRANSFER_CASES[0]
    raw, source_target, drafts = _source_drafts(source_id)
    partial = tuple(
        draft
        for draft in drafts
        if not (draft.derivation == "number_to_words" and draft.source_text == "TEN")
    )
    with pytest.raises(ValueError, match="only partly transferred"):
        package_count_surfaces.normalize_sum_package_categories(
            raw=raw, drafts=partial, source_target=source_target
        )


def test_parenthesized_plural_aggregate_changes_with_package_category() -> None:
    source_id, _ = OWNER_TRANSFER_CASES[2]
    raw, source_target, drafts = _source_drafts(source_id)
    normalized = package_count_surfaces.normalize_sum_package_categories(
        raw=raw, drafts=drafts, source_target=source_target
    )
    count = next(draft for draft in normalized if draft.source_text == "13 BOX(ES)")
    target = copy.deepcopy(source_target)
    for package in target["documentPatch"]["cargoPackages"][:3]:
        package["typeCategory"] = "PACKAGE_CARTON"
    slot = SimpleNamespace(
        slot_id="aggregate",
        source_text=count.source_text,
        byte_start=len(raw[: count.char_start].encode()),
        byte_end=len(raw[: count.char_end].encode()),
    )
    binding = SimpleNamespace(
        logical_key=count.logical_key,
        dependency_paths=count.dependency_paths,
        dependency_bindings=(),
        occurrences=(slot,),
    )
    rendered = package_count_surfaces.render_sum_package_nouns(
        binding=binding,
        template=None,
        source=raw.encode(),
        source_target=source_target,
        target=target,
        replacements={"aggregate": "13 BOX(ES)"},
        target_quantity=13,
    )
    assert rendered == {"aggregate": "13 CARTON(S)"}
