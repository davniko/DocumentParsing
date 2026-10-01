from __future__ import annotations

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.raw_text_template import (
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
)
from document_ocr.synthesis.template_compiler import source_only_lot_ids
from document_ocr.synthesis.template_compiler.descendant import _direct_auxiliary_route
from document_ocr.synthesis.template_compiler.host import SpanDraft, binding_realization
from document_ocr.synthesis.template_compiler.models import SemanticBinding

SOURCE_LINE = "LOT NO. 12032, 12033, 12034"
SOURCE_PREFIX = (
    "Description of Packages and Goods\n"
    "3 X 20' DC CONTAINERS\n"
    "60 PACKAGES\n"
    "60,000KG FERRO MOLYBDENUM\n"
    "NET WEIGHT: 60,000KG\n"
    "GROSS WEIGHT: 60,960KG\n"
    "60 PALLETS ( 60 BAGS )\n"
    "HS CODE: 7202.7000\n"
)
SOURCE = (
    SOURCE_PREFIX + SOURCE_LINE + "\nTERMS OF DELIVERY ACCORDING TO INCOTERMS 2020\n"
).encode()
TARGET = {"documentPatch": {"cargoGroups": [{"description": "FERRO MOLYBDENUM"}]}}


def _binding(
    source_line: str = SOURCE_LINE,
    *,
    group_key: str = "cargo:0",
    target_paths: tuple[str, ...] = (),
) -> SemanticBinding:
    start = len(SOURCE_PREFIX.encode())
    slot = build_template_slot(
        slot_id="slot_0001",
        byte_start=start,
        byte_end=start + len(source_line),
        source_text=source_line,
        target_paths=target_paths,
        semantic_role=group_key,
        evidence_origin="audited_source_auxiliary",
        render_policy="natural_text",
    )
    draft = SpanDraft(
        draft_id="aux_lot_0001",
        logical_key="aux:lot_identifier_list:cargo:0",
        render_mode="deterministic_auxiliary",
        value_kind="lot_identifier_list",
        group_kind="cargo",
        group_key=group_key,
        target_paths=target_paths,
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(source_line),
        source_text=source_line,
        evidence_origin="audited_source_auxiliary",
        render_policy="natural_text",
        rationale="The audited goods block prints a source-only three-ID LOT list.",
    )
    return SemanticBinding.model_validate(
        {
            "binding_id": "binding_0001",
            "logical_key": draft.logical_key,
            "render_mode": draft.render_mode,
            "value_kind": draft.value_kind,
            "group_kind": draft.group_kind,
            "group_key": draft.group_key,
            "target_paths": draft.target_paths,
            "target_relationship": "none",
            "derivation": None,
            "dependency_paths": (),
            "dependency_bindings": (),
            "occurrences": (slot,),
            "realization": binding_realization(draft=draft, slots=(slot,), source_target=TARGET),
            "source_relationships": (),
            "rationale": draft.rationale,
        }
    )


def test_source_only_lot_list_uses_deterministic_renderer_and_preserves_literal_bytes() -> None:
    binding = _binding()
    assert _direct_auxiliary_route(binding)[0]
    stream = DeterministicStream(151, "lot-test", "sample-one")
    rendered = source_only_lot_ids.render_binding(
        binding, stream=stream, source=SOURCE, target=TARGET
    )
    assert rendered == source_only_lot_ids.render_binding(
        binding, stream=stream, source=SOURCE, target=TARGET
    )
    assert rendered != SOURCE_LINE
    source_only_lot_ids.parse_source_line(rendered)
    assert rendered != source_only_lot_ids.render_binding(
        binding,
        stream=DeterministicStream(151, "lot-test", "sample-two"),
        source=SOURCE,
        target=TARGET,
    )
    template = compile_raw_text_template(
        document_id="sample-one", source=SOURCE, slots=binding.occurrences
    )
    output, proof = render_compiled_template(
        source=SOURCE, template=template, bindings={"slot_0001": rendered}
    )
    assert output == SOURCE.replace(SOURCE_LINE.encode(), rendered.encode())
    assert proof.exact_literal_regions and proof.format_envelopes_valid


@pytest.mark.parametrize(
    "source_line",
    [
        "LOT NO. 12032, 12034, 12035",
        "LOT NO. 12032, 12033",
        "LOT NO. 12032 / 12033 / 12034",
        "LOT NO. 12032, 12033, 120345",
        "LOT NO. 12032, 12033, 12034  ",
    ],
)
def test_uncertified_lot_surface_fails_closed(source_line: str) -> None:
    binding = _binding(source_line)
    assert not _direct_auxiliary_route(binding)[0]
    with pytest.raises(ValueError):
        source_only_lot_ids.render_binding(
            binding,
            stream=DeterministicStream(151, "lot-test", "sample-one"),
            source=SOURCE,
            target=TARGET,
        )


def test_lot_owner_and_target_contamination_are_rejected() -> None:
    binding = _binding(group_key="cargo:1")
    with pytest.raises(ValueError, match="owner is absent"):
        source_only_lot_ids.render_binding(
            binding,
            stream=DeterministicStream(151, "lot-test", "sample-one"),
            source=SOURCE,
            target=TARGET,
        )
    contaminated = {
        "documentPatch": {
            "cargoGroups": [
                {"description": "FERRO MOLYBDENUM", "additionalInformation": [SOURCE_LINE]}
            ]
        }
    }
    with pytest.raises(ValueError, match="duplicated"):
        source_only_lot_ids.validate_binding(_binding(), source=SOURCE, target=contaminated)


def test_lot_source_span_and_target_backing_are_rejected() -> None:
    binding = _binding()
    with pytest.raises(ValueError, match="pinned source"):
        source_only_lot_ids.validate_binding(binding, source=SOURCE.replace(b"12032", b"99999"))
    bad_binding = binding.model_copy(
        update={"target_paths": ("documentPatch.cargoGroups[0].additionalInformation[0]",)}
    )
    with pytest.raises(ValueError, match="source-only goods contract"):
        source_only_lot_ids.validate_binding(bad_binding)


def test_lot_goods_context_requires_source_proven_section_and_unique_line() -> None:
    alternative = SOURCE.decode().replace(
        "Description of Packages and Goods",
        "PARTICULARS FURNISHED BY SHIPPER\nMarks and Numbers\nNo.of Containers or Other Pkgs",
    )
    source_only_lot_ids.validate_source_context(alternative.encode(), source_text=SOURCE_LINE)
    for invalid in (
        SOURCE.replace(b"Description of Packages and Goods\n", b""),
        SOURCE + SOURCE_LINE.encode() + b"\n",
        SOURCE.replace(b"HS CODE: 7202.7000\n", b""),
    ):
        with pytest.raises(ValueError):
            source_only_lot_ids.validate_source_context(invalid, source_text=SOURCE_LINE)
