from __future__ import annotations

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.raw_text_template import (
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
)
from document_ocr.synthesis.template_compiler import source_only_ped_ids
from document_ocr.synthesis.template_compiler.descendant import _direct_auxiliary_route
from document_ocr.synthesis.template_compiler.host import SpanDraft, binding_realization
from document_ocr.synthesis.template_compiler.models import SemanticBinding

PREFIX = "FREIGHT PREPAID AT ABROAD BY SEARA ALIMENTOS LTDA - BRAND: SEARA - "
PED = "PED.\n100043.11, 100043.12, 100043.10"
SUFFIX = " - NCM:02071412 - MNBU4015403 SEAL: 020759/SIF3837\n"
SOURCE = (PREFIX + PED + SUFFIX).encode()
TARGET = {
    "documentPatch": {
        "cargoGroups": [{"description": "FROZEN CHICKEN WHOLE LEG BONE IN BRAND: SEARA"}]
    }
}


def _binding(
    *,
    source_text: str = PED,
    target_paths: tuple[str, ...] = (),
    evidence_origin: str = "audited_source_auxiliary",
) -> SemanticBinding:
    start = len(PREFIX.encode())
    slot = build_template_slot(
        slot_id="slot_0001",
        byte_start=start,
        byte_end=start + len(source_text.encode()),
        source_text=source_text,
        target_paths=target_paths,
        semantic_role="cargo:0",
        evidence_origin=evidence_origin,
        render_policy="natural_text",
    )
    draft = SpanDraft(
        draft_id="aux_ped_0001",
        logical_key="aux:ped_identifier_list:cargo:0",
        render_mode="deterministic_auxiliary",
        value_kind="ped_identifier_list",
        group_kind="cargo",
        group_key="cargo:0",
        target_paths=target_paths,
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(source_text),
        source_text=source_text,
        evidence_origin=evidence_origin,
        render_policy="natural_text",
        rationale="Source-owned consignment PED list; no package/container assignment.",
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


def test_source_only_ped_list_round_trips_and_changes_together() -> None:
    binding = _binding()
    assert _direct_auxiliary_route(binding)[0]
    stream = DeterministicStream(59, "ped-test", "sample-one")
    rendered = source_only_ped_ids.render_binding(
        binding, stream=stream, source=SOURCE, target=TARGET
    )
    assert rendered != PED
    assert source_only_ped_ids.parse_source_surface(rendered)
    assert rendered == source_only_ped_ids.render_binding(
        binding, stream=stream, source=SOURCE, target=TARGET
    )
    template = compile_raw_text_template(
        document_id="sample-one", source=SOURCE, slots=binding.occurrences
    )
    output, proof = render_compiled_template(
        source=SOURCE, template=template, bindings={"slot_0001": rendered}
    )
    assert output == SOURCE.replace(PED.encode(), rendered.encode())
    assert proof.exact_literal_regions and proof.format_envelopes_valid


@pytest.mark.parametrize(
    "source_text",
    [
        "PED.\n100043.11, 100044.12, 100043.10",
        "PED.\n100043.11, 100043.13, 100043.10",
        "PED.\n100043.11, 100043.12",
        "PED. 100043.11, 100043.12, 100043.10",
    ],
)
def test_ped_surface_must_prove_one_shared_triplet(source_text: str) -> None:
    binding = _binding(source_text=source_text)
    assert not _direct_auxiliary_route(binding)[0]


def test_ped_source_context_and_label_separation_fail_closed() -> None:
    binding = _binding()
    for invalid in (
        SOURCE.replace(b"BRAND: SEARA", b"BRAND: OTHER"),
        SOURCE.replace(b" - NCM:", b" - OTHER:"),
        SOURCE + b"PED.\n",
    ):
        with pytest.raises(ValueError):
            source_only_ped_ids.validate_binding(binding, source=invalid, target=TARGET)
    bad_binding = binding.model_copy(
        update={"target_paths": ("documentPatch.cargoGroups[0].additionalInformation[0]",)}
    )
    with pytest.raises(ValueError, match="source-only"):
        source_only_ped_ids.validate_binding(bad_binding)
    contaminated = {
        "documentPatch": {"cargoGroups": [{"description": "FROZEN CHICKEN PED. 100043.11"}]}
    }
    with pytest.raises(ValueError, match="remains"):
        source_only_ped_ids.validate_binding(binding, source=SOURCE, target=contaminated)
