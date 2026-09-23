from __future__ import annotations

import json
from pathlib import Path

import pytest

from document_ocr.synthesis.template_compiler.coherence import validate_render_coherence
from document_ocr.synthesis.template_compiler.descendant import (
    _active_identifier_relationships,
    _equipment_semantics_match,
    _render_agent_target_binding,
    _resolve_path,
)
from document_ocr.synthesis.template_compiler.latest_target import (
    LatestTargetConstructionError,
    latest_target_from_source,
)
from document_ocr.synthesis.template_compiler.models import CertifiedSemanticTemplate

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOG = (
    _PROJECT_ROOT
    / "artifacts/kie-synthesis-production/template-base/catalogs/"
    "mpci-bl-production-template-catalog1510-v5"
)
_PRINTED_TEMPERATURE_WITHOUT_EQUIPMENT_TYPE = {
    "doc_243706979d0c12ffcdcd311dc31d417d9baf5c738232797cfef21ec20ade8bab",
    "doc_5734315e62a9036a810859819fdd12fa3dedf3ea9a52c51d044d01336c373fed",
    "doc_7043674b2e2f27b44082a2e7404fa807c42fe478184ead07783b73e1065ee8c5",
}


def _source_target(document_id: str) -> dict[str, object]:
    value = json.loads((_CATALOG / "cases" / document_id / "source-label.json").read_bytes())
    assert isinstance(value, dict)
    return value


def test_latest_target_population_is_fail_closed_and_exact() -> None:
    incompatible: set[str] = set()
    compatible = 0
    for line in (_CATALOG / "catalog.jsonl").read_text(encoding="utf-8").splitlines():
        document_id = json.loads(line)["documentId"]
        try:
            target = latest_target_from_source(_source_target(document_id))
        except LatestTargetConstructionError:
            incompatible.add(document_id)
        else:
            compatible += 1
            assert target["schemaVersion"] == "5.0.0-experimental"

    assert compatible == 1_510
    assert not incompatible


@pytest.mark.parametrize("document_id", sorted(_PRINTED_TEMPERATURE_WITHOUT_EQUIPMENT_TYPE))
def test_latest_target_preserves_printed_temperature_without_inventing_equipment(
    document_id: str,
) -> None:
    source = _source_target(document_id)
    target = latest_target_from_source(source)
    before = source["documentPatch"]["containers"]
    after = target["documentPatch"]["containers"]
    assert len(before) == len(after)
    for old, new in zip(before, after, strict=True):
        assert new["temperatureSetpoint"] == old["temperatureSetpoint"]
        assert "typeCategory" not in new
        assert "sizeCategory" not in new
        assert "typeDescription" not in new


def test_compiled_dangerous_goods_paths_resolve_their_v5_locations() -> None:
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "cargoGroups": [
                {
                    "dangerousGoods": [
                        {
                            "subsidiaryHazardCategories": ["TOXIC_SUBSTANCES"],
                            "packingGroupCategory": "MEDIUM_DANGER",
                            "flashPoint": {
                                "temperature": {"value": 12.0, "unit": "celsius"}
                            },
                        }
                    ]
                }
            ]
        },
    }
    subsidiary_path = (
        "documentPatch.cargoGroups[0].dangerousGoods[0].subsidiaryHazardCategory"
    )
    packing_path = (
        "documentPatch.cargoGroups[0].dangerousGoods[0].flashPoint."
        "packingGroupCategory"
    )

    assert _resolve_path(target, subsidiary_path) == "TOXIC_SUBSTANCES"
    assert _resolve_path(target, packing_path) == "MEDIUM_DANGER"


def test_equipment_matcher_uses_the_same_reviewed_grammar_as_v5_enrichment() -> None:
    assert _equipment_semantics_match(
        {
            "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
            "typeCategory": "GENERAL_PURPOSE",
        },
        "20'CONT",
    )


def test_composite_range_coherence_validates_without_modifying_target() -> None:
    document_id = "doc_2520425e6bd3a4ea5b06ec6e83322468a17a78668d3e8900188e7d14f8d685a7"
    source = _source_target(document_id)
    target = latest_target_from_source(source)
    template = CertifiedSemanticTemplate.model_validate_json(
        (_CATALOG / "cases" / document_id / "template.json").read_bytes(), strict=True
    )

    before = json.dumps(target, sort_keys=True)
    validate_render_coherence(
        source_target=source,
        target=target,
        bindings=template.bindings,
        constraints=template.coherence_constraints,
        outputs=None,
    )

    assert json.dumps(target, sort_keys=True) == before


def test_incidental_short_identifier_overlap_is_not_a_runtime_relationship() -> None:
    document_id = "doc_13bff9170593de35f1f2e19292a9611c0f7630a7f67caf108a9b2b5b65bef6ef"
    template = CertifiedSemanticTemplate.model_validate_json(
        (_CATALOG / "cases" / document_id / "template.json").read_bytes(), strict=True
    )

    active = _active_identifier_relationships(template)

    assert active["agent:equipment:container_row_number:0"] == ()


def test_incomplete_equipment_surface_requires_compilation_ownership_repair() -> None:
    document_id = "doc_872132b13424286de434aadcac0a823d05005bc3355ca259f946e1b32f72b138"
    source = _source_target(document_id)
    target = latest_target_from_source(source)
    template = CertifiedSemanticTemplate.model_validate_json(
        (_CATALOG / "cases" / document_id / "template.json").read_bytes(), strict=True
    )
    binding = next(
        row
        for row in template.bindings
        if row.logical_key == "anchor:documentPatch.containers[0].typeDescription"
    )

    with pytest.raises(ValueError, match="partial equipment surface"):
        _render_agent_target_binding(binding, source_target=source, target=target)
