from __future__ import annotations

from pathlib import Path

import pytest

from document_ocr.synthesis.equipment_registry import (
    EquipmentRegistryError,
    canonical_equipment_receipt_json,
    load_bic_equipment_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    PROJECT_ROOT
    / "data/registries/equipment/bic-iso6346-2022-web-snapshot-20260831/source-manifest.json"
)


def test_bic_snapshot_compiles_assigned_size_and_type_semantics() -> None:
    registry = load_bic_equipment_registry(MANIFEST)

    assert len(registry.length_codes) >= 10
    assert len(registry.height_width_codes) >= 10
    assert len(registry.type_codes) >= 60
    assert registry.excluded_unassigned_rows > 0
    assert registry.excluded_invalid_source_codes == ("VEJ",)
    assert registry.classify("45G1").type.thermal_capability == "none"
    assert registry.classify("40RA").type.supports_temperature_setpoint is True
    assert registry.classify("40HR").type.supports_temperature_setpoint is True
    heated = registry.classify("40RH").type
    assert heated.thermal_capability == "heated"
    assert heated.supports_temperature_setpoint is True
    assert canonical_equipment_receipt_json(registry).endswith(b"\n")


@pytest.mark.parametrize("surface", ["40HC", "20DV", "HC40", "40' HC", "40G4"])
def test_equipment_aliases_and_unassigned_codes_fail_closed(surface: str) -> None:
    registry = load_bic_equipment_registry(MANIFEST)

    with pytest.raises(EquipmentRegistryError):
        registry.classify(surface)


def test_equipment_source_hash_drift_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bic"
    root.mkdir()
    manifest = MANIFEST.read_text()
    for name in ("size-type-code.html", "type-code-designation.html"):
        (root / name).write_bytes((MANIFEST.parent / name).read_bytes())
    (root / "type-code-designation.html").write_bytes(b"changed")
    (root / "source-manifest.json").write_text(manifest)

    with pytest.raises(EquipmentRegistryError, match="receipt mismatch"):
        load_bic_equipment_registry(root / "source-manifest.json")
