"""Printed package sums must be true at compilation, not only at rendering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.package_sum_certificate import certify_source


def _binding(*surfaces: str) -> SimpleNamespace:
    return SimpleNamespace(
        logical_key="sum:package_quantities",
        derivation="sum_package_quantity",
        dependency_paths=(
            "documentPatch.cargoPackages[0].quantity",
            "documentPatch.cargoPackages[1].quantity",
        ),
        dependency_bindings=(),
        occurrences=tuple(
            SimpleNamespace(slot_id=f"slot_{index:04d}", source_text=surface)
            for index, surface in enumerate(surfaces)
        ),
    )


_TARGET = {
    "documentPatch": {
        "cargoPackages": [
            {"quantity": 667, "typeCategory": "PACKAGE_CARTON"},
            {"quantity": 640, "typeCategory": "PACKAGE_CARTON"},
        ]
    }
}


def test_compiler_accepts_printed_total_with_complete_components() -> None:
    certify_source(bindings=[_binding("1307", "1,307")], source_target=_TARGET)


@pytest.mark.parametrize("surface", ["667", "640"])
def test_compiler_rejects_component_mislabeled_as_aggregate(surface: str) -> None:
    with pytest.raises(ValueError, match="printed package sum fails source arithmetic"):
        certify_source(bindings=[_binding(surface)], source_target=_TARGET)
