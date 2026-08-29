from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "analyze_kie_scaling_evidence.py"
    spec = importlib.util.spec_from_file_location("analyze_kie_scaling_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def test_normalize_path_erases_only_numeric_list_indices() -> None:
    assert (
        TOOL.normalize_path("$.documentPatch.cargoGroups[12].marksAndNumbers[0]")
        == "$.documentPatch.cargoGroups[].marksAndNumbers[]"
    )
    assert TOOL.normalize_path("$.documentPatch.value[x]") == "$.documentPatch.value[x]"


def test_prf_uses_exact_micro_counts() -> None:
    precision, recall, f1 = TOOL._prf((3, 4, 5))
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.6)
    assert f1 == pytest.approx(2 / 3)


def test_flatten_preserves_list_positions_before_normalization() -> None:
    value = {"a": [{"b": 1}, {"b": 2}]}
    assert TOOL._flatten(value) == {"$.a[0].b": 1, "$.a[1].b": 2}
    assert TOOL._normalized_leaf_paths(value) == [
        "$.documentPatch.a[].b",
        "$.documentPatch.a[].b",
    ]
