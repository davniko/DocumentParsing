from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from document_ocr.training.metrics import PredictionAssessment


def _load_tool() -> ModuleType:
    tools = Path(__file__).parents[1] / "tools"
    sys.path.insert(0, str(tools))
    path = tools / "analyze_kie_training_program.py"
    spec = importlib.util.spec_from_file_location("analyze_kie_training_program", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TOOL = _load_tool()


def _diagnostic(
    document_id: str,
    *,
    true_positive: int,
    predicted: int,
    reference: int,
    json_valid: bool,
    schema_valid: bool,
) -> object:
    assessment = PredictionAssessment(
        generated_text="{}",
        reference_text="{}",
        json_valid=json_valid,
        schema_valid=schema_valid,
        canonical_exact_match=False,
        predicted_field_values=frozenset(),
        reference_field_values=frozenset(),
        predicted_cargo_relation_facts=frozenset(),
        reference_cargo_relation_facts=frozenset(),
        predicted_category_values=frozenset(),
        reference_category_values=frozenset(),
    )
    record = SimpleNamespace(document_id=document_id)
    return _TOOL.Diagnostic(
        record=record,
        assessment=assessment,
        true_positive=true_positive,
        predicted=predicted,
        reference=reference,
        compared_paths=reference,
        index_true_positive=true_positive,
        schema_failure_class="none" if schema_valid else "schema_validation_error",
        schema_failure_detail="",
        generated_characters=2,
        reference_characters=2,
    )


def test_numeric_conversion_preserves_numeric_looking_group_labels() -> None:
    result = _TOOL._as_numbers([{"group": "1", "group_type": "cargo_groups", "f1": "0.85"}])

    assert result == [{"group": "1", "group_type": "cargo_groups", "f1": 0.85}]


def test_quantile_supports_non_default_bootstrap_sizes() -> None:
    assert _TOOL._quantile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    assert _TOOL._quantile([0.0, 10.0], 0.975) == pytest.approx(9.75)


def test_validity_subsets_reconcile_pooled_counts() -> None:
    diagnostics = {
        "schema": _diagnostic(
            "schema", true_positive=8, predicted=9, reference=10, json_valid=True, schema_valid=True
        ),
        "schema-invalid": _diagnostic(
            "schema-invalid",
            true_positive=3,
            predicted=5,
            reference=6,
            json_valid=True,
            schema_valid=False,
        ),
        "json-invalid": _diagnostic(
            "json-invalid",
            true_positive=0,
            predicted=0,
            reference=4,
            json_valid=False,
            schema_valid=False,
        ),
    }

    rows = _TOOL._validity_subset_rows(diagnostics)
    by_name = {row["validity_subset"]: row for row in rows}

    assert by_name["schema valid"]["f1"] == pytest.approx(16 / 19)
    assert by_name["JSON valid, schema invalid"]["recall"] == pytest.approx(0.5)
    assert sum(row["reference"] for row in rows) == 20
    assert sum(row["predicted"] for row in rows) == 14
    assert sum(row["true_positive"] for row in rows) == 11


def test_error_family_aggregation_rejects_unclassified_mechanisms() -> None:
    with pytest.raises(ValueError, match="unmapped error mechanism"):
        _TOOL._actionable_error_families(
            [{"mechanism": "new_unclassified_failure", "leaf_errors": 1}]
        )
