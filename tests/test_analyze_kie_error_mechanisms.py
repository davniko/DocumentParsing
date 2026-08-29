from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_tool() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "analyze_kie_error_mechanisms.py"
    spec = importlib.util.spec_from_file_location("analyze_kie_error_mechanisms", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "error_type": "substitution",
        "reference_path": "$.documentPatch.parties.shipper.address",
        "predicted_path": "$.documentPatch.parties.shipper.address",
        "reference_value": '"ALPHA STREET, CAIRO"',
        "predicted_value": '"ALPHA STREET"',
        "value_similarity": 0.75,
        "predicted_ocr_grounded": True,
    }
    row.update(changes)
    return row


def test_classifies_boundary_copy_and_scale_errors() -> None:
    mechanism, _, _ = TOOL.classify_leaf_error(_row())
    assert mechanism == "incomplete_or_shortened_value"

    mechanism, _, _ = TOOL.classify_leaf_error(
        _row(
            reference_value="500.0",
            predicted_value="500000.0",
            reference_path="$.documentPatch.cargoGroups[0].grossWeight.value",
            predicted_path="$.documentPatch.cargoGroups[0].grossWeight.value",
        ),
    )
    assert mechanism == "numeric_scale_error"


def test_exact_scalar_under_another_field_is_wrong_field_assignment() -> None:
    row = _row(
        error_type="addition",
        reference_path="",
        predicted_path="$.documentPatch.parties.carrier.name",
        reference_value="",
        predicted_value='"ACME"',
    )
    mechanism, _, paths = TOOL.classify_leaf_error(
        row,
        cross_field_paths=("$.documentPatch.parties.shipper.name",),
    )
    assert mechanism == "wrong_field_assignment"
    assert paths == ("$.documentPatch.parties.shipper.name",)


def test_unsupported_is_not_named_hallucination() -> None:
    mechanism, basis, _ = TOOL.classify_leaf_error(
        _row(
            reference_value='"ALPHA"',
            predicted_value='"UNRELATED"',
            value_similarity=0.1,
            predicted_ocr_grounded=False,
        ),
    )
    assert mechanism == "unsupported_text_candidate"
    assert "OCR substring" in basis
    assert "hallucin" not in mechanism


def test_synthetic_relation_fields_are_not_grounding_failures() -> None:
    mechanism, _, _ = TOOL.classify_leaf_error(
        _row(
            error_type="addition",
            reference_path="",
            predicted_path="$.documentPatch.cargoAllocationGroups[0].packageIds[0]",
            reference_value="",
            predicted_value='"p1"',
            predicted_ocr_grounded=False,
        ),
    )
    assert mechanism == "relation_or_structure_error"


def test_cross_field_pair_requires_unique_addition_and_omission() -> None:
    rows = [
        {
            "document_id": "doc",
            "error_type": "addition",
            "predicted_value": '"ACME"',
            "predicted_path": "$.documentPatch.parties.carrier.name",
            "reference_value": "",
            "reference_path": "",
        },
        {
            "document_id": "doc",
            "error_type": "omission",
            "reference_value": '"ACME"',
            "reference_path": "$.documentPatch.parties.shipper.name",
            "predicted_value": "",
            "predicted_path": "",
        },
    ]
    assert TOOL._cross_field_pairs(rows) == {
        0: ("$.documentPatch.parties.shipper.name",),
        1: ("$.documentPatch.parties.carrier.name",),
    }

    rows.append(dict(rows[1], reference_path="$.documentPatch.parties.consignee.name"))
    assert TOOL._cross_field_pairs(rows) == {}


def test_output_contract_counts_do_not_double_count_invalid_json() -> None:
    rows = [
        {"issue_type": "invalid_json"},
        {"issue_type": "extra_forbidden"},
        {"issue_type": "value_error"},
    ]
    predictions = [{"json_valid": False}, {"json_valid": True}]

    assert TOOL._output_failure_rows(rows, predictions) == [
        {"failure_type": "invalid_json", "occurrences": 1},
        {"failure_type": "wrong_or_extra_key", "occurrences": 1},
        {"failure_type": "missing_required_key", "occurrences": 0},
        {"failure_type": "invalid_value_or_relationship", "occurrences": 1},
    ]
