from __future__ import annotations

import json
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.config import load_synthesis_deterministic_smoke_config
from document_ocr.synthesis.drafts import validate_change_ledger
from document_ocr.synthesis.generation_models import SemanticChange
from document_ocr.synthesis.policies import validate_policy_registry
from document_ocr.synthesis.rendering import (
    apply_patch_plan,
    build_patch_plan,
    render_date_surface,
    render_number_surface,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl"
)
CONFIG = PROJECT_ROOT / "configs/synthesis/mpci_bl_combined1157_deterministic_smoke20.yaml"


def _change(path: str, family: str, old: object, new: object) -> SemanticChange:
    role = path.replace("[0]", "[]")
    return SemanticChange.model_validate(
        {
            "target_path": path,
            "role_path": role,
            "family": family,
            "old_value": old,
            "new_value": new,
            "method": "test",
            "coupling_group": "test",
        },
        strict=True,
    )


def test_deterministic_smoke_config_pins_twenty_non_publishable_drafts() -> None:
    config = load_synthesis_deterministic_smoke_config(CONFIG)

    assert config.selection.requested_documents == 20
    assert sum(config.selection.family_exact.values()) == 20
    assert config.selection.minimum_carriers == 8
    assert config.generation.publish_training_records is False


def test_field_policy_is_exhaustive_over_real_1157_target_contract() -> None:
    targets = tuple(json.loads(line)["target"] for line in SOURCE.read_text().splitlines())

    audit = validate_policy_registry(targets)

    assert audit == {
        "schema_leaf_families": 131,
        "observed_leaf_families": 114,
        "zero_support_leaf_families": 17,
        "policy_derive": 62,
        "policy_forbidden": 6,
        "policy_legitimately_absent": 11,
        "policy_pending_linguistic": 10,
        "policy_preserve_nonidentifying": 8,
        "policy_regenerate": 15,
        "policy_resample": 19,
    }


def test_surface_renderers_preserve_date_and_decimal_format() -> None:
    assert render_date_surface("01/02/2024", "2024-02-01", "2025-12-03") == "03/12/2025"
    assert render_date_surface("FEB 1, 2024", "2024-02-01", "2025-12-03") == "DEC 3, 2025"
    assert render_number_surface("1.234,50 KG", 1234.5, 2345.75) == "2.345,75 KG"
    assert render_number_surface("1,234 PCS", 1234, 2345) == "2,345 PCS"


def test_patch_plan_changes_only_audited_page_spans() -> None:
    raw = "--- PAGE 1 ---\nMSCU6639870\nKEEP\n\n--- PAGE 2 ---\nMSCU6639870\nTAIL\n"
    source_hash = sha256_bytes(raw.encode())
    change = _change(
        "documentPatch.containers[0].containerNumber",
        "container_identifier",
        "MSCU6639870",
        "MSCU1234566",
    )
    anchors = [
        {
            "relation_target_path": change.target_path,
            "patchable": True,
            "page_number": 1,
            "page_start": 0,
            "page_end": 11,
            "raw_value": "MSCU6639870",
        },
        {
            "relation_target_path": change.target_path,
            "patchable": True,
            "page_number": 2,
            "page_start": 0,
            "page_end": 11,
            "raw_value": "MSCU6639870",
        },
    ]

    plan = build_patch_plan(
        synthetic_document_id="syn_test",
        source_raw_text_sha256=source_hash,
        deterministic_target_sha256="1" * 64,
        changes=(change,),
        anchors=anchors,
    )
    rendered, changed = apply_patch_plan(raw, plan)

    assert changed == 22
    assert rendered == raw.replace("MSCU6639870", "MSCU1234566")
    assert rendered.replace("MSCU1234566", "MSCU6639870") == raw
    assert plan.unrendered_change_paths == ()


def test_patch_plan_fails_closed_when_any_evidence_span_is_ambiguous() -> None:
    change = _change(
        "documentPatch.containers[0].sealNumbers[0]",
        "seal_identifier",
        "ABC123",
        "XYZ987",
    )
    plan = build_patch_plan(
        synthetic_document_id="syn_test",
        source_raw_text_sha256="0" * 64,
        deterministic_target_sha256="1" * 64,
        changes=(change,),
        anchors=(
            {
                "relation_target_path": change.target_path,
                "patchable": False,
                "page_number": 1,
                "page_start": None,
                "page_end": None,
                "raw_value": "ABC123",
            },
        ),
    )

    assert plan.edits == ()
    assert plan.unrendered_change_paths == (change.target_path,)


def test_change_ledger_must_equal_the_complete_target_diff() -> None:
    source = {"documentPatch": {"issueDate": "2024-01-01", "negotiability": "negotiable"}}
    target = {"documentPatch": {"issueDate": "2025-01-01", "negotiability": "negotiable"}}
    change = _change(
        "documentPatch.issueDate",
        "document_date",
        "2024-01-01",
        "2025-01-01",
    )

    validate_change_ledger(source, target, (change,))

    changed_again = {
        "documentPatch": {"issueDate": "2025-01-01", "negotiability": "non_negotiable"}
    }
    with pytest.raises(ValueError, match="ledger mismatch"):
        validate_change_ledger(source, changed_again, (change,))
