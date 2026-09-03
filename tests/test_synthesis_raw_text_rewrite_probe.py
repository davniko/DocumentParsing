from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.config import load_synthesis_raw_text_rewrite_probe_config
from document_ocr.synthesis.raw_text_rewrite_probe import (
    ExactTextEdit,
    RewriteCompletionReceipt,
    RewriteWorkspace,
    apply_edit_batch,
    changed_leaves,
    inspect_current_diff,
    validate_completion_receipt,
)


def _target_edit(
    *,
    old: str,
    new: str,
    left: str = "",
    right: str = "",
    path: str = "documentPatch.billOfLadingNumber",
) -> ExactTextEdit:
    return ExactTextEdit.model_validate(
        {
            "oldText": old,
            "newText": new,
            "leftContext": left,
            "rightContext": right,
            "expectedMatches": 1,
            "applyToAllMatches": False,
            "category": "target_field",
            "targetPaths": (path,),
            "auxiliaryKind": None,
        },
        strict=True,
    )


def _workspace(text: str, *paths: str) -> RewriteWorkspace:
    return RewriteWorkspace(
        document_id="doc_" + "a" * 64,
        original_text=text,
        current_text=text,
        required_paths=frozenset(paths or ("documentPatch.billOfLadingNumber",)),
    )


def test_exact_contextual_edit_preserves_page_and_newline_topology() -> None:
    source = "--- PAGE 1 ---\r\nB/L: ABC123\r\n--- PAGE 2 ---\r\nCOPY: ABC123\r\n"
    workspace = _workspace(source)

    receipts = apply_edit_batch(
        workspace,
        (_target_edit(old="ABC123", new="XYZ789", left="B/L: ", right="\r\n"),),
    )

    assert workspace.current_text == source.replace("B/L: ABC123", "B/L: XYZ789")
    assert receipts[0].matchedOccurrences == 1
    assert workspace.current_text.splitlines()[0] == "--- PAGE 1 ---"
    assert workspace.current_text.count("\r\n") == source.count("\r\n")


def test_ambiguous_or_wrong_match_count_is_rejected_without_mutation() -> None:
    source = "B/L ABC123\nCOPY ABC123\n"
    workspace = _workspace(source)
    edit = _target_edit(old="ABC123", new="XYZ789")

    with pytest.raises(ValueError, match="match count is 2, expected 1"):
        apply_edit_batch(workspace, (edit,))

    assert workspace.current_text == source
    assert workspace.applied_edits == []


def test_overlapping_batch_is_atomic() -> None:
    source = "--- PAGE 1 ---\nABCDEFG\n"
    workspace = _workspace(
        source,
        "documentPatch.billOfLadingNumber",
        "documentPatch.originalBillOfLadingNumber",
    )
    first = _target_edit(old="ABCDE", new="12345")
    second = _target_edit(
        old="CDEFG",
        new="67890",
        path="documentPatch.originalBillOfLadingNumber",
    )

    with pytest.raises(ValueError, match="overlapping"):
        apply_edit_batch(workspace, (first, second))

    assert workspace.current_text == source
    assert workspace.batch_count == 0


def test_newline_or_edge_whitespace_changes_are_rejected() -> None:
    workspace = _workspace("--- PAGE 1 ---\nNAME: OLD VALUE\n")
    newline_edit = _target_edit(old="OLD VALUE", new="NEW\nVALUE")
    with pytest.raises(ValueError, match="newline sequence"):
        apply_edit_batch(workspace, (newline_edit,))

    whitespace_edit = _target_edit(old="OLD VALUE", new=" NEW VALUE")
    with pytest.raises(ValueError, match="horizontal whitespace"):
        apply_edit_batch(workspace, (whitespace_edit,))


def test_auxiliary_personal_data_must_be_replaced_not_deleted() -> None:
    with pytest.raises(ValidationError, match="non-empty replacement"):
        ExactTextEdit.model_validate(
            {
                "oldText": "VAT123",
                "newText": "",
                "leftContext": "VAT: ",
                "rightContext": "\n",
                "expectedMatches": 1,
                "applyToAllMatches": False,
                "category": "auxiliary_personal_data",
                "targetPaths": (),
                "auxiliaryKind": "tax_identifier",
            },
            strict=True,
        )


def test_changed_leaf_inventory_separates_printed_and_nonprinted_metadata() -> None:
    source = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "billOfLadingNumber": "OLD",
            "cargoGroups": [{"groupId": "g1", "description": "OLD GOODS"}],
        },
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "billOfLadingNumber": "NEW",
            "cargoGroups": [{"groupId": "g2", "description": "NEW GOODS"}],
        },
    }

    rows = {row.path: row for row in changed_leaves(source, target)}

    assert rows["schemaVersion"].evidenceClass == "schema_metadata"
    assert rows["schemaVersion"].requiresTextEdit is False
    assert rows["documentPatch.cargoGroups[0].groupId"].evidenceClass == "relation_metadata"
    assert rows["documentPatch.cargoGroups[0].groupId"].requiresTextEdit is False
    assert rows["documentPatch.billOfLadingNumber"].requiresTextEdit is True
    assert rows["documentPatch.cargoGroups[0].description"].requiresTextEdit is True


def test_final_receipt_requires_complete_path_coverage_and_final_diff_inspection() -> None:
    path = "documentPatch.billOfLadingNumber"
    workspace = _workspace("--- PAGE 1 ---\nB/L: OLD\n", path)
    apply_edit_batch(
        workspace,
        (_target_edit(old="OLD", new="NEW", left="B/L: ", right="\n", path=path),),
    )
    receipt = RewriteCompletionReceipt.model_validate(
        {
            "status": "complete",
            "currentTextSha256": workspace.current_sha256,
            "unresolvedChangedPaths": (),
            "summary": "All target facts were rewritten.",
        },
        strict=True,
    )

    with pytest.raises(ValueError, match="inspect_current_diff"):
        validate_completion_receipt(workspace, receipt)

    class _Context:
        deps = workspace

    inspect_current_diff(_Context())  # type: ignore[arg-type]
    assert validate_completion_receipt(workspace, receipt) == receipt


def test_real_rewrite_probe_config_is_strict_and_selects_ten_cases() -> None:
    config = load_synthesis_raw_text_rewrite_probe_config(
        Path("configs/synthesis/mpci_bl_combined2174_raw_text_rewrite_probe10_luna_high.yaml")
    )

    assert len(config.cases) == 10
    assert config.workflow.max_concurrent_cases == 5
    assert config.provider.generation_settings is None
    assert config.workflow.publish_training_records is False
