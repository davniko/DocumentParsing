from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools/analyze_raw_text_pipeline_outcome.py"
_SPEC = importlib.util.spec_from_file_location("analyze_raw_text_pipeline_outcome", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _case(document_id: str, status: str = "certified") -> Any:
    return SimpleNamespace(lineage=SimpleNamespace(documentId=document_id, status=status))


def test_edit_metrics_preserve_exact_topology_contract() -> None:
    source = "--- PAGE 1 ---\nSHIPPER\nOLD NAME\n\n"
    candidate = "--- PAGE 1 ---\nSHIPPER\nNEW NAME\n\n"

    metrics = _MODULE._edit_metrics(source, candidate)

    assert metrics == {
        "source_characters": 33,
        "candidate_characters": 33,
        "character_delta": 0,
        "source_lines": 4,
        "candidate_lines": 4,
        "line_count_delta": 0,
        "blank_line_delta": 0,
        "changed_source_lines": 1,
        "changed_candidate_lines": 1,
        "changed_blocks": 1,
        "unchanged_line_fraction": 0.75,
        "pages": 1,
        "page_markers_preserved": True,
        "terminal_newline_preserved": True,
    }


def test_manual_audit_requires_exact_reproducible_random_sample() -> None:
    pipeline_sha = "a" * 64
    ids = tuple(f"doc_{number:064x}" for number in range(1, 6))
    cases = tuple(_case(document_id) for document_id in ids)
    expected = tuple(
        sorted(
            ids,
            key=lambda document_id: _MODULE.sha256_bytes(
                f"{pipeline_sha}:{document_id}".encode()
            ),
        )[:2]
    )
    audit = _MODULE.ManualAudit.model_validate(
        {
            "schemaVersion": 1,
            "pipelineCommitSha256": pipeline_sha,
            "randomSampleSize": 2,
            "cases": tuple(
                {
                    "documentId": document_id,
                    "semanticConsistency": True,
                    "templateFidelity": True,
                    "privacyRewritten": True,
                    "selectionBasis": ("deterministic_random",),
                    "notes": "Independent review passed.",
                }
                for document_id in expected
            ),
        }
    )

    rows, observed = _MODULE._manual_audit_rows(
        audit=audit,
        cases=cases,
        pipeline_commit_sha256=pipeline_sha,
    )

    assert observed == expected
    assert len(rows) == 2
    assert all(row["all_checks_passed"] is True for row in rows)


def test_manual_audit_rejects_cherry_picked_random_label() -> None:
    pipeline_sha = "b" * 64
    ids = tuple(f"doc_{number:064x}" for number in range(1, 4))
    cases = tuple(_case(document_id) for document_id in ids)
    expected = min(
        ids,
        key=lambda document_id: _MODULE.sha256_bytes(
            f"{pipeline_sha}:{document_id}".encode()
        ),
    )
    wrong = next(document_id for document_id in ids if document_id != expected)
    audit = _MODULE.ManualAudit.model_validate(
        {
            "schemaVersion": 1,
            "pipelineCommitSha256": pipeline_sha,
            "randomSampleSize": 1,
            "cases": (
                {
                    "documentId": wrong,
                    "semanticConsistency": True,
                    "templateFidelity": True,
                    "privacyRewritten": True,
                    "selectionBasis": ("deterministic_random",),
                    "notes": "Wrong sample on purpose.",
                },
            ),
        }
    )

    with pytest.raises(ValueError, match="exact deterministic random sample"):
        _MODULE._manual_audit_rows(
            audit=audit,
            cases=cases,
            pipeline_commit_sha256=pipeline_sha,
        )


def test_manual_audit_cannot_promote_a_quarantined_case() -> None:
    pipeline_sha = "c" * 64
    accepted_id = f"doc_{1:064x}"
    blocked_id = f"doc_{2:064x}"
    audit = _MODULE.ManualAudit.model_validate(
        {
            "schemaVersion": 1,
            "pipelineCommitSha256": pipeline_sha,
            "randomSampleSize": 1,
            "cases": (
                {
                    "documentId": accepted_id,
                    "semanticConsistency": True,
                    "templateFidelity": True,
                    "privacyRewritten": True,
                    "selectionBasis": ("deterministic_random",),
                    "notes": "Accepted case.",
                },
                {
                    "documentId": blocked_id,
                    "semanticConsistency": True,
                    "templateFidelity": True,
                    "privacyRewritten": True,
                    "selectionBasis": ("risk_weighted",),
                    "notes": "Attempted blocked promotion.",
                },
            ),
        }
    )

    with pytest.raises(ValueError, match="non-certified cases"):
        _MODULE._manual_audit_rows(
            audit=audit,
            cases=(_case(accepted_id), _case(blocked_id, "blocked")),
            pipeline_commit_sha256=pipeline_sha,
        )


@pytest.mark.parametrize(
    ("status", "manual", "expected"),
    (
        ("certified", None, "provisional"),
        ("certified", {"all_checks_passed": True}, "confirmed"),
        ("certified", {"all_checks_passed": False}, "manual_quarantine"),
        ("blocked", None, "pipeline_quarantine"),
    ),
)
def test_audit_disposition_never_treats_machine_certification_as_acceptance(
    status: str, manual: dict[str, bool] | None, expected: str
) -> None:
    assert _MODULE._audit_disposition(status=status, manual_audit=manual) == expected


def test_audit_disposition_rejects_manual_promotion_of_pipeline_block() -> None:
    with pytest.raises(ValueError, match="pipeline-quarantined"):
        _MODULE._audit_disposition(
            status="blocked", manual_audit={"all_checks_passed": True}
        )


def test_svg_escapes_labels_and_scales_zero_values() -> None:
    payload = _MODULE._bar_svg(
        title="A < B",
        rows=(("x & y", 0),),
        x_label="documents",
    ).decode()

    assert "A &lt; B" in payload
    assert "x &amp; y" in payload
    assert 'width="0.000"' in payload


def test_wilson_interval_is_bounded_and_non_degenerate() -> None:
    lower, upper = _MODULE._wilson_interval(25, 25)

    assert lower == pytest.approx(0.866807, abs=1e-6)
    assert upper == pytest.approx(1.0)
