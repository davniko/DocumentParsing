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
        "source_blank_lines": 1,
        "candidate_blank_lines": 1,
        "source_nonblank_lines": 3,
        "candidate_nonblank_lines": 3,
        "source_tokens": 5,
        "candidate_tokens": 5,
        "source_unique_tokens": 5,
        "candidate_unique_tokens": 5,
        "source_digit_fraction": 1 / 33,
        "candidate_digit_fraction": 1 / 33,
        "source_uppercase_fraction": 1.0,
        "candidate_uppercase_fraction": 1.0,
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
            key=lambda document_id: _MODULE.sha256_bytes(f"{pipeline_sha}:{document_id}".encode()),
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
        key=lambda document_id: _MODULE.sha256_bytes(f"{pipeline_sha}:{document_id}".encode()),
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
        _MODULE._audit_disposition(status="blocked", manual_audit={"all_checks_passed": True})


def test_matplotlib_seaborn_plot_is_a_deterministic_png() -> None:
    values = _MODULE.pd.DataFrame(
        [
            {
                "document_id": "doc_1",
                "status": "blocked",
                "variant": variant,
                "dimension": "package_type",
                "role": "g1",
                "value": value,
            }
            for variant, value in (("source", "PACKAGE_BOX"), ("synthetic_target", "PACKAGE_BAG"))
        ]
    )

    first = _MODULE._paired_category_png(
        values,
        dimension="package_type",
        title="Source < target & categories",
    )
    second = _MODULE._paired_category_png(
        values,
        dimension="package_type",
        title="Source < target & categories",
    )

    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert first == second


def test_semantic_transitions_preserve_source_target_signatures() -> None:
    rows = [
        {
            "document_id": "doc_1",
            "status": "blocked",
            "variant": variant,
            "dimension": "container_type",
            "role": "",
            "value": value,
        }
        for variant, value in (
            ("source", "GENERAL_PURPOSE"),
            ("synthetic_target", "REFRIGERATED"),
        )
    ]

    assert _MODULE._semantic_transition_rows(rows) == [
        {
            "document_id": "doc_1",
            "status": "blocked",
            "dimension": "container_type",
            "source_values": "GENERAL_PURPOSE",
            "target_values": "REFRIGERATED",
            "changed": True,
        }
    ]


def test_wilson_interval_is_bounded_and_non_degenerate() -> None:
    lower, upper = _MODULE._wilson_interval(25, 25)

    assert lower == pytest.approx(0.866807, abs=1e-6)
    assert upper == pytest.approx(1.0)


def test_clean_yield_uses_direct_cohort_interval_for_complete_census() -> None:
    estimate, lower, upper, method = _MODULE._clean_yield_estimate(
        documents=100,
        certified=6,
        manual_passes=4,
        random_passes=4,
        random_documents=6,
        certified_census_complete=True,
    )

    assert estimate == 0.04
    assert lower == pytest.approx(0.0156633, abs=1e-7)
    assert upper == pytest.approx(0.0983707, abs=1e-7)
    assert method == "complete_machine_certified_census"


def test_clean_yield_scales_conditional_sample_when_census_is_incomplete() -> None:
    estimate, lower, upper, method = _MODULE._clean_yield_estimate(
        documents=100,
        certified=20,
        manual_passes=3,
        random_passes=3,
        random_documents=5,
        certified_census_complete=False,
    )
    conditional_lower, conditional_upper = _MODULE._wilson_interval(3, 5)

    assert estimate == pytest.approx(0.12)
    assert lower == pytest.approx(0.2 * conditional_lower)
    assert upper == pytest.approx(0.2 * conditional_upper)
    assert method == "screening_fraction_times_sampled_conditional_precision"


def test_finding_rows_merge_deterministic_and_semantic_evidence() -> None:
    invariant = SimpleNamespace(
        findings=(
            SimpleNamespace(
                findingKind="source_only_private_or_auxiliary_fact",
                dimension="source_private_and_auxiliary_replacement",
                invariantId="D0123456789abcdef",
                evidence=(SimpleNamespace(lineId="L00002", currentLine="OLD CARRIER"),),
                targetPaths=("documentPatch.parties.carrier.name",),
                repairs=(SimpleNamespace(),),
                problem="A source carrier survived.",
            ),
        )
    )
    case = SimpleNamespace(
        lineage=SimpleNamespace(documentId="doc_1", status="blocked"),
        invariant_audit=invariant,
        semantic_findings=(
            {
                "findingKind": "target_fact_mismatch",
                "dimension": "target_fact_fidelity",
                "evidence": [{"lineId": "L00003", "currentFragment": "WRONG WEIGHT"}],
                "obligationIds": ["O1"],
                "problem": "Weight differs.",
            },
        ),
    )

    rows = _MODULE._finding_rows((case,))

    assert [row["origin"] for row in rows] == ["deterministic", "semantic"]
    assert rows[0]["invariant_id"] == "D0123456789abcdef"
    assert rows[0]["repair_count"] == 1
    assert rows[1]["obligation_ids"] == "O1"
    assert rows[1]["line_ids"] == "L00003"


def test_case_workbook_embeds_manual_false_positive_evidence() -> None:
    case = SimpleNamespace(
        lineage=SimpleNamespace(
            documentId="doc_1",
            status="certified",
            correctionRounds=0,
            currentCandidateSha256="a" * 64,
            blockingReason=None,
        ),
        source="--- PAGE 1 ---\nMaersk A/S\nhttps://www.maersk.com\n",
        candidate="--- PAGE 1 ---\nNew Carrier Ltd.\nhttps://www.maersk.com\n",
        target_label={"documentPatch": {}},
        invariant_audit=None,
        semantic_findings=(),
    )
    manual = {
        "semantic_consistency": False,
        "template_fidelity": True,
        "privacy_rewritten": False,
        "all_checks_passed": False,
        "selection_basis": "deterministic_random;carrier_reference_risk",
        "notes": "The unchanged carrier URL still points to maersk.com.",
    }

    workbook = _MODULE._case_workbook(
        case,
        audit_disposition="manual_quarantine",
        manual_audit=manual,
    ).decode()

    assert "Review disposition: **manual_quarantine**" in workbook
    assert "Semantic consistency: **false**" in workbook
    assert "Privacy/source replacement: **false**" in workbook
    assert "The unchanged carrier URL still points to maersk.com." in workbook
    assert workbook.count("## Latest deterministic and independent findings") == 1
