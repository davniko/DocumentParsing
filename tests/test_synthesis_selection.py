from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from document_ocr.synthesis.selection import (
    SelectionCandidate,
    SelectionRequest,
    solve_selection,
)


def _candidate(
    document_id: str,
    row_index: int,
    *,
    template: str,
    carrier: str,
    document_type: str,
    corpus: str,
    families: frozenset[str],
    contexts: frozenset[str] = frozenset(),
    risk: int = 0,
) -> SelectionCandidate:
    return SelectionCandidate(
        document_id=document_id,
        source_row_index=row_index,
        template_id=template,
        template_size=3,
        carrier_family=carrier,
        strata={"document_type": document_type, "source_corpus": corpus},
        contexts=contexts,
        eligible_families=families,
        risk_score=risk,
    )


def _candidates() -> tuple[SelectionCandidate, ...]:
    return (
        _candidate(
            "doc_1",
            1,
            template="template_1",
            carrier="carrier_1",
            document_type="bill_of_lading",
            corpus="current",
            families=frozenset({"container", "date"}),
            contexts=frozenset({"rare"}),
        ),
        _candidate(
            "doc_2",
            2,
            template="template_2",
            carrier="carrier_1",
            document_type="sea_waybill",
            corpus="legacy",
            families=frozenset({"container"}),
        ),
        _candidate(
            "doc_3",
            3,
            template="template_3",
            carrier="carrier_2",
            document_type="bill_of_lading",
            corpus="legacy",
            families=frozenset({"date"}),
            contexts=frozenset({"agent"}),
        ),
        _candidate(
            "doc_4",
            4,
            template="template_4",
            carrier="carrier_2",
            document_type="sea_waybill",
            corpus="current",
            families=frozenset({"container", "date"}),
            contexts=frozenset({"rare"}),
        ),
        _candidate(
            "doc_5",
            5,
            template="template_5",
            carrier="carrier_3",
            document_type="bill_of_lading",
            corpus="current",
            families=frozenset({"container", "date"}),
            risk=10,
        ),
        _candidate(
            "doc_6",
            6,
            template="template_6",
            carrier="carrier_3",
            document_type="sea_waybill",
            corpus="legacy",
            families=frozenset({"container", "date"}),
            risk=10,
        ),
    )


def _request(**overrides: object) -> SelectionRequest:
    values: dict[str, object] = {
        "requested_documents": 4,
        "family_exact": {"container": 2, "date": 2},
        "strata_exact": {
            "document_type": {"bill_of_lading": 2, "sea_waybill": 2},
            "source_corpus": {"current": 2, "legacy": 2},
        },
        "context_minimums": {"rare": 2, "agent": 1},
        "maximum_per_template": 1,
        "maximum_per_carrier": 2,
        "minimum_carriers": 2,
        "seed": 4242,
    }
    values.update(overrides)
    return SelectionRequest(**values)  # type: ignore[arg-type]


def test_selection_enforces_exact_constraints_and_is_order_stable() -> None:
    candidates = _candidates()
    request = _request()

    first = solve_selection(candidates, request)
    second = solve_selection(tuple(reversed(candidates)), request)

    assert first.assignments == second.assignments
    assert first.solver_receipt == second.solver_receipt
    assert [row.position for row in first.assignments] == [1, 2, 3, 4]
    assert len({row.document_id for row in first.assignments}) == 4
    assert Counter(row.family for row in first.assignments) == {"container": 2, "date": 2}
    assert first.feasibility["all_constraints_satisfied"] is True
    assert first.feasibility["achieved"]["strata"]["document_type"] == {
        "bill_of_lading": 2,
        "sea_waybill": 2,
    }
    assert first.feasibility["achieved"]["strata"]["source_corpus"] == {
        "current": 2,
        "legacy": 2,
    }
    assert first.feasibility["achieved"]["contexts"]["rare"] >= 2
    assert first.feasibility["achieved"]["contexts"]["agent"] >= 1
    assert first.feasibility["achieved"]["carriers"] >= 2
    assert first.feasibility["achieved"]["templates"] == 4

    receipt = first.solver_receipt
    assert receipt["risk_status"] == 0
    assert receipt["tie_break_status"] == 0
    assert receipt["risk_mip_gap"] == 0.0
    assert receipt["tie_break_mip_gap"] == 0.0
    assert receipt["constraint_labels"][-1] == "risk_objective_exact"
    assert first.feasibility["constraint_count"] == len(receipt["constraint_labels"])
    for field in (
        "candidate_inventory_sha256",
        "request_sha256",
        "matrix_sha256",
        "problem_sha256",
    ):
        assert len(receipt[field]) == 64


def test_selection_receipt_covers_all_candidate_and_request_fields() -> None:
    candidates = _candidates()
    baseline = solve_selection(candidates, _request())
    changed_candidate = replace(candidates[-1], contexts=frozenset({"receipt_only_context"}))
    changed = solve_selection((*candidates[:-1], changed_candidate), _request())
    changed_request = solve_selection(candidates, _request(seed=4243))

    assert (
        baseline.solver_receipt["candidate_inventory_sha256"]
        != changed.solver_receipt["candidate_inventory_sha256"]
    )
    assert baseline.solver_receipt["problem_sha256"] != changed.solver_receipt["problem_sha256"]
    assert (
        baseline.solver_receipt["request_sha256"]
        != changed_request.solver_receipt["request_sha256"]
    )
    assert (
        baseline.solver_receipt["problem_sha256"]
        != changed_request.solver_receipt["problem_sha256"]
    )


def test_selection_is_fail_closed_when_joint_constraints_are_infeasible() -> None:
    with pytest.raises(ValueError, match="selection MILP is infeasible or non-optimal") as error:
        solve_selection(_candidates(), _request(context_minimums={"absent": 1}))

    assert "context:absent" in str(error.value)


@pytest.mark.parametrize(
    "override, message",
    (
        ({"requested_documents": 0}, "requested_documents must be a positive integer"),
        ({"family_exact": {"container": 1}}, "exact family counts must sum"),
        (
            {"strata_exact": {"document_type": {"bill_of_lading": 1}}},
            "exact stratum counts",
        ),
        ({"context_minimums": {"rare": -1}}, "non-negative integer"),
        ({"maximum_per_template": 0}, "maximum_per_template must be a positive integer"),
        ({"minimum_carriers": 5}, "minimum_carriers must not exceed"),
        ({"seed": -1}, "unsigned 64-bit"),
    ),
)
def test_selection_request_rejects_invalid_values(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**override)


def test_selection_candidate_rejects_duplicate_or_invalid_families() -> None:
    with pytest.raises(ValueError, match="eligible families contain duplicates"):
        _candidate(
            "doc_duplicate_family",
            10,
            template="template_x",
            carrier="carrier_x",
            document_type="bill_of_lading",
            corpus="current",
            families=("container", "container"),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least one eligible family"):
        _candidate(
            "doc_no_family",
            11,
            template="template_y",
            carrier="carrier_y",
            document_type="bill_of_lading",
            corpus="current",
            families=frozenset(),
        )
    with pytest.raises(ValueError, match="collection of strings"):
        _candidate(
            "doc_string_family",
            12,
            template="template_z",
            carrier="carrier_z",
            document_type="bill_of_lading",
            corpus="current",
            families="container",  # type: ignore[arg-type]
        )


def test_selection_rejects_duplicate_documents_and_source_rows() -> None:
    candidates = _candidates()
    with pytest.raises(ValueError, match="duplicate document IDs"):
        solve_selection((*candidates, candidates[0]), _request())
    duplicate_source_row = replace(candidates[-1], source_row_index=candidates[0].source_row_index)
    with pytest.raises(ValueError, match="duplicate source row indexes"):
        solve_selection((*candidates[:-1], duplicate_source_row), _request())


def test_selection_copies_mutable_request_and_candidate_mappings() -> None:
    strata = {"document_type": "bill_of_lading"}
    candidate = SelectionCandidate(
        document_id="doc_frozen",
        source_row_index=1,
        template_id="template_frozen",
        template_size=3,
        carrier_family="carrier_frozen",
        strata=strata,
        contexts=frozenset({"rare"}),
        eligible_families=frozenset({"container"}),
        risk_score=0,
    )
    family_exact = {"container": 1}
    strata_exact = {"document_type": {"bill_of_lading": 1}}
    request = SelectionRequest(
        requested_documents=1,
        family_exact=family_exact,
        strata_exact=strata_exact,
        context_minimums={},
        maximum_per_template=1,
        maximum_per_carrier=1,
        minimum_carriers=1,
        seed=1,
    )

    strata["document_type"] = "sea_waybill"
    family_exact["container"] = 0
    strata_exact["document_type"]["bill_of_lading"] = 0

    assert candidate.strata["document_type"] == "bill_of_lading"
    assert request.family_exact["container"] == 1
    assert request.strata_exact["document_type"]["bill_of_lading"] == 1
