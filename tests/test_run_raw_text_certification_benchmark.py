from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from document_ocr.synthesis.config import (
    CommittedArtifactDirectoryConfig,
    load_synthesis_raw_text_certification_config,
)

_TOOL = Path(__file__).resolve().parents[1] / "tools/run_raw_text_certification_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("run_raw_text_certification_benchmark", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_ROOT = Path(__file__).resolve().parents[1]
_LOW_CONFIG = (
    _ROOT / "configs/synthesis/mpci_bl_raw_text_certification_benchmark15_holdout_low_v1.yaml"
)
_HIGH_CONFIG = (
    _ROOT / "configs/synthesis/mpci_bl_raw_text_certification_benchmark15_holdout_high_v1.yaml"
)


def _issue(
    issue_id: str,
    *groups: tuple[str, ...],
    dimensions: tuple[str, ...] = ("repeated_and_derived_relations",),
) -> Any:
    return _MODULE.ExpectedIssue.model_validate(
        {
            "issueId": issue_id,
            "primaryDimension": dimensions[0],
            "acceptableDimensions": dimensions,
            "requiredEvidenceGroups": groups,
            "description": f"Ground-truth defect {issue_id}.",
        },
        strict=True,
    )


def _finding(kind: str, *line_ids: str) -> Any:
    return SimpleNamespace(
        findingKind=kind,
        evidence=tuple(SimpleNamespace(lineId=line_id) for line_id in line_ids),
    )


def test_expected_issue_requires_distinct_nonempty_evidence_groups() -> None:
    with pytest.raises(ValueError, match="repeats an evidence group"):
        _issue("duplicate_groups", ("L00001",), ("L00001",))

    with pytest.raises(ValueError):
        _MODULE.ExpectedIssue.model_validate(
            {
                "issueId": "empty_group",
                "primaryDimension": "repeated_and_derived_relations",
                "acceptableDimensions": ("repeated_and_derived_relations",),
                "requiredEvidenceGroups": ((),),
                "description": "Invalid empty evidence boundary.",
            },
            strict=True,
        )


def test_issue_matching_is_maximum_one_to_one_and_requires_every_group() -> None:
    issues = (
        _issue("flexible", ("L00001", "L00002")),
        _issue("line_one_only", ("L00001",)),
        _issue("needs_both_sides", ("L00003",), ("L00004",)),
    )
    case = SimpleNamespace(
        replay=SimpleNamespace(
            findings=(
                _finding("repeated_or_derived_fact_mismatch", "L00001"),
                _finding("repeated_or_derived_fact_mismatch", "L00002"),
                _finding("repeated_or_derived_fact_mismatch", "L00003"),
            )
        )
    )

    matches = _MODULE._match_expected_issues(issues, case)

    assert matches == {0: 1, 1: 0}
    assert 2 not in matches


def test_arm_transport_allows_only_zero_cost_retryable_prefix_before_one_success() -> None:
    zero_usage = SimpleNamespace(
        requests=0,
        inputTokens=0,
        outputTokens=0,
        reasoningTokens=0,
        visibleOutputTokens=0,
        cacheReadTokens=0,
        cacheWriteTokens=0,
        estimatedCostUsd=Decimal(0),
        providerReportedCostUsd=None,
        providerResponseIds=(),
        downstreamProviders=(),
        finishReasons=(),
    )
    success_usage = SimpleNamespace(requests=1)
    retry = SimpleNamespace(
        errorType="ModelHTTPError",
        retryableRouteError=True,
        routeError=SimpleNamespace(statusCode=429),
        hostError=None,
        modelOutput=None,
        usage=zero_usage,
    )
    success = SimpleNamespace(
        errorType=None,
        retryableRouteError=False,
        routeError=None,
        hostError=None,
        modelOutput=SimpleNamespace(),
        usage=success_usage,
    )
    case = SimpleNamespace(stages=(retry, success), result=SimpleNamespace(usage=success_usage))

    assert _MODULE._case_has_one_success_after_zero_cost_retryable_prefix(case)
    assert not _MODULE._case_has_one_success_after_zero_cost_retryable_prefix(
        SimpleNamespace(
            stages=(
                SimpleNamespace(
                    **{**retry.__dict__, "retryableRouteError": False},
                ),
                success,
            ),
            result=case.result,
        )
    )
    assert not _MODULE._case_has_one_success_after_zero_cost_retryable_prefix(
        SimpleNamespace(stages=(success, retry), result=case.result)
    )


def test_weight_mismatch_gold_requires_every_operand_and_unit_anchor() -> None:
    cohort = _MODULE._load_config(
        _ROOT / "configs/synthesis/mpci_bl_raw_text_certification_benchmark15_cohort_v3.json",
        _MODULE.CohortBuildConfig,
    )
    specification = next(
        row
        for row in cohort.cases
        if row.documentId == "doc_3438a61a5ab18c0fe93d4c77f2552c0246107c37584aa6d1096f18fe62aa0234"
    )
    issue = next(
        row
        for row in specification.issues
        if row.issueId == "container_weights_factor_thousand_mismatch"
    )
    incomplete_total = SimpleNamespace(
        replay=SimpleNamespace(
            findings=(
                _finding(
                    "repeated_or_derived_fact_mismatch",
                    "L00080",
                    "L00094",
                    "L00111",
                    "L00118",
                    "L00129",
                ),
            )
        )
    )
    incomplete_components = SimpleNamespace(
        replay=SimpleNamespace(
            findings=(
                _finding(
                    "repeated_or_derived_fact_mismatch",
                    "L00080",
                    "L00111",
                    "L00117",
                    "L00118",
                    "L00129",
                ),
            )
        )
    )
    complete = SimpleNamespace(
        replay=SimpleNamespace(
            findings=(
                _finding(
                    "repeated_or_derived_fact_mismatch",
                    "L00080",
                    "L00094",
                    "L00111",
                    "L00117",
                    "L00118",
                    "L00129",
                ),
            )
        )
    )

    assert not _MODULE._finding_matches_issue(issue, incomplete_total, 0)
    assert not _MODULE._finding_matches_issue(issue, incomplete_components, 0)
    assert _MODULE._finding_matches_issue(issue, complete, 0)


def test_holdout_cohort_issue_count_is_derived_from_frozen_ground_truth() -> None:
    cohort = _MODULE._load_config(
        _ROOT
        / "configs/synthesis/mpci_bl_raw_text_certification_benchmark15_holdout_cohort_v2.json",
        _MODULE.CohortBuildConfig,
    )

    assert len(cohort.cases) == 15
    assert sum(not row.expectedClean for row in cohort.cases) == 9
    assert sum(len(row.issues) for row in cohort.cases) == 18
    assert {issue.primaryDimension for row in cohort.cases for issue in row.issues} == set(
        _MODULE._AUDIT_DIMENSIONS
    )
    issue_ids = {issue.issueId for row in cohort.cases for issue in row.issues}
    assert "component_weights_conflict_crossed_aggregate" not in issue_ids
    assert {
        "target_container_gross_weight_not_rendered",
        "derived_target_package_total_not_rendered",
    } <= issue_ids


def test_secondary_cohort_replaces_disproved_control_without_claiming_untouched_status() -> None:
    cohort = _MODULE._load_config(
        _ROOT
        / "configs/synthesis/mpci_bl_raw_text_certification_benchmark15_secondary_cohort_v1.json",
        _MODULE.CohortBuildConfig,
    )
    development = _MODULE._load_config(
        _ROOT / "configs/synthesis/mpci_bl_raw_text_certification_benchmark15_cohort_v3.json",
        _MODULE.CohortBuildConfig,
    )

    assert "secondary" in cohort.run.runId
    assert sum(not row.expectedClean for row in cohort.cases) == 9
    assert sum(row.expectedClean for row in cohort.cases) == 6
    secondary_controls = {row.documentId for row in cohort.cases if row.expectedClean}
    development_controls = {row.documentId for row in development.cases if row.expectedClean}
    assert (
        "doc_2809b1ea448cee68911e4c468afb3ef8f1c12b0a94ee2f8ca1922e2b5538c208"
        not in secondary_controls
    )
    assert secondary_controls & development_controls == {
        "doc_c68c55e06e89726858c2eb9138be7a1edbd12963320e8a8f02c7fea86c034811"
    }


def test_current_cohorts_pin_compiler_inputs_and_remove_disproven_annotations() -> None:
    cohort = _MODULE._load_config(
        _ROOT / "configs/synthesis/mpci_bl_raw_text_certification_benchmark15_cohort_v3.json",
        _MODULE.CohortBuildConfig,
    )

    assert cohort.schemaVersion == 2
    assert cohort.compilerInventoryConfig is not None
    assert cohort.invariantInputs is not None
    issue_ids = {issue.issueId for row in cohort.cases for issue in row.issues}
    assert "orphaned_model_artifact_marker" not in issue_ids
    assert sum(len(row.issues) for row in cohort.cases) == 29
    control_ids = {row.documentId for row in cohort.cases if row.expectedClean}
    assert {
        "doc_c68c55e06e89726858c2eb9138be7a1edbd12963320e8a8f02c7fea86c034811",
        "doc_6ba2f406c06f73af60c2cbc8eded47c563a6c9e8383be9603e42bb759f753c26",
        "doc_d5498799bb1a50c8e5b9fe9cc82a59bf217c1ad6ee7225d922f90109946c20ec",
    } <= control_ids
    assert (
        not {
            "doc_e836b643a799620af0e897e0fa36d0673bd4b23f3c839e454855518d6f98ac4b",
            "doc_428aa4805d1feaa2bac0f0507db90fd5893deee3632a0454733784116b036017",
            "doc_d422eaa8085d05e468b52d16a4c77b03715924308d50a64700bfe5f78e140440",
        }
        & control_ids
    )

    missing_compiler = cohort.model_dump(mode="python")
    missing_compiler["compilerInventoryConfig"] = None
    with pytest.raises(ValueError, match="requires compiler and invariant inputs"):
        _MODULE.CohortBuildConfig.model_validate(missing_compiler, strict=True)


def test_deterministic_issue_matching_is_one_to_one_and_evidence_complete() -> None:
    issues = (
        _issue("first", ("L00001",), ("L00002",)),
        _issue("second", ("L00001",)),
    )
    findings = (
        SimpleNamespace(
            dimension="repeated_and_derived_relations",
            evidence=(SimpleNamespace(lineId="L00001"),),
        ),
        SimpleNamespace(
            dimension="repeated_and_derived_relations",
            evidence=(SimpleNamespace(lineId="L00001"), SimpleNamespace(lineId="L00002")),
        ),
    )

    assert _MODULE._match_deterministic_issues(issues, findings) == {0: 1, 1: 0}


def test_deterministic_precision_does_not_count_unmatchable_classic_host_strings() -> None:
    summary = _MODULE._deterministic_finding_summary(
        (
            {"invariantFindings": 2, "classicHostFindings": 1},
            {"invariantFindings": 1, "classicHostFindings": 2},
        ),
        issues_matched=3,
    )

    assert summary == {
        "findings": 3,
        "classicHostFindings": 3,
        "allAuditFindings": 6,
        "findingPrecision": 1.0,
    }


def test_cohort_contract_handoff_accepts_only_direct_or_authorized_refinement() -> None:
    source = "CARRIER: CMA CGM Société Anonyme\nSIGNED FOR THE CARRIER CMA CGM S.A.\n"
    inventory_contract = {
        "changedLeaves": [
            {
                "path": "documentPatch.parties.carrier.name",
                "sourceValue": "CMA CGM Société Anonyme",
            }
        ],
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": ["documentPatch.parties.carrier.name"],
                "targetValue": "Asterline Maritime Carriers Pte. Ltd.",
                "requiredOccurrences": 1,
            }
        ],
    }
    refined = _MODULE.raw_text_correction_module._refine_carrier_occurrence_contract(
        source=source,
        contract=inventory_contract,
    )

    _MODULE._validate_inventory_certification_contract_handoff(
        source=source,
        inventory_contract=inventory_contract,
        certification_contract=inventory_contract,
    )
    _MODULE._validate_inventory_certification_contract_handoff(
        source=source,
        inventory_contract=inventory_contract,
        certification_contract=refined,
    )
    with pytest.raises(ValueError, match="contract handoff differs"):
        _MODULE._validate_inventory_certification_contract_handoff(
            source=source,
            inventory_contract=inventory_contract,
            certification_contract={**refined, "unexpected": True},
        )


def _case_row(
    document_id: str,
    *,
    expected_clean: bool,
    predicted_clean: bool,
    gold_issue_caught: bool,
    cost: str,
    findings: int,
) -> dict[str, Any]:
    return {
        "documentId": document_id,
        "expectedClean": expected_clean,
        "predictedClean": predicted_clean,
        "correct": (
            predicted_clean if expected_clean else (not predicted_clean and gold_issue_caught)
        ),
        "goldIssueCaught": gold_issue_caught,
        "providerReportedCostUsd": cost,
        "findings": findings,
    }


def _issue_row(document_id: str, issue_id: str, *, matched: bool) -> dict[str, Any]:
    return {
        "documentId": document_id,
        "issueId": issue_id,
        "acceptableDimensions": ["repeated_and_derived_relations"],
        "requiredEvidenceGroups": [["L00001"]],
        "matched": matched,
    }


def test_counterfactual_cascade_does_not_borrow_challenger_findings_after_baseline_reject() -> None:
    document_id = f"doc_{1:064x}"
    low_cases = [
        _case_row(
            document_id,
            expected_clean=False,
            predicted_clean=False,
            gold_issue_caught=True,
            cost="0.01",
            findings=1,
        )
    ]
    high_cases = [
        _case_row(
            document_id,
            expected_clean=False,
            predicted_clean=False,
            gold_issue_caught=True,
            cost="0.03",
            findings=3,
        )
    ]
    low_issues = [
        _issue_row(document_id, f"issue_{index}", matched=index == 0) for index in range(3)
    ]
    high_issues = [_issue_row(document_id, f"issue_{index}", matched=True) for index in range(3)]

    summary, rows = _MODULE._counterfactual_cascade(
        baseline_cases=low_cases,
        challenger_cases=high_cases,
        baseline_issues=low_issues,
        challenger_issues=high_issues,
    )

    assert summary["correct"] == 1
    assert summary["challengerCalls"] == 0
    assert summary["issuesMatched"] == 1
    assert summary["issueRecall"] == pytest.approx(1 / 3)
    assert summary["providerReportedCostUsd"] == "0.01"
    assert summary["challengerOnlyMatchesOnBaselineRejectedCases"] == 2
    assert sum(row["matched"] for row in rows) == 1


def test_counterfactual_cascade_runs_challenger_only_after_baseline_clean() -> None:
    document_id = f"doc_{2:064x}"
    low_cases = [
        _case_row(
            document_id,
            expected_clean=False,
            predicted_clean=True,
            gold_issue_caught=False,
            cost="0.01",
            findings=0,
        )
    ]
    high_cases = [
        _case_row(
            document_id,
            expected_clean=False,
            predicted_clean=False,
            gold_issue_caught=True,
            cost="0.03",
            findings=1,
        )
    ]

    summary, _rows = _MODULE._counterfactual_cascade(
        baseline_cases=low_cases,
        challenger_cases=high_cases,
        baseline_issues=[_issue_row(document_id, "caught_by_high", matched=False)],
        challenger_issues=[_issue_row(document_id, "caught_by_high", matched=True)],
    )

    assert summary["correct"] == 1
    assert summary["challengerCalls"] == 1
    assert summary["issuesMatched"] == 1
    assert summary["providerReportedCostUsd"] == "0.04"


def _plan_for_templates() -> Any:
    low = load_synthesis_raw_text_certification_config(_LOW_CONFIG)
    high = load_synthesis_raw_text_certification_config(_HIGH_CONFIG)
    cohort_pin = _MODULE.RunPin(
        path=low.input_run.path,
        commitSha256=low.input_run.commit_sha256,
        transactionSha256=low.input_run.transaction_sha256,
    )
    cohort = SimpleNamespace(
        config=SimpleNamespace(prompt=low.prompt),
        reference=SimpleNamespace(pin=cohort_pin),
        cases=tuple(
            SimpleNamespace(specification=SimpleNamespace(documentId=document_id))
            for document_id in low.case_ids
        ),
    )
    plan_pin = _MODULE.RunPin(
        path="artifacts/kie-synthesis/benchmark-plan",
        commitSha256="a" * 64,
        transactionSha256="b" * 64,
    )
    plan_config = _MODULE.BenchmarkPlanConfig(
        schemaVersion=2,
        task="raw_text_certification_contract_v2_benchmark_plan_v2",
        run=_MODULE.BenchmarkRunConfig(
            runId="benchmark-plan",
            outputDir="artifacts/kie-synthesis",
        ),
        cohortRun=cohort_pin,
        baseline=_MODULE.BenchmarkPlannedArm(
            effort="low",
            config=_MODULE.FilePin(
                path=_LOW_CONFIG.relative_to(_ROOT).as_posix(),
                sha256=_MODULE.sha256_file(_LOW_CONFIG),
            ),
        ),
        challenger=_MODULE.BenchmarkPlannedArm(
            effort="high",
            config=_MODULE.FilePin(
                path=_HIGH_CONFIG.relative_to(_ROOT).as_posix(),
                sha256=_MODULE.sha256_file(_HIGH_CONFIG),
            ),
        ),
    )
    return _MODULE.LoadedBenchmarkPlan(
        reference=_MODULE.ResolvedRun(pin=plan_pin, root=_ROOT / plan_pin.path),
        config=plan_config,
        cohort=cohort,
        baseline_config=low,
        challenger_config=high,
        baseline_config_payload=_LOW_CONFIG.read_bytes(),
        challenger_config_payload=_HIGH_CONFIG.read_bytes(),
    )


def _authorized(template: Any, plan: Any) -> Any:
    authority = CommittedArtifactDirectoryConfig(
        path=plan.reference.pin.path,
        commit_sha256=plan.reference.pin.commitSha256,
        transaction_sha256=plan.reference.pin.transactionSha256,
    )
    return template.model_copy(update={"benchmark_plan": authority})


def test_authorized_config_is_exact_plan_derivative() -> None:
    plan = _plan_for_templates()
    authorized = _authorized(plan.baseline_config, plan)

    _MODULE._validate_authorized_config(
        config=authorized,
        template=plan.baseline_config,
        plan=plan,
        effort="low",
    )

    changed_provider = authorized.provider.model_copy(update={"max_output_tokens": 4096})
    altered = authorized.model_copy(update={"provider": changed_provider})
    with pytest.raises(ValueError, match="configuration"):
        _MODULE._validate_authorized_config(
            config=altered,
            template=plan.baseline_config,
            plan=plan,
            effort="low",
        )

    wrong_authority = authorized.model_copy(
        update={
            "benchmark_plan": authorized.benchmark_plan.model_copy(
                update={"commit_sha256": "c" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="plan-authorized derivative"):
        _MODULE._validate_authorized_config(
            config=wrong_authority,
            template=plan.baseline_config,
            plan=plan,
            effort="low",
        )


def test_planned_arms_cannot_reuse_one_run_identity() -> None:
    plan = _plan_for_templates()
    repeated_challenger = plan.challenger_config.model_copy(
        update={"run": plan.baseline_config.run}
    )

    with pytest.raises(ValueError, match="reuse one run identity"):
        _MODULE._validate_planned_configs(
            cohort=plan.cohort,
            baseline_config=plan.baseline_config,
            challenger_config=repeated_challenger,
            baseline_effort="low",
            challenger_effort="high",
        )


def test_plan_accepts_low_medium_and_rejects_nonincreasing_effort() -> None:
    plan = _plan_for_templates()
    value = plan.config.model_dump(mode="json")
    value["challenger"]["effort"] = "medium"
    medium = _MODULE.BenchmarkPlanConfig.model_validate(value, strict=True)

    assert medium.baseline.effort == "low"
    assert medium.challenger.effort == "medium"

    value["challenger"]["effort"] = "low"
    with pytest.raises(ValueError, match="efforts must be distinct"):
        _MODULE.BenchmarkPlanConfig.model_validate(value, strict=True)


def test_challenger_arm_resolves_medium_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan_for_templates()
    medium_provider = plan.challenger_config.provider.model_copy(
        update={"reasoning_effort": "medium"}
    )
    medium_template = plan.challenger_config.model_copy(update={"provider": medium_provider})
    plan_value = plan.config.model_dump(mode="json")
    plan_value["challenger"]["effort"] = "medium"
    medium_plan = _MODULE.LoadedBenchmarkPlan(
        reference=plan.reference,
        config=_MODULE.BenchmarkPlanConfig.model_validate(plan_value, strict=True),
        cohort=SimpleNamespace(cases=()),
        baseline_config=plan.baseline_config,
        challenger_config=medium_template,
        baseline_config_payload=plan.baseline_config_payload,
        challenger_config_payload=plan.challenger_config_payload,
    )
    authority = _authorized(medium_template, medium_plan).benchmark_plan
    assert authority is not None
    root = tmp_path / medium_template.run.output_dir / medium_template.run.run_id
    root.mkdir(parents=True)
    implementation = _MODULE.certification_implementation_contract()
    runtime = _MODULE._certification_runtime_contract()
    run = SimpleNamespace(
        root=root,
        config=_authorized(medium_template, medium_plan),
        transaction={
            "benchmarkPlan": authority.model_dump(mode="json"),
            "implementationSha256": implementation["implementationSha256"],
            "dependencyImplementationSha256": implementation["dependencyImplementationSha256"],
            "runtime": runtime,
        },
        cases=(),
    )
    pin = _MODULE.RunPin(
        path=root.relative_to(tmp_path).as_posix(),
        commitSha256="f" * 64,
        transactionSha256="1" * 64,
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        _MODULE,
        "_validate_run_pin",
        lambda *_args, **_kwargs: SimpleNamespace(root=root),
    )
    monkeypatch.setattr(_MODULE, "load_validated_certification_run", lambda _root: run)

    def validate_authority(*, template: Any, effort: str, **_kwargs: Any) -> None:
        observed.update(template=template, effort=effort)

    monkeypatch.setattr(_MODULE, "_validate_authorized_config", validate_authority)

    assert (
        _MODULE._validate_arm(
            project_root=tmp_path,
            pin=pin,
            plan=medium_plan,
            role="challenger",
        )
        is run
    )
    assert observed == {"template": medium_template, "effort": "medium"}


def test_report_names_actual_efforts_and_hidden_challenger_matches() -> None:
    summary = {
        "promotionGatePassed": False,
        "documents": 1,
        "knownDefects": 1,
        "cleanControls": 0,
        "baselineEffort": "low",
        "baseline": {"correct": 1, "issueRecall": 0.5, "findingPrecision": 1.0},
        "challengerEffort": "medium",
        "challenger": {"correct": 1, "issueRecall": 1.0, "findingPrecision": 1.0},
        "counterfactualCascade": {
            "correct": 1,
            "issueRecall": 0.5,
            "providerReportedCostUsd": "0.01",
            "challengerOnlyMatchesOnBaselineRejectedCases": 2,
        },
        "combinedArmCostUsd": "0.04",
        "medianReasoningRatio": "2",
        "medianCostRatio": "2",
    }

    report = _MODULE._report(summary, []).decode()

    assert "Baseline (LOW) accuracy: **1/1**" in report
    assert "Challenger (MEDIUM) accuracy: **1/1**" in report
    assert (
        "Challenger-only issue matches hidden by baseline-reject short-circuiting: **2**" in report
    )


def test_launch_validates_both_arms_before_any_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "launch.json"
    config_path.write_text("{}", encoding="utf-8")
    plan = _plan_for_templates()
    launch = _MODULE.BenchmarkLaunchConfig(
        schemaVersion=2,
        task="raw_text_certification_contract_v2_benchmark_launch_v2",
        planRun=plan.reference.pin,
        baselineConfig=_MODULE.FilePin(path="low.yaml", sha256="d" * 64),
        challengerConfig=_MODULE.FilePin(path="high.yaml", sha256="e" * 64),
    )
    low = _authorized(plan.baseline_config, plan)
    high = _authorized(plan.challenger_config, plan)
    calls: list[str] = []

    monkeypatch.setattr(_MODULE, "_load_config", lambda *_args: launch)
    monkeypatch.setattr(_MODULE, "_load_plan", lambda *_args: plan)
    monkeypatch.setattr(
        _MODULE,
        "_load_certification_config_pin",
        lambda _root, pin, **_kwargs: (
            (low, b"low") if pin == launch.baselineConfig else (high, b"high")
        ),
    )

    def validate(*, effort: str, **_kwargs: Any) -> None:
        if effort == "high":
            raise ValueError("challenger configuration is altered")

    monkeypatch.setattr(_MODULE, "_validate_authorized_config", validate)

    def run_provider(**_kwargs: Any) -> dict[str, Any]:
        calls.append("provider")
        return {}

    monkeypatch.setattr(_MODULE, "run_raw_text_certification", run_provider)

    with pytest.raises(ValueError, match="challenger configuration is altered"):
        _MODULE.launch_benchmark(project_root=tmp_path, config_path=config_path)

    assert calls == []


def test_launch_validates_completed_baseline_arm_before_starting_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "launch.json"
    config_path.write_text("{}", encoding="utf-8")
    plan = _plan_for_templates()
    launch = _MODULE.BenchmarkLaunchConfig(
        schemaVersion=2,
        task="raw_text_certification_contract_v2_benchmark_launch_v2",
        planRun=plan.reference.pin,
        baselineConfig=_MODULE.FilePin(path="low.yaml", sha256="d" * 64),
        challengerConfig=_MODULE.FilePin(path="high.yaml", sha256="e" * 64),
    )
    low = _authorized(plan.baseline_config, plan)
    high = _authorized(plan.challenger_config, plan)
    calls: list[str] = []

    monkeypatch.setattr(_MODULE, "_load_config", lambda *_args: launch)
    monkeypatch.setattr(_MODULE, "_load_plan", lambda *_args: plan)
    monkeypatch.setattr(
        _MODULE,
        "_load_certification_config_pin",
        lambda _root, pin, **_kwargs: (
            (low, b"low") if pin == launch.baselineConfig else (high, b"high")
        ),
    )
    monkeypatch.setattr(_MODULE, "_validate_authorized_config", lambda **_kwargs: None)

    def run_arm(*, config: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append(config.provider.reasoning_effort)
        return {}

    monkeypatch.setattr(_MODULE, "run_raw_text_certification", run_arm)
    monkeypatch.setattr(
        _MODULE,
        "_committed_arm_pin",
        lambda **_kwargs: _MODULE.RunPin(
            path="arm", commitSha256="f" * 64, transactionSha256="1" * 64
        ),
    )

    def reject_baseline(**_kwargs: Any) -> None:
        raise ValueError("baseline arm incomplete")

    monkeypatch.setattr(_MODULE, "_validate_arm", reject_baseline)

    with pytest.raises(ValueError, match="baseline arm incomplete"):
        _MODULE.launch_benchmark(project_root=tmp_path, config_path=config_path)

    assert calls == ["low"]


def test_arm_rejects_certification_implementation_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan_for_templates()
    authority = _authorized(plan.baseline_config, plan).benchmark_plan
    assert authority is not None
    root = tmp_path / plan.baseline_config.run.output_dir / plan.baseline_config.run.run_id
    root.mkdir(parents=True)
    run = SimpleNamespace(
        root=root,
        config=_authorized(plan.baseline_config, plan),
        transaction={
            "benchmarkPlan": authority.model_dump(mode="json"),
            "implementationSha256": "0" * 64,
            "dependencyImplementationSha256": {"providerRuntime": "0" * 64},
        },
        cases=(),
    )
    pinned = _MODULE.RunPin(
        path=root.relative_to(tmp_path).as_posix(),
        commitSha256="f" * 64,
        transactionSha256="1" * 64,
    )
    fake_plan = _MODULE.LoadedBenchmarkPlan(
        reference=plan.reference,
        config=plan.config,
        cohort=SimpleNamespace(cases=()),
        baseline_config=plan.baseline_config,
        challenger_config=plan.challenger_config,
        baseline_config_payload=plan.baseline_config_payload,
        challenger_config_payload=plan.challenger_config_payload,
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_run_pin",
        lambda *_args, **_kwargs: SimpleNamespace(root=root),
    )
    monkeypatch.setattr(_MODULE, "load_validated_certification_run", lambda _root: run)
    monkeypatch.setattr(_MODULE, "_validate_authorized_config", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="pre-registered plan"):
        _MODULE._validate_arm(
            project_root=tmp_path,
            pin=pinned,
            plan=fake_plan,
            role="baseline",
        )


def test_arm_rejects_runtime_version_drift_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan_for_templates()
    authority = _authorized(plan.baseline_config, plan).benchmark_plan
    assert authority is not None
    root = tmp_path / plan.baseline_config.run.output_dir / plan.baseline_config.run.run_id
    root.mkdir(parents=True)
    implementation = _MODULE.certification_implementation_contract()
    runtime = _MODULE._certification_runtime_contract()
    run = SimpleNamespace(
        root=root,
        config=_authorized(plan.baseline_config, plan),
        transaction={
            "benchmarkPlan": authority.model_dump(mode="json"),
            "implementationSha256": implementation["implementationSha256"],
            "dependencyImplementationSha256": implementation["dependencyImplementationSha256"],
            "runtime": {
                "pydanticAiVersion": runtime["pydanticAiVersion"],
                "openaiVersion": "0.0.0",
            },
        },
        cases=(),
    )
    pinned = _MODULE.RunPin(
        path=root.relative_to(tmp_path).as_posix(),
        commitSha256="f" * 64,
        transactionSha256="1" * 64,
    )
    fake_plan = _MODULE.LoadedBenchmarkPlan(
        reference=plan.reference,
        config=plan.config,
        cohort=SimpleNamespace(cases=()),
        baseline_config=plan.baseline_config,
        challenger_config=plan.challenger_config,
        baseline_config_payload=plan.baseline_config_payload,
        challenger_config_payload=plan.challenger_config_payload,
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_run_pin",
        lambda *_args, **_kwargs: SimpleNamespace(root=root),
    )
    monkeypatch.setattr(_MODULE, "load_validated_certification_run", lambda _root: run)
    monkeypatch.setattr(_MODULE, "_validate_authorized_config", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="pre-registered plan"):
        _MODULE._validate_arm(
            project_root=tmp_path,
            pin=pinned,
            plan=fake_plan,
            role="baseline",
        )
