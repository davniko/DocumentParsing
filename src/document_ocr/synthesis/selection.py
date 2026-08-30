"""Deterministic MILP selection of source documents and mutation families."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import scipy  # type: ignore[import-untyped]
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]

from document_ocr.hashing import canonical_json_bytes, identity_sha256, sha256_bytes


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    document_id: str
    source_row_index: int
    template_id: str
    template_size: int
    carrier_family: str
    strata: Mapping[str, str]
    contexts: frozenset[str]
    eligible_families: frozenset[str]
    risk_score: int

    def __post_init__(self) -> None:
        for name, value in (
            ("document_id", self.document_id),
            ("template_id", self.template_id),
            ("carrier_family", self.carrier_family),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"selection candidate {name} must be a non-empty string")
        if not isinstance(self.source_row_index, int) or isinstance(self.source_row_index, bool):
            raise ValueError("selection candidate source_row_index must be an integer")
        if self.source_row_index < 0:
            raise ValueError("selection candidate source_row_index must not be negative")
        if not isinstance(self.template_size, int) or isinstance(self.template_size, bool):
            raise ValueError("selection candidate template_size must be an integer")
        if self.template_size <= 0:
            raise ValueError("selection candidate template_size must be positive")
        if not isinstance(self.risk_score, int) or isinstance(self.risk_score, bool):
            raise ValueError("selection candidate risk_score must be an integer")
        if self.risk_score < 0:
            raise ValueError("selection candidate risk_score must not be negative")

        if not isinstance(self.strata, Mapping):
            raise ValueError("selection candidate strata must be a mapping")
        strata = dict(self.strata)
        for field, value in strata.items():
            if not isinstance(field, str) or not field.strip():
                raise ValueError("selection candidate stratum fields must be non-empty strings")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("selection candidate stratum values must be non-empty strings")
        if isinstance(self.contexts, str):
            raise ValueError("selection candidate contexts must be a collection of strings")
        contexts = tuple(self.contexts)
        if len(contexts) != len(set(contexts)):
            raise ValueError("selection candidate contexts contain duplicates")
        if any(not isinstance(value, str) or not value.strip() for value in contexts):
            raise ValueError("selection candidate contexts must be non-empty strings")
        if isinstance(self.eligible_families, str):
            raise ValueError(
                "selection candidate eligible families must be a collection of strings"
            )
        families = tuple(self.eligible_families)
        if len(families) != len(set(families)):
            raise ValueError("selection candidate eligible families contain duplicates")
        if not families:
            raise ValueError("selection candidate requires at least one eligible family")
        if any(not isinstance(value, str) or not value.strip() for value in families):
            raise ValueError("selection candidate eligible families must be non-empty strings")
        object.__setattr__(self, "strata", MappingProxyType(strata))
        object.__setattr__(self, "contexts", frozenset(contexts))
        object.__setattr__(self, "eligible_families", frozenset(families))


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    requested_documents: int
    family_exact: Mapping[str, int]
    strata_exact: Mapping[str, Mapping[str, int]]
    context_minimums: Mapping[str, int]
    maximum_per_template: int
    maximum_per_carrier: int
    minimum_carriers: int
    seed: int

    def __post_init__(self) -> None:
        _positive_integer(self.requested_documents, "requested_documents")
        _positive_integer(self.maximum_per_template, "maximum_per_template")
        _positive_integer(self.maximum_per_carrier, "maximum_per_carrier")
        _positive_integer(self.minimum_carriers, "minimum_carriers")
        if self.minimum_carriers > self.requested_documents:
            raise ValueError("minimum_carriers must not exceed requested_documents")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("selection seed must be an integer")
        if not 0 <= self.seed <= (2**64 - 1):
            raise ValueError("selection seed must fit an unsigned 64-bit integer")

        family_exact = _validated_counts(self.family_exact, "family_exact")
        if not family_exact:
            raise ValueError("family_exact must not be empty")
        if sum(family_exact.values()) != self.requested_documents:
            raise ValueError("exact family counts must sum to requested documents")

        if not isinstance(self.strata_exact, Mapping):
            raise ValueError("strata_exact must be a mapping")
        strata_exact: dict[str, Mapping[str, int]] = {}
        for field, values in self.strata_exact.items():
            if not isinstance(field, str) or not field.strip():
                raise ValueError("strata_exact fields must be non-empty strings")
            validated = _validated_counts(values, f"strata_exact.{field}")
            if not validated:
                raise ValueError(f"strata_exact.{field} must not be empty")
            if sum(validated.values()) != self.requested_documents:
                raise ValueError(
                    f"exact stratum counts for {field!r} must sum to requested documents"
                )
            strata_exact[field] = MappingProxyType(validated)

        context_minimums = _validated_counts(self.context_minimums, "context_minimums")
        for context, minimum in context_minimums.items():
            if minimum > self.requested_documents:
                raise ValueError(
                    f"context minimum for {context!r} must not exceed requested_documents"
                )
        object.__setattr__(self, "family_exact", MappingProxyType(family_exact))
        object.__setattr__(self, "strata_exact", MappingProxyType(strata_exact))
        object.__setattr__(self, "context_minimums", MappingProxyType(context_minimums))


@dataclass(frozen=True, slots=True)
class SelectionAssignment:
    position: int
    document_id: str
    family: str


@dataclass(frozen=True, slots=True)
class SelectionResult:
    assignments: tuple[SelectionAssignment, ...]
    feasibility: dict[str, Any]
    solver_receipt: dict[str, Any]


def _positive_integer(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _validated_counts(values: Mapping[str, int], label: str) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping")
    output: dict[str, int] = {}
    for name, count in values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{label}.{name} must be a non-negative integer")
        output[name] = count
    return output


def _stable_ranks(document_ids: Sequence[str], seed: int) -> dict[str, int]:
    ordered = sorted(
        document_ids,
        key=lambda document_id: identity_sha256("synthesis-selection-rank-v1", seed, document_id),
    )
    return {document_id: rank for rank, document_id in enumerate(ordered)}


def solve_selection(
    candidates: Sequence[SelectionCandidate], request: SelectionRequest
) -> SelectionResult:
    """Solve exact family/stratum requirements without weakening infeasible constraints."""

    if not candidates:
        raise ValueError("selection candidate inventory is empty")
    document_ids = [row.document_id for row in candidates]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("selection candidates contain duplicate document IDs")
    source_rows = [row.source_row_index for row in candidates]
    if len(source_rows) != len(set(source_rows)):
        raise ValueError("selection candidates contain duplicate source row indexes")
    maximum_risk = max(row.risk_score for row in candidates)
    if maximum_risk * request.requested_documents > 2**52:
        raise ValueError("selection risk objective exceeds exact float64 integer range")

    pairs = [
        (candidate, family)
        for candidate in sorted(candidates, key=lambda row: row.document_id)
        for family in sorted(candidate.eligible_families)
        if family in request.family_exact
    ]
    if not pairs:
        raise ValueError("no candidate supports a requested mutation family")
    carriers = sorted({row.carrier_family for row in candidates})
    carrier_offset = len(pairs)
    carrier_index = {name: carrier_offset + index for index, name in enumerate(carriers)}
    variable_count = len(pairs) + len(carriers)

    sparse_rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []
    labels: list[str] = []

    def constrain(coefficients: Mapping[int, float], lb: float, ub: float, label: str) -> None:
        for index, value in coefficients.items():
            if not 0 <= index < variable_count:
                raise RuntimeError(f"constraint {label!r} references an invalid variable")
            if not math.isfinite(value) or value == 0:
                raise RuntimeError(f"constraint {label!r} has an invalid coefficient")
        sparse_rows.append(dict(sorted(coefficients.items())))
        lower.append(lb)
        upper.append(ub)
        labels.append(label)

    all_pair_indexes = range(len(pairs))
    constrain(
        {index: 1.0 for index in all_pair_indexes},
        request.requested_documents,
        request.requested_documents,
        "total_documents",
    )
    for document_id in sorted(document_ids):
        indexes = [
            index for index, (row, _family) in enumerate(pairs) if row.document_id == document_id
        ]
        constrain({index: 1.0 for index in indexes}, 0.0, 1.0, f"document:{document_id}")
    for family, exact in sorted(request.family_exact.items()):
        indexes = [index for index, (_row, value) in enumerate(pairs) if value == family]
        constrain({index: 1.0 for index in indexes}, exact, exact, f"family:{family}")
    for field, values in sorted(request.strata_exact.items()):
        for value, exact in sorted(values.items()):
            indexes = [
                index
                for index, (row, _family) in enumerate(pairs)
                if row.strata.get(field) == value
            ]
            constrain({index: 1.0 for index in indexes}, exact, exact, f"stratum:{field}:{value}")
    for context, minimum in sorted(request.context_minimums.items()):
        indexes = [index for index, (row, _family) in enumerate(pairs) if context in row.contexts]
        constrain(
            {index: 1.0 for index in indexes},
            minimum,
            np.inf,
            f"context:{context}",
        )
    for template_id in sorted({row.template_id for row in candidates}):
        indexes = [
            index for index, (row, _family) in enumerate(pairs) if row.template_id == template_id
        ]
        constrain(
            {index: 1.0 for index in indexes},
            0.0,
            request.maximum_per_template,
            f"template:{template_id}",
        )
    for carrier in carriers:
        indexes = [
            index for index, (row, _family) in enumerate(pairs) if row.carrier_family == carrier
        ]
        used_index = carrier_index[carrier]
        constrain(
            {**{index: 1.0 for index in indexes}, used_index: -request.maximum_per_carrier},
            -np.inf,
            0.0,
            f"carrier_link_upper:{carrier}",
        )
        constrain(
            {**{index: -1.0 for index in indexes}, used_index: 1.0},
            -np.inf,
            0.0,
            f"carrier_link_lower:{carrier}",
        )
    constrain(
        {index: 1.0 for index in carrier_index.values()},
        request.minimum_carriers,
        np.inf,
        "minimum_carriers",
    )

    matrix_row_indexes: list[int] = []
    matrix_column_indexes: list[int] = []
    matrix_values: list[float] = []
    for row_index, coefficients in enumerate(sparse_rows):
        for column_index, coefficient in coefficients.items():
            matrix_row_indexes.append(row_index)
            matrix_column_indexes.append(column_index)
            matrix_values.append(coefficient)
    matrix = coo_matrix(
        (matrix_values, (matrix_row_indexes, matrix_column_indexes)),
        shape=(len(sparse_rows), variable_count),
        dtype=np.float64,
    ).tocsr()
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    integrality = np.ones(variable_count, dtype=np.int32)
    base_constraints: list[LinearConstraint] = [
        LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    ]
    risk = np.zeros(variable_count, dtype=np.float64)
    for index, (candidate, _family) in enumerate(pairs):
        risk[index] = candidate.risk_score
    first = milp(
        c=risk,
        integrality=integrality,
        bounds=bounds,
        constraints=base_constraints,
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not first.success or first.x is None:
        ceilings = _support_ceilings(candidates, request)
        raise ValueError(
            f"selection MILP is infeasible or non-optimal: status={first.status}, "
            f"message={first.message}, ceilings={ceilings}"
        )
    first_gap = _optimal_result_gap(first, "risk")
    _validate_binary_solution(first.x, variable_count)
    first_objective = float(first.fun)
    optimum_risk = round(first_objective)
    if not math.isclose(first_objective, optimum_risk, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError("risk MILP returned a non-integral objective")

    ranks = _stable_ranks(document_ids, request.seed)
    pair_ranks = {
        pair: rank
        for rank, pair in enumerate(
            sorted(
                ((candidate.document_id, family) for candidate, family in pairs),
                key=lambda pair: identity_sha256(
                    "synthesis-selection-pair-rank-v1", request.seed, pair[0], pair[1]
                ),
            )
        )
    }
    rank_objective = np.zeros(variable_count, dtype=np.float64)
    for index, (candidate, family) in enumerate(pairs):
        rank_objective[index] = pair_ranks[(candidate.document_id, family)]
    exact_risk = LinearConstraint(csr_matrix(risk.reshape(1, -1)), [optimum_risk], [optimum_risk])
    second = milp(
        c=rank_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[*base_constraints, exact_risk],
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not second.success or second.x is None:
        raise RuntimeError("deterministic tie-break MILP failed after an optimal risk solve")
    second_gap = _optimal_result_gap(second, "tie-break")
    _validate_binary_solution(second.x, variable_count)
    second_objective = float(second.fun)
    rounded_second_objective = round(second_objective)
    if not math.isclose(second_objective, rounded_second_objective, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError("tie-break MILP returned a non-integral objective")
    selected = [pairs[index] for index, value in enumerate(second.x[: len(pairs)]) if value > 0.5]
    if len(selected) != request.requested_documents:
        raise RuntimeError("MILP result does not contain the requested assignment count")
    _verify_selected(selected, request)
    selected.sort(key=lambda item: (ranks[item[0].document_id], item[1], item[0].document_id))
    assignments = tuple(
        SelectionAssignment(position=index, document_id=row.document_id, family=family)
        for index, (row, family) in enumerate(selected, start=1)
    )
    achieved = _achieved(selected)
    feasibility = {
        "requested_documents": request.requested_documents,
        "achieved": achieved,
        "support_ceilings": _support_ceilings(candidates, request),
        "constraint_count": len(labels) + 1,
        "all_constraints_satisfied": True,
    }
    candidate_payload = [
        _candidate_payload(row) for row in sorted(candidates, key=lambda item: item.document_id)
    ]
    request_payload = _request_payload(request)
    variable_payload = [
        {"kind": "assignment", "document_id": candidate.document_id, "family": family}
        for candidate, family in pairs
    ] + [{"kind": "carrier_used", "carrier_family": carrier} for carrier in carriers]
    constraint_payload = [
        {
            "label": label,
            "lower": _finite_bound(lb),
            "upper": _finite_bound(ub),
            "coefficients": [[index, value] for index, value in coefficients.items()],
        }
        for label, lb, ub, coefficients in zip(labels, lower, upper, sparse_rows, strict=True)
    ]
    constraint_payload.append(
        {
            "label": "risk_objective_exact",
            "lower": optimum_risk,
            "upper": optimum_risk,
            "coefficients": [[index, int(value)] for index, value in enumerate(risk) if value != 0],
        }
    )
    matrix_payload = {
        "variables": variable_payload,
        "constraints": constraint_payload,
        "bounds": {"lower": 0, "upper": 1},
        "integrality": "binary",
    }
    problem_payload = {
        "candidates": candidate_payload,
        "request": request_payload,
        "matrix": matrix_payload,
        "risk_objective": [int(value) for value in risk],
        "tie_break_objective": [int(value) for value in rank_objective],
    }
    solver_receipt = {
        "solver": "scipy.optimize.milp_highs",
        "scipy_version": scipy.__version__,
        "candidate_inventory_sha256": sha256_bytes(canonical_json_bytes(candidate_payload)),
        "request_sha256": sha256_bytes(canonical_json_bytes(request_payload)),
        "matrix_sha256": sha256_bytes(canonical_json_bytes(matrix_payload)),
        "problem_sha256": sha256_bytes(canonical_json_bytes(problem_payload)),
        "candidate_count": len(candidates),
        "assignment_variable_count": len(pairs),
        "binary_variable_count": variable_count,
        "constraint_labels": [*labels, "risk_objective_exact"],
        "risk_status": int(first.status),
        "risk_message": str(first.message),
        "risk_objective": optimum_risk,
        "risk_mip_gap": first_gap,
        "tie_break_status": int(second.status),
        "tie_break_message": str(second.message),
        "tie_break_objective": rounded_second_objective,
        "tie_break_mip_gap": second_gap,
        "seed": request.seed,
    }
    return SelectionResult(assignments, feasibility, solver_receipt)


def _optimal_result_gap(result: Any, label: str) -> float:
    if result.status != 0 or not result.success:
        raise RuntimeError(f"{label} MILP did not report an optimal result")
    if result.fun is None or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"{label} MILP did not return a finite objective")
    gap_value = result.get("mip_gap")
    if gap_value is None or not math.isfinite(float(gap_value)):
        raise RuntimeError(f"{label} MILP did not report a finite MIP gap")
    gap = float(gap_value)
    if gap != 0.0:
        raise RuntimeError(f"{label} MILP returned a non-zero MIP gap: {gap}")
    return gap


def _validate_binary_solution(values: np.ndarray, expected_size: int) -> None:
    if values.shape != (expected_size,) or not np.all(np.isfinite(values)):
        raise RuntimeError("MILP returned a malformed solution vector")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-7):
        raise RuntimeError("MILP returned a non-integral solution vector")
    if np.any((rounded < 0) | (rounded > 1)):
        raise RuntimeError("MILP returned a solution outside its binary bounds")


def _verify_selected(
    selected: Sequence[tuple[SelectionCandidate, str]], request: SelectionRequest
) -> None:
    if len(selected) != request.requested_documents:
        raise RuntimeError("selection verification found an incorrect document count")
    document_ids = [row.document_id for row, _family in selected]
    if len(document_ids) != len(set(document_ids)):
        raise RuntimeError("selection verification found a duplicate document")
    for candidate, family in selected:
        if family not in candidate.eligible_families or family not in request.family_exact:
            raise RuntimeError("selection verification found an ineligible family assignment")
    family_counts = Counter(family for _row, family in selected)
    if any(family_counts[family] != exact for family, exact in request.family_exact.items()):
        raise RuntimeError("selection verification found incorrect exact family counts")
    for field, values in request.strata_exact.items():
        counts = Counter(row.strata.get(field) for row, _family in selected)
        if any(counts[value] != exact for value, exact in values.items()):
            raise RuntimeError(
                f"selection verification found incorrect exact stratum counts for {field!r}"
            )
    for context, minimum in request.context_minimums.items():
        count = sum(context in row.contexts for row, _family in selected)
        if count < minimum:
            raise RuntimeError(f"selection verification found insufficient context {context!r}")
    template_counts = Counter(row.template_id for row, _family in selected)
    if max(template_counts.values(), default=0) > request.maximum_per_template:
        raise RuntimeError("selection verification found an exceeded template cap")
    carrier_counts = Counter(row.carrier_family for row, _family in selected)
    if max(carrier_counts.values(), default=0) > request.maximum_per_carrier:
        raise RuntimeError("selection verification found an exceeded carrier cap")
    if len(carrier_counts) < request.minimum_carriers:
        raise RuntimeError("selection verification found insufficient carrier diversity")


def _candidate_payload(candidate: SelectionCandidate) -> dict[str, Any]:
    return {
        "document_id": candidate.document_id,
        "source_row_index": candidate.source_row_index,
        "template_id": candidate.template_id,
        "template_size": candidate.template_size,
        "carrier_family": candidate.carrier_family,
        "strata": dict(sorted(candidate.strata.items())),
        "contexts": sorted(candidate.contexts),
        "eligible_families": sorted(candidate.eligible_families),
        "risk_score": candidate.risk_score,
    }


def _request_payload(request: SelectionRequest) -> dict[str, Any]:
    return {
        "requested_documents": request.requested_documents,
        "family_exact": dict(sorted(request.family_exact.items())),
        "strata_exact": {
            field: dict(sorted(values.items()))
            for field, values in sorted(request.strata_exact.items())
        },
        "context_minimums": dict(sorted(request.context_minimums.items())),
        "maximum_per_template": request.maximum_per_template,
        "maximum_per_carrier": request.maximum_per_carrier,
        "minimum_carriers": request.minimum_carriers,
        "seed": request.seed,
    }


def _finite_bound(value: float) -> float | str:
    if value == np.inf:
        return "positive_infinity"
    if value == -np.inf:
        return "negative_infinity"
    if not math.isfinite(value):
        raise RuntimeError("constraint has an unsupported non-finite bound")
    return value


def _support_ceilings(
    candidates: Sequence[SelectionCandidate], request: SelectionRequest
) -> dict[str, int]:
    output = {
        f"family:{family}": sum(family in row.eligible_families for row in candidates)
        for family in request.family_exact
    }
    for field, values in request.strata_exact.items():
        for value in values:
            output[f"stratum:{field}:{value}"] = sum(
                row.strata.get(field) == value for row in candidates
            )
    for context in request.context_minimums:
        output[f"context:{context}"] = sum(context in row.contexts for row in candidates)
    output["carriers"] = len({row.carrier_family for row in candidates})
    output["templates"] = len({row.template_id for row in candidates})
    return dict(sorted(output.items()))


def _achieved(selected: Sequence[tuple[SelectionCandidate, str]]) -> dict[str, Any]:
    return {
        "documents": len(selected),
        "families": dict(sorted(Counter(family for _row, family in selected).items())),
        "strata": {
            field: dict(sorted(Counter(row.strata[field] for row, _family in selected).items()))
            for field in sorted({field for row, _family in selected for field in row.strata})
        },
        "contexts": dict(
            sorted(
                Counter(context for row, _family in selected for context in row.contexts).items()
            )
        ),
        "carriers": len({row.carrier_family for row, _family in selected}),
        "templates": len({row.template_id for row, _family in selected}),
    }
