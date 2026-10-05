"""One-to-one, order-independent JSON list matching for extraction metrics.

Only the scoring view is aligned; serialized predictions and training labels are
unchanged. Rows remain whole, nested lists stay with their parent, and unmatched
occurrences occupy separate positions so duplicate/missing rows are not erased.
"""

from __future__ import annotations

from typing import Any

# A scoring-only gap, distinct from an explicitly predicted JSON null.
UNMATCHED = object()

# Prefer an exact entity identity over coincidentally equal attributes elsewhere.
# Description is the goods identity when the target has no explicit goods ID.
_IDENTITY_FIELDS = (
    "equipmentIdentifier",
    "containerNumber",
    "packageId",
    "groupId",
    "unNumber",
    "description",
    "name",
)


def unordered_key(value: Any) -> tuple[Any, ...]:
    """Hashable JSON identity with multiset (not set) semantics for every list."""

    if isinstance(value, dict):
        return (0, tuple((key, unordered_key(child)) for key, child in sorted(value.items())))
    if isinstance(value, list):
        return (1, tuple(sorted(unordered_key(child) for child in value)))
    # JSON distinguishes true from 1, and the existing scalar metric distinguishes
    # 1.0 from 1. Keep those distinctions even for schema-invalid predictions.
    return (2, type(value).__name__, repr(value) if isinstance(value, float) else value)


def _identity(value: Any) -> tuple[str, str] | None:
    if isinstance(value, dict):
        for field in _IDENTITY_FIELDS:
            if isinstance(value.get(field), str):
                return field, value[field]
    return None


def _scalar_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_scalar_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_scalar_count(child) for child in value)
    return 1


def _maximum_weight_pairs(weights: list[list[int]]) -> list[tuple[int, int]]:
    """Exact rectangular assignment via shortest augmenting paths (Hungarian).

    Every row of the smaller side gets one distinct partner. Callers put rows in
    content order first, making ties independent of their original list order.
    Complexity is O(min(n,m)**2 * max(n,m)); no factorial permutation search.
    """

    if not weights or not weights[0]:
        return []
    rows, columns = len(weights), len(weights[0])
    if rows > columns:
        return [
            (j, i) for i, j in _maximum_weight_pairs([list(x) for x in zip(*weights, strict=True)])
        ]
    row_potential = [0] * (rows + 1)
    col_potential = [0] * (columns + 1)
    owner = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row in range(1, rows + 1):
        owner[0] = row
        column = 0
        distance = [float("inf")] * (columns + 1)
        visited = [False] * (columns + 1)
        while True:
            visited[column] = True
            current_row = owner[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, columns + 1):
                if visited[candidate]:
                    continue
                cost = (
                    -weights[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - col_potential[candidate]
                )
                if cost < distance[candidate]:
                    distance[candidate] = cost
                    predecessor[candidate] = column
                if distance[candidate] < delta:
                    delta = distance[candidate]
                    next_column = candidate
            # All costs are finite integers and at least one column is unvisited.
            step = int(delta)
            for candidate in range(columns + 1):
                if visited[candidate]:
                    row_potential[owner[candidate]] += step
                    col_potential[candidate] -= step
                else:
                    distance[candidate] -= step
            column = next_column
            if owner[column] == 0:
                break
        while column:
            previous = predecessor[column]
            owner[column] = owner[previous]
            column = previous
    return [(owner[j] - 1, j - 1) for j in range(1, columns + 1) if owner[j]]


def align_list_items(predicted: Any, reference: Any) -> tuple[Any, Any, int]:
    """Align lists for exact leaf scoring, with identities taking precedence.

    Reference indices are retained for diagnostic paths. Unmatched predictions
    follow the reference positions; gaps never contribute fields or relations.
    For rows without matching identities, maximize exact nested leaf agreement.
    No fuzzy text, value normalization, or movement across parent fields occurs.
    """

    if isinstance(predicted, dict) and isinstance(reference, dict):
        if unordered_key(predicted) == unordered_key(reference):
            return reference, reference, _scalar_count(reference)
        aligned_predicted_dict, aligned_reference_dict = predicted.copy(), reference.copy()
        score = 0
        for key in predicted.keys() & reference.keys():
            left, right, child_score = align_list_items(predicted[key], reference[key])
            aligned_predicted_dict[key], aligned_reference_dict[key] = left, right
            score += child_score
        return aligned_predicted_dict, aligned_reference_dict, score
    if isinstance(predicted, list) and isinstance(reference, list):
        if not predicted:
            return [UNMATCHED] * len(reference), reference, 0
        if not reference:
            return predicted, [UNMATCHED] * len(predicted), 0
        if len(predicted) == len(reference) == 1:
            left, right, score = align_list_items(predicted[0], reference[0])
            return [left], [right], score
        pred_keys = [unordered_key(row) for row in predicted]
        ref_keys = [unordered_key(row) for row in reference]
        pred_order = sorted(range(len(predicted)), key=pred_keys.__getitem__)
        ref_order = sorted(range(len(reference)), key=ref_keys.__getitem__)
        pairs = [
            [
                (reference[j], reference[j], _scalar_count(reference[j]))
                if pred_keys[i] == ref_keys[j]
                else align_list_items(predicted[i], reference[j])
                for j in ref_order
            ]
            for i in pred_order
        ]
        # One additional identity match dominates the entire leaf-match objective.
        identity_weight = 1 + sum(max(pair[2] for pair in row) for row in pairs)
        weights = [
            [
                pairs[i][j][2]
                + identity_weight
                * int(
                    _identity(predicted[pi]) is not None
                    and _identity(predicted[pi]) == _identity(reference[rj])
                )
                for j, rj in enumerate(ref_order)
            ]
            for i, pi in enumerate(pred_order)
        ]
        aligned_predicted: list[Any] = [UNMATCHED] * len(reference)
        aligned_reference: list[Any] = reference.copy()
        used_predicted: set[int] = set()
        score = 0
        for i, j in _maximum_weight_pairs(weights):
            left, right, child_score = pairs[i][j]
            aligned_predicted[ref_order[j]] = left
            aligned_reference[ref_order[j]] = right
            used_predicted.add(pred_order[i])
            score += child_score
        for index in pred_order:
            if index not in used_predicted:
                aligned_predicted.append(predicted[index])
                aligned_reference.append(UNMATCHED)
        return aligned_predicted, aligned_reference, score
    if isinstance(predicted, (list, dict)) or isinstance(reference, (list, dict)):
        return predicted, reference, 0
    return predicted, reference, int(unordered_key(predicted) == unordered_key(reference))
