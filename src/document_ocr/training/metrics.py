"""Structured generation metrics for sparse KIE decoder targets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from document_ocr.label_schemas.bill_of_lading_v3 import RelationExplicitDocumentPatch
from document_ocr.label_schemas.bill_of_lading_v4 import RelationExplicitDocumentPatchV4
from document_ocr.label_schemas.bill_of_lading_v5 import RelationExplicitDocumentPatchV5
from document_ocr.label_schemas.bill_of_lading_v6 import BillOfLadingDocumentPatchV6
from document_ocr.training.tasks import TrainingTask, canonical_json


class DecoderTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class PredictionAssessment:
    generated_text: str
    reference_text: str
    json_valid: bool
    schema_valid: bool
    canonical_exact_match: bool
    predicted_field_values: frozenset[tuple[str, str]]
    reference_field_values: frozenset[tuple[str, str]]
    predicted_cargo_relation_facts: frozenset[tuple[str, ...]]
    reference_cargo_relation_facts: frozenset[tuple[str, ...]]
    predicted_category_values: frozenset[tuple[str, ...]]
    reference_category_values: frozenset[tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_text": self.generated_text,
            "reference_text": self.reference_text,
            "json_valid": self.json_valid,
            "schema_valid": self.schema_valid,
            "canonical_exact_match": self.canonical_exact_match,
        }


@dataclass(frozen=True, slots=True)
class _RelationMetricProfile:
    supported: bool
    dangerous_goods_categories: bool
    container_size_category: bool
    extraction_facts: bool
    mpci_aligned_v6: bool


def _relation_metric_profile(task: TrainingTask) -> _RelationMetricProfile:
    """Derive metric coverage from the registered target schema, not its task name."""

    patch_model = task.target_model.model_fields["documentPatch"].annotation
    if not isinstance(patch_model, type):
        raise TypeError("training target documentPatch annotation must be a model class")
    v5_schema = issubclass(patch_model, RelationExplicitDocumentPatchV5)
    v6_schema = issubclass(patch_model, BillOfLadingDocumentPatchV6)
    return _RelationMetricProfile(
        supported=issubclass(patch_model, RelationExplicitDocumentPatch) or v6_schema,
        dangerous_goods_categories=(
            issubclass(patch_model, RelationExplicitDocumentPatchV4) or v6_schema
        ),
        container_size_category=v5_schema or v6_schema,
        extraction_facts=v5_schema or v6_schema,
        mpci_aligned_v6=v6_schema,
    )


def _field_value_items(value: Any, path: str = "$.documentPatch") -> set[tuple[str, str]]:
    if isinstance(value, dict):
        field_values: set[tuple[str, str]] = set()
        for key in sorted(value):
            field_values.update(_field_value_items(value[key], f"{path}.{key}"))
        return field_values
    if isinstance(value, list):
        field_values = set()
        for index, child in enumerate(value):
            field_values.update(_field_value_items(child, f"{path}[{index}]"))
        return field_values
    return {
        (
            path,
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    }


def _document_patch(value: dict[str, Any]) -> dict[str, Any]:
    patch = value.get("documentPatch")
    if not isinstance(patch, dict):
        raise ValueError("canonical decoder target must contain a documentPatch object")
    return patch


def _optional_document_patch(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    patch = value.get("documentPatch")
    return patch if isinstance(patch, dict) else None


def _relation_explicit_facts(
    patch: dict[str, Any] | None,
    *,
    profile: _RelationMetricProfile,
) -> tuple[frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]]:
    """Project cargo edges and anchored categories without container array positions."""

    if patch is None:
        return frozenset(), frozenset()
    relations: set[tuple[str, ...]] = set()
    categories: set[tuple[str, ...]] = set()

    containers = patch.get("containers")
    if isinstance(containers, list):
        for container in containers:
            if not isinstance(container, dict):
                continue
            number = container.get("containerNumber")
            category = container.get("typeCategory")
            if isinstance(number, str) and isinstance(category, str):
                categories.add(("container_type", number, category))
            size = container.get("sizeCategory")
            if (
                profile.container_size_category
                and isinstance(number, str)
                and isinstance(size, str)
            ):
                categories.add(("container_size", number, size))

    if profile.dangerous_goods_categories:
        groups = patch.get("cargoGroups")
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, dict) or not isinstance(group.get("groupId"), str):
                    continue
                dangerous_goods = group.get("dangerousGoods")
                if not isinstance(dangerous_goods, list):
                    continue
                for index, dangerous in enumerate(dangerous_goods):
                    if not isinstance(dangerous, dict):
                        continue
                    un_number = dangerous.get("unNumber")
                    anchor = un_number if isinstance(un_number, str) else f"index:{index}"
                    for field in ("hazardCategory", "packingGroupCategory"):
                        value = dangerous.get(field)
                        if isinstance(value, str):
                            categories.add((field, group["groupId"], anchor, value))
                    subsidiaries = dangerous.get("subsidiaryHazardCategories")
                    if isinstance(subsidiaries, list):
                        for value in subsidiaries:
                            if isinstance(value, str):
                                categories.add(
                                    ("subsidiaryHazardCategory", group["groupId"], anchor, value)
                                )

    packages = patch.get("cargoPackages")
    if isinstance(packages, list):
        for package in packages:
            if not isinstance(package, dict):
                continue
            group_id = package.get("groupId")
            package_id = package.get("packageId")
            if isinstance(group_id, str) and isinstance(package_id, str):
                relations.add(("group_has_package", group_id, package_id))
                category = package.get("typeCategory")
                if isinstance(category, str):
                    categories.add(("package_type", group_id, package_id, category))

    allocation_groups = patch.get("cargoAllocationGroups")
    if isinstance(allocation_groups, list):
        for allocation_group in allocation_groups:
            if not isinstance(allocation_group, dict):
                continue
            group_id = allocation_group.get("groupId")
            coverage = allocation_group.get("coverage")
            if not isinstance(group_id, str):
                continue
            if isinstance(coverage, str):
                relations.add(("allocation_coverage", group_id, coverage))
            package_ids = allocation_group.get("packageIds")
            if isinstance(package_ids, list):
                for package_id in package_ids:
                    if isinstance(package_id, str):
                        relations.add(("allocation_covers_package", group_id, package_id))
            allocations = allocation_group.get("allocations")
            if not isinstance(allocations, list):
                continue
            for allocation in allocations:
                if not isinstance(allocation, dict):
                    continue
                container_number = allocation.get("containerNumber")
                if not isinstance(container_number, str):
                    continue
                relations.add(("group_uses_container", group_id, container_number))
                quantity = allocation.get("packageQuantity")
                if isinstance(quantity, int) and not isinstance(quantity, bool):
                    relations.add(
                        (
                            "container_package_quantity",
                            group_id,
                            container_number,
                            str(quantity),
                        )
                    )
                package_id = allocation.get("packageId")
                if isinstance(package_id, str):
                    relations.add(("container_has_package", group_id, container_number, package_id))
    return frozenset(relations), frozenset(categories)


def _mpci_aligned_facts(
    patch: dict[str, Any] | None,
) -> tuple[frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]]:
    """Score nested goods-to-equipment edges without redundant coverage labels."""

    if patch is None:
        return frozenset(), frozenset()
    relations: set[tuple[str, ...]] = set()
    categories: set[tuple[str, ...]] = set()
    containers = patch.get("containerInformation")
    if not isinstance(containers, list):
        containers = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        identifier = container.get("equipmentIdentifier")
        if not isinstance(identifier, str):
            continue
        for field, metric_name in (
            ("sizeCategory", "container_size"),
            ("typeCategory", "container_type"),
        ):
            value = container.get(field)
            if isinstance(value, str):
                categories.add((metric_name, identifier, value))
    goods_items = patch.get("goodsItemDetails")
    if not isinstance(goods_items, list):
        goods_items = []
    for goods_index, goods in enumerate(goods_items):
        if not isinstance(goods, dict):
            continue
        anchor = str(goods_index)
        packages = goods.get("numberAndTypeOfPackages")
        if not isinstance(packages, list):
            packages = []
        for package_index, package in enumerate(packages):
            if isinstance(package, dict) and isinstance(package.get("typeCategory"), str):
                categories.add(
                    ("package_type", anchor, str(package_index), package["typeCategory"])
                )
        dangerous_goods = goods.get("dangerousGoods")
        if not isinstance(dangerous_goods, list):
            dangerous_goods = []
        for dangerous_index, dangerous in enumerate(dangerous_goods):
            if not isinstance(dangerous, dict):
                continue
            un_number = dangerous.get("unNumber")
            dangerous_anchor = (
                un_number if isinstance(un_number, str) else f"index:{dangerous_index}"
            )
            for field in ("hazardCategory", "packingGroupCategory"):
                value = dangerous.get(field)
                if isinstance(value, str):
                    categories.add((field, anchor, dangerous_anchor, value))
            subsidiaries = dangerous.get("subsidiaryHazardCategories")
            if not isinstance(subsidiaries, list):
                subsidiaries = []
            for subsidiary in subsidiaries:
                if isinstance(subsidiary, str):
                    categories.add(
                        ("subsidiaryHazardCategory", anchor, dangerous_anchor, subsidiary)
                    )
        placements = goods.get("splitGoodsPlacement")
        if not isinstance(placements, list):
            placements = []
        for placement_index, placement in enumerate(placements):
            if not isinstance(placement, dict):
                continue
            identifier = placement.get("equipmentIdentifier")
            if not isinstance(identifier, str):
                continue
            relations.add(("goods_uses_container", anchor, identifier))
            quantity = placement.get("packageQuantity")
            if isinstance(quantity, int) and not isinstance(quantity, bool):
                # The row index distinguishes two printed placements into one container.
                relations.add(
                    (
                        "container_package_quantity",
                        anchor,
                        str(placement_index),
                        identifier,
                        str(quantity),
                    )
                )
    return frozenset(relations), frozenset(categories)


def _is_relation_or_scaffold_path(path: str) -> bool:
    if path.startswith("$.documentPatch.goodsItemDetails[") and ".splitGoodsPlacement[" in path:
        return True
    if path.startswith("$.documentPatch.cargoAllocationGroups["):
        return True
    if path.startswith("$.documentPatch.cargoGroups[") and path.endswith(".groupId"):
        return True
    return path.startswith("$.documentPatch.cargoPackages[") and path.endswith(
        (".groupId", ".packageId")
    )


def _micro_set_metrics(
    predicted_sets: Sequence[frozenset[tuple[str, ...]]],
    reference_sets: Sequence[frozenset[tuple[str, ...]]],
) -> tuple[float, float, float, float]:
    if len(predicted_sets) != len(reference_sets):
        raise ValueError("predicted and reference fact-set counts differ")
    true_positive = predicted_total = reference_total = 0
    for predicted, reference in zip(predicted_sets, reference_sets, strict=True):
        true_positive += len(predicted & reference)
        predicted_total += len(predicted)
        reference_total += len(reference)
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Accuracy is micro Jaccard, not document exact match: no true-negative universe
    # exists for sparse facts and multi-valued category/relation slots are permitted.
    compared_total = predicted_total + reference_total - true_positive
    accuracy = true_positive / compared_total if compared_total else 1.0
    return precision, recall, f1, accuracy


def assess_prediction(
    generated_text: str,
    reference_text: str,
    task: TrainingTask,
) -> PredictionAssessment:
    """Parse, schema-check, and compare one generated sparse target."""

    return _assess_prediction(generated_text, reference_text, task, _relation_metric_profile(task))


def _assess_prediction(
    generated_text: str,
    reference_text: str,
    task: TrainingTask,
    profile: _RelationMetricProfile,
) -> PredictionAssessment:
    reference_value = json.loads(reference_text)
    if not isinstance(reference_value, dict):
        raise ValueError("reference decoder target must be a JSON object")
    reference = task.canonicalize(cast(dict[str, Any], reference_value))
    reference_canonical = canonical_json(reference)

    try:
        parsed = json.loads(generated_text)
    except json.JSONDecodeError:
        parsed = None
    json_valid = isinstance(parsed, dict)
    parsed_object = cast(dict[str, Any], parsed) if json_valid else None
    predicted: dict[str, Any] | None = None
    if parsed_object is not None:
        try:
            predicted = task.canonicalize(parsed_object)
        except ValueError:
            predicted = None
    schema_valid = predicted is not None
    generated_canonical = canonical_json(predicted) if predicted is not None else None
    # Exact field-value scoring remains informative when one unrelated field makes an
    # otherwise parseable prediction schema-invalid. Schema validity is reported separately.
    predicted_patch = _optional_document_patch(
        predicted if predicted is not None else parsed_object
    )
    if profile.supported:
        fact_projector = _mpci_aligned_facts if profile.mpci_aligned_v6 else None
        if fact_projector is not None:
            predicted_relations, predicted_categories = fact_projector(predicted_patch)
            reference_relations, reference_categories = fact_projector(_document_patch(reference))
        else:
            predicted_relations, predicted_categories = _relation_explicit_facts(
                predicted_patch, profile=profile
            )
            reference_relations, reference_categories = _relation_explicit_facts(
                _document_patch(reference), profile=profile
            )
    else:
        predicted_relations = reference_relations = frozenset()
        predicted_categories = reference_categories = frozenset()
    return PredictionAssessment(
        generated_text=generated_text,
        reference_text=reference_canonical,
        json_valid=json_valid,
        schema_valid=schema_valid,
        canonical_exact_match=generated_canonical == reference_canonical,
        predicted_field_values=(
            frozenset(_field_value_items(predicted_patch))
            if predicted_patch is not None
            else frozenset()
        ),
        reference_field_values=frozenset(_field_value_items(_document_patch(reference))),
        predicted_cargo_relation_facts=predicted_relations,
        reference_cargo_relation_facts=reference_relations,
        predicted_category_values=predicted_categories,
        reference_category_values=reference_categories,
    )


def structured_metrics(
    generated_texts: Sequence[str],
    reference_texts: Sequence[str],
    task: TrainingTask,
) -> tuple[dict[str, float], list[PredictionAssessment]]:
    """Compute document validity/exactness and micro exact field-value metrics."""

    if len(generated_texts) != len(reference_texts):
        raise ValueError("generated and reference sequence counts differ")
    if not generated_texts:
        raise ValueError("structured metrics require at least one prediction")
    profile = _relation_metric_profile(task)
    assessments = [
        _assess_prediction(generated, reference, task, profile)
        for generated, reference in zip(generated_texts, reference_texts, strict=True)
    ]
    true_positive = sum(
        len(item.predicted_field_values & item.reference_field_values) for item in assessments
    )
    predicted_total = sum(len(item.predicted_field_values) for item in assessments)
    reference_total = sum(len(item.reference_field_values) for item in assessments)
    compared_field_total = sum(
        len(
            {path for path, _ in item.predicted_field_values}
            | {path for path, _ in item.reference_field_values}
        )
        for item in assessments
    )
    accuracy = true_positive / compared_field_total if compared_field_total else 1.0
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count = len(assessments)
    metrics = {
        "json_valid": sum(item.json_valid for item in assessments) / count,
        "schema_valid": sum(item.schema_valid for item in assessments) / count,
        "canonical_exact_match": (sum(item.canonical_exact_match for item in assessments) / count),
        "field_value_accuracy": accuracy,
        "field_value_precision": precision,
        "field_value_recall": recall,
        "field_value_f1": f1,
    }
    if profile.supported:
        relation_precision, relation_recall, relation_f1, relation_accuracy = _micro_set_metrics(
            [row.predicted_cargo_relation_facts for row in assessments],
            [row.reference_cargo_relation_facts for row in assessments],
        )
        category_precision, category_recall, category_f1, category_accuracy = _micro_set_metrics(
            [row.predicted_category_values for row in assessments],
            [row.reference_category_values for row in assessments],
        )
        metrics.update(
            {
                "cargo_relation_precision": relation_precision,
                "cargo_relation_recall": relation_recall,
                "cargo_relation_f1": relation_f1,
                "cargo_relation_accuracy": relation_accuracy,
                "cargo_relation_exact_match": sum(
                    row.predicted_cargo_relation_facts == row.reference_cargo_relation_facts
                    for row in assessments
                )
                / count,
                "cargo_relation_support_fraction": sum(
                    bool(row.reference_cargo_relation_facts) for row in assessments
                )
                / count,
                "category_value_precision": category_precision,
                "category_value_recall": category_recall,
                "category_value_f1": category_f1,
                "category_value_accuracy": category_accuracy,
                "category_value_exact_match": sum(
                    row.predicted_category_values == row.reference_category_values
                    for row in assessments
                )
                / count,
                "category_value_support_fraction": sum(
                    bool(row.reference_category_values) for row in assessments
                )
                / count,
            }
        )
        if profile.extraction_facts:
            fact_precision, fact_recall, fact_f1, _ = _micro_set_metrics(
                [
                    frozenset(
                        value
                        for value in row.predicted_field_values
                        if not _is_relation_or_scaffold_path(value[0])
                    )
                    for row in assessments
                ],
                [
                    frozenset(
                        value
                        for value in row.reference_field_values
                        if not _is_relation_or_scaffold_path(value[0])
                    )
                    for row in assessments
                ],
            )
            metrics.update(
                {
                    "extraction_fact_precision": fact_precision,
                    "extraction_fact_recall": fact_recall,
                    "extraction_fact_f1": fact_f1,
                }
            )
    return metrics, assessments


def make_compute_metrics(tokenizer: DecoderTokenizer, task: TrainingTask) -> Any:
    """Create the Transformers Trainer metric callback without retaining model inputs."""

    def compute_metrics(evaluation_prediction: Any) -> Mapping[str, float]:
        predictions = evaluation_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = evaluation_prediction.label_ids
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id for generated evaluation")
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id for generated evaluation")
        # NumPy is already a Trainer dependency; copy so the Trainer-owned array is never mutated.
        labels_for_decode = labels.copy()
        labels_for_decode[labels_for_decode == -100] = pad_token_id
        generated_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        reference_texts = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
        metrics, _ = structured_metrics(generated_texts, reference_texts, task)
        generated_token_counts = (predictions != pad_token_id).sum(axis=1)
        eos_reached = (predictions == eos_token_id).any(axis=1)
        metrics.update(
            {
                "generated_tokens_mean": float(generated_token_counts.mean()),
                "generated_tokens_max": float(generated_token_counts.max()),
                "generation_eos_reached_fraction": float(eos_reached.mean()),
            }
        )
        return metrics

    return compute_metrics
