from __future__ import annotations

import copy
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from document_ocr.training.metric_matching import _maximum_weight_pairs, unordered_key
from document_ocr.training.metrics import make_compute_metrics, structured_metrics
from document_ocr.training.prediction import SamplerAwarePredictionMixin
from document_ocr.training.runtime import _predict_and_publish
from document_ocr.training.tasks import canonical_json, get_training_task


def _target(number: str) -> str:
    return canonical_json(
        {
            "schemaVersion": "2.0.0",
            "documentPatch": {"billOfLadingNumber": number},
        }
    )


def _relation_target(
    *,
    reverse_containers_and_allocations: bool = False,
    allocation_quantities: tuple[int, int] = (10, 20),
) -> str:
    containers = [
        {"containerNumber": "MSKU1200040", "typeCategory": "FORTY_FOOT_DRY"},
        {"containerNumber": "FCIU3651201", "typeCategory": "FORTY_FOOT_HIGH_CUBE_DRY"},
    ]
    allocations = [
        {"containerNumber": "MSKU1200040", "packageQuantity": allocation_quantities[0]},
        {"containerNumber": "FCIU3651201", "packageQuantity": allocation_quantities[1]},
    ]
    if reverse_containers_and_allocations:
        containers.reverse()
        allocations.reverse()
    return canonical_json(
        {
            "schemaVersion": "3.0.0-experimental",
            "documentPatch": {
                "containers": containers,
                "cargoGroups": [{"groupId": "g1", "description": "MACHINERY"}],
                "cargoPackages": [
                    {
                        "packageId": "p1",
                        "groupId": "g1",
                        "quantity": 30,
                        "typeCategory": "CARTON",
                    }
                ],
                "cargoAllocationGroups": [
                    {
                        "groupId": "g1",
                        "coverage": "single_package_level",
                        "packageIds": ["p1"],
                        "allocations": allocations,
                    }
                ],
            },
        }
    )


def _v5_target(*, reverse: bool = False, hazard: str = "FLAMMABLE_LIQUIDS") -> str:
    containers = [
        {
            "containerNumber": "MSKU1200040",
            "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
            "typeCategory": "GENERAL_PURPOSE",
        },
        {
            "containerNumber": "FCIU3651201",
            "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
            "typeCategory": "GENERAL_PURPOSE",
        },
    ]
    allocations = [
        {"containerNumber": "MSKU1200040", "packageQuantity": 10},
        {"containerNumber": "FCIU3651201", "packageQuantity": 20},
    ]
    if reverse:
        containers.reverse()
        allocations.reverse()
    return canonical_json(
        {
            "schemaVersion": "5.0.0-experimental",
            "documentPatch": {
                "containers": containers,
                "cargoGroups": [
                    {
                        "groupId": "g1",
                        "description": "PAINT",
                        "dangerousGoods": [{"unNumber": "1203", "hazardCategory": hazard}],
                    }
                ],
                "cargoPackages": [
                    {
                        "packageId": "p1",
                        "groupId": "g1",
                        "quantity": 30,
                        "typeCategory": "PACKAGE_CARTON",
                    }
                ],
                "cargoAllocationGroups": [
                    {
                        "groupId": "g1",
                        "coverage": "single_package_level",
                        "packageIds": ["p1"],
                        "allocations": allocations,
                    }
                ],
            },
        }
    )


def _v4_target(*, hazard: str = "FLAMMABLE_LIQUIDS") -> str:
    target = json.loads(_relation_target())
    target["schemaVersion"] = "4.0.0-experimental"
    target["documentPatch"]["cargoGroups"][0]["dangerousGoods"] = [
        {
            "unNumber": "1203",
            "hazardCategory": hazard,
            "packingGroupCategory": "LOW_DANGER",
            "subsidiaryHazardCategories": ["GASES", "CORROSIVE_SUBSTANCES"],
        }
    ]
    return canonical_json(target)


def test_structured_metrics_distinguish_json_schema_and_exactness() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    reference = _target("ABC")

    metrics, assessments = structured_metrics(
        [reference, _target("WRONG"), "not json"],
        [reference, reference, reference],
        task,
    )

    assert metrics["json_valid"] == 2 / 3
    assert metrics["schema_valid"] == 2 / 3
    assert metrics["canonical_exact_match"] == 1 / 3
    # schemaVersion is structural metadata, not an extracted field-value success.
    assert metrics["field_value_accuracy"] == 1 / 3
    assert metrics["field_value_precision"] == 1 / 2
    assert metrics["field_value_recall"] == 1 / 3
    assert metrics["field_value_f1"] == 0.4
    assert assessments[0].canonical_exact_match
    assert not assessments[2].json_valid


def test_field_value_metrics_penalize_missing_extra_and_wrong_values() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    reference = canonical_json(
        {
            "schemaVersion": "2.0.0",
            "documentPatch": {
                "billOfLadingNumber": "ABC",
                "originalBillOfLadingNumber": "ORIGINAL",
            },
        }
    )
    prediction = canonical_json(
        {
            "schemaVersion": "2.0.0",
            "documentPatch": {
                "billOfLadingNumber": "ABC",
                "masterBillOfLadingNumber": "EXTRA",
            },
        }
    )

    metrics, _ = structured_metrics([prediction], [reference], task)

    assert metrics["field_value_accuracy"] == 1 / 3
    assert metrics["field_value_precision"] == 1 / 2
    assert metrics["field_value_recall"] == 1 / 2
    assert metrics["field_value_f1"] == 1 / 2


def test_field_value_metrics_require_the_complete_scalar_value() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")

    metrics, _ = structured_metrics([_target("ABX")], [_target("ABC")], task)

    assert metrics["field_value_accuracy"] == 0.0
    assert metrics["field_value_precision"] == 0.0
    assert metrics["field_value_recall"] == 0.0
    assert metrics["field_value_f1"] == 0.0


def test_field_value_metrics_preserve_partial_credit_when_schema_is_invalid() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    prediction = canonical_json(
        {
            "schemaVersion": "WRONG",
            "documentPatch": {"billOfLadingNumber": "ABC"},
        }
    )

    metrics, _ = structured_metrics([prediction], [_target("ABC")], task)

    assert metrics["json_valid"] == 1.0
    assert metrics["schema_valid"] == 0.0
    assert metrics["canonical_exact_match"] == 0.0
    assert metrics["field_value_accuracy"] == 1.0
    assert metrics["field_value_precision"] == 1.0
    assert metrics["field_value_recall"] == 1.0
    assert metrics["field_value_f1"] == 1.0


def test_relation_explicit_metrics_ignore_array_position_but_anchor_entities() -> None:
    task = get_training_task("bill_of_lading_relation_explicit_v3")
    reference = _relation_target()
    prediction = _relation_target(reverse_containers_and_allocations=True)

    metrics, assessments = structured_metrics([prediction], [reference], task)

    assert metrics["canonical_exact_match"] == 1.0
    assert metrics["field_value_f1"] == 1.0
    assert metrics["cargo_relation_precision"] == 1.0
    assert metrics["cargo_relation_recall"] == 1.0
    assert metrics["cargo_relation_f1"] == 1.0
    assert metrics["cargo_relation_exact_match"] == 1.0
    assert metrics["category_value_f1"] == 1.0
    assert metrics["category_value_exact_match"] == 1.0
    assert assessments[0].predicted_cargo_relation_facts


def test_relation_explicit_metrics_penalize_wrong_anchored_allocation_values() -> None:
    task = get_training_task("bill_of_lading_relation_explicit_v3")

    metrics, _ = structured_metrics(
        [_relation_target(allocation_quantities=(20, 10))],
        [_relation_target()],
        task,
    )

    assert metrics["schema_valid"] == 1.0
    assert metrics["cargo_relation_precision"] == 5 / 7
    assert metrics["cargo_relation_recall"] == 5 / 7
    assert metrics["cargo_relation_f1"] == 5 / 7
    assert metrics["cargo_relation_accuracy"] == 5 / 9
    assert metrics["cargo_relation_exact_match"] == 0.0
    assert metrics["category_value_f1"] == 1.0


def test_v4_metrics_include_relations_and_all_modeled_dangerous_goods_categories() -> None:
    task = get_training_task("bill_of_lading_relation_explicit_v4")
    reference = _v4_target()
    predicted = _v4_target(hazard="OXIDIZING_SUBSTANCES_AND_ORGANIC_PEROXIDES")

    metrics, assessments = structured_metrics([predicted], [reference], task)

    assert metrics["schema_valid"] == 1.0
    assert metrics["cargo_relation_accuracy"] == 1.0
    assert metrics["cargo_relation_precision"] == 1.0
    assert metrics["cargo_relation_recall"] == 1.0
    assert metrics["cargo_relation_f1"] == 1.0
    assert metrics["category_value_precision"] == 6 / 7
    assert metrics["category_value_recall"] == 6 / 7
    assert metrics["category_value_f1"] == 6 / 7
    assert metrics["category_value_accuracy"] == 6 / 8
    assert metrics["category_value_exact_match"] == 0.0
    assert len(assessments[0].reference_category_values) == 7


@pytest.mark.parametrize(
    ("task_name", "reference"),
    [
        ("bill_of_lading_relation_explicit_v3", _relation_target()),
        ("bill_of_lading_relation_explicit_v4", _v4_target()),
        ("bill_of_lading_relation_explicit_v5", _v5_target()),
    ],
)
def test_trainer_metric_callback_emits_relation_and_category_metrics_for_each_schema(
    task_name: str, reference: str
) -> None:
    class StubTokenizer:
        pad_token_id: int | None = 0
        eos_token_id: int | None = 2

        def batch_decode(self, sequences: Any, **_: Any) -> list[str]:
            return [reference for _ in sequences]

    compute_metrics = make_compute_metrics(StubTokenizer(), get_training_task(task_name))
    metrics = compute_metrics(
        SimpleNamespace(
            predictions=np.asarray([[1, 2, 0]], dtype=np.int64),
            label_ids=np.asarray([[1, 2, -100]], dtype=np.int64),
        )
    )

    for prefix in ("cargo_relation", "category_value"):
        assert metrics[f"{prefix}_precision"] == 1.0
        assert metrics[f"{prefix}_recall"] == 1.0
        assert metrics[f"{prefix}_f1"] == 1.0
        assert metrics[f"{prefix}_accuracy"] == 1.0
    assert metrics["generation_eos_reached_fraction"] == 1.0


def test_v5_schema_requires_explicit_package_ids_and_diagnostics_anchor_relations() -> None:
    task = get_training_task("bill_of_lading_relation_explicit_v5")
    schema = json.loads(task.prompt_schema_json())
    assert "packageIds" in schema["$defs"]["CargoAllocationGroupV5"]["required"]

    reference = _v5_target()
    reordered = _v5_target(reverse=True)
    metrics, _ = structured_metrics([reordered], [reference], task)
    assert metrics["schema_valid"] == 1.0
    assert metrics["cargo_relation_f1"] == 1.0
    assert metrics["category_value_f1"] == 1.0
    assert metrics["field_value_f1"] == 1.0
    assert metrics["extraction_fact_f1"] == 1.0

    missing_package_ids = json.loads(reference)
    del missing_package_ids["documentPatch"]["cargoAllocationGroups"][0]["packageIds"]
    metrics, _ = structured_metrics([json.dumps(missing_package_ids)], [reference], task)
    assert metrics["schema_valid"] == 0.0


def test_v5_category_metric_detects_wrong_dangerous_goods_category() -> None:
    task = get_training_task("bill_of_lading_relation_explicit_v5")
    metrics, _ = structured_metrics([_v5_target(hazard="GASES")], [_v5_target()], task)
    assert metrics["schema_valid"] == 1.0
    assert metrics["cargo_relation_f1"] == 1.0
    assert metrics["category_value_f1"] < 1.0
    assert metrics["category_value_accuracy"] == 5 / 7


def test_relation_and_category_accuracy_are_one_when_both_fact_sets_are_empty() -> None:
    task = get_training_task("bill_of_lading_relation_explicit_v5")
    target = canonical_json(
        {
            "schemaVersion": "5.0.0-experimental",
            "documentPatch": {"billOfLadingNumber": "ABC"},
        }
    )

    metrics, _ = structured_metrics([target], [target], task)

    assert metrics["cargo_relation_accuracy"] == 1.0
    assert metrics["category_value_accuracy"] == 1.0
    assert metrics["cargo_relation_support_fraction"] == 0.0
    assert metrics["category_value_support_fraction"] == 0.0


def _v7_target() -> dict[str, Any]:
    return {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "billOfLadingNumber": "ABC",
            "containerInformation": [
                {"equipmentIdentifier": "MSKU1200040", "sealNumbers": ["SEAL-A", "SEAL-B"]},
                {"equipmentIdentifier": "FCIU3651201", "sealNumbers": ["SEAL-C"]},
            ],
            "goodsItemDetails": [
                {
                    "description": "PAINT",
                    "hsCodes": ["320810", "320820"],
                    "dangerousGoods": [
                        {"unNumber": "1263", "hazardCategory": "FLAMMABLE_LIQUIDS"},
                        {"hazardCategory": "GASES"},
                    ],
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 20, "typeCategory": "PACKAGE_CARTON"},
                        {"packageQuantity": 3, "typeCategory": "PACKAGE_DRUM"},
                    ],
                    "splitGoodsPlacement": [
                        {"equipmentIdentifier": "MSKU1200040", "packageQuantity": 10},
                        {"equipmentIdentifier": "MSKU1200040", "packageQuantity": 10},
                        {"equipmentIdentifier": "FCIU3651201", "packageQuantity": 3},
                    ],
                },
                {
                    "description": "MACHINERY",
                    "numberAndTypeOfPackages": [{"packageQuantity": 2}],
                    "splitGoodsPlacement": [{"equipmentIdentifier": "FCIU3651201"}],
                },
            ],
        },
    }


def _shuffle_lists(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        return {key: _shuffle_lists(child, rng) for key, child in value.items()}
    if isinstance(value, list):
        result = [_shuffle_lists(child, rng) for child in value]
        rng.shuffle(result)
        return result
    return value


@pytest.mark.parametrize(
    "task_name", ["bill_of_lading_extraction_v7", "bill_of_lading_extraction_v7_reduced"]
)
@pytest.mark.parametrize("errors", [False, True])
def test_all_metrics_ignore_nested_list_permutations_with_partial_credit(
    task_name: str, errors: bool
) -> None:
    reference = _v7_target()
    prediction = copy.deepcopy(reference)
    if errors:
        patch = prediction["documentPatch"]
        patch["billOfLadingNumber"] = "WRONG"
        patch["containerInformation"][0]["sealNumbers"][0] = "WRONG-SEAL"
        goods = patch["goodsItemDetails"][0]
        goods["numberAndTypeOfPackages"][1]["packageQuantity"] = 4
        goods["splitGoodsPlacement"][0]["packageQuantity"] = 11
        goods["dangerousGoods"][1]["hazardCategory"] = "CORROSIVE_SUBSTANCES"
    task = get_training_task(task_name)
    baseline, _ = structured_metrics([json.dumps(prediction)], [json.dumps(reference)], task)
    assert baseline["canonical_exact_match"] == int(not errors)
    for prefix in ("field_value", "extraction_fact", "category_value", "cargo_relation"):
        assert (baseline[f"{prefix}_f1"] < 1.0) == errors
    for seed in range(15):
        rng = random.Random(seed)
        metrics, _ = structured_metrics(
            [json.dumps(_shuffle_lists(prediction, rng))],
            [json.dumps(_shuffle_lists(reference, rng))],
            task,
        )
        assert metrics == baseline


def test_list_matching_preserves_entity_ownership_and_missing_duplicate_occurrences() -> None:
    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    reference = _v7_target()
    wire = json.dumps(reference)
    prediction = copy.deepcopy(reference)
    containers = prediction["documentPatch"]["containerInformation"]
    containers[0]["sealNumbers"], containers[1]["sealNumbers"] = (
        containers[1]["sealNumbers"],
        containers[0]["sealNumbers"],
    )
    metrics, _ = structured_metrics([json.dumps(prediction)], [wire], task)
    assert metrics["canonical_exact_match"] == 0
    assert metrics["field_value_f1"] < 1

    prediction = copy.deepcopy(reference)
    goods = prediction["documentPatch"]["goodsItemDetails"]
    goods[0]["splitGoodsPlacement"], goods[1]["splitGoodsPlacement"] = (
        goods[1]["splitGoodsPlacement"],
        goods[0]["splitGoodsPlacement"],
    )
    metrics, _ = structured_metrics([json.dumps(prediction)], [wire], task)
    assert metrics["cargo_relation_f1"] < 1

    prediction = copy.deepcopy(reference)
    prediction["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"].pop(0)
    metrics, assessments = structured_metrics([json.dumps(prediction)], [wire], task)
    assert metrics["schema_valid"] == 1
    assert metrics["field_value_precision"] == 1
    assert metrics["field_value_recall"] < 1
    assert metrics["cargo_relation_precision"] == 1
    assert metrics["cargo_relation_recall"] < 1
    row = assessments[0]
    assert len(row.reference_field_values) - len(row.predicted_field_values) == 2
    assert len(row.reference_cargo_relation_facts) - len(row.predicted_cargo_relation_facts) == 1
    assert json.loads(row.reference_text) == reference
    assert json.loads(row.generated_text) == prediction

    extra, _ = structured_metrics([wire], [json.dumps(prediction)], task)
    assert extra["field_value_precision"] < 1
    assert extra["field_value_recall"] == 1
    assert extra["cargo_relation_precision"] < 1


def test_list_matching_retains_scalar_multiplicity_and_invalid_structure_penalties() -> None:
    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    reference = _v7_target()
    prediction = copy.deepcopy(reference)
    prediction["documentPatch"]["containerInformation"][0]["sealNumbers"].append("SEAL-A")
    metrics, _ = structured_metrics([json.dumps(prediction)], [json.dumps(reference)], task)
    assert metrics["schema_valid"] == 0
    assert metrics["field_value_precision"] < 1
    assert metrics["field_value_recall"] == 1
    assert metrics["canonical_exact_match"] == 0
    assert unordered_key([1]) != unordered_key([True])
    assert unordered_key([1]) != unordered_key([1.0])
    assert unordered_key([-0.0]) != unordered_key([0.0])
    assert unordered_key(["A", "A"]) != unordered_key(["A"])


def test_notify_party_list_matching_keeps_addresses_with_the_named_party() -> None:
    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    reference = _v7_target()
    reference["documentPatch"]["parties"] = {
        "notifyParties": [
            {"name": "COMPANY A", "addressLine": "1 ROAD", "country": "CHINA"},
            {"name": "COMPANY B", "addressLine": "2 STREET", "country": "EGYPT"},
        ]
    }
    prediction = copy.deepcopy(reference)
    parties = prediction["documentPatch"]["parties"]["notifyParties"]
    parties[0]["name"], parties[1]["name"] = parties[1]["name"], parties[0]["name"]
    scores, rows = structured_metrics([json.dumps(prediction)], [json.dumps(reference)], task)
    assert scores["schema_valid"] == 1
    misses = rows[0].reference_field_values - rows[0].predicted_field_values
    assert len(misses) == 4  # Wrong address and country for each named party.
    assert all(path.endswith((".addressLine", ".country")) for path, _ in misses)


def test_matching_missing_goods_does_not_shift_other_goods_or_category_ownership() -> None:
    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    reference = _v7_target()
    prediction = copy.deepcopy(reference)
    prediction["documentPatch"]["goodsItemDetails"].pop(0)
    metrics, _ = structured_metrics([json.dumps(prediction)], [json.dumps(reference)], task)
    assert metrics["field_value_precision"] == 1
    assert metrics["field_value_recall"] < 1
    assert metrics["cargo_relation_precision"] == 1
    assert metrics["cargo_relation_recall"] < 1


def test_assignment_is_globally_optimal_not_greedy_and_handles_rectangular_lists() -> None:
    rng = random.Random(13)
    matrices = [[[9, 8], [8, 0]]]
    for rows in range(1, 5):
        for columns in range(1, 5):
            matrices.extend(
                [[rng.randrange(10) for _ in range(columns)] for _ in range(rows)]
                for _ in range(10)
            )
    for weights in matrices:
        pairs = _maximum_weight_pairs(weights)
        assert len({i for i, _ in pairs}) == len(pairs)
        assert len({j for _, j in pairs}) == len(pairs)
        oriented = weights if len(weights) <= len(weights[0]) else list(zip(*weights, strict=True))
        optimum = max(
            sum(oriented[i][j] for i, j in enumerate(partners))
            for partners in itertools.permutations(range(len(oriented[0])), len(oriented))
        )
        assert sum(weights[i][j] for i, j in pairs) == optimum
    assert _maximum_weight_pairs([]) == []
    assert _maximum_weight_pairs([[]]) == []


def test_trainer_metric_callback_publishes_exact_field_value_metrics() -> None:
    class StubTokenizer:
        pad_token_id: int | None = 0
        eos_token_id: int | None = 2

        def batch_decode(self, sequences: Any, **_: Any) -> list[str]:
            number = "ABC" if int(sequences[0][0]) in {1, 2} else "WRONG"
            return [_target(number)]

    compute_metrics = make_compute_metrics(
        StubTokenizer(), get_training_task("bill_of_lading_semantic_v2")
    )

    metrics = compute_metrics(
        SimpleNamespace(
            predictions=np.asarray([[1, 2, 0]], dtype=np.int64),
            label_ids=np.asarray([[2]], dtype=np.int64),
        )
    )

    assert metrics == {
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "canonical_exact_match": 1.0,
        "field_value_accuracy": 1.0,
        "field_value_precision": 1.0,
        "field_value_recall": 1.0,
        "field_value_f1": 1.0,
        "generated_tokens_mean": 2.0,
        "generated_tokens_max": 2.0,
        "generation_eos_reached_fraction": 1.0,
    }


def test_best_model_prediction_is_scored_and_persisted_in_one_pass(tmp_path: Path) -> None:
    class StubTokenizer:
        pad_token_id = 0

        def batch_decode(self, sequences: Any, **_: Any) -> list[str]:
            return [_target("ABC") for _ in sequences]

    expected = {
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "canonical_exact_match": 1.0,
        "field_value_accuracy": 1.0,
        "field_value_precision": 1.0,
        "field_value_recall": 1.0,
        "field_value_f1": 1.0,
    }

    class StubTrainerBase:
        calls = 0
        args = SimpleNamespace(world_size=1, dataloader_drop_last=False)

        def predict(self, dataset: Any, metric_key_prefix: str) -> Any:
            self.calls += 1
            indices = list(self._get_eval_sampler(dataset))
            assert indices == [0]
            return SimpleNamespace(
                predictions=np.asarray([[1, 2, 0]], dtype=np.int64),
                label_ids=np.asarray([[1, 2, -100]], dtype=np.int64),
                metrics={
                    **{f"{metric_key_prefix}_{name}": value for name, value in expected.items()},
                    f"{metric_key_prefix}_loss": 0.1,
                },
            )

        def _get_eval_sampler(self, _: Any) -> list[int]:
            return [0]

    class StubTrainer(SamplerAwarePredictionMixin, StubTrainerBase):
        pass

    trainer = StubTrainer()
    metrics, prediction_path = _predict_and_publish(
        trainer=trainer,
        tokenizer=StubTokenizer(),
        task=get_training_task("bill_of_lading_semantic_v2"),
        dataset={"document_id": ["doc_one"]},
        split="validation",
        metric_key_prefix="eval",
        predictions_dir=tmp_path / "predictions",
    )

    assert trainer.calls == 1
    assert metrics["eval_field_value_f1"] == 1.0
    row = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert row["document_id"] == "doc_one"
    assert row["json_valid"] is True


def test_prediction_publication_uses_exact_length_sampler_identity_order(
    tmp_path: Path,
) -> None:
    targets = [_target("SHORT"), _target("LONG"), _target("MEDIUM")]

    class StubTokenizer:
        pad_token_id = 0

        def batch_decode(self, sequences: Any, **_: Any) -> list[str]:
            return [targets[int(sequence[0]) - 1] for sequence in sequences]

    expected = {
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "canonical_exact_match": 1.0,
        "field_value_accuracy": 1.0,
        "field_value_precision": 1.0,
        "field_value_recall": 1.0,
        "field_value_f1": 1.0,
    }

    class StubTrainerBase:
        calls = 0
        args = SimpleNamespace(world_size=1, dataloader_drop_last=False)

        def _get_eval_sampler(self, _: Any) -> list[int]:
            # This is the order that previously made source-order IDs incorrect.
            return [2, 0, 1]

        def predict(self, dataset: Any, metric_key_prefix: str) -> Any:
            self.calls += 1
            indices = list(self._get_eval_sampler(dataset))
            tokens = np.asarray([[index + 1, 0] for index in indices], dtype=np.int64)
            return SimpleNamespace(
                predictions=tokens,
                label_ids=tokens.copy(),
                metrics={
                    **{f"{metric_key_prefix}_{name}": value for name, value in expected.items()},
                    f"{metric_key_prefix}_loss": 0.1,
                },
            )

    class StubTrainer(SamplerAwarePredictionMixin, StubTrainerBase):
        pass

    trainer = StubTrainer()
    _, prediction_path = _predict_and_publish(
        trainer=trainer,
        tokenizer=StubTokenizer(),
        task=get_training_task("bill_of_lading_semantic_v2"),
        dataset={
            "document_id": ["doc_short", "doc_long", "doc_medium"],
            "input_length": [10, 100, 50],
        },
        split="validation",
        metric_key_prefix="eval",
        predictions_dir=tmp_path / "predictions",
    )

    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert trainer.calls == 1
    assert [row["document_id"] for row in rows] == [
        "doc_medium",
        "doc_short",
        "doc_long",
    ]
    reference_numbers = [
        json.loads(row["reference_text"])["documentPatch"]["billOfLadingNumber"] for row in rows
    ]
    assert reference_numbers == [
        "MEDIUM",
        "SHORT",
        "LONG",
    ]
    assert all(row["canonical_exact_match"] for row in rows)
