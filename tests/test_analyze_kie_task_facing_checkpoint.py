from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
from safetensors.torch import save_file

from document_ocr.training.metrics import PredictionAssessment


def _load_tool() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "analyze_kie_task_facing_checkpoint.py"
    spec = importlib.util.spec_from_file_location("analyze_kie_task_facing_checkpoint", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TOOL = _load_tool()


def _record(document_id: str, *, split: str, target: dict) -> object:
    return _TOOL.Record(
        document_id=document_id,
        split=split,
        raw_text="ALPHA BETA GAMMA",
        target={"documentPatch": target},
        source_corpus="test",
        source_path="test.pdf",
        page_count=1,
        input_tokens=10,
        target_tokens=10,
        carrier_name="<MISSING>",
        carrier_family="<UNAVAILABLE>",
        template_id="<UNAVAILABLE>",
        template_seen_in_train="unknown",
        container_count=0,
        cargo_group_count=0,
        package_count=0,
        allocation_group_count=0,
        target_leaves=len(_TOOL._flatten(target)),
    )


def _diagnostic(
    predicted: frozenset[tuple[str, str]], reference: frozenset[tuple[str, str]]
) -> object:
    assessment = PredictionAssessment(
        generated_text="{}",
        reference_text="{}",
        json_valid=True,
        schema_valid=True,
        canonical_exact_match=False,
        predicted_field_values=predicted,
        reference_field_values=reference,
        predicted_cargo_relation_facts=frozenset(),
        reference_cargo_relation_facts=frozenset(),
        predicted_category_values=frozenset(),
        reference_category_values=frozenset(),
    )
    return _TOOL.Diagnostic(
        record=_record("doc", split="validation", target={}),
        assessment=assessment,
        true_positive=len(predicted & reference),
        predicted=len(predicted),
        reference=len(reference),
        compared_paths=len({path for path, _ in predicted | reference}),
        index_true_positive=_TOOL._index_true_positive(predicted, reference),
        schema_failure_class="none",
        schema_failure_detail="",
        generated_characters=2,
        reference_characters=2,
    )


def test_error_taxonomy_distinguishes_index_substitution_omission_and_addition() -> None:
    reference = frozenset(
        {
            ("$.documentPatch.cargoGroups[0].description", '"ALPHA"'),
            ("$.documentPatch.cargoGroups[1].description", '"BETA"'),
            ("$.documentPatch.billOfLadingNumber", '"REF"'),
            ("$.documentPatch.voyageNumber", '"V1"'),
        }
    )
    predicted = frozenset(
        {
            ("$.documentPatch.cargoGroups[0].description", '"BETA"'),
            ("$.documentPatch.cargoGroups[1].description", '"WRONG"'),
            ("$.documentPatch.billOfLadingNumber", '"DIFFERENT"'),
            ("$.documentPatch.placeOfIssue", '"PORT"'),
        }
    )

    rows = _TOOL._classify_errors(_diagnostic(predicted, reference))

    assert {row["error_type"] for row in rows} == {
        "right_value_wrong_index",
        "list_alignment_or_value_mismatch",
        "substitution",
        "omission",
        "addition",
    }


def test_novel_value_support_is_independent_of_record_iteration_order() -> None:
    field = "$.documentPatch.parties.shipper.name"
    assessment = PredictionAssessment(
        generated_text="{}",
        reference_text="{}",
        json_valid=True,
        schema_valid=True,
        canonical_exact_match=True,
        predicted_field_values=frozenset({(field, '"KNOWN"')}),
        reference_field_values=frozenset({(field, '"KNOWN"')}),
        predicted_cargo_relation_facts=frozenset(),
        reference_cargo_relation_facts=frozenset(),
        predicted_category_values=frozenset(),
        reference_category_values=frozenset(),
    )
    validation = _record(
        "validation", split="validation", target={"parties": {"shipper": {"name": "KNOWN"}}}
    )
    train = _record("train", split="train", target={"parties": {"shipper": {"name": "KNOWN"}}})
    diagnostic = _TOOL.Diagnostic(
        record=validation,
        assessment=assessment,
        true_positive=1,
        predicted=1,
        reference=1,
        compared_paths=1,
        index_true_positive=1,
        schema_failure_class="none",
        schema_failure_detail="",
        generated_characters=2,
        reference_characters=2,
    )

    rows = _TOOL._field_rows([diagnostic], {"validation": validation, "train": train})

    assert rows[0]["validation_novel_values"] == 0
    assert rows[0]["train_documents"] == 1


def test_dataset_complexity_uses_declared_split_denominators() -> None:
    train = _record("train", split="train", target={})
    validation = _record("validation", split="validation", target={})

    rows = _TOOL._dataset_complexity_rows({"train": train, "validation": validation})

    assert len(rows) == 16
    assert {row["documents"] for row in rows} == {1}
    assert {row["fraction"] for row in rows} == {0.0}


def test_adapter_inventory_counts_encoder_and_decoder_parameters(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    save_file(
        {
            "base_model.model.model.encoder.layer.lora_A.weight": torch.zeros((2, 3)),
            "base_model.model.model.encoder.layer.lora_B.weight": torch.zeros((4, 2)),
            "base_model.model.model.decoder.layer.lora_A.weight": torch.zeros((2, 5)),
            "base_model.model.model.decoder.layer.lora_B.weight": torch.zeros((6, 2)),
        },
        path,
    )

    inventory = _TOOL._adapter_inventory(path)

    assert inventory["tensor_count"] == 4
    assert inventory["adapted_module_count"] == 2
    assert inventory["parameter_count"] == 36
    assert inventory["encoder_parameters"] == 14
    assert inventory["decoder_parameters"] == 22
