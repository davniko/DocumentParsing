"""R30 contract and explicitly approved hardware-specific training schedules."""

import importlib.util
import json
from copy import deepcopy
from math import ceil
from pathlib import Path

import pytest

from document_ocr.semantic_v3.transform import CategoryRegistry
from document_ocr.spatial_inputs.paddle import pages_from_record, serialize
from document_ocr.training.config import load_training_config
from document_ocr.training.tasks import load_training_task

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/training/production"
PREFIX = "t5gemma2_270m_lora.mpci_bl_real600_synthetic9400_"


def config(version, hardware):
    mode = "positions" if version == "r28" else "paddle_centroids"
    return load_training_config(
        DIRECTORY / f"{PREFIX}{version}_{mode}_inputonly_v7_e10_compact_eva_a32_r64_"
        f"{hardware}_schedulefree_v1.yaml"
    )


@pytest.mark.parametrize("hardware", ["local", "runpod"])
def test_preserves_r28_except_approved_overrides_and_binds_r30_contract(hardware):
    old = config("r28", hardware).model_dump(mode="json")
    current = config("r30", hardware)
    new = current.model_dump(mode="json")
    expected = deepcopy(old)
    for section in ("run", "dataset", "task_constraints", "logging"):
        expected[section] = new[section]
    expected["peft"]["adapter_name"] = new["peft"]["adapter_name"]
    expected["evaluation"]["generation_max_length"] = 3329
    if hardware == "runpod":
        expected["optimization"]["gradient_accumulation_steps"] = 12
        expected["optimization"]["schedule_free_adamw"]["warmup_steps"] = 105
        expected["evaluation"]["steps"] = 522
        expected["checkpoint"]["steps"] = 522
    assert new == expected
    assert current.run.run_id != old["run"]["run_id"]
    assert current.checkpoint.resume_from_checkpoint is None
    assert current.logging.mlflow.resume_run_id is None
    assert current.dataset.fields.input_sha256 is None
    assert current.dataset.fields.input_text == "positionedText"
    assert current.dataset.preprocessing.max_target_length == 5500
    sources = (*current.dataset.splits.train, *current.dataset.splits.validation)
    assert [s.records for s in sources] == [9400, 600, 60]
    assert all("mpci-bl-r30-paddle-centroids-v1/" in s.path for s in sources)
    task = load_training_task(ROOT, current)
    assert len(task.constraints.packageCategoryTokens) == 49
    assert "PACKAGE_PALLET_PLASTIC" in task.constraints.packageCategoryTokens


@pytest.mark.parametrize(
    "hardware,microbatch,accumulation,effective,updates,logging_steps",
    [("local", 1, 32, 32, 3130, 5), ("runpod", 4, 12, 48, 2090, 3)],
)
def test_batch_and_step_schedule_track_dataset_and_hardware(
    hardware, microbatch, accumulation, effective, updates, logging_steps
):
    current = config("r30", hardware)
    optimization = current.optimization
    assert current.runtime.distributed_processes == 1
    assert not current.dataloader.drop_last
    assert optimization.max_steps == -1
    assert optimization.num_train_epochs == 10.0
    assert optimization.per_device_train_batch_size == microbatch
    assert optimization.gradient_accumulation_steps == accumulation
    assert microbatch * accumulation == effective
    records = sum(source.records for source in current.dataset.splits.train)
    # Trainer flushes the final partial accumulation at each epoch boundary.
    per_epoch = ceil(ceil(records / microbatch) / accumulation)
    assert ceil(per_epoch * optimization.num_train_epochs) == updates
    assert optimization.schedule_free_adamw.warmup_steps == ceil(updates * 0.05)
    assert current.evaluation.steps == current.checkpoint.steps == int(per_epoch * 2.5)
    assert current.evaluation.run_final_evaluation
    assert current.evaluation.per_device_batch_size == 2
    assert current.logging.steps == logging_steps
    tags = current.logging.mlflow.tags
    assert tags["effective_optimizer_batch"] == str(effective)
    assert tags["projected_optimizer_updates"] == str(updates)
    assert tags["microbatch_size"] == str(microbatch)
    assert tags["gradient_accumulation_steps"] == str(accumulation)


def test_registry_propagates_reviewed_category_without_inventing_submission_code():
    directory = ROOT / "artifacts/sampling-registries"
    old = json.loads((directory / "package-extraction-vocabulary-v2.json").read_text())
    path = directory / "package-extraction-vocabulary-r30.json"
    new = json.loads(path.read_text())
    assert new["entries"][:-1] == old["entries"]
    assert new["entries"][-1]["categoryToken"] == "PACKAGE_PALLET_PLASTIC"
    registry = CategoryRegistry.model_validate_json(path.read_bytes(), strict=True)
    registry.verify_application_source(path)
    with pytest.raises(ValueError, match="no exact application-code mapping"):
        registry.application_code("PACKAGE_PALLET_PLASTIC")


def coverage_function():
    spec = importlib.util.spec_from_file_location(
        "r30_coordinate_audit", ROOT / "scripts/migrations/audit_r30_config_and_coordinates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.region_coverage


def sample():
    row = {
        "documentId": "control",
        "pages": [
            {
                "index": 0,
                "width": 100,
                "height": 100,
                "regions": [
                    {"index": 0, "text": "CARGO", "box": [1, 2, 40, 8]},
                    {"index": 1, "text": "", "box": [10, 20, 40, 28]},
                ],
            }
        ],
    }
    pages = pages_from_record(row)
    row["positionedText"] = serialize(pages)
    row["bboxText"] = serialize(pages, geometry="bbox")
    return row


def test_complete_coordinates_include_empty_ocr_regions_but_not_page_markers():
    assert coverage_function()(sample()) == (2, 1, 1)


def test_missing_geometry_is_not_counted_as_covered():
    row = sample()
    del row["pages"][0]["regions"][0]["box"]
    with pytest.raises(KeyError):
        coverage_function()(row)


def test_missing_serialized_suffix_is_not_counted_as_covered():
    row = sample()
    row["positionedText"] = row["positionedText"].replace(" || 205,50", "")
    with pytest.raises(ValueError, match="positionedText"):
        coverage_function()(row)
