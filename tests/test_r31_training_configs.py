"""R31 changes data and dataset-sized schedules, not the rank-64 baseline."""

import json
from copy import deepcopy
from math import ceil
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.training.config import load_training_config
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import load_training_task

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/training/production"
MANIFEST = ROOT / "data/curated/mpci-bl-r31-paddle-centroids-v1/manifest.json"


def config(version, hardware):
    count = 10150 if version == "r31" else 9400
    return load_training_config(DIRECTORY / (
        f"t5gemma2_270m_lora.mpci_bl_real600_synthetic{count}_{version}_paddle_centroids_"
        f"inputonly_v7_e10_compact_eva_a32_r64_{hardware}_schedulefree_v1.yaml"
    ))


@pytest.mark.parametrize("hardware", ["local", "runpod"])
def test_r31_preserves_rank64_settings_except_data_identity_and_scaled_schedule(hardware):
    old = config("r30", hardware).model_dump(mode="json")
    new = config("r31", hardware).model_dump(mode="json")
    expected = deepcopy(old)
    expected["run"]["run_id"] = old["run"]["run_id"].replace("synthetic9400-r30", "synthetic10150-r31")
    expected["peft"]["adapter_name"] = old["peft"]["adapter_name"].replace("synthetic9400_r30", "synthetic10150_r31")
    expected["dataset"]["splits"] = new["dataset"]["splits"]
    expected["optimization"]["schedule_free_adamw"]["warmup_steps"] = 168 if hardware == "local" else 112
    expected["evaluation"]["steps"] = expected["checkpoint"]["steps"] = 840 if hardware == "local" else 560
    expected["logging"]["mlflow"]["tags"].update(
        dataset="r31-paddle-centroids-v1",
        labels_source="r31-printed-casing-and-boundary-extension-20261109",
        synthetic_train_records="10150",
        training_records="10750",
        projected_optimizer_updates="3360" if hardware == "local" else "2240",
        target_casing="printed-party-and-goods-v1",
        boundary_extension_records="750",
        dataset_manifest_sha256=sha256_file(MANIFEST),
    )
    assert new == expected
    assert new["checkpoint"]["resume_from_checkpoint"] is None
    assert new["logging"]["mlflow"]["resume_run_id"] is None


@pytest.mark.parametrize("hardware,micro,accumulation,updates,final_batch", [
    ("local", 1, 32, 3360, 30), ("runpod", 4, 12, 2240, 46),
])
def test_r31_epoch_flush_and_training_schedule(hardware, micro, accumulation, updates, final_batch):
    c = config("r31", hardware)
    assert c.optimization.per_device_train_batch_size == micro
    assert c.optimization.gradient_accumulation_steps == accumulation
    assert c.runtime.distributed_processes == 1 and not c.dataloader.drop_last
    count = sum(s.records for s in c.dataset.splits.train)
    per_epoch = ceil(ceil(count / micro) / accumulation)
    assert count == 10750
    assert per_epoch * c.optimization.num_train_epochs == updates
    assert count - (per_epoch - 1) * micro * accumulation == final_batch
    assert c.optimization.schedule_free_adamw.warmup_steps == ceil(updates * .05)
    assert c.evaluation.steps == c.checkpoint.steps == per_epoch * 2.5
    assert c.evaluation.run_final_evaluation
    assert c.evaluation.generation_max_length == 3329
    assert c.optimization.train_sampling_strategy == "group_by_length"


@pytest.mark.parametrize("hardware", ["local", "runpod"])
def test_r31_files_constraints_and_source_case_contract(hardware):
    c = config("r31", hardware)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["stageB"]["status"] == "complete"
    assert manifest["targetCasing"] == "printed-party-and-goods-v1"
    sources = (*c.dataset.splits.train, *c.dataset.splits.validation)
    for source, split, count in zip(sources, ("synthetic-train", "real-train", "validation"), (10150, 600, 60), strict=True):
        assert source.path == manifest["aggregateRecords"][split]["path"]
        assert source.sha256 == manifest["aggregateRecords"][split]["sha256"]
        assert source.records == count
    task = load_training_task(ROOT, c)
    assert len(task.constraints.packageCategoryTokens) == 49
    prompt = load_prompt(ROOT, c.prompt, task)
    assert prompt.render("Printed Name Ltd || 1,2") == "Printed Name Ltd || 1,2"
    target = {"negotiability": "non_negotiable",
              "parties": {"shipper": {"name": "Printed Name Ltd", "addressLine": "12 New Road"}},
              "goodsItemDetails": [{"description": "Printed goods Description"}]}
    assert task.canonicalize(target) == target
    assert c.dataset.fields.input_sha256 is None
    assert c.dataset.fields.input_text == "positionedText"
    assert not c.evaluation.case_sensitive  # Same scoring as R30, not uppercase targets.


def test_r31_offline_pod_deployment_only_changes_storage_and_model_access():
    expected = config('r31', 'runpod').model_dump(mode='json')
    expected['run']['output_dir'] = '/workspace/DocumentParsing/artifacts/kie-training'
    expected['model']['token_env'] = None
    expected['model']['local_files_only'] = True
    deployed = load_training_config(ROOT / 'configs/training/deployment/r31_runpod_offline.yaml')
    assert deployed.model_dump(mode='json') == expected
    assert not Path(deployed.dataset.preprocessing.cache_dir).is_absolute()
