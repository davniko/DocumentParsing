"""Rank96 preserves hyperparameters and pins the reviewed seven-record amendment."""

import json
from copy import deepcopy
from math import ceil
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.training.config import load_training_config
from document_ocr.training.tasks import load_training_task

ROOT = Path(__file__).parents[1]
PREFIX = (
    "t5gemma2_270m_lora.mpci_bl_real600_synthetic9400_r30_paddle_centroids_"
    "inputonly_v7_e10_compact_eva_a32_"
)


@pytest.mark.parametrize("hardware", ["local", "runpod"])
def test_only_rank_run_identity_and_reviewed_data_amendment_change(hardware):
    path = ROOT / "configs/training/production"
    old = load_training_config(path / f"{PREFIX}r64_{hardware}_schedulefree_v1.yaml").model_dump(
        mode="json"
    )
    config = load_training_config(path / f"{PREFIX}r96_{hardware}_schedulefree_v1.yaml")
    actual = config.model_dump(mode="json")
    expected = deepcopy(old)
    expected["run"]["run_id"] = expected["run"]["run_id"].replace("a32-r64", "a32-r96")
    expected["peft"]["adapter_name"] = expected["peft"]["adapter_name"].replace(
        "a32_r64", "a32_r96"
    )
    expected["peft"]["rank"] = 96
    expected["logging"]["mlflow"]["tags"]["lora_rank"] = "96"
    dataset = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1"
    current = json.loads((dataset / "manifest.json").read_text())["aggregateRecords"]
    frozen = json.loads(
        (dataset / "audit-history/posttraining-20261108/frozen-splits.json").read_text()
    )
    for split, names in [
        ("train", ["synthetic-train", "real-train"]),
        ("validation", ["validation"]),
    ]:
        for source, name in zip(expected["dataset"]["splits"][split], names, strict=True):
            assert source["path"] == frozen[name]["path"]
            assert source["sha256"] == frozen[name]["sha256"]
            source.update(current[name])
            assert sha256_file(ROOT / source["path"]) == source["sha256"]
    assert actual == expected
    assert config.peft.alpha == 32 and config.peft.use_rslora
    assert config.peft.eva.rho == 1.0
    assert config.checkpoint.resume_from_checkpoint is None
    assert config.logging.mlflow.resume_run_id is None
    assert config.optimization.train_sampling_strategy == "group_by_length"
    batch = (
        config.optimization.per_device_train_batch_size
        * config.optimization.gradient_accumulation_steps
    )
    assert batch == (48 if hardware == "runpod" else 32)
    steps = ceil(10000 / batch)
    assert config.evaluation.steps == config.checkpoint.steps == int(steps * 2.5)
    assert config.optimization.schedule_free_adamw.warmup_steps == ceil(steps * 10 * 0.05)
    assert len(load_training_task(ROOT, config).constraints.packageCategoryTokens) == 49


def test_rank96_runpod_deployment_only_changes_access_and_storage():
    ordinary = load_training_config(
        ROOT / f"configs/training/production/{PREFIX}r96_runpod_schedulefree_v1.yaml"
    ).model_dump(mode="json")
    ordinary["model"]["token_env"] = None
    ordinary["model"]["local_files_only"] = True
    ordinary["run"]["output_dir"] = "/workspace/DocumentParsing/artifacts/kie-training"
    deployment = load_training_config(
        ROOT / "configs/training/deployment/r30_r96_runpod_offline.yaml"
    )
    assert deployment.model_dump(mode="json") == ordinary


def test_length_grouping_reshuffles_and_preserves_coverage():
    torch = pytest.importorskip("torch")
    from transformers.trainer_pt_utils import LengthGroupedSampler

    lengths = [(i * 41) % 1000 + 1 for i in range(10000)]
    torch.manual_seed(424)
    sampler = LengthGroupedSampler(batch_size=48, lengths=lengths)
    first, second = list(sampler), list(sampler)
    assert sorted(first) == sorted(second) == list(range(10000))
    assert first != second
    torch.manual_seed(424)
    assert list(sampler) == first
