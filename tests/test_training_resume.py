from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from document_ocr.training.config import TrainingConfig, load_training_config
from document_ocr.training.prompting import load_prompt
from document_ocr.training.runtime import _prepare_run_directory
from document_ocr.training.tasks import get_training_task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "t5gemma2_270m_lora.pilot106.yaml"
)


def _payload(config: TrainingConfig) -> bytes:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")


def test_resume_keeps_training_contract_immutable_and_records_invocation(
    tmp_path: Path,
) -> None:
    base = load_training_config(CONFIG_PATH)
    base = base.model_copy(
        update={
            "run": base.run.model_copy(
                update={"run_id": "resume-contract", "output_dir": str(tmp_path)}
            )
        }
    )
    prompt = load_prompt(PROJECT_ROOT, base.prompt, get_training_task(base.task))
    prepared = SimpleNamespace(report=lambda: {"cache_identity": "test"})
    environment = {"packages": {"test": "1"}}

    run_dir, resume, artifacts = _prepare_run_directory(
        project_root=PROJECT_ROOT,
        config_payload=_payload(base),
        config=base,
        prompt=prompt,
        prepared=prepared,
        environment=environment,
    )
    assert resume is None
    assert len(artifacts) == 2

    checkpoint = run_dir / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    resumed = base.model_copy(
        update={
            "checkpoint": base.checkpoint.model_copy(
                update={"resume_from_checkpoint": str(checkpoint)}
            ),
            "logging": base.logging.model_copy(
                update={
                    "mlflow": base.logging.mlflow.model_copy(
                        update={"resume_run_id": "0123456789abcdef0123456789abcdef"}
                    )
                }
            ),
        }
    )
    resumed_run_dir, resume, artifacts = _prepare_run_directory(
        project_root=PROJECT_ROOT,
        config_payload=_payload(resumed),
        config=resumed,
        prompt=prompt,
        prepared=prepared,
        environment=environment,
    )

    assert resumed_run_dir == run_dir
    assert resume == checkpoint
    assert artifacts[-1] == run_dir / "resume-invocations" / "checkpoint-1.yaml"
    assert artifacts[-1].is_file()

    changed = resumed.model_copy(
        update={
            "optimization": resumed.optimization.model_copy(
                update={"learning_rate": 0.0003}
            )
        }
    )
    with pytest.raises(RuntimeError, match="differs from its immutable run contract"):
        _prepare_run_directory(
            project_root=PROJECT_ROOT,
            config_payload=_payload(changed),
            config=changed,
            prompt=prompt,
            prepared=prepared,
            environment=environment,
        )
