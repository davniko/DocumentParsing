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
    prepared = SimpleNamespace(
        report=lambda: {
            "cache_identity": "test",
            "source_files": ({"path": "source.jsonl"},),
        }
    )
    environment = {
        "packages": {"test": "1"},
        "source_code": [{"path": "runtime.py", "sha256": "original"}],
    }

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
                update={
                    "resume_from_checkpoint": str(checkpoint),
                    "allow_resume_source_code_drift": True,
                }
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
        environment={
            **environment,
            "source_code": [{"path": "runtime.py", "sha256": "resumed"}],
        },
    )

    assert resumed_run_dir == run_dir
    assert resume == checkpoint
    invocation_artifacts = [
        path for path in artifacts if path.parent == run_dir / "resume-invocations"
    ]
    assert len(invocation_artifacts) == 2
    assert {path.suffix for path in invocation_artifacts} == {".yaml", ".json"}
    assert all(path.name.startswith("checkpoint-1-") for path in invocation_artifacts)
    assert all(path.is_file() for path in invocation_artifacts)

    changed_prepared = SimpleNamespace(
        report=lambda: {"cache_identity": "different-partition"}
    )
    with pytest.raises(RuntimeError, match="differs from its immutable run report"):
        _prepare_run_directory(
            project_root=PROJECT_ROOT,
            config_payload=_payload(resumed),
            config=resumed,
            prompt=prompt,
            prepared=changed_prepared,
            environment={
                **environment,
                "source_code": [{"path": "runtime.py", "sha256": "resumed"}],
            },
        )

    with pytest.raises(RuntimeError, match="outside the explicitly allowed"):
        _prepare_run_directory(
            project_root=PROJECT_ROOT,
            config_payload=_payload(resumed),
            config=resumed,
            prompt=prompt,
            prepared=prepared,
            environment={
                "packages": {"test": "2"},
                "source_code": [{"path": "runtime.py", "sha256": "resumed"}],
            },
        )

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
            environment={
                **environment,
                "source_code": [{"path": "runtime.py", "sha256": "resumed"}],
            },
        )
