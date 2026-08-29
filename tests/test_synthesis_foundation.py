from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError
from yaml.constructor import ConstructorError

from document_ocr.synthesis.config import (
    SynthesisFoundationConfig,
    load_synthesis_foundation_config,
)
from document_ocr.synthesis.normalization import denormalize_target, normalize_target
from document_ocr.synthesis.pipeline import prepare_synthesis_foundation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/synthesis/mpci_bl_combined1157_foundation.yaml"
SOURCE = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl"
)


def test_value_graph_round_trips_every_json_type() -> None:
    target = {
        "object": {"text": "Å", "integer": 3, "number": 1.5, "boolean": True},
        "list": [None, "x", {"nested": []}],
    }

    nodes = normalize_target("doc_test", target)

    assert denormalize_target(nodes) == target
    assert nodes[0].node_kind == "object"
    assert [node.node_id for node in nodes] == [
        f"doc_test:n{index:06d}" for index in range(1, len(nodes) + 1)
    ]


def test_synthesis_yaml_rejects_duplicate_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ConstructorError, match="duplicate key"):
        load_synthesis_foundation_config(config_path)


def test_relation_synthesis_requires_constraints() -> None:
    base = load_synthesis_foundation_config(CONFIG_PATH).model_dump(mode="json")
    base["task_constraints"] = None

    with pytest.raises(ValidationError, match="requires task_constraints"):
        SynthesisFoundationConfig.model_validate(base, strict=True)


def test_value_graph_rejects_disconnected_nodes() -> None:
    nodes = normalize_target("doc_test", {"a": 1})
    disconnected = replace(
        nodes[1],
        node_id="doc_test:n999999",
        parent_node_id="doc_test:n999998",
    )
    with pytest.raises(ValueError, match="absent parent"):
        denormalize_target((nodes[0], disconnected))


def test_foundation_round_trips_and_publishes_one_real_record(tmp_path: Path) -> None:
    first_line = SOURCE.read_bytes().splitlines(keepends=True)[0]
    source = tmp_path / "source.jsonl"
    source.write_bytes(first_line)
    base = load_synthesis_foundation_config(CONFIG_PATH).model_dump(mode="json")
    base["run"] = {"run_id": "test-foundation", "output_dir": str(tmp_path / "output")}
    base["source"]["file"] = {
        "path": str(source),
        "sha256": hashlib.sha256(first_line).hexdigest(),
        "records": 1,
    }
    base["sdv"]["validate_with_sdv"] = False
    config = SynthesisFoundationConfig.model_validate(base, strict=True)

    manifest = prepare_synthesis_foundation(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        config=config,
    )
    manifest_bytes = (tmp_path / "output/test-foundation/manifest.json").read_bytes()
    repeated = prepare_synthesis_foundation(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        config=config,
    )

    run_dir = tmp_path / "output" / "test-foundation"
    document = json.loads((run_dir / "tables/documents.jsonl").read_text().strip())
    assert manifest["roundTripValidatedRecords"] == 1
    assert manifest["sdvValidation"]["enabled"] is False
    assert document["node_count"] > 1
    assert (run_dir / "tables/nodes.jsonl").is_file()
    assert (run_dir / "manifest.json").read_bytes() == manifest_bytes
    assert repeated["files"] == manifest["files"]


def test_foundation_rejects_wrong_input_digest(tmp_path: Path) -> None:
    row = json.loads(SOURCE.read_text().splitlines()[0])
    row["joinedRawTextSha256"] = "0" * 64
    encoded = json.dumps(row, separators=(",", ":")).encode() + b"\n"
    source = tmp_path / "source.jsonl"
    source.write_bytes(encoded)
    base = load_synthesis_foundation_config(CONFIG_PATH).model_dump(mode="json")
    base["run"] = {"run_id": "bad-input-digest", "output_dir": str(tmp_path / "output")}
    base["source"]["file"] = {
        "path": str(source),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "records": 1,
    }
    base["sdv"]["validate_with_sdv"] = False
    config = SynthesisFoundationConfig.model_validate(base, strict=True)

    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        prepare_synthesis_foundation(
            project_root=PROJECT_ROOT,
            config_path=CONFIG_PATH,
            config=config,
        )
