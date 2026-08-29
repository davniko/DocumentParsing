from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes


def _load_tool() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "materialize_kie_table_input.py"
    spec = importlib.util.spec_from_file_location("materialize_kie_table_input", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    files = []
    for split, kind in (("train", "train_records"), ("validation", "validation_records")):
        raw = f"RAW {split}"
        table = f"--- PAGE 1 ---\n<table><tr><td>{split}</td></tr></table>"
        row = {
            "documentId": f"doc_{split}",
            "joinedRawText": raw,
            "joinedRawTextSha256": sha256_bytes(raw.encode()),
            "target": {"documentPatch": {"billOfLadingNumber": split}},
            "auxiliaryViews": {
                "glmOcrTableRecognition": {
                    "documentPageCount": 1,
                    "joinedTableText": table,
                    "joinedTableTextSha256": sha256_bytes(table.encode()),
                    "pages": [
                        {
                            "pageIndex": 0,
                            "pageNumber": 1,
                            "rawTableText": f"<table><tr><td>{split}</td></tr></table>",
                            "rawTableTextSha256": sha256_bytes(
                                f"<table><tr><td>{split}</td></tr></table>".encode()
                            ),
                        }
                    ],
                }
            },
        }
        payload = canonical_json_bytes(row) + b"\n"
        (root / f"{split}.jsonl").write_bytes(payload)
        files.append(
            {
                "kind": kind,
                "path": f"{split}.jsonl",
                "records": 1,
                "sha256": sha256_bytes(payload),
            }
        )
    (root / "manifest.json").write_bytes(
        canonical_json_bytes({"dataset_id": "source", "files": files}) + b"\n"
    )
    return root


def test_materialize_preserves_source_and_adds_audited_dual_view(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "output"

    manifest = TOOL.materialize(source_dir=source, output_dir=output, dataset_id="dual-view")

    row = json.loads((output / "train.jsonl").read_text())
    assert row["joinedRawText"] == "RAW train"
    assert row["target"]["documentPatch"]["billOfLadingNumber"] == "train"
    assert row["modelInputText"].startswith("<raw_ocr_text>\nRAW train")
    assert "<glm_ocr_table_view>" in row["modelInputText"]
    assert row["modelInputTextSha256"] == sha256_bytes(row["modelInputText"].encode())
    assert manifest["documents"] == 2
    assert manifest["input_contract"]["raw_text_is_value_truth_boundary"] is True


def test_materialize_rejects_drifted_table_digest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = source / "train.jsonl"
    row = json.loads(path.read_text())
    row["auxiliaryViews"]["glmOcrTableRecognition"]["joinedTableTextSha256"] = "0" * 64
    payload = canonical_json_bytes(row) + b"\n"
    path.write_bytes(payload)
    manifest = json.loads((source / "manifest.json").read_text())
    next(item for item in manifest["files"] if item["kind"] == "train_records")["sha256"] = (
        sha256_bytes(payload)
    )
    (source / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="joinedTableText SHA-256 mismatch"):
        TOOL.materialize(source_dir=source, output_dir=tmp_path / "output", dataset_id="bad")


def test_materialize_can_bind_current_targets_without_changing_ocr_views(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_row = json.loads((source / "train.jsonl").read_text())
    rows = []
    for split in ("train", "validation"):
        row = json.loads((source / f"{split}.jsonl").read_text())
        row["target"] = {"documentPatch": {"billOfLadingNumber": f"CURRENT-{split}"}}
        rows.append(row)
    target_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    target_path = tmp_path / "targets.jsonl"
    target_path.write_bytes(target_payload)

    output = tmp_path / "output"
    manifest = TOOL.materialize(
        source_dir=source,
        output_dir=output,
        dataset_id="current-targets",
        target_records=target_path,
        target_records_sha256=sha256_bytes(target_payload),
        target_records_count=2,
    )

    projected = json.loads((output / "train.jsonl").read_text())
    assert projected["target"]["documentPatch"]["billOfLadingNumber"] == "CURRENT-train"
    assert projected["joinedRawText"] == source_row["joinedRawText"]
    assert projected["auxiliaryViews"] == source_row["auxiliaryViews"]
    assert manifest["target_source"]["matched_records"] == 2
