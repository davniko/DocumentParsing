"""Negative controls for the finite, source-grounded retained-row repair."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "docs/analysis/synthesis-refresh900-20261014"
spec = importlib.util.spec_from_file_location("f256_retained", HERE / "f256_retained_rows.py")
repair_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair_module)
authority_spec = importlib.util.spec_from_file_location(
    "f256_authority", HERE / "f256_seal_rebase.py"
)
authority_module = importlib.util.module_from_spec(authority_spec)
authority_spec.loader.exec_module(authority_module)


def source_row():
    row = {
        "documentId": "syn_full_v7_test_seal_repair",
        "sourceDocumentId": repair_module.SID,
        "joinedRawText": "--- PAGE 1 ---\nAAAA0000001\nSEAL OLD1\nC/S\nBOLT51026581\n"
        "BBBB0000002\nSEAL OLD2\nC/S\nBOLT51026582\n",
        "positionedText": "--- PAGE 1 ---\nAAAA0000001 || 100,100\nSEAL OLD1 || 100,110\n"
        "C/S || 100,120\nBOLT51026581 || 100,130\nBBBB0000002 || 100,140\n"
        "SEAL OLD2 || 100,150\nC/S || 100,160\nBOLT51026582 || 100,170\n",
        "target": {
            "containerInformation": [
                {"equipmentIdentifier": "AAAA0000001", "sealNumbers": ["OLD1"]},
                {"equipmentIdentifier": "BBBB0000002", "sealNumbers": ["OLD2"]},
            ],
            "negotiability": "non_negotiable",
        },
    }
    for field in ("joinedRawText", "positionedText"):
        row[field + "Sha256"] = repair_module.digest(row[field].encode())
    return row


def test_synthetic_second_seals_preserve_text_width_geometry_and_other_targets():
    before = source_row()
    after, edits, added, changed = repair_module.repair(before, real=False)
    assert before == source_row()
    assert len(edits) == len(added) == 2
    assert repair_module.positional_suffixes(after["positionedText"]) == (
        repair_module.positional_suffixes(before["positionedText"])
    )
    assert after["target"]["negotiability"] == before["target"]["negotiability"]
    assert len(after["joinedRawText"].encode()) == len(before["joinedRawText"].encode())
    assert all(
        value not in repair_module.SEALS and value.startswith("BOLT") for value in added.values()
    )
    assert after == repair_module.repair(before, real=False)[0]
    assert changed == [
        "joinedRawText",
        "joinedRawTextSha256",
        "positionedText",
        "positionedTextSha256",
        "target",
    ]


def test_real_second_seals_add_only_supported_labels():
    before = source_row()
    before["documentId"] = repair_module.SID
    after, edits, added, changed = repair_module.repair(before, real=True)
    assert edits == [] and changed == ["target"]
    assert set(added.values()) == set(repair_module.SEALS)
    assert after["joinedRawText"] == before["joinedRawText"]
    assert after["positionedText"] == before["positionedText"]


@pytest.mark.parametrize("field", ["joinedRawText", "positionedText"])
def test_second_seal_requires_exact_text_in_both_input_views(field):
    row = source_row()
    row[field] = row[field].replace("BOLT51026582", "")
    with pytest.raises(ValueError, match="printed second seal occurrence"):
        repair_module.repair(row, real=False)


def test_duplicate_second_seal_occurrence_is_not_guessed():
    row = source_row()
    row["joinedRawText"] += "BOLT51026581\n"
    with pytest.raises(ValueError, match="printed second seal occurrence"):
        repair_module.repair(row, real=False)


def test_container_order_cannot_silently_reassign_second_seals():
    row = source_row()
    row["target"]["containerInformation"].reverse()
    with pytest.raises(ValueError, match="different container"):
        repair_module.repair(row, real=False)


def test_already_present_second_seals_are_not_duplicated():
    row = source_row()
    row["target"]["containerInformation"][0]["sealNumbers"].append("BOLT51026581")
    with pytest.raises(ValueError, match="two one-seal"):
        repair_module.repair(row, real=False)


def test_wrapped_training_target_is_rejected_instead_of_adapted():
    row = source_row()
    row["target"] = {"documentPatch": deepcopy(row["target"]), "schemaVersion": "7.0.0"}
    with pytest.raises(ValueError, match="direct reduced object"):
        repair_module.repair(row, real=False)


def test_all_actual31_staged_rows_have_exactly_two_added_seals():
    receipt = json.loads((HERE / "f256-source-rebase/retained-row-corrections.json").read_text())
    assert len(receipt["records"]) == 31
    for record in receipt["records"]:
        before = json.loads((ROOT / record["beforePath"]).read_text())
        expected = json.loads((ROOT / record["afterPath"]).read_text())
        actual, *_ = repair_module.repair(before, real=record["pool"] == "real")
        assert actual == expected


def authority_fixture(path):
    path.mkdir()
    (path / "alignments").mkdir()
    manifest = {
        "alignment_sha256": {}, "source_records": [], "split_records": {}, "output_sha256": {}
    }
    for split in ("train", "validation"):
        sid = f"doc_{split}"
        raw = f"--- PAGE 1 ---\n{split.upper()}\n"
        positioned = f"--- PAGE 1 ---\n{split.upper()} || 100,200\n"
        row = {"documentId": sid, "joinedRawText": raw, "positionedText": positioned, "target": {}}
        row["joinedRawTextSha256"] = authority_module.digest(raw.encode())
        row["positionedTextSha256"] = authority_module.digest(positioned.encode())
        alignment = {
            "document_id": sid,
            "original_input_sha256": row["joinedRawTextSha256"],
            "positioned_input_sha256": row["positionedTextSha256"],
            "lines": [],
        }
        payload = authority_module.json_artifact_bytes(alignment)
        (path / "alignments" / f"{sid}.json").write_bytes(payload)
        pin = authority_module.digest(payload)
        manifest["alignment_sha256"][f"{sid}.json"] = pin
        manifest["source_records"].append(
            {
                "documentId": sid,
                "split": split,
                "alignmentSha256": pin,
                "sourceInputSha256": row["joinedRawTextSha256"],
                "positionedInputSha256": row["positionedTextSha256"],
            }
        )
        (path / f"{split}.jsonl").write_bytes(json.dumps(row).encode() + b"\n")
        manifest["split_records"][split] = 1
    for name in ("prompt-schema.json", "target-schema.json", "task-constraints.json"):
        (path / name).write_bytes(b"{}\n")
    for name in authority_module.AUTHORITY_CORE_FILES - {"manifest.json"}:
        manifest["output_sha256"][name] = authority_module.digest((path / name).read_bytes())
    (path / "manifest.json").write_bytes(authority_module.json_artifact_bytes(manifest))
    return manifest


def test_authority_publication_includes_train_and_validation_alignments(tmp_path):
    original = tmp_path / "original"
    revised = tmp_path / "revised"
    destination = tmp_path / "published"
    authority_fixture(original)
    revised.mkdir()
    for name in authority_module.AUTHORITY_CORE_FILES:
        (revised / name).write_bytes((original / name).read_bytes())
    expected = authority_module.inherit_authority_alignments(original, revised)
    assert authority_module.publish_authority_bundle(revised, destination) == expected
    assert authority_module.publish_authority_bundle(revised, destination) == expected
    assert expected["splitRecords"] == {"train": 1, "validation": 1}
    assert len(expected["alignmentFiles"]) == 2


def test_incomplete_authority_is_rejected_before_any_destination_write(tmp_path):
    original = tmp_path / "original"
    destination = tmp_path / "published"
    authority_fixture(original)
    for path in (original / "alignments").iterdir():
        path.unlink()
    (original / "alignments").rmdir()
    with pytest.raises(ValueError, match="all six core files and alignments"):
        authority_module.publish_authority_bundle(original, destination)
    assert not destination.exists()


@pytest.mark.parametrize("corruption", ["alignment_symlink", "conflicting_core"])
def test_authority_destination_conflicts_are_rejected_before_writes(tmp_path, corruption):
    original = tmp_path / "original"
    destination = tmp_path / "published"
    authority_fixture(original)
    destination.mkdir()
    if corruption == "alignment_symlink":
        (destination / "alignments").symlink_to(original / "alignments", target_is_directory=True)
    else:
        (destination / "train.jsonl").write_bytes(b"CONFLICT\n")
    before = {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        authority_module.publish_authority_bundle(original, destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()} == before
    assert not (destination / "manifest.json").exists()


@pytest.mark.parametrize("corruption", ["missing", "extra", "symlink", "bytes", "wrong_document"])
def test_authority_alignment_corruptions_are_rejected(tmp_path, corruption):
    path = tmp_path / "authority"
    manifest = authority_fixture(path)
    alignment = path / "alignments/doc_train.json"
    if corruption == "missing":
        alignment.unlink()
    elif corruption == "extra":
        (path / "alignments/unpinned.json").write_bytes(b"{}")
    elif corruption == "symlink":
        saved = tmp_path / "saved.json"
        saved.write_bytes(alignment.read_bytes())
        alignment.unlink()
        alignment.symlink_to(saved)
    elif corruption == "bytes":
        alignment.write_bytes(alignment.read_bytes() + b" ")
    else:
        value = json.loads(alignment.read_bytes())
        value["document_id"] = "doc_wrong"
        alignment.write_bytes(authority_module.json_artifact_bytes(value))
        pin = authority_module.digest(alignment.read_bytes())
        manifest["alignment_sha256"][alignment.name] = pin
        manifest["source_records"][0]["alignmentSha256"] = pin
        (path / "manifest.json").write_bytes(authority_module.json_artifact_bytes(manifest))
    with pytest.raises(ValueError):
        authority_module.validate_authority_bundle(path)


@pytest.mark.parametrize("field", ["joinedRawText", "positionedText"])
def test_authority_alignment_must_match_actual_input_bytes(tmp_path, field):
    path = tmp_path / "authority"
    manifest = authority_fixture(path)
    row = json.loads((path / "train.jsonl").read_bytes())
    row[field] += "UNSUPPORTED\n"
    row[field + "Sha256"] = authority_module.digest(row[field].encode())
    (path / "train.jsonl").write_bytes(json.dumps(row).encode() + b"\n")
    manifest["output_sha256"]["train.jsonl"] = authority_module.digest(
        (path / "train.jsonl").read_bytes()
    )
    (path / "manifest.json").write_bytes(authority_module.json_artifact_bytes(manifest))
    with pytest.raises(ValueError, match="input/alignment hash mismatch"):
        authority_module.validate_authority_bundle(path)


def test_target_only_authority_revision_cannot_change_inherited_geometry_pins(tmp_path):
    original = tmp_path / "original"
    revised = tmp_path / "revised"
    manifest = authority_fixture(original)
    revised.mkdir()
    for name in authority_module.AUTHORITY_CORE_FILES:
        (revised / name).write_bytes((original / name).read_bytes())
    manifest["alignment_sha256"]["doc_train.json"] = "0" * 64
    (revised / "manifest.json").write_bytes(authority_module.json_artifact_bytes(manifest))
    with pytest.raises(ValueError, match="changed inherited geometry pins"):
        authority_module.inherit_authority_alignments(original, revised)
    assert not (revised / "alignments").exists()


@pytest.mark.parametrize("corruption", ["source_record_duplicate", "split", "input_row_duplicate"])
def test_authority_geometry_inventory_requires_unique_correct_split_rows(tmp_path, corruption):
    path = tmp_path / "authority"
    manifest = authority_fixture(path)
    if corruption == "source_record_duplicate":
        manifest["source_records"].append(deepcopy(manifest["source_records"][0]))
    elif corruption == "split":
        manifest["source_records"][0]["split"] = "validation"
    else:
        payload = (path / "train.jsonl").read_bytes() * 2
        (path / "train.jsonl").write_bytes(payload)
        manifest["output_sha256"]["train.jsonl"] = authority_module.digest(payload)
        manifest["split_records"]["train"] = 2
    (path / "manifest.json").write_bytes(authority_module.json_artifact_bytes(manifest))
    with pytest.raises(ValueError):
        authority_module.validate_authority_bundle(path)
