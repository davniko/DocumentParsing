"""Campaign helpers must stop on account refusal, not submit the next batch."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts/synthesis" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expansion_stops_new_batches_after_credit_refusal(tmp_path, monkeypatch):
    module = load_script("run_r28_expansion_phase")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path)
    jobs = []
    for number in range(1, 4):
        path = tmp_path / str(number) / "config.json"
        path.parent.mkdir()
        path.write_text("{}")
        jobs.append(
            {
                "config": str(path.relative_to(tmp_path)),
                "batch": number,
                "phase": "ordinary",
                "configSha256": module.sha256_file(path),
            }
        )
    (tmp_path / "selection.json").write_text(json.dumps({"cohortConfigs": jobs}))
    calls = []

    class Process:
        async def wait(self):
            return 0

    async def launch(*args, **kwargs):
        calls.append(args)
        kwargs["stdout"].write(b"ModelHTTPError: status_code: 402 Insufficient credits\n")
        return Process()

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", launch)
    with pytest.raises(RuntimeError):
        asyncio.run(module.run("ordinary", 1, 3, 1))
    assert len(calls) == 1
    receipt = json.loads((tmp_path / "1/execution-receipt.json").read_text())
    assert receipt["status"] == "PROVIDER_CREDITS_BLOCKED"
    assert not (tmp_path / "2/execution.log").exists()


def test_scoped_audit_requires_a_ledger_for_multiple_public_rows():
    module = load_script("audit_r28_multirow_wording")
    row = {
        "scenario": {"provenance": {}},
        "target": {"documentPatch": {"goodsItemDetails": [{"numberAndTypeOfPackages": [{}, {}]}]}},
    }
    with pytest.raises(ValueError, match="require their declared-package ledger"):
        module.packet(row)


def test_credit_resume_pins_only_exact_account_refusals(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/synthesis"))
    module = load_script("run_package_alias_batch")
    calls = tmp_path / "calls"
    calls.mkdir()
    for name, row in {
        "credit": {"output": None, "error": "status_code: 402 Insufficient credits"},
        "credit2": {"output": None, "error": "status_code: 402 This request requires more credits"},
        "invalid": {"output": None, "error": "UnexpectedModelBehavior"},
        "rate": {"output": None, "error": "status_code: 429"},
        "success": {"output": {}, "error": None},
    }.items():
        (calls / (name + ".json")).write_text(json.dumps(row))
    assert module.credit_refusal_pins(tmp_path) == [
        {"path": str(calls / name), "sha256": module.sha256_file(calls / name)}
        for name in ("credit.json", "credit2.json")
    ]
    assert module.transient_refusal_pins(tmp_path) == [
        {"path": str(calls / "rate.json"), "sha256": module.sha256_file(calls / "rate.json")}
    ]


def test_rebased_source_manifest_preserves_spatial_pins(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/synthesis"))
    module = load_script("repair_expansion_source_owners")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    old, new = tmp_path / "old", tmp_path / "new"
    for folder in (old, new):
        folder.mkdir()
        (folder / "train.jsonl").write_text(folder.name)
        (folder / "validation.jsonl").write_text("unchanged")
        (folder / "alignments").mkdir()
        (folder / "alignments/alignment.json").write_text("unchanged geometry")
    manifest = {
        "dataset": "old",
        "output_sha256": {
            n: module.sha256_file(old / n) for n in ("train.jsonl", "validation.jsonl")
        },
        "alignment_sha256": {
            "alignment.json": module.sha256_file(old / "alignments/alignment.json")
        },
        "config": {"alignment": {"grid_size": 1000}},
    }
    (old / "manifest.json").write_text(json.dumps(manifest))
    module.publish_rebased_manifest(old, new, "doc_reviewed")
    result = json.loads((new / "manifest.json").read_text())
    assert result["output_sha256"]["train.jsonl"] == module.sha256_file(new / "train.jsonl")
    assert result["alignment_sha256"] == manifest["alignment_sha256"]
    assert result["config"] == manifest["config"]
    assert result["dataset"] == "new"
    module.publish_rebased_manifest(old, new, "doc_reviewed")
    (new / "alignments/alignment.json").write_text("changed")
    with pytest.raises(ValueError, match="source alignment changed"):
        module.publish_rebased_manifest(old, new, "doc_reviewed")


def test_redraw_carries_complete_source_authority_without_changing_quotas(tmp_path, monkeypatch):
    module = load_script("manage_r28_expansion")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    config = {
        "source_ids": ["a", "b"],
        "capabilities": {"a": {"family": "frozen"}, "b": {}},
        "seed": 12,
        "template_sampling": {"source_counts": {"a": 2, "b": 1}},
        "declared_packages": {"source_group_counts": {"a": [1, 2], "b": [1]}, "contracts": "old"},
        "independent_preparation": {
            "mixed_source_ids": ["a", "b"],
            "thermal_goods_modes": {"a": "produce"},
        },
        "sampling": {},
        "independent": {"numeric_certificates": "old"},
        "aliases": {"presentations": {"certificates": "old", "short_weight": 0.3}},
    }
    authority = {
        "dataset": "repaired-data",
        "source_contracts": "repaired-cases",
        "ownership": "repaired-owners",
        "auxiliary": "repaired-auxiliary",
        "independent": {"numeric_certificates": "repaired-numeric"},
        "declared_packages": {"contracts": "repaired-declarations"},
        "aliases": {"presentations": {"certificates": "repaired-presentations"}},
    }
    paths = [tmp_path / name for name in ("config.json", "review.json", "authority.json")]
    for path, value in zip(
        paths, (config, {"redrawSources": {"a": "reviewed repair"}}, authority), strict=True
    ):
        path.write_text(json.dumps(value))
    module.redraw(paths[0], paths[1], 1, paths[2])
    result = json.loads((tmp_path / "redraw-1/config.json").read_text())
    for key in ("dataset", "source_contracts", "ownership", "auxiliary", "independent"):
        assert result[key] == authority[key]
    assert result["declared_packages"]["contracts"] == "repaired-declarations"
    assert result["aliases"]["presentations"] == {
        "certificates": "repaired-presentations",
        "short_weight": 0.3,
    }
    assert result["template_sampling"] == {"samples": 2, "source_counts": {"a": 2}}
    assert result["declared_packages"]["source_group_counts"] == {"a": [1, 2]}
    assert result["independent_preparation"] == {
        "mixed_source_ids": ["a"],
        "thermal_goods_modes": {"a": "produce"},
    }


def test_final_expansion_requires_exact_unique_source_quota():
    module = load_script("finalize_r28_expansion")
    rows = [
        {"documentId": str(i), "sourceDocumentId": "a" if i < 2400 else "b"} for i in range(2500)
    ]
    module.require_schedule(rows, {"a": 2400, "b": 100})
    with pytest.raises(ValueError, match="exactly 2500"):
        module.require_schedule(rows[:-1], {"a": 2400, "b": 100})
    with pytest.raises(ValueError, match="exactly 2500"):
        module.require_schedule([*rows[:-1], rows[0]], {"a": 2400, "b": 100})
    with pytest.raises(ValueError, match="approved original schedule"):
        module.require_schedule(rows, {"a": 2399, "b": 101})


@pytest.mark.parametrize(
    "field,value",
    [
        ("missingFreshSources", ["excluded-payable", "unreviewed"]),
        ("missingReplacementModes", {"source": [3]}),
        ("sourceDeltasWithoutFreshEvidence", {"source": ["ownership"]}),
    ],
)
def test_expansion_authority_promotion_rejects_incomplete_evidence(monkeypatch, field, value):
    module = load_script("certify_expansion_authorities")
    report = {
        "missingFreshSources": ["excluded-payable"],
        "missingReplacementModes": {},
        "sourceDeltasWithoutFreshEvidence": {},
    }
    report[field] = value
    monkeypatch.setattr(module, "investigate", lambda: ({}, {}, {}, {}, report))
    with pytest.raises(ValueError, match="current source or mode evidence incomplete"):
        module.run(publish=True)


def test_expansion_authority_pin_rejects_stale_receipt(tmp_path, monkeypatch):
    module = load_script("certify_expansion_authorities")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        module.checked({"path": path.name, "sha256": "0" * 64})


def test_final_census_preserves_text_and_coordinate_grid(monkeypatch):
    module = load_script("finalize_r28_expansion")
    monkeypatch.setattr(module, "feature_function", lambda: lambda target: {})
    row = {
        "documentId": "one",
        "target": {},
        "joinedRawText": "--- PAGE 1 ---\nA\nB",
        "positionedText": "--- PAGE 1 ---\nA || 0,1000\nB ||",
    }
    assert module.census([row])["coordinates"] == {
        "knownLines": 1,
        "eligibleNonemptyLines": 2,
        "coverage": 0.5,
    }
    with pytest.raises(ValueError, match="changed OCR text"):
        module.census([{**row, "positionedText": "--- PAGE 1 ---\nC || 0,1000\nB ||"}])
    with pytest.raises(ValueError, match="outside normalized grid"):
        module.census([{**row, "positionedText": "--- PAGE 1 ---\nA || 1001,1000\nB ||"}])


def test_final_append_rejects_changed_current_dataset(tmp_path, monkeypatch):
    module = load_script("finalize_r28_expansion")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "audit")
    monkeypatch.setattr(module, "DATA", tmp_path / "data")
    final = tmp_path / "audit/final"
    final.mkdir(parents=True)
    (tmp_path / "data").mkdir()
    data = tmp_path / "data/synthetic-train.jsonl"
    data.write_text("old\n")
    report = final / "audit.json"
    report.write_text(json.dumps({"datasetBeforeAppend": {data.name: module.sha256_file(data)}}))
    (final / "main-approval.json").write_text(
        json.dumps({"status": "APPROVED", "auditSha256": module.sha256_file(report)})
    )
    data.write_text("changed\n")
    with pytest.raises(ValueError, match="dataset changed after audit"):
        module.append()
    assert data.read_text() == "changed\n"
    assert not (final / "publication.json").exists()
