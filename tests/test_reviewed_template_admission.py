"""Scratch-only proof of cohort-sized admission, never live registry mutation."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/synthesis/reviewed_template_admission.py"
)
SPEC = importlib.util.spec_from_file_location("reviewed_admission", MODULE_PATH)
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)


def artifact(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(admission.json_artifact_bytes(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def sid(number):
    return "doc_" + f"{number:064x}"


def context_fixture(tmp_path, sources=30, old=2):
    audit = tmp_path / "audit"
    selection = [
        {
            "sourceId": sid(n + 100),
            "group": "thermal" if n < 25 else "nor",
            "evidence": "Reviewed source.",
        }
        for n in range(sources)
    ]
    artifact(audit / "selection/final.json", {"status": "FROZEN", "sources": selection})
    groups = {"thermal": 25, "nor": sources - 25}
    spec = {
        "schema_version": 1,
        "cohort_id": "thermal-test",
        "registry": "registry",
        "selection": {
            "path": "audit/selection/final.json",
            "sha256": admission.sha256_file(audit / "selection/final.json"),
        },
        "groups": groups,
        "variants_per_source": 3,
        "registry_sources_before": old,
        "authority_relative": "source-authorities/thermal-test/authority",
        "production_config": "configs/synthesis/thermal.yaml",
        "report": "docs/thermal.md",
    }
    artifact(audit / "cohort.json", spec)
    (tmp_path / "configs/synthesis").mkdir(parents=True)
    return admission.Context.load(tmp_path, Path("audit"))


def source_row(source):
    raw = f"SOURCE {source}"
    return {
        "documentId": source,
        "joinedRawText": raw,
        "positionedText": raw + " || 100,200",
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "billOfLadingNumber": "BL" + source[-8:],
                "goodsItemDetails": [{"description": "GOODS"}],
            },
        },
    }


def case(folder, row):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source.txt").write_text(row["joinedRawText"])
    (folder / "source-positioned.txt").write_text(row["positionedText"])
    artifact(folder / "target.json", row["target"])
    artifact(
        folder / "contract.json",
        {
            "sourceSha256": admission.digest(row["joinedRawText"].encode()),
            "targetSha256": admission.digest(row["target"]),
        },
    )
    artifact(folder / "template.json", {"source": row["documentId"]})
    artifact(folder / "alignment.json", {"source": row["documentId"]})
    return {name: admission.sha256_file(folder / name) for name in admission.ASSETS}


class StubCampaign:
    """Models only I/O/replay dependencies; domain accuracy is not claimed here."""

    executions: ClassVar[list] = []
    schema: ClassVar[dict] = {"type": "object", "properties": {"current": {"type": "string"}}}

    def __init__(self, root, config):
        self.root, self.config = root, config
        self.rows = admission.unique_rows(
            admission.records(root / config["dataset"] / "train.jsonl"), "documentId"
        )
        self.source_ids = config["source_ids"]
        self.calls = SimpleNamespace(
            task=SimpleNamespace(
                name="test",
                target_schema=lambda: self.schema,
                base_prompt_schema_sha256=lambda: "b" * 64,
                prompt_schema_json=lambda: json.dumps(self.schema),
            )
        )

    def validate_source_readiness(self):
        assert set(self.source_ids) <= self.rows.keys()

    def plans(self, source):
        return [(source, variant) for variant in range(1, 4)]

    def preflight(self, plans):
        self.executions.extend(plans)

    def replay_candidate(self, candidate):
        assert candidate["sourceDocumentId"] in self.source_ids


def prepared_fixture(tmp_path, monkeypatch, sources=30):
    ctx = context_fixture(tmp_path, sources)
    monkeypatch.setattr(admission, "Campaign", StubCampaign)
    old_ids = [sid(1), sid(2)]
    new_ids = list(admission.selected_sources(ctx))
    held_id = sid(500)
    prior = ctx.registry / "source-authorities/prior/base"
    source_rows = {source: source_row(source) for source in [*old_ids, *new_ids, held_id]}
    prior_train = [source_rows[source] for source in [*old_ids, new_ids[0]]]
    validation = [source_rows[held_id]]
    jsonl(prior / "train.jsonl", prior_train)
    jsonl(prior / "validation.jsonl", validation)
    prior_alignments = {}
    for row in prior_train + validation:
        path = prior / "alignments" / f"{row['documentId']}.json"
        artifact(path, {"source": row["documentId"]})
        prior_alignments[path.name] = admission.sha256_file(path)
    artifact(prior / "manifest.json", {"alignment_sha256": prior_alignments})
    original_cases = {}
    for source in old_ids:
        pins = case(ctx.registry / "cases" / source, source_rows[source])
        (ctx.registry / "cases" / source / "optional-proof.txt").write_text(
            "Keep this extra file.\n"
        )
        original_cases[source] = {
            "files": pins,
            "capability": {"family": "ambient"},
            "sourceDataset": str(prior.relative_to(tmp_path)),
        }
    ownership = {
        "schema_version": 1,
        "sources": {
            source: {
                "private_contacts": {"emails": {"issuer": {"domain_policy": "public_carrier"}}}
            }
            for source in old_ids
        },
    }
    auxiliary = {"schema_version": 1, "sources": {source: {} for source in old_ids}}
    for name, value in (("ownership.yaml", ownership), ("auxiliary.yaml", auxiliary)):
        (ctx.registry / name).write_bytes(admission.yaml_bytes(value))
    prior_pin = {
        "path": str(prior.relative_to(tmp_path)),
        "files": {
            name: admission.sha256_file(prior / name)
            for name in ("train.jsonl", "validation.jsonl", "manifest.json")
        },
    }
    manifest = {
        "sourceCount": len(old_ids),
        "cases": original_cases,
        "sharedFiles": {
            name: admission.sha256_file(ctx.registry / name) for name in admission.SHARED
        },
        "dependencies": {"sourceDataset": prior_pin},
        "admissions": [],
    }
    artifact(ctx.registry / "manifest.json", manifest)
    for name in (*admission.SHARED, "manifest.json"):
        dest = ctx.audit / "before" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ctx.registry / name).read_bytes())
    dataset_pins = {}
    for name in ("protected-old", "protected-new"):
        path = tmp_path / name / "train.jsonl"
        jsonl(path, [{"unchanged": name}])
        dataset_pins[name] = {"train.jsonl": admission.sha256_file(path)}
    before = {
        "registry": {
            name: admission.sha256_file(ctx.registry / name)
            for name in (*admission.SHARED, "manifest.json")
        },
        "cases": {source: original_cases[source]["files"] for source in old_ids},
        "sourceAuthority": prior_pin,
        "datasets": dataset_pins,
    }
    artifact(ctx.audit / "before/pins.json", before)
    staged = ctx.audit / "source-authority"
    jsonl(staged / "train.jsonl", [source_rows[source] for source in [*old_ids, *new_ids]])
    jsonl(staged / "validation.jsonl", validation)
    for source in [*old_ids, *new_ids, held_id]:
        artifact(staged / "alignments" / f"{source}.json", {"source": source})
    for source in new_ids:
        case(ctx.audit / "cases" / source, source_rows[source])
    config = {
        "dataset": "audit/source-authority",
        "output": "out",
        "source_ids": new_ids,
        "variants_per_source": 3,
        "seed": 123,
        "capabilities": {source: {"family": "ambient"} for source in new_ids},
        "source_contracts": "audit/cases",
        "historical_catalog": "audit/cases",
        "ownership": "audit/ownership.yaml",
        "auxiliary": "audit/auxiliary.yaml",
        "task_constraints": "constraints.json",
        "positions": {"output_subdirectory": "positioned"},
    }
    ctx.config_path.write_bytes(admission.yaml_bytes(config))
    staged_owners, staged_auxiliary = deepcopy(ownership), deepcopy(auxiliary)
    reviews = []
    for source in new_ids:
        worker = ctx.audit / "workers" / source
        worker.mkdir(parents=True)
        for name in ("source.txt", "target.json", "contract.json", "template.json"):
            (worker / name).write_bytes((ctx.audit / "cases" / source / name).read_bytes())
        artifact(worker / "adjudication.json", {"reviewed": source})
        artifact(worker / "ownership.json", {})
        artifact(worker / "auxiliary.json", {})
        staged_owners["sources"][source] = {}
        staged_auxiliary["sources"][source] = {}
        reviews.append(
            {
                "sourceId": source,
                "caseDirectory": str(worker.relative_to(tmp_path)),
                "status": "PASS_SOURCE_POLICY_REVIEW",
                "reviewedCompleteOcr": True,
                "reviewedCurrentTargetAndAdjudication": True,
                "openFindings": [],
                "assetsSha256": {
                    path.name: admission.sha256_file(path) for path in worker.iterdir()
                },
            }
        )
    for name, value in (("ownership.yaml", staged_owners), ("auxiliary.yaml", staged_auxiliary)):
        (ctx.audit / name).write_bytes(admission.yaml_bytes(value))
    (tmp_path / "policy.md").write_text("Reviewed current policy.\n")
    policy = {
        "status": "PASS",
        "sourcesReviewed": sources,
        "sourcePolicyPassCount": sources,
        "openFindingCount": 0,
        "policyPath": "policy.md",
        "policySha256": admission.sha256_file(tmp_path / "policy.md"),
        "sources": reviews,
    }
    artifact(ctx.audit / "source-policy-review.json", policy)
    artifact(
        tmp_path / "constraints.json",
        {
            "targetSchemaSha256": admission.canonical_json_sha256(StubCampaign.schema),
            "basePromptSchemaSha256": "b" * 64,
        },
    )
    artifact(
        tmp_path / "data/corpora/paddle-bl-swb-filtered-max5/selection.json",
        {
            "documents": [
                {"document_ids": [source], "source_sha256": admission.digest(source.encode())}
                for source in source_rows
            ]
        },
    )
    artifact(
        ctx.audit / "spatial/final/manifest.json",
        {
            "config": {
                "paddle_run": "paddle",
                "alignment": {"grid_size": 1000},
                "selection_manifest": "data/corpora/paddle-bl-swb-filtered-max5/selection.json",
            },
            "source_sha256": {},
        },
    )
    plain, positioned, decisions = [], [], []
    output, positions = tmp_path / "out", tmp_path / "out/positioned"
    for source in new_ids:
        for variant in range(1, 4):
            sample = f"sample-{source}-{variant}"
            raw = {
                "documentId": sample,
                "sourceDocumentId": source,
                "joinedRawText": f"RENDER {sample}",
                "target": source_rows[source]["target"]["documentPatch"],
            }
            final = {**raw, "positionedText": raw["joinedRawText"] + " || 100,200"}
            plain.append(raw)
            positioned.append(final)
            candidate = {**raw, "target": source_rows[source]["target"]}
            artifact(output / "candidates" / f"{sample}.json", candidate)
            artifact(positions / f"{sample}.json", {"sample": sample})
            decisions.append(
                {
                    "sampleId": sample,
                    "sourceId": source,
                    "status": "PASS",
                    "sourceInputSha256": admission.digest(
                        source_rows[source]["joinedRawText"].encode()
                    ),
                    "renderedInputSha256": admission.digest(raw["joinedRawText"].encode()),
                    "positionedInputSha256": admission.digest(final["positionedText"].encode()),
                    "targetSha256": admission.digest(final["target"]),
                    "candidateFileSha256": admission.sha256_file(
                        output / "candidates" / f"{sample}.json"
                    ),
                    "geometryReceiptSha256": admission.sha256_file(positions / f"{sample}.json"),
                    "notes": "Reviewed final text, target and geometry.",
                }
            )
    jsonl(output / "dataset.jsonl", plain)
    jsonl(positions / "dataset.jsonl", positioned)
    artifact(
        ctx.audit / "manual-review.json",
        {
            "status": "PASS",
            "sources": sources,
            "samples": sources * 3,
            "configFileSha256": admission.sha256_file(ctx.config_path),
            "plainDatasetSha256": admission.sha256_file(output / "dataset.jsonl"),
            "positionedDatasetSha256": admission.sha256_file(positions / "dataset.jsonl"),
            "records": decisions,
        },
    )
    artifact(
        ctx.audit / "validation.json",
        {
            "status": "PASS",
            "sources": sources,
            "samples": sources * 3,
            "trainingDatasetUnchanged": True,
            "targetSchemaSha256": admission.canonical_json_sha256(StubCampaign.schema),
            "sourcePolicyReviewSha256": admission.sha256_file(
                ctx.audit / "source-policy-review.json"
            ),
            "plainDatasetSha256": admission.sha256_file(output / "dataset.jsonl"),
            "positionedDatasetSha256": admission.sha256_file(positions / "dataset.jsonl"),
            "datasetSha256": admission.sha256_file(positions / "dataset.jsonl"),
            "results": [
                {
                    "sampleId": row["documentId"],
                    "sourceId": row["sourceDocumentId"],
                    "status": "PASS",
                }
                for row in positioned
            ],
            "negativeControls": [
                {"sourceId": source, "mutations": [{"mutation": "corruption", "rejected": True}]}
                for source in new_ids
            ],
            "sourceGeometryAudit": {"sources": sources, "unprovedRegionReuse": 0},
        },
    )
    artifact(
        ctx.audit / "regression-before-admission.json",
        {"status": "PASS", "sources": 2, "plans": 6, "unchangedExistingCaseFiles": True},
    )
    artifact(ctx.audit / "preflight.json", {"status": "PASS"})
    return ctx, before, manifest


@pytest.mark.parametrize("sources", [29, 30])
def test_prepare_derives_sizes_and_preserves_all_old_case_metadata(tmp_path, monkeypatch, sources):
    ctx, before, manifest = prepared_fixture(tmp_path, monkeypatch, sources)
    plan = admission.prepare(ctx)
    assert plan.report["samples"] == sources * 3
    assert plan.report["sourceAuthorityTrain"] == sources + 2
    proposed = json.loads(plan.files[Path("manifest.json")])
    assert proposed["sourceCount"] == sources + 2
    assert all(proposed["cases"][source] == entry for source, entry in manifest["cases"].items())
    assert proposed["admissions"][-1]["newWiderSources"] == sources - 1
    assert proposed["admissions"][-1]["revisedUnregisteredSourceRows"] == 1
    assert admission.read(ctx.registry / "manifest.json") == manifest
    admission.protected_dataset_pins(ctx, before)
    assert (
        not ctx.production_config.exists() and not (ctx.registry / ctx.authority_relative).exists()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "old_owner",
        "old_capability",
        "unselected_row",
        "role_presence",
        "old_positions",
        "protected_dataset",
        "schema_gate",
        "policy_gate",
        "missing_manual",
        "group_underfill",
        "unrejected_control",
    ],
)
def test_prepare_rejects_every_unapproved_old_data_or_review_change(
    tmp_path, monkeypatch, mutation
):
    ctx, _before, _manifest = prepared_fixture(tmp_path, monkeypatch)
    source = sid(1)
    if mutation == "old_owner":
        path = ctx.audit / "ownership.yaml"
        value = admission.yaml.safe_load(path.read_text())
        value["sources"][source]["private_contacts"]["emails"]["issuer"]["domain_policy"] = (
            "private_company"
        )
        path.write_bytes(admission.yaml_bytes(value))
    elif mutation == "old_capability":
        path = ctx.registry / "manifest.json"
        value = admission.read(path)
        value["cases"][source]["capability"]["family"] = "changed"
        artifact(path, value)
    elif mutation in {"unselected_row", "role_presence"}:
        path = ctx.audit / "source-authority/train.jsonl"
        rows = admission.records(path)
        if mutation == "unselected_row":
            rows[0]["joinedRawText"] += " ALTERED"
        else:
            rows[0]["target"]["documentPatch"]["parties"] = {"consignee": {"name": "UNSUPPORTED"}}
        jsonl(path, rows)
    elif mutation == "old_positions":
        (ctx.audit / "source-authority/alignments" / f"{source}.json").write_text("{}")
    elif mutation == "protected_dataset":
        (tmp_path / "protected-new/train.jsonl").write_text("changed")
    elif mutation == "missing_manual":
        path = ctx.audit / "manual-review.json"
        value = admission.read(path)
        value["records"].pop()
        artifact(path, value)
    elif mutation == "group_underfill":
        path = tmp_path / ctx.spec.selection.path
        value = admission.read(path)
        value["sources"][-1]["group"] = "thermal"
        artifact(path, value)
    else:
        path = ctx.audit / "validation.json"
        value = admission.read(path)
        if mutation == "schema_gate":
            value["targetSchemaSha256"] = "a" * 64
        elif mutation == "policy_gate":
            value["sourcePolicyReviewSha256"] = "a" * 64
        else:
            value["negativeControls"][0]["mutations"][0]["rejected"] = False
        artifact(path, value)
    with pytest.raises(ValueError):
        admission.prepare(ctx)
    assert not (ctx.audit / "admission-transaction").exists()


@pytest.mark.parametrize("column", ["plain", "preserved", "positioned", "missing_positioned"])
def test_manual_text_columns_are_independently_pinned(tmp_path, monkeypatch, column):
    ctx, _, _ = prepared_fixture(tmp_path, monkeypatch)
    output, positions = tmp_path / "out", tmp_path / "out/positioned"
    plain = admission.unique_rows(admission.records(output / "dataset.jsonl"), "documentId")
    positioned = admission.unique_rows(admission.records(positions / "dataset.jsonl"), "documentId")
    sample = next(iter(plain))
    if column == "plain":
        plain[sample]["joinedRawText"] += " ALTERED"
    elif column == "preserved":
        positioned[sample]["joinedRawText"] += " ALTERED"
    elif column == "positioned":
        positioned[sample]["positionedText"] += " ALTERED"
    else:
        positioned[sample].pop("positionedText")
    review = admission.read(ctx.audit / "manual-review.json")
    for folder, rows, key in (
        (output, plain, "plainDatasetSha256"),
        (positions, positioned, "positionedDatasetSha256"),
    ):
        jsonl(folder / "dataset.jsonl", list(rows.values()))
        review[key] = admission.sha256_file(folder / "dataset.jsonl")
    sources = admission.unique_rows(
        admission.records(ctx.audit / "source-authority/train.jsonl"), "documentId"
    )
    with pytest.raises((ValueError, KeyError)):
        admission.manual_review_gate(ctx, review, sources, plain, positioned, output, positions)


def test_live_precommit_exercises_new_and_every_old_source_before_manifest_commit(
    tmp_path, monkeypatch
):
    ctx, before, original = prepared_fixture(tmp_path, monkeypatch)
    plan = admission.prepare(ctx)
    StubCampaign.executions = []

    def live(current):
        assert admission.read(ctx.registry / "manifest.json") == original
        admission.validate_installed(current)

    admission.install(plan, validate_live=live)
    assert len(StubCampaign.executions) == (30 + 2) * 3
    assert admission.read(ctx.registry / "manifest.json")["sourceCount"] == 32
    assert all(
        (ctx.registry / "cases" / source / "optional-proof.txt").read_text()
        == "Keep this extra file.\n"
        for source in original["cases"]
    )
    admission.protected_dataset_pins(ctx, before)


def test_failed_precommit_rolls_back_and_keeps_recoverable_new_payload(tmp_path, monkeypatch):
    ctx, _, original = prepared_fixture(tmp_path, monkeypatch)
    plan = admission.prepare(ctx)

    def fail(_):
        raise ValueError("Injected live proof failure")

    with pytest.raises(ValueError, match="Injected"):
        admission.install(plan, validate_live=fail)
    assert admission.read(ctx.registry / "manifest.json") == original
    assert (
        not ctx.production_config.exists() and not (ctx.registry / ctx.authority_relative).exists()
    )
    assert (
        ctx.audit / "admission-transaction/payload" / ctx.authority_relative / "train.jsonl"
    ).exists()
    assert all(
        not (ctx.registry / "cases" / source).exists() for source in plan.config["source_ids"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unsafe",
        "scope",
        "case_missing",
        "authority_missing",
        "extra_case",
        "existing_config",
        "existing_authority",
        "existing_case",
        "existing_transaction",
    ],
)
def test_install_rejects_scope_and_destination_corruption(tmp_path, monkeypatch, mutation):
    ctx, _, original = prepared_fixture(tmp_path, monkeypatch)
    plan = admission.prepare(ctx)
    if mutation == "unsafe":
        plan.files[Path("../../escape")] = b"NO"
    elif mutation == "scope":
        plan.files[Path("unrelated.json")] = b"NO"
    elif mutation == "case_missing":
        plan.files.pop(Path("cases") / plan.config["source_ids"][0] / "source.txt")
    elif mutation == "authority_missing":
        plan.files.pop(ctx.authority_relative / "train.jsonl")
    elif mutation == "extra_case":
        plan.files[Path("cases") / sid(900) / "source.txt"] = b"NO"
    else:
        path = {
            "existing_config": ctx.production_config,
            "existing_authority": ctx.registry / ctx.authority_relative,
            "existing_case": ctx.registry / "cases" / plan.config["source_ids"][0],
            "existing_transaction": ctx.audit / "admission-transaction",
        }[mutation]
        if mutation == "existing_config":
            path.write_text("Do not overwrite.")
        else:
            path.mkdir(parents=True)
    with pytest.raises(ValueError):
        admission.install(plan, validate_live=lambda _: None)
    assert admission.read(ctx.registry / "manifest.json") == original
