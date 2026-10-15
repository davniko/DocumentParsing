"""Fail-closed admission of a pinned, completely reviewed template cohort.

Paths and cardinality are explicit; domain-specific validation is a separately
pinned gate. Default operation is read-only. Only --apply publishes the registry.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import resource
import stat
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.atomic import atomic_publish_bytes, atomic_write_bytes, json_artifact_bytes
from document_ocr.hashing import canonical_json_sha256, sha256_bytes, sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_admission import (
    ASSETS,
    SHARED,
    merge_shared,
    pin_bytes,
    read,
    records,
    require,
    unique_rows,
    validate_case,
    validate_exclusion,
    verify_pins,
    yaml_bytes,
)
from document_ocr.synthesis.curated_campaign import Campaign

PositiveInt = Annotated[int, Field(strict=True, gt=0)]


def relative(path: str) -> Path:
    result = Path(path)
    require(
        bool(path) and not result.is_absolute() and ".." not in result.parts, f"Unsafe path: {path}"
    )
    require(result.parts and result != Path("."), f"Empty scoped path: {path}")
    return result


class SelectionPin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CohortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    cohort_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
    registry: str
    selection: SelectionPin
    groups: dict[str, PositiveInt] = Field(min_length=1)
    variants_per_source: Literal[3]
    registry_sources_before: PositiveInt
    authority_relative: str
    production_config: str
    report: str

    @model_validator(mode="after")
    def scoped_paths(self):
        for value in (self.registry, self.selection.path, self.production_config, self.report):
            relative(value)
        authority = relative(self.authority_relative)
        require(
            len(authority.parts) >= 3
            and authority.parts[:2] == ("source-authorities", self.cohort_id),
            "Immutable authority must be scoped to this cohort",
        )
        require(
            all(re.fullmatch(r"[a-z][a-z0-9_]*", key) for key in self.groups), "Unsafe cohort group"
        )
        return self


@dataclass(frozen=True)
class Context:
    root: Path
    audit: Path
    spec: CohortSpec

    @property
    def registry(self):
        return self.root / self.spec.registry

    @property
    def authority_relative(self):
        return Path(self.spec.authority_relative)

    @property
    def config_path(self):
        return self.audit / "config.yaml"

    @property
    def production_config(self):
        return self.root / self.spec.production_config

    @property
    def sources(self):
        return sum(self.spec.groups.values())

    @property
    def samples(self):
        return self.sources * self.spec.variants_per_source

    @classmethod
    def load(cls, root, audit):
        root = root.resolve()
        audit = (root / audit).resolve()
        require(audit.is_relative_to(root), "Audit directory must be inside the project")
        return cls(root, audit, CohortSpec.model_validate(read(audit / "cohort.json")))


def selected_sources(ctx):
    path = ctx.root / ctx.spec.selection.path
    require(sha256_file(path) == ctx.spec.selection.sha256, "Frozen source selection changed")
    selection = read(path)
    require(selection["status"] == "FROZEN", "Source selection is not frozen")
    sources = unique_rows(selection["sources"], "sourceId")
    require(len(sources) == ctx.sources, "Frozen source selection cardinality differs")
    require(
        Counter(row["group"] for row in sources.values()) == ctx.spec.groups,
        "Frozen source selection group coverage differs",
    )
    require(
        all(re.fullmatch(r"doc_[0-9a-f]{64}", sid) for sid in sources),
        "Unsafe selected source identifier",
    )
    return sources


def protected_dataset_pins(ctx, before):
    require(bool(before["datasets"]), "Protected training dataset inventory is empty")
    pins = {}
    for folder, files in before["datasets"].items():
        require(bool(files), f"Protected dataset has no files: {folder}")
        for name, expected in files.items():
            pins[ctx.root / relative(folder) / relative(name)] = expected
    verify_pins(pins)
    return pins


def source_policy_gate(ctx, report, config):
    require(
        report["status"] == "PASS"
        and report["sourcesReviewed"] == report["sourcePolicyPassCount"] == ctx.sources
        and report["openFindingCount"] == 0,
        "Source-policy review is incomplete",
    )
    policy = ctx.root / relative(report["policyPath"])
    require(sha256_file(policy) == report["policySha256"], "Reviewed policy changed")
    reviews = unique_rows(report["sources"], "sourceId")
    require(set(reviews) == set(config["source_ids"]), "Source-policy coverage differs")
    shared = {
        key: yaml.safe_load((ctx.root / config[key]).read_text())
        for key in ("ownership", "auxiliary")
    }
    pins = {policy: report["policySha256"]}
    required = {
        "source.txt",
        "target.json",
        "adjudication.json",
        "ownership.json",
        "auxiliary.json",
        "contract.json",
        "template.json",
    }
    for sid, review in reviews.items():
        require(
            review["status"] == "PASS_SOURCE_POLICY_REVIEW"
            and review["reviewedCompleteOcr"] is True
            and review["reviewedCurrentTargetAndAdjudication"] is True
            and review["openFindings"] == [],
            f"Unresolved source review: {sid}",
        )
        require(required <= review["assetsSha256"].keys(), f"Incomplete reviewed assets: {sid}")
        worker = ctx.root / relative(review["caseDirectory"])
        for name, expected in review["assetsSha256"].items():
            path = worker / relative(name)
            require(sha256_file(path) == expected, f"Reviewed source asset changed: {sid}/{name}")
            pins[path] = expected
        for name in ("source.txt", "target.json", "contract.json", "template.json"):
            common = ctx.root / config["source_contracts"] / sid / name
            require(
                sha256_file(common) == review["assetsSha256"][name],
                f"Worker/common asset differs: {sid}/{name}",
            )
            pins[common] = review["assetsSha256"][name]
        for name, value in shared.items():
            require(
                read(worker / f"{name}.json") == value["sources"][sid],
                f"Worker/common {name} differs: {sid}",
            )
    for name in shared:
        pins[ctx.root / config[name]] = sha256_file(ctx.root / config[name])
    return pins


def manual_review_gate(ctx, report, source_rows, plain, positioned, output, positions):
    require(
        report["status"] == "PASS"
        and report["sources"] == ctx.sources
        and report["samples"] == ctx.samples,
        "Final manual review is incomplete",
    )
    require(
        report["configFileSha256"] == sha256_file(ctx.config_path), "Manual review config changed"
    )
    files = {
        output / "dataset.jsonl": report["plainDatasetSha256"],
        positions / "dataset.jsonl": report["positionedDatasetSha256"],
    }
    verify_pins(files)
    decisions = unique_rows(report["records"], "sampleId")
    require(
        len(decisions) == ctx.samples and set(decisions) == set(plain) == set(positioned),
        "Manual sample coverage differs",
    )
    counts = Counter(row["sourceDocumentId"] for row in positioned.values())
    require(
        len(counts) == ctx.sources and set(counts.values()) == {ctx.spec.variants_per_source},
        "Manual source/variant coverage differs",
    )
    pins = {ctx.config_path: report["configFileSha256"], **files}
    for sample, decision in decisions.items():
        final, raw = positioned[sample], plain[sample]
        sid = final["sourceDocumentId"]
        require(
            decision["status"] == "PASS"
            and isinstance(decision["notes"], str)
            and bool(decision["notes"].strip()),
            f"Unresolved manual review: {sample}",
        )
        require(
            decision["sourceId"] == sid == raw["sourceDocumentId"],
            f"Manual source differs: {sample}",
        )
        require(final["target"] == raw["target"], f"Positioning changed target: {sample}")
        require(
            final["joinedRawText"] == raw["joinedRawText"],
            f"Positioning changed preserved plain text: {sample}",
        )
        for key, actual in (
            ("sourceInputSha256", digest(source_rows[sid]["joinedRawText"].encode())),
            ("renderedInputSha256", digest(raw["joinedRawText"].encode())),
            ("positionedInputSha256", digest(final["positionedText"].encode())),
            ("targetSha256", digest(final["target"])),
        ):
            require(decision[key] == actual, f"Reviewed semantic value changed: {sample}/{key}")
        for key, path in (
            ("candidateFileSha256", output / "candidates" / f"{sample}.json"),
            ("geometryReceiptSha256", positions / f"{sample}.json"),
        ):
            require(
                decision[key] == sha256_file(path), f"Reviewed artifact changed: {sample}/{key}"
            )
            pins[path] = decision[key]
    return pins


def validation_gate(ctx, report, task, output, positions):
    require(
        report["status"] == "PASS"
        and report["sources"] == ctx.sources
        and report["samples"] == ctx.samples
        and report["trainingDatasetUnchanged"] is True,
        "Domain/publication validation is incomplete",
    )
    checks = unique_rows(report["results"], "sampleId")
    require(
        len(checks) == ctx.samples and all(row["status"] == "PASS" for row in checks.values()),
        "Unresolved domain validation",
    )
    require(
        report["targetSchemaSha256"] == canonical_json_sha256(task.target_schema()),
        "Domain validation schema changed",
    )
    require(
        report["sourcePolicyReviewSha256"] == sha256_file(ctx.audit / "source-policy-review.json"),
        "Domain validation source-policy review changed",
    )
    require(
        report["plainDatasetSha256"] == sha256_file(output / "dataset.jsonl"),
        "Validated plain publication changed",
    )
    require(
        report["positionedDatasetSha256"]
        == report["datasetSha256"]
        == sha256_file(positions / "dataset.jsonl"),
        "Validated positioned publication changed",
    )
    controls = unique_rows(report["negativeControls"], "sourceId")
    require(
        set(controls) == set(selected_sources(ctx))
        and all(
            row["mutations"] and all(mutation["rejected"] is True for mutation in row["mutations"])
            for row in controls.values()
        ),
        "Domain/geometry corruption-control coverage is incomplete",
    )
    require(
        report["sourceGeometryAudit"]["sources"] == ctx.sources
        and report["sourceGeometryAudit"]["unprovedRegionReuse"] == 0,
        "Source geometry ownership audit is incomplete",
    )
    return checks


def authority_manifest(
    ctx, task, config, train, validation, output_hashes, alignments, pins, prior, selection
):
    constraints = read(ctx.root / config["task_constraints"])
    require(
        constraints["targetSchemaSha256"] == canonical_json_sha256(task.target_schema()),
        "Current schema differs from frozen constraints",
    )
    require(
        constraints["basePromptSchemaSha256"] == task.base_prompt_schema_sha256(),
        "Current base prompt differs from frozen constraints",
    )
    spatial = read(ctx.audit / "spatial/final/manifest.json")
    return {
        "schema_version": 1,
        "status": "certified_reviewed_cohort_source_authority",
        "dataset": str((ctx.registry / ctx.authority_relative).relative_to(ctx.root)),
        "task": task.name,
        "targetSchemaSha256": canonical_json_sha256(task.target_schema()),
        "basePromptSchemaSha256": task.base_prompt_schema_sha256(),
        "taskConstraintsSha256": sha256_file(ctx.root / config["task_constraints"]),
        "train_real": len(train),
        "train_synthetic": 0,
        "validation_real": len(validation),
        "split_records": {"train": len(train), "validation": len(validation)},
        "output_sha256": output_hashes,
        "alignment_sha256": alignments,
        "source_sha256": {str(path.resolve()): expected for path, expected in pins.items()},
        "config": {
            "schema_version": 1,
            "source_dataset": prior["path"],
            "paddle_run": spatial["config"]["paddle_run"],
            "alignment": spatial["config"]["alignment"],
            "selection_manifest": spatial["config"]["selection_manifest"],
            "selection_sha256": sha256_file(ctx.root / spatial["config"]["selection_manifest"]),
            "splits": [
                {"name": name, "records": len(rows), "sha256": output_hashes[f"{name}.jsonl"]}
                for name, rows in (("train", train), ("validation", validation))
            ],
        },
        "source_records": [
            {
                "documentId": row["documentId"],
                "split": split,
                "sourceInputSha256": digest(row["joinedRawText"].encode()),
                "positionedInputSha256": digest(row["positionedText"].encode()),
                "targetSha256": digest(row["target"]),
                "canonicalTargetSha256": canonical_json_sha256(row["target"]),
                "alignmentSha256": alignments[f"{row['documentId']}.json"],
            }
            for split, rows in (("train", train), ("validation", validation))
            for row in rows
        ],
        "provenance": {
            "cohort": ctx.spec.model_dump(),
            "priorAuthority": prior,
            "selectedSources": selection,
            "sourcePolicyReview": str(
                (ctx.audit / "source-policy-review.json").relative_to(ctx.root)
            ),
            "manualReview": str((ctx.audit / "manual-review.json").relative_to(ctx.root)),
            "validation": str((ctx.audit / "validation.json").relative_to(ctx.root)),
            "oldCaseMetadataUnchanged": True,
            "trainingDatasetUnchanged": True,
        },
    }


@dataclass(frozen=True)
class Admission:
    context: Context
    files: dict[Path, Path | bytes]
    config: dict
    config_bytes: bytes
    pins: dict[Path, str]
    report: dict


def prepare(ctx):
    started = perf_counter()
    require(not (ctx.audit / "admission.json").exists(), "Admission receipt already exists")
    selection = selected_sources(ctx)
    before = read(ctx.audit / "before/pins.json")
    pins = {ctx.registry / name: expected for name, expected in before["registry"].items()}
    verify_pins(pins)
    pins.update(protected_dataset_pins(ctx, before))
    original = read(ctx.registry / "manifest.json")
    old_ids, new_ids = set(original["cases"]), set(selection)
    require(
        original["sourceCount"] == len(old_ids) == ctx.spec.registry_sources_before,
        "Baseline registry count differs",
    )
    require(
        set(before["cases"]) == old_ids and not old_ids & new_ids, "Existing/new case scope differs"
    )
    require(
        {p.name for p in (ctx.registry / "cases").iterdir() if p.is_dir()} == old_ids,
        "Registry case directory inventory differs",
    )
    require(
        {"manifest.json", *original["sharedFiles"]} <= before["registry"].keys(),
        "Baseline registry/shared pin inventory is incomplete",
    )
    require(
        original["dependencies"]["sourceDataset"] == before["sourceAuthority"],
        "Baseline source-authority metadata differs",
    )
    config = yaml.safe_load(ctx.config_path.read_text())
    for key in (
        "dataset",
        "source_contracts",
        "historical_catalog",
        "ownership",
        "auxiliary",
        "task_constraints",
        "output",
    ):
        relative(config[key])
    require(
        set(config["source_ids"]) == new_ids
        and len(config["source_ids"]) == ctx.sources
        and set(config["capabilities"]) == new_ids,
        "Configured source/capability inventory differs",
    )
    require(
        config["variants_per_source"] == ctx.spec.variants_per_source
        and "template_sampling" not in config,
        "Configured descendant scope differs",
    )
    staged = ctx.root / relative(config["dataset"])
    prior = ctx.root / before["sourceAuthority"]["path"]
    prior_pins = {
        prior / name: expected for name, expected in before["sourceAuthority"]["files"].items()
    }
    verify_pins(prior_pins)
    pins.update(prior_pins)
    train, validation = records(staged / "train.jsonl"), records(staged / "validation.jsonl")
    rows, held = unique_rows(train, "documentId"), unique_rows(validation, "documentId")
    old_rows = unique_rows(records(prior / "train.jsonl"), "documentId")
    require(
        set(rows) == old_rows.keys() | new_ids and not rows.keys() & held.keys(),
        "Promoted authority identity/split differs",
    )
    require(
        all(rows[sid] == old_rows[sid] for sid in old_rows.keys() - new_ids),
        "Unselected source-authority row changed",
    )
    require(
        (staged / "validation.jsonl").read_bytes() == (prior / "validation.jsonl").read_bytes(),
        "Validation authority changed",
    )
    require(
        len(rows) == len(old_rows) + len(new_ids - old_rows.keys()),
        "Source-authority growth differs",
    )
    source_policy = read(ctx.audit / "source-policy-review.json")
    pins.update(source_policy_gate(ctx, source_policy, config))
    output = ctx.root / relative(config["output"])
    positions = output / relative(config["positions"]["output_subdirectory"])
    plain = unique_rows(records(output / "dataset.jsonl"), "documentId")
    positioned = unique_rows(records(positions / "dataset.jsonl"), "documentId")
    pins.update(
        manual_review_gate(
            ctx, read(ctx.audit / "manual-review.json"), rows, plain, positioned, output, positions
        )
    )
    require(
        set(r["sourceDocumentId"] for r in positioned.values()) == new_ids,
        "Published source identities differ",
    )
    campaign = Campaign(ctx.root, config)
    campaign.validate_source_readiness()
    checked = validation_gate(
        ctx, read(ctx.audit / "validation.json"), campaign.calls.task, output, positions
    )
    require(set(checked) == set(positioned), "Domain validation sample identities differ")
    regression = read(ctx.audit / "regression-before-admission.json")
    require(
        regression["status"] == "PASS"
        and regression["sources"] == len(old_ids)
        and regression["plans"] == len(old_ids) * ctx.spec.variants_per_source
        and regression["unchangedExistingCaseFiles"] is True,
        "Whole-registry regression gate is incomplete",
    )
    selection_path = ctx.root / "data/corpora/paddle-bl-swb-filtered-max5/selection.json"
    validate_exclusion([rows[sid] for sid in new_ids], validation, read(selection_path))
    pins[selection_path] = sha256_file(selection_path)
    for path in (
        ctx.audit / "cohort.json",
        ctx.config_path,
        ctx.audit / "before/pins.json",
        ctx.audit / "before/manifest.json",
        ctx.root / ctx.spec.selection.path,
    ):
        pins[path] = sha256_file(path)
    for name in (
        "source-policy-review.json",
        "manual-review.json",
        "validation.json",
        "regression-before-admission.json",
        "preflight.json",
    ):
        pins[ctx.audit / name] = sha256_file(ctx.audit / name)
    require(
        read(ctx.audit / "before/manifest.json") == original, "Frozen baseline manifest differs"
    )
    files, updated = {}, deepcopy(original)
    for sid in sorted(old_ids):
        validate_case(rows[sid], ctx.registry / "cases" / sid, expected=before["cases"][sid])
        pins.update(
            {
                ctx.registry / "cases" / sid / name: value
                for name, value in before["cases"][sid].items()
            }
        )
    for name in SHARED:
        path = ctx.root / config[name.split(".")[0]]
        merged = merge_shared(
            yaml.safe_load((ctx.registry / name).read_text()),
            yaml.safe_load(path.read_text()),
            old_ids,
            new_ids,
        )
        files[Path(name)] = yaml_bytes(merged)
        pins[path] = sha256_file(path)
        updated["sharedFiles"][name] = pin_bytes(files[Path(name)])
    for sid in sorted(new_ids):
        folder = ctx.root / config["source_contracts"] / sid
        require(not (ctx.registry / "cases" / sid).exists(), f"New case destination exists: {sid}")
        asset_pins = validate_case(rows[sid], folder)
        for name in ASSETS:
            files[Path("cases") / sid / name] = folder / name
            pins[folder / name] = asset_pins[name]
        updated["cases"][sid] = {
            "status": "admitted_3_variants_content_and_geometry_reviewed",
            "capability": config["capabilities"][sid],
            "files": asset_pins,
            "sourceInputSha256": asset_pins["source.txt"],
            "targetSha256": digest(rows[sid]["target"]),
            "sourceDataset": str((ctx.registry / ctx.authority_relative).relative_to(ctx.root)),
            "admissionCampaign": config["output"],
            "cohort": ctx.spec.cohort_id,
            "group": selection[sid]["group"],
        }
        campaign.preflight(campaign.plans(sid))
    for sample, row in plain.items():
        candidate = read(output / "candidates" / f"{sample}.json")
        campaign.replay_candidate(candidate)
        require(
            candidate["documentId"] == sample
            and candidate["sourceDocumentId"] == row["sourceDocumentId"]
            and candidate["joinedRawText"] == row["joinedRawText"]
            and candidate["target"]["documentPatch"] == row["target"],
            f"Candidate/publication differs: {sample}",
        )
        require(
            "schemaVersion" not in row["target"] and "documentPatch" not in row["target"],
            f"Published target contains metadata: {sample}",
        )
    require(
        not (ctx.registry / ctx.authority_relative).exists() and not ctx.production_config.exists(),
        "Immutable authority/config destination exists",
    )
    authority_files = {name: staged / name for name in ("train.jsonl", "validation.jsonl")}
    authority_files["task-constraints.json"] = ctx.root / config["task_constraints"]
    for name, path in authority_files.items():
        files[ctx.authority_relative / name] = path
        pins[path] = sha256_file(path)
    for name, value in (
        ("target-schema.json", campaign.calls.task.target_schema()),
        ("prompt-schema.json", json.loads(campaign.calls.task.prompt_schema_json())),
    ):
        files[ctx.authority_relative / name] = json_artifact_bytes(value)
    prior_manifest = read(prior / "manifest.json")
    require(
        set(prior_manifest["alignment_sha256"])
        == {f"{sid}.json" for sid in old_rows.keys() | held.keys()},
        "Prior alignment pin inventory differs",
    )
    alignments = {}
    for sid in sorted(rows.keys() | held.keys()):
        path = staged / "alignments" / f"{sid}.json"
        expected = sha256_file(path)
        if sid not in new_ids:
            old_expected = prior_manifest["alignment_sha256"][path.name]
            require(
                expected == old_expected == sha256_file(prior / "alignments" / path.name),
                f"Unselected alignment changed: {sid}",
            )
        files[ctx.authority_relative / "alignments" / path.name] = path
        pins[path] = expected
        alignments[path.name] = expected
    spatial_path = ctx.audit / "spatial/final/manifest.json"
    pins[spatial_path] = sha256_file(spatial_path)
    for path, expected in read(spatial_path)["source_sha256"].items():
        resolved = Path(path)
        require(resolved.is_absolute(), "Spatial source pins must be resolved absolute paths")
        require(sha256_file(resolved) == expected, f"Spatial prerequisite changed: {path}")
        pins[resolved] = expected
    authority_payload = authority_manifest(
        ctx,
        campaign.calls.task,
        config,
        train,
        validation,
        {
            name: pin_bytes(files[ctx.authority_relative / name])
            for name in (*authority_files, "target-schema.json", "prompt-schema.json")
        },
        alignments,
        {
            path: value
            for path, value in pins.items()
            if path not in {ctx.registry / name for name in (*SHARED, "manifest.json")}
        },
        before["sourceAuthority"],
        list(selection.values()),
    )
    files[ctx.authority_relative / "manifest.json"] = json_artifact_bytes(authority_payload)
    promoted = deepcopy(config)
    promoted.update(
        dataset=str((ctx.registry / ctx.authority_relative).relative_to(ctx.root)),
        ownership=str((ctx.registry / "ownership.yaml").relative_to(ctx.root)),
        auxiliary=str((ctx.registry / "auxiliary.yaml").relative_to(ctx.root)),
        source_contracts=str((ctx.registry / "cases").relative_to(ctx.root)),
        historical_catalog=str((ctx.registry / "cases").relative_to(ctx.root)),
        output=f"artifacts/kie-synthesis-production/{ctx.spec.cohort_id}-registry-validation",
    )
    config_bytes = yaml_bytes(promoted)
    updated["dependencies"]["sourceDataset"] = {
        "path": promoted["dataset"],
        "files": {
            name: pin_bytes(files[ctx.authority_relative / name])
            for name in ("train.jsonl", "validation.jsonl", "manifest.json")
        },
    }
    updated["dependencies"].setdefault("reviewedTemplateCohorts", {})[ctx.spec.cohort_id] = {
        "path": ctx.spec.production_config,
        "sha256": sha256_bytes(config_bytes),
    }
    for sid, override in updated.get("sourceAuthorityOverrides", {}).items():
        require(
            sid in old_ids
            and override["sourceContractSha256"] == before["cases"][sid]["contract.json"],
            "Protected source override contract changed",
        )
        require(
            sha256_file(ctx.root / override["revisionReceipt"])
            == override["revisionReceiptSha256"],
            "Protected source revision receipt changed",
        )
        pins[ctx.root / override["revisionReceipt"]] = override["revisionReceiptSha256"]
        override.update(
            dataset=promoted["dataset"],
            datasetManifestSha256=updated["dependencies"]["sourceDataset"]["files"][
                "manifest.json"
            ],
            trainSha256=updated["dependencies"]["sourceDataset"]["files"]["train.jsonl"],
            auxiliarySha256=updated["sharedFiles"]["auxiliary.yaml"],
        )
    require(
        all(updated["cases"][sid] == original["cases"][sid] for sid in old_ids),
        "Old case metadata changed",
    )
    updated["sourceCount"] = len(updated["cases"])
    require(updated["sourceCount"] == len(old_ids) + ctx.sources, "Final registry growth differs")
    event = {
        "cohort": ctx.spec.model_dump(),
        "sourceCount": updated["sourceCount"],
        "samples": ctx.samples,
        "newWiderSources": len(new_ids - old_rows.keys()),
        "revisedUnregisteredSourceRows": len(new_ids & old_rows.keys()),
        "campaignConfigFileSha256": sha256_bytes(config_bytes),
        "sourceAuthority": updated["dependencies"]["sourceDataset"],
        "reviewReceipts": {
            name: {
                "path": str((ctx.audit / name).relative_to(ctx.root)),
                "sha256": sha256_file(ctx.audit / name),
            }
            for name in ("source-policy-review.json", "manual-review.json", "validation.json")
        },
        "oldCaseMetadataUnchanged": True,
        "trainingDatasetChanged": False,
    }
    updated.setdefault("admissions", []).append(event)
    files[Path("manifest.json")] = json_artifact_bytes(updated)
    verify_pins(pins)
    report = {
        "status": "PASS",
        "dryRun": True,
        "sourcesBefore": len(old_ids),
        "sourcesAfter": updated["sourceCount"],
        "newSources": ctx.sources,
        "samples": ctx.samples,
        "groups": ctx.spec.groups,
        "sourceAuthorityTrain": len(train),
        "sourceAuthorityValidation": len(validation),
        "oldCaseAssetPinsPreserved": True,
        "oldCaseMetadataUnchanged": True,
        "trainingDatasetUnchanged": True,
        "sourceAuthority": updated["dependencies"]["sourceDataset"],
        "productionConfig": ctx.spec.production_config,
        "manifestSha256": pin_bytes(files[Path("manifest.json")]),
        "seconds": perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "paidCalls": 0,
    }
    return Admission(ctx, files, promoted, config_bytes, pins, report)


def install(plan, *, validate_live: Callable[[Admission], None]):
    ctx = plan.context
    transaction = ctx.audit / "admission-transaction"
    names = {
        "train.jsonl",
        "validation.jsonl",
        "task-constraints.json",
        "target-schema.json",
        "prompt-schema.json",
        "manifest.json",
    }
    for path in plan.files:
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe admission payload")
        case = len(path.parts) == 3 and path.parts[0] == "cases" and path.parts[-1] in ASSETS
        parts = path.parts[len(ctx.authority_relative.parts) :]
        authority = path.parts[
            : len(ctx.authority_relative.parts)
        ] == ctx.authority_relative.parts and (
            (len(parts) == 1 and parts[0] in names)
            or (
                len(parts) == 2
                and parts[0] == "alignments"
                and re.fullmatch(r"doc_[0-9a-f]{64}\.json", parts[1]) is not None
            )
        )
        require(
            path in {Path(name) for name in (*SHARED, "manifest.json")} or case or authority,
            f"Out-of-scope payload: {path}",
        )
    require(
        not transaction.exists()
        and not ctx.production_config.exists()
        and not (ctx.registry / ctx.authority_relative).exists(),
        "Immutable admission destination exists",
    )
    new_ids = sorted({p.parts[1] for p in plan.files if p.parts[0] == "cases"})
    require(
        set(new_ids) == set(plan.config["source_ids"]) and len(new_ids) == ctx.sources,
        "Admission payload case inventory differs from the reviewed cohort",
    )
    for sid in new_ids:
        require(
            {p.name for p in plan.files if p.parts[:2] == ("cases", sid)} == ASSETS,
            f"Admission payload has incomplete case assets: {sid}",
        )
    require(
        {Path(name) for name in (*SHARED, "manifest.json")}
        | {ctx.authority_relative / name for name in names}
        <= plan.files.keys(),
        "Admission payload is missing required authority/shared assets",
    )
    require(
        all(not (ctx.registry / "cases" / sid).exists() for sid in new_ids),
        "Admission would overwrite a case",
    )
    backup, payload = transaction / "before", transaction / "payload"
    modes = {}
    for name in (*SHARED, "manifest.json"):
        modes[name] = stat.S_IMODE((ctx.registry / name).stat().st_mode)
        atomic_publish_bytes(backup / name, (ctx.registry / name).read_bytes())
    for path, value in plan.files.items():
        atomic_publish_bytes(
            payload / path, value.read_bytes() if isinstance(value, Path) else value
        )
    atomic_publish_bytes(payload / "production-config.yaml", plan.config_bytes)
    moved, authority_moved, config_moved = [], False, False
    try:
        verify_pins(plan.pins)
        for sid in new_ids:
            (payload / "cases" / sid).rename(ctx.registry / "cases" / sid)
            moved.append(sid)
        (ctx.registry / ctx.authority_relative).parent.mkdir(parents=True, exist_ok=True)
        (payload / ctx.authority_relative).rename(ctx.registry / ctx.authority_relative)
        authority_moved = True
        for name in SHARED:
            atomic_write_bytes(ctx.registry / name, (payload / name).read_bytes())
            (ctx.registry / name).chmod(modes[name])
        (payload / "production-config.yaml").rename(ctx.production_config)
        config_moved = True
        validate_live(plan)
        atomic_write_bytes(ctx.registry / "manifest.json", (payload / "manifest.json").read_bytes())
        (ctx.registry / "manifest.json").chmod(modes["manifest.json"])
    except BaseException:
        for name in (*SHARED, "manifest.json"):
            atomic_write_bytes(ctx.registry / name, (backup / name).read_bytes())
            (ctx.registry / name).chmod(modes[name])
        if config_moved:
            ctx.production_config.rename(payload / "production-config.yaml")
        if authority_moved:
            (ctx.registry / ctx.authority_relative).rename(payload / ctx.authority_relative)
        for sid in reversed(moved):
            (ctx.registry / "cases" / sid).rename(payload / "cases" / sid)
        raise
    return backup


def validate_installed(plan):
    ctx = plan.context
    changing = {ctx.registry / name for name in SHARED}
    fixed = {path: value for path, value in plan.pins.items() if path not in changing}
    verify_pins(fixed)
    for path, value in plan.files.items():
        if path != Path("manifest.json"):
            require(
                sha256_file(ctx.registry / path) == pin_bytes(value),
                f"Installed payload differs: {path}",
            )
    require(
        ctx.production_config.read_bytes() == plan.config_bytes,
        "Installed production config differs",
    )
    inspection = yaml.safe_load(ctx.config_path.read_text())
    replay = deepcopy(plan.config)
    replay["output"] = inspection["output"]
    campaign = Campaign(ctx.root, replay)
    campaign.validate_source_readiness()
    for sid in campaign.source_ids:
        campaign.preflight(campaign.plans(sid))
    for row in records(ctx.root / inspection["output"] / "dataset.jsonl"):
        campaign.replay_candidate(
            read(ctx.root / inspection["output"] / "candidates" / f"{row['documentId']}.json")
        )
    old_manifest = read(ctx.audit / "before/manifest.json")
    config = deepcopy(plan.config)
    config.update(
        source_ids=sorted(old_manifest["cases"]),
        capabilities={sid: entry["capability"] for sid, entry in old_manifest["cases"].items()},
        output=str((ctx.audit / "regression-admission-precommit").relative_to(ctx.root)),
    )
    old = Campaign(ctx.root, config)
    old.validate_source_readiness()
    for sid in old.source_ids:
        old.preflight(old.plans(sid))
        validate_case(
            old.rows[sid],
            ctx.registry / "cases" / sid,
            expected=old_manifest["cases"][sid]["files"],
        )
    proposed = read(ctx.registry / "manifest.json")
    require(proposed == old_manifest, "Baseline manifest changed before final commit")
    for name in SHARED:
        original = yaml.safe_load((ctx.audit / "before" / name).read_text())
        installed = yaml.safe_load((ctx.registry / name).read_text())
        require(
            all(installed["sources"][sid] == original["sources"][sid] for sid in old.source_ids),
            f"Old shared metadata changed: {name}",
        )
    verify_pins(fixed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    ctx = Context.load(args.project_root, args.audit_dir)
    plan = prepare(ctx)
    if not args.apply:
        print(json.dumps(plan.report, indent=2))
        return
    with (ctx.registry / ".catalog-admission.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_pins(plan.pins)
        backup = install(plan, validate_live=validate_installed)
    report = {
        **plan.report,
        "status": "ADMITTED",
        "dryRun": False,
        "backup": str(backup.relative_to(ctx.root)),
        "manifestSha256": sha256_file(ctx.registry / "manifest.json"),
    }
    atomic_publish_bytes(ctx.audit / "admission.json", json_artifact_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
