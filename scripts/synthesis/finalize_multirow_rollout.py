"""Stage, replay, then activate the fully reviewed registry; never publish training data."""

import argparse
import copy
import json
import resource
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from certify_multirow_rollout import build, pin
from multirow_context_rebase import rebase
from prepare_multirow_current_views import subset

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    atomic_write_json,
)
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import Campaign, create_campaign, wording_request_hash
from document_ocr.synthesis.declared_package_acceptance import require_certified

REGISTRY = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed")
ROLLOUT = REGISTRY / "multirow-rollout-20261031"
RUNTIME = REGISTRY / "declared-packages-v1"
CONFIG = Path("configs/synthesis/mpci_bl_curated_v7_multirow.json")
DECISIONS = ROLLOUT / "main-adjudication-02.json"
REPLAY = ROLLOUT / "final-authority-replay-v4"
CONTEXT_ARCHIVE = ROLLOUT / "wording-context-before-v2"


def read(path):
    return json.loads(path.read_text())


def reconcile_catalogue_pin(ledger, before, after):
    count = 0
    if "certificateFileSha256" in ledger:
        if ledger["certificateFileSha256"] != before:
            raise ValueError("candidate numeric catalogue hash differs from its authority")
        ledger["certificateFileSha256"] = after
        count += 1
    children = list(ledger.get("scopeLedgers", {}).values())
    children.extend(ledger["facts"].get("declaredGroupLedgers", []))
    return count + sum(reconcile_catalogue_pin(child, before, after) for child in children)


def stage():
    decisions = read(DECISIONS)
    if decisions["summary"]["reviewedPassingSources"] != 456 or decisions["summary"]["heldSources"]:
        raise ValueError("all source decisions must pass before staging production")
    config = read(ROLLOUT / "resolved-base-config-v2.json")
    contracts = {}
    for path in sorted(ROLLOUT.glob("batch-*/source-contracts.json")):
        contracts.update(read(path)["sources"])
    contracts = {sid: contracts[sid] for sid in config["source_ids"]}
    atomic_publish_json(RUNTIME / "source-contracts.json", {"version": 1, "sources": contracts})
    for name in ("numeric_certificates", "geography_certificates"):
        old = config["independent"][name]
        if sha256_file(Path(old["path"])) != old["sha256"]:
            raise ValueError("staged authority changed")
        target = RUNTIME / (name + ".json")
        atomic_publish_bytes(target, Path(old["path"]).read_bytes())
        config["independent"][name] = pin(target)
    config["declared_packages"] = {
        "contracts": pin(RUNTIME / "source-contracts.json"),
        "group_counts": [1, 2, 3],
        # The complete certified cohort uses this explicit planning envelope.
        "utilization": [0.35, 0.75],
        "source_group_counts": {
            sid: [1]
            for sid in config["source_ids"]
            if sid[4:12] in {"bef02d48", "458d2fa0", "0dbc75a0", "2763b891"}
        },
    }
    config["variants_per_source"] = 3
    config["output"] = "artifacts/kie-synthesis-production/r28-multirow"
    config["seed"] = 2026103109
    atomic_publish_json(ROLLOUT / "staged-production-config.json", config)
    # Guard against overwriting unrelated edits during activation.
    snapshots = {}
    for name in ("ownership.yaml", "auxiliary.yaml", "manifest.json"):
        path = REGISTRY / name
        backup = ROLLOUT / "pre-promotion" / name
        atomic_publish_bytes(backup, path.read_bytes())
        snapshots[name] = {"original": pin(path), "backup": pin(backup)}
    atomic_publish_json(ROLLOUT / "pre-promotion/pins.json", snapshots)
    print(json.dumps({"stagedSources": len(contracts), "defaultChanged": False}), flush=True)


def replay():
    config = read(ROLLOUT / "staged-production-config.json")
    decisions = read(DECISIONS)
    selected = defaultdict(list)
    for sid, row in decisions["sources"].items():
        for control in row["controls"]:
            selected[control["batch"]].append((sid, control))
    summaries = []
    started = time.perf_counter()
    for batch_name, controls in sorted(selected.items()):
        batch = ROLLOUT / batch_name
        candidates = [
            batch / "production-replay/config.json",
            batch / "reviewed-authorities/config.json",
            batch / "prepared-config.json",
        ]
        old = read(next(p for p in candidates if p.exists()))
        original_output = Path(old["output"])
        sources = sorted({sid for sid, _ in controls})
        numeric_before_pin = old["independent"]["numeric_certificates"]
        if sha256_file(Path(numeric_before_pin["path"])) != numeric_before_pin["sha256"]:
            raise ValueError("original numerical authority changed")
        numeric_before = read(Path(numeric_before_pin["path"]))["sources"]
        numeric_after_pin = config["independent"]["numeric_certificates"]
        numeric_after = read(Path(numeric_after_pin["path"]))["sources"]
        if any(numeric_before[sid] != numeric_after[sid] for sid in sources):
            raise ValueError(
                "selected source numeric certificate changed; fresh synthesis required"
            )
        for name in ("ownership", "auxiliary"):
            old[name] = config[name]
        for name in ("numeric_certificates", "geography_certificates", "wording_certificates"):
            old["independent"][name] = config["independent"][name]
        old["declared_packages"]["contracts"] = config["declared_packages"]["contracts"]
        directory = REPLAY / batch_name
        old["output"] = str(directory / "campaign")
        current = subset(old, sources, directory)
        begin = time.perf_counter()
        campaign = create_campaign(Path.cwd(), current)
        spent = campaign.calls.spent
        rows = []
        for sid in sources:
            contacts = original_output / "contacts" / f"{sid}.json"
            if contacts.exists():
                atomic_publish_bytes(
                    campaign.output / "contacts" / contacts.name, contacts.read_bytes()
                )
        archive_path = CONTEXT_ARCHIVE / f"{batch_name}.json"
        archive = (
            {row["sampleId"]: row for row in read(archive_path)["rows"]}
            if archive_path.exists()
            else {}
        )
        for sid, control in controls:
            expected = read(Path(control["candidate"]["path"]))
            sample = expected["documentId"]
            original_wording = original_output / "wording" / f"{sample}.json"
            contacts = original_output / "contacts/samples" / f"{sample}.json"
            if contacts.exists():
                atomic_publish_bytes(
                    campaign.output / "contacts/samples" / contacts.name, contacts.read_bytes()
                )
            receipt = read(original_wording)
            if digest(receipt) != expected["wordingReceiptSha256"]:
                raise ValueError(
                    "selected wording receipt changed since reviewed rendering: " + sample
                )
            blueprint, scenario, sample, target = campaign.plan(sid, expected["variant"])
            request = campaign.wording_request(blueprint, scenario, sample, target)
            if receipt["requestSha256"] != wording_request_hash(request):
                if sample not in archive:
                    raise ValueError("request changed without an exact original archive: " + sample)
                receipt = rebase(receipt, archive[sample], request, archive_pin=pin(archive_path))
                atomic_publish_json(
                    directory / "context-rebases" / f"{sample}.json",
                    {
                        "originalReceipt": pin(original_wording),
                        "audit": receipt["validationContractRebase"],
                        "oldRequestSha256": archive[sample]["requestSha256"],
                        "newRequestSha256": receipt["requestSha256"],
                    },
                )
                atomic_publish_json(campaign.output / "wording" / f"{sample}.json", receipt)
            else:
                atomic_publish_bytes(
                    campaign.output / "wording" / f"{sample}.json", original_wording.read_bytes()
                )
            try:
                actual = campaign.render(sid, expected["variant"], persist=False)
            except Exception as error:
                raise ValueError(f"final replay {sid}:{expected['variant']}: {error}") from error
            normalized = copy.deepcopy(expected)
            normalized["wordingReceiptSha256"] = digest(receipt)
            # Only the catalogue-level provenance hash changes: this source's
            # certificate was checked equal above. Text, targets, numeric facts,
            # owner receipts and all other metadata must remain exactly equal.
            provenance_pins = 0
            for record in (normalized["physical"], normalized["scenario"]["provenance"]):
                private = record.get("freshPrivatePacking")
                if private is not None:
                    provenance_pins += reconcile_catalogue_pin(
                        private, numeric_before_pin["sha256"], numeric_after_pin["sha256"]
                    )
            if "compositeLocalityProjection" in normalized:
                # This digest covers the entire *pre-projection* candidate,
                # including its authority metadata. Prove the old and new
                # digests explicitly instead of ignoring a changed hash.
                before_projection = copy.deepcopy(
                    Campaign.render(campaign, sid, expected["variant"], persist=False)
                )
                projection = actual["compositeLocalityProjection"]
                if digest(before_projection) != projection["beforeCandidateSha256"]:
                    raise ValueError("current locality projection lineage is not reproducible")
                before_projection["wordingReceiptSha256"] = expected["wordingReceiptSha256"]
                for record in (
                    before_projection["physical"],
                    before_projection["scenario"]["provenance"],
                ):
                    reconcile_catalogue_pin(
                        record["freshPrivatePacking"],
                        numeric_after_pin["sha256"],
                        numeric_before_pin["sha256"],
                    )
                if (
                    digest(before_projection)
                    != normalized["compositeLocalityProjection"]["beforeCandidateSha256"]
                ):
                    raise ValueError(
                        "original locality projection differs beyond authority metadata"
                    )
                normalized["compositeLocalityProjection"]["beforeCandidateSha256"] = projection[
                    "beforeCandidateSha256"
                ]
            if digest(actual) != digest(normalized):
                differing = [key for key in normalized if actual.get(key) != normalized[key]]
                raise ValueError(
                    f"final authority replay differs: {sid}:{expected['variant']}: {differing}"
                )
            rows.append(
                {
                    "sourceId": sid,
                    "sampleId": actual["documentId"],
                    "variant": expected["variant"],
                    "candidateSha256": control["candidate"]["sha256"],
                    "fullCandidateDigest": digest(actual),
                    "catalogueProvenancePinsReconciled": provenance_pins,
                    "wordingValidationContextRebased": "validationContractRebase" in receipt,
                    "status": "PASS",
                }
            )
        if campaign.calls.spent != spent:
            raise ValueError("deterministic replay unexpectedly made a provider call")
        report = {
            "batch": batch_name,
            "sources": len(sources),
            "controls": len(rows),
            "seconds": time.perf_counter() - begin,
            "nativeCalls": 0,
            "rows": rows,
        }
        atomic_publish_json(directory / "result.json", report)
        summaries.append(
            {
                "batch": batch_name,
                "sources": len(sources),
                "controls": len(rows),
                "result": pin(directory / "result.json"),
            }
        )
        print(
            json.dumps({k: report[k] for k in ("batch", "sources", "controls", "seconds")}),
            flush=True,
        )
    atomic_publish_json(
        REPLAY / "summary.json",
        {
            "status": "PASS",
            "decisions": pin(DECISIONS),
            "config": pin(ROLLOUT / "staged-production-config.json"),
            "sources": len(decisions["sources"]),
            "controls": sum(row["controls"] for row in summaries),
            "batches": summaries,
            "seconds": time.perf_counter() - started,
            "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "fullCandidateIdentityExceptReviewedAuthorityAndValidationProvenance": True,
            "originalWordingContextArchive": pin(CONTEXT_ARCHIVE / "summary.json"),
            "rebasedValidationContexts": sum(
                row["wordingValidationContextRebased"]
                for batch in summaries
                for row in read(Path(batch["result"]["path"]))["rows"]
            ),
            "rawTextAndTargetsIdentical": True,
            "nativeCalls": 0,
            "trainingDatasetMutation": False,
        },
    )


def activate():
    replay_path = REPLAY / "summary.json"
    replay_result = read(replay_path)
    if replay_result["status"] != "PASS" or (
        replay_result["sources"],
        replay_result["controls"],
    ) != (456, 1368):
        raise ValueError("full final-authority replay is required")
    if replay_result["decisions"] != pin(DECISIONS) or replay_result["config"] != pin(
        ROLLOUT / "staged-production-config.json"
    ):
        raise ValueError("final replay authority changed")
    for row in replay_result["batches"]:
        if sha256_file(Path(row["result"]["path"])) != row["result"]["sha256"]:
            raise ValueError("batch replay result changed")
    fresh_path = ROLLOUT / "fresh-context-controls.json"
    fresh = read(fresh_path)
    if (fresh["status"], fresh["controls"], fresh["unresolvedFindings"]) != ("PASS", 9, 0):
        raise ValueError("fresh prompt controls must pass final text and geometry review")
    if fresh["projectionImplementation"] != pin(
        Path("src/document_ocr/synthesis/independent_presentation.py")
    ):
        raise ValueError("fresh controls tested a different prompt projection")
    for row in fresh["rows"]:
        for name in ("candidate", "positions", "review"):
            if row[name] != pin(Path(row[name]["path"])):
                raise ValueError("fresh prompt control evidence changed")
    backups = read(ROLLOUT / "pre-promotion/pins.json")
    for record in backups.values():
        if sha256_file(Path(record["original"]["path"])) != record["original"]["sha256"]:
            raise ValueError("live authority changed since staging; do not overwrite")
    config = read(ROLLOUT / "staged-production-config.json")
    contracts = read(RUNTIME / "source-contracts.json")
    # Validate all controls before touching live authorities. Moving identical
    # reviewed bytes changes their path, not the content they certify.
    acceptance = build(Path.cwd(), config, contracts, read(DECISIONS), DECISIONS)
    for key, name in (("ownership", "ownership.yaml"), ("auxiliary", "auxiliary.yaml")):
        atomic_write_bytes(REGISTRY / name, Path(config[key]).read_bytes())
        config[key] = str(REGISTRY / name)
        acceptance["authorities"][key]["path"] = config[key]
    acceptance["finalAuthorityReplay"] = pin(replay_path)
    acceptance["freshPromptControls"] = pin(fresh_path)
    acceptance_path = RUNTIME / "acceptance.json"
    atomic_publish_json(acceptance_path, acceptance)
    config["declared_packages"]["acceptance"] = pin(acceptance_path)
    proxy = SimpleNamespace(
        root=Path.cwd(), config=config, declared_package_contracts=contracts["sources"]
    )
    require_certified(proxy, {sid: 3 for sid in config["source_ids"]})
    atomic_publish_json(CONFIG, config)
    manifest = read(REGISTRY / "manifest.json")
    for name in ("ownership.yaml", "auxiliary.yaml"):
        manifest["sharedFiles"][name] = sha256_file(REGISTRY / name)
    manifest["multirowPackages20261031"] = {
        "status": "certified_native_synthesis_and_geometry",
        "sources": 456,
        "multirowSources": 452,
        "singleRowOnlySources": 4,
        "controls": 1368,
        "config": pin(CONFIG),
        "acceptance": pin(acceptance_path),
        "replay": pin(replay_path),
        "freshPromptControls": pin(fresh_path),
        "trainingDatasetMutation": False,
        "report": "docs/analysis/r28-multirow-production-20261031/REPORT.md",
    }
    manifest["currentProductionConfig"] = pin(CONFIG)
    atomic_write_json(REGISTRY / "manifest.json", manifest)
    atomic_publish_json(
        ROLLOUT / "publication.json",
        {
            "status": "PROMOTED",
            "config": pin(CONFIG),
            "manifest": pin(REGISTRY / "manifest.json"),
            "acceptance": pin(acceptance_path),
            "backups": backups,
            "trainingDatasetMutation": False,
            "positiveRuntimeAdmission": "PASS",
        },
    )
    print(json.dumps({"status": "PROMOTED", "sources": 456, "controls": 1368}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stage", "replay", "activate"])
    args = parser.parse_args()
    {"stage": stage, "replay": replay, "activate": activate}[args.stage]()


if __name__ == "__main__":
    main()
