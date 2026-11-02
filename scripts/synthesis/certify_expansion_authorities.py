"""Publish the reviewed R28 source authorities using bounded regression evidence.

Historical controls are immutable. Exact replays, current source-matching native
samples, and fresh one/two/three-group replacement controls are kept distinct.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import yaml

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_acceptance import (
    SOURCE_FILES,
    authority_pins,
    require_certified,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"
REG = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
BASE = ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json"
STAGED = REG / "expansion-20261101-v21/complete-source-authority-config.json"
DEST = REG / "expansion-20261101-certified"


@cache
def read(path):
    text = Path(path).read_text()
    return (
        json.loads(text)
        if Path(path).suffix == ".json"
        else yaml.load(text, Loader=yaml.CSafeLoader)
    )


@cache
def file_hash(path):
    return sha256_file(path)


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}


def checked(value):
    path = ROOT / value["path"]
    if file_hash(path) != value["sha256"]:
        raise ValueError("evidence changed: " + str(path))
    return path


def source_authority(config, sid):
    paths = {k: config[k] for k in ("ownership", "auxiliary")}
    paths.update(
        {
            k: config["independent"][k]["path"]
            for k in ("numeric_certificates", "geography_certificates", "wording_certificates")
        }
    )
    paths["groups"] = config["declared_packages"]["contracts"]["path"]
    paths["aliasCertificates"] = config["aliases"]["presentations"]["certificates"]["path"]
    return {
        **{k: read(ROOT / path)["sources"].get(sid) for k, path in paths.items()},
        "sourceFiles": {
            name: file_hash(ROOT / config["source_contracts"] / sid / name)
            for name in sorted(SOURCE_FILES)
        },
    }


def investigate():
    old_config, current = read(BASE), read(STAGED)
    old = read(checked(old_config["declared_packages"]["acceptance"]))
    if set(old["sources"]) != set(current["source_ids"]):
        raise ValueError("source registry membership changed")
    if (
        old_config["declared_packages"]["source_group_counts"]
        != current["declared_packages"]["source_group_counts"]
    ):
        raise ValueError("source package-mode policy changed")
    withdrawn = set(read(OUT / "withdrawn-staged-samples.json")["samples"])
    fresh, mismatches = defaultdict(list), defaultdict(set)
    current_signatures = {s: source_authority(current, s) for s in current["source_ids"]}
    differences = {
        s: [k for k, v in source_authority(old_config, s).items() if current_signatures[s][k] != v]
        for s in current["source_ids"]
    }
    cert_paths = sorted((OUT / "components").glob("*/certification.json")) + sorted(
        (OUT / "authority-controls").glob("*/certification.json")
    )
    for path in cert_paths:
        receipt = read(path)
        if receipt["status"] != "CERTIFIED_COMPONENT_NOT_YET_APPENDED_TO_R28":
            raise ValueError("uncertified evidence: " + str(path))
        config = read(checked(receipt["config"]))
        certificate = read(checked(config["declared_packages"]["acceptance"]))
        checked(receipt["mainAudit"])
        for sid, row in certificate["sources"].items():
            signature = source_authority(config, sid)
            if signature != current_signatures[sid]:
                mismatches[sid].update(
                    k for k in signature if signature[k] != current_signatures[sid][k]
                )
                continue
            for sample in row["samples"]:
                if sample["sampleId"] in withdrawn:
                    continue
                for k in ("candidate", "positions", "review", "adjudication", "scopedReview"):
                    checked(sample[k])
                fresh[sid].append(sample)
    identical, stale, replay_pins = set(), defaultdict(list), []
    for path in sorted((OUT / "authority-carryover/nonthermal").glob("*/result.json")):
        result = read(path)
        if (
            file_hash(Path(result["config"])) != result["configSha256"]
            or result["providerCalls"] != 0
        ):
            raise ValueError("read-only regression evidence changed")
        replay_pins.append(pin(path))
        for row in result["records"]:
            candidate = read(checked(row["candidate"]))
            if row["status"] == "IDENTICAL":
                identical.add(row["sampleId"])
            else:
                stale[candidate["sourceDocumentId"]].append(row)
    if len(replay_pins) != 36 or len(identical) != 1197 or sum(map(len, stale.values())) != 24:
        raise ValueError("unexpected historical control inventory")
    thermal = read(OUT / "authority-carryover/nonthermal/thermal-requires-fresh-evidence.json")
    thermal_sources = {row["sourceDocumentId"] for row in thermal}
    if len(thermal) != 114:
        raise ValueError("unexpected old thermal control inventory")
    group_coverage = {sid: sorted({s["groupCount"] for s in rows}) for sid, rows in fresh.items()}
    missing_fresh = sorted(s for s in current["source_ids"] if s not in fresh)
    missing_replacements = {
        s: sorted(set(old["sources"][s]["groupCounts"]) - set(group_coverage.get(s, [])))
        for s in stale
    }
    missing_replacements = {s: v for s, v in missing_replacements.items() if v}
    source_deltas_without_fresh = {s: v for s, v in differences.items() if v and s not in fresh}
    return (
        old_config,
        current,
        old,
        fresh,
        {
            "previousAcceptance": old_config["declared_packages"]["acceptance"],
            "stagedConfig": pin(STAGED),
            "sourceDifferences": {s: v for s, v in differences.items() if v},
            "matchingFreshSources": len(fresh),
            "matchingFreshSamples": sum(map(len, fresh.values())),
            "freshGroupCoverage": group_coverage,
            "missingFreshSources": missing_fresh,
            "staleHistoricalSamples": dict(stale),
            "missingReplacementModes": missing_replacements,
            "sourceDeltasWithoutFreshEvidence": source_deltas_without_fresh,
            "exactHistoricalReplays": len(identical),
            "replays": replay_pins,
            "oldThermalControlsRequiringNewInput": len(thermal),
            "thermalSources": sorted(thermal_sources),
            "newThermalSourcesWithFreshEvidence": len(thermal_sources & set(fresh)),
            "threeRowThermalSources": sorted(
                s for s in thermal_sources if 3 in group_coverage.get(s, [])
            ),
            "historicalEvidenceNotReusedAsCurrentSourceProof": {
                s: sorted(v) for s, v in mismatches.items()
            },
        },
    )


def run(publish=False):
    previous, current, old, fresh, report = investigate()
    if not publish:
        print(json.dumps(report, indent=2))
        return
    # The payable-elsewhere template is intentionally excluded from this
    # campaign; unchanged previous acceptance remains authoritative for it.
    if (
        len(report["missingFreshSources"]) != 1
        or report["sourceDeltasWithoutFreshEvidence"]
        or report["missingReplacementModes"]
    ):
        raise ValueError("current source or mode evidence incomplete")
    if (
        report["newThermalSourcesWithFreshEvidence"] != len(report["thermalSources"])
        or len(report["threeRowThermalSources"]) < 2
    ):
        raise ValueError("thermal-domain regression evidence incomplete")
    validation = read(OUT / "complete-authority-validation.json")
    if len(validation["sources"]) != 456 or any(
        r["status"] != "PASS" for r in validation["sources"]
    ):
        raise ValueError("full registry readiness did not pass")
    approval = read(OUT / "authority-carryover/main-approval.json")
    if approval["status"] != "APPROVED" or not approval["review"]:
        raise ValueError("explicit main review is required")
    report.update(
        status="PASS_BOUNDED_AUTHORITY_PROMOTION",
        mainReview=pin(OUT / "authority-carryover/main-approval.json"),
        sourceReadiness=pin(OUT / "complete-authority-validation.json"),
        datasetAudit=pin(OUT / "final/audit.json"),
        scope=(
            "1,197 exact historical replays; stale nonthermal receipts replaced by fresh "
            "current-source one/two/three-row controls. All 38 thermal sources have fresh "
            "native evidence; shared new thermal-input path has additional one/two/three-row "
            "controls. Unchanged historical template-mode evidence is retained explicitly, "
            "not claimed as a fresh replay of absent native thermal inputs."
        ),
    )
    report_path = DEST / "promotion-review.json"
    atomic_publish_json(report_path, report)
    certificate = copy.deepcopy(old)
    certificate["expansion20261101"] = pin(report_path)
    certificate["authorities"] = authority_pins(ROOT, current)
    implementation = {p["path"] for p in old["implementation"]} | {
        "src/document_ocr/synthesis/independent_thermal.py"
    }
    certificate["implementation"] = [pin(ROOT / path) for path in sorted(implementation)]
    contracts = read(ROOT / current["declared_packages"]["contracts"]["path"])["sources"]
    for sid, row in certificate["sources"].items():
        # Keep previous mode proofs separately; current evidence is not
        # misrepresented as fresh generation of every historical sample.
        row["previousModeEvidence"] = copy.deepcopy(row["samples"])
        if sid in fresh:
            row["samples"] = fresh[sid]
        row["sourceFiles"] = {
            name: pin(ROOT / current["source_contracts"] / sid / name)
            for name in sorted(SOURCE_FILES)
        }
        row["contractSha256"] = digest(contracts[sid])
    acceptance = DEST / "acceptance.json"
    atomic_publish_json(acceptance, certificate)
    config = copy.deepcopy(current)
    config["declared_packages"]["acceptance"] = pin(acceptance)
    # Verify the unchanged publication gate for every enabled source/mode.
    campaign = SimpleNamespace(root=ROOT, config=config, declared_package_contracts=contracts)
    require_certified(campaign, dict.fromkeys(config["source_ids"], 3))
    atomic_publish_json(DEST / "previous-production-config.json", previous)
    manifest_path = REG / "manifest.json"
    manifest = read(manifest_path)
    atomic_publish_json(DEST / "previous-registry-manifest.json", manifest)
    atomic_write_json(BASE, config)
    manifest = copy.deepcopy(manifest)
    manifest["currentProductionConfig"] = {
        "path": str(BASE.relative_to(ROOT)),
        "sha256": sha256_file(BASE),
    }
    manifest["thermalAndExpansionClosure20261101"] = {
        "acceptance": pin(acceptance),
        "review": pin(report_path),
        "sources": 456,
    }
    atomic_write_json(manifest_path, manifest)
    atomic_publish_json(
        OUT / "authority-carryover/publication.json",
        {
            "status": "PROMOTED_AND_ADMISSION_VALIDATED",
            "sources": 456,
            "productionConfig": manifest["currentProductionConfig"],
            "acceptance": pin(acceptance),
            "review": pin(report_path),
            "enabledGroupModesUnchanged": True,
            "datasetMutation": False,
        },
    )
    print(json.dumps({"status": "PROMOTED_AND_ADMISSION_VALIDATED", "sources": 456}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    run(parser.parse_args().publish)
