"""Build all-registry multi-row acceptance from reviewed, current replay evidence.

No automatic approval of native findings; each selected sample needs a hash-bound
main-review decision. This tool writes evidence, not training records or defaults.
"""

import argparse
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_acceptance import SOURCE_FILES, authority_pins
from document_ocr.synthesis.declared_packages import DeclaredPackages


def pin(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def build(root, config, contracts, decisions, decision_path):
    source_ids = set(config["source_ids"])
    if set(decisions["sources"]) != source_ids or set(contracts["sources"]) != source_ids:
        raise ValueError("every registered source needs an explicit disposition and contract")
    records = {}
    for sid in sorted(source_ids):
        decision = decisions["sources"][sid]
        if decision["status"] != "PASS" or not decision["reason"].strip():
            raise ValueError("source still lacks final acceptance: " + sid)
        controls = []
        counts = set()
        samples = set()
        for row in decision["controls"]:
            cp, pp, rp = (root / row[name]["path"] for name in ("candidate", "positions", "review"))
            for name, path in zip(("candidate", "positions", "review"), (cp, pp, rp), strict=True):
                if sha256_file(path) != row[name]["sha256"]:
                    raise ValueError("reviewed evidence changed: " + name)
            candidate, positions, review = (json.loads(p.read_text()) for p in (cp, pp, rp))
            sample = candidate["documentId"]
            if candidate["sourceDocumentId"] != sid or sample in samples:
                raise ValueError("control source/sample identity is not unique")
            samples.add(sample)
            if (
                positions["candidateSha256"] != sha256_file(cp)
                or positions["target"] != candidate["target"]
                or review["candidateHashes"].get(sample) != digest(candidate)
            ):
                raise ValueError("geometry/native review does not belong to the accepted candidate")
            findings = [f for f in review["review"]["findings"] if f["sample_id"] == sample]
            if {digest(f) for f in findings} != set(row["findingDecisions"]):
                raise ValueError("every native finding needs its exact main-review decision")
            if any(not value.strip() for value in row["findingDecisions"].values()):
                raise ValueError("empty finding decision")
            provenance = candidate["scenario"]["provenance"]
            ledger = provenance.get("declaredPackages")
            count = len(DeclaredPackages.from_receipt(ledger).groups) if ledger else 1
            counts.add(count)
            if ledger and provenance["declaredPackageContract"] != contracts["sources"][sid]:
                raise ValueError("candidate used a different source presentation contract")
            controls.append(
                {
                    "sampleId": sample,
                    "groupCount": count,
                    **{name: row[name] for name in ("candidate", "positions", "review")},
                    "adjudication": pin(decision_path),
                }
            )
        required = set(
            config["declared_packages"]
            .get("source_group_counts", {})
            .get(sid, config["declared_packages"]["group_counts"])
        )
        if not required <= counts or len(controls) < 3:
            raise ValueError(
                "source requires three fresh controls and every configured group count"
            )
        records[sid] = {
            "status": "PASS",
            "groupCounts": sorted(counts),
            "contractSha256": digest(contracts["sources"][sid]),
            "samples": controls,
            "sourceFiles": {
                name: pin(root / config["source_contracts"] / sid / name)
                for name in sorted(SOURCE_FILES)
            },
        }
        # Keep repository-relative pins portable and aligned with runtime admission.
        for value in records[sid]["sourceFiles"].values():
            value["path"] = str(Path(value["path"]).relative_to(root))
    implementation = [
        pin(path) for path in sorted((root / "src/document_ocr/synthesis").rglob("*.py"))
    ]
    for value in implementation:
        value["path"] = str(Path(value["path"]).relative_to(root))
    return {
        "version": 1,
        "status": "certified_native_synthesis_and_geometry",
        "authorities": authority_pins(root, config),
        "implementation": implementation,
        "sources": records,
        "trainingDatasetMutation": False,
        "reviewScope": decisions["reviewScope"],
        "claims": (
            "Reviewed synthesis and replay acceptance, not first-draw accuracy "
            "or a regulatory filing certificate."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    authority = config["declared_packages"]["contracts"]
    if sha256_file(Path(authority["path"])) != authority["sha256"]:
        raise ValueError("declared source contracts changed")
    report = build(
        Path.cwd(),
        config,
        json.loads(Path(authority["path"]).read_text()),
        json.loads(args.decisions.read_text()),
        args.decisions,
    )
    atomic_publish_json(args.output, report)
    print(json.dumps({"status": report["status"], "sources": len(report["sources"])}))


if __name__ == "__main__":
    main()
