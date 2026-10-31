"""Close the bounded vehicle-eligibility and private Marks authority repair.

Historical evidence remains immutable. Frozen refresh campaigns retain their
actual generation authorities; the current production configuration receives
the new private-owner authority. Both certificates pin the tested current code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_acceptance import authority_pins

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/analysis/r28-multirow-refresh1500-20261031"
REGISTRY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
REVISION = REGISTRY / "declared-packages-refresh-20261031"
FRESH = (
    ROOT
    / "artifacts/kie-synthesis-production/r28-multirow-refresh1500-20261031"
    / "upstream-regression-controls"
)


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def run():
    old_path = REGISTRY / "declared-packages-v1/acceptance.json"
    old = read(old_path)
    config_path = ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json"
    previous = read(config_path)
    current = read(REVISION / "staged-production-config.json")
    private_source = next(s for s in current["source_ids"] if s.startswith("doc_c08ce779"))
    old_numeric = read(ROOT / old["authorities"]["numeric_certificates"]["path"])
    new_numeric = read(ROOT / current["independent"]["numeric_certificates"]["path"])
    if {
        s for s in old_numeric["sources"] if old_numeric["sources"][s] != new_numeric["sources"][s]
    } != {private_source}:
        raise ValueError("numerical authority changed beyond the reviewed private owner")
    old_owners = yaml.safe_load((ROOT / old["authorities"]["ownership"]["path"]).read_text())
    new_owners = yaml.safe_load((ROOT / current["ownership"]).read_text())
    if set(old_owners) != set(new_owners) or {
        k: v for k, v in old_owners.items() if k != "sources"
    } != {k: v for k, v in new_owners.items() if k != "sources"}:
        raise ValueError("ownership root shape changed")
    owner_changes = {
        key
        for key in old_owners["sources"]
        if old_owners["sources"][key] != new_owners["sources"][key]
    }
    if owner_changes != {private_source}:
        raise ValueError(f"ownership changes are not source-local: {owner_changes}")
    replay_paths = sorted((REPORT / "final-authority-carryover").glob("*/result.json"))
    carried = []
    for path in replay_paths:
        result = read(path)
        if result["status"] != "PASS" or result["providerCalls"] != 0:
            raise ValueError("retained control replay did not pass")
        if sha256_file(Path(result["config"])) != result["configSha256"]:
            raise ValueError("replay config changed")
        carried.extend(row["sampleId"] for row in result["records"])
    if len(carried) != 1335 or len(set(carried)) != 1335:
        raise ValueError("retained controls are incomplete")
    fresh = {}
    for directory in (FRESH, FRESH / "redraw-2", FRESH / "redraw-3"):
        summary_path = sorted((directory / "stage-runs").glob("*/summary.json"))[-1]
        summary = read(summary_path)
        for stage in ("generate", "postal", "render", "review"):
            if summary["stages"][stage].get("errors"):
                raise ValueError("fresh control stage has errors")
        for row in summary["stages"]["geometry"]["rows"]:
            if row["status"] != "PASS":
                raise ValueError("fresh geometry held")
            candidate_path = ROOT / row["candidate"]
            candidate = read(candidate_path)
            sid = candidate["sourceDocumentId"]
            review_path = directory / "campaign/reviews" / (sid + ".json")
            review = read(review_path)
            if review["candidateHashes"][candidate["documentId"]] != digest(candidate):
                raise ValueError("fresh candidate differs from its native review")
            decisions = []
            for index, finding in enumerate(review["review"]["findings"]):
                if not finding["sample_id"].endswith("c741f52d"):
                    raise ValueError("unreviewed fresh native finding")
                quote = "EL TOR, ÉGYPTE"
                reviewed_candidate = read(
                    directory / "campaign/candidates" / (finding["sample_id"] + ".json")
                )
                if quote not in reviewed_candidate["joinedRawText"]:
                    raise ValueError("country adjudication evidence changed")
                decisions.append(
                    {
                        "findingIndex": index,
                        "findingSha256": digest(finding),
                        "decision": "rejected",
                        "evidence": [quote],
                        "reason": (
                            "ÉGYPTE is an eligible French rendering in the configured country "
                            "registry. Canonical target country names remain English; rendering "
                            "need not be English. This is not a route-role defect."
                        ),
                    }
                )
            if decisions:
                atomic_publish_json(
                    directory / "campaign/adjudications" / (sid + ".json"),
                    {
                        "sourceDocumentId": sid,
                        "reviewSha256": digest(review),
                        "candidateHashes": review["candidateHashes"],
                        "decisions": decisions,
                    },
                )
            if candidate["variant"] > 1:
                scoped_path = (
                    FRESH.parent
                    / "scoped-wording-audit/reviews"
                    / (candidate["documentId"] + "-" + sha256_file(candidate_path) + ".json")
                )
                scoped = read(scoped_path)
                if scoped["candidate"] != pin(candidate_path) or scoped["review"]["findings"]:
                    raise ValueError("fresh packing-scope control not clear")
            fresh.setdefault(sid, []).append(
                {
                    "sampleId": candidate["documentId"],
                    "groupCount": candidate["variant"],
                    "candidate": pin(candidate_path),
                    "positions": pin(ROOT / row["positions"]),
                    "review": pin(review_path),
                }
            )
    if len(fresh) != 11 or any(
        sorted(x["groupCount"] for x in rows) != [1, 2, 3] for rows in fresh.values()
    ):
        raise ValueError("fresh source controls incomplete")
    adjudication_path = REVISION / "main-adjudication.json"
    atomic_publish_json(
        adjudication_path,
        {
            "status": "PASS",
            "sources": fresh,
            "reviewScope": (
                "Main review of all fresh cargo descriptions/package rows; native full-text "
                "review; separate packing-scope review of all 22 multirow controls; production "
                "geometry checks. Not live customs certification or raster inspection of every PDF."
            ),
            "retainedControlReplays": [pin(p) for p in replay_paths],
            "retainedControls": len(carried),
            "freshControls": 33,
            "authorityDifference": (
                "Only the c08 private Marks organization owner and its source-contract digest "
                "changed. Other source numerical and ownership entries are identical."
            ),
        },
    )
    evidence = copy.deepcopy(old)
    evidence["boundedRefreshRepair"] = pin(adjudication_path)
    evidence["implementation"] = [
        {**p, "sha256": sha256_file(ROOT / p["path"])} for p in old["implementation"]
    ]
    changed = {
        p["path"]
        for p, q in zip(old["implementation"], evidence["implementation"], strict=True)
        if p != q
    }
    expected = {
        "src/document_ocr/synthesis/independent_sampling.py",
        "src/document_ocr/synthesis/independent_cargo_presentation.py",
    }
    if changed != expected:
        raise ValueError(f"unexpected production changes: {changed}")
    for sid, samples in fresh.items():
        row = evidence["sources"][sid]
        row["samples"] = [{**s, "adjudication": pin(adjudication_path)} for s in samples]
        row["sourceFiles"] = {
            name: pin(REGISTRY / "cases" / sid / name) for name in row["sourceFiles"]
        }
    # This certificate admits the actual, frozen pre-repair generation inputs;
    # c08 cannot use it because its source authority has changed.
    frozen = copy.deepcopy(evidence)
    del frozen["sources"][private_source]
    atomic_publish_json(REVISION / "frozen-generation-acceptance.json", frozen)
    evidence["authorities"] = authority_pins(ROOT, current)
    atomic_publish_json(REVISION / "acceptance.json", evidence)
    current["declared_packages"]["acceptance"] = pin(REVISION / "acceptance.json")
    atomic_publish_json(REVISION / "previous-production-config.json", previous)
    atomic_write_json(config_path, current)
    manifest_path = REGISTRY / "manifest.json"
    manifest = read(manifest_path)
    atomic_publish_json(REVISION / "previous-registry-manifest.json", manifest)
    manifest["currentProductionConfig"] = pin(config_path)
    manifest["multirowRefreshAuthorityRepair20261031"] = {
        "acceptance": pin(REVISION / "acceptance.json"),
        "adjudication": pin(adjudication_path),
        "sources": 456,
        "retainedControls": 1335,
        "freshControls": 33,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {"status": "CERTIFIED", "sources": 456, "retainedControls": 1335, "freshControls": 33}
        )
    )


if __name__ == "__main__":
    run()
