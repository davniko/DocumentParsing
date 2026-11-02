"""Certify a main-reviewed source-group subset without admitting unresolved drafts.

All paid reviews, exact candidate replay, scoped cargo checks, schema validation
and geometry remain mandatory. This creates an immutable campaign publication;
it does not append anything to the working R28 dataset.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

from check_package_alias_candidates import check
from refresh_r28_multirow import subset_config

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_positions import position_campaign
from document_ocr.synthesis.curated_publication import _adjudicate, publish_campaign
from document_ocr.synthesis.curated_wording import RenderedReview
from document_ocr.synthesis.declared_package_acceptance import SOURCE_FILES, authority_pins
from document_ocr.synthesis.declared_package_preparation import selection

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--control", action="store_true", help="Certify validation controls outside the training-addition inventory")
    args = parser.parse_args()
    started = time.perf_counter()
    config_path, review_path = args.config.resolve(), args.review.resolve()
    config, review = read(config_path), read(review_path)
    if not review.get("mainReview") or review["status"] != "APPROVED_FOR_COMPONENT_VALIDATION":
        raise ValueError("an explicit main-agent review is required")
    excluded = review["excludedSources"]
    if not all(excluded.values()) or not set(excluded) <= set(config["source_ids"]):
        raise ValueError("excluded sources require exact identities and reasons")
    sources = [s for s in config["source_ids"] if s not in excluded]
    directory = OUT / ("authority-controls" if args.control else "components") / args.name
    path = subset_config(config_path, sources, directory)
    current = read(path)
    campaign = create_campaign(ROOT, current)
    with (campaign.output / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (campaign.output / "manifest.json").exists():
            raise ValueError("component already published; verify its receipt instead of replaying")
        samples, selected_ids = {}, set()
        for sid in sources:
            receipt_path = campaign.output / "reviews" / (sid + ".json")
            receipt = read(receipt_path)
            candidates = {}
            for variant in range(1, campaign.variant_counts[sid] + 1):
                sample = "syn_full_v7_" + digest([sid, current["seed"], variant])[:24]
                cp = campaign.output / "candidates" / (sample + ".json")
                candidate = read(cp)
                if receipt["candidateHashes"].get(sample) != digest(candidate):
                    raise ValueError("native review is stale: " + sample)
                scope = (
                    RUN / "scoped-cargo-review/reviews" / (sample + "-" + sha256_file(cp) + ".json")
                )
                scoped = read(scope)
                if scoped["candidate"] != pin(cp):
                    raise ValueError("scoped cargo review is stale")
                for index, finding in enumerate(scoped["review"]["findings"]):
                    decision = review.get("rejectedScopedFindings", {}).get(f"{sample}:{index}")
                    if (
                        not decision
                        or decision["finding"] != finding
                        or not decision["reason"]
                        or not decision["evidence"]
                        or any(q not in candidate["joinedRawText"] for q in decision["evidence"])
                    ):
                        raise ValueError("unresolved scoped cargo finding: " + sample)
                selected_ids.add(sample)
                candidates[sample] = candidate
                samples[sample] = {
                    "sourceDocumentId": sid,
                    "variant": variant,
                    "groupCount": selection(current["declared_packages"], sid, variant),
                    "candidate": pin(cp),
                    "review": pin(receipt_path),
                    "scopedReview": pin(scope),
                }
            decisions = []
            for index, finding in enumerate(receipt["review"]["findings"]):
                decision = review["rejectedFindings"].get(f"{sid[4:12]}:{index}")
                if not decision or not decision["reason"] or not decision["evidence"]:
                    raise ValueError("unresolved complete-document finding: " + sid)
                if decision["finding"] != finding:
                    raise ValueError("main decision belongs to a different native finding")
                decisions.append(
                    {
                        "findingIndex": index,
                        "findingSha256": digest(finding),
                        "decision": "rejected",
                        "reason": decision["reason"],
                        "evidence": decision["evidence"],
                    }
                )
            if decisions:
                atomic_publish_json(
                    campaign.output / "adjudications" / (sid + ".json"),
                    {
                        "sourceDocumentId": sid,
                        "reviewSha256": digest(receipt),
                        "candidateHashes": receipt["candidateHashes"],
                        "decisions": decisions,
                    },
                )
            _adjudicate(
                campaign.output,
                sid,
                receipt,
                candidates,
                RenderedReview.model_validate(receipt["review"]),
            )
        geometry = check(path, directory / "geometry")
        rows = geometry["rows"]
        if len(rows) != len(samples) or any(r["status"] != "PASS" for r in rows):
            raise ValueError("component has missing or held replay/geometry")
        for row in rows:
            samples[row["sampleId"]]["positions"] = pin(ROOT / row["positions"])
        audited_path = directory / "main-audit.json"
        atomic_publish_json(
            audited_path,
            {
                "status": "PASS",
                "decision": pin(review_path),
                "preparedConfig": pin(config_path),
                "samples": samples,
                "excludedSources": excluded,
                "basis": "Current full-text native review and explicit main adjudication, separate cargo review, exact replay, schema and geometry. Not automatic admission from structural checks alone.",
            },
        )
        old = read(ROOT / current["declared_packages"]["acceptance"]["path"])
        implementation = {p["path"] for p in old["implementation"]}
        implementation.add("src/document_ocr/synthesis/independent_thermal.py")
        certificate = {
            "status": "certified_native_synthesis_and_geometry",
            "authorities": authority_pins(ROOT, current),
            "implementation": [pin(ROOT / p) for p in sorted(implementation)],
            "sources": {},
        }
        for sid in sources:
            retained = [
                dict(v, sampleId=k, adjudication=pin(audited_path))
                for k, v in samples.items()
                if v["sourceDocumentId"] == sid
            ]
            certificate["sources"][sid] = {
                "status": "PASS",
                "groupCounts": sorted({v["groupCount"] for v in retained}),
                "contractSha256": digest(campaign.declared_package_contracts[sid]),
                "sourceFiles": {
                    n: pin(ROOT / current["source_contracts"] / sid / n)
                    for n in sorted(SOURCE_FILES)
                },
                "samples": retained,
            }
        cert_path = directory / "acceptance.json"
        atomic_publish_json(cert_path, certificate)
        current["declared_packages"]["acceptance"] = pin(cert_path)
        admission_config = directory / "admission-config.json"
        atomic_publish_json(admission_config, current)
        campaign = create_campaign(ROOT, current)
        # Preserve unresolved drafts, but keep the publication directory exactly
        # equal to its explicitly reviewed source-group scope.
        held = []
        for p in sorted((campaign.output / "candidates").glob("*.json")):
            if p.stem in selected_ids:
                continue
            candidate = read(p)
            sid = candidate["sourceDocumentId"]
            if sid not in excluded:
                raise ValueError("unexpected candidate outside reviewed exclusions")
            dest = campaign.output / "held-candidates" / (p.stem + "-" + sha256_file(p) + ".json")
            held.append(
                {
                    "original": pin(p),
                    "preservedAt": str(dest.relative_to(ROOT)),
                    "reason": excluded[sid],
                }
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise ValueError("held draft already has an archive; refusing to overwrite")
            p.rename(dest)
        atomic_publish_json(directory / "held-drafts.json", held)
        published = publish_campaign(campaign, publish=True)
        positions = position_campaign(campaign)
        receipt = {
            "status": "CERTIFIED_COMPONENT_NOT_YET_APPENDED_TO_R28",
            "samples": len(samples),
            "sources": len(sources),
            "config": pin(admission_config),
            "publication": published,
            "positions": positions,
            "mainAudit": pin(audited_path),
            "seconds": time.perf_counter() - started,
        }
        atomic_publish_json(directory / "certification.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
