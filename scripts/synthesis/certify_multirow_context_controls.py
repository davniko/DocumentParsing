"""Record main-reviewed live controls for the final semantic-only prompt projection."""

import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest

ROOT = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def main():
    rows = []
    batches = []
    for number in (40, 41, 42):
        batch = ROOT / f"batch-{number}"
        summary_path = sorted((batch / "stage-runs").glob("*/summary.json"))[-1]
        summary = read(summary_path)
        stages = summary["stages"]
        for stage in ("generate", "postal", "render", "review"):
            if stages[stage]["errors"]:
                raise ValueError("live control still has a failed stage")
        if stages["render"]["accepted"] != 3 or stages["geometry"]["counts"] != {"PASS": 3}:
            raise ValueError("three rendered and geometry-checked controls are required")
        for row in stages["geometry"]["rows"]:
            cp, pp = Path(row["candidate"]), Path(row["positions"])
            candidate, positions = read(cp), read(pp)
            rp = batch / "campaign/reviews" / (candidate["sourceDocumentId"] + ".json")
            review = read(rp)
            sample = candidate["documentId"]
            if (
                positions["candidateSha256"] != sha256_file(cp)
                or positions["target"] != candidate["target"]
                or review["candidateHashes"][sample] != digest(candidate)
            ):
                raise ValueError("live control evidence is stale")
            goods = candidate["target"]["documentPatch"]["goodsItemDetails"]
            if len(goods) != 1 or len(goods[0]["numberAndTypeOfPackages"]) != candidate["variant"]:
                raise ValueError("live control lost its public group count")
            decisions = {}
            for finding in review["review"]["findings"]:
                if finding["sample_id"] != sample:
                    continue
                if number != 42:
                    raise ValueError("an unreviewed fresh finding remains")
                if finding["problem"].startswith("None"):
                    reason = "Affirmative consistency observation; no defect was alleged."
                elif finding["field"] == "rider gross cargo weight block":
                    if len(goods[0]["splitGoodsPlacement"]) != 8 or any(
                        "packageQuantity" in allocation
                        for allocation in goods[0]["splitGoodsPlacement"]
                    ):
                        raise ValueError("source allocation absence was not preserved")
                    reason = (
                        "The reviewed source has eight containers and seven visible rider "
                        "weight cells. Retain that occurrence mask. Shipment totals cover "
                        "all eight privately planned loads; an omitted eighth printed weight "
                        "is not contradictory arithmetic and must not be invented. This is "
                        "the same source-supported decision as the accepted batch-30 controls."
                    )
                else:
                    raise ValueError("unexpected finding requires main-agent review")
                decisions[digest(finding)] = reason
            if "ACETIC ANHYDRIQUE" in candidate["joinedRawText"]:
                raise ValueError("reviewed lexical artifact remains")
            rows.append(
                {
                    "status": "PASS",
                    "batch": number,
                    "sourceId": candidate["sourceDocumentId"],
                    "sampleId": sample,
                    "groupCount": candidate["variant"],
                    "candidate": pin(cp),
                    "positions": pin(pp),
                    "review": pin(rp),
                    "findingDecisions": decisions,
                    "mainReview": (
                        "Read generated party/cargo text, target roles, declaration/description "
                        "boundaries, repeated group quantities and net/gross facts; compared "
                        "public source visibility and inspected coordinate-check evidence."
                    ),
                }
            )
        batches.append(pin(summary_path))
    atomic_publish_json(
        ROOT / "fresh-context-controls.json",
        {
            "status": "PASS",
            "controls": len(rows),
            "sources": 3,
            "batches": batches,
            "rows": rows,
            "unresolvedFindings": 0,
            "lexicalCorrection": pin(ROOT / "batch-41/context-lexical-correction.json"),
            "sampling": (
                "Fresh native wording of nine previously frozen shipment controls, "
                "not nine new independent identities."
            ),
            "projectionImplementation": pin(
                Path("src/document_ocr/synthesis/independent_presentation.py")
            ),
            "trainingDatasetMutation": False,
        },
    )


if __name__ == "__main__":
    main()
