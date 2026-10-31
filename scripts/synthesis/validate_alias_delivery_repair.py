"""Check the seven-record country patch and refresh only its manifest hash."""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json, atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.synthesis.curated_reflow_policy import FontMetrics


def main():
    root = Path.cwd()
    work = root / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/package-alias-rollout-20261030"
    evidence = work / "delivery-country-repair"
    review = json.loads((evidence / "reviewed-patch.json").read_text())
    path = Path("data/curated/mpci-bl-r28-positions-v1/synthetic-train.jsonl")
    before = (evidence / "before" / path).read_text().splitlines(keepends=True)
    after = path.read_text().splitlines(keepends=True)
    assert len(before) == len(after) == 6900
    changed = []
    for a, b in zip(before, after, strict=True):
        if a == b:
            continue
        old, new = json.loads(a), json.loads(b)
        assert new["sourceDocumentId"] == review["sourceId"]
        BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(new["target"]))
        expected = next(
            r for r in review["records"] if r["after"]["documentId"] == new["documentId"]
        )
        assert old == expected["before"] and new == expected["after"]
        # Reverse only the approved target/input deltas; everything else agrees.
        check = copy.deepcopy(new)
        agent = check["target"]["parties"]["deliveryAgent"]
        assert agent.pop("country") == "UNITED ARAB EMIRATES"
        assert agent["addressLine"].endswith(" UAE")
        agent["addressLine"] = agent["addressLine"][:-4]
        for key in ("joinedRawText", "positionedText"):
            assert len(new[key].splitlines()) == len(old[key].splitlines())
            if key == "positionedText":
                assert [line.rsplit(" ||", 1)[-1] for line in new[key].splitlines()] == [
                    line.rsplit(" ||", 1)[-1] for line in old[key].splitlines()
                ]
            check[key] = old[key]
            check[key + "Sha256"] = old[key + "Sha256"]
        assert check == old
        changed.append(new["documentId"])
    assert len(changed) == 7
    config = json.loads((work / "inspection/inventory.json").read_text())["config"]
    font = config["positions"]["reflow"]
    metrics = FontMetrics(root / font["font_path"], font["font_sha256"])
    assert metrics.length("UAE") < metrics.length("EGYPT")
    result = {
        "status": "PASS",
        "changedRecords": changed,
        "unchangedRecords": 6893,
        "fields": [
            "deliveryAgent.addressLine",
            "deliveryAgent.country",
            "isolated country input span",
        ],
        "allCoordinateSuffixesPreserved": True,
        "allOtherTextAndTargetsPreserved": True,
        "outputSha256": sha256_file(path),
    }
    atomic_publish_json(evidence / "validation.json", result)
    manifest_path = path.parent / "manifest.json"
    atomic_write_bytes(evidence / "before/r28-manifest.json", manifest_path.read_bytes())
    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"][path.name] == review["beforeHashes"][str(path)]
    manifest["files"][path.name] = sha256_file(path)
    manifest["aliasRolloutDeliveryCountry20261030"] = {
        "records": 7,
        "evidence": str(evidence.relative_to(root) / "validation.json"),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
