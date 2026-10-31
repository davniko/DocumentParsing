"""Read back bounded dataset repairs and account for the full alias rollout."""

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget


def main():
    work = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/package-alias-rollout-20261030")
    acceptance = json.loads((work / "acceptance.json").read_text())
    inventory = json.loads((work / "inspection/inventory.json").read_text())
    families = Counter(r["family"] for r in inventory["sources"])
    draws = list((work / "batches").glob("**/campaign/candidates/*.json"))
    calls = list((work / "batches").glob("**/campaign/calls/*.json"))
    costs, stages, errors, billing = Decimal(0), Counter(), Counter(), Counter()
    tokens = Counter()
    for path in calls:
        call = json.loads(path.read_text())
        stages[call["stage"]] += 1
        billing[call["costStatus"]] += 1
        if call["costUsd"] is not None:
            costs += Decimal(call["costUsd"])
        if call["error"]:
            errors[call["stage"]] += 1
        for key in ("inputTokens", "outputTokens", "cacheReadTokens"):
            tokens[key] += call.get("usage", {}).get(key, 0) or 0
    styles, categories, nouns, roles = Counter(), Counter(), Counter(), Counter()
    changed_documents, shortened, compounds, noops = [], [], [], []
    for record in acceptance["records"]:
        row = json.loads(Path(record["candidate"]["path"]).read_text())
        slots = row["packagePresentations"]["slots"]
        for slot in slots:
            styles[slot["style"]] += 1
            roles[slot["role"]] += 1
            nouns[slot["after"]] += 1
            categories[slot["category"]] += 1
        if any(s["before"] != s["after"] for s in slots):
            changed_documents.append(record["sourceId"])
        else:
            noops.append(record["sourceId"])
        if any(s["category"] != s["presentedCategory"] for s in slots):
            shortened.append(record["sourceId"])
        if any(s["mode"] == "compound" for s in row["packagePresentations"]["nonScalarOwners"]):
            compounds.append(record["sourceId"])
    dataset = Path("data/curated/mpci-bl-r28-positions-v1")
    current_path = dataset / "synthetic-train.jsonl"
    before_path = work / "delivery-country-repair/before" / current_path
    expected = {}
    for folder in ("delivery-country-repair", "legacy-literal-repair"):
        patch = json.loads((work / folder / "reviewed-patch.json").read_text())
        for row in patch["records"]:
            sid = row["after"]["documentId"]
            assert sid not in expected
            expected[sid] = row
    changed = []
    before_lines = before_path.read_text().splitlines(keepends=True)
    after_lines = current_path.read_text().splitlines(keepends=True)
    assert len(before_lines) == len(after_lines) == 6900
    for old, new in zip(before_lines, after_lines, strict=True):
        if old == new:
            continue
        a, b = json.loads(old), json.loads(new)
        sid = b["documentId"]
        assert expected[sid] == {**expected[sid], "before": a, "after": b}
        BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(b["target"]))
        assert [s.rsplit(" ||", 1)[-1] for s in a["positionedText"].splitlines()] == [
            s.rsplit(" ||", 1)[-1] for s in b["positionedText"].splitlines()
        ]
        changed.append(sid)
    assert set(changed) == set(expected) and len(changed) == 19
    # Pre-rollout file hashes recorded before any alias integration mutation.
    untouched = {
        "real-train.jsonl": "e52fd7b93eb86d972c28c80f93aabdb747c87cf75c254da54f42c99afd609a19",
        "validation.jsonl": "580ed0d40aea03d4edb0e085d614377149a24a2a468bb01ff1d9574e4b96e5d2",
        "previous-uae-synthetic.jsonl": (
            "b6fd9f57fb75c5e2b3f52b0e20b6c5532f868cbe305d51fb385d099df6854c9f"
        ),
        "old-uae-addition.jsonl": (
            "cea21e1579317e3ac3ef08858f8cdf85cd452b16034e0b63657a6f9857ab1375"
        ),
    }
    for name, expected_hash in untouched.items():
        assert sha256_file(dataset / name) == expected_hash, name
    manifest = json.loads((dataset / "manifest.json").read_text())
    assert manifest["files"][current_path.name] == sha256_file(current_path)
    result = {
        "status": "PASS",
        "acceptedNativeDocuments": len(acceptance["records"]),
        "renderedCandidateFilesIncludingSuperseded": len(draws),
        "callReceiptsIncludingRetries": len(calls),
        "callStages": dict(stages),
        "callErrors": dict(errors),
        "costStatuses": dict(billing),
        "recordedCostUsd": str(costs),
        "tokens": dict(tokens),
        "sourceFamilies": dict(families),
        "scalarStyles": dict(styles),
        "scalarRoles": dict(roles),
        "scalarCategories": dict(categories),
        "distinctScalarRenderedForms": len(nouns),
        "scalarChanges": sum(r["aliasChanges"] for r in acceptance["records"]),
        "documentsWithScalarChanges": len(changed_documents),
        "documentsWithoutScalarChanges": noops,
        "documentsWithFamilyShortening": shortened,
        "documentsWithCompoundOwners": compounds,
        "dataset": {
            "changedRecords": changed,
            "unchangedRecords": 6881,
            "syntheticSha256": sha256_file(current_path),
            "untouchedFiles": untouched,
            "membershipAndCoordinateSuffixesPreserved": True,
        },
    }
    atomic_publish_json(work / "final-accounting.json", result)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}))


if __name__ == "__main__":
    main()
