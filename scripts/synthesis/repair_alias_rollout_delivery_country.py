"""Apply the PDF-reviewed ab905 delivery-country continuation repair.

The page-three country continues the page-two agent address. Existing seven UAE
descendants use the approved shorter UAE alias in that isolated, centered region;
it fits inside the old EGYPT region without moving its centroid or other lines.
Original records and template authorities are retained in the evidence directory.
"""

import argparse
import copy
import json
import re
from pathlib import Path

from document_ocr.atomic import atomic_publish_json, atomic_write_bytes, atomic_write_json
from document_ocr.hashing import canonical_json_sha256, sha256_file
from document_ocr.synthesis.curated import digest

SID = "doc_ab905fd40d2afadc45840dc00b569e510bfd6a1017218d16c2ce125ea38288a7"
REG = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    evidence = REG / "package-alias-rollout-20261030/delivery-country-repair"
    dataset = REG / "source-authorities/handling-20261030/source-dataset"
    train = dataset / "train.jsonl"
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    source = next(r for r in rows if r["documentId"] == SID)
    original_source = copy.deepcopy(source)
    party = source["target"]["documentPatch"]["parties"]["deliveryAgent"]
    assert "country" not in party and party["addressLine"].endswith("CAIRO")
    party.update(addressLine=party["addressLine"] + " EGYPT", country="EGYPT")
    case = REG / "cases" / SID
    contract = json.loads((case / "contract.json").read_text())
    original_contract = copy.deepcopy(contract)
    offset = source["joinedRawText"].index("\nEGYPT\nShipped on Board") + 1
    occurrence = sum(m.start() <= offset for m in re.finditer("EGYPT", source["joinedRawText"]))
    contract["contract"]["variables"].append(
        {
            "key": "delivery_country_continuation",
            "kind": "postal",
            "value": "EGYPT",
            "meaning": "Delivery-agent country continues from page 2 to page 3; country only",
            "required_literals": [],
            "occurrences": [{"text": "EGYPT", "occurrence": occurrence, "presentation": "text"}],
        }
    )
    for binding in contract["contract"]["targets"]:
        if binding["path"] == "documentPatch.parties.deliveryAgent.addressLine":
            assert binding["expression"] == "{postal_5}"
            binding["expression"] += " {delivery_country_continuation}"
    contract["contract"]["targets"].append(
        {
            "path": "documentPatch.parties.deliveryAgent.country",
            "expression": "{delivery_country_continuation}",
        }
    )
    contract["targetSha256"] = digest(source["target"])
    # Explicitly reviewed current descendants; no country inference from ports.
    current = Path("data/curated/mpci-bl-r28-positions-v1/synthetic-train.jsonl")
    lines, changes = [], []
    for line in current.read_text().splitlines(keepends=True):
        row = json.loads(line)
        if row.get("sourceDocumentId") != SID:
            lines.append(line)
            continue
        before = copy.deepcopy(row)
        text = row["joinedRawText"]
        assert text.count("\nEGYPT\nShipped on Board") == 1
        row["joinedRawText"] = text.replace("\nEGYPT\nShipped on Board", "\nUAE\nShipped on Board")
        positioned, count = re.subn(
            r"(?m)^EGYPT( \|\| [^\n]+)\n(?=Shipped on Board)", r"UAE\1\n", row["positionedText"]
        )
        assert count == 1
        row["positionedText"] = positioned
        target = row["target"].get("documentPatch", row["target"])
        agent = target["parties"]["deliveryAgent"]
        assert "country" not in agent
        agent["addressLine"] += " UAE"
        agent["country"] = "UNITED ARAB EMIRATES"
        for key in ("joinedRawText", "positionedText"):
            row[key + "Sha256"] = digest(row[key].encode())
        changes.append({"before": before, "after": row})
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    assert len(changes) == 7
    result = {
        "sourceId": SID,
        "sourceBefore": original_source,
        "sourceAfter": source,
        "contractBefore": original_contract,
        "contractAfter": contract,
        "records": changes,
        "countryEvidence": (
            "PDF page 2 delivery-agent address ends CAIRO; page 3 starts EGYPT "
            "in same continued column"
        ),
        "syntheticCountryDecision": (
            "Seven manually reviewed UAE delivery-agent addresses; render UAE, "
            "normalized country UNITED ARAB EMIRATES"
        ),
        "coordinates": (
            "Shorter country alias centered in unchanged isolated region; "
            "centroid and every other coordinate unchanged"
        ),
        "beforeHashes": {
            str(p): sha256_file(p)
            for p in (train, current, case / "target.json", case / "contract.json")
        },
    }
    atomic_publish_json(evidence / "reviewed-patch.json", result)
    if args.apply:
        for p in (
            train,
            current,
            case / "target.json",
            case / "contract.json",
            dataset / "manifest.json",
        ):
            backup = evidence / "before" / p
            if not backup.exists():
                atomic_write_bytes(backup, p.read_bytes())
        atomic_write_json(case / "contract.json", contract)
        atomic_write_json(case / "target.json", source["target"])
        atomic_write_bytes(
            train,
            (
                "".join(
                    json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
                )
            ).encode(),
        )
        atomic_write_bytes(current, "".join(lines).encode())
        manifest = json.loads((dataset / "manifest.json").read_text())
        manifest["output_sha256"]["train.jsonl"] = sha256_file(train)
        for row in manifest["source_records"]:
            if row["documentId"] == SID:
                row["canonicalTargetSha256"] = canonical_json_sha256(source["target"])
                row["targetSha256"] = digest(source["target"])
        manifest["deliveryCountryRepair20261030"] = {
            "sourceId": SID,
            "evidence": str(evidence / "reviewed-patch.json"),
        }
        atomic_write_json(dataset / "manifest.json", manifest)
    print(json.dumps({"source": SID, "syntheticRecords": len(changes), "applied": args.apply}))


if __name__ == "__main__":
    main()
