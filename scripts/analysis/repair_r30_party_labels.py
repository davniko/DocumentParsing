"""Stage reviewed party-only amendments with immutable original evidence."""

import copy
import hashlib
import json
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1"
OUT = ROOT / "artifacts/synthesis-probes/r30-binding-repair-20261108"
PARTY = OUT / "party-amendments"


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def rename_candidate(row, catalog):
    oldpath = ROOT / row["evidence"]["candidate"]["path"]
    candidate = json.loads(oldpath.read_text())
    replacements = {
        "v009": ("Reef Al Fujairah", "Al Rimal Trading"),
        "v011": ("Phum Anlung\nVil", "Mekong Loom\nCo."),
    }
    for key, (old, new) in replacements.items():
        assert candidate["renderValues"][key] == old
        candidate["renderValues"][key] = new
        matches = [e for e in candidate["proof"]["edits"] if e["key"] == key]
        assert matches and all(e["after"] == old for e in matches)
        for e in matches:
            e["after"] = new
    parties = candidate["target"]["documentPatch"]["parties"]
    parties["shipper"]["name"] = "MEKONG LOOM CO."
    parties["consignee"]["name"] = "AL RIMAL TRADING"
    parties["notifyParties"][0]["name"] = "AL RIMAL TRADING"
    source = (
        ROOT / catalog.entries[row["sourceDocumentId"]]["samplingSource"]["path"]
    ).read_bytes()
    cursor = 0
    chunks = []
    for e in candidate["proof"]["edits"]:
        a, b = e["byteStart"], e["byteEnd"]
        assert cursor <= a < b and source[a:b] == e["before"].encode()
        chunks.extend([source[cursor:a], e["after"].encode()])
        cursor = b
    chunks.append(source[cursor:])
    raw = b"".join(chunks)
    candidate["joinedRawText"] = raw.decode()
    candidate["joinedRawTextSha256"] = hashlib.sha256(raw).hexdigest()
    candidate["partyAmendment"] = {
        "previousCandidate": pin(oldpath),
        "scope": "Two name owners only; locality and postal values unchanged.",
    }
    path = PARTY / row["documentId"] / "candidate.json"
    atomic_write_json(path, candidate)
    rendered = catalog.render(path)
    # Preserve all reviewed target corrections unrelated to these names.
    expected = copy.deepcopy(row["target"])
    for role in ["shipper", "consignee"]:
        expected["parties"][role]["name"] = parties[role]["name"]
    expected["parties"]["notifyParties"][0]["name"] = parties["notifyParties"][0]["name"]
    assert rendered["target"] == expected, "Unrelated target drift during name-only correction"
    for role in ["shipper", "consignee"]:
        assert (
            rendered["target"]["parties"][role]["addressLine"]
            == row["target"]["parties"][role]["addressLine"]
        )
    renderpath = PARTY / row["documentId"] / "rendered.json"
    atomic_write_json(renderpath, rendered)
    result = copy.deepcopy(row)
    for key in ["bboxText", "joinedRawText", "pages", "positionedText", "target"]:
        result[key] = rendered[key]
    for key, text in [
        ("joinedRawTextSha256", "joinedRawText"),
        ("positionedTextSha256", "positionedText"),
        ("bboxTextSha256", "bboxText"),
    ]:
        if key in result:
            result[key] = hashlib.sha256(result[text].encode()).hexdigest()
    result["evidence"] = {**row["evidence"], "candidate": pin(path), "rendered": pin(renderpath)}
    return result


def stage():
    catalogpath = OUT / "catalog-draft.json"
    catalog = PaddleTemplateCatalog(ROOT, pin(catalogpath))
    branch = {
        "75006aac": ("deliveryAgent", "ALEXANDRIA BRANCH"),
        "5d6b477c": ("consignee", "TOWER BRANCH"),
        "70cdb6c8": ("notifyParties", "EL RAYHAN BRANCH"),
        "7f8fafe7": ("consignee", "EL MEHALLA EL KOBRA BRANCH"),
        "e745f5ba": ("consignee", "GARDEN BRANCH"),
    }
    review = []
    for split in ["real-train", "validation", "synthetic-train"]:
        for line in (DATA / f"{split}.jsonl").open():
            row = json.loads(line)
            old = copy.deepcopy(row)
            prefix = row["documentId"][4:12]
            if prefix in branch:
                role, qualifier = branch[prefix]
                p = row["target"]["parties"][role]
                p = p[0] if isinstance(p, list) else p
                assert p["addressLine"].startswith(qualifier)
                p["name"] += " " + qualifier
                p["addressLine"] = p["addressLine"][len(qualifier) :].lstrip(" ,")
            elif row["documentId"] == "syn_full_v7_825e02f81577153dbb10833c":
                row = rename_candidate(row, catalog)
            elif row["documentId"] == "syn_full_v7_6f0508c4703cf78f5137fd84":
                party = row["target"]["parties"]["forwardingAgent"]
                assert party["addressLine"] == "AL WARQAA BRANCH"
                party["name"] += " " + party.pop("addressLine")
                oldpath = ROOT / row["evidence"]["candidate"]["path"]
                candidate = json.loads(oldpath.read_text())
                original = candidate["target"]["documentPatch"]["parties"]["forwardingAgent"]
                original["name"] = party["name"]
                original.pop("addressLine")
                cp = PARTY / row["documentId"] / "candidate.json"
                atomic_write_json(cp, candidate)
                wording = None
                if row["evidence"].get("wording"):
                    wp = ROOT / row["evidence"]["wording"]["path"]
                    wording = json.loads(wp.read_text())
                    wording["candidateSha256"] = sha256_file(cp)
                    wording["review"] = (
                        "Main-agent target-only branch ownership correction; all requested "
                        "native wording and shipment facts unchanged."
                    )
                    newwp = PARTY / row["documentId"] / "wording.json"
                    atomic_write_json(newwp, wording)
                    row["evidence"]["wording"] = pin(newwp)
                rendered = catalog.render(cp, wording_receipt=wording)
                assert rendered["target"] == row["target"]
                assert rendered["pages"] == row["pages"]
                rp = PARTY / row["documentId"] / "rendered.json"
                atomic_write_json(rp, rendered)
                row["evidence"]["candidate"] = pin(cp)
                row["evidence"]["rendered"] = pin(rp)
            else:
                continue
            assert (
                BillOfLadingExtractionV7DirectTarget.model_validate_json(
                    json.dumps(row["target"])
                ).canonical_target()
                == row["target"]
            )
            oldpath = PARTY / row["documentId"] / "before.json"
            atomic_write_json(oldpath, old)
            row["evidence"]["posttrainingPartyCorrection"] = {
                "previousRecord": pin(oldpath),
                "review": (
                    "Explicit own-branch qualifier belongs to identity, not street address; "
                    "not a landmark or care-of recipient."
                )
                if prefix in branch
                else (
                    "Reviewed replacement names; addresses/localities and all unrelated facts "
                    "retained."
                ),
            }
            newpath = PARTY / row["documentId"] / "record.json"
            atomic_write_json(newpath, row)
            review.append(
                {
                    "split": split,
                    "id": row["documentId"],
                    "before": old["target"]["parties"],
                    "after": row["target"]["parties"],
                    "record": pin(newpath),
                }
            )
    assert len(review) == 7
    atomic_write_json(PARTY / "review.json", {"records": review})
    print(json.dumps([{"id": r["id"], "split": r["split"]} for r in review]))


if __name__ == "__main__":
    stage()
