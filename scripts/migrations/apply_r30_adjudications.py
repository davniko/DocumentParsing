"""Stage explicit validation decisions; never publish or anonymize training rows.

Inputs remain native Paddle text. A missing pointer, proposal or decision fails
the batch rather than restoring an earlier label implicitly.
"""

from __future__ import annotations

import copy
import json
import re
import time

from jsonschema import Draft202012Validator
from prepare_r30_paddle_pilot import ART, BASE, OUT, load_pages, read
from review_r30_paddle import differences

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.spatial_inputs.paddle import fingerprint, serialize


def parts(pointer):
    if not pointer.startswith("/") or pointer == "/":
        raise ValueError(f"invalid field pointer: {pointer}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def get(value, pointer):
    for part in parts(pointer):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def assign(target, pointer, value):
    keys = parts(pointer)
    parent = target
    for part in keys[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    key = int(keys[-1]) if isinstance(parent, list) else keys[-1]
    parent[key] = copy.deepcopy(value)


def adjudicate(current, candidate, decision):
    target = copy.deepcopy(current)
    if not decision.get("notes"):
        raise ValueError("decision requires review rationale")
    for path in decision["acceptCandidatePaths"]:
        assign(target, path, get(candidate, path))
    for path, value in decision["set"].items():
        assign(target, path, value)
    for path in decision["remove"]:
        keys = parts(path)
        parent = target
        for part in keys[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        del parent[int(keys[-1]) if isinstance(parent, list) else keys[-1]]
    return target


def original_lookup(record):
    """Undo only exact, still-current anonymized leaves for private review."""
    result = copy.deepcopy(record["current"]["target"])
    for update in record["privacyPlan"]["targetUpdates"]:
        keys = re.findall(r"[^.\[\]]+", update["path"])
        if keys[0] == "documentPatch":
            keys = keys[1:]
        path = "/" + "/".join(keys)
        try:
            current = get(result, path)
        except (KeyError, IndexError):
            continue  # A retired field must not be restored by privacy reversal.
        if current == update["after"]:
            assign(result, path, update["before"])
    return result


def main():
    start = time.perf_counter()
    path = OUT / "adjudications.json"
    decisions = read(path)["documents"]
    validator = Draft202012Validator(read(BASE / "schema.json"))
    staged, errors, real = [], [], []
    for source_path in sorted((ART / "sources").glob("*.json")):
        record = read(source_path)
        did = record["documentId"]
        try:
            decision = decisions[did[4:12]]
            proposal_path = ART / "target-reviews" / source_path.name
            proposal = read(proposal_path) if proposal_path.exists() else None
            candidate = proposal["candidateTarget"] if proposal else None
            if record["split"] == "real-train":
                old = proposal["lookupTarget"] if proposal else original_lookup(record)
            else:
                old = record["current"]["target"]
            target = adjudicate(old, candidate, decision)
            validator.validate(target)
            BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(target))
            pages = load_pages(record)
            if fingerprint(pages) != record["nativeFingerprint"]:
                raise ValueError("native source fingerprint changed")
            row = {
                "status": "ADJUDICATED_VALIDATION_REQUIRES_FINAL_EVIDENCE_AUDIT"
                if record["split"] == "validation"
                else "PRIVATE_REAL_TARGET_REQUIRES_ANONYMIZATION_AND_AUDIT",
                "documentId": did,
                "target": target,
                "sourceSha256": sha256_file(source_path),
                "decisionsSha256": sha256_file(path),
                "nativeFingerprint": fingerprint(pages),
                "positionedText": serialize(pages),
                "bboxText": serialize(pages, geometry="bbox"),
                "joinedRawText": serialize(pages, geometry="none"),
                "changes": list(differences(old, target)),
                "review": decision,
            }
            (staged if record["split"] == "validation" else real).append(row)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            errors.append({"id": did, "error": str(exc)})
        except Exception as exc:
            # JSON Schema errors retain their precise path but not the verbose
            # full schema/value dump in the operational receipt.
            if not hasattr(exc, "json_path"):
                raise
            errors.append({"id": did, "error": f"{exc.json_path}: {exc.message}"})
    receipt = {
        "validation": len(staged),
        "privateRealReview": len(real),
        "errors": errors,
        "seconds": time.perf_counter() - start,
    }
    atomic_write_json(OUT / "validation-adjudication-check.json", receipt)
    if errors or len(staged) != 60 or len(real) != 20:
        print(json.dumps(receipt, indent=2))
        raise ValueError("staging requires all 80 reviewed, valid decisions")
    for row in staged:
        atomic_write_json(ART / "validation-rebased" / (row["documentId"] + ".json"), row)
    for row in real:
        atomic_write_json(ART / "real-target-review" / (row["documentId"] + ".json"), row)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
