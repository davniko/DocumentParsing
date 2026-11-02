"""Align the explicitly polyethylene source category and its linked authorities."""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_geography import IndependentGeographyGate, load_independent_geography_certificates
from document_ocr.synthesis.independent_numeric import _digest as numeric_digest
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
OUT = REGISTRY / "expansion-20261101-v20"


def main():
    parent = REGISTRY / "expansion-20261101-v19/complete-source-authority-config.json"
    base = json.loads(parent.read_text())
    config = copy.deepcopy(base)
    sid = next(s for s in base["source_ids"] if s.startswith("doc_630cd159"))
    old = _SourceOwnershipCampaign(ROOT, base).blueprint(sid)
    expected = copy.deepcopy(old.target)
    package = expected["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]
    if package != {"packageQuantity": 72, "typeCategory": "PACKAGE_DRUM"}:
        raise ValueError("reviewed source category changed")
    if "POLYETHYLENE DRUM" not in old.source:
        raise ValueError("source material evidence absent")
    package["typeCategory"] = "PACKAGE_DRUM_PLASTIC"
    dataset = ROOT / base["dataset"]
    rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
    matches = [row for row in rows if row["documentId"] == sid]
    if len(matches) != 1 or matches[0]["target"] != old.target:
        raise ValueError("source dataset authority differs")
    matches[0]["target"] = expected
    new_data = OUT / "source-dataset"
    atomic_publish_bytes(new_data / "train.jsonl", b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows
    ))
    for path in dataset.rglob("*"):
        relative = path.relative_to(dataset)
        if path.is_file() and relative not in {Path("train.jsonl"), Path("manifest.json")}:
            atomic_publish_bytes(new_data / relative, path.read_bytes())
    manifest = json.loads((dataset / "manifest.json").read_text())
    for name, value in manifest["output_sha256"].items():
        if sha256_file(dataset / name) != value:
            raise ValueError("source dataset input hash changed")
        if name != "train.jsonl" and sha256_file(new_data / name) != value:
            raise ValueError("unrelated source data changed")
    manifest["dataset"] = str(new_data.relative_to(ROOT))
    manifest["output_sha256"]["train.jsonl"] = sha256_file(new_data / "train.jsonl")
    manifest.setdefault("provenance", {})["drumMaterial20261101"] = {
        "sourceDocumentId": sid, "change": "Explicit POLYETHYLENE DRUM establishes PLASTIC for the single public drum identity; no source text, counts or allocations change.",
    }
    atomic_publish_json(new_data / "manifest.json", manifest)
    new_cases = OUT / "cases"
    for source in base["source_ids"]:
        for path in (ROOT / base["source_contracts"] / source).iterdir():
            if not path.is_file():
                continue
            destination = new_cases / source / path.name
            if source == sid and path.name == "target.json":
                atomic_publish_json(destination, expected)
            elif source == sid and path.name == "contract.json":
                envelope = json.loads(path.read_text())
                if envelope["targetSha256"] != digest(old.target):
                    raise ValueError("source contract target pin changed")
                envelope["targetSha256"] = digest(expected)
                atomic_publish_json(destination, envelope)
            else:
                atomic_publish_bytes(destination, path.read_bytes())
    config.update(dataset=str(new_data.relative_to(ROOT)), source_contracts=str(new_cases.relative_to(ROOT)))
    after = _SourceOwnershipCampaign(ROOT, config).blueprint(sid)
    if after.source != old.source or after.target != expected or after.contract != old.contract:
        raise ValueError("material target rebase changed unrelated source owners")
    for title, path in (
        ("numeric", base["independent"]["numeric_certificates"]),
        ("groups", base["declared_packages"]["contracts"]),
        ("aliases", base["aliases"]["presentations"]["certificates"]),
    ):
        catalog = json.loads((ROOT / path["path"]).read_text())
        entry = catalog["sources"][sid]
        field = "sourcePublicTargetSha256" if title == "numeric" else "sourceTargetSha256"
        hash_fn = numeric_digest if title == "numeric" else digest
        if entry[field] != hash_fn(old.target):
            raise ValueError("original authority pin differs: " + title)
        entry[field] = hash_fn(after.target)
        destination = OUT / (title + "-certificates.json")
        atomic_publish_json(destination, catalog)
        pin = {"path": str(destination.relative_to(ROOT)), "sha256": sha256_file(destination)}
        if title == "numeric":
            config["independent"]["numeric_certificates"] = pin
        elif title == "groups":
            config["declared_packages"]["contracts"] = pin
        else:
            config["aliases"]["presentations"]["certificates"] = pin
    gp = base["independent"]["geography_certificates"]
    raw = (ROOT / gp["path"]).read_bytes()
    gate = IndependentGeographyGate(load_independent_geography_certificates(raw, expected_sha256=gp["sha256"]))
    certificate = gate._certificate(old, allow_prepared=False)
    if certificate is not None:
        if gate._postal_state(old, certificate) != gate._postal_state(after, certificate):
            raise ValueError("postal state changed")
        catalog = json.loads(raw)
        pins = gate._blueprint_pins(after)
        catalog["sources"][sid].update(effective_contract_sha256=pins[0], ownership_sha256=pins[1])
        path = OUT / "geography_certificates.json"
        atomic_publish_json(path, catalog)
        config["independent"]["geography_certificates"] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    atomic_publish_json(OUT / "complete-source-authority-config.json", config)
    atomic_publish_json(OUT / "review.json", {
        "status": "REQUIRES_FRESH_SYNTHESIS", "sourceDocumentId": sid,
        "beforeTarget": old.target, "afterTarget": after.target,
        "sourceTextAndOwnersUnchanged": True, "parent": str(parent.relative_to(ROOT)),
    })
    print(OUT.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
