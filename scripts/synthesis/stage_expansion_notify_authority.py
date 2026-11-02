"""Repair a reviewed shared notify-contact footnote without changing its OCR."""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_private_contacts import validate_private_person_coverage
from document_ocr.synthesis.independent_geography import IndependentGeographyGate, load_independent_geography_certificates
from document_ocr.synthesis.independent_numeric import _digest
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
OUT = REGISTRY / "expansion-20261101-v21"
SOURCE = "doc_ab905fd40d2afadc45840dc00b569e510bfd6a1017218d16c2ce125ea38288a7"
PREFIX = "documentPatch.parties.notifyParties[0].contactDetails."


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def main():
    parent = REGISTRY / "expansion-20261101-v20/complete-source-authority-config.json"
    base = read(parent)
    config = copy.deepcopy(base)
    before = _SourceOwnershipCampaign(ROOT, base).blueprint(SOURCE)
    if before.source.count("ATTENTION.: OMAR DAWABA**") != 2:
        raise ValueError("reviewed shared footnote anchors changed")
    if "**EMAIL ADDRESS.: OMAR@ROCHEM.COM.EG\nPHONE NUMBER.: 203 582 2141" not in before.source:
        raise ValueError("reviewed footnote changed")
    target = copy.deepcopy(before.target)
    contact = target["documentPatch"]["parties"]["notifyParties"][0]["contactDetails"]
    if contact != {"contactName": "OMAR DAWABA"}:
        raise ValueError("reviewed original notify contact differs")
    contact.update(emailAddresses=["OMAR@ROCHEM.COM.EG"], phoneNumbers=["203 582 2141"])
    dataset, new_data = ROOT / base["dataset"], OUT / "source-dataset"
    rows = [json.loads(line) for line in (dataset / "train.jsonl").open()]
    selected = [r for r in rows if r["documentId"] == SOURCE]
    if len(selected) != 1 or selected[0]["target"] != before.target:
        raise ValueError("source target authorities differ")
    selected[0]["target"] = target
    atomic_publish_bytes(new_data / "train.jsonl", b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows))
    for p in dataset.rglob("*"):
        rel = p.relative_to(dataset)
        if p.is_file() and rel not in {Path("train.jsonl"), Path("manifest.json")}:
            atomic_publish_bytes(new_data / rel, p.read_bytes())
    manifest = read(dataset / "manifest.json")
    for name, expected in manifest["output_sha256"].items():
        if sha256_file(dataset / name) != expected or (name != "train.jsonl" and sha256_file(new_data / name) != expected):
            raise ValueError("unrelated source dataset output changed")
    manifest["dataset"] = str(new_data.relative_to(ROOT))
    manifest["output_sha256"]["train.jsonl"] = sha256_file(new_data / "train.jsonl")
    manifest.setdefault("provenance", {})["notifyFootnote20261101"] = {
        "sourceDocumentId": SOURCE, "change": "Extract the explicitly shared ** footnote for notify only. Consignee goods-area contact exclusion remains unchanged; raw text and coordinates unchanged."}
    atomic_publish_json(new_data / "manifest.json", manifest)
    cases = OUT / "cases"
    for sid in base["source_ids"]:
        for p in (ROOT / base["source_contracts"] / sid).iterdir():
            if not p.is_file():
                continue
            dest = cases / sid / p.name
            if sid == SOURCE and p.name == "target.json":
                atomic_publish_json(dest, target)
            elif sid == SOURCE and p.name == "contract.json":
                value = read(p)
                if value["targetSha256"] != digest(before.target):
                    raise ValueError("source contract pin differs")
                value["targetSha256"] = digest(target)
                atomic_publish_json(dest, value)
            else:
                atomic_publish_bytes(dest, p.read_bytes())
    owners, aux = read(ROOT / base["ownership"]), read(ROOT / base["auxiliary"])
    owner = owners["sources"][SOURCE]
    for key, field in [("source_personal_email", "emailAddresses[0]"), ("consignee_footer_phone", "phoneNumbers[0]")]:
        if key in owner["bindings"]:
            raise ValueError("reviewed new binding unexpectedly exists")
        owner["bindings"][key] = {"paths": [PREFIX + field]}
    private = owner["private_contacts"]
    del private["emails"]["source_personal_email"]
    private["email_paths"] = [PREFIX + "emailAddresses[0]"]
    private["replace_domain_paths"] = [PREFIX + "emailAddresses[0]"]
    del aux["sources"][SOURCE]["bindings"]["consignee_footer_phone"]
    atomic_publish_json(OUT / "ownership.json", owners)
    atomic_publish_json(OUT / "auxiliary.json", aux)
    config.update(dataset=str(new_data.relative_to(ROOT)), source_contracts=str(cases.relative_to(ROOT)),
                  ownership=str((OUT / "ownership.json").relative_to(ROOT)), auxiliary=str((OUT / "auxiliary.json").relative_to(ROOT)))
    after = _SourceOwnershipCampaign(ROOT, config).blueprint(SOURCE)
    validate_private_person_coverage(after)
    if after.source != before.source or after.target != target or after.contract != before.contract:
        raise ValueError("notify repair changed another source authority")
    for key, field, value in [("source_personal_email", "emailAddresses[0]", "OMAR@ROCHEM.COM.EG"), ("consignee_footer_phone", "phoneNumbers[0]", "203 582 2141")]:
        regions = [r for r in after.regions if r.key == key]
        if len(regions) != 1 or regions[0].source != value or tuple(regions[0].target_paths) != (PREFIX + field,):
            raise ValueError("notify occurrence was not bound exactly")
    for name, section in [("numeric", base["independent"]["numeric_certificates"]), ("groups", base["declared_packages"]["contracts"]), ("aliases", base["aliases"]["presentations"]["certificates"])]:
        catalog = read(ROOT / section["path"])
        entry = catalog["sources"][SOURCE]
        field, hash_fn = ("sourcePublicTargetSha256", _digest) if name == "numeric" else ("sourceTargetSha256", digest)
        if entry[field] != hash_fn(before.target):
            raise ValueError("old target authority changed")
        entry[field] = hash_fn(after.target)
        dest = OUT / (name + "-certificates.json")
        atomic_publish_json(dest, catalog)
        if name == "numeric": config["independent"]["numeric_certificates"] = pin(dest)
        elif name == "groups": config["declared_packages"]["contracts"] = pin(dest)
        else: config["aliases"]["presentations"]["certificates"] = pin(dest)
    gp = base["independent"]["geography_certificates"]
    raw = (ROOT / gp["path"]).read_bytes()
    gate = IndependentGeographyGate(load_independent_geography_certificates(raw, expected_sha256=gp["sha256"]))
    certificate = gate._certificate(before, allow_prepared=False)
    if certificate is not None:
        if gate._postal_state(before, certificate) != gate._postal_state(after, certificate):
            raise ValueError("postal state changed")
        catalog = json.loads(raw)
        pins = gate._blueprint_pins(after)
        catalog["sources"][SOURCE].update(effective_contract_sha256=pins[0], ownership_sha256=pins[1])
        dest = OUT / "geography_certificates.json"
        atomic_publish_json(dest, catalog)
        config["independent"]["geography_certificates"] = pin(dest)
    atomic_publish_json(OUT / "complete-source-authority-config.json", config)
    atomic_publish_json(OUT / "review.json", {"status": "REQUIRES_FRESH_SYNTHESIS", "sourceDocumentId": SOURCE,
        "parent": pin(parent), "beforeTarget": before.target, "afterTarget": after.target,
        "sourceTextUnchanged": True, "consigneeTargetUnchanged": True, "policy": "Shared notify-role footnote contacts are permitted; direct-block restriction is consignee-specific."})
    print(OUT.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
