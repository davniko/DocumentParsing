"""Renew acceptance after the bounded contact-order and ownership repairs.

Retained cargo controls are preserved. New controls replace evidence only for
the three contact layouts and the complete locality-frame repair. Certificates
for frozen refresh inputs explicitly retain those actual generation authorities.
"""

import copy
import json
from pathlib import Path

import yaml

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.declared_package_acceptance import authority_pins

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-multirow-refresh1500-20261031"
RUN = ROOT / "artifacts/kie-synthesis-production/r28-multirow-refresh1500-20261031"
REG = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
REV = REG / "declared-packages-refresh-final-20261031"


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    path = path.resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256_file(path))


def run():
    config_path = ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json"
    previous = read(config_path)
    old = read(REG / "declared-packages-refresh-20261031/acceptance.json")
    current = copy.deepcopy(previous)
    current["ownership"] = str((REG / "ownership-refresh-final-20261031.yaml").relative_to(ROOT))
    current["independent"]["geography_certificates"] = pin(REV / "geography_certificates.json")
    owners = [
        yaml.load((ROOT / p).read_text(), Loader=yaml.CSafeLoader)
        for p in (previous["ownership"], current["ownership"])
    ]
    changed = {
        sid
        for sid in owners[0]["sources"]
        if owners[0]["sources"][sid] != owners[1]["sources"][sid]
    }
    if {s[4:12] for s in changed} != {"04f208b1", "0ba3ad4f"}:
        raise ValueError("unexpected ownership changes")
    impact = read(OUT / "contact-order-impact.json")
    if impact["controlsInspected"] != 1368 or len(impact["changed"]) != 7:
        raise ValueError("contact impact proof incomplete")
    if not read(RUN / "contact-owner-repair/geography-rebase.json")[
        "sourceTargetAndPostalStateUnchanged"
    ]:
        raise ValueError("postal-preserving proof missing")
    fresh, fresh_contracts = {}, {}
    directories = [
        RUN / s
        for s in (
            "contact-owner-repair",
            "contact-owner-repair/0ba-controls",
            "locality-frame-repair/control-redraw",
        )
    ]
    for directory in directories:
        summary = read(sorted((directory / "stage-runs").glob("*/summary.json"))[-1])
        campaign = create_campaign(ROOT, read(ROOT / summary["preparedConfig"]))
        if any(
            summary["stages"][k].get("errors") for k in ("generate", "postal", "render", "review")
        ):
            raise ValueError("fresh control errors")
        for result in summary["stages"]["geometry"]["rows"]:
            if result["status"] != "PASS":
                raise ValueError("fresh control geometry held")
            path = ROOT / result["candidate"]
            candidate = read(path)
            sid = candidate["sourceDocumentId"]
            fresh_contracts[sid] = digest(campaign.declared_package_contracts[sid])
            review_path = directory / "campaign/reviews" / (sid + ".json")
            review = read(review_path)
            if (
                review["candidateHashes"][candidate["documentId"]] != digest(candidate)
                or review["review"]["findings"]
            ):
                raise ValueError("fresh native review not clear/current")
            if candidate["variant"] > 1:
                scoped = read(
                    RUN
                    / "scoped-wording-audit/reviews"
                    / (candidate["documentId"] + "-" + sha256_file(path) + ".json")
                )
                if scoped["candidate"] != pin(path) or scoped["review"]["findings"]:
                    raise ValueError("fresh packing review not clear/current")
            fresh.setdefault(sid, []).append(
                dict(
                    sampleId=candidate["documentId"],
                    groupCount=candidate["variant"],
                    candidate=pin(path),
                    positions=pin(ROOT / result["positions"]),
                    review=pin(review_path),
                )
            )
    if len(fresh) != 4 or any(
        sorted(v["groupCount"] for v in values) != [1, 2, 3] for values in fresh.values()
    ):
        raise ValueError("fresh single/two/three-group controls incomplete")
    adjudication = REV / "main-adjudication.json"
    atomic_publish_json(
        adjudication,
        dict(
            status="PASS",
            freshControls=fresh,
            contactImpact=pin(OUT / "contact-order-impact.json"),
            geographyProof=pin(RUN / "contact-owner-repair/geography-rebase.json"),
            localityProof=pin(RUN / "locality-frame-repair/repair.json"),
            review=(
                "Main review of rendered name/email order and locality frames; native complete "
                "review, exact replay, packing review and geometry. Cargo-only ownership outside "
                "these repairs unchanged."
            ),
        ),
    )
    evidence = copy.deepcopy(old)
    evidence["contactAndFrameRepair"] = pin(adjudication)
    evidence["implementation"] = [
        {**p, "sha256": sha256_file(ROOT / p["path"])} for p in old["implementation"]
    ]
    differences = {
        a["path"]
        for a, b in zip(old["implementation"], evidence["implementation"], strict=True)
        if a != b
    }
    if differences != {"src/document_ocr/synthesis/curated_contacts.py"}:
        raise ValueError(f"unexpected code change: {differences}")
    for sid, samples in fresh.items():
        entry = evidence["sources"][sid]
        if entry["contractSha256"] != fresh_contracts[sid]:
            raise ValueError("noncargo ownership repair altered declared-package contract")
        entry["samples"] = [{**s, "adjudication": pin(adjudication)} for s in samples]
    evidence["authorities"] = authority_pins(ROOT, current)
    atomic_publish_json(REV / "acceptance.json", evidence)
    current["declared_packages"]["acceptance"] = pin(REV / "acceptance.json")
    index = read(OUT / "final-components.json")
    accepted = {}
    for batch, row in index.items():
        configs = []
        for filename in row["components"]:
            config = read(ROOT / filename)
            authorities = authority_pins(ROOT, config)
            identity = digest(authorities)[:16]
            path = REV / f"frozen-authority-{identity}.json"
            certificate = copy.deepcopy(evidence)
            certificate["authorities"] = authorities
            actual = yaml.load((ROOT / config["ownership"]).read_text(), Loader=yaml.CSafeLoader)
            for sid in list(certificate["sources"]):
                if actual["sources"].get(sid) != owners[1]["sources"][sid]:
                    del certificate["sources"][sid]
            if not set(config["source_ids"]) <= certificate["sources"].keys():
                raise ValueError("selected refresh source uses obsolete repaired ownership")
            atomic_publish_json(path, certificate)
            config["declared_packages"]["acceptance"] = pin(path)
            dest = (ROOT / filename).with_name("admission-config.json")
            atomic_publish_json(dest, config)
            configs.append(str(dest.relative_to(ROOT)))
        accepted[batch] = {**row, "components": configs}
    atomic_publish_json(OUT / "admission-components.json", accepted)
    atomic_publish_json(REV / "previous-production-config.json", previous)
    atomic_write_json(config_path, current)
    manifest_path = REG / "manifest.json"
    manifest = read(manifest_path)
    manifest["currentProductionConfig"] = pin(config_path)
    manifest["multirowRefreshContactAndFrameClosure20261031"] = dict(
        acceptance=pin(REV / "acceptance.json"),
        adjudication=pin(adjudication),
        sources=len(evidence["sources"]),
        freshControls=12,
    )
    for sid in manifest["cases"]:
        if sid.startswith("doc_c08ce779"):
            manifest["cases"][sid]["files"]["contract.json"] = sha256_file(
                REG / "cases" / sid / "contract.json"
            )
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(dict(status="CERTIFIED", sources=len(evidence["sources"]), freshControls=12)))


if __name__ == "__main__":
    run()
