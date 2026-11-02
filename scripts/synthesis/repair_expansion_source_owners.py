"""Stage reviewed source-only ownership repairs without changing sampled targets."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from rebase_reviewed_locality_frames import frame_proposals, rebase_owner_frames
from rebase_reviewed_geography import validate_postal_preserving_rebase

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_numeric import _digest as numeric_digest
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign
from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate, load_independent_geography_certificates,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v17"
COUNTRY = "doc_b2b9200dc9c342c00181c08441a6448967f04a9bc9c52d2c0a87307b5faa667e"
PHONE = "doc_5242647c15b8d9847aef3c4321fb5d93df660576d1aaab7e6e4e2f261124de04"
IMPORTER = "doc_0bd3ce7b5dd84812d58987e15385efa2e95506302e90d847a9b689495b9cd009"
MARKS = "doc_d383908ff8a8d131a545d9e610fbe39c4424f566f3c77e051daa2ae45f592290"
BRAND = "doc_551f3a340047d71e39413508f5f4f189705044a7c78972479731f63db1793c65"
IMPORTER_SECOND = "doc_388e54a6628e5be102e49a12f307f489cf69ceceadd90b2e3bd65f9f824cdfdc"
LOCALITY_FRAME = "doc_551657f34bc93b4b7db213ad8cf35ac0be759fa35d32967097f1b4a766ff6525"
IMPORTER_CAPTIONS = {
    "654e017c": "EGYPTIAN IMPORTER",
    "e28ff6f7": "EGYPTIAN IMPORTER NAME:",
    "1e903c99": "EGYPTIAN IMPORTER\n",
    "f766e12f": "Egyptian Importer",
    "ef2a1959": "EGYPTIAN IMPORTER",
    "7484c804": "EGYPTIAN IMPORTER TAX NUMBER:",
    "1e9d32e5": "EGYPTIAN CONSIGNEE TAX",
}
LOCALITY_REPETITIONS = {
    "6297815a": [("MUGLA", 2, "shipper", "name")],
    "cbd4c105": [("10TH OF RAMADAN CITY", 3, "consignee", "name"),
                 ("EGYPT", 6, "consignee", "country")],
}


def publish_rebased_manifest(dataset, new_data, source_id):
    """Preserve spatial provenance while pinning the reviewed training target edit."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        if sha256_file(dataset / name) != expected:
            raise ValueError("source manifest output differs: " + name)
        if name != "train.jsonl" and sha256_file(new_data / name) != expected:
            raise ValueError("non-training output changed: " + name)
    for name, expected in manifest["alignment_sha256"].items():
        if sha256_file(new_data / "alignments" / name) != expected:
            raise ValueError("source alignment changed: " + name)
    updated = copy.deepcopy(manifest)
    updated["dataset"] = str(new_data.relative_to(ROOT))
    updated["output_sha256"]["train.jsonl"] = sha256_file(new_data / "train.jsonl")
    updated.setdefault("provenance", {})["descriptionBoundary20261101"] = {
        "sourceDocumentId": source_id,
        "parentManifest": {
            "path": str((dataset / "manifest.json").relative_to(ROOT)),
            "sha256": sha256_file(dataset / "manifest.json"),
        },
        "change": "Remove separately owned shipment mass and introductory OF from the product description target; OCR and spatial alignment unchanged.",
    }
    atomic_publish_json(new_data / "manifest.json", updated)


def rebase_description_authority(base, config, old_campaign):
    """Rebase the reviewed product-only boundary, preserving every numeric owner."""
    sid = next(s for s in base["source_ids"] if s.startswith("doc_48973902"))
    before = old_campaign.blueprint(sid)
    dataset = ROOT / base["dataset"]
    rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
    original = next(row for row in rows if row["documentId"] == sid)
    row = copy.deepcopy(original)
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    if goods["description"] != "202.236 MT OF WHITE TOP TESTLINER COATED":
        raise ValueError("reviewed description target changed")
    goods["description"] = "WHITE TOP TESTLINER COATED"
    expected_target = copy.deepcopy(original["target"])
    expected_target["documentPatch"]["goodsItemDetails"][0]["description"] = goods["description"]
    assert row["target"] == expected_target and row["joinedRawText"] == before.source
    new_data = OUT / "source-dataset"
    atomic_publish_bytes(new_data / "train.jsonl", b"".join(
        (json.dumps(row if item["documentId"] == sid else item, ensure_ascii=False) + "\n").encode()
        for item in rows
    ))
    for path in dataset.rglob("*"):
        relative = path.relative_to(dataset)
        if path.is_file() and relative not in {Path("train.jsonl"), Path("manifest.json")}:
            atomic_publish_bytes(new_data / relative, path.read_bytes())
    publish_rebased_manifest(dataset, new_data, sid)
    new_contracts = OUT / "cases"
    for source in base["source_ids"]:
        folder = ROOT / base["source_contracts"] / source
        for path in folder.iterdir():
            if not path.is_file():
                continue
            target = new_contracts / source / path.name
            if source == sid and path.name == "contract.json":
                envelope = json.loads(path.read_text())
                if envelope["targetSha256"] != digest(original["target"]):
                    raise ValueError("original source contract differs")
                variable = next(v for v in envelope["contract"]["variables"] if v["key"] == "v017")
                if variable["value"] != "OF WHITE TOP TESTLINER COATED":
                    raise ValueError("original product owner differs")
                variable["value"] = "WHITE TOP TESTLINER COATED"
                variable["occurrences"] = [{"text": "WHITE TOP TESTLINER\nCOATED", "occurrence": 1, "presentation": "text"}]
                envelope["targetSha256"] = digest(row["target"])
                atomic_publish_json(target, envelope)
            elif source == sid and path.name == "target.json":
                atomic_publish_json(target, row["target"])
            else:
                atomic_publish_bytes(target, path.read_bytes())
    config["dataset"] = str(new_data.relative_to(ROOT))
    config["source_contracts"] = str(new_contracts.relative_to(ROOT))
    after = _SourceOwnershipCampaign(ROOT, config).blueprint(sid)
    assert after.source == before.source and after.target == expected_target
    old_contract = before.contract.model_dump(mode="json")
    new_contract = after.contract.model_dump(mode="json")
    # Only the reviewed product owner may change. The private/public numeric,
    # party, identifier and route owners must be byte-for-byte equivalent.
    for contract in (old_contract, new_contract):
        contract["variables"] = [v for v in contract["variables"] if v["key"] != "v017"]
    if old_contract != new_contract:
        raise ValueError("product boundary rebase changed unrelated source owners")
    numeric_pin = base["independent"]["numeric_certificates"]
    numeric = json.loads((ROOT / numeric_pin["path"]).read_text())
    certificate = numeric["sources"][sid]
    if certificate["sourceContractSha256"] != numeric_digest(before.contract.model_dump(mode="json")) or certificate["sourcePublicTargetSha256"] != numeric_digest(before.target):
        raise ValueError("original numeric certificate differs")
    certificate["sourceContractSha256"] = numeric_digest(after.contract.model_dump(mode="json"))
    certificate["sourcePublicTargetSha256"] = numeric_digest(after.target)
    numeric_path = OUT / "numeric_certificates.json"
    atomic_publish_json(numeric_path, numeric)
    config["independent"]["numeric_certificates"] = {"path": str(numeric_path.relative_to(ROOT)), "sha256": sha256_file(numeric_path)}
    package_pin = base["declared_packages"]["contracts"]
    packages = json.loads((ROOT / package_pin["path"]).read_text())
    package = packages["sources"][sid]
    if package["sourceTargetSha256"] != digest(before.target):
        raise ValueError("original declared-package authority differs")
    # Declaration and allocation structure are identical; only the product
    # description target changed. Retain all approved presentation decisions.
    package["sourceTargetSha256"] = digest(after.target)
    packages_path = OUT / "declared-package-contracts.json"
    atomic_publish_json(packages_path, packages)
    config["declared_packages"]["contracts"] = {
        "path": str(packages_path.relative_to(ROOT)), "sha256": sha256_file(packages_path),
    }
    alias_pin = base["aliases"]["presentations"]["certificates"]
    aliases = json.loads((ROOT / alias_pin["path"]).read_text())
    alias_certificate = aliases["sources"][sid]
    if alias_certificate["sourceTargetSha256"] != digest(before.target):
        raise ValueError("original package presentation authority differs")
    alias_certificate["sourceTargetSha256"] = digest(after.target)
    alias_path = OUT / "package-presentation-certificates.json"
    atomic_publish_json(alias_path, aliases)
    config["aliases"]["presentations"]["certificates"] = {
        "path": str(alias_path.relative_to(ROOT)), "sha256": sha256_file(alias_path),
    }
    atomic_publish_json(OUT / "description-rebase.json", {
        "sourceDocumentId": sid, "beforeTarget": original["target"], "afterTarget": row["target"],
        "sourceTextUnchanged": True, "allNonProductOwnersUnchanged": True,
        "numericFactsAndOwnershipUnchanged": True,
        "requiresFreshSynthesis": True,
    })


def main():
    base = json.loads((ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json").read_text())
    ownership = yaml.safe_load((ROOT / base["ownership"]).read_text())
    auxiliary = yaml.safe_load((ROOT / base["auxiliary"]).read_text())
    changes = []
    declarations = [
        (
            COUNTRY,
            "agent:customs:exporter_registration_country",
            "UNITED STATES",
            "other_text",
            {"party_location": "documentPatch.parties.shipper", "field": "country"},
        ),
        (
            PHONE,
            "anchor:documentPatch.parties.consignee.contactDetails.phoneNumbers[1]",
            "353",
            "identifier",
            {"identifier": "Private untyped telephone suffix", "format": "source"},
        ),
        (
            PHONE,
            "anchor:documentPatch.parties.consignee.contactDetails.phoneNumbers[2]",
            "131",
            "identifier",
            {"identifier": "Private untyped telephone suffix", "format": "source"},
        ),
    ]
    for sid, key, text, kind, recipe in declarations:
        source = (ROOT / base["source_contracts"] / sid / "source.txt").read_text()
        if sid == COUNTRY:
            marker = "Exporter Registration Country: " + text
            reason = (
                "Explicit exporter registration country repeats the source's US shipper. "
                "Keep the country tied to that sampled party, not a fixed USA literal."
            )
        else:
            marker = " , " + text if text == "353" else " , 131 MOB:"
            reason = (
                "Uncaptioned three-digit telephone suffix: anonymize format-preservingly "
                "without inventing a complete phone or guessing extension versus suffix."
            )
        assert source.count(marker) == 1
        offset = source.index(marker) + marker.index(text)
        occurrence = source[:offset].count(text) + 1
        row = ownership["sources"][sid]
        # These occurrences already have historical owners. Repair those owners
        # rather than adding overlapping regions to an otherwise complete template.
        row.setdefault("bindings", {})[key] = {
            "kind": kind,
            "paths": [],
            "occurrences": [{"text": text, "occurrence": occurrence}],
        }
        if kind == "identifier":
            row.setdefault("identifier_privacy", {})[key] = "private"
        auxiliary["sources"][sid]["bindings"][key] = recipe
        changes.append(
            {
                "sourceId": sid,
                "key": key,
                "sourceText": text,
                "occurrence": occurrence,
                "reason": reason,
            }
        )
    source = (ROOT / base["source_contracts"] / IMPORTER / "source.txt").read_text()
    if source.count("EGYPT IMPORTER") != 1:
        raise ValueError("reviewed importer-caption occurrence changed")
    auxiliary["sources"][IMPORTER].setdefault("spans", []).append(
        {"quote": "EGYPT IMPORTER", "count": 1, "value": {"text": "IMPORTER"}}
    )
    changes.append(
        {
            "sourceId": IMPORTER,
            "sourceText": "EGYPT IMPORTER",
            "replacement": "IMPORTER",
            "reason": "Neutralize a filing-country caption when parties/routes are independently sampled; no party country or private identifier is inferred from this caption.",
        }
    )
    source = (ROOT / base["source_contracts"] / MARKS / "source.txt").read_text()
    marks_key = "anchor:documentPatch.cargoGroups[0].marksAndNumbers[1]"
    ownership["sources"][MARKS]["bindings"][marks_key] = {"kind": "other_text", "paths": []}
    auxiliary["sources"][MARKS]["bindings"][marks_key] = {
        "text": "{destination_port}, {destination_country}"
    }
    changes.append(
        {
            "sourceId": MARKS,
            "key": marks_key,
            "sourceText": "ALEXANDRIA, EGYPT",
            "replacement": "{destination_port}, {destination_country}",
            "reason": "Existing source-only destination-mark owner follows the sampled discharge port, not a retained literal.",
        }
    )
    # Marks retain their private prefix owner. Only the adjoining destination
    # words follow the sampled discharge port; they do not create new targets.
    spans = [
        ("-EGYPT", "-{destination_country}"),
        (
            "Reloading empty containers to remain for receiver's account at any port of discharge in Egypt",
            "Reloading empty containers to remain for receiver's account at the port of discharge",
        ),
        (
            "Carrier drew Merchant’s attention to the fact that as per Egyptian New Customs Law No. 207 for the year 2020, published in the Egyptian Official Gazette on Nov. 11, 2020, cargo shall be auctioned by Customs without any notice if Merchant fails to take delivery within 1(one) month from the date of discharge.",
            "Uncollected cargo remains subject to applicable destination customs procedures.",
        ),
        (
            "Preliminary Customs registration number for shipment bound to Egypt as declared by shipper.",
            "Preliminary customs registration reference as declared by shipper.",
        ),
    ]
    for quote, replacement in spans:
        if source.count(quote) != 1:
            raise ValueError("reviewed destination-context occurrence changed: " + quote)
        auxiliary["sources"][MARKS].setdefault("spans", []).append(
            {"quote": quote, "count": 1, "value": {"text": replacement}}
        )
        changes.append(
            {
                "sourceId": MARKS,
                "sourceText": quote,
                "replacement": replacement,
                "reason": "Keep destination marks and source-only customs context coherent with independent route sampling, without adding extraction targets or altering the private marks-prefix owner.",
            }
        )
    source = (ROOT / base["source_contracts"] / BRAND / "source.txt").read_text()
    quote = 'BRAND "HEG" INDIA'
    if source.count(quote) != 1:
        raise ValueError("reviewed linked brand occurrence changed")
    brand_key = "agent:shipper:brand_alias"
    offset = source.index(quote) + len('BRAND "')
    # The reduced V7 compiler retires the historical additionalInformation
    # parent. Add ownership of the surviving literal, not a correction to that
    # retired historical key.
    ownership["sources"][BRAND].setdefault("add", []).append({
        "key": brand_key, "kind": "organization", "paths": [],
        "occurrences": [{"text": "HEG", "occurrence": source[:offset].count("HEG") + 1}],
    })
    auxiliary["sources"][BRAND]["bindings"][brand_key] = {
        "target": "documentPatch.parties.shipper.name"
    }
    # closed_goods_origin already owns INDIA. Preserve that certified owner.
    changes.append({
        "sourceId": BRAND, "key": brand_key, "sourceText": quote,
        "replacement": "BRAND <sampled shipper name> <sampled origin country>",
        "reason": "The brand occurrence repeats the source shipper HEG LIMITED. Reuse the already generated shipper identity rather than retaining that private source-party alias after the shipper changes. Keep the administrative brand/origin line outside the existing product-description target.",
    })
    source = (ROOT / base["source_contracts"] / IMPORTER_SECOND / "source.txt").read_text()
    ownership["sources"][IMPORTER_SECOND].setdefault("retire", {})[
        "agent:customs:importer_country"
    ] = "Complete reviewed neutral importer caption owns this source-only country occurrence."
    for quote, replacement in [
        ("EGYPT IMPORTER TAX ID:", "IMPORTER TAX ID:"),
        (
            "Reloading empty containers to remain for receiver’s account at any port of discharge in Egypt.",
            "Reloading empty containers to remain for receiver’s account at the port of discharge.",
        ),
        (
            "Off loading of containers full from truck and reloading of empty onto truck at Port Said to be at Receiver’s risk and expenses.",
            "Off loading of containers full from truck and reloading of empty onto truck at the discharge terminal to be at Receiver’s risk and expenses.",
        ),
        (
            "Carrier draw Merchant’s attention to the fact that as per Egyptian New Customs Law No. 207 for the year 2020, published in the Egyptian Official Gazette on Nov. 11, 2020, cargo shall be auctioned by Customs without any notice if Merchant fails to take delivery within 1(one) month from the date of discharge.",
            "Uncollected cargo remains subject to applicable destination customs procedures.",
        ),
        (
            "Preliminary Customs registration number for shipment bound to Egypt as declared by shipper.",
            "Preliminary customs registration reference as declared by shipper.",
        ),
    ]:
        if source.count(quote) != 1:
            raise ValueError("reviewed importer/destination context changed: " + quote)
        auxiliary["sources"][IMPORTER_SECOND].setdefault("spans", []).append(
            {"quote": quote, "count": 1, "value": {"text": replacement}}
        )
        changes.append({
            "sourceId": IMPORTER_SECOND, "sourceText": quote, "replacement": replacement,
            "reason": "Use the same source-only destination-neutral treatment as the reviewed d383908f layout. Keep the sampled importer identifier owner and all extraction targets unchanged; no claim about real customs rules is introduced.",
        })
    for prefix, quote in IMPORTER_CAPTIONS.items():
        matches = [s for s in auxiliary["sources"] if s.startswith("doc_" + prefix)]
        if len(matches) != 1:
            raise ValueError("reviewed importer caption source is not unique")
        sid = matches[0]
        source = (ROOT / base["source_contracts"] / sid / "source.txt").read_text()
        if source.count(quote) != 1:
            raise ValueError("reviewed importer caption occurrence changed: " + sid)
        replacement = quote.split(" ", 1)[1]
        auxiliary["sources"][sid].setdefault("spans", []).append(
            {"quote": quote, "count": 1, "value": {"text": replacement}}
        )
        changes.append({
            "sourceId": sid, "sourceText": quote, "replacement": replacement,
            "reason": "The printed importer is already bound to the newly sampled consignee/name or identifier. Neutralize only the fixed nationality caption; keep all names, identifiers and public targets under their existing owners.",
        })
    for prefix, repetitions in LOCALITY_REPETITIONS.items():
        sid = next(s for s in base["source_ids"] if s.startswith("doc_" + prefix))
        source = (ROOT / base["source_contracts"] / sid / "source.txt").read_text()
        if prefix == "cbd4c105":
            ownership["sources"][sid].setdefault("retire", {})[
                "anchor:documentPatch.cargoGroups[0].marksAndNumbers[1]"
            ] = "Replace the complete fixed Marks locality with separately owned sampled consignee city/country; preserve the literal hyphen and public target absence."
        for text, occurrence, party, field in repetitions:
            if source.count(text) < occurrence:
                raise ValueError("reviewed repeated locality occurrence changed")
            key = f"reviewed:source_only:{party}:{field}"
            ownership["sources"][sid].setdefault("add", []).append({
                "key": key, "kind": "other_text", "paths": [],
                "occurrences": [{"text": text, "occurrence": occurrence}],
            })
            auxiliary["sources"][sid]["bindings"][key] = {
                "party_location": f"documentPatch.parties.{party}", "field": field,
            }
            changes.append({
                "sourceId": sid, "key": key, "sourceText": text,
                "occurrence": occurrence, "party": party, "field": field,
                "reason": "Source-only tax-office/Marks locality repeats the original named party locality; sample from that same party without adding target fields.",
            })
    sid = next(s for s in base["source_ids"] if s.startswith("doc_9b35d0b6"))
    source = (ROOT / base["source_contracts"] / sid / "source.txt").read_text()
    quote = "PORT SAID FREEZONE"
    if source.count(quote) != 1:
        raise ValueError("reviewed free-zone destination occurrence changed")
    auxiliary["sources"][sid].setdefault("spans", []).append({
        "quote": quote, "count": 1, "value": {"text": "{destination_port} FREEZONE"},
    })
    changes.append({
        "sourceId": sid, "sourceText": quote, "replacement": "{destination_port} FREEZONE",
        "reason": "Reuse the established destination free-zone presentation mechanism for the separately printed discharge-destination repetition. No new goods/route target.",
    })
    sid = next(s for s in base["source_ids"] if s.startswith("doc_600a1e89"))
    source = (ROOT / base["source_contracts"] / sid / "source.txt").read_text()
    if source.count(",EGYI") != 4:
        raise ValueError("reviewed clipped-country occurrences changed")
    ownership["sources"][sid].setdefault("retire", {})[
        "anchor:documentPatch.route.portOfDischarge.country"
    ] = "The two unrecognized source-only EGYI route suffixes are removed by reviewed exact context spans. No country target exists. The separately generated postal owner is unchanged."
    for suffix in ["\nFOR TRANSSHIPMENT TO (13)", "\n\nPARTICULARS FURNISHED BY SHIPPER"]:
        quote = ",EGYI" + suffix
        if source.count(quote) != 1:
            raise ValueError("reviewed route-only clipped-country context changed")
        auxiliary["sources"][sid].setdefault("spans", []).append({
            "quote": quote, "count": 1, "value": {"text": suffix},
        })
    changes.append({
        "sourceId": sid, "sourceText": ",EGYI", "occurrences": 2, "replacement": "",
        "reason": "PDF readback shows clipped Egypt at the end of the repeated locality boxes. The two postal occurrences already belong to the generated postal owner and do not survive in its descendants. Remove only the two unrecognized route-only suffixes while retaining their exact following headings. Keep absent country targets. Original OCR, original PDF and real targets remain unchanged.",
        "pdf": "data/corpora/paddle-bl-swb-filtered-max5/files/958fc87deb09b08367f75d5a37c0162d8e3ab40532da6f4f983f8cffabbfecb4.pdf",
    })
    sid = next(s for s in base["source_ids"] if s.startswith("doc_48973902"))
    source = (ROOT / base["source_contracts"] / sid / "source.txt").read_text()
    quote = "ALEXANDRIA"
    if source.count(quote) != 1 or "EGYPT HOLLAND\nALEXANDRIA" not in source:
        raise ValueError("reviewed Marks discharge-locality occurrence changed")
    auxiliary["sources"][sid].setdefault("spans", []).append({
        "quote": quote, "count": 1, "value": {"text": "{destination_port}"},
    })
    changes.append({
        "sourceId": sid, "sourceText": quote, "replacement": "{destination_port}",
        "reason": "The shipping mark repeats the source discharge port ALEXANDRIA, not the consignee city GAMASA. Bind this source-only repetition to the sampled discharge port while retaining the already anonymized consignee mark and private reference owners.",
    })
    field = ownership["sources"][sid]["description_blocks"]["fields"][0]
    expected = [{"text": "202.236 MT\nOF WHITE TOP TESTLINER\nCOATED", "occurrence": 1}]
    if field["spans"] != expected:
        raise ValueError("reviewed accounting/product boundary changed")
    field["spans"] = [{"text": "WHITE TOP TESTLINER\nCOATED", "occurrence": 1}]
    changes.append({
        "sourceId": sid,
        "sourceText": expected[0]["text"],
        "replacement": "WHITE TOP TESTLINER\nCOATED",
        "reason": "The preceding 161 REELS / 202.236 MT is the whole-shipment accounting summary, already owned as quantity/gross mass. The product-description projection must begin with the commercial product, not duplicate that accounting mass. Source OCR and numeric owners are unchanged; this is a description-target boundary correction.",
    })
    # Both MANILA, PHILIPPINES occurrences are complete inline route pairs.
    # The existing renderer already projects SINGAPORE, SINGAPORE to one token;
    # it needs ownership of the comma as well as the two scalar values.
    campaign = _SourceOwnershipCampaign(ROOT, base)
    frames = frame_proposals(campaign.blueprint(LOCALITY_FRAME))
    if len(frames) != 1 or frames[0]["path"] != "documentPatch.route.placeOfReceipt.country":
        raise ValueError("reviewed route-frame scope differs")
    if frames[0]["occurrences"] != [
        {"text": "MANILA, PHILIPPINES", "occurrence": 1},
        {"text": "MANILA, PHILIPPINES", "occurrence": 2},
    ]:
        raise ValueError("both reviewed route-frame occurrences must be owned")
    ownership["sources"][LOCALITY_FRAME] = rebase_owner_frames(
        ownership["sources"][LOCALITY_FRAME], frames
    )
    changes.append({
        "sourceId": LOCALITY_FRAME, "frames": frames,
        "reason": "Bind each complete repeated receipt-locality/country pair, including its separator, so existing one-token/one-label homonym projection can operate. Source bytes and source targets are unchanged.",
    })
    atomic_publish_json(OUT / "ownership.json", ownership)
    atomic_publish_json(OUT / "auxiliary.json", auxiliary)
    geographic_pin = base["independent"]["geography_certificates"]
    geographic_path = ROOT / geographic_pin["path"]
    geography = json.loads(geographic_path.read_bytes())
    gate = IndependentGeographyGate(load_independent_geography_certificates(
        geographic_path.read_bytes(), expected_sha256=geographic_pin["sha256"]
    ))
    changed_config = copy.deepcopy(base)
    changed_config["ownership"] = str((OUT / "ownership.json").relative_to(ROOT))
    changed_config["auxiliary"] = str((OUT / "auxiliary.json").relative_to(ROOT))
    rebase_description_authority(base, changed_config, campaign)
    updated_campaign = _SourceOwnershipCampaign(ROOT, changed_config)
    before_bp = campaign.blueprint(LOCALITY_FRAME)
    after_bp = updated_campaign.blueprint(LOCALITY_FRAME)
    pins = validate_postal_preserving_rebase(gate, before_bp, after_bp)
    geography["sources"][LOCALITY_FRAME].update(
        effective_contract_sha256=pins[0], ownership_sha256=pins[1]
    )
    atomic_publish_json(OUT / "geography_certificates.json", geography)
    changed_config["independent"]["geography_certificates"] = {
        "path": str((OUT / "geography_certificates.json").relative_to(ROOT)),
        "sha256": sha256_file(OUT / "geography_certificates.json"),
    }
    atomic_publish_json(OUT / "complete-source-authority-config.json", changed_config)
    atomic_publish_json(OUT / "geography-rebase.json", {
        "sourceDocumentId": LOCALITY_FRAME,
        "previousCatalog": geographic_pin,
        "exactOriginalCertificateVerified": True,
        "protectedPostalStateUnchanged": True,
        "sourceAndPublicTargetUnchanged": True,
        "newPins": pins,
    })
    atomic_publish_json(
        OUT / "review.json",
        {
            "status": "REVIEWED_REQUIRES_REPLAY_AND_SYNTHESIS",
            "changes": changes,
            "previous": {
                k: {"path": base[k], "sha256": sha256_file(ROOT / base[k])}
                for k in ("ownership", "auxiliary")
            },
            "targets": "Only 48973902 description boundary changes: exclude the separately owned shipment-mass prefix. Other targets unchanged. No guessed telephone expansion; exporter country is source-only.",
        },
    )
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
