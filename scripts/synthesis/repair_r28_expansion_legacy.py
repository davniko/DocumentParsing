"""Stage/apply reviewed, occurrence-pinned legacy owner corrections in R28 only.

This is a bounded migration, not a production inference heuristic. It never
reconstructs records from original private sources or changes shipment facts.
"""

import argparse
import copy
import json
import re
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json, atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_reflow_policy import FontMetrics
from document_ocr.synthesis.package_alias_refresh import apply_reviewed_cleanup

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101/legacy-repair"
AUTH = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v17"
FILES = ("synthetic-train.jsonl", "real-train.jsonl", "previous-uae-synthetic.jsonl", "old-uae-addition.jsonl")
BRANDS = {
    "HORLOGERIE BOIS-FRANCS LTEE": "HBF", "MIQUELON CHIMIE INDUSTRIE SARL": "MCI",
    "JUBBA TANNIN EXPORTERS CO.": "JTE", "SELVA ANDINA PULP S.A.C.": "SAP",
    "HORTO TORRE LLOBETA S.A.": "HTL", "AVK LIMITED": "AVK",
}
MARK_CITIES = {
    "syn_full_v7_2b440fce2176d484aaf84aa3": "ABU HAYL",
    "syn_full_v7_1142d289b4a2c0e4b7587e37": "AL AIN CITY",
    "syn_full_v7_26589a63ccc42940650fa3ca": "ADH DHAYD",
    "syn_full_v7_8b2eaa6498615f2abc0cdb63": "JUMAYRA",
    "syn_full_v7_e6bbc7c881219bc05e009a4a": "REEF AL FUJAIRAH CITY",
}


def read(p):
    return json.loads(p.read_text())


def normalize(value):
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def stage():
    started = time.perf_counter()
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    font = cfg["positions"]["reflow"]
    metrics = FontMetrics(Path(font["font_path"]), font["font_sha256"])
    preferred = read(ROOT / "artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/preferred-names.json")
    countries = {normalize(v): k for k, v in preferred["names"].items()}
    ports = {}
    with (ROOT / cfg["sampling"]["locations"]["path"]).open() as stream:
        for line in stream:
            item = json.loads(line)
            for value in (item["name"], item["name_without_diacritics"]):
                ports.setdefault(normalize(value), set()).add(item["locode"])
    revisions = read(AUTH / "review.json")["changes"]
    literal = {}
    for change in revisions:
        before, after = change.get("sourceText"), change.get("replacement")
        if before and after is not None and "{" not in after and before != after and (
            "IMPORTER" in before.upper() or "CONSIGNEE TAX" in before.upper()
            or "Reloading empty containers" in before
            or "Carrier draw" in before or "Preliminary Customs" in before
        ):
            literal.setdefault(change["sourceId"][4:12], []).append((before, after, change["reason"]))
    records, copies, hashes = {}, Counter(), {}
    for name in FILES:
        path = DATA / name
        hashes[name] = sha256_file(path)
        for line in path.open():
            row = json.loads(line)
            sid = row["documentId"]
            if sid in records and records[sid] != row:
                raise ValueError("mirrored row differs: " + sid)
            records[sid] = row
            copies[sid] += 1
    repairs, holds = [], []
    for sid, row in records.items():
        prefix = row.get("sourceDocumentId", sid)[4:12]
        raw, target = row["joinedRawText"], row["target"]
        edits, targets, reasons = [], [], []
        synthetic = sid.startswith("syn_")

        def add(before, after, reason, *, all_occurrences=False):
            if before == after or before not in raw:
                return
            found = list(re.finditer(re.escape(before), raw))
            if len(found) != 1 and not all_occurrences:
                raise ValueError("reviewed owner not unique: " + sid + "/" + before)
            if "\n" in before or "\n" in after:
                holds.append({"id": sid, "before": before, "after": after, "reason": "multiline_requires_individual_geometry"})
                return
            if metrics.length(after) > metrics.length(before) + 1e-7:
                holds.append({"id": sid, "before": before, "after": after, "reason": "longer_glyph_advance"})
                return
            for i in range(len(found)):
                edits.append({"before": before, "after": after, "occurrence": i + 1,
                              "beforeAdvance": metrics.length(before), "afterAdvance": metrics.length(after)})
            reasons.append(reason)

        def port_alias(before, suffix=""):
            port = target["route"]["portOfDischarge"]["name"]
            full = port + suffix
            if metrics.length(full) <= metrics.length(before):
                return full
            matches = ports.get(normalize(port), set())
            declared_country = target["route"]["portOfDischarge"].get("country")
            if declared_country:
                code = countries[normalize(declared_country)]
                matches = {v for v in matches if v.startswith(code)}
            if len(matches) != 1:
                raise ValueError("port needs manual alias review: " + sid + "/" + port + str(matches))
            # Explicit reviewed compact presentation, not a fabricated location:
            # UN/LOCODE identifies this very same declared discharge port.
            return next(iter(matches))

        if prefix == "551f3a34" and '"HEG"' in raw:
            name = target["parties"]["shipper"]["name"]
            add('"HEG"', '"' + BRANDS[name] + '"', "Replace the residual private shipper brand with a reviewed three-letter alias of the already anonymized shipper.")
        if synthetic and prefix == "5242647c":
            for number in ("353", "131"):
                before = " , " + number
                if before in raw:
                    value = int(digest([sid, "opaque-telephone-suffix", number])[:12], 16) % 1000
                    if value == int(number):
                        value = (value + 1) % 1000
                    add(before, " , " + f"{value:03d}", "Anonymize the untyped three-digit phone suffix; complete phone targets remain unchanged.", all_occurrences=True)
        if prefix == "48973902":
            description = target["goodsItemDetails"][0]["description"]
            match = re.match(r"^[0-9][0-9.,]*\s+MT\s+(?:OF\s+)?", description)
            if match:
                targets.append({"path": "goodsItemDetails[0].description", "before": match[0], "after": ""})
                reasons.append("Exclude independently declared shipment mass and OF introduction from the product-description target, without changing OCR or mass labels.")
            if synthetic and "ALEXANDRIA" in raw:
                add("ALEXANDRIA", port_alias("ALEXANDRIA"), "Bind the shipping-mark port repetition to the actual discharge port; use a registry UN/LOCODE only when the full name would exceed the existing footprint.")
        if synthetic:
            for before, after, reason in literal.get(prefix, []):
                add(before.rstrip("\n"), after.rstrip("\n"), reason, all_occurrences=True)
            if prefix == "600a1e89":
                add(",EGYI", "", "Remove only the clipped source-only country residue; preserve absent public country labels.", all_occurrences=True)
            if prefix == "9b35d0b6":
                add("PORT SAID FREEZONE", "DESTINATION ZONE", "Make the non-target free-zone context destination-relative rather than retain an unrelated source port.")
            if prefix == "d383908f":
                add("ALEXANDRIA, EGYPT", port_alias("ALEXANDRIA, EGYPT", ", UAE"), "Correct the source-only destination shipping mark; compact registry code is used only if needed for the existing footprint.")
                add("-EGYPT", "-UAE", "Correct the country repetition in destination Marks, preserving its private prefix.")
            if prefix == "b2b9200d" and "Exporter Registration Country: UNITED STATES" in raw:
                code = countries[normalize(target["parties"]["shipper"]["country"])]
                add("Exporter Registration Country: UNITED STATES", "Exporter Registration Country: " + code,
                    "Replace the source exporter's retained USA registration-country repetition with an ISO presentation of the current sampled shipper country. This is not applied to the distinct 74860ac9 care-of case.")
            if prefix == "cbd4c105" and "10TH OF RAMADAN CITY - EGYPT" in raw:
                city = MARK_CITIES.get(sid)
                if city is None:
                    holds.append({"id": sid, "reason": "reviewed_city_mapping_required"})
                else:
                    address = target["parties"]["consignee"]["addressLine"]
                    if city not in address.upper():
                        raise ValueError("reviewed Marks city absent from current address: " + sid)
                    add("10TH OF RAMADAN CITY - EGYPT", city + " - UAE", "Reuse the reviewed consignee locality/country in the source-only Marks repetition.")
        if not edits and not targets:
            continue
        repair = {"documentId": sid, "sourceDocumentId": row.get("sourceDocumentId", sid),
                  "beforeRecordSha256": digest(row), "rawEdits": edits, "targetEdits": targets,
                  "reasons": list(dict.fromkeys(reasons)), "copies": copies[sid]}
        updated, proof = apply_reviewed_cleanup(row, repair)
        if len(updated["joinedRawText"].splitlines()) != len(raw.splitlines()):
            raise ValueError("line count changed")
        if [s.rsplit(" ||", 1)[-1] for s in updated["positionedText"].splitlines()] != [s.rsplit(" ||", 1)[-1] for s in row["positionedText"].splitlines()]:
            raise ValueError("coordinate suffix changed")
        repair["expectedAfterSha256"] = digest(updated)
        repair["proof"] = proof
        repairs.append(repair)
    result = {"status": "STAGED_REQUIRES_MAIN_REVIEW_NOT_APPLIED", "files": hashes,
              "repairs": repairs, "holds": holds, "seconds": time.perf_counter() - started,
              "counts": dict(Counter(r["sourceDocumentId"][4:12] for r in repairs))}
    atomic_publish_json(OUT / "plan.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"repairs", "files"}}))


def apply():
    plan = read(OUT / "plan.json")
    approval = read(OUT / "main-review.json")
    if approval["planSha256"] != sha256_file(OUT / "plan.json") or approval["status"] != "APPROVED":
        raise ValueError("explicit pinned main approval required")
    if plan["holds"]:
        raise ValueError("resolve every staged geometry/owner hold before applying")
    changes = {r["documentId"]: r for r in plan["repairs"]}
    manifest = read(DATA / "manifest.json")
    payloads, counts = {}, {}
    for name, expected in plan["files"].items():
        path = DATA / name
        if sha256_file(path) != expected or manifest["files"][name] != expected:
            raise ValueError("dataset changed before reviewed repair: " + name)
        chunks, count = [], 0
        for line in path.read_bytes().splitlines(keepends=True):
            row = json.loads(line)
            change = changes.get(row["documentId"])
            if change is None:
                chunks.append(line)
                continue
            updated, _ = apply_reviewed_cleanup(row, change)
            if digest(updated) != change["expectedAfterSha256"]:
                raise ValueError("reviewed result changed")
            chunks.append((json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
            count += 1
        payloads[name] = b"".join(chunks)
        counts[name] = count
    # The whole dataset was snapshotted before this campaign. Also pin this
    # exact migration's inputs and expected outputs before any publication.
    atomic_publish_json(OUT / "publication.json", {"before": plan["files"],
        "after": {n: digest(v) for n, v in payloads.items()}, "counts": counts,
        "review": approval, "uniqueRecords": len(changes)})
    for name, data in payloads.items():
        atomic_write_bytes(DATA / name, data)
        manifest["files"][name] = digest(data)
    manifest["expansionLegacyOwnerRepair20261101"] = {"records": len(changes),
        "copiesByFile": counts, "evidence": str((OUT / "publication.json").relative_to(ROOT))}
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps({"status": "APPLIED", "uniqueRecords": len(changes), "copies": counts}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "apply"))
    arguments = parser.parse_args()
    stage() if arguments.action == "stage" else apply()
