"""Audit the exact certified expansion, then explicitly append it to R28."""

import argparse
import importlib.util
import json
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_publication import decoder_target
from document_ocr.synthesis.curated_thermal import ventilation_meaning
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"


def read(path):
    return json.loads(path.read_text())


def verified(pin):
    path = ROOT / pin["path"]
    if sha256_file(path) != pin["sha256"]:
        raise ValueError("pinned artifact changed: " + str(path))
    return path


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def add_source_counts(rows):
    return dict(Counter(row["sourceDocumentId"] for row in rows))


def require_schedule(rows, scheduled):
    if len(rows) != 2500 or len({r["documentId"] for r in rows}) != 2500:
        raise ValueError("expansion must contain exactly 2500 unique accepted records")
    if add_source_counts(rows) != scheduled:
        raise ValueError("accepted source counts differ from the approved original schedule")


def feature_function():
    path = ROOT / "docs/analysis/r28-new2500-postrefresh-20261031/audit.py"
    spec = importlib.util.spec_from_file_location("r28_postrefresh_census", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.features


def census(rows):
    features = feature_function()
    flags, packages, types, temperatures, vents = Counter(), Counter(), Counter(), Counter(), Counter()
    hs_lengths, dg_un, hazards, description_lengths = Counter(), Counter(), Counter(), Counter()
    known = total = 0
    source_categories = defaultdict(set)
    for row in rows:
        target = row["target"]
        flags.update(k for k, v in features(target).items() if v)
        flags["transshipment"] += bool(target.get("route", {}).get("transshipmentPort"))
        for container in target.get("containerInformation", []):
            types[f"{container.get('sizeCategory', 'ABSENT')}/{container.get('typeCategory', 'ABSENT')}"] += 1
        for value, unit in {(c["temperatureSetpoint"]["value"], c["temperatureSetpoint"]["unit"])
                            for c in target.get("containerInformation", []) if "temperatureSetpoint" in c}:
            temperatures[f"{value} {unit}"] += 1
        seen_categories, seen_vents = set(), set()
        for goods in target.get("goodsItemDetails", []):
            words = len(goods.get("description", "").split())
            description_lengths["short_1_10" if words <= 10 else "medium_11_30" if words <= 30 else "long_31_plus"] += 1
            for package in goods.get("numberAndTypeOfPackages", []):
                seen_categories.add(package.get("typeCategory", "FREE_TEXT"))
            for code in goods.get("hsCodes", []):
                hs_lengths[str(len(re.sub(r"\D", "", code)))] += 1
            for dg in goods.get("dangerousGoods", []):
                dg_un[dg.get("unNumber", "ABSENT")] += 1
                hazards[dg.get("hazardCategory", "ABSENT")] += 1
            for clause in goods.get("handlingInstructions", []):
                meaning = ventilation_meaning(clause)
                if meaning is not None:
                    seen_vents.add(str(meaning.rate_cbm_per_hour))
        packages.update(seen_categories)
        vents.update(seen_vents)
        for category in seen_categories:
            source_categories[category].add(row.get("sourceDocumentId", row["documentId"]))
        raw_lines, positioned = row["joinedRawText"].splitlines(), row["positionedText"].splitlines()
        if len(raw_lines) != len(positioned):
            raise ValueError("positioned text line count differs")
        for raw, line in zip(raw_lines, positioned, strict=True):
            match = re.fullmatch(r"(.*?) \|\|(?: (\d+),(\d+))?", line)
            if match:
                if match[1] != raw:
                    raise ValueError("coordinate serialization changed OCR text")
                total += bool(raw.strip())
                if match[2]:
                    if not all(0 <= int(v) <= 1000 for v in (match[2], match[3])):
                        raise ValueError("coordinate outside normalized grid")
                    known += bool(raw.strip())
            elif line != raw or (raw.strip() and not re.fullmatch(r"--- PAGE \d+ ---", raw.strip())):
                raise ValueError("unexpected positioned-text line")
    return {"records": len(rows), "flags": dict(flags), "packageDocumentCounts": dict(packages),
            "packageDistinctParents": {k: len(v) for k, v in source_categories.items()},
            "equipmentRows": dict(types), "temperatureDocumentCounts": dict(temperatures),
            "ventilationDocumentRatesM3H": dict(vents), "hsDigitLengths": dict(hs_lengths),
            "distinctUNNumbers": len(dg_un), "UNDeclarationCounts": dict(dg_un),
            "hazardDeclarationCounts": dict(hazards), "descriptionLengths": dict(description_lengths),
            "coordinates": {"knownLines": known, "eligibleNonemptyLines": total,
                            "coverage": known / total if total else 0}}


def audit():
    started = time.perf_counter()
    withdrawn = read(OUT / "withdrawn-staged-samples.json")["samples"]
    schedule = read(OUT / "selection.json")
    constraints = RelationExplicitTaskConstraints.model_validate_json((DATA / "task-constraints.json").read_bytes())
    task = get_training_task(constraints.task).bind_constraints(constraints)
    records, evidence, components = {}, {}, []
    regimes, modes, new_hs = Counter(), Counter(), set()
    for path in sorted((OUT / "components").glob("*/certification.json")):
        receipt = read(path)
        if receipt["status"] != "CERTIFIED_COMPONENT_NOT_YET_APPENDED_TO_R28":
            raise ValueError("component not certified")
        cfg = read(verified(receipt["config"]))
        reviewed = read(verified(receipt["mainAudit"]))
        campaign = ROOT / cfg["output"]
        dataset = campaign / "positions-reflow-v1/dataset.jsonl"
        if sha256_file(dataset) != receipt["positions"]["datasetSha256"]:
            raise ValueError("positioned publication changed")
        rows = [json.loads(line) for line in dataset.open()]
        if len(rows) != receipt["samples"] or set(reviewed["samples"]) != {r["documentId"] for r in rows}:
            raise ValueError("component cardinality differs from main review")
        selected = 0
        for row in rows:
            did = row["documentId"]
            if did in withdrawn:
                continue
            if did in records:
                raise ValueError("duplicate accepted identity")
            item = reviewed["samples"][did]
            candidate = read(verified(item["candidate"]))
            for key in ("review", "scopedReview", "positions"):
                verified(item[key])
            if candidate["joinedRawText"] != row["joinedRawText"] or decoder_target(candidate["target"]) != row["target"]:
                raise ValueError("published data differs from reviewed candidate")
            if candidate["scenario"]["destination"]["country_code"] != "AE":
                raise ValueError("new sample imports outside UAE")
            if task.canonicalize(row["target"]) != row["target"]:
                raise ValueError("target is not canonical under current R28 constraints")
            for field in ("joinedRawText", "positionedText"):
                if digest(row[field].encode()) != row[field + "Sha256"]:
                    raise ValueError("data hash differs")
            provenance = candidate["scenario"]["provenance"]
            regimes[provenance["independentDraw"]["provenance"]["regime"]] += 1
            if provenance["independentDraw"]["provenance"]["sourceShipmentDependence"]:
                raise ValueError("new draw used a commodity donor")
            modes[str(item["groupCount"])] += 1
            new_hs.update(g["hs6"] for g in candidate["scenario"]["goods_identities"] if g.get("hs6"))
            records[did] = row
            evidence[did] = {"sourceDocumentId": row["sourceDocumentId"], "component": pin(path),
                             "candidate": item["candidate"], "mainAudit": receipt["mainAudit"],
                             "recordSha256": digest(row)}
            selected += 1
        components.append({"certificate": pin(path), "selected": selected, "historicallyCertified": receipt["samples"]})
    rows = [records[k] for k in sorted(records)]
    require_schedule(rows, schedule["sourceCounts"])
    base = [json.loads(line) for line in (DATA / "synthetic-train.jsonl").open()]
    real = [json.loads(line) for line in (DATA / "real-train.jsonl").open()]
    validation = [json.loads(line) for line in (DATA / "validation.jsonl").open()]
    if (len(base), len(real), len(validation)) != (6900, 600, 60):
        raise ValueError("working R28 membership changed")
    if set(records) & {r["documentId"] for r in [*base, *real, *validation]}:
        raise ValueError("new identities overlap current membership")
    if set(schedule["sourceCounts"]) & {r["documentId"] for r in validation}:
        raise ValueError("validation source entered synthesis")
    for row in [*base, *real, *validation]:
        if task.canonicalize(row["target"]) != row["target"]:
            raise ValueError("existing R28 schema inconsistency")
    new_stats, final_stats = census(rows), census([*base, *rows, *real])
    if (new_stats["flags"]["dg"], new_stats["flags"]["thermal"], new_stats["flags"]["transshipment"]) != (560, 460, 400):
        raise ValueError("realized primary cohorts differ from schedule")
    for flag in ("dg", "thermal", "forwarding"):
        if final_stats["flags"].get(flag, 0) < 800:
            raise ValueError("final minimum coverage not met: " + flag)
    if sha256_file(DATA / "validation.jsonl") != sha256_file(OUT / "before/validation.jsonl"):
        raise ValueError("validation changed during expansion")
    payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for row in rows)
    output = OUT / "final/new2500.jsonl"
    atomic_publish_bytes(output, payload)
    atomic_publish_json(OUT / "final/accepted-records.json", evidence)
    result = {"status": "FINAL_DATA_AUDIT_PASS_NOT_YET_APPENDED", "newDataset": pin(output),
              "acceptedRecords": pin(OUT / "final/accepted-records.json"), "components": components,
              "withdrawn": pin(OUT / "withdrawn-staged-samples.json"), "new": new_stats, "finalTraining": final_stats,
              "regimes": dict(regimes), "requestedPublicGroupCounts": dict(modes), "distinctSampledHS6": len(new_hs),
              "datasetBeforeAppend": {p.name: sha256_file(p) for p in DATA.iterdir() if p.is_file()},
              "seconds": time.perf_counter() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "scope": "All current accepted candidates have exact hash-pinned full/native/scoped review, main adjudication and per-component replay/schema/geometry. Source gaps retain absent coordinates. This is dataset certification, not live filing or regulatory certification."}
    atomic_publish_json(OUT / "final/audit.json", result)
    print(json.dumps({k: result[k] for k in ("status", "regimes", "distinctSampledHS6", "seconds", "peakRssKiB")}), flush=True)


def append():
    report = read(OUT / "final/audit.json")
    approval = read(OUT / "final/main-approval.json")
    if approval["status"] != "APPROVED" or approval["auditSha256"] != sha256_file(OUT / "final/audit.json"):
        raise ValueError("exact final main approval required")
    for name, value in report["datasetBeforeAppend"].items():
        if sha256_file(DATA / name) != value:
            raise ValueError("dataset changed after audit: " + name)
    incoming = verified(report["newDataset"]).read_bytes()
    current = (DATA / "synthetic-train.jsonl").read_bytes()
    if not current.endswith(b"\n"):
        raise ValueError("existing JSONL lacks final newline")
    updated = current + incoming
    atomic_publish_bytes(OUT / "final/before-append-synthetic-train.jsonl", current)
    manifest = read(DATA / "manifest.json")
    atomic_publish_json(OUT / "final/before-append-manifest.json", manifest)
    receipt = {"status": "CERTIFIED_AND_APPENDED", "added": 2500, "synthetic": 9400,
               "real": 600, "trainingTotal": 10000, "validation": 60,
               "beforeSha256": digest(current), "afterSha256": digest(updated),
               "audit": pin(OUT / "final/audit.json"), "mainApproval": pin(OUT / "final/main-approval.json")}
    atomic_publish_json(OUT / "final/publication.json", receipt)
    atomic_write_bytes(DATA / "synthetic-train.jsonl", updated)
    manifest["files"]["synthetic-train.jsonl"] = digest(updated)
    manifest["counts"]["synthetic-train.jsonl"] = 9400
    manifest["uniqueSyntheticDocumentsIncludingSidecars"] += 2500
    manifest["expansion2500_20261101"] = {"publication": pin(OUT / "final/publication.json"),
                                         "new": report["new"], "trainingTotal": 10000}
    manifest["status"] = "r28_expansion2500_audited_training10000"
    atomic_write_json(DATA / "manifest.json", manifest)
    if sha256_file(DATA / "synthetic-train.jsonl") != receipt["afterSha256"]:
        raise ValueError("published dataset hash mismatch")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "append"])
    args = parser.parse_args()
    audit() if args.command == "audit" else append()
