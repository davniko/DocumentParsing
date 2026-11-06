"""Replay explicit audited overlay changes against all three retained controls.

Writes an isolated candidate catalog only. The production catalog and its
historical control receipts stay unchanged until complete semantic readback.
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time

from publish_r30_sources import TEMPLATES, pin, verify_pin
from rebase_r30_sources import ART as REBASE_ART
from rebase_r30_sources import BASE, PILOT_ART, ROOT, read, rows
from rebase_r30_sources import OUT as REBASE_OUT
from render_r30_paddle_pilot import load_current
from review_r30_paddle import differences

from document_ocr.atomic import atomic_write_json
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog

OUT = ROOT / "docs/analysis/r30-independent-audit-20261103"
ART = ROOT / "artifacts/paddle-migration/r30-independent-audit-20261103/template-repairs"


def apply_overlay_review(overlay, decision):
    if not decision.get("reason") or decision.keys() - {
        "reason",
        "add",
        "append",
        "replace",
        "replaceItems",
        "remove",
    }:
        raise ValueError("overlay correction needs explicit rationale and known operations")
    result = copy.deepcopy(overlay)
    for key, value in decision.get("add", {}).items():
        if key in result:
            raise ValueError("new overlay property already exists; use reviewed replacement")
        result[key] = copy.deepcopy(value)
    for key, replacements in decision.get("replaceItems", {}).items():
        for replacement in replacements:
            if set(replacement) != {"before", "after"}:
                raise ValueError("item replacement requires reviewed before and after values")
            before, after = replacement["before"], replacement["after"]
            if before == after or result[key].count(before) != 1:
                raise ValueError("item replacement must change one unique existing entry")
            result[key][result[key].index(before)] = copy.deepcopy(after)
    for key, values in decision.get("append", {}).items():
        if not isinstance(values, list) or not values:
            raise ValueError("append needs a nonempty list")
        prior = result.setdefault(key, [])
        if not isinstance(prior, list) or any(v in prior for v in values):
            raise ValueError("cannot append duplicate entries or append to a non-list")
        prior.extend(copy.deepcopy(values))
    for key, replacement in decision.get("replace", {}).items():
        if set(replacement) != {"before", "after"} or result[key] != replacement["before"]:
            raise ValueError("overlay replacement does not match reviewed before-value")
        if replacement["before"] == replacement["after"]:
            raise ValueError("overlay replacement changes nothing")
        result[key] = copy.deepcopy(replacement["after"])
    for key, values in decision.get("remove", {}).items():
        for value in values:
            if result[key].count(value) != 1:
                raise ValueError("removed overlay entry is not unique")
            result[key].remove(value)
    if overlay == result:
        raise ValueError("overlay review changes nothing")
    owners = [e["owner"] for e in result["extras"]]
    if len(owners) != len(set(owners)):
        raise ValueError("duplicate extra owner")
    return result


def stage(review_paths):
    started = time.perf_counter()
    old_catalog_pin = pin(TEMPLATES / "catalog.json")
    manifest = read(verify_pin(old_catalog_pin))
    controls = read(REBASE_OUT / "handoff-controls.json")
    baseline = PaddleTemplateCatalog(ROOT, old_catalog_pin)
    decisions = {}
    for path in review_paths:
        review = read(path)
        if review["status"] != "MAIN_AGENT_REVIEWED_OWNERS_REQUIRE_THREE_REPLAYS":
            raise ValueError("requires main-agent review, not provider proposals")
        if decisions.keys() & review["sources"].keys():
            raise ValueError("duplicate reviewed template")
        decisions.update(review["sources"])
    sources = {did[4:12]: did for did in manifest["templates"]}
    if len(sources) != len(manifest["templates"]) or decisions.keys() - sources.keys():
        raise ValueError("reviewed prefixes are not unique admitted templates")
    current = {r["documentId"]: r for r in rows(BASE / "synthetic-train.jsonl")}
    for prefix, decision in decisions.items():
        did = sources[prefix]
        template = read(verify_pin(manifest["templates"][did]["template"]))
        template["overlay"] = apply_overlay_review(template["overlay"], decision)
        path = ART / "cases" / (did + ".json")
        atomic_write_json(path, template)
        manifest["templates"][did]["template"] = pin(path)
    catalog_path = ART / "catalog.json"
    atomic_write_json(catalog_path, manifest)
    candidate_catalog = PaddleTemplateCatalog(ROOT, pin(catalog_path))
    failures, results, before_seconds, after_seconds = [], [], 0.0, 0.0
    for prefix in decisions:
        did = sources[prefix]
        template, _, transfer, _ = candidate_catalog._template(did)
        expected = {read(verify_pin(p))["documentId"]: p for p in controls[did]}
        artifact = (
            PILOT_ART
            if manifest["templates"][did]["privacy"]["kind"] == "reviewed_replacement_spans"
            else REBASE_ART
        )
        for selection in transfer["descendants"]:
            identity = selection["id"]
            path = ROOT / selection["path"]
            try:
                prior = load_current(selection, current)
                spec = candidate_catalog.wording_specification(path)
                wording = read(artifact / "new-wording" / (identity + ".json")) if spec else None
                tick = time.perf_counter()
                before = baseline.render(path, current=prior, wording_receipt=wording)
                before_seconds += time.perf_counter() - tick
                saved = read(verify_pin(expected[identity]))
                saved.pop("migrationPins", None)
                if before != saved:
                    raise ValueError("baseline replay differs from historical control")
                tick = time.perf_counter()
                after = candidate_catalog.render(path, current=prior, wording_receipt=wording)
                after_seconds += time.perf_counter() - tick
                if after != candidate_catalog.render(path, current=prior, wording_receipt=wording):
                    raise ValueError("amended replay is not deterministic")
                output = ART / "controls" / (identity + ".json")
                atomic_write_json(output, after)
                results.append(
                    {
                        "source": did,
                        "id": identity,
                        "before": expected[identity],
                        "after": pin(output),
                        "targetChanges": list(differences(before["target"], after["target"])),
                        "status": "REPLAY_PASSED_REQUIRES_TEXT_TARGET_READBACK",
                    }
                )
            except (ValueError, KeyError, IndexError) as exc:
                failures.append({"source": did, "id": identity, "error": str(exc)})
        print(
            json.dumps({"source": prefix, "passed": len(results), "failed": len(failures)}),
            flush=True,
        )
    report = {
        "status": "STAGED_TEMPLATE_REPAIRS_NOT_PUBLISHED",
        "baseCatalog": old_catalog_pin,
        "candidateCatalog": pin(catalog_path),
        "reviewPins": [pin(p) for p in review_paths],
        "templates": len(decisions),
        "results": results,
        "failures": failures,
        "benchmark": {
            "baselineRenderSeconds": before_seconds,
            "amendedRenderSeconds": after_seconds,
        },
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "template-repair-staging.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    if failures or len(results) != 3 * len(decisions):
        raise ValueError("template replay incomplete; no publication")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", nargs="+", help="Project-relative reviewed overlay amendments")
    stage([ROOT / p for p in parser.parse_args().reviews])
