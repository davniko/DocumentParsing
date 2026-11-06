"""Check migration binding expansion against every certified native control."""

import argparse
import copy
import json
import resource
import time

from migrate_r30_synthetic import DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin
from rebase_r30_sources import ART as REBASE_ART
from rebase_r30_sources import BASE, PILOT_ART, rows
from render_r30_paddle_pilot import load_current

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog


def semantic_changes(old, new, path=""):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            yield from semantic_changes(
                old.get(key, "ABSENT"), new.get(key, "ABSENT"), path + "/" + key
            )
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            yield from semantic_changes(a, b, path + "/" + str(index))
    elif old != new:
        yield path, old, new


def main(catalog_path, prefixes=(), review_path=None):
    started = time.perf_counter()
    manifest = read(DATA / "manifest.json")
    catalog_pin = pin(catalog_path)
    catalog = PaddleTemplateCatalog(ROOT, catalog_pin)
    expected = read(verify_pin(manifest["templateControls"]))
    reviewed = read(review_path) if review_path else {}
    if reviewed and reviewed["catalog"] != catalog_pin:
        raise ValueError("semantic review belongs to a different catalog")
    controls, changes = {}, []
    current = {r["documentId"]: r for r in rows(BASE / "synthetic-train.jsonl")}
    repaired = read(
        ROOT
        / "artifacts/paddle-migration/r30-independent-audit-20261103/product-repair/receipt.json"
    )
    corrected = read(verify_pin(repaired["after"]))
    failures, passed, render_seconds = [], 0, 0.0
    selected = [s for s in sorted(catalog.entries) if not prefixes or s[4:12] in prefixes]
    if prefixes and len(selected) != len(prefixes):
        raise ValueError("requested source prefixes do not resolve uniquely")
    for did in selected:
        _, _, transfer, _ = catalog._template(did)
        old_by_id = {read(verify_pin(p))["documentId"]: p for p in expected[did]}
        controls[did] = []
        for selection in transfer["descendants"]:
            identity, path = selection["id"], ROOT / selection["path"]
            try:
                spec = catalog.wording_specification(path)
                artifact = (
                    PILOT_ART
                    if catalog.entries[did]["privacy"]["kind"] == "reviewed_replacement_spans"
                    else REBASE_ART
                )
                # A corrected control can acquire additional native-only owners.
                # Its newly reviewed receipt is pinned in the transfer; never
                # overwrite the original wording file used by prior certificates.
                wording_path = (verify_pin(selection["wording"]) if "wording" in selection
                                else artifact / "new-wording" / (identity + ".json"))
                wording = read(wording_path) if spec else None
                prior = load_current(selection, current)
                if identity == corrected["documentId"]:
                    prior = copy.deepcopy(prior)
                    prior["joinedRawText"] = corrected["joinedRawText"]
                    prior["target"]["goodsItemDetails"][0]["description"] = corrected["target"][
                        "documentPatch"
                    ]["goodsItemDetails"][0]["description"]
                tick = time.perf_counter()
                result = catalog.render(path, current=prior, wording_receipt=wording)
                render_seconds += time.perf_counter() - tick
                previous = read(verify_pin(old_by_id[identity]))
                previous.pop("migrationPins", None)
                if result != previous:
                    scope = reviewed.get("sources", {}).get(did[4:12])
                    if scope is None:
                        raise ValueError("binding expansion changed an unreviewed control")
                    delta = list(semantic_changes(previous["target"], result["target"]))
                    for change in delta:
                        if not any(change[0].startswith(p) for p in scope["targetPaths"]):
                            raise ValueError(f"unreviewed target change: {change}")
                    if not scope.get("inputChanges"):
                        a, b = copy.deepcopy(previous), copy.deepcopy(result)
                        a.pop("target")
                        b.pop("target")
                        if a != b:
                            raise ValueError("target-only correction changed input or geometry")
                    changes.append(
                        {
                            "source": did,
                            "id": identity,
                            "targetChanges": delta,
                            "inputChanged": result["pages"] != previous["pages"],
                        }
                    )
                if review_path:
                    destination = catalog_path.parent / "controls" / (identity + ".json")
                    atomic_publish_json(destination, result)
                    controls[did].append(pin(destination))
                passed += 1
            except (ValueError, KeyError, IndexError) as exc:
                failures.append({"source": did, "id": identity, "error": str(exc)})
        if (passed + len(failures)) % 60 == 0:
            print(json.dumps({"passed": passed, "failures": len(failures)}), flush=True)
    report = {
        "catalog": catalog_pin,
        "priorControls": manifest["templateControls"],
        "passed": passed,
        "sources": selected,
        "failures": failures,
        "renderSeconds": render_seconds,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "review": pin(review_path) if review_path else None,
        "reviewedChanges": changes,
    }
    if review_path and not failures:
        atomic_publish_json(catalog_path.parent / "controls.json", controls)
    atomic_publish_json(OUT / (catalog_path.parent.name + "-replay.json"), report)
    print(json.dumps(report, indent=2))
    if failures or passed != 3 * len(selected):
        raise ValueError("candidate catalog did not preserve all reviewed controls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog")
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument("--review")
    args = parser.parse_args()
    main(ROOT / args.catalog, args.sources, ROOT / args.review if args.review else None)
