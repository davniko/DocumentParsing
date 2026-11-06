"""Replay every active control through the standalone native catalog interface."""

from __future__ import annotations

import json
import resource
import time

from publish_r30_sources import TEMPLATES, pin, verify_pin
from rebase_r30_sources import ART, BASE, OUT, PILOT_ART, ROOT, read, rows
from render_r30_paddle_pilot import load_current

from document_ocr.atomic import atomic_write_json
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog


def main():
    started = time.perf_counter()
    manifest_pin = pin(TEMPLATES / "catalog.json")
    catalog = PaddleTemplateCatalog(ROOT, manifest_pin)
    receipts = read(OUT / "handoff-controls.json")
    current = {r["documentId"]: r for r in rows(BASE / "synthetic-train.jsonl")}
    if receipts.keys() != catalog.entries.keys():
        raise ValueError("control inventory differs from catalog")
    failures, count, fresh, fresh_sources = [], 0, 0, set()
    render_seconds = 0
    for did, control_pins in receipts.items():
        _, _, transfer, _ = catalog._template(did)
        artifact = (
            PILOT_ART
            if catalog.entries[did]["privacy"]["kind"] == "reviewed_replacement_spans"
            else ART
        )
        expected = {read(verify_pin(p))["documentId"]: p for p in control_pins}
        for selection in transfer["descendants"]:
            count += 1
            path = ROOT / selection["path"]
            try:
                spec = catalog.wording_specification(path)
                wording = (
                    read(artifact / "new-wording" / (selection["id"] + ".json")) if spec else None
                )
                prior = load_current(selection, current)
                t0 = time.perf_counter()
                result = catalog.render(path, current=prior, wording_receipt=wording)
                render_seconds += time.perf_counter() - t0
                saved = read(verify_pin(expected[selection["id"]]))
                saved.pop("migrationPins", None)
                if result != saved:
                    from review_r30_paddle import differences

                    raise ValueError(
                        str(
                            [
                                (p, str(a)[:80], str(b)[:80])
                                for p, a, b in differences(result, saved)
                            ]
                        )
                    )
                if "freshRecord" in selection:
                    independent = catalog.render(path, wording_receipt=wording)
                    if independent != result:
                        raise ValueError("fresh mode differs from its reviewed sampled receipt")
                    fresh += 1
                    fresh_sources.add(did)
            except Exception as exc:
                failures.append(
                    {
                        "source": did,
                        "sample": selection["id"],
                        "errorType": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if count % 60 == 0:
            print(json.dumps({"controls": count, "failures": len(failures)}), flush=True)
    report = {
        "catalog": manifest_pin,
        "templates": len(receipts),
        "controls": count,
        "freshStandaloneControls": fresh,
        "freshSources": sorted(fresh_sources),
        "failures": failures,
        "elapsedSeconds": time.perf_counter() - started,
        "renderAndValidationSeconds": render_seconds,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "handoff-catalog-validation.json", report)
    print(json.dumps(report, indent=2))
    if failures:
        raise ValueError("catalog validation failed; no aggregate admission")


if __name__ == "__main__":
    main()
