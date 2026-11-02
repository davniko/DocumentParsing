"""Full source-only readiness, repaired homonym frames and paired preflight timing."""

import ast
import copy
import json
import resource
import statistics
import subprocess
import time
import tracemalloc
from pathlib import Path

from pydantic import BaseModel
from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_geography import project_composite_locality_homonyms
from document_ocr.synthesis import independent_preparation as preparation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"
AUTHORITY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v21/complete-source-authority-config.json"


class Scenario(BaseModel):
    replacements: dict[str, str]


def main():
    started = time.perf_counter()
    config = json.loads(AUTHORITY.read_text())
    context = preparation._load_source_context(ROOT, config)
    readiness = []
    for sid in config["source_ids"]:
        compiled, hold = preparation._source_readiness(context, sid, set())
        readiness.append({"sourceDocumentId": sid, "status": "PASS" if hold is None else "HELD", "hold": hold})
    frames = []
    for prefix in ("551657f3", "48973902", "35288b68"):
        sid = next(s for s in config["source_ids"] if s[4:12] == prefix)
        bp = context.source.blueprint(sid)
        recipes = bp.ownership_data.get("surfaces", {})
        paths = {p for r in bp.regions for p in r.target_paths if p.endswith(".country") and not p.startswith("documentPatch.parties.")}
        checked = 0
        for country in sorted(paths):
            role, name = country.rsplit(".", 1)[0], country.removesuffix(".country") + ".name"
            owners = [r for r in bp.regions if country in r.target_paths]
            if not owners or not all(set(r.target_paths) == {country, name} and r.key in recipes for r in owners):
                continue
            target = copy.deepcopy(bp.target)
            node = target
            for part in role.split("."):
                node = node[part]
            node.update(name="SINGAPORE", country="SINGAPORE")
            edits = [{"key": r.key, "occurrence": r.occurrence, "byteStart": r.start, "byteEnd": r.end,
                      "before": r.source, "after": recipes[r.key]["expression"].replace("{"+name+"}", "SINGAPORE").replace("{"+country+"}", "SINGAPORE"), "targetPaths": list(r.target_paths)} for r in owners]
            candidate = {"target": target, "proof": {"sourceSha256": digest(bp.source.encode()), "edits": edits}}
            plan, receipt = project_composite_locality_homonyms((bp, Scenario(replacements={}), "control", target), candidate)
            if receipt is None:
                raise ValueError("reviewed complete frame did not project")
            current = plan[3]
            for part in role.split("."):
                current = current[part]
            if current != {"name": "SINGAPORE"} or bp.target == target:
                raise ValueError("homonym projection mutated or retained duplicate target")
            frames.append({"sourceDocumentId": sid, "role": role, "occurrences": len(owners), "status": "PASS"})
            checked += 1
        if not checked:
            raise ValueError("repaired source has no tested composite frame")
    previous = subprocess.check_output(["git", "show", "HEAD:src/document_ocr/synthesis/independent_preparation.py"], cwd=ROOT, text=True)
    fn = next(n for n in ast.parse(previous).body if isinstance(n, ast.FunctionDef) and n.name == "_compile_source")
    namespace = dict(vars(preparation))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "old_preflight", "exec"), namespace)
    old = namespace["_compile_source"]
    # Pair successful old/new uniform checks; the new mixed cases are correctness controls.
    sources = config["source_ids"][:20]
    timing = {}
    for label, function in [("before", old), ("after", preparation._compile_source)]:
        samples = []
        tracemalloc.start()
        for _ in range(5):
            start = time.perf_counter()
            for sid in sources:
                function(context, sid, set())
            samples.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing[label] = {"medianSecondsFor20Sources": statistics.median(samples), "peakTracedBytes": peak}
    mixed = []
    for prefix in ("d0d9e558", "4985afa4"):
        sid = next(s for s in config["source_ids"] if s[4:12] == prefix)
        preparation._compile_source(context, sid, {sid})
        try:
            old(context, sid, {sid})
        except ValueError as exc:
            mixed.append({"sourceDocumentId": sid, "oldFailure": str(exc), "current": "PASS"})
        else:
            raise ValueError("mixed regression no longer reproduces")
    result = {"status": "PASS" if all(r["status"] == "PASS" for r in readiness) else "HELD", "sources": readiness,
              "homonymFrames": frames, "mixedRegressions": mixed, "benchmark": timing,
              "seconds": time.perf_counter() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    atomic_publish_json(OUT / "complete-authority-validation.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "sources"}), flush=True)
    if result["status"] != "PASS":
        raise SystemExit("Some source prerequisites remain held.")


if __name__ == "__main__":
    main()
