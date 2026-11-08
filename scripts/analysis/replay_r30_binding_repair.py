"""Paired native rendering: affected descendants plus every other admitted parent."""

import json
import resource
import time
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/synthesis-probes/r30-binding-repair-20261108"


def main():
    start = time.perf_counter()
    oldpin = json.loads((OUT / "inventory.json").read_text())["catalog"]
    newpath = OUT / "catalog-draft.json"
    old = PaddleTemplateCatalog(ROOT, oldpin)
    new = PaddleTemplateCatalog(
        ROOT, {"path": str(newpath.relative_to(ROOT)), "sha256": sha256_file(newpath)}
    )
    affected = {r["source"] for r in json.loads((OUT / "inventory.json").read_text())["rows"]}
    seen = set()
    results = []
    times = [0.0, 0.0]
    for line in (
        ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1/synthetic-train.jsonl"
    ).open():
        row = json.loads(line)
        sid = row["sourceDocumentId"]
        if sid in seen and sid not in affected:
            continue
        seen.add(sid)
        e = row["evidence"]
        path = ROOT / e["candidate"]["path"]
        candidate = json.loads(path.read_text())
        wording = (
            json.loads((ROOT / e["wording"]["path"]).read_text()) if e.get("wording") else None
        )
        if "joinedRawText" not in candidate:
            results.append(
                {
                    "id": row["documentId"],
                    "source": sid,
                    "status": "source-preserving-receipt",
                    "path": e["candidate"]["path"],
                }
            )
            continue
        try:
            values = []
            for i, cat in enumerate([old, new]):
                t = time.perf_counter()
                try:
                    values.append(cat.render(path, wording_receipt=wording))
                except Exception as exc:
                    values.append({"baselineError": f"{type(exc).__name__}: {exc}"})
                times[i] += time.perf_counter() - t
            if values[0] != values[1]:
                raise ValueError("paired rendering changed")
            results.append(
                {
                    "id": row["documentId"],
                    "source": sid,
                    "status": "baseline-error" if "baselineError" in values[0] else "exact",
                    **(
                        {"reason": values[0]["baselineError"]}
                        if "baselineError" in values[0]
                        else {}
                    ),
                }
            )
        except Exception as exc:
            results.append(
                {"id": row["documentId"], "source": sid, "error": f"{type(exc).__name__}: {exc}"}
            )
        if len(results) % 100 == 0:
            print(
                json.dumps({"done": len(results), "errors": sum("error" in r for r in results)}),
                flush=True,
            )
            atomic_write_json(OUT / "paired-progress.json", results)
    result = {
        "results": results,
        "parents": len(seen),
        "seconds": time.perf_counter() - start,
        "pairedSeconds": times,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "paired-replay.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    print(json.dumps([r for r in results if "error" in r]), flush=True)


if __name__ == "__main__":
    main()
