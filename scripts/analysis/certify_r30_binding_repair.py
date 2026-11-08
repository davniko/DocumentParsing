"""Final replay/census for the reviewed package-mode repair controls."""

import json
import resource
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/synthesis-probes/r30-binding-repair-20261108"


def main():
    started = time.perf_counter()
    p = OUT / "catalog-draft.json"
    cat = PaddleTemplateCatalog(ROOT, {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)})
    for sid in cat.entries:
        cat._template(sid)
    results = []
    sources = {}
    regions = 0
    directories = [
        OUT / f"fresh-{m}{suffix}"
        for m, suffix in [
            (1, ""),
            (1, "-retry-1"),
            (2, ""),
            (2, "-retry-1"),
            (2, "-retry-2"),
            (3, ""),
            (1, "-retry-3"),
            (2, "-retry-3"),
        ]
    ]
    directories.append(ROOT / "artifacts/synthesis-probes/r30-posttraining-20261108")
    for d in directories:
        for candidate in sorted((d / "sampling/candidates").glob("*.json")):
            wording_dir = (
                "wording-reviewed"
                if d.name == "r30-posttraining-20261108"
                else "native-wording-reviewed"
            )
            wp = d / wording_dir / candidate.name
            try:
                receipt = json.loads(wp.read_text()) if wp.exists() else None
                r = cat.render(candidate, wording_receipt=receipt)
                replay = cat.render(candidate, wording_receipt=receipt)
                assert r == replay
                cp = json.loads(candidate.read_text())
                actual = r["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
                expected = cp["target"]["documentPatch"]["goodsItemDetails"][0][
                    "numberAndTypeOfPackages"
                ]
                assert actual == expected
                public_rows = len(actual)
                sources.setdefault(cp["sourceDocumentId"], set()).add(public_rows)
                regions += sum(len(page["regions"]) for page in r["pages"])
                rp = OUT / "final-controls" / candidate.name
                atomic_write_json(rp, r)
                results.append(
                    {
                        "id": r["documentId"],
                        "source": r["sourceDocumentId"],
                        "rows": public_rows,
                        "candidate": str(candidate.relative_to(ROOT)),
                        "rendered": {"path": str(rp.relative_to(ROOT)), "sha256": sha256_file(rp)},
                        "wording": str(wp.relative_to(ROOT)) if wp.exists() else None,
                        "description": r["target"]["goodsItemDetails"][0].get("description"),
                        "packages": actual,
                        "status": "PASS",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "id": candidate.stem,
                        "candidate": str(candidate.relative_to(ROOT)),
                        "status": "FAIL",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    calls = []
    for d in directories[:-1]:
        calls.extend((d / "sampling/calls").glob("*.json"))
        calls.extend((d / "native-wording/provider/calls").glob("*.json"))
    usage = Counter()
    cost = Decimal(0)
    for path in calls:
        x = json.loads(path.read_text())
        cost += Decimal(x["costUsd"])
        usage.update({k: v for k, v in x["usage"].items() if type(v) is int})
    report = {
        "status": "MECHANICAL_REVIEW_COMPLETE",
        "catalogSha256": sha256_file(p),
        "templateModeCoverage": len(cat.entries),
        "freshControls": len(results),
        "counts": Counter(r["status"] for r in results),
        "regions": regions,
        "modeCoverage": {sid: sorted(rows) for sid, rows in sources.items()},
        "results": results,
        "costUsd": str(cost),
        "allAttemptUsage": usage,
        "seconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "final-control-audit.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["results", "modeCoverage"]}),
        flush=True,
    )
    print(json.dumps([r for r in results if r["status"] == "FAIL"]), flush=True)


if __name__ == "__main__":
    main()
