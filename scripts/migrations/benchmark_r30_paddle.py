"""Replay the native input contract against the audit prototype on all 660 sources."""

import importlib.util
import json
import resource
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from document_ocr.atomic import atomic_write_json
from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion, serialize

ROOT = Path(__file__).resolve().parents[2]


def main():
    spec = importlib.util.spec_from_file_location(
        "probe", ROOT / "docs/analysis/paddle-migration-scope-20261102/probe.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    ids = {"pdf_" + d["source_sha256"] for d in probe.read(probe.PRIOR / "documents.json")}
    ps = pq.read_table(
        probe.OCR / "pages.parquet", filters=[("document_id", "in", sorted(ids))]
    ).to_pylist()
    rs = pq.read_table(
        probe.OCR / "lines.parquet", filters=[("document_id", "in", sorted(ids))]
    ).to_pylist()
    pages, regions = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in ps:
        pages[r["document_id"]].append(r)
    for r in rs:
        regions[r["document_id"]][r["page_index"]].append(r)
    cases, rounding = [], []
    for did in sorted(ids):
        frame = tuple(
            PaddlePage(
                p["page_index"],
                p["width"],
                p["height"],
                tuple(
                    PaddleRegion(
                        r["line_index"], r["text"], tuple(r[k] for k in ("x1", "y1", "x2", "y2"))
                    )
                    for r in sorted(regions[did][p["page_index"]], key=lambda r: r["line_index"])
                ),
            )
            for p in sorted(pages[did], key=lambda p: p["page_index"])
        )
        for mode in ("centroid", "bbox"):
            old = probe.serialize_regions(pages[did], regions[did], mode)
            new = serialize(frame, geometry=mode)
            for a, b in zip(old.splitlines(), new.splitlines(), strict=True):
                if a != b:
                    ta, ca = a.rsplit(" || ", 1)
                    tb, cb = b.rsplit(" || ", 1)
                    if (
                        ta != tb
                        or max(
                            abs(int(x) - int(y))
                            for x, y in zip(ca.split(","), cb.split(","), strict=True)
                        )
                        > 1
                    ):
                        raise ValueError("text/order or non-rounding geometry regression")
                    rounding.append({"pdf": did, "mode": mode, "before": a, "after": b})
        cases.append((did, frame))
    timings, heap_peaks = {}, {}
    for mode in ("centroid", "bbox"):
        for name in ("prototype", "contract"):
            repeats = []
            for _ in range(5):
                t = time.perf_counter()
                for did, frame in cases:
                    if name == "prototype":
                        probe.serialize_regions(pages[did], regions[did], mode)
                    else:
                        serialize(frame, geometry=mode)
                repeats.append(time.perf_counter() - t)
            timings[mode + "_" + name] = repeats
            tracemalloc.start()
            for did, frame in cases:
                if name == "prototype":
                    probe.serialize_regions(pages[did], regions[did], mode)
                else:
                    serialize(frame, geometry=mode)
            heap_peaks[mode + "_" + name] = tracemalloc.get_traced_memory()[1] / 1024
            tracemalloc.stop()
    result = {
        "documents": len(cases),
        "pages": len(ps),
        "regions": len(rs),
        "exactTextAndOrderViews": len(cases) * 2,
        "oneGridRoundingDifferences": rounding,
        "roundingExplanation": (
            "Scale before division avoids intermediate binary half-tie drift; no box changes."
        ),
        "timingsSeconds": timings,
        "serializerPeakPythonHeapKiB": heap_peaks,
        "memoryScope": "incremental serialization heap; source inventory loaded before tracing",
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(
        ROOT / "docs/analysis/r30-paddle-pilot-20261102/serializer-benchmark.json", result
    )
    print(json.dumps({**result, "oneGridRoundingDifferences": len(rounding)}, indent=2))


if __name__ == "__main__":
    main()
