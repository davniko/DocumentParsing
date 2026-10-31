"""Measure the final prompt-only projection against its exact archived predecessor."""

import ast
import copy
import gc
import hashlib
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis import independent_presentation as implementation

ROOT = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")


def main():
    path = Path(implementation.__file__)
    source = path.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "semantic_private_packing"
    )
    branch = next(
        node
        for node in function.body
        if isinstance(node, ast.If) and "declaredGroupLedgers" in ast.unparse(node.test)
    )
    lines = source.splitlines(keepends=True)
    original_source = "".join(lines[: branch.lineno - 1] + lines[branch.end_lineno :])
    archive = json.loads((ROOT / "wording-context-before-v2/summary.json").read_text())
    if (
        hashlib.sha256(original_source.encode()).hexdigest()
        != archive["projectionImplementationBefore"]["sha256"]
    ):
        raise ValueError("benchmark baseline is not the exact archived implementation")
    previous = copy.deepcopy(function)
    previous.body = [node for node in previous.body if node.lineno != branch.lineno]
    namespace = vars(implementation).copy()
    exec(compile(ast.Module(body=[previous], type_ignores=[]), str(path), "exec"), namespace)
    baseline = namespace["semantic_private_packing"]
    candidate = json.loads(
        (
            ROOT / "batch-41/campaign/candidates/syn_full_v7_40dc8782e73bb49361d6bf4a.json"
        ).read_text()
    )
    ledger = candidate["scenario"]["provenance"]["freshPrivatePacking"]
    before = copy.deepcopy(ledger)
    results = []
    for name, operation in [
        ("before", baseline),
        ("after", implementation.semantic_private_packing),
        ("after", implementation.semantic_private_packing),
        ("before", baseline),
    ]:
        timings = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            for _ in range(1000):
                operation(ledger)
            timings.append((time.perf_counter() - start) * 1000 / 1000)
        tracemalloc.start()
        projected = operation(ledger)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append(
            {
                "mode": name,
                "medianMs": statistics.median(timings),
                "peakTemporaryBytes": peak,
                "projectedJsonBytes": len(json.dumps(projected)),
            }
        )
    if ledger != before:
        raise ValueError("prompt projection mutated host evidence")
    atomic_publish_json(
        ROOT / "benchmark-context-projection.json",
        {
            "runs": results,
            "inputGroups": 3,
            "hostLedgerUnchanged": True,
            "iterationsPerTiming": 1000,
            "timingsPerRun": 5,
            "scope": "Prompt-only function, not provider latency or full-campaign throughput.",
        },
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
