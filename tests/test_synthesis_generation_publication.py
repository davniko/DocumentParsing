import asyncio
import threading
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedRunError
from document_ocr.synthesis.template_compiler import complete_pipeline as pipeline

FIELDS = {
    "target.json": "target",
    "target-receipt.json": "targetReceipt",
    "auxiliary-values.json": "auxiliaryValues",
    "numeric-auxiliary.json": "numericAuxiliary",
    "agent-stage.json": "agentStage",
    "generation-ledger.json": "ledger",
    "generation-attempts.json": "attempts",
    "residual-attempts.json": "residualAttempts",
    "goods-identities.json": "goodsIdentities",
    "host-lexical-facts.json": "hostLexicalFacts",
    "host-lexical-evidence.json": "hostLexicalEvidence",
    "dangerous-goods-facts.json": "dangerousGoodsFacts",
    "cargo-scenario.json": "cargoScenario",
    "route-projection.json": "routeProjection",
    "customs-presentation.json": "customsPresentation",
    "render-result.json": "renderResult",
}


def cases(count=7):
    source = NS(source="é source\r\n".encode(), source_target={"source": True})
    rows = [
        dict(
            sampleId=f"sample-{i}",
            sourceDocumentId="source",
            rendered=f"é new {i}\r\n",
            **{key: {"field": key, "index": i} for key in FIELDS.values()},
        )
        for i in range(count)
    ]
    return {"source": source}, rows


def test_concurrent_atomic_publication_preserves_every_byte_and_plan_order(tmp_path):
    import json

    sources, rows = cases()
    frozen = deepcopy(rows)
    inventories = []
    for workers in (1, 4):
        run = StagedArtifactRun(
            output_parent=tmp_path, run_name=f"workers-{workers}", transaction_sha256="a" * 64
        )
        paths = asyncio.run(pipeline._publish_generated_cases(run, sources, rows, workers=workers))
        assert len(paths) == len(rows) * 19
        assert [p.split("/")[1] for p in paths[::19]] == [r["sampleId"] for r in rows]
        for row in rows:
            folder = run.stage_root / "cases" / row["sampleId"]
            assert (folder / "source.txt").read_bytes() == sources["source"].source
            assert (folder / "rendered.txt").read_bytes() == row["rendered"].encode()
            assert (
                json.loads((folder / "source-target.json").read_bytes())
                == sources["source"].source_target
            )
            for name, key in FIELDS.items():
                assert json.loads((folder / name).read_bytes()) == row[key]
        # Restart uses byte-identical existing artifacts, not an overwrite.
        assert (
            asyncio.run(pipeline._publish_generated_cases(run, sources, rows, workers=workers))
            == paths
        )
        inventories.append({p: sha256_file(run.stage_root / p) for p in paths})
        run.commit(expected_artifacts=paths, metadata={"documents": len(rows)})
        assert run.completed
    assert inventories[0] == inventories[1]
    assert rows == frozen


def test_publication_conflicts_surface_without_committing_or_overwriting(tmp_path):
    sources, rows = cases(2)
    run = StagedArtifactRun(
        output_parent=tmp_path, run_name="conflict", transaction_sha256="b" * 64
    )
    paths = asyncio.run(pipeline._publish_generated_cases(run, sources, rows, workers=2))
    pinned = {p: sha256_file(run.stage_root / p) for p in paths}
    rows[0]["target"] = {"different": True}
    with pytest.raises(StagedRunError, match="conflict"):
        asyncio.run(pipeline._publish_generated_cases(run, sources, rows, workers=2))
    assert not run.final_root.exists()
    assert pinned == {p: sha256_file(run.stage_root / p) for p in paths}


def test_publication_really_runs_concurrently_with_bounded_admission(monkeypatch):
    sources, rows = cases(8)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = peak = calls = 0

    def publish(_run, _source, row):
        nonlocal active, peak, calls
        with lock:
            active += 1
            peak = max(peak, active)
            calls += 1
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return (row["sampleId"],)

    monkeypatch.setattr(pipeline, "_publish_generated_case", publish)
    result = asyncio.run(pipeline._publish_generated_cases(None, sources, rows, workers=4))
    assert result == [row["sampleId"] for row in rows]
    assert peak == 4 and calls == 8 and active == 0


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_limits_are_explicit(workers):
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(pipeline._publish_generated_cases(None, {}, [], workers=workers))


def test_duplicate_sample_ownership_is_rejected_before_writing(monkeypatch):
    sources, rows = cases(1)

    def unexpected(*args):
        pytest.fail("duplicate identity must be rejected before filesystem work")

    monkeypatch.setattr(pipeline, "_publish_generated_case", unexpected)
    with pytest.raises(ValueError, match="identities must be unique"):
        asyncio.run(pipeline._publish_generated_cases(None, sources, rows * 2, workers=2))
