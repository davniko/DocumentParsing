import asyncio
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/migrations"
sys.path.insert(0, str(SCRIPTS))
from audit_r30_synthetic_batch import review_pending_entries


def test_refills_worker_without_waiting_for_slow_sibling():
    async def run():
        release = asyncio.Event()
        starts = []

        async def one(entry):
            starts.append(entry["id"])
            if entry["id"] == "slow":
                await asyncio.wait_for(release.wait(), timeout=1)
            if entry["id"] == "third":
                release.set()

        result = await review_pending_entries(
            [{"id": v} for v in ("slow", "fast", "third")], one, 2
        )
        assert starts == ["slow", "fast", "third"]
        assert result == ([], False, [])

    asyncio.run(run())


def test_concurrency_bound_and_every_identity_once():
    async def run():
        active = peak = 0
        seen = []

        async def one(entry):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            seen.append(entry["id"])
            await asyncio.sleep(0)
            active -= 1

        assert await review_pending_entries([{"id": i} for i in range(100)], one, 8) == ([], False, [])
        assert peak == 8 and sorted(seen) == list(range(100)) and active == 0

    asyncio.run(run())


def test_account_rejection_stops_new_dispatch_but_preserves_inflight():
    async def run():
        release = asyncio.Event()
        completed = []

        async def one(entry):
            if entry["id"] == 0:
                await release.wait()
                raise RuntimeError("ModelHTTPError: status_code: 402, insufficient credits")
            release.set()
            await asyncio.sleep(0.01)
            completed.append(entry["id"])

        failures, blocked, unattempted = await review_pending_entries(
            [{"id": i} for i in range(10)], one, 2
        )
        assert blocked and [f["id"] for f in failures] == [0]
        assert completed == [1] and unattempted == list(range(2, 10))

    asyncio.run(run())


def test_failures_remain_ordered_and_do_not_hide_other_results():
    async def one(entry):
        await asyncio.sleep(0)
        if entry["id"] % 2:
            raise ValueError(f"invalid evidence {entry['id']}")

    failures, blocked, unattempted = asyncio.run(
        review_pending_entries([{"id": i} for i in range(6)], one, 3)
    )
    assert [f["id"] for f in failures] == [1, 3, 5]
    assert not blocked and not unattempted


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_invalid_width_rejected(width):
    async def one(entry):
        pytest.fail("must not dispatch")

    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(review_pending_entries([], one, width))


def test_empty_queue():
    async def one(entry):
        pytest.fail("must not dispatch")

    assert asyncio.run(review_pending_entries([], one, 8)) == ([], False, [])


def test_cohorts_are_pinned_selected_and_deduplicated(tmp_path, monkeypatch):
    import audit_r30_synthetic_batch as audit

    documents = {
        "001-staging.json": {"results": [{"id": "a"}, {"id": "skip"}]},
        "002-staging.json": {"results": [{"id": "b"}]},
    }
    monkeypatch.setattr(audit, "OUT", tmp_path)
    monkeypatch.setattr(audit, "read", lambda path: documents[path.name])
    monkeypatch.setattr(audit, "pin", lambda path: {"path": str(path)})
    entries, pins = audit.collect_review_cohorts({2: ["b"], 1: ["a"]})
    assert entries == [{"id": "a"}, {"id": "b"}]
    assert set(pins) == {"1", "2"}
    with pytest.raises(ValueError, match="distinct rendered"):
        audit.collect_review_cohorts({1: ["missing"]})
    documents["002-staging.json"]["results"] = [{"id": "a"}]
    with pytest.raises(ValueError, match="across cohorts"):
        audit.collect_review_cohorts({1: ["a"], 2: ["a"]})
    with pytest.raises(ValueError, match="at least one"):
        audit.collect_review_cohorts({})


def test_shared_queue_refills_across_old_batch_boundary():
    async def run():
        release = asyncio.Event()
        started = []

        async def one(entry):
            started.append(entry["id"])
            if entry["id"] == "batch1-slow":
                await asyncio.wait_for(release.wait(), 1)
            if entry["id"] == "batch2-first":
                release.set()

        entries = [{"id": name} for name in (
            "batch1-slow", "batch1-fast", "batch2-first", "batch2-second"
        )]
        assert await review_pending_entries(entries, one, 2) == ([], False, [])
        assert started == [entry["id"] for entry in entries]

    asyncio.run(run())
