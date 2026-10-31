import importlib.util
from collections import Counter
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/synthesis/refresh_r28_multirow.py"
    spec = importlib.util.spec_from_file_location("r28_multirow_refresh", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def groups():
    return {
        f"source-{i}": [
            {
                "documentId": f"record-{i}-{j}",
                "candidatePath": f"candidate-{i}-{j}",
                "donorDocumentId": f"donor-{i}" if j < 6 else None,
            }
            for j in range(4 if i < 15 else 8)
        ]
        for i in range(300)
    }


def test_reproducible_exact_quotas_and_no_repeated_records():
    script = module()
    original = groups()
    batches = script.select_batches(original, seed=2026103115)
    assert script.select_batches(original, seed=2026103115) == batches
    assert original == groups()
    records = [row for batch in batches for row in batch["records"]]
    assert len(batches) == 15
    assert len({row["documentId"] for row in records}) == 1500
    assert len({row["sourceDocumentId"] for row in records}) == 300
    assert Counter(row["packageRows"] for row in records) == {2: 1200, 3: 300}
    for batch in batches:
        assert len(batch["records"]) == 100
        assert Counter(row["packageRows"] for row in batch["records"]) == {2: 80, 3: 20}
        assert all(row["previousDonor"] is not None for row in batch["records"])
        for source, count in batch["sourceCounts"].items():
            assert 3 <= count <= min(6, len(original[source]))
            assert [
                r["variant"] for r in batch["records"] if r["sourceDocumentId"] == source
            ] == list(range(1, count + 1))


def test_insufficient_sources_or_records_are_not_silently_replaced():
    script = module()
    with pytest.raises(ValueError, match="fewer than 300"):
        script.select_batches({"one": groups()["source-1"]}, seed=1)
    with pytest.raises(ValueError, match="cannot supply 100"):
        script.select_batches({sid: rows[:3] for sid, rows in groups().items()}, seed=1)
