import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from r30_batch_schedule import reschedule


def fixture():
    plan = {"batchSize": 100, "batches": [[str(i) for i in range(n, n + 100)] for n in range(0, 400, 100)]}
    pending = [i for n, batch in enumerate(plan["batches"], 1) if n not in {1, 3} for i in batch]
    return plan, {i: int(i) % 4 for i in pending}


def test_schedule_preserves_every_identity_and_published_batches():
    plan, priorities = fixture()
    before = copy.deepcopy(plan)
    result = reschedule(plan, {1, 3}, priorities)
    assert plan == before
    assert result["batches"][0] == plan["batches"][0]
    assert result["batches"][2] == plan["batches"][2]
    pending = result["batches"][1] + result["batches"][3]
    assert set(pending) == set(priorities)
    assert [priorities[i] for i in pending] == sorted(priorities.values())
    assert [i for i in pending if priorities[i] == 3] == [i for i in priorities if priorities[i] == 3]


@pytest.mark.parametrize("change", ["omit", "extra", "duplicate", "short", "published", "invalid_priority"])
def test_bad_inventory_never_drops_or_changes_a_record(change):
    plan, priorities = fixture()
    published = {1, 3}
    if change == "omit": priorities.pop("100")
    elif change == "extra": priorities["unexpected"] = 0
    elif change == "duplicate": plan["batches"][1][0] = "0"
    elif change == "short": plan["batches"][1].pop()
    elif change == "published": published.add(5)
    elif change == "invalid_priority": priorities["100"] = True
    with pytest.raises(ValueError): reschedule(plan, published, priorities)
