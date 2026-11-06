"""Reorder unfinished work without changing membership or published batches."""

import copy


def reschedule(plan, published_batches, priorities):
    size = plan["batchSize"]
    batches = plan["batches"]
    if size != 100 or any(len(batch) != size for batch in batches):
        raise ValueError("schedule requires complete 100-record batches")
    identities = [identity for batch in batches for identity in batch]
    if len(set(identities)) != len(identities):
        raise ValueError("schedule contains duplicate identities")
    published = set(published_batches)
    if any(type(n) is not int or not 1 <= n <= len(batches) for n in published):
        raise ValueError("published batch is outside schedule")
    pending = [identity for n, batch in enumerate(batches, 1) if n not in published for identity in batch]
    if set(priorities) != set(pending):
        raise ValueError("priority inventory must cover every pending identity exactly")
    if any(type(value) is not int or value not in range(4) for value in priorities.values()):
        raise ValueError("invalid explicit readiness priority")
    # Stable ordering within readiness classes keeps the earlier schedule's
    # relative order. Held records remain in the same complete membership.
    ordered = sorted(pending, key=priorities.__getitem__)
    result = copy.deepcopy(plan)
    cursor = 0
    for n in range(1, len(batches) + 1):
        if n not in published:
            result["batches"][n - 1] = ordered[cursor : cursor + size]
            cursor += size
    assert cursor == len(pending)
    assert set(identity for batch in result["batches"] for identity in batch) == set(identities)
    assert all(result["batches"][n - 1] == batches[n - 1] for n in published)
    return result
