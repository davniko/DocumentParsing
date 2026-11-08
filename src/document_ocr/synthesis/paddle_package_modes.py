"""Admission checks for explicit scalar and compound package presentations."""


def validate_package_mode_owners(source_id, contract, template, transfer):
    """Check every declared compound mode, not merely historical replay edits."""
    slots = {s["owner"]: s for s in transfer["slots"]}
    overrides = template["overlay"]["overrides"]
    absent = template["overlay"].get("absentEdits", [])
    for i, declaration in enumerate(contract["declarations"]):
        key = f"declared-packages:{i}"
        start, end = declaration["start"], declaration["end"]
        owner = f"{key}@{start}:{end}"
        explicit = [
            r
            for r in overrides
            if r["key"] == key
            and r.get("byteStart", start) == start
            and r.get("byteEnd", end) == end
        ]
        if len(explicit) == 1:
            continue
        if owner in slots and slots[owner]["status"] == "exact_measured_owner_candidate":
            continue
        member = {"key": key, "byteStart": start, "before": declaration["source"]}
        # A compound owner can be absent only through a reviewed exact disposition,
        # including alternate scalar/compound topology for the same missing line.
        if any(
            all(d.get(k) == v for k, v in member.items())
            or any(member in branch for branch in d.get("alternatives", []))
            for d in absent
        ):
            continue
        raise ValueError(
            f"native package mode lacks a reviewed complete owner: {source_id}: {owner}"
        )
