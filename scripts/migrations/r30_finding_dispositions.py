"""Require exact, per-finding human dispositions before scheduling flagged rows."""

from collections import defaultdict

from migrate_r30_synthetic import read
from publish_r30_sources import verify_pin


def load_dispositions(path):
    if path is None:
        return {}
    document = read(path)
    if document["status"] != "MAIN_AGENT_REVIEWED_IDENTIFIED_FINDINGS":
        raise ValueError("individual finding dispositions require main-agent review")
    result = defaultdict(dict)
    for item in document["records"]:
        review = read(verify_pin(item["review"]))
        index = item["finding"]
        if (review["id"] != item["id"] or review["rendered"] != item["rendered"]
                or type(index) is not int or not 0 <= index < len(review["review"]["findings"])
                or not isinstance(item["decision"], str) or not item["decision"].strip()):
            raise ValueError("finding disposition has changed identity, render, index or decision")
        key = (item["review"]["sha256"], index)
        if key in result[item["id"]]:
            raise ValueError("duplicate finding disposition")
        result[item["id"]][key] = item
    return dict(result)


def complete_decision(entry, review_items, dispositions):
    """Return a decision only when every exact review's finding is covered."""
    notes = []
    for item in review_items:
        review = read(verify_pin(item["pin"]))
        if (review["id"] != entry["id"] or review["source"] != entry["source"]
                or review["rendered"] != entry["rendered"]
                or review["status"] != "PROPOSED_REQUIRES_MAIN_AGENT_ADJUDICATION"):
            raise ValueError("finding readiness review is not exact-current")
        findings = review["review"]["findings"]
        if (("findings" in item and item["findings"] != len(findings))
                or ("invalidEvidence" in item and item["invalidEvidence"] != len(review["invalidEvidence"]))):
            raise ValueError("review inventory finding counts differ from pinned evidence")
        if any(type(e["finding"]) is not int or not 0 <= e["finding"] < len(findings)
               for e in review["invalidEvidence"]):
            raise ValueError("invalid evidence references an absent finding")
        for index in range(len(findings)):
            disposition = dispositions.get(entry["id"], {}).get((item["pin"]["sha256"], index))
            if disposition is None:
                return None
            if disposition["rendered"] != entry["rendered"]:
                raise ValueError("finding disposition belongs to an older render")
            notes.append(f"Review {item['pin']['sha256']} finding {index}: {disposition['decision']}")
    return "\n".join(notes) if notes else "No findings; exact-current review verified."
