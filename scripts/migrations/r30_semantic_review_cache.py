"""Reattach semantic evidence only when the complete reviewed input is identical."""

import copy
import hashlib
import json


def rebound_review_name(review):
    """Distinct provenance chains must not collide at one immutable artifact path."""
    content = json.dumps(review, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return f"{review['requestSha256']}-rebound-{hashlib.sha256(content).hexdigest()}.json"


def rebind_review(review, previous_render, current_render, current_pin, request, previous_pin):
    if (review["status"] != "PROPOSED_REQUIRES_MAIN_AGENT_ADJUDICATION"
            or review["requestSha256"] != request):
        raise ValueError("cached review request differs")
    for key in ("documentId", "sourceDocumentId", "target", "pages"):
        if previous_render[key] != current_render[key]:
            raise ValueError(f"cached semantic input changed: {key}")
    if (review["id"] != current_render["documentId"]
            or review["source"] != current_render["sourceDocumentId"]):
        raise ValueError("cached review identity differs")
    result = copy.deepcopy(review)
    result["rendered"] = current_pin
    result["unchangedSemanticInputReuse"] = {"previousReview": previous_pin}
    return result
