"""Apply explicit, pinned reviews of omitted facts in retained goods wording.

This is migration evidence, not an extractor. No country detection or instruction
classification runs here: a human review names the exact record, passage and
target. Unchanged records take the no-op path.
"""

import copy
import json
import re

from publish_r30_sources import verify_pin

ABSENT = {"state": "ABSENT"}


def apply_literal_target_correction(rendered, review_pin):
    if review_pin is None:
        return rendered
    review = json.loads(verify_pin(review_pin).read_text())
    return apply_review(rendered, review)


def apply_review(rendered, review):
    if (
        review["status"] != "MAIN_AGENT_REVIEWED_LITERAL_TARGET_CORRECTION"
        or review["documentId"] != rendered["documentId"]
        or review["sourceDocumentId"] != rendered["sourceDocumentId"]
        or not review["reason"].strip()
    ):
        raise ValueError("literal target review has wrong identity or authority")
    match = re.fullmatch(
        r"/goodsItemDetails/(\d+)/(origin|handlingInstructions|dangerousGoods/(\d+)/packingGroupCategory)",
        review["path"],
    )
    if match is None:
        raise ValueError("literal target review exceeds reviewed goods-fact scope")
    index, field = int(match[1]), match[2]
    goods = rendered["target"]["goodsItemDetails"][index]
    owner = goods if match[3] is None else goods["dangerousGoods"][int(match[3])]
    leaf = field if match[3] is None else "packingGroupCategory"
    current = owner.get(leaf, ABSENT)
    if field != "handlingInstructions" and current != review["before"]:
        raise ValueError("literal target review baseline changed")
    quote = review["descriptionQuote"]
    if not isinstance(quote, str) or not quote.strip() or quote not in goods["description"]:
        raise ValueError("reviewed evidence is not in the retained description")
    value = review["after"]
    if field == "origin":
        if (
            not isinstance(value, dict)
            or set(value) != {"name"}
            or not isinstance(value["name"], str)
            or not value["name"]
            or value["name"] not in quote
        ):
            raise ValueError("origin must preserve the reviewed literal country wording")
    elif field == "handlingInstructions":
        before = [] if review["before"] == ABSENT else review["before"]
        retained = [] if current == ABSENT else current
        if (
            not isinstance(before, list)
            or not isinstance(value, list)
            or value[: len(before)] != before
            or len(value) != len(before) + 1
            or not isinstance(value[-1], str)
            or not value[-1]
            or value[-1] not in quote
            or value[-1] in before
        ):
            raise ValueError(
                "handling repair must add one literal instruction without rewriting others"
            )
        # Independently reviewed template clauses may have been appended since
        # this description-owned addition was reviewed. Preserve those clauses,
        # but still require the complete original prefix and literal evidence.
        if (
            not isinstance(retained, list)
            or retained[: len(before)] != before
            or any(not isinstance(item, str) or not item.strip() for item in retained)
            or value[-1] in retained
        ):
            raise ValueError("literal target review baseline changed")
        value = retained + [value[-1]]
    else:
        # This is an explicit per-record review, not an automatic DG extractor.
        # Require the quoted passage to identify just this UN and one packing group.
        identities = set(re.findall(r"\bUN\s*([0-9]{4})\b", quote, re.IGNORECASE))
        groups = set(re.findall(r"\bPG\s+(III|II|I)\b", quote, re.IGNORECASE))
        mapping = {"I": "HIGH_DANGER", "II": "MEDIUM_DANGER", "III": "LOW_DANGER"}
        if (
            review["before"] != ABSENT
            or identities != {owner.get("unNumber")}
            or len(groups) != 1
            or mapping[next(iter(groups)).upper()] != value
        ):
            raise ValueError("packing-group addition lacks unique matching UN/PG evidence")
    if value == review["before"]:
        raise ValueError("literal target review makes no change")
    result = copy.deepcopy(rendered)
    destination = result["target"]["goodsItemDetails"][index]
    if match[3] is not None:
        destination = destination["dangerousGoods"][int(match[3])]
    destination[leaf] = copy.deepcopy(value)
    return result


def correction_index(manifest):
    pin = manifest.get("literalTargetCorrections")
    if pin is None:
        return {}
    index = json.loads(verify_pin(pin).read_text())
    if index["status"] != "MAIN_AGENT_REVIEWED_LITERAL_TARGET_CORRECTION_INDEX":
        raise ValueError("literal target correction index is not reviewed")
    return index["records"]
