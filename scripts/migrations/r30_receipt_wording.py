"""Reuse reviewed linguistic values only under an exactly unchanged request."""

import copy
import json


def rebind_wording(wording, specification, previous_pin):
    if wording["status"] != "REVIEWED":
        raise ValueError("wording is not reviewed")
    if wording["documentId"] != specification["documentId"]:
        raise ValueError("wording identity changed")
    if wording["sourceFingerprint"] != specification["sourceFingerprint"]:
        raise ValueError("wording source changed")
    if json.loads(wording["prompt"]) != {
        "requested": specification["requested"], **specification["context"]
    }:
        raise ValueError("wording request or shipment context changed")
    if set(wording["values"]) != set(specification["requested"]):
        raise ValueError("wording owner membership changed")
    result = copy.deepcopy(wording)
    result["candidateSha256"] = specification["candidateSha256"]
    result["unchangedRequestReuse"] = {"previousWording": previous_pin}
    return result
