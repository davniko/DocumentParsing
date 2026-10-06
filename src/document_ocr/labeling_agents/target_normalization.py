"""Final, loss-bounded text normalization for newly annotated V7 targets.

Human-readable text has one casing convention. Identifiers, contact endpoints,
category tokens and units remain byte-for-byte unchanged. This is deliberately
separate from schema validation and evaluation: it must not conceal prediction
errors or reinterpret historical labels.
"""

from __future__ import annotations

from typing import Any

_LOCATIONS = (
    "placeOfIssue",
    "freight.paymentPlace",
    *(
        f"route.{role}"
        for role in (
            "placeOfReceipt",
            "portOfLoading",
            "transshipmentPort",
            "portOfDischarge",
            "placeOfDelivery",
            "finalDestination",
        )
    ),
)
_PARTIES = (
    "shipper",
    "consignee",
    "notifyParties[]",
    "carrier",
    "forwardingAgent",
    "deliveryAgent",
    "consolidator",
)
UPPERCASE_FIELDS = frozenset(
    [f"{location}.{field}" for location in _LOCATIONS for field in ("name", "country")]
    + [
        f"parties.{party}.{field}"
        for party in _PARTIES
        for field in ("name", "addressLine", "country", "contactDetails.contactName")
    ]
    + [
        "transport.vesselName",
        "transport.vesselFlagCountry",
        "containerInformation[].typeDescription",
        "goodsItemDetails[].description",
        "goodsItemDetails[].handlingInstructions[]",
        "goodsItemDetails[].origin.name",
        "goodsItemDetails[].numberAndTypeOfPackages[].typeOfPackages",
    ]
)


def normalize_target_casing(target: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy a V7 target, uppercasing only the explicit human-readable field set.

    Receipts contain every changed leaf. No source text, key, list order, spelling,
    punctuation, or numeric value is altered. Unicode case expansions (e.g. ß
    to SS) follow Python's Unicode uppercase mapping; this is casing, not ASCII
    transliteration. The operation is idempotent and never mutates its input.
    """
    if target.get("schemaVersion") != "7.0.0" or not isinstance(target.get("documentPatch"), dict):
        raise ValueError("casing normalization requires a V7 target envelope")
    changes: list[dict[str, Any]] = []

    def visit(value: Any, field: str, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: visit(child, f"{field}.{key}".lstrip("."), f"{path}.{key}")
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [visit(child, field + "[]", f"{path}[{i}]") for i, child in enumerate(value)]
        if isinstance(value, str) and field in UPPERCASE_FIELDS:
            normalized = value.upper()
            if normalized != value:
                changes.append({"path": path, "before": value, "after": normalized})
            return normalized
        return value

    result = {
        **target,
        "documentPatch": visit(target["documentPatch"], "", "documentPatch"),
    }
    return result, changes
