"""Training publication requires genuine new identities, not a changed file hash."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# TAB, CR and LF are legitimate document layout; C0/C1 device controls are not text.
INVALID_TEXT_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_LEXICAL_SURFACE = re.compile(
    r"\.parties\.[^.]+\.(?:name|address|contactDetails\.(?:contactName|"
    r"(?:phoneNumbers|emailAddresses|websiteUrls)\[\d+\]))$|"
    r"\.cargoGroups\[\d+\]\.(?:description|(?:marksAndNumbers|additionalInformation|"
    r"handlingInstructions)\[\d+\])$|\.forwardingAndExportReferences\[\d+\]$"
)


def validate_unbound_lexical_surfaces(
    *, target: Mapping[str, Any], binding_paths: set[str], rendered: str
) -> None:
    """Auxiliary/shared facts without a direct target binding must still be printed.

    Direct binding adapters already prove segmented and transformed values. This
    additional boundary covers fields conveyed by shared parties or auxiliary
    entities, which previously escaped the binding-only validation loop.
    """
    if INVALID_TEXT_CONTROL.search(rendered):
        raise ValueError(
            "rendered document contains an invalid control character; training input is invalid"
        )
    unbound = {
        path: value
        for path, value in leaves(target).items()
        if isinstance(value, str)
        and _LEXICAL_SURFACE.search(path)
        and path not in binding_paths
        and not any(path.startswith(p + ".") or path.startswith(p + "[") for p in binding_paths)
        and not order_party_reference(path, value)
    }
    if not unbound:
        return
    text = "".join(c for c in rendered.casefold() if c.isalnum())
    missing = [
        path
        for path, value in unbound.items()
        if "".join(c for c in value.casefold() if c.isalnum()) not in text
    ]
    if missing:
        raise ValueError(
            "unbound target values absent from rendered document: " + ", ".join(missing)
        )


def leaves(value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            key: item
            for name, child in value.items()
            for key, item in leaves(child, f"{path}.{name}" if path else name).items()
        }
    if isinstance(value, list):
        return {
            key: item
            for index, child in enumerate(value)
            for key, item in leaves(child, f"{path}[{index}]").items()
        }
    return {path: value}


def order_party_reference(path: str, value: Any) -> bool:
    return bool(
        re.search(r"\.parties\.[^.]+\.name$", path)
        and isinstance(value, str)
        and value.strip().upper()
        in {
            "TO ORDER",
            "TO THE ORDER",
            "TO ORDER OF SHIPPER",
            "TO THE ORDER OF SHIPPER",
            "SAME AS CONSIGNEE",
            "SAME AS SHIPPER",
        }
    )


def require_complete_variation(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> tuple[str, ...]:
    original, generated = leaves(source), leaves(target)
    invalid = [
        p
        for p, value in generated.items()
        if isinstance(value, str) and INVALID_TEXT_CONTROL.search(value)
    ]
    if invalid:
        raise ValueError(
            "synthetic target contains an invalid control character: " + ", ".join(invalid)
        )
    required = tuple(
        p
        for p, v in original.items()
        if re.search(
            r"\.parties\.(?!carrier\.)[^.]+(?:\[\d+\])?\.(name|address)$|"
            r"\.cargoGroups\[\d+\]\.description$|\.containers\[\d+\]\.containerNumber$",
            p,
        )
        and not order_party_reference(p, v)
    )
    unchanged = [
        p
        for p in required
        if p not in generated
        or "".join(c for c in str(original[p]).casefold() if c.isalnum())
        == "".join(c for c in str(generated[p]).casefold() if c.isalnum())
    ]
    if unchanged:
        raise ValueError(
            "required party/cargo/equipment synthesis did not occur: " + ", ".join(unchanged)
        )
    return required
