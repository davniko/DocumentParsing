"""Compose independently prepared DG identities without merging their facts.

Each component retains the existing native HS/form evidence contract. A group
owns no invented UN/HS correspondence. Regulatory occurrences explicitly address
a DG index; reviewed HS-list blocks follow the resulting classification list.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


def components(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    group = request.get("dgComponents")
    if group is None:
        return [request] if request.get("regime") == "dg" else []
    if (
        request.get("regime") != "dg"
        or not isinstance(group, list)
        or len(group) < 2
        or any(not isinstance(row, Mapping) or "dgComponents" in row for row in group)
    ):
        raise ValueError("DG group requires at least two non-nested atomic components")
    return group


def refresh(request: dict[str, Any]) -> None:
    """Project received private context from components, never replace identities."""
    if "dgComponents" not in request:
        return
    rows = components(request)
    request["dangerousGoods"] = [deepcopy(row["dangerousGoods"]) for row in rows]
    request["goods"] = [deepcopy(row["goods"][0]) for row in rows]
    request["goodsForm"] = "; ".join(row["description"] for row in request["goods"])


def prepare(
    sampler,
    *,
    source_document_id: str,
    variant: int,
    identity_count: int,
    seed: int,
    equipment_mode: str,
    requirements: Sequence[Mapping[str, Any]],
    excluded_identifiers: Sequence[str],
    cargo_presentation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if len(requirements) < 2 or identity_count != len(requirements):
        raise ValueError(
            "DG group needs exactly one sampled identity per declaration; "
            "spare HS slots are not extra goods"
        )
    if cargo_presentation is not None:
        raise ValueError("a multi-DG group cannot reuse a single-article presentation contract")
    group = []
    selected = set()
    for index, fields in enumerate(requirements):
        eligible = set(sampler._dg_pool(fields))
        excluded = (set(excluded_identifiers) | selected) & eligible
        # Previously used variants may be recycled; distinct declarations in
        # this shipment may not be silently collapsed after domain exhaustion.
        if not eligible - selected:
            raise ValueError("DG group exhausted distinct eligible registry identities")
        if not eligible - excluded:
            excluded = selected & eligible
        row = sampler.prepare_request(
            source_document_id=f"{source_document_id}/dg/{index}",
            variant=variant,
            regime="dg",
            identity_count=1,
            seed=seed,
            equipment_mode=equipment_mode,
            dg_requirements=fields,
            excluded_identifiers=sorted(excluded),
        )
        selected.add(row["dangerousGoods"]["recordId"])
        group.append(row)
    result = {
        "sourceDocumentId": source_document_id,
        "variant": variant,
        "regime": "dg",
        "equipmentMode": equipment_mode,
        "dgComponents": group,
        "dgRequirements": deepcopy(list(requirements)),
        "sourceCargoUsedForSelection": False,
    }
    refresh(result)
    return result


def validate(sampler, request: Mapping[str, Any], *, require_context: bool) -> None:
    rows = components(request)
    fields = request.get("dgRequirements")
    if not isinstance(fields, list) or len(fields) != len(rows):
        raise ValueError("DG group field requirements do not match its components")
    ids = set()
    for index, row in enumerate(rows):
        if (
            row["sourceDocumentId"] != f"{request['sourceDocumentId']}/dg/{index}"
            or row["variant"] != request["variant"]
            or row["equipmentMode"] != request["equipmentMode"]
            or row["dgRequirements"] != fields[index]
        ):
            raise ValueError("DG component ownership differs from its group")
        sampler.validate_request(row, require_context=require_context)
        ids.add(row["dangerousGoods"]["recordId"])
    if len(ids) != len(rows):
        raise ValueError("DG group repeats a sampled identity")
    if "additionalGoods" in request:
        raise ValueError("additional goods require their own reviewed product ownership")
    expected = deepcopy(dict(request))
    refresh(expected)
    for key in ("goods", "goodsForm", "dangerousGoods"):
        if request[key] != expected[key]:
            raise ValueError("DG group changed the component projection: " + key)


def declarations(value):
    """Return declared ordered entries, preserving the single-entry wire shape."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, list) or not value or any(not isinstance(v, Mapping) for v in value):
        raise ValueError("DG declarations must be a nonempty ordered collection")
    return value
