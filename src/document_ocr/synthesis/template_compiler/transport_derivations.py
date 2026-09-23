"""Explicit independent-leg vessel identities, sampled once from the pinned fleet.

The carrier and extraction-label topology remain unchanged. These fictional
shipment scenarios do not assert that a named vessel actually sails a schedule.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from document_ocr.synthesis.generators import DeterministicStream

DERIVATIONS = frozenset({"sampled_auxiliary_vessel"})

_INDEPENDENT_CAPTION = re.compile(r"(?i)(?:FEEDER|FIRST\s+LEG|INTENDED)\s+VESSEL\s*[:*]?\s*$")


def normalize_explicit_names(
    *, raw: str, drafts: Sequence[Any], source_target: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Type only whole separate vessel names with a caption at every occurrence.

    Do not infer a name from a combined vessel/voyage/flag field or use equal
    text to turn an uncaptioned location into an independent transport leg.
    Unresolved cases remain unchanged for explicit contract review.
    """
    main = source_target["documentPatch"].get("transport", {}).get("vesselName")
    if not isinstance(main, str) or not main.strip():
        return tuple(drafts)
    normalized_main = re.sub(r"[^A-Z0-9]", "", main.upper())
    groups: dict[str, list[Any]] = {}
    for draft in drafts:
        groups.setdefault(draft.logical_key, []).append(draft)
    accepted = set()
    for key, rows in groups.items():
        if not all(
            d.group_kind in {"transport", "route"}
            and d.value_kind == "equipment"
            and not d.target_paths
            and not d.dependency_paths
            and not d.dependency_bindings
            and d.derivation is None
            and d.render_mode not in {"literal_static", "carrier_static"}
            and re.fullmatch(r"[A-Za-z][A-Za-z .'-]*[A-Za-z]", d.source_text)
            and _INDEPENDENT_CAPTION.search(raw[max(0, d.char_start - 80) : d.char_start])
            and raw[d.char_end : d.char_end + 1] in {"", " ", "\n", "\r"}
            for d in rows
        ):
            continue
        names = {re.sub(r"[^A-Z0-9]", "", d.source_text.upper()) for d in rows}
        if len(names) == 1 and normalized_main not in names:
            accepted.add(key)
    return tuple(
        replace(
            d,
            render_mode="deterministic_derived",
            group_kind="transport",
            derivation="sampled_auxiliary_vessel",
            dependency_paths=("documentPatch.transport.vesselName",),
            render_policy="natural_text",
            rationale=(
                "Explicit separate-vessel caption at every occurrence; "
                "distinct from the target-backed main vessel."
            ),
        )
        if d.logical_key in accepted
        else d
        for d in drafts
    )


def validate(binding: Any) -> None:
    if (
        binding.derivation not in DERIVATIONS
        or binding.target_paths
        or binding.dependency_paths
        not in {
            ("documentPatch.transport.vesselName",),
            ("documentPatch.transport",),
        }
        or binding.dependency_bindings
        or binding.render_mode != "deterministic_derived"
        or binding.group_kind != "transport"
        or binding.value_kind != "equipment"
    ):
        raise ValueError("independent vessel requires an explicit source-only transport contract")


def _main_vessel(bindings: Sequence[Any], target: Mapping[str, Any]) -> str | None:
    """A private leg does not require inventing an absent main-vessel label."""
    owner = target["documentPatch"].get("transport")
    if not isinstance(owner, Mapping) or not owner:
        raise ValueError(
            "independent vessel contract requires a target-backed main transport owner"
        )
    main = owner.get("vesselName")
    if main is not None and (not isinstance(main, str) or not main.strip()):
        raise ValueError("target-backed main vessel is invalid")
    if main is None and any(
        b.dependency_paths == ("documentPatch.transport.vesselName",) for b in bindings
    ):
        raise ValueError("independent vessel contract requires its target-backed main vessel")
    return main


def validate_source(bindings: Sequence[Any], target: Mapping[str, Any]) -> None:
    required = [b for b in bindings if b.derivation in DERIVATIONS]
    if not required:
        return
    main = _main_vessel(required, target)
    normalized_main = re.sub(r"[^A-Z0-9]", "", main.upper()) if main is not None else None
    for binding in required:
        validate(binding)
        names = {re.sub(r"[^A-Z0-9]", "", slot.source_text.upper()) for slot in binding.occurrences}
        if len(names) != 1 or normalized_main in names:
            raise ValueError("independent vessel source names are inconsistent or alias the main")


def values(
    bindings: Sequence[Any],
    *,
    vessel_names: Sequence[str],
    target: Mapping[str, Any],
    seed: int,
    sample_id: str,
) -> dict[str, str]:
    required = [b for b in bindings if b.derivation in DERIVATIONS]
    if not required:
        return {}

    def normalized(value: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", value.upper())

    main = _main_vessel(required, target)
    excluded = ({normalized(main)} if main is not None else set()) | {
        normalized(s.source_text) for b in required for s in b.occurrences
    }
    candidates = tuple(sorted({n for n in vessel_names if normalized(n) not in excluded}))
    if len(candidates) < len(required):
        raise ValueError("pinned vessel registry lacks distinct independent-leg identities")
    stream = DeterministicStream(seed, "compiled-independent-vessels-v1", sample_id)
    result = {}
    for binding in sorted(required, key=lambda b: b.logical_key):
        validate(binding)
        choices = tuple(n for n in candidates if normalized(n) not in excluded)
        value = choices[stream.derive(binding.logical_key).randbelow(len(choices))]
        result[binding.logical_key] = value
        excluded.add(normalized(value))
    return result
