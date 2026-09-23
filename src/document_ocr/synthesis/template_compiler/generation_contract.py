"""Training publication requires genuine new identities, not a changed file hash."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .geographic_context import _terminal_country
from .models import AuxiliaryEntity, SemanticBinding

# TAB, CR and LF are legitimate document layout; C0/C1 device controls are not text.
INVALID_TEXT_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_LEXICAL_SURFACE = re.compile(
    r"\.parties\.[^.]+\.(?:name|address|contactDetails\.(?:contactName|"
    r"(?:phoneNumbers|emailAddresses|websiteUrls)\[\d+\]))$|"
    r"\.cargoGroups\[\d+\]\.(?:description|(?:marksAndNumbers|additionalInformation|"
    r"handlingInstructions)\[\d+\])$|\.forwardingAndExportReferences\[\d+\]$"
)
_PARTY_PATH = re.compile(r"^(documentPatch\.parties\.[^.]+)\.")
_PARTY_SURFACE = re.compile(
    r"^documentPatch\.parties\.(?!carrier\.)[^.]+\.(?:name|address|city|country|"
    r"contactDetails\.(?:contactName|(?:phoneNumbers|emailAddresses|websiteUrls)\[\d+\]))$"
)


def party_owned_surfaces(
    bindings: Sequence[SemanticBinding],
    slot_values: Mapping[str, str],
    entities: Sequence[AuxiliaryEntity],
) -> dict[str, tuple[str, ...]]:
    """Collect evidence by explicit party ownership, never by equal source values.

    Join segments within one binding only. Equal names under notify and consignee
    headings do not prove that the unprinted role exists. A genuinely shared
    surface must already declare both target roles in its binding contract.
    """
    owned: dict[str, list[str]] = defaultdict(list)
    linked: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        if entity.relationship == "same_as_target_party":
            if entity.target_party_path is None:
                raise ValueError("linked auxiliary party has no target owner")
            for member in entity.members:
                linked[member.logical_key].add(entity.target_party_path)
    for binding in bindings:
        parties = {
            match[1]
            for path in binding.target_paths
            if (match := _PARTY_PATH.match(path)) is not None
        }
        parties.update(linked.get(binding.logical_key, ()))
        if not parties:
            continue
        values = tuple(slot_values[slot.slot_id] for slot in binding.occurrences)
        for party in parties:
            owned[party].extend(values)
            if len(values) > 1:
                owned[party].append(" ".join(values))
    return {party: tuple(surfaces) for party, surfaces in owned.items()}


def validate_party_evidence(
    *,
    target: Mapping[str, Any],
    binding_paths: set[str],
    party_surfaces: Mapping[str, Sequence[str]],
    country_codes: Mapping[str, str] | None = None,
) -> None:
    """Unbound extraction labels need visible evidence in their own party block.

    Geography may be embedded in an owned address; it need not have a separate
    line. This is text evidence, not geocoding or an address-to-city lookup.
    Direct bindings retain their existing typed/segmented rendering proofs.
    """
    missing = []
    for path, value in leaves(target).items():
        if (
            not isinstance(value, str)
            or not _PARTY_SURFACE.fullmatch(path)
            or path in binding_paths
            or any(path.startswith(p + ".") or path.startswith(p + "[") for p in binding_paths)
            or order_party_reference(path, value)
        ):
            continue
        match = _PARTY_PATH.match(path)
        assert match is not None
        surfaces = party_surfaces.get(match[1], ())
        if path.endswith(".country"):
            compact_country = "".join(c for c in value.casefold() if c.isalnum())
            countries = country_codes if country_codes is not None else {compact_country: value}
            expected = countries.get(compact_country)
            if expected is not None and any(
                _terminal_country(component, countries) == expected
                for text in surfaces
                # Wrapping is presentation: UNITED\nSTATES is one country.
                # Also inspect original line components for an explicitly
                # separated country followed by a locality on the next line.
                for component in (
                    re.sub(r"\s+", " ", text),
                    *re.split(r"[,;]", re.sub(r"\s+", " ", text)),
                    *text.splitlines(),
                )
            ):
                continue
            missing.append(path)
            continue
        if ".contactDetails.phoneNumbers[" in path:
            # A local phone suffix is not proof of a different complete number.
            # Normalize punctuation within complete phone-like spans, not across
            # unrelated address words or between distinct comma-separated values.
            digits = re.sub(r"\D", "", value)
            if digits and any(
                re.sub(r"\D", "", candidate) == digits
                for text in surfaces
                for candidate in re.findall(r"(?<!\w)\+?\d[\d\s()./-]*\d(?!\w)", text)
            ):
                continue
            missing.append(path)
            continue
        # Preserve word boundaries: HAM must not become evidence for HAMBURG.
        words = re.findall(r"[^\W_]+", value.casefold())
        pattern = " " + " ".join(words) + " "
        if not words or not any(
            pattern in " " + " ".join(re.findall(r"[^\W_]+", text.casefold())) + " "
            for text in surfaces
        ):
            missing.append(path)
    if missing:
        raise ValueError("party target lacks role-owned printed evidence: " + ", ".join(missing))


def validate_unbound_lexical_surfaces(
    *,
    target: Mapping[str, Any],
    binding_paths: set[str],
    rendered: str,
    party_surfaces: Mapping[str, Sequence[str]],
    country_codes: Mapping[str, str] | None = None,
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
    validate_party_evidence(
        target=target,
        binding_paths=binding_paths,
        party_surfaces=party_surfaces,
        country_codes=country_codes,
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


def fixed_carrier_role_paths(
    source: Mapping[str, Any], bindings: Sequence[SemanticBinding]
) -> frozenset[str]:
    """Explicit shared static surfaces, not equal-name inference between parties.

    Some bills explicitly identify their carrier as the forwarding agent too.
    Only a certified binding owning BOTH corresponding fields establishes this
    exception. A second static binding or an equal source value is insufficient.
    """
    original = leaves(source)
    result = set()
    prefix = "documentPatch.parties.carrier."
    for binding in bindings:
        if binding.render_mode != "carrier_static":
            continue
        carrier = {p for p in binding.target_paths if p.startswith(prefix)}
        if not carrier:
            continue
        for path in set(binding.target_paths) - carrier:
            match = _PARTY_SURFACE.fullmatch(path)
            owner = _PARTY_PATH.match(path)
            corresponding = prefix + path[owner.end() :] if owner else None
            if (
                match is None
                or corresponding not in carrier
                or corresponding not in original
                or path not in original
                or original[path] != original[corresponding]
            ):
                raise ValueError(
                    "shared carrier-role binding lacks equal corresponding source fields"
                )
            result.add(path)
    return frozenset(result)


def require_complete_variation(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    bindings: Sequence[SemanticBinding] = (),
) -> tuple[str, ...]:
    original, generated = leaves(source), leaves(target)
    fixed_roles = fixed_carrier_role_paths(source, bindings)
    fixed_paths = fixed_roles | {
        p for p in original if p.startswith("documentPatch.parties.carrier.")
    }
    if any(p not in generated or generated[p] != original[p] for p in fixed_paths):
        raise ValueError("fixed carrier-role identity changed during synthesis")
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
        and p not in fixed_roles
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
