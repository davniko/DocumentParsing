"""Training publication requires genuine new identities, not a changed file hash."""

from __future__ import annotations

import bisect
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
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
_SEAL_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.sealNumbers\[([0-9]+)\]$")
_PARTY_AFFILIATION = re.compile(
    r"(?i)(?:\bon\s+behal(?:f|df)\s+of\b|\bas\s+agent\s+of\b|"
    r"\bc\s*/\s*o\b|\bo\s*/\s*b\b)"
)
_PAGE_MARKER = re.compile(rb"(?m)^--- PAGE [0-9]+ ---")
_PARTY_CONTINUATION_MARKER = re.compile(r"(?:^|[\s,])(?:FW|CN|NP)>", re.IGNORECASE)


def validate_rendered_party_boundaries(target: Mapping[str, Any], rendered: str) -> None:
    """Reject fused name/address text even when both complete facts are present."""

    patch = target.get("documentPatch")
    parties = patch.get("parties") if isinstance(patch, Mapping) else None
    if not isinstance(parties, Mapping):
        return
    for role, party in parties.items():
        if not isinstance(party, Mapping):
            continue
        name, address = party.get("name"), party.get("address")
        if not isinstance(name, str) or not isinstance(address, str):
            continue
        name_tokens = re.findall(r"[A-Za-z0-9]+", name)
        address_tokens = re.findall(r"[A-Za-z0-9]+", address)
        if not name_tokens or not address_tokens:
            continue
        name_tail = name_tokens[-2:] if len(name_tokens[-1]) >= 3 else name_tokens[-3:]
        prior = r"\W+".join(re.escape(token) for token in name_tail[:-1])
        final = re.escape(name_tail[-1])
        first = re.escape(address_tokens[0])
        prefix = rf"(?<!\w){prior}\W+" if prior else r"(?<!\w)"
        pattern = re.compile(rf"{prefix}{final}[^\w\s]{{0,3}}{first}\b", re.IGNORECASE)
        if pattern.search(rendered):
            raise ValueError(f"rendered party name and address lack a separator: {role}")


def validate_repeated_agent_party_pages(
    *,
    source: bytes,
    rendered: bytes,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    bindings: Sequence[SemanticBinding],
) -> None:
    """A source-complete party page may not become a partial generated copy.

    This applies only to mutable, agent-edited party names/addresses. Source
    fragments that genuinely continue onto another page are not treated as
    independent copies, and unaltered source surfaces remain untouched.
    """

    candidates = [
        binding
        for binding in bindings
        if binding.realization.mode == "agent_required"
        and len(binding.occurrences) > 1
        and len(binding.target_paths) == 1
        and _PARTY_PATH.match(binding.target_paths[0])
        and binding.target_paths[0].endswith((".name", ".address"))
    ]
    if not candidates:
        return
    source_values = leaves(source_target)
    target_values = leaves(target)
    starts = [match.start() for match in _PAGE_MARKER.finditer(source)]
    rendered_starts = [match.start() for match in _PAGE_MARKER.finditer(rendered)]
    if not starts or starts[0] != 0 or len(starts) != len(rendered_starts):
        raise ValueError("repeated-party proof requires matching source and rendered pages")
    source_pages = [
        source[start:end].decode("utf-8")
        for start, end in zip(starts, [*starts[1:], len(source)], strict=True)
    ]
    rendered_pages = [
        rendered[start:end].decode("utf-8")
        for start, end in zip(
            rendered_starts, [*rendered_starts[1:], len(rendered)], strict=True
        )
    ]

    def comparable(value: str) -> str:
        folded = unicodedata.normalize("NFKD", value.casefold().replace("\u0131", "i"))
        return "".join(
            char for char in folded if char.isalnum() and not unicodedata.combining(char)
        )

    for binding in candidates:
        path = binding.target_paths[0]
        before = source_values.get(path)
        after = target_values.get(path)
        if not isinstance(before, str) or not isinstance(after, str) or before == after:
            continue
        expected_source = comparable(before)
        expected_target = comparable(after)
        if not expected_source or not expected_target:
            raise ValueError(f"repeated party scalar is empty: {path}")
        owned_pages = {
            bisect.bisect_right(starts, slot.byte_start) - 1
            for slot in binding.occurrences
        }
        for page in owned_pages:
            source_has_value = expected_source in comparable(source_pages[page])
            rendered_has_value = expected_target in comparable(rendered_pages[page])
            if source_has_value and not rendered_has_value:
                raise ValueError(
                    f"repeated party page {page + 1} lacks its generated target: {path}"
                )


def validate_compiled_party_contract(
    *,
    raw: bytes,
    source_target: Mapping[str, Any],
    bindings: Sequence[SemanticBinding],
    entities: Sequence[AuxiliaryEntity],
) -> None:
    """Reject party contracts that can independently change one labelled identity.

    A printed affiliation inside ``party.name`` belongs to that label, not to
    an independently sampled source-only entity. Phone and email slots must
    have a literal separator; otherwise valid replacements can form a single
    OCR token such as ``034840282name@example.com``. Name/address boundaries
    require whitespace even when source punctuation lies between their slots.
    """
    by_key = {binding.logical_key: binding for binding in bindings}
    if len(by_key) != len(bindings):
        raise ValueError("compiled party contract repeats a logical binding key")
    for binding in bindings:
        if (
            binding.group_kind != "party"
            or binding.value_kind != "address"
            or binding.realization.mode != "segmented_surface"
            or len(binding.occurrences) < 2
            or not all(slot.source_text.strip().isdigit() for slot in binding.occurrences)
        ):
            continue
        first = min(slot.byte_start for slot in binding.occurrences)
        last = max(slot.byte_end for slot in binding.occurrences)
        if any(
            other.group_key == binding.group_key
            and any(path.endswith(".city") for path in other.target_paths)
            and any(first < slot.byte_start < last for slot in other.occurrences)
            for other in bindings
        ):
            raise ValueError(
                "numeric-only address slots interleave a separately generated city; "
                "the street/postal components cannot be independently rendered: "
                + binding.logical_key
            )
    ordered = sorted(
        ((slot, binding) for binding in bindings for slot in binding.occurrences),
        key=lambda row: row[0].byte_start,
    )
    for (left_slot, left), (right_slot, right) in pairwise(ordered):
        if (
            left.group_key == right.group_key
            and left.group_kind == right.group_kind == "party"
            and any(path.endswith(".name") for path in left.target_paths)
            and any(path.endswith(".address") for path in right.target_paths)
            and not any(
                char.isspace()
                for char in raw[left_slot.byte_end : right_slot.byte_start].decode("utf-8")
            )
        ):
            raise ValueError(
                "adjacent party name/address slots lack a literal separator: "
                + left.logical_key
                + " / "
                + right.logical_key
            )
        if (
            left_slot.byte_end == right_slot.byte_start
            and left.group_key == right.group_key
            and left.group_kind == "party"
            and right.group_kind == "party"
            and left.value_kind == "phone"
            and right.value_kind == "email"
            and left_slot.source_text[-1].isdigit()
            and right_slot.source_text[0].isalnum()
        ):
            raise ValueError(
                "adjacent party phone/email slots lack a literal separator: "
                + left.logical_key
                + " / "
                + right.logical_key
            )
    source_values = leaves(source_target)
    for path, value in source_values.items():
        if (
            _PARTY_SURFACE.fullmatch(path) is not None
            and path.endswith(".address")
            and isinstance(value, str)
            and _PARTY_CONTINUATION_MARKER.search(value)
        ):
            raise ValueError("party address includes an OCR continuation marker: " + path)
    owners: dict[str, list[SemanticBinding]] = defaultdict(list)
    for binding in bindings:
        for path in binding.target_paths:
            owners[path].append(binding)
    for path, name in source_values.items():
        if (
            not path.startswith("documentPatch.parties.")
            or not path.endswith(".name")
            or not isinstance(name, str)
            or not name.split()
            or not owners.get(path)
        ):
            continue
        suffix = "".join(char for char in name.split()[-1].casefold() if char.isalnum())
        if not 3 <= len(suffix) <= 16:
            continue
        name_surfaces = "".join(
            char
            for binding in owners[path]
            for slot in binding.occurrences
            for char in slot.source_text.casefold()
            if char.isalnum()
        )
        if suffix in name_surfaces:
            continue
        for binding in owners.get(path.removesuffix(".name") + ".address", ()):
            if any(
                "".join(char for char in slot.source_text.casefold() if char.isalnum()).startswith(
                    suffix
                )
                for slot in binding.occurrences
            ):
                raise ValueError("party-name suffix is swallowed by its address binding: " + path)
    for entity in entities:
        party_path = entity.target_party_path
        if entity.relationship != "same_as_target_party" or party_path is None:
            continue
        name = source_values.get(party_path + ".name")
        if not isinstance(name, str) or _PARTY_AFFILIATION.search(name) is None:
            continue
        normalized_name = "".join(char for char in name.casefold() if char.isalnum())
        for member in entity.members:
            if member.field not in {"name", "other"}:
                continue
            binding = by_key[member.logical_key]
            if binding.target_paths or binding.render_mode == "literal_static":
                continue
            if any(
                len(
                    surface := "".join(
                        char for char in slot.source_text.casefold() if char.isalnum()
                    )
                )
                >= 6
                and surface in normalized_name
                for slot in binding.occurrences
            ):
                raise ValueError(
                    "labelled party affiliation is independently generated: "
                    + party_path
                    + ".name / "
                    + binding.logical_key
                )


def validate_seal_realization(
    *,
    target: Mapping[str, Any],
    bindings: Sequence[SemanticBinding],
    slot_values: Mapping[str, str] | None = None,
    rendered: str | None = None,
) -> None:
    """Every seal leaf must be rendered by its own container's target binding.

    A parent container/count binding and a coincidental global string match are
    not evidence for a child seal. Check every physical occurrence so repeated
    pages cannot retain a stale independently generated seal. The compiled
    renderer separately proves that every checked slot reaches final text.
    Without slot values this is also a cheap preflight before provider calls.
    Offline audits may additionally check the complete rendered text.
    """
    if rendered is not None and slot_values is None:
        raise ValueError("rendered seal evidence requires slot values")
    patch = target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return
    containers = patch.get("containers")
    if containers is None:
        return
    if not isinstance(containers, list):
        raise ValueError("documentPatch.containers must be a list")
    seal_rows = [
        (container_index, seal_index, value)
        for container_index, container in enumerate(containers)
        if isinstance(container, Mapping)
        for seal_index, value in enumerate(container.get("sealNumbers") or ())
    ]
    if not seal_rows:
        return
    rendered_compact = (
        "".join(char for char in rendered.casefold() if char.isalnum())
        if rendered is not None
        else None
    )
    owners: dict[str, list[SemanticBinding]] = defaultdict(list)
    for binding in bindings:
        for path in binding.target_paths:
            if _SEAL_PATH.fullmatch(path):
                owners[path].append(binding)
    for container_index, seal_index, value in seal_rows:
        path = f"documentPatch.containers[{container_index}].sealNumbers[{seal_index}]"
        candidates = owners.get(path, ())
        if len(candidates) != 1:
            raise ValueError(f"seal target requires exactly one leaf binding: {path}")
        owner = candidates[0]
        if owner.group_kind != "equipment" or owner.group_key != f"container:{container_index}":
            raise ValueError(f"seal binding has wrong container ownership: {path}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"seal target must be a nonempty string: {path}")
        expected = "".join(char for char in value.casefold() if char.isalnum())
        if not expected:
            raise ValueError(f"seal target has no identifier characters: {path}")
        if rendered_compact is not None and expected not in rendered_compact:
            raise ValueError(f"rendered document lacks target seal: {path}")
        if slot_values is None:
            continue
        observed_parts: list[str] = []
        for occurrence in owner.occurrences:
            replacement = slot_values.get(occurrence.slot_id)
            if replacement is None:
                raise ValueError(f"seal binding has no slot output: {path} {occurrence.slot_id}")
            observed_parts.append(
                "".join(char for char in replacement.casefold() if char.isalnum())
            )
        mode = getattr(getattr(owner, "realization", None), "mode", None)
        if mode in {"single_surface", "repeated_surface", None} or len(observed_parts) == 1:
            # A complete seal slot cannot silently absorb an adjacent OCR
            # character. Source separators, including mistyped ones, must
            # remain literal regions outside the identifier binding.
            if all(observed == expected for observed in observed_parts):
                continue
        elif mode in {"segmented_surface", "agent_required"}:
            fragments_match = all(
                observed and (observed in expected or expected in observed)
                for observed in observed_parts
            )
            complete = any(expected in observed for observed in observed_parts) or (
                "".join(observed_parts) == expected
            )
            if fragments_match and complete:
                continue
        raise ValueError(f"seal slot does not realize its container target: {path}")


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
