"""Fail closed on unowned source-only route and customs claims after rendering.

Only claims whose source owner is proved by the source target are checked. An
optional claim may be wholly omitted; a retained claim must agree with the
sampled physical endpoint at every printed occurrence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_FREE_TIME_PORT = re.compile(r"(?im)^(?P<days>\d+) DAYS DEM+UR+AGE FREETIME IN (?P<place>[^\n]+)$")
_FREE_TIME_COUNTRY = re.compile(r"(?im)Free Time in\s*\n\s*(?P<country>[A-Za-z ]+):\s*\d+\s+days")
_LINER = re.compile(r"(?im)^LINER IN (?P<origin>[^\n]+?) - FREE OUT (?P<port>[^\n]+)$")
_ALEXANDRIA_TERMINAL = re.compile(
    r"(?i)Any transportation expss of containers at Alexandria terminal"
)
_PORT_CONTAINERS_TERMINAL = re.compile(
    r"(?i)ALL LANDING, DISCHARGING AD RELOADING OPERATIONS TO BE EFFECTED BY THE "
    r"(?P<place>[^\n]+?) PORT CONTAINERS TERMINAL"
)
_EGYPT_AGENT = re.compile(
    r"(?im)^NAME AND FULL ADDRESS OF SHIPPING AGENT IN EGYPT\r?\n(?P<place>[^\n]+)"
)
_EGYPTIAN_LAWS = re.compile(r"(?i)AS PER EGYPTIAN LAWS")
_ACID = re.compile(r"(?im)^\s*ACID\s*[:\u00b7-]\s*\d{19}\s*$")
_TOKEN = re.compile(r"[a-z0-9]+")
_EGYPT = frozenset({"egypt", "eg", "egy", "arab republic of egypt"})


def _normal(value: str) -> str:
    return " ".join(_TOKEN.findall(value.casefold()))


def _same_port(declared: str, endpoint: str) -> bool:
    observed = _normal(declared)
    actual = _normal(endpoint)
    if not observed or not actual:
        return False
    return f" {observed} " in f" {actual} " or f" {actual} " in f" {observed} "


def _field(target: Mapping[str, Any], role: str, field: str) -> str | None:
    route = target.get("documentPatch", {}).get("route", {})
    endpoint = route.get(role, {})
    value = endpoint.get(field)
    return value if isinstance(value, str) and value.strip() else None


def _consignee_country(target: Mapping[str, Any]) -> str | None:
    party = target.get("documentPatch", {}).get("parties", {}).get("consignee", {})
    value = party.get("country")
    return value if isinstance(value, str) and value.strip() else None


def _liner_owners(template: Any, source_line: str) -> tuple[str, str] | None:
    """Return both physical owners only when compilation explicitly records them."""
    origin_roles = {"portOfLoading", "placeOfReceipt"}
    matched = [
        binding
        for binding in template.bindings
        if any(slot.source_text.strip() == source_line for slot in binding.occurrences)
    ]
    if len(matched) != 1:
        return None
    paths = (*matched[0].target_paths, *matched[0].dependency_paths)
    discharge = any(path.startswith("documentPatch.route.portOfDischarge") for path in paths)
    origins = {
        role
        for role in origin_roles
        if any(path.startswith(f"documentPatch.route.{role}") for path in paths)
    }
    if not discharge or len(origins) != 1:
        return None
    return origins.pop(), "portOfDischarge"


def validate(
    *,
    source: bytes,
    rendered: bytes,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    template: Any,
) -> None:
    """Validate every repeated source-owned claim, without guessing missing owners."""
    source_text = source.decode("utf-8")
    output = rendered.decode("utf-8")
    source_port = _field(source_target, "portOfDischarge", "name")
    target_port = _field(target, "portOfDischarge", "name")
    if source_port and target_port:
        source_claims = tuple(
            match
            for match in _FREE_TIME_PORT.finditer(source_text)
            if _same_port(match["place"], source_port)
        )
        if source_claims:
            realized = tuple(_FREE_TIME_PORT.finditer(output))
            if len(realized) not in {0, len(source_claims)}:
                raise ValueError("source-only free-time port claim was only partly realized")
            if any(not _same_port(match["place"], target_port) for match in realized):
                raise ValueError("source-only free-time port contradicts sampled discharge")

        source_liner = tuple(
            match
            for match in _LINER.finditer(source_text)
            if _same_port(match["port"], source_port)
        )
        if source_liner and not _same_port(source_port, target_port):
            realized_liner = tuple(_LINER.finditer(output))
            if len(realized_liner) not in {0, len(source_liner)}:
                raise ValueError("source-only liner route was only partly realized")
            if realized_liner:
                # The printed loading/discharge pair can itself be proven from
                # explicit target endpoints. A compiled auxiliary dependency is
                # needed only when the loading endpoint is not task-labelled.
                origin = _field(target, "portOfLoading", "name")
                if origin is None:
                    owners = {_liner_owners(template, match.group(0)) for match in source_liner}
                    if len(owners) != 1 or None in owners:
                        raise ValueError(
                            "retained liner route has no compiled two-endpoint contract"
                        )
                    owner = owners.pop()
                    if owner is None:
                        raise ValueError("retained liner route has no compiled origin owner")
                    origin = _field(target, owner[0], "name")
                if not origin or any(
                    not _same_port(match["origin"], origin)
                    or not _same_port(match["port"], target_port)
                    for match in realized_liner
                ):
                    raise ValueError("source-only liner route contradicts sampled endpoints")

        if _same_port("Alexandria", source_port) and not _same_port("Alexandria", target_port):
            if _ALEXANDRIA_TERMINAL.search(source_text) and _ALEXANDRIA_TERMINAL.search(output):
                raise ValueError("source-only Alexandria terminal clause contradicts discharge")
            source_terminal = _PORT_CONTAINERS_TERMINAL.search(source_text)
            if (
                source_terminal
                and _same_port(source_terminal["place"], source_port)
                and _PORT_CONTAINERS_TERMINAL.search(output)
            ):
                raise ValueError("source-only port-terminal facility was not omitted")

    source_agent = _EGYPT_AGENT.search(source_text)
    rendered_agent = _EGYPT_AGENT.search(output)
    if (
        source_agent
        and rendered_agent
        and _normal(source_agent["place"]) != _normal(rendered_agent["place"])
    ):
        raise ValueError("Egypt-specific shipping-agent heading has changed agent locality")

    source_country = _field(source_target, "portOfDischarge", "country")
    target_country = _field(target, "portOfDischarge", "country")
    if source_country and target_country:
        source_country_claims = tuple(
            match
            for match in _FREE_TIME_COUNTRY.finditer(source_text)
            if _normal(match["country"]) == _normal(source_country)
        )
        if source_country_claims:
            realized_country = tuple(_FREE_TIME_COUNTRY.finditer(output))
            if len(realized_country) not in {0, len(source_country_claims)}:
                raise ValueError("source-only free-time country was only partly realized")
            if any(
                _normal(match["country"]) != _normal(target_country) for match in realized_country
            ):
                raise ValueError("source-only free-time country contradicts sampled discharge")

    source_importer = _consignee_country(source_target)
    target_importer = _consignee_country(target)
    source_egypt = bool(source_country and _normal(source_country) in _EGYPT)
    foreign_route = bool(target_country and _normal(target_country) not in _EGYPT)
    foreign_importer = bool(target_importer and _normal(target_importer) not in _EGYPT)
    if (
        source_egypt
        and foreign_route
        and foreign_importer
        and source_importer
        and _normal(source_importer) in _EGYPT
        and _ACID.search(source_text)
        and _ACID.search(output)
    ):
        raise ValueError("Egypt ACID declaration contradicts foreign route and importer")
    if _EGYPTIAN_LAWS.search(source_text) and _EGYPTIAN_LAWS.search(output):
        if source_egypt and foreign_route:
            raise ValueError("Egyptian-law clause contradicts foreign discharge")
        if (
            source_port
            and target_port
            and not _same_port(source_port, target_port)
            and not target_country
            and foreign_importer
        ):
            raise ValueError("Egyptian-law clause has no sampled jurisdiction owner")
