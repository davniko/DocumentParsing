"""Source-certified shared postal localities, independent of company identity."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.country_registry import CountryRegistry


class PostalOccurrence(BaseModel):
    """Exact source text owned by one party's postal region."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str = Field(min_length=1)
    occurrence: int = Field(ge=1)

    def span(self, source: str) -> tuple[int, int]:
        matches = list(re.finditer(re.escape(self.text), source))
        if self.occurrence > len(matches):
            raise ValueError("shared postal evidence occurrence is absent")
        match = matches[self.occurrence - 1]
        return len(source[: match.start()].encode()), len(source[: match.end()].encode())


def _postal_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold()))


def concrete_parties(patch: Mapping[str, Any]) -> dict[str, dict]:
    return {
        f"documentPatch.parties.{role}" + (f"[{i}]" if isinstance(value, list) else ""): party
        for role, value in patch.get("parties", {}).items()
        for i, party in enumerate(value if isinstance(value, list) else [value])
        if party.get("sameAs") is None
    }


class PartyLocalityGroup(BaseModel):
    """Reviewed co-location of concrete parties sharing one postal variable.

    Identity aliases, care-of identities and optional postal suffixes may differ.
    The common source-owned postal component, not company-name similarity, is
    the authority for sharing one sampled locality. Country/route constraints
    remain independent and must agree.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shared_postal_key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    members: dict[str, PostalOccurrence] = Field(min_length=2)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_members(self) -> PartyLocalityGroup:
        if any(
            re.fullmatch(r"documentPatch\.parties\.[A-Za-z]+(?:\[\d+\])?", path) is None
            for path in self.members
        ):
            raise ValueError("locality groups require exact concrete party paths")
        texts = {_postal_tokens(item.text) for item in self.members.values()}
        if len(texts) != 1 or not next(iter(texts)):
            raise ValueError("locality group witnesses must express the same postal component")
        return self

    def validate_source(self, row: Mapping[str, Any], countries: CountryRegistry) -> None:
        source = row["joinedRawText"]
        if sha256_bytes(source.encode()) != self.source_sha256:
            raise ValueError("shared postal source hash differs from certificate")
        parties = concrete_parties(row["target"]["documentPatch"])
        if not self.members.keys() <= parties.keys():
            raise ValueError("locality groups must name present concrete parties")
        country_codes, spans = set(), set()
        for path, evidence in self.members.items():
            span = evidence.span(source)
            if span in spans:
                raise ValueError("shared postal members require their own printed occurrences")
            spans.add(span)
            address = _postal_tokens(parties[path].get("addressLine", ""))
            component = _postal_tokens(evidence.text)
            if not any(
                address[i : i + len(component)] == component
                for i in range(len(address) - len(component) + 1)
            ):
                raise ValueError("shared postal witness is absent from its party address target")
            if "country" in parties[path]:
                country_codes.add(countries.require(parties[path]["country"], field=path))
        if len(country_codes) > 1:
            raise ValueError("shared postal source parties have conflicting countries")

    def validate_blueprint(self, blueprint: Any) -> None:
        """Verify the certificate against independently compiled lexical ownership."""
        for path, evidence in self.members.items():
            start, end = evidence.span(blueprint.source)
            owners = [
                region
                for region in blueprint.regions
                if region.curated_key == self.shared_postal_key
                and region.kind == "postal"
                and region.start <= start < end <= region.end
                and path + ".addressLine" in region.target_paths
            ]
            if len(owners) != 1:
                raise ValueError("locality certificate lacks exact same-party shared postal owner")


def share_party_localities(
    groups: tuple[PartyLocalityGroup, ...],
    patch: Mapping[str, Any],
    sampled: Mapping[str, Any],
    country_assignments: Mapping[str, str],
    signature: Any,
) -> dict[str, Any]:
    """Unify certified components and existing repeated identities transitively.

    The lexicographically first role supplies its existing deterministic draw.
    Independent party draws are untouched. No country is silently replaced.
    """
    parent = {path: path for path in sampled}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(paths: Any) -> None:
        paths = sorted(paths)
        first = find(paths[0])
        for path in paths[1:]:
            other = find(path)
            if other != first:
                parent[other] = first

    for group in groups:
        union(group.members)
    repeated = {}
    for path, party in concrete_parties(patch).items():
        value = signature(party)
        if all(value):
            repeated.setdefault(value, []).append(path)
    for paths in repeated.values():
        union(paths)
    components = {}
    for path in sampled:
        components.setdefault(find(path), []).append(path)
    output = dict(sampled)
    for paths in components.values():
        if len({country_assignments[path] for path in paths}) != 1:
            raise ValueError("shared postal group conflicts with sampled route countries")
        location = sampled[min(paths)]
        output.update({path: location for path in paths})
    return output
