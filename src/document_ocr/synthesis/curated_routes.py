"""Explicit source route topology, independent of cargo and lexical generation.

Registered ports establish geography, not a carrier service or sailing schedule.
Node sharing represents source ownership (for example, loading at the via hub).
It is never inferred from an unfilled transshipment caption.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RouteField = Literal[
    "placeOfReceipt",
    "portOfLoading",
    "transshipmentPort",
    "portOfDischarge",
    "placeOfDelivery",
    "finalDestination",
]


class GoodsOriginIdentifier(BaseModel):
    """Reviewed origin-code representation and its owned route location."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["country_alpha2", "unlocode"]
    node: str = "origin"

    def validate_source(self, value: str) -> None:
        pattern = r"[A-Z]{2}" if self.kind == "country_alpha2" else r"[A-Z]{2}[A-Z0-9]{3}"
        if re.fullmatch(pattern, value) is None:
            raise ValueError("goods-origin identifier does not match its declared representation")

    def render(self, *, country_code: str, registry: str, registry_id: str) -> str:
        if self.kind == "country_alpha2":
            return country_code
        if registry != "unlocode_wpi" or not registry_id.startswith(country_code):
            raise ValueError("goods-origin UN/LOCODE requires an owned UN/LOCODE route node")
        return registry_id


class RouteNode(BaseModel):
    """An additional port or locality, with an explicit country dependency."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["port", "locality"] = "port"
    country_from: Literal["origin", "destination", "independent"] = "independent"
    port_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_domain(self) -> RouteNode:
        if self.port_ids and self.kind != "port":
            raise ValueError("port IDs cannot constrain a non-port route node")
        if len(set(self.port_ids)) != len(self.port_ids):
            raise ValueError("route port domain contains duplicate IDs")
        return self


class RouteTopology(BaseModel):
    """Reviewed mapping of every populated source route field to a sampled node.

    Origin and destination are the existing endpoint draws. Extra nodes express
    hubs, inland receipt or onward delivery without collapsing distinct places.
    The source contract independently pins exact printed ownership and labels.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    nodes: dict[str, RouteNode] = Field(default_factory=dict)
    fields: dict[RouteField, str]
    party_nodes: dict[str, str] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description=(
            "Exact populated party paths mapped to route nodes. The party's generated "
            "postal locality belongs to that node's country; it need not be the port itself. "
            "Symbolic sameAs parties inherit their referenced party and are not mapped."
        ),
    )
    location_nodes: dict[Literal["documentPatch.freight.paymentPlace"], str] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Existing non-route location fields with an explicit sampled-node owner.",
    )
    issue_node: str = "origin"
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> RouteTopology:
        if {"origin", "destination"} & self.nodes.keys():
            raise ValueError("route nodes cannot redefine shipment endpoints")
        known = {"origin", "destination", *self.nodes}
        if (
            not self.fields
            or not set(self.fields.values()) <= known
            or not set(self.party_nodes.values()) <= known
            or not set(self.location_nodes.values()) <= known
            or self.issue_node not in known
        ):
            raise ValueError("route topology contains an undefined node")
        if not self.nodes.keys() <= (
            set(self.fields.values())
            | set(self.party_nodes.values())
            | set(self.location_nodes.values())
            | {self.issue_node}
        ):
            raise ValueError("route topology has unused nodes")
        if any(not name.isidentifier() for name in self.nodes):
            raise ValueError("route node names must be identifiers")
        return self

    def validate_source(self, patch: dict) -> None:
        if self.location_nodes and "paymentPlace" not in patch.get("freight", {}):
            raise ValueError("route location nodes must name a populated freight payment place")
        concrete_parties = {
            f"documentPatch.parties.{role}" + (f"[{i}]" if isinstance(value, list) else "")
            for role, value in patch.get("parties", {}).items()
            for i, party in enumerate(value if isinstance(value, list) else [value])
            if "sameAs" not in party
        }
        if not self.party_nodes.keys() <= concrete_parties:
            raise ValueError("route party nodes must name exact populated concrete party paths")
        route = patch.get("route", {})
        if set(route) != set(self.fields):
            raise ValueError("route topology must cover exactly the populated source route fields")
        # Explicit node sharing must agree with source labels. Conversely, do
        # not turn a repeated source port into two independently sampled ports.
        places = {}
        for field, node in self.fields.items():
            value = route[field]
            if "name" not in value:
                raise ValueError("route topology requires source location names")
            name = " ".join(value["name"].upper().split())
            places.setdefault(node, set()).add(name)
        if any(len(names) != 1 for names in places.values()):
            raise ValueError("route topology merges distinct source locations")
        names = [next(iter(values)) for values in places.values()]
        if len(set(names)) != len(names):
            raise ValueError("route topology splits a repeated source location")


def validate_route_values(source: dict, candidate: dict, replacements: dict) -> None:
    """Reject invented/missing fields or altered sampled routes at publication."""
    old = source.get("route", {})
    new = candidate.get("route", {})
    if set(old) != set(new):
        raise ValueError("rendered route field presence differs from source")
    for field, values in old.items():
        if set(values) != set(new[field]):
            raise ValueError("rendered route location components differ from source")
        for key in values:
            expected = replacements[f"documentPatch.route.{field}.{key}"]
            if str(new[field][key]).casefold() != str(expected).casefold():
                raise ValueError(f"rendered route differs from sampled node: {field}.{key}")
