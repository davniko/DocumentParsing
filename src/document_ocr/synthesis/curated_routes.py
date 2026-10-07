"""Explicit source route topology, independent of cargo and lexical generation.

Registered ports establish geography, not a carrier service or sailing schedule.
Node sharing represents source ownership (for example, loading at the via hub).
It is never inferred from an unfilled transshipment caption.
"""

from __future__ import annotations

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
            or self.issue_node not in known
        ):
            raise ValueError("route topology contains an undefined node")
        if not self.nodes.keys() <= set(self.fields.values()):
            raise ValueError("route topology has unused nodes")
        if any(not name.isidentifier() for name in self.nodes):
            raise ValueError("route node names must be identifiers")
        return self

    def validate_source(self, patch: dict) -> None:
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
