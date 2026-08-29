"""Lossless, deterministic JSON value graph used before synthesis policy is chosen."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

NodeKind = Literal["object", "list", "string", "integer", "number", "boolean", "null"]
EdgeKind = Literal["root", "object_field", "list_item"]


@dataclass(frozen=True, slots=True)
class NormalizedNode:
    document_id: str
    node_id: str
    parent_node_id: str | None
    edge_kind: EdgeKind
    edge_name: str | None
    edge_index: int | None
    child_order: int
    node_kind: NodeKind
    scalar_json: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _node_kind(value: Any) -> NodeKind:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"
    raise TypeError(f"unsupported JSON value type: {type(value).__qualname__}")


def normalize_target(document_id: str, target: dict[str, Any]) -> tuple[NormalizedNode, ...]:
    """Encode one JSON target as a deterministic preorder graph."""

    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty")
    nodes: list[NormalizedNode] = []

    def visit(
        value: Any,
        *,
        parent: str | None,
        edge_kind: EdgeKind,
        edge_name: str | None,
        edge_index: int | None,
        child_order: int,
    ) -> str:
        node_id = f"{document_id}:n{len(nodes) + 1:06d}"
        kind = _node_kind(value)
        scalar = (
            None
            if kind in {"object", "list"}
            else json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        )
        nodes.append(
            NormalizedNode(
                document_id=document_id,
                node_id=node_id,
                parent_node_id=parent,
                edge_kind=edge_kind,
                edge_name=edge_name,
                edge_index=edge_index,
                child_order=child_order,
                node_kind=kind,
                scalar_json=scalar,
            )
        )
        if isinstance(value, dict):
            for order, (name, child) in enumerate(value.items()):
                visit(
                    child,
                    parent=node_id,
                    edge_kind="object_field",
                    edge_name=name,
                    edge_index=None,
                    child_order=order,
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(
                    child,
                    parent=node_id,
                    edge_kind="list_item",
                    edge_name=None,
                    edge_index=index,
                    child_order=index,
                )
        return node_id

    visit(
        target,
        parent=None,
        edge_kind="root",
        edge_name=None,
        edge_index=None,
        child_order=0,
    )
    return tuple(nodes)


def denormalize_target(nodes: tuple[NormalizedNode, ...]) -> dict[str, Any]:
    """Rebuild one target while rejecting missing, duplicate, cyclic, or malformed edges."""

    if not nodes:
        raise ValueError("normalized target contains no nodes")
    by_id = {row.node_id: row for row in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("normalized target contains duplicate node IDs")
    document_ids = {row.document_id for row in nodes}
    if len(document_ids) != 1:
        raise ValueError("normalized target mixes document IDs")
    roots = [row for row in nodes if row.parent_node_id is None]
    if len(roots) != 1 or roots[0].edge_kind != "root":
        raise ValueError("normalized target must contain exactly one root edge")
    children: dict[str, list[NormalizedNode]] = {}
    for row in nodes:
        if row.parent_node_id is None:
            continue
        if row.parent_node_id not in by_id:
            raise ValueError(f"node references absent parent: {row.node_id}")
        children.setdefault(row.parent_node_id, []).append(row)
    visiting: set[str] = set()
    visited: set[str] = set()

    def rebuild(node_id: str) -> Any:
        if node_id in visiting:
            raise ValueError("normalized target contains a cycle")
        visiting.add(node_id)
        row = by_id[node_id]
        descendants = sorted(children.get(node_id, []), key=lambda child: child.child_order)
        if row.node_kind == "object":
            if row.scalar_json is not None:
                raise ValueError("object node cannot carry scalar JSON")
            result: dict[str, Any] = {}
            expected_orders = list(range(len(descendants)))
            if [child.child_order for child in descendants] != expected_orders:
                raise ValueError("object child order is not contiguous")
            for child in descendants:
                if (
                    child.edge_kind != "object_field"
                    or child.edge_name is None
                    or child.edge_index is not None
                ):
                    raise ValueError("object child has an invalid edge")
                if child.edge_name in result:
                    raise ValueError("object contains a duplicate field edge")
                result[child.edge_name] = rebuild(child.node_id)
            value: Any = result
        elif row.node_kind == "list":
            if row.scalar_json is not None:
                raise ValueError("list node cannot carry scalar JSON")
            if [child.edge_index for child in descendants] != list(range(len(descendants))):
                raise ValueError("list indexes are not contiguous")
            if any(
                child.edge_kind != "list_item" or child.edge_name is not None
                for child in descendants
            ):
                raise ValueError("list child has an invalid edge")
            value = [rebuild(child.node_id) for child in descendants]
        else:
            if descendants:
                raise ValueError("scalar node cannot have children")
            if row.scalar_json is None:
                raise ValueError("scalar node is missing scalar JSON")
            value = json.loads(row.scalar_json)
            if _node_kind(value) != row.node_kind:
                raise ValueError("scalar JSON type differs from node_kind")
        visiting.remove(node_id)
        visited.add(node_id)
        return value

    result = rebuild(roots[0].node_id)
    if visited != set(by_id):
        raise ValueError("normalized target contains nodes disconnected from the root")
    if not isinstance(result, dict):
        raise ValueError("target root must reconstruct as an object")
    return result
