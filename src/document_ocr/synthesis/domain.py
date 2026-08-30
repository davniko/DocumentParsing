"""Task-specific relational projections used by synthesis preparation.

The generic value graph remains the byte-for-byte losslessness guard.  Domain
adapters expose a second, semantic representation whose tables are useful to
statistical synthesizers and deterministic constraint engines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class RelationalTables:
    """Ordered rows grouped by table name."""

    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add(self, table: str, row: Mapping[str, Any]) -> None:
        self.rows.setdefault(table, []).append(dict(row))

    def extend(self, other: RelationalTables) -> None:
        for table, rows in other.rows.items():
            self.rows.setdefault(table, []).extend(dict(row) for row in rows)

    def table(self, name: str) -> tuple[dict[str, Any], ...]:
        return tuple(self.rows.get(name, ()))


class DomainAdapter(Protocol):
    """Concrete task contract for a lossless semantic table bundle."""

    task: str
    table_order: tuple[str, ...]

    def project(
        self, *, document_id: str, source_row_index: int, target: Mapping[str, Any]
    ) -> RelationalTables: ...

    def reconstruct(
        self, *, document_id: str, tables: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> dict[str, Any]: ...

    def sdv_metadata(self) -> dict[str, Any]: ...


def rows_for_document(
    tables: Mapping[str, Sequence[Mapping[str, Any]]], document_id: str
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Select one document without relying on table adjacency."""

    return {
        name: tuple(row for row in rows if row.get("document_id") == document_id)
        for name, rows in tables.items()
    }
