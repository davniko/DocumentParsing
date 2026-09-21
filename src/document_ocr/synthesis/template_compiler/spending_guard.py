"""Durable, cross-process reservations for a explicitly capped synthesis run.

Uncertain/abandoned requests keep their reservation. Restarting never resets the
ledger, and concurrent workers reserve before sending a request. This bounds
estimated charges at the configured prices; it is not a provider invoice API.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from uuid import uuid4

_SCALE = Decimal(10**12)


class SpendingLimitExceeded(RuntimeError):
    pass


def _units(value: Decimal) -> int:
    if not value.is_finite() or value < 0:
        raise ValueError("cost must be finite and nonnegative")
    return int((value * _SCALE).to_integral_value(rounding=ROUND_CEILING))


class SpendingGuard:
    def __init__(self, path: Path, *, limit_usd: Decimal, pricing_sha256: str) -> None:
        if limit_usd <= 0:
            raise ValueError("an explicit positive spending limit is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._inflight: set[str] = set()
        self._settled = asyncio.Event()
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS budget (singleton INTEGER PRIMARY KEY "
                "CHECK(singleton=1), limit_units INTEGER NOT NULL, pricing TEXT NOT NULL, "
                "occupied INTEGER NOT NULL, violated INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, "
                "request_sha TEXT NOT NULL, reserved INTEGER NOT NULL, settled INTEGER, "
                "uncertain INTEGER NOT NULL)"
            )
            row = db.execute("SELECT limit_units, pricing FROM budget WHERE singleton=1").fetchone()
            expected = (_units(limit_usd), pricing_sha256)
            if row is None:
                db.execute("INSERT INTO budget VALUES (1, ?, ?, 0, 0)", expected)
            elif row != expected:
                raise ValueError(
                    "existing spending ledger limit/pricing differs; never reset it implicitly"
                )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        # Each operation has its own connection so the ledger is safe across
        # threads and processes. SQLite rolls back an interrupted transaction.
        with closing(sqlite3.connect(self.path, isolation_level=None)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    def reserve(self, request_sha256: str, maximum_cost_usd: Decimal) -> str:
        amount = _units(maximum_cost_usd)
        if amount <= 0:
            raise ValueError("request reservation must be positive")
        identity = uuid4().hex
        with self._transaction() as db:
            limit, occupied, violated = db.execute(
                "SELECT limit_units, occupied, violated FROM budget WHERE singleton=1"
            ).fetchone()
            if violated or occupied + amount > limit:
                raise SpendingLimitExceeded(
                    "synthesis spending limit reached; no additional provider request sent"
                )
            db.execute(
                "INSERT INTO calls VALUES (?, ?, ?, NULL, 0)", (identity, request_sha256, amount)
            )
            db.execute("UPDATE budget SET occupied=occupied+? WHERE singleton=1", (amount,))
        return identity

    async def acquire(self, request_sha256: str, maximum_cost_usd: Decimal) -> str:
        """Wait for this process's active reservations, never for abandoned ones."""
        while True:
            self._settled.clear()
            try:
                identity = self.reserve(request_sha256, maximum_cost_usd)
            except SpendingLimitExceeded:
                if not self._inflight or self.snapshot()["stopped"]:
                    raise
                await self._settled.wait()
            else:
                self._inflight.add(identity)
                return identity

    def settle(self, identity: str, actual_cost_usd: Decimal | None) -> None:
        try:
            self._settle(identity, actual_cost_usd)
        finally:
            self._inflight.discard(identity)
            self._settled.set()

    def _settle(self, identity: str, actual_cost_usd: Decimal | None) -> None:
        violation = False
        with self._transaction() as db:
            row = db.execute(
                "SELECT reserved, settled, uncertain FROM calls WHERE id=?", (identity,)
            ).fetchone()
            if row is None or row[1] is not None or row[2]:
                raise ValueError("unknown or already settled provider reservation")
            reserved = row[0]
            if actual_cost_usd is None:
                db.execute("UPDATE calls SET uncertain=1 WHERE id=?", (identity,))
                return
            actual = _units(actual_cost_usd)
            violation = actual > reserved
            db.execute("UPDATE calls SET settled=? WHERE id=?", (actual, identity))
            db.execute(
                "UPDATE budget SET occupied=occupied+?, violated=MAX(violated, ?) "
                "WHERE singleton=1",
                (actual - reserved, int(violation)),
            )
        if violation:
            raise SpendingLimitExceeded(
                "provider receipt exceeded its cost reservation; "
                "spending ledger permanently stopped"
            )

    def snapshot(self) -> dict[str, object]:
        with self._transaction() as db:
            limit, occupied, violated = db.execute(
                "SELECT limit_units, occupied, violated FROM budget WHERE singleton=1"
            ).fetchone()
            settled, calls, uncertain = db.execute(
                "SELECT COALESCE(SUM(settled),0), COUNT(*), "
                "SUM(CASE WHEN settled IS NULL THEN 1 ELSE 0 END) FROM calls"
            ).fetchone()
        return dict(
            limitUsd=str(Decimal(limit) / _SCALE),
            occupiedUsd=str(Decimal(occupied) / _SCALE),
            settledUsd=str(Decimal(settled) / _SCALE),
            requests=calls,
            uncertainRequests=uncertain or 0,
            stopped=bool(violated),
        )
