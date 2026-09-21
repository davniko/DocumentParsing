import asyncio
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from document_ocr.synthesis.template_compiler.spending_guard import (
    SpendingGuard,
    SpendingLimitExceeded,
)


def test_concurrent_reservations_cannot_oversubscribe(tmp_path):
    guard = SpendingGuard(
        tmp_path / "cost.sqlite3", limit_usd=Decimal("1"), pricing_sha256="prices"
    )

    def reserve(_):
        try:
            return guard.reserve("request", Decimal("0.2"))
        except SpendingLimitExceeded:
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        reservations = list(executor.map(reserve, range(50)))
    assert len([x for x in reservations if x]) == 5
    assert guard.snapshot()["occupiedUsd"] == "1"


def test_restart_does_not_forget_uncertain_calls(tmp_path):
    path = tmp_path / "cost.sqlite3"
    guard = SpendingGuard(path, limit_usd=Decimal("1"), pricing_sha256="prices")
    receipt = guard.reserve("x", Decimal("0.8"))
    guard.settle(receipt, None)
    resumed = SpendingGuard(path, limit_usd=Decimal("1"), pricing_sha256="prices")
    with pytest.raises(SpendingLimitExceeded):
        resumed.reserve("y", Decimal("0.3"))
    with pytest.raises(ValueError, match="differs"):
        SpendingGuard(path, limit_usd=Decimal("2"), pricing_sha256="prices")


def test_settlement_refunds_unused_reservation_once(tmp_path):
    guard = SpendingGuard(
        tmp_path / "cost.sqlite3", limit_usd=Decimal("1"), pricing_sha256="prices"
    )
    receipt = guard.reserve("x", Decimal("0.9"))
    guard.settle(receipt, Decimal("0.1"))
    guard.reserve("y", Decimal("0.9"))
    with pytest.raises(ValueError):
        guard.settle(receipt, Decimal("0.1"))
    assert guard.snapshot()["settledUsd"] == "0.1"


def test_underreserved_provider_receipt_permanently_stops_run(tmp_path):
    path = tmp_path / "cost.sqlite3"
    guard = SpendingGuard(path, limit_usd=Decimal("1"), pricing_sha256="prices")
    receipt = guard.reserve("x", Decimal("0.1"))
    with pytest.raises(SpendingLimitExceeded):
        guard.settle(receipt, Decimal("0.2"))
    resumed = SpendingGuard(path, limit_usd=Decimal("1"), pricing_sha256="prices")
    with pytest.raises(SpendingLimitExceeded):
        resumed.reserve("y", Decimal("0.1"))


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity", "0"])
def test_invalid_reservations_fail(tmp_path, value):
    guard = SpendingGuard(
        tmp_path / "cost.sqlite3", limit_usd=Decimal("1"), pricing_sha256="prices"
    )
    with pytest.raises(ValueError):
        guard.reserve("x", Decimal(value))


def test_active_reservations_throttle_concurrency_without_false_exhaustion(tmp_path):
    async def run():
        guard = SpendingGuard(
            tmp_path / "cost.sqlite3", limit_usd=Decimal("0.5"), pricing_sha256="prices"
        )

        async def request(i):
            reservation = await guard.acquire(str(i), Decimal("0.3"))
            await asyncio.sleep(0)
            guard.settle(reservation, Decimal("0.001"))

        await asyncio.wait_for(asyncio.gather(*(request(i) for i in range(50))), 2)
        assert guard.snapshot()["settledUsd"] == "0.05"
        assert guard.snapshot()["uncertainRequests"] == 0

    asyncio.run(run())


def test_abandoned_reservations_do_not_wait_forever(tmp_path):
    async def run():
        path = tmp_path / "cost.sqlite3"
        old = SpendingGuard(path, limit_usd=Decimal("0.5"), pricing_sha256="prices")
        await old.acquire("abandoned", Decimal("0.4"))
        restarted = SpendingGuard(path, limit_usd=Decimal("0.5"), pricing_sha256="prices")
        with pytest.raises(SpendingLimitExceeded):
            await asyncio.wait_for(restarted.acquire("new", Decimal("0.3")), 0.1)

    asyncio.run(run())
