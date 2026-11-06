import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))

from audit_r30_synthetic_batch import account_blocked


def test_only_explicit_provider_account_errors_stop_remaining_batches():
    for code in (401, 402, 403):
        assert account_blocked(RuntimeError(f"ModelHTTPError: status_code: {code}, body: rejected"))
    for message in (
        "ModelHTTPError: status_code: 429, body: throttled",
        "ModelHTTPError: status_code: 500, body: server error",
        "UnexpectedModelBehavior: token limit exceeded",
        "Document contains 402 packages",
    ):
        assert not account_blocked(RuntimeError(message))


def test_account_rejection_saves_resume_report_and_does_not_submit_next_group(
    tmp_path, monkeypatch
):
    import audit_r30_synthetic_batch as audit

    policy = tmp_path / "docs/kie-real-baseline-label-policy-2026-10-05.md"
    policy.parent.mkdir()
    policy.write_text(
        "policy\n## Current published baseline: R12\nold\n"
        "## Source-owner repair clarifications\ncurrent"
    )
    entries = [{"id": str(i), "rendered": {"path": str(i)}} for i in range(20)]
    calls, reports = [], []

    class RejectedRunner:
        spent = Decimal("0.5")

        def __init__(self, root, config):
            assert config["concurrency"] == 8

        async def call(self, stage, identity, *args, **kwargs):
            calls.append(identity)
            raise RuntimeError("ModelHTTPError: status_code: 402, body: Insufficient credits")

    def read(path):
        if path.name == "001-staging.json":
            return {"results": entries}
        if path.name == "mpci_bl_curated_v7_paddle.json":
            return {"provider": {}}
        if path.name == "schema.json":
            return {}
        if path.name == "preferred-names.json":
            return {"names": {}}
        return {"documentId": path.name, "target": {}, "pages": []}

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "ART", tmp_path / "artifacts")
    monkeypatch.setattr(audit, "DATA", tmp_path / "data")
    monkeypatch.setattr(audit, "OUT", tmp_path / "out")
    monkeypatch.setattr(audit, "Runner", RejectedRunner)
    monkeypatch.setattr(audit, "CuratedSynthesisConfig", SimpleNamespace(model_fields={"provider"}))
    monkeypatch.setattr(audit, "read", read)
    monkeypatch.setattr(audit, "verify_pin", lambda p: Path(p["path"]))
    monkeypatch.setattr(audit, "pin", lambda p: {"path": str(p)})
    monkeypatch.setattr(audit, "atomic_publish_json", lambda p, r: reports.append(r))
    with pytest.raises(RuntimeError, match="Provider account blocked"):
        asyncio.run(audit.main(1, ids=[entry["id"] for entry in entries]))
    # This provider rejects synchronously: no other request has entered flight.
    assert calls == ["0"]
    (report,) = reports
    assert report["providerAccountBlocked"] is True
    assert report["unattempted"] == [str(i) for i in range(1, 20)]
    assert report["allAttemptCostUsd"] == "0.0"
    assert report["cumulativeProviderCostUsd"] == "0.5"
