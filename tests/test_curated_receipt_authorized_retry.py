"""Exact failed-receipt authority, with no network or implicit retry policy."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import pytest
from pydantic import BaseModel

from document_ocr.atomic import atomic_write_bytes
from document_ocr.synthesis import curated


class Answer(BaseModel):
    value: str


def cached_runner(tmp_path):
    runner = curated.Runner.__new__(curated.Runner)
    runner.output = tmp_path
    runner.config = {"provider": {"route": "unchanged"}}
    runner.retry_rate_limited = False
    runner.retry_invalid_output = False
    runner.retry_failed_receipt_sha256s = frozenset()
    runner.model = None
    runner.spent = Decimal("0.01784")
    key = curated.digest(
        [
            "wording",
            "source",
            Answer.model_json_schema(),
            "sys",
            "prompt",
            runner.config["provider"],
        ]
    )
    path = tmp_path / "calls" / f"wording-{key}.json"
    payload = json.dumps(
        {
            "output": None,
            "error": "ModelHTTPError: status_code: 503, temporarily unavailable",
            "costStatus": "reserved_unknown",
            "costUsd": "0.01784",
        }
    ).encode()
    atomic_write_bytes(path, payload)
    return runner, path, payload


def call(runner):
    return asyncio.run(runner.call("wording", "source", Answer, "sys", "prompt"))


def test_unlisted_failure_has_no_retry_or_extra_hash_work(monkeypatch, tmp_path):
    runner, path, payload = cached_runner(tmp_path)
    actual_digest = curated.digest

    def key_only(value):
        if isinstance(value, bytes):
            raise AssertionError("empty authorization must not hash receipt payloads")
        return actual_digest(value)

    monkeypatch.setattr(curated, "digest", key_only)
    with pytest.raises(RuntimeError, match="cached failed call"):
        call(runner)
    assert path.read_bytes() == payload and runner.model is None
    assert list(path.parent.iterdir()) == [path]


def test_only_exact_failed_bytes_authorize_attempt2(tmp_path):
    runner, path, payload = cached_runner(tmp_path)
    runner.retry_failed_receipt_sha256s = frozenset({curated.digest(payload)})
    attempt2 = path.with_name(path.stem + "-attempt2.json")
    atomic_write_bytes(attempt2, b'{"output":{"value":"accepted exact request"}}')
    assert call(runner).value == "accepted exact request"
    assert path.read_bytes() == payload
    assert runner.spent == Decimal("0.01784") and runner.model is None
    assert not path.with_name(path.stem + "-attempt3.json").exists()


def test_changed_failed_bytes_invalidate_authority(tmp_path):
    runner, path, payload = cached_runner(tmp_path)
    runner.retry_failed_receipt_sha256s = frozenset({curated.digest(payload)})
    atomic_write_bytes(path, payload + b"\n")
    with pytest.raises(RuntimeError, match="cached failed call"):
        call(runner)
    assert list(path.parent.iterdir()) == [path]


def test_new_failed_attempt_does_not_inherit_previous_authorization(tmp_path):
    runner, path, payload = cached_runner(tmp_path)
    runner.retry_failed_receipt_sha256s = frozenset({curated.digest(payload)})
    attempt2 = path.with_name(path.stem + "-attempt2.json")
    failed2 = json.dumps({**json.loads(payload), "seconds": 1}).encode()
    atomic_write_bytes(attempt2, failed2)
    with pytest.raises(RuntimeError, match=r"attempt2\.json"):
        call(runner)
    assert path.read_bytes() == payload and attempt2.read_bytes() == failed2
    assert not path.with_name(path.stem + "-attempt3.json").exists()


def test_authorized_hash_never_retries_a_successful_receipt(tmp_path):
    runner, path, _ = cached_runner(tmp_path)
    payload = b'{"output":{"value":"cached success"},"costUsd":"0.0002"}'
    atomic_write_bytes(path, payload)
    runner.retry_failed_receipt_sha256s = frozenset({curated.digest(payload)})
    assert call(runner).value == "cached success"
    assert list(path.parent.iterdir()) == [path]


def test_recorded_successful_retry_is_reusable_without_new_authorization(tmp_path):
    runner, path, payload = cached_runner(tmp_path)
    attempt2 = path.with_name(path.stem + "-attempt2.json")
    accepted = b'{"output":{"value":"approved retry already completed"}}'
    atomic_write_bytes(attempt2, accepted)
    assert not runner.retry_failed_receipt_sha256s
    assert call(runner).value == "approved retry already completed"
    assert path.read_bytes() == payload and attempt2.read_bytes() == accepted
    assert runner.model is None and runner.spent == Decimal("0.01784")
    assert not path.with_name(path.stem + "-attempt3.json").exists()


def test_recorded_failed_retry_tail_still_requires_its_own_authority(tmp_path):
    runner, path, payload = cached_runner(tmp_path)
    attempt2 = path.with_name(path.stem + "-attempt2.json")
    atomic_write_bytes(attempt2, payload)
    with pytest.raises(RuntimeError, match=r"attempt2\.json"):
        call(runner)
    assert runner.model is None
    assert not path.with_name(path.stem + "-attempt3.json").exists()


def test_noncontiguous_receipt_is_not_a_retry_successor(tmp_path):
    runner, path, _ = cached_runner(tmp_path)
    atomic_write_bytes(path.with_name(path.stem + "-attempt3.json"),
                       b'{"output":{"value":"unrelated gap"}}')
    with pytest.raises(RuntimeError, match="cached failed call"):
        call(runner)
    assert runner.model is None


@pytest.mark.parametrize(
    "field,value",
    [("prompt", "changed"), ("system", "changed"), ("provider", {"route": "changed"})],
)
def test_authority_does_not_redirect_a_different_request(tmp_path, field, value):
    runner, path, payload = cached_runner(tmp_path)
    runner.retry_failed_receipt_sha256s = frozenset({curated.digest(payload)})
    prompt, system = "prompt", "sys"
    if field == "prompt":
        prompt = value
    elif field == "system":
        system = value
    else:
        runner.config["provider"] = value

    class NoNetwork:
        async def __aenter__(self):
            raise RuntimeError("distinct request reached its own uncached path")

        async def __aexit__(self, *args):
            return False

    runner.limiter = NoNetwork()
    with pytest.raises(RuntimeError, match="distinct request reached its own uncached path"):
        asyncio.run(runner.call("wording", "source", Answer, system, prompt))
    assert path.read_bytes() == payload and runner.model is None
