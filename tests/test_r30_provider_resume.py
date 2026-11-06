"""Credit recovery must not discard failures or retry unrelated failed outputs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_resume_authorizes_only_pinned_credit_and_rate_rejections(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/migrations/r30_provider_resume.py"
    spec = importlib.util.spec_from_file_location("r30_provider_resume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = tmp_path / "provider/calls"
    calls.mkdir(parents=True)
    for code in (401, 402, 403, 429, 500):
        (calls / f"{code}.json").write_text(
            json.dumps(
                {
                    "output": None,
                    "costStatus": "request_rejected",
                    "error": f"ModelHTTPError: status_code: {code}, model_name: test",
                }
            )
        )
    (calls / "invalid.json").write_text(
        json.dumps(
            {
                "output": None,
                "costStatus": "billed",
                "error": "UnexpectedModelBehavior",
            }
        )
    )
    for name, status, output, error in (
        (
            "billed_then_rejected",
            "provider_reported",
            None,
            "ModelHTTPError: status_code: 402, model_name: test",
        ),
        (
            "unknown_cost",
            "reserved_unknown",
            None,
            "ModelHTTPError: status_code: 402, model_name: test",
        ),
        (
            "successful",
            "provider_reported",
            {"ok": True},
            "ModelHTTPError: status_code: 402, model_name: test",
        ),
        (
            "nested_reference",
            "provider_reported",
            None,
            "UnexpectedModelBehavior: text mentions ModelHTTPError: status_code: 402,",
        ),
    ):
        (calls / f"{name}.json").write_text(
            json.dumps(
                {
                    "output": output,
                    "costStatus": status,
                    "error": error,
                    "costUsd": "0.00814095",
                    "usage": {"requests": 1, "outputTokens": 12000},
                }
            )
        )
    before = {p: p.read_bytes() for p in calls.iterdir()}
    runner = SimpleNamespace(root=tmp_path, output=calls.parent)
    pins = module.authorize_rejected_calls(runner)
    assert {Path(p["path"]).stem for p in pins} == {"402", "429", "billed_then_rejected"}
    assert runner.retry_failed_receipt_sha256s == frozenset(p["sha256"] for p in pins)
    assert {p: p.read_bytes() for p in calls.iterdir()} == before
