"""Explicitly resume immutable, rejected provider calls after account recovery."""

import json

from document_ocr.synthesis.curated import digest


def authorize_rejected_calls(runner):
    """Authorize terminal credit/rate rejections, preserving any earlier billed response.

    A native-output correction can be rejected after its first response was
    billed. Cost status describes accounting, not the terminal failure type.
    Unknown billing and unrelated model/output failures remain unauthorized.
    """
    authorized = []
    for path in sorted((runner.output / "calls").glob("*.json")):
        payload = path.read_bytes()
        receipt = json.loads(payload)
        if (
            receipt.get("output") is None
            and receipt.get("costStatus") in {"request_rejected", "provider_reported"}
            and any(
                receipt.get("error", "").startswith(f"ModelHTTPError: status_code: {code},")
                for code in (402, 429)
            )
        ):
            authorized.append(
                {"path": str(path.relative_to(runner.root)), "sha256": digest(payload)}
            )
    runner.retry_failed_receipt_sha256s = frozenset(p["sha256"] for p in authorized)
    return authorized
