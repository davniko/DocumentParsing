"""Verify immutable native packets against actual canonical Runner receipt bytes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.curated import digest


def _read_pin(root: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    if set(pin) != {"path", "sha256"}:
        raise ValueError("native evidence requires an exact path/SHA-256 file pin")
    relative = Path(pin["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("native evidence paths must remain relative to the project root")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("native evidence escapes the project root")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != pin["sha256"]:
        raise ValueError("native evidence file SHA-256 differs")
    return cast(dict[str, Any], json.loads(payload))


def verify_native_evidence(
    root: Path,
    evidence: Mapping[str, Any],
    *,
    stage: str,
    identity: str,
    request: Mapping[str, Any],
    model: type[BaseModel],
    system: str,
    prompt: str,
    provider: Mapping[str, Any],
) -> dict[str, Any]:
    """A packet alone is not proof: require matching received output on the real wire."""
    if set(evidence) != {"request", "output", "packet", "packetSha256", "callReceipts"}:
        raise ValueError("native decision lacks complete immutable evidence")
    packet = _read_pin(root, evidence["packet"])
    expected = {
        "stage": stage,
        "identity": identity,
        "request": request,
        "schema": model.model_json_schema(),
        "system": system,
        "prompt": prompt,
        "provider": dict(provider),
    }
    if (
        packet != expected
        or evidence["request"] != request
        or digest(packet) != evidence["packetSha256"]
    ):
        raise ValueError("native evidence packet differs from current request/schema/provider")
    pins = evidence["callReceipts"]
    if not isinstance(pins, list) or not pins:
        raise ValueError("native decision has no canonical Runner receipts")
    paths = [pin["path"] for pin in pins]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate native receipt pin")
    call_key = digest([stage, identity, model.model_json_schema(), system, prompt, dict(provider)])
    wire_name = re.compile(
        re.escape(stage + "-" + call_key) + r"(?:-attempt(?:[2-9]|[1-9][0-9]+))?\.json"
    )
    received = 0
    for pin in pins:
        if wire_name.fullmatch(Path(pin["path"]).name) is None:
            raise ValueError("receipt filename differs from the canonical Runner wire identity")
        receipt = _read_pin(root, pin)
        if any(
            receipt.get(key) != value
            for key, value in {
                "stage": stage,
                "identity": identity,
                "promptSha256": digest(prompt),
                "schemaSha256": digest(model.model_json_schema()),
            }.items()
        ):
            raise ValueError("native receipt belongs to a different request")
        if receipt.get("output") is not None:
            if receipt.get("error") is not None or receipt["output"] != evidence["output"]:
                raise ValueError("native evidence differs from the actual received output")
            received += 1
    if received != 1:
        raise ValueError("native decision requires exactly one unambiguous successful receipt")
    return model.model_validate(evidence["output"]).model_dump(mode="json")
