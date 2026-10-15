"""Verify scoped lexical extensions without rewriting their generation history.

An extension certifies newly requested source-only fields. The exact previous
receipt, including manual corrections, remains immutable and supplies all old
values. A current validation-contract hash is not presented as an old model call.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_wording import (
    WORDING_PROMPT,
    WordingBatch,
    WordingField,
    WordingRequest,
    identity_key,
    native_wording_prompt,
    unpack_wording,
    validate_wording,
    wording_output_type,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def wording_request_data(request: WordingRequest) -> dict:
    data = asdict(request)
    data["fields"] = list(data["fields"])
    data["description_groups"] = [list(group) for group in data["description_groups"]]
    data["gross_weight_kg"] = (
        str(request.gross_weight_kg) if request.gross_weight_kg is not None else None
    )
    return data


def request_from_data(data: dict) -> WordingRequest:
    require(
        isinstance(data, dict)
        and set(data) == {
            "sample_id", "context", "fields", "description_groups", "gross_weight_kg",
        },
        "scoped wording request has unknown or missing fields",
    )
    require(
        isinstance(data["sample_id"], str) and data["sample_id"]
        and isinstance(data["context"], str) and data["context"]
        and isinstance(data["fields"], (list, tuple)) and data["fields"],
        "scoped wording request identity/context/fields are invalid",
    )
    fields = tuple(WordingField(**field) for field in data["fields"])
    require(len({field.key for field in fields}) == len(fields), "duplicate scoped wording key")
    gross = data["gross_weight_kg"]
    return WordingRequest(
        data["sample_id"], data["context"], fields,
        tuple(tuple(group) for group in data["description_groups"]),
        Decimal(gross) if gross is not None else None,
    )


def wording_request_hash(request: WordingRequest) -> str:
    """Hash the validation contract independently of any generation call."""
    return digest({
        "system": WORDING_PROMPT,
        "request": native_wording_prompt([request]),
        "schema": wording_output_type([request]).model_json_schema(),
        "gross_weight_kg": (
            str(request.gross_weight_kg) if request.gross_weight_kg is not None else None
        ),
    })


def validate_private_postal(request: WordingRequest, values: dict[str, str]) -> None:
    """Reject a copied full private address, not an unchanged country component."""
    for field in request.fields:
        if "postal" in field.role.casefold():
            require(
                identity_key(values[field.key]) != identity_key(field.example),
                f"{field.key}: copied complete private postal surface",
            )


def values_batch(sample_id: str, values: dict[str, str]) -> WordingBatch:
    return WordingBatch.model_validate({"shipments": [{
        "sample_id": sample_id,
        "values": [{"key": key, "text": value} for key, value in values.items()],
    }]})


def pinned_output_reference(output: Path, reference: dict, directory: str) -> tuple[Path, dict]:
    require(
        isinstance(reference, dict) and set(reference) == {"path", "sha256"},
        "wording lineage reference is invalid",
    )
    relative = reference["path"]
    require(
        isinstance(relative, str) and relative and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts
        and Path(relative).parent == Path(directory),
        "wording lineage reference escapes its owned output directory",
    )
    require(
        isinstance(reference["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]),
        "wording lineage reference SHA-256 is invalid",
    )
    path = output / relative
    payload = read_regular_file_bytes(path)
    require(digest(payload) == reference["sha256"], "wording lineage reference changed")
    value = json.loads(payload)
    require(isinstance(value, dict), "wording lineage reference is not an object")
    return path, value


def verify_scoped_call(output: Path, provider: dict, request: WordingRequest, reference: dict):
    path, receipt = pinned_output_reference(output, reference, "calls")
    schema = wording_output_type([request])
    prompt = native_wording_prompt([request])
    key = digest([
        "scoped-private-wording", request.sample_id, schema.model_json_schema(),
        WORDING_PROMPT, prompt, provider,
    ])
    require(
        re.fullmatch(
            rf"scoped-private-wording-{key}(?:-attempt(?:[2-9]|[1-9]\d+))?\.json", path.name
        ),
        "scoped native call does not belong to the exact request/provider",
    )
    require(
        receipt["stage"] == "scoped-private-wording"
        and receipt["identity"] == request.sample_id
        and receipt["output"] is not None and receipt["error"] is None
        and receipt["promptSha256"] == digest(prompt)
        and receipt["schemaSha256"] == digest(schema.model_json_schema()),
        "scoped native call is unsuccessful or belongs to a different contract",
    )
    systems, users, responses = [], [], []
    for message in receipt["messages"]:
        if message["kind"] == "request":
            for part in message["parts"]:
                if part["part_kind"] == "system-prompt":
                    systems.append(part["content"])
                elif part["part_kind"] == "user-prompt":
                    users.append(part["content"])
        elif message["kind"] == "response":
            responses.append(message)
    require(systems == [WORDING_PROMPT] and users == [prompt], "scoped native transcript differs")
    require(len(responses) == 1, "scoped native response coverage differs")
    visible = "".join(
        part["content"] for part in responses[0]["parts"] if part["part_kind"] == "text"
    )
    actual = schema.model_validate_json(visible)
    require(
        actual.model_dump(mode="json") == schema.model_validate(receipt["output"]).model_dump(),
        "scoped native output differs from the actual provider response",
    )
    values = unpack_wording(actual, [request])[request.sample_id]
    validate_private_postal(request, values)
    return values


def validate_wording_receipt(
    receipt: dict, request: WordingRequest, *, output: Path, provider: dict | None
) -> dict[str, str]:
    """Validate ordinary receipts and, when present, immutable scoped lineage."""
    require(
        receipt["requestSha256"] == wording_request_hash(request),
        "wording request changed; regenerate or explicitly extend this source",
    )
    extension = receipt.get("scopedExtension")
    if extension is not None:
        require(isinstance(provider, dict), "scoped wording lineage needs the exact provider")
        require(
            isinstance(extension, dict) and set(extension) == {
                "schemaVersion", "baseReceipt", "baseCandidate", "baseRequest", "scopedRequest",
                "scopedRequestSha256", "call", "packetSha256", "addedValuesSha256",
                "mergedValuesSha256",
            } and extension["schemaVersion"] == 1,
            "scoped wording lineage has unknown or missing fields",
        )
        _, base = pinned_output_reference(output, extension["baseReceipt"], "wording-history")
        _, candidate = pinned_output_reference(
            output, extension["baseCandidate"], "wording-history"
        )
        require("scopedExtension" not in base, "nested scoped wording extensions are unsupported")
        original = request_from_data(extension["baseRequest"])
        scoped = request_from_data(extension["scopedRequest"])
        require(
            request.sample_id == original.sample_id == scoped.sample_id
            == base["documentId"] == receipt["documentId"]
            and base["sourceDocumentId"] == receipt["sourceDocumentId"]
            and candidate["documentId"] == receipt["documentId"]
            and candidate["sourceDocumentId"] == receipt["sourceDocumentId"]
            and base["scenario"] == receipt["scenario"]
            and base["targetBeforeWording"] == receipt["targetBeforeWording"],
            "scoped wording lineage changed existing shipment identity or facts",
        )
        require(
            base["requestSha256"] == wording_request_hash(original)
            and extension["scopedRequestSha256"] == wording_request_hash(scoped)
            and re.fullmatch(r"[0-9a-f]{64}", extension["packetSha256"]),
            "scoped wording lineage request pins differ",
        )
        old_fields = {field.key: field for field in original.fields}
        new_fields = {field.key: field for field in scoped.fields}
        current_fields = {field.key: field for field in request.fields}
        require(
            not old_fields.keys() & new_fields.keys()
            and old_fields | new_fields == current_fields
            and not scoped.description_groups and scoped.gross_weight_kg is None
            and all("name" in field.role or "postal" in field.role for field in scoped.fields),
            "scoped wording extension must own exactly new private name/postal fields",
        )
        # The old receipt can contain manually adjudicated wording. Its exact
        # bytes and values are preserved; a new call cannot silently edit them.
        require(
            set(base["values"]) == set(old_fields)
            and all(receipt["values"].get(key) == value for key, value in base["values"].items()),
            "scoped wording extension altered accepted old values",
        )
        validate_wording(values_batch(original.sample_id, base["values"]), [original])
        added = verify_scoped_call(output, provider, scoped, extension["call"])
        require(
            base["values"] | added == receipt["values"]
            and digest(added) == extension["addedValuesSha256"]
            and digest(receipt["values"]) == extension["mergedValuesSha256"],
            "scoped wording extension differs from its native values or merged pins",
        )
    return validate_wording(values_batch(request.sample_id, receipt["values"]), [request])[
        request.sample_id
    ]
