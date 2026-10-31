"""Receipt checks share one schema construction without caching evidence bytes."""

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_native_evidence import verify_native_evidence


def test_schema_built_once_and_changed_wire_receipt_still_rejected(tmp_path):
    class Answer(BaseModel):
        quantity: int

    schema = Answer.model_json_schema()
    provider = {"model": "offline-test"}
    stage, identity, system, prompt = "test", "source:1", "system", "prompt"
    key = digest([stage, identity, schema, system, prompt, provider])

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": name, "sha256": sha256_file(path)}

    packet = {
        "stage": stage,
        "identity": identity,
        "request": {},
        "schema": schema,
        "system": system,
        "prompt": prompt,
        "provider": provider,
    }
    receipt = {
        "stage": stage,
        "identity": identity,
        "promptSha256": digest(prompt),
        "schemaSha256": digest(schema),
        "output": {"quantity": 2},
        "error": None,
    }
    wire = write(f"{stage}-{key}.json", receipt)
    evidence = {
        "request": {},
        "output": {"quantity": 2},
        "packet": write("packet.json", packet),
        "packetSha256": digest(packet),
        "callReceipts": [wire],
    }
    kwargs = dict(
        stage=stage,
        identity=identity,
        request={},
        model=Answer,
        system=system,
        prompt=prompt,
        provider=provider,
    )
    with patch.object(Answer, "model_json_schema", wraps=Answer.model_json_schema) as build:
        assert verify_native_evidence(tmp_path, evidence, **kwargs) == {"quantity": 2}
        assert build.call_count == 1
    (tmp_path / wire["path"]).write_text("changed")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        verify_native_evidence(tmp_path, evidence, **kwargs)
