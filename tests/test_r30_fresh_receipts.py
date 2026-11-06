import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
import stage_r30_fresh_receipts as migration


@pytest.fixture
def source(tmp_path, monkeypatch):
    artifact = tmp_path / "migration"
    did = "doc_12345678"
    authority = b"OLD BILL"
    transfer = {"source": did, "descendants": [], "slots": []}
    candidate = {
        "sourceDocumentId": did,
        "documentId": "synthetic_one",
        "joinedRawText": "NEW BILL",
        "target": {"documentPatch": {"negotiability": "non_negotiable"}},
        "proof": {
            "sourceSha256": hashlib.sha256(authority).hexdigest(),
            "edits": [
                {
                    "key": "bill",
                    "byteStart": 0,
                    "byteEnd": 8,
                    "before": "OLD BILL",
                    "after": "NEW BILL",
                    "targetPaths": [],
                }
            ],
        },
    }
    files = {
        "configs/synthesis/mpci_bl_curated_v7_multirow.json": {"source_contracts": "contracts"},
        "migration/transfers/doc_12345678.json": transfer,
        "migration/sources/doc_12345678.json": {
            "documentId": did,
            "pages": [{"index": 0, "regions": [{"index": 0, "text": "OLD BILL"}]}],
        },
        "candidates/one.json": candidate,
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    contract = tmp_path / "contracts" / did / "source.txt"
    contract.parent.mkdir(parents=True)
    contract.write_bytes(authority)
    monkeypatch.setattr(migration, "ROOT", tmp_path)
    monkeypatch.setattr(migration, "PILOT_ART", artifact)
    return tmp_path, artifact, candidate


def test_first_fresh_receipt_uses_source_authority_without_old_descendant(source):
    _, artifact, _ = source
    migration.main("12345678", "candidates", scope="pilot")
    result = json.loads((artifact / "transfers/doc_12345678.json").read_text())
    assert result["remainingDescendants"] == 2
    assert result["descendants"][0]["id"] == "synthetic_one"
    assert result["slots"][0]["before"] == "OLD BILL"
    assert (artifact / "fresh-records/synthetic_one.json").exists()
    with pytest.raises(ValueError, match="already staged"):
        migration.main("12345678", "candidates", scope="pilot")


@pytest.mark.parametrize("invalid", ["hash", "edit"])
def test_fresh_receipt_rejects_wrong_authority_before_publishing(source, invalid):
    root, artifact, candidate = source
    if invalid == "hash":
        candidate["proof"]["sourceSha256"] = "0" * 64
    else:
        candidate["proof"]["edits"][0]["before"] = "BAD BILL"
    (root / "candidates/one.json").write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="source authority"):
        migration.main("12345678", "candidates", scope="pilot")
    assert not (artifact / "fresh-records").exists()
    assert json.loads((artifact / "transfers/doc_12345678.json").read_text())["descendants"] == []
