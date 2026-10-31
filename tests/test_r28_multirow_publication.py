import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest


def script(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "scripts/synthesis"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "refresh_publication", folder / "publish_r28_multirow_refresh.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    original = dict(
        documentId="existing",
        sourceDocumentId="source",
        joinedRawText="old",
        joinedRawTextSha256=digest(b"old"),
        positionedText="old ||",
        positionedTextSha256=digest(b"old ||"),
        target={
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 10, "typeCategory": "PACKAGE_BOX"}
                    ]
                }
            ]
        },
    )
    native = deepcopy(original)
    native.update(
        documentId="native",
        joinedRawText="12 IBCS + 8 PALLETS",
        joinedRawTextSha256=digest(b"12 IBCS + 8 PALLETS"),
        positionedText="12 IBCS + 8 PALLETS ||",
        positionedTextSha256=digest(b"12 IBCS + 8 PALLETS ||"),
    )
    native["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"] = [
        {"packageQuantity": 12, "typeCategory": "PACKAGE_INTERMEDIATE_BULK_CONTAINER"},
        {"packageQuantity": 8, "typeCategory": "PACKAGE_PALLET"},
    ]
    receipt = dict(
        previousRecordSha256=digest(original),
        nativeRecordSha256=digest(native),
        candidateId="native",
    )
    return original, native, receipt


def test_replacement_preserves_r28_identity_and_complete_native_facts(monkeypatch):
    module = script(monkeypatch)
    original, native, receipt = fixture()
    result = module.replacement(original, native, receipt)
    assert result == {**native, "documentId": "existing"}
    assert original["joinedRawText"] == "old"
    result["target"].clear()
    assert native["target"]


@pytest.mark.parametrize("kind", ["current", "native", "source", "identity", "single_row"])
def test_replacement_rejects_stale_or_substituted_evidence(monkeypatch, kind):
    module = script(monkeypatch)
    original, native, receipt = fixture()
    if kind == "current":
        original["joinedRawText"] = "concurrent edit"
    elif kind == "native":
        native["joinedRawText"] = "unreviewed"
    elif kind == "source":
        native["sourceDocumentId"] = "other"
        receipt["nativeRecordSha256"] = digest(native)
    elif kind == "identity":
        receipt["candidateId"] = "other"
    else:
        native["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"].pop()
        receipt["nativeRecordSha256"] = digest(native)
    with pytest.raises(ValueError):
        module.replacement(original, native, receipt)


@pytest.mark.parametrize("fail_readback", [False, True])
def test_publication_readback_is_part_of_rollback(monkeypatch, tmp_path, fail_readback):
    module = script(monkeypatch)
    dataset, publication, before = [tmp_path / name for name in ("data", "stage", "before")]
    for folder in (dataset, publication, before):
        folder.mkdir()
    old, new = {}, {}
    for name in ("first.jsonl", "second.jsonl"):
        (dataset / name).write_bytes(b"original\n")
        (before / name).write_bytes(b"original\n")
        (publication / name).write_bytes(b"replacement\n")
        old[name] = sha256_file(before / name)
        new[name] = sha256_file(publication / name)

    def verify():
        assert all(sha256_file(dataset / name) == pin for name, pin in new.items())
        if fail_readback:
            raise ValueError("readback failed")

    if fail_readback:
        with pytest.raises(ValueError, match="readback failed"):
            module.promote_and_verify(dataset, publication, before, old, new, verify)
    else:
        module.promote_and_verify(dataset, publication, before, old, new, verify)
    assert all(
        sha256_file(dataset / name) == pin for name, pin in (old if fail_readback else new).items()
    )
