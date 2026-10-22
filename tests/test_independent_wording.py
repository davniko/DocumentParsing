"""Pinned, reviewed complete-phrase certificates for independent synthesis."""

from __future__ import annotations

import json

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_wording import load_certificates


def test_wording_certificates_require_exact_file_pin(tmp_path) -> None:
    sid = "doc_" + "f" * 64
    path = tmp_path / "wording.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    sid: {
                        "sourceSha256": "a" * 64,
                        "reviewStatus": "reviewed",
                        "operations": [],
                        "thermalReview": False,
                        "countedMaterialReview": False,
                    }
                },
            }
        )
    )
    expected = sha256_file(path)
    assert set(load_certificates(path, expected_sha256=expected)) == {sid}
    with pytest.raises(ValueError, match="SHA-256"):
        load_certificates(path, expected_sha256="0" * 64)


def test_wording_certificate_rejects_unreviewed_operation(tmp_path) -> None:
    path = tmp_path / "wording.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "doc_" + "a" * 64: {
                        "sourceSha256": "b" * 64,
                        "reviewStatus": "pending",
                        "operations": [],
                        "thermalReview": False,
                        "countedMaterialReview": False,
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="validation error"):
        load_certificates(path, expected_sha256=sha256_file(path))
