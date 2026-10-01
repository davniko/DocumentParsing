"""A compiled case cannot enter synthesis without its pinned goods-role certificate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.run_safety import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.template_compiler.descendant import _case_files
from document_ocr.synthesis.template_compiler.goods_role_certificate import (
    GoodsRoleCertificate,
    OriginalAdditionalInformation,
)


def _case(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "catalog"
    case = root / "cases" / "doc_test"
    case.mkdir(parents=True)
    source = b"Description of Goods\nTEST ARTICLE\n"
    label = canonical_json_bytes(
        {"documentPatch": {"cargoGroups": [{"groupId": "g1", "description": "TEST ARTICLE"}]}}
    )
    template = b'{"template":"test-only"}\n'
    (case / "source.txt").write_bytes(source)
    (case / "source-label.json").write_bytes(label)
    (case / "template.json").write_bytes(template)
    certificate = GoodsRoleCertificate(
        schema_version=1,
        document_id="doc_test",
        admission="source_has_no_additional_information",
        source_sha256=sha256_bytes(source),
        source_label_sha256=sha256_bytes(label),
        template_sha256=sha256_bytes(template),
        original_source_label_sha256=sha256_bytes(label),
        original_additional_information=(),
        critic_stage_sha256=None,
        changed_scenario_proof_sha256=None,
        fixed_inner_package_category=None,
    )
    (case / "goods-role-certificate.json").write_bytes(
        canonical_json_bytes(certificate.model_dump(mode="json")) + b"\n"
    )
    return root, case


def test_case_loader_accepts_exact_no_aai_certificate(tmp_path: Path) -> None:
    root, _case_dir = _case(tmp_path)
    source, label, template = _case_files(root, "doc_test")
    assert source.endswith(b"TEST ARTICLE\n")
    assert label["documentPatch"]["cargoGroups"][0]["description"] == "TEST ARTICLE"
    assert template.startswith(b'{"template"')


def test_case_loader_rejects_missing_and_stale_certificates(tmp_path: Path) -> None:
    root, case = _case(tmp_path)
    (case / "goods-role-certificate.json").unlink()
    with pytest.raises(ValueError, match=r"goods-role-certificate\.json"):
        _case_files(root, "doc_test")
    root, case = _case(tmp_path / "other")
    (case / "source.txt").write_text("CHANGED\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not pin case bytes"):
        _case_files(root, "doc_test")


def test_case_loader_rejects_old_aai_with_a_false_empty_certificate(tmp_path: Path) -> None:
    root, case = _case(tmp_path)
    label_path = case / "source-label.json"
    label = json.loads(label_path.read_bytes())
    label["documentPatch"]["cargoGroups"][0]["additionalInformation"] = ["LOT NO. 1"]
    label_path.write_bytes(canonical_json_bytes(label))
    cert_path = case / "goods-role-certificate.json"
    certificate = json.loads(cert_path.read_bytes())
    certificate["source_label_sha256"] = sha256_bytes(label_path.read_bytes())
    certificate["original_source_label_sha256"] = certificate["source_label_sha256"]
    cert_path.write_bytes(canonical_json_bytes(certificate))
    with pytest.raises(ValueError, match="still contains additionalInformation"):
        _case_files(root, "doc_test")


def test_description_reference_admission_requires_all_three_pinned_reviews() -> None:
    sha = "a" * 64
    values = dict(
        schema_version=1,
        document_id="doc_test",
        admission="certified_description_reference",
        source_sha256=sha,
        source_label_sha256=sha,
        template_sha256=sha,
        original_source_label_sha256=sha,
        original_additional_information=(
            OriginalAdditionalInformation(
                group_index=0,
                value_index=0,
                value="FABRIC NO:AB1234",
                role="description_labelled_reference",
            ),
        ),
        critic_stage_sha256=sha,
        changed_scenario_proof_sha256=sha,
        fixed_inner_package_category=None,
        reviewed_lexical_contract_sha256=sha,
    )
    assert GoodsRoleCertificate(**values).admission == "certified_description_reference"
    for missing in (
        "critic_stage_sha256",
        "changed_scenario_proof_sha256",
        "reviewed_lexical_contract_sha256",
    ):
        incomplete = {**values, missing: None}
        with pytest.raises(ValidationError, match="lacks complete source and scenario proof"):
            GoodsRoleCertificate(**incomplete)
    with pytest.raises(ValidationError, match="lacks complete source and scenario proof"):
        GoodsRoleCertificate(
            **{
                **values,
                "original_additional_information": (
                    OriginalAdditionalInformation(
                        group_index=0,
                        value_index=0,
                        value="FABRIC NO:AB1234",
                        role="private_lot_identifier_list",
                    ),
                ),
            }
        )
