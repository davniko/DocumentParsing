"""DG context must not rename products or forge HS/physical provenance."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_dg_context import (
    FORM_STAGE,
    FORM_SYSTEM,
    HS_STAGE,
    HS_SYSTEM,
    HsProposal,
    PhysicalProposal,
    form_prompt,
    form_request,
    hs_output_type,
    hs_prompt,
    hs_request,
    known_physical_form,
    named_eligible,
    physical_form,
    validate_form,
    validate_hs,
    verify_dg_context_evidence,
)


def frozen():
    return {
        "sourceDocumentId": "doc_test",
        "variant": 1,
        "regime": "dg",
        "goods": [{"hs6": "850760", "description": "Lithium ion batteries"}],
        "dangerousGoods": {
            "recordId": "hmt_test",
            "unNumber": "3480",
            "properShippingName": "Lithium ion batteries",
            "class": "9",
            "packingGroup": None,
            "subsidiaryHazards": [],
        },
    }


def test_hs_wire_has_no_product_rewriting_and_is_six_ascii_digits():
    assert set(HsProposal.model_json_schema()["properties"]) == {"hs6"}
    for code in ("850760", None):
        assert HsProposal(hs6=code).hs6 == code
    for code in ("8507.60", "12345", "1234567", "\uff11\uff12\uff13\uff14\uff15\uff16", 850760):
        with pytest.raises(ValidationError):
            HsProposal(hs6=code)
    with pytest.raises(ValidationError):
        HsProposal(hs6="850760", product="renamed")


def test_requests_exclude_mutable_hs_goods_package_equipment_context():
    row = frozen()
    baseline = hs_request(row)
    row.update(goods=[{"hs6": "123456", "description": "wrong"}], selectedCategory="PACKAGE_SET")
    assert hs_request(row) == baseline
    assert form_request(row).model_dump() == baseline.model_dump()
    assert "123456" not in hs_prompt(baseline)
    row["dangerousGoods"]["properShippingName"] = "Changed identity"
    assert baseline.dangerousGoods["properShippingName"] == "Lithium ion batteries"
    assert hs_request(row) != baseline


def test_hs_unknown_membership_and_presentation_are_explicit():
    row = frozen()
    assert validate_hs(row, {"hs6": "850760"}, {"850760"}).hs6 == "850760"
    for code in (None, "123456"):
        with pytest.raises(ValueError):
            validate_hs(row, {"hs6": code}, {"850760"})
    row["cargoPresentation"] = {
        "hsHeadings": ["8703"],
        "excludedHs6": [],
        "packageTypes": ["PACKAGE_SET"],
    }
    with pytest.raises(ValueError, match="presentation"):
        validate_hs(row, {"hs6": "850760"}, {"850760"})
    row["cargoPresentation"]["hsHeadings"] = ["8507"]
    row["cargoPresentation"]["excludedHs6"] = ["850760"]
    with pytest.raises(ValueError, match="presentation"):
        validate_hs(row, {"hs6": "850760"}, {"850760"})


def record(**updates):
    values = dict(
        maritime_eligible=True,
        nos_entry=False,
        technical_name_required=False,
        source_fields={"Non-bulk Packaging": "185"},
        packing_group_code=None,
        exact_hazard_class="9",
        un_number="3480",
    )
    return SimpleNamespace(**(values | updates))


@pytest.mark.parametrize(
    "updates,expected",
    [
        (
            {
                "source_fields": {"Non-bulk Packaging": "202"},
                "packing_group_code": "II",
                "exact_hazard_class": "3",
            },
            "liquid",
        ),
        (
            {
                "source_fields": {"Non-bulk Packaging": "212"},
                "packing_group_code": "II",
                "exact_hazard_class": "6.1",
            },
            "solid",
        ),
        ({"exact_hazard_class": "2.2"}, None),
        ({"exact_hazard_class": "2.2", "source_fields": {"Non-bulk Packaging": "302"}}, "gas"),
        ({"exact_hazard_class": "2.2", "source_fields": {"Non-bulk Packaging": "306"}}, None),
        ({"un_number": "3166"}, "article"),
        ({"un_number": "3171"}, "article"),
        ({}, None),
        ({"technical_name_required": True}, None),
        ({"nos_entry": True}, None),
        ({"maritime_eligible": False}, None),
    ],
)
def test_known_forms_do_not_treat_all_dg_as_liquid_or_solid(updates, expected):
    assert known_physical_form(record(**updates)) == expected


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({}, True),
        ({"nos_entry": True}, False),
        ({"technical_name_required": True}, False),
        ({"maritime_eligible": False}, False),
    ],
)
def test_named_candidate_scope(updates, expected):
    assert named_eligible(record(**updates)) is expected


def test_physical_form_requires_evidence_and_rejects_known_contradiction():
    row = frozen()
    assert physical_form(row, {"hmt_test": "article"}) == "article"
    with pytest.raises(ValueError, match="evidence"):
        physical_form(row, {})
    row["nativeDgPhysicalEvidence"] = {"output": {"form": "article"}}
    assert physical_form(row, {}) == "article"
    with pytest.raises(ValueError, match="contradicts"):
        physical_form(row, {"hmt_test": "liquid"})
    with pytest.raises(ValueError, match="UNKNOWN"):
        validate_form(row, {"form": None}, {})
    row["regime"] = "ambient"
    with pytest.raises(ValueError, match="non-DG"):
        physical_form(row, {})
    row.pop("nativeDgPhysicalEvidence")
    assert physical_form(row, {}) is None


def evidence(root, row, *, physical=False):
    stage, model, system, make_request, prompt = (
        (FORM_STAGE, PhysicalProposal, FORM_SYSTEM, form_request, form_prompt)
        if physical
        else (HS_STAGE, hs_output_type(row, {"850760"}), HS_SYSTEM, hs_request, hs_prompt)
    )
    request = make_request(row).model_dump(mode="json")
    prompt_text = prompt(make_request(row))
    identity = "doc_test:1:" + ("form" if physical else "hs")
    provider = {"model": "test"}
    output = {"form": "article"} if physical else {"hs6": "850760"}
    packet = dict(
        stage=stage,
        identity=identity,
        request=request,
        schema=model.model_json_schema(),
        system=system,
        prompt=prompt_text,
        provider=provider,
    )
    packet_path = root / (stage + "-packet.json")
    atomic_publish_json(packet_path, packet)
    key = digest([stage, identity, model.model_json_schema(), system, prompt_text, provider])
    receipt_path = root / (stage + "-" + key + ".json")
    atomic_publish_json(
        receipt_path,
        dict(
            stage=stage,
            identity=identity,
            promptSha256=digest(prompt_text),
            schemaSha256=digest(model.model_json_schema()),
            output=output,
            error=None,
        ),
    )

    def pin(path):
        return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}

    return dict(
        request=request,
        output=output,
        packet=pin(packet_path),
        packetSha256=digest(packet),
        callReceipts=[pin(receipt_path)],
    )


def verify(tmp_path: Path, row, *, known=None, joined=None):
    verify_dg_context_evidence(
        tmp_path,
        row,
        provider={"model": "test"},
        known_states=known or {},
        hs_codes={"850760", "850650"},
        hs_links=joined or {},
    )


def test_full_immutable_native_hs_and_physical_evidence(tmp_path):
    row = frozen()
    row["nativeDgHsEvidence"] = evidence(tmp_path, row)
    row["nativeDgPhysicalEvidence"] = evidence(tmp_path, row, physical=True)
    verify(tmp_path, row)
    changed = deepcopy(row)
    changed["goods"][0]["hs6"] = "850650"
    with pytest.raises(ValueError, match="received"):
        verify(tmp_path, changed)
    changed = deepcopy(row)
    changed["dangerousGoods"]["properShippingName"] = "Lithium METAL batteries"
    with pytest.raises(ValueError, match="packet"):
        verify(tmp_path, changed)


def test_unjoined_and_unknown_forms_never_silently_pass(tmp_path):
    row = frozen()
    with pytest.raises(ValueError, match="synthetic HS evidence"):
        verify(tmp_path, row)
    with pytest.raises(ValueError, match="physical evidence"):
        verify(tmp_path, row, joined={"hmt_test": {"850760"}})
    verify(tmp_path, row, joined={"hmt_test": {"850760"}}, known={"hmt_test": "article"})


def test_known_join_or_form_rejects_redundant_native_authority(tmp_path):
    row = frozen()
    row["nativeDgHsEvidence"] = evidence(tmp_path, row)
    with pytest.raises(ValueError, match="exact HS join"):
        verify(tmp_path, row, joined={"hmt_test": {"850760"}}, known={"hmt_test": "article"})
    row.pop("nativeDgHsEvidence")
    row["nativeDgPhysicalEvidence"] = evidence(tmp_path, row, physical=True)
    with pytest.raises(ValueError, match="redundant"):
        verify(tmp_path, row, joined={"hmt_test": {"850760"}}, known={"hmt_test": "article"})


def test_missing_or_mutated_wire_receipt_blocks_native_context(tmp_path):
    row = frozen()
    row["nativeDgHsEvidence"] = evidence(tmp_path, row)
    row["nativeDgPhysicalEvidence"] = evidence(tmp_path, row, physical=True)
    receipt = tmp_path / row["nativeDgHsEvidence"]["callReceipts"][0]["path"]
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        verify(tmp_path, row)
