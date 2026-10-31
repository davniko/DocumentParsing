"""Additional group decisions retain source-scoped native relationship authority."""

from types import SimpleNamespace

import pytest

from document_ocr.synthesis.declared_package_preparation import _private_request


@pytest.fixture
def context():
    campaign = SimpleNamespace(
        numeric=SimpleNamespace(allowed_private_relations=lambda sid: ("CONTAINS", "NONE"))
    )
    request = {
        "sourceDocumentId": "test-source",
        "selectedCategory": "PACKAGE_PALLET",
        "nativeChoiceEvidence": {
            "request": {
                "goods": [{"hs6": "520100", "description": "Cotton"}],
                "regime": "ambient",
                "equipment": "general-purpose container",
                "form": "cotton fibre",
                "requiredCategory": "PACKAGE_BALE",
            }
        },
    }
    return campaign, request


def test_native_none_must_be_redecided_for_new_counted_category(context):
    campaign, request = context
    native = _private_request(
        campaign, request, {"origin": "native_typed_private_plan", "privatePackingRelation": "NONE"}
    )
    assert native.countedCategory == "PACKAGE_PALLET"
    assert native.allowedRelations == ["CONTAINS", "NONE"]
    assert "requiredCategory" not in native.model_dump()


def test_source_without_private_relationship_needs_no_new_call(context):
    campaign, request = context
    assert (
        _private_request(
            campaign,
            request,
            {"origin": "reviewed_no_private_relation", "privatePackingRelation": "NONE"},
        )
        is None
    )


def test_mixed_relationship_uses_existing_mixed_accounting_schema(context):
    from document_ocr.synthesis.independent_accounting import MixedPackingRequest

    campaign, request = context
    request["selectedCategory"] = "PACKAGE_PACKAGE"
    native = _private_request(
        campaign, request, {"origin": "native_mixed_accounting", "privatePackingRelation": "MIXED"}
    )
    assert isinstance(native, MixedPackingRequest)
    assert native.countedCategory == "PACKAGE_PACKAGE"


def test_mixed_accounting_rejects_reinterpreting_inner_units_as_public_type(context):
    campaign, request = context
    with pytest.raises(ValueError, match="PACKAGE_PACKAGE"):
        _private_request(
            campaign,
            request,
            {"origin": "native_mixed_accounting", "privatePackingRelation": "MIXED"},
        )
