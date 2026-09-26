"""Location/payment extraction facts require an OCR-owned source slot."""

from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.host import (
    validate_compiled_location_payment_grounding,
    validate_compiler_location_payment_grounding,
)


def test_route_leaf_cannot_be_semantic_only_or_unbound() -> None:
    target = {"documentPatch": {"route": {"portOfLoading": {"name": "QINGDAO"}}}}
    semantic = (SimpleNamespace(target_path="documentPatch.route.portOfLoading.name"),)
    with pytest.raises(ValueError, match="semantic-only location/payment"):
        validate_compiler_location_payment_grounding(
            source_target=target, drafts=(), semantic_only_target_facts=semantic
        )
    with pytest.raises(ValueError, match="compiled location/payment"):
        validate_compiled_location_payment_grounding(
            source_target=target, template=SimpleNamespace(bindings=())
        )


def test_each_distinct_role_needs_its_own_owner_even_when_values_match() -> None:
    target = {
        "documentPatch": {
            "route": {
                "portOfDischarge": {"name": "ALEXANDRIA"},
                "placeOfDelivery": {"name": "ALEXANDRIA"},
            },
            "freight": {"paymentArrangement": "collect"},
        }
    }
    owners = (
        SimpleNamespace(target_paths=("documentPatch.route.portOfDischarge.name",)),
        SimpleNamespace(target_paths=("documentPatch.freight.paymentArrangement",)),
    )
    with pytest.raises(ValueError, match=r"route\.placeOfDelivery\.name"):
        validate_compiler_location_payment_grounding(
            source_target=target, drafts=owners, semantic_only_target_facts=()
        )
    complete = (
        *owners,
        SimpleNamespace(target_paths=("documentPatch.route.placeOfDelivery.name",)),
    )
    validate_compiler_location_payment_grounding(
        source_target=target, drafts=complete, semantic_only_target_facts=()
    )
    validate_compiled_location_payment_grounding(
        source_target=target, template=SimpleNamespace(bindings=complete)
    )
