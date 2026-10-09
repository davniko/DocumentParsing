"""Boundary policy guards preserve product specifications and hold genuine uncertainty."""

import pytest

from document_ocr.label_schemas.bill_of_lading_v7 import GoodsItemDetailsV7
from document_ocr.labeling_agents.direct_cargo import CargoProduct
from document_ocr.labeling_agents.direct_models import SectionReview
from document_ocr.synthesis.curated_wording import (
    WordingBatch,
    WordingField,
    WordingRequest,
    has_shipment_accounting,
    review_output_type,
    unpack_review,
    validate_wording,
)


@pytest.mark.parametrize(
    "text",
    [
        "YARN\nTOTAL: 491 CARTONS",
        "RESIN\nTOTAL: 802 BAGS.",
        "LUMBER\nTOTAL: 17565 KG, 25.7 CBM",
        "AUTOMOTIVE PARTS. TOTAL: 1718 CARTONS.",
        "CHARCOAL\n688 CARTONS",
        "COTTON YARN\n410 CARTONS",
        "INTEGRATED CIRCUITS\n146 BOXES PER SHIPMENT UNIT COUNT, PROTECTIVE PACKING",
        "PRODUCT DETAILS\n300 CARTONS OF COMPONENTS",
        "1,718 boxes vehicle braking systems and components",
        "COFFEE TOTAL PACKAGES - 9",
        "NEW GOODS\nGross Weight: 1250 KG",
    ],
)
def test_generated_standalone_accounting_cannot_pass_wording(text):
    request = WordingRequest(
        "s", "Products only", (WordingField("g", "goods description", "OLD GOODS", "Product"),)
    )
    output = WordingBatch.model_validate(
        {"shipments": [{"sample_id": "s", "values": [{"key": "g", "text": text}]}]}
    )
    with pytest.raises(ValueError, match="shipment accounting"):
        validate_wording(output, [request])


@pytest.mark.parametrize(
    "text",
    [
        "WALNUTS PACKED IN 15KG BAGS",
        "S.J 95%\nCOT 30/1 + 5% SPANDEX",
        "DOMESTIC SEWING MACHINE\nMODEL: JA2-1 WITHOUT FEED\nDROP BLACK COLOUR",
        "1012322163DXH DEGREE 1 ELBOW, REDUCER, SOCKET",
        "BENNI BRAND, SKIMMED MILK POWDER TOTAL 1000 BAGS , 25 KG EACH",
        "25,925 MTCHAMBRIL WHITE WOODFREE WRITING AND PRINTING PAPER",
        "PSN: LIGHTERS",
        "1 UNIT NEW RANGE ROVER",
        "100% COTTON",
        "HYDRAULIC PUMP TOTAL DISPLACEMENT 100 CC",
        "TOTAL PERFORMANCE ADDITIVE",
        "ACID NUMBER 15 MG KOH/G",
        "MODEL NP-10, CAPACITY 500 LITRES",
    ],
)
def test_accounting_guard_is_not_a_product_number_or_total_word_blacklist(text):
    assert not has_shipment_accounting(text)


def test_cargo_map_and_extraction_share_main_passage_definition():
    definition = GoodsItemDetailsV7.model_fields["description"].description
    assert CargoProduct.model_fields["description"].description == definition
    assert "main product-description block" in definition
    assert "detached tracking/batch" in definition
    assert "across pages" in definition


def test_review_uncertainty_retains_recommendation_and_cannot_be_a_pass():
    finding = dict(
        field="documentPatch.goodsItemDetails[0].description",
        issue="ambiguous",
        explanation="Two plausible product boundaries remain.",
        suggestedCorrection="Review the adjacent column in the source layout.",
    )
    # The real extraction review already has a typed unresolved state.
    review = SectionReview(
        status="unresolved", explanation="Boundary undecided.", findings=[finding]
    )
    assert review.findings[0].suggestedCorrection
    with pytest.raises(ValueError, match="pass requires no findings"):
        SectionReview(status="pass", explanation="Wrong pass.", findings=review.findings)
    schema = review_output_type(1)
    result = unpack_review(
        schema.model_validate(
            {
                "s0": {
                    "findings": [
                        dict(
                            field="description",
                            problem="Unresolved: product continuation or tracking table.",
                            evidence="PRODUCT / BATCH",
                            correction="Inspect the source boundary before publication.",
                        )
                    ]
                }
            }
        ),
        ["sample"],
    )
    assert len(result.findings) == 1
    assert result.findings[0].correction.startswith("Inspect")
