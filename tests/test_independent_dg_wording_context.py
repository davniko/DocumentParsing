"""DG prose receives frozen physical context instead of inferring it from HS."""

from types import SimpleNamespace

import pytest

from document_ocr.synthesis import independent_campaign as module
from document_ocr.synthesis.curated_campaign import Campaign
from document_ocr.synthesis.curated_wording import WordingField, WordingRequest


@pytest.mark.parametrize("unjoined", [False, True])
@pytest.mark.parametrize("article", [False, True])
def test_dg_wording_gets_frozen_form_hs_authority_and_exact_article_groups(
    monkeypatch, unjoined, article
):
    campaign = module.IndependentCampaign.__new__(module.IndependentCampaign)
    request = {
        "regime": "dg",
        "dangerousGoods": {"recordId": "identity"},
    }
    if unjoined:
        request["nativeDgHsEvidence"] = {"output": {"hs6": "281119"}}
    groups = [["passenger car", "motorbike"], ["passenger car"], ["passenger car"]]
    if article:
        request["cargoPresentation"] = {"articleGroups": groups}
    campaign.requests = {"source:1": request}
    campaign.sampler = SimpleNamespace(hmt_states={"identity": "article" if article else "solid"})
    campaign.numeric = SimpleNamespace(wording_context=lambda _: "")
    campaign.wording_certificates = {}
    campaign.locality_owner_reviews = SimpleNamespace(for_document=lambda _: None)
    campaign.distinct_actor_reviews = SimpleNamespace(for_document=lambda _: None)
    campaign.geography = SimpleNamespace(prepare_request=lambda request, *args: request)
    original = WordingRequest(
        "sample", "base", (WordingField("product", "goods description", "OLD", "Generate."),)
    )
    monkeypatch.setattr(Campaign, "wording_request", lambda *args: original)
    monkeypatch.setattr(module, "wording_scenario", lambda scenario: scenario)
    monkeypatch.setattr(module, "semantic_private_packing", lambda _: {})
    monkeypatch.setattr(module, "resolve_reviewed_locality_assemblies", lambda *a, **kw: ())
    monkeypatch.setattr(module, "resolve_reviewed_distinct_actors", lambda *a: ())
    monkeypatch.setattr(module, "strengthen_presentation_request", lambda r, **kw: r)
    scenario = SimpleNamespace(
        variant=1,
        provenance={
            "independentDraw": {
                "internalCategory": "PACKAGE_SET" if article else "PACKAGE_BOX",
                "package": {"packageQuantity": 3},
                "equipment": [],
            },
            "freshPrivatePacking": {},
        },
    )
    wording = campaign.wording_request(
        SimpleNamespace(document_id="source"), scenario, "sample", {}
    )
    assert f"dangerousGoodsPhysicalForm: {'article' if article else 'solid'}" in wording.context
    assert ("unverified synthetic HS proposal" in wording.context) == unjoined
    assert ("exact pinned registry join" in wording.context) != unjoined
    if article:
        assert "physicalArticleCount: 4" in wording.context
        assert "Every group is one accounting unit, not one vehicle" in wording.context
    else:
        assert "physicalArticleCount" not in wording.context
