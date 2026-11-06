import copy
from decimal import Decimal
from pathlib import Path

import pytest

from document_ocr.synthesis.paddle_render import extra_value
from document_ocr.synthesis.paddle_render import coalesced_edits


def fixture(value="18031.34355051170331340561354"):
    return (
        {"physical": {"authority": "explicit_physical_facts", "field": "grossWeight",
                      "unit": "kilogram", "grouped": True, "minPlaces": 3,
                      "displayPlaces": 3, "prefix": "", "suffix": " KG"}},
        {"scenario": {"provenance": {"explicitPhysicalFacts": {
            "grossWeight": {"value": value, "unit": "kilogram"}}}}},
        {"target": {"goodsItemDetails": [{}]}},
    )


def call(extra, candidate, current):
    return extra_value(extra, candidate, current, "sample", None, project_root=Path("."))


def test_private_precision_projects_printed_value_without_changing_facts():
    extra, candidate, current = fixture()
    before = copy.deepcopy((candidate, current))
    text, fact = call(extra, candidate, current)
    assert text == "18,031.344 KG"
    assert fact == {"value": 18031.344, "unit": "kilogram"}
    assert (candidate, current) == before
    extra["physical"]["decimalSeparator"] = ","
    assert call(extra, candidate, current)[0] == "18.031,344 KG"


def test_precision_is_explicit_and_rounds_binary_artifact():
    extra, candidate, current = fixture("5.9999999999999994")
    assert call(extra, candidate, current)[0] == "6.000 KG"
    del extra["physical"]["displayPlaces"]
    assert call(extra, candidate, current)[0] == "5.9999999999999994 KG"


@pytest.mark.parametrize("places", [-1, True, 1.5, "3"])
def test_invalid_precision_is_rejected(places):
    extra, candidate, current = fixture()
    extra["physical"]["displayPlaces"] = places
    with pytest.raises(ValueError, match="display precision"):
        call(extra, candidate, current)


def test_precision_cannot_erase_positive_value_or_conflict_with_conversion():
    extra, candidate, current = fixture("0.0001")
    with pytest.raises(ValueError, match="erases"):
        call(extra, candidate, current)
    extra["physical"]["renderUnit"] = "metric_tonne"
    with pytest.raises(ValueError, match="fact unit"):
        call(extra, candidate, current)


def test_retained_public_measure_cannot_change():
    extra, candidate, current = fixture()
    extra["physical"].update(authority="public_target", goods=0)
    current["target"]["goodsItemDetails"][0]["grossWeight"] = {
        "value": Decimal("18031.34355"), "unit": "kilogram"}
    with pytest.raises(ValueError, match="retained public"):
        call(extra, candidate, current)
    current["target"]["goodsItemDetails"][0]["grossWeight"]["value"] = 18031.344
    assert call(extra, candidate, current)[0] == "18,031.344 KG"


def test_new_shipment_measure_uses_printed_precision_and_rejects_existing_conflict(monkeypatch):
    import document_ocr.synthesis.paddle_render as module
    extra, candidate, current = fixture("6.2333333333333322")
    extra["physical"].update(authority="sampled_shipment", field="volume", unit="cubic_metre")
    monkeypatch.setattr(module, "sampled_shipment_fact", lambda *_: {
        "value": Decimal("6.2333333333333322"), "unit": "cubic_metre"})
    assert call(extra, candidate, current)[1] == {"value": 6.233, "unit": "cubic_metre"}
    current["target"]["goodsItemDetails"][0]["volume"] = {
        "value": 6.2333333333333322, "unit": "cubic_metre"}
    with pytest.raises(ValueError, match="retained public"):
        call(extra, candidate, current)


def test_coalescence_may_explicitly_ignore_case_but_never_distinct_values():
    edits = [{"key": "first", "byteStart": 0, "before": "NINGBO", "after": "SAIPAN"},
             {"key": "second", "byteStart": 10, "before": "NINGBO", "after": "Saipan"}]
    rule = {"members": [{k: e[k] for k in ("key", "byteStart", "before")} for e in edits]}
    with pytest.raises(ValueError, match="distinct sampled"):
        coalesced_edits(edits, [rule])
    rule["comparison"] = "casefold"
    assert coalesced_edits(edits, [rule]) == [edits[0]]
    edits[1]["after"] = "Saipan North"
    with pytest.raises(ValueError, match="distinct sampled"):
        coalesced_edits(edits, [rule])
    rule["comparison"] = "fuzzy"
    with pytest.raises(ValueError, match="unknown coalescence"):
        coalesced_edits(edits, [rule])
