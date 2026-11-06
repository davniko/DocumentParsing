import copy
from pathlib import Path

import pytest

from document_ocr.synthesis.paddle_render import (
    extra_value,
    literal_owner_value,
    preserve_boundary_markers,
    project_qualified_owner_countries,
)


def test_public_mass_authority_preserves_value_and_native_format():
    current = {
        "target": {"goodsItemDetails": [{"grossWeight": {"value": 1234.56, "unit": "kilogram"}}]}
    }
    recipe = {
        "owner": "attachment-mass",
        "physical": {
            "authority": "public_target",
            "goods": 0,
            "field": "grossWeight",
            "unit": "kilogram",
            "grouped": True,
            "minPlaces": 2,
            "prefix": "",
            "suffix": "",
        },
    }
    before = copy.deepcopy(current)
    text, value = extra_value(recipe, {}, current, "sample", None, project_root=Path("."))
    assert text == "1,234.56"
    assert value == {"value": 1234.56, "unit": "kilogram"}
    assert current == before
    recipe["physical"]["unit"] = "metric_tonne"
    with pytest.raises(ValueError, match="incompatible sampled unit"):
        extra_value(recipe, {}, current, "sample", None, project_root=Path("."))
    recipe["physical"]["unit"] = "kilogram"
    current["target"]["goodsItemDetails"][0].clear()
    with pytest.raises(KeyError):
        extra_value(recipe, {}, current, "sample", None, project_root=Path("."))


@pytest.mark.parametrize("already", [False, True])
def test_preserve_source_owned_continuation_markers(already):
    value = "New Road+" if already else "New Road"
    assert (
        preserve_boundary_markers(value, "Original Road+", {"prefix": "", "suffix": "+"})
        == "New Road+"
    )
    value = "+GNB" if already else "GNB"
    assert preserve_boundary_markers(value, "+ROAD CHINA", {"prefix": "+", "suffix": ""}) == "+GNB"


@pytest.mark.parametrize(
    "markers",
    [{"prefix": "+", "suffix": ""}, {"prefix": "", "suffix": "ROAD"}, {"prefix": "", "suffix": ""}],
)
def test_markers_cannot_invent_unprinted_or_lexical_content(markers):
    with pytest.raises(ValueError):
        preserve_boundary_markers("New Road", "Old Road", markers)


def test_postal_continuation_deduplicates_whole_components_only():
    receipts = [
        {"owner": "front@0", "after": "Plot 14, Industrial District, New York, USA"},
        {"owner": "tail@1", "after": "Plot 14, Industrial District"},
        {"owner": "country@2", "after": "USA"},
    ]
    keys = ["front", "tail", "country"]
    assert (
        literal_owner_value(receipts, keys, " ", deduplicate_postal_components=True)
        == receipts[0]["after"]
    )
    assert literal_owner_value(receipts, keys, " ").endswith("Industrial District USA")
    receipts[1]["after"] = "York"
    assert literal_owner_value(receipts, keys, " ", deduplicate_postal_components=True).endswith(
        "USA York"
    )


def test_postal_distinct_continuation_is_preserved():
    receipts = [
        {"owner": "a@0", "after": "Road 1, Gharbia, Egypt"},
        {"owner": "b@2", "after": "Tanta, Egypt"},
    ]
    assert (
        literal_owner_value(receipts, ["a", "b"], " ", deduplicate_postal_components=True)
        == "Road 1, Gharbia, Egypt Tanta"
    )


def qualified_fixture():
    return (
        {"route": {"portOfLoading": {"name": "PORT SUDAN"}}, "placeOfIssue": {"name": "KHARTOUM"}},
        {"scenario": {"origin": {"name": "Port Sudan", "country": "Sudan"}}},
        [{"owner": "incoterm@0", "after": "FOB PORT SUDAN, SUDAN"}],
        [
            {
                "key": "incoterm",
                "prefix": "FOB ",
                "separator": ", ",
                "namePath": "/origin/name",
                "countryPath": "/origin/country",
                "targets": ["/route/portOfLoading", "/placeOfIssue"],
            }
        ],
    )


def test_qualified_owner_needs_literal_evidence_and_same_place():
    target, candidate, receipts, actions = qualified_fixture()
    project_qualified_owner_countries(target, candidate, receipts, actions)
    assert target["route"]["portOfLoading"]["country"] == "SUDAN"
    assert "country" not in target["placeOfIssue"]


@pytest.mark.parametrize(
    "defect", ["missing", "stale", "conflict", "ambiguous", "empty", "duplicate"]
)
def test_qualified_owner_rejects_unproved_geography(defect):
    target, candidate, receipts, actions = copy.deepcopy(qualified_fixture())
    if defect == "missing":
        receipts.clear()
    elif defect == "stale":
        receipts[0]["after"] = "FOB SHENZHEN, CHINA"
    elif defect == "conflict":
        target["route"]["portOfLoading"]["country"] = "EGYPT"
    elif defect == "ambiguous":
        receipts.append({"owner": "incoterm@9", "after": "FOB KHARTOUM, SUDAN"})
    elif defect == "empty":
        candidate["scenario"]["origin"]["country"] = ""
    elif defect == "duplicate":
        actions[0]["targets"].append(actions[0]["targets"][0])
    with pytest.raises(ValueError):
        project_qualified_owner_countries(target, candidate, receipts, actions)
