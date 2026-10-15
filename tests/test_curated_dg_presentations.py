"""Registry tuples must fit each separately owned DG printout."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_curated_physical import PREFIX, fixture
from test_curated_scenarios import catalog, source_row

from document_ocr.synthesis.curated_physical import _operational, _packing_group_text
from document_ocr.synthesis.curated_scenarios import SourceCapabilities, _observation
from document_ocr.synthesis.generators import DeterministicStream


def record(*, subsidiaries=(), packing="LOW_DANGER"):
    return SimpleNamespace(
        proper_shipping_name="TEST REGISTRY COMMODITY",
        packing_group_category=packing,
        subsidiary_hazard_categories=subsidiaries,
        un_number="1824",
        hazard_category="CORROSIVE_SUBSTANCES",
        packing_group_code="III" if packing == "LOW_DANGER" else "II",
        exact_hazard_class="8",
        record_id="test_record",
        exact_subsidiary_hazards=tuple(
            "3" if value == "FLAMMABLE_LIQUIDS" else "8" for value in subsidiaries
        ),
    )


def test_identity_matches_source_packing_and_subsidiary_cardinality():
    row = source_row()
    row["target"]["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = [
        {"packingGroupCategory": "LOW_DANGER", "subsidiaryHazardCategories": ["FLAMMABLE_LIQUIDS"]}
    ]
    support = catalog([row])
    selected = record(subsidiaries=("CORROSIVE_SUBSTANCES",))
    support.chemical_candidates = tuple(
        (value, "520512")
        for value in (
            record(subsidiaries=("CORROSIVE_SUBSTANCES",), packing=None),
            record(),
            record(subsidiaries=("CORROSIVE_SUBSTANCES", "FLAMMABLE_LIQUIDS")),
            selected,
        )
    )
    source = _observation(row)
    identities, actual = support._identities(
        source,
        source,
        SourceCapabilities(family="dg_chemical"),
        DeterministicStream(1, "test", "source"),
    )
    assert actual is selected
    assert identities[0].hs6 == "520512"
    support.chemical_candidates = support.chemical_candidates[:-1]
    with pytest.raises(ValueError, match="no eligible identity matching source-owned"):
        support._identities(
            source,
            source,
            SourceCapabilities(family="dg_chemical"),
            DeterministicStream(1, "test", "source"),
        )


def test_sampled_tuple_updates_the_existing_target_slots_without_adding_hidden_ones():
    row = source_row()
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods["dangerousGoods"] = [
        {
            "unNumber": "2903",
            "hazardCategory": "TOXIC_AND_INFECTIOUS_SUBSTANCES",
            "packingGroupCategory": "LOW_DANGER",
            "subsidiaryHazardCategories": ["FLAMMABLE_LIQUIDS"],
        }
    ]
    support = catalog([row])
    selected = record(subsidiaries=("CORROSIVE_SUBSTANCES",), packing="MEDIUM_DANGER")
    support.chemical_candidates = ((selected, "520512"),)
    result = support.sample(
        row, seed=19, variant=1, capabilities=SourceCapabilities(family="dg_chemical")
    )
    actual = result.cargo["goodsItemDetails"][0]["dangerousGoods"][0]
    assert actual == {
        "unNumber": "1824",
        "hazardCategory": "CORROSIVE_SUBSTANCES",
        "packingGroupCategory": "MEDIUM_DANGER",
        "subsidiaryHazardCategories": ["CORROSIVE_SUBSTANCES"],
    }
    assert set(actual) == set(goods["dangerousGoods"][0])
    assert result.provenance["dgPrintedFacts"]["subsidiaryHazards"] == ["8"]


def test_flashpoint_cannot_survive_an_unbacked_registry_identity_change():
    row = source_row()
    row["target"]["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = [
        {"flashPoint": {"temperature": {"value": 12, "unit": "celsius"}}}
    ]
    support = catalog([row])
    source = _observation(row)
    with pytest.raises(ValueError, match="formulation-property contract"):
        support._identities(
            source,
            source,
            SourceCapabilities(family="dg_chemical"),
            DeterministicStream(1, "test", "source"),
        )


def operational_fixture(text="SUBSIDIARY CLASS: 3", *, index=0):
    row, blueprint, scenario, _ = fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = [
        {"subsidiaryHazardCategories": ["FLAMMABLE_LIQUIDS"]}
    ]
    path = f"{PREFIX}.dangerousGoods[0].subsidiaryHazardCategories[{index}]"
    bindings = {"subsidiary": {"target_paths": [path], "occurrences": [{"source_text": text}] * 2}}
    blueprint = replace(
        blueprint,
        target=deepcopy(target),
        historical_bindings=bindings,
        regions=(*blueprint.regions, SimpleNamespace(key="subsidiary", curated_key=None)),
    )
    scenario.provenance["dgPrintedFacts"] = {"subsidiaryHazards": ["5.1"]}
    return blueprint, scenario, target


def test_subsidiary_rendering_preserves_caption_and_all_repeated_occurrences():
    blueprint, scenario, target = operational_fixture()
    surfaces = {}
    _operational(blueprint, scenario, target, surfaces, {})
    assert surfaces == {"subsidiary": ["SUBSIDIARY CLASS: 5.1"] * 2}


def test_second_subsidiary_index_has_its_own_exact_printed_class():
    blueprint, scenario, target = operational_fixture("8", index=1)
    blueprint.target["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"][0][
        "subsidiaryHazardCategories"
    ].append("CORROSIVE_SUBSTANCES")
    scenario.provenance["dgPrintedFacts"]["subsidiaryHazards"] = ["3", "6.1"]
    surfaces = {}
    _operational(blueprint, scenario, target, surfaces, {})
    assert surfaces == {"subsidiary": ["6.1"] * 2}


@pytest.mark.parametrize("text", ["3 / 2903", "SUBSIDIARY UNKNOWN", "SUBSIDIARY CLASS: 6.1"])
def test_subsidiary_renderer_rejects_bad_numeric_ownership(text):
    blueprint, scenario, target = operational_fixture(text)
    with pytest.raises(ValueError, match="subsidiary"):
        _operational(blueprint, scenario, target, {}, {})


def test_subsidiary_renderer_rejects_missing_sampled_slot():
    blueprint, scenario, target = operational_fixture()
    scenario.provenance["dgPrintedFacts"]["subsidiaryHazards"] = []
    with pytest.raises(ValueError, match="lacks the source subsidiary"):
        _operational(blueprint, scenario, target, {}, {})


def test_composite_owner_cannot_silently_rewrite_only_primary_class():
    blueprint, scenario, target = operational_fixture()
    blueprint.historical_bindings["subsidiary"]["target_paths"].append(
        f"{PREFIX}.dangerousGoods[0].hazardCategory"
    )
    with pytest.raises(ValueError, match="one separately owned regulatory field"):
        _operational(blueprint, scenario, target, {}, {})


@pytest.mark.parametrize("text", ["PGIII", "P.G. III", "PG: III", "III"])
def test_packing_group_preserves_compact_or_captioned_source_frame(text):
    row, blueprint, scenario, _ = fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = [
        {"packingGroupCategory": "LOW_DANGER"}
    ]
    path = PREFIX + ".dangerousGoods[0].packingGroupCategory"
    blueprint = replace(
        blueprint,
        target=deepcopy(target),
        historical_bindings={
            "packing": {
                "target_paths": [path],
                "occurrences": [
                    {"source_text": text},
                    {"source_text": text},
                ],
            }
        },
        regions=(*blueprint.regions, SimpleNamespace(key="packing", curated_key=None)),
    )
    scenario.provenance["dgPrintedFacts"] = {"packingGroup": "II"}
    surfaces = {}
    _operational(blueprint, scenario, target, surfaces, {})
    assert surfaces == {"packing": [text.replace("III", "II")] * 2}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("II", "III"),
        ("PGII", "PGIII"),
        ("PG.II", "PG.III"),
        ("P.G. II", "P.G. III"),
        ("PG: II", "PG: III"),
        ("2", "3"),
        ("PACKING GROUP: 2", "PACKING GROUP: 3"),
        ("PACKING GROUP:2", "PACKING GROUP:3"),
        ("PACKINGGROUP=2", "PACKINGGROUP=3"),
        ("PG2", "PG3"),
        ("PG.2", "PG.3"),
        ("  P.G. 2,  ", "  P.G. 3,  "),
    ],
)
def test_each_owned_packing_group_preserves_its_roman_or_numeric_frame(source, expected):
    assert _packing_group_text(source, "II", "III") == expected


@pytest.mark.parametrize("source_code", ["I", "II", "III"])
@pytest.mark.parametrize("sampled_code", ["I", "II", "III"])
def test_all_numeric_packing_group_transitions(source_code, sampled_code):
    assert _packing_group_text(str(len(source_code)), source_code, sampled_code) == str(
        len(sampled_code)
    )


@pytest.mark.parametrize(
    "surface",
    [
        "PACKAGES: 2",
        "UN2",
        "CLASS:2",
        "2.1",
        "PACKING GROUP: 4",
        "IV",
        "PACKING GROUP: II / 2",
        "UN 3463 PG2",
        "2 KG",
        "12",
        "PACKING GROUP: 1",
        "III",
    ],
)
def test_packing_group_rejects_unrelated_composite_or_wrong_source_values(surface):
    with pytest.raises(ValueError, match="exactly its source group"):
        _packing_group_text(surface, "II", "III")


def test_packing_group_rejects_noncanonical_regulatory_arguments():
    with pytest.raises(ValueError, match="canonical"):
        _packing_group_text("2", "2", "III")
    with pytest.raises(ValueError, match="canonical"):
        _packing_group_text("2", "II", "4")


def test_hapag_style_roman_and_numeric_repetitions_change_as_one_regulatory_fact():
    row, blueprint, scenario, _ = fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = [
        {"packingGroupCategory": "MEDIUM_DANGER"}
    ]
    path = PREFIX + ".dangerousGoods[0].packingGroupCategory"
    blueprint = replace(
        blueprint,
        target=deepcopy(target),
        historical_bindings={
            "packing": {
                "target_paths": [path],
                "occurrences": [
                    {"source_text": "II"},
                    {"source_text": "2"},
                ],
            }
        },
        regions=(*blueprint.regions, SimpleNamespace(key="packing", curated_key=None)),
    )
    scenario.provenance["dgPrintedFacts"] = {"packingGroup": "III"}
    surfaces = {}
    _operational(blueprint, scenario, target, surfaces, {})
    assert surfaces == {"packing": ["III", "3"]}
