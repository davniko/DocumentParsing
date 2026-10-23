from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from document_ocr.synthesis.country_presentations import (
    CountryFamilyWeights,
    CountryLanguageContext,
    CountryPresentationConfig,
    Registry,
)
from document_ocr.synthesis.curated import Occurrence, SourceContract, TargetBinding, Variable, flat
from document_ocr.synthesis.curated_auxiliary import (
    augment_auxiliary_blueprint,
    load_auxiliary_contract,
)
from document_ocr.synthesis.curated_countries import (
    bind_postal_country_alias_owners,
    country_code_caption,
    country_frame,
    owned_country_spans,
    prepare_country_rendering,
    present_owned_country,
    terminal_country_conflict,
)
from document_ocr.synthesis.curated_ownership import build_owned_blueprint, ownership_surfaces
from document_ocr.synthesis.curated_templates import (
    RenderRegion,
    SamplingBlueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.generators import DeterministicStream

ROOT = Path(__file__).resolve().parents[1]
ROLE = "documentPatch.parties.shipper"


@pytest.fixture(scope="module")
def registry():
    return Registry.load(ROOT)


def policy(*, alias=0.0, family=None):
    weights = {key: 0.0 for key in CountryFamilyWeights.model_fields}
    if family:
        weights[family] = 1.0
    else:
        weights = {key: 1.0 for key in weights}
    return CountryPresentationConfig(alias_probability=alias, family_weights=weights)


def blueprint(*, repeat=1, country="FRANCE", address="12 RUE TEST FRANCE"):
    source, regions = "", []
    for occurrence in range(repeat):
        source += f"SHIPPER {occurrence}\nEXAMPLE COMPANY\n"
        start = len(source.encode())
        source += address
        regions.append(
            RenderRegion(
                "postal",
                occurrence,
                start,
                len(source.encode()),
                address,
                (ROLE + ".addressLine", ROLE + ".country"),
                "postal",
            )
        )
        source += "\nEND\n"
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "EXAMPLE COMPANY",
                    "addressLine": address,
                    "country": country,
                }
            }
        },
    }
    return SamplingBlueprint(
        "source",
        source,
        target,
        SourceContract(
            variables=[
                Variable(
                    key="company",
                    kind="name",
                    value="EXAMPLE COMPANY",
                    meaning="Unchanged company identity in this country-only fixture",
                    required_literals=[],
                    occurrences=[
                        Occurrence(
                            text="EXAMPLE COMPANY",
                            occurrence=index + 1,
                            presentation="text",
                        )
                        for index in range(repeat)
                    ],
                )
            ],
            targets=[TargetBinding(path=ROLE + ".name", expression="{company}")],
            fixed_context="Only country presentations change in this fixture",
        ),
        tuple(regions),
        {"postal": {"occurrences": [{}] * repeat}},
        {},
        (),
    )


def sampled(*, country="SPAIN", address="19 CALLE TEST SPAIN"):
    return {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "EXAMPLE COMPANY",
                    "addressLine": address,
                    "country": country,
                }
            }
        },
    }


def test_address_target_does_not_expand_salvador_in_street_to_country(registry):
    bp = blueprint()
    postal = replace(bp.regions[0], curated_key="postal")
    contract = bp.contract.model_copy(
        update={
            "variables": [
                *bp.contract.variables,
                Variable(
                    key="postal",
                    kind="postal",
                    value="12 RUE TEST FRANCE",
                    meaning="Complete postal block",
                    required_literals=[],
                    occurrences=[
                        Occurrence(text="12 RUE TEST FRANCE", occurrence=1, presentation="text")
                    ],
                ),
            ],
            "targets": [
                *bp.contract.targets,
                TargetBinding(path=ROLE + ".addressLine", expression="{postal}"),
            ],
        }
    )
    bp = replace(bp, regions=(postal,), contract=contract)
    address = "Km 3 Carretera de San Salvador\nEdificio 2\nCUSCATANCINGO\nEL SALVADOR"
    target = sampled(country="EL SALVADOR", address=" ".join(address.upper().split()))
    projected = plan(bp, target, registry, protected={ROLE: "CUSCATANCINGO"}).apply_regions(
        [(postal, address)],
        variable_values={"postal": address},
    )
    assert projected.target_values[ROLE + ".addressLine"] == " ".join(address.upper().split())
    assert "SAN EL SALVADOR" not in projected.target_values[ROLE + ".addressLine"]


def unowned_postal_alias(*, country="ETHIOPIA", address="12 MAIN ROAD (ET)"):
    bp = blueprint(country=country, address=address)
    return replace(
        bp,
        regions=tuple(replace(r, target_paths=(ROLE + ".addressLine",)) for r in bp.regions),
    )


def test_normalized_country_is_grounded_in_same_party_complete_postal_alias(registry):
    bp = unowned_postal_alias()
    before_target = deepcopy(bp.target)
    actual = bind_postal_country_alias_owners(bp, registry)
    assert actual.regions[0].target_paths == (ROLE + ".addressLine", ROLE + ".country")
    assert actual.target == before_target
    assert actual.source == bp.source
    assert actual.regions[0].start == bp.regions[0].start
    assert actual.regions[0].end == bp.regions[0].end
    assert bp.regions[0].target_paths == (ROLE + ".addressLine",)


@pytest.mark.parametrize("address", ["12 MAIN ROAD (DE)", "12 STREET", "12 ETCHING ROAD"])
def test_wrong_identity_and_substrings_do_not_supply_postal_country_ownership(registry, address):
    bp = unowned_postal_alias(address=address)
    assert bind_postal_country_alias_owners(bp, registry) is bp


@pytest.mark.parametrize(
    "address",
    [
        "12 MAIN ROAD contact@example.et",
        "12 MAIN ROAD https://example.et",
        "12 MAIN ROAD https://example.com/ET",
    ],
)
def test_contact_tokens_are_not_postal_country_witnesses(registry, address):
    bp = unowned_postal_alias(address=address)
    assert bind_postal_country_alias_owners(bp, registry) is bp


def test_country_in_another_party_cannot_supply_missing_owner(registry):
    bp = unowned_postal_alias()
    changed_target = deepcopy(bp.target)
    changed_target["documentPatch"]["parties"]["consignee"] = {"country": "ETHIOPIA"}
    actual = bind_postal_country_alias_owners(replace(bp, target=changed_target), registry)
    assert all(
        "documentPatch.parties.consignee.country" not in r.target_paths for r in actual.regions
    )


def test_partial_address_region_does_not_become_a_complete_country_owner(registry):
    bp = unowned_postal_alias()
    region = replace(bp.regions[0], source="ET", end=bp.regions[0].start + 2)
    bp = replace(bp, regions=(region,))
    assert bind_postal_country_alias_owners(bp, registry) is bp


def test_existing_reviewed_country_owner_is_not_reinterpreted(registry):
    # The reviewed c706 source interprets a bare ARE occurrence as Egypt.
    # This helper must not override an already established source owner.
    bp = blueprint(country="EGYPT", address="12 MAIN ROAD ARE")
    assert bind_postal_country_alias_owners(bp, registry) is bp


def plan(bp, target, registry, *, selected=None, protected=None):
    return prepare_country_rendering(
        bp,
        target,
        registry=registry,
        policy=selected or policy(),
        context_country_codes={
            registry.require(target["documentPatch"]["parties"]["shipper"]["country"])
        },
        stream=DeterministicStream(42, "country-unit", "sample"),
        protected_localities=protected,
    )


@pytest.mark.parametrize(
    ("code", "text"),
    [
        ("CD", "CONGO(DRC)"),
        ("CD", "CONGO(RDC)"),
        ("MM", "MYANMAR(BURMA)"),
        ("GB", "ROYAUME-UNI"),
        ("NL", "PAYS-BAS"),
    ],
)
def test_atomic_country_names_are_not_name_code_frames(registry, code, text):
    assert country_frame(text, code, registry) is None


def test_true_name_code_frame_preserves_both_halves(registry):
    text, receipt = present_owned_country(
        "France (FR)",
        "FR",
        registry,
        policy(),
        random.Random(1),
        alias_opportunity=False,
    )
    assert text == "France (FR)"
    assert receipt["family"] == "composite"
    assert registry.require(text) == "FR"


def test_contradictory_name_code_frame_is_rejected(registry):
    with pytest.raises(ValueError, match="different country identities"):
        country_frame("France (ES)", "FR", registry)


@pytest.mark.parametrize(
    ("text", "code", "expected"),
    [
        ("Garmsar, Iran, Islamic Republic Of", "IR", "Iran, Islamic Republic Of"),
        ("PHILIPPINES, ILOCOS REGION, ROAD 12", "PH", "PHILIPPINES"),
        ("Ukraine Zip:03022", "UA", "Ukraine"),
        ("Brunei Darussalam BA1717", "BN", "Brunei Darussalam"),
        ("Isle Of Man, IM2 4QB", "IM", "Isle Of Man"),
        ("CHAI HK 999077", "HK", "HK"),
    ],
)
def test_country_before_postcode_or_region_is_supported(registry, text, code, expected):
    spans = owned_country_spans(text, code, registry, postal=True)
    assert [span[2] for span in spans] == [expected]


def test_highway_identifier_is_not_a_country(registry):
    text = "ESTRADA FEDERAL BR-116,NO.702,\n- ESTEIO\nRS BRAZIL"
    assert [span[2] for span in owned_country_spans(text, "BR", registry, postal=True)] == [
        "BRAZIL"
    ]


def test_country_inside_exact_locality_is_protected(registry):
    text = "SANTIAGO DE CUBA, CUBA"
    spans = owned_country_spans(
        text, "CU", registry, postal=True, protected_names=("SANTIAGO DE CUBA",)
    )
    assert spans == ((18, 22, "CUBA"),)


def test_country_inside_company_is_protected(registry):
    text = "CHINA MACHINERY LTD\n22 ROAD, CHINA"
    spans = owned_country_spans(
        text, "CN", registry, postal=True, protected_names=("CHINA MACHINERY LTD",)
    )
    assert [span[2] for span in spans] == ["CHINA"]
    assert spans[0][0] == text.rindex("CHINA")


def test_word_wrapped_country_has_exact_original_offsets(registry):
    text = "17 ROAD\nSAINT\nLUCIA"
    assert owned_country_spans(text, "LC", registry, postal=True) == ((8, 19, "SAINT\nLUCIA"),)
    result, _ = present_owned_country(
        "SAINT\nLUCIA",
        "LC",
        registry,
        policy(),
        random.Random(5),
        alias_opportunity=False,
    )
    assert result == "SAINT\nLUCIA"


def test_short_base_name_does_not_destroy_certified_physical_lines(registry):
    text = "IRAN,\nISLAMIC REPUBLIC OF"
    result, receipt = present_owned_country(
        text,
        "IR",
        registry,
        policy(),
        random.Random(5),
        alias_opportunity=False,
    )
    assert result == text
    assert receipt["reason"] == "preferred_name_cannot_preserve_owned_physical_lines"
    assert receipt["nonVaryingSlot"]


def test_shortcode_cannot_replace_multiline_name_with_empty_lines(registry):
    with pytest.raises(ValueError, match="no eligible"):
        present_owned_country(
            "SAINT\nLUCIA",
            "LC",
            registry,
            policy(alias=1.0, family="iso2"),
            random.Random(5),
            alias_opportunity=True,
        )


def test_multiple_independent_named_countries_remain_multiple_spans(registry):
    text = "Gonzaga Brasil CEP: 11060-001\nSantos\nBrazil"
    assert [span[2] for span in owned_country_spans(text, "BR", registry, postal=True)] == [
        "Brasil",
        "Brazil",
    ]


def test_plain_region_and_address_labels_project_together(registry):
    bp, target = blueprint(), sampled()
    prepared = plan(bp, target, registry, selected=policy(alias=1.0, family="iso2"))
    prepared.validate_for(bp, target)
    projection = prepared.apply_regions(
        [(bp.regions[0], "19 CALLE TEST SPAIN")], variable_values={}
    )
    assert projection.regions[0][1] == "19 CALLE TEST ES"
    assert projection.target_values[ROLE + ".addressLine"] == "19 CALLE TEST ES"
    assert projection.target_values[ROLE + ".country"] == "SPAIN"
    assert projection.receipt["inputValues"][ROLE + ".addressLine"] == "19 CALLE TEST SPAIN"
    assert projection.receipt["aliasOpportunity"]


def test_repeated_same_owner_uses_one_presentation_without_casing_codes(registry):
    bp, target = blueprint(repeat=2), sampled()
    prepared = plan(bp, target, registry, selected=policy(alias=1.0, family="iso2"))
    projection = prepared.apply_regions(
        [(region, "19 Calle Test Spain") for region in bp.regions],
        variable_values={},
        render_casing="title",
    )
    assert [text for _, text in projection.regions] == ["19 Calle Test ES"] * 2
    assert projection.target_values[ROLE + ".addressLine"] == "19 CALLE TEST ES"


def test_pinned_input_and_source_ownership_cannot_be_changed(registry):
    bp, target = blueprint(), sampled()
    prepared = plan(bp, target, registry)
    with pytest.raises(ValueError, match="pinned"):
        prepared.validate_for(bp, sampled(address="OTHER STREET SPAIN"))
    with pytest.raises(ValueError, match="pinned"):
        prepared.validate_for(replace(bp, regions=()), target)
    with pytest.raises(ValueError, match="pinned"):
        prepared.validate_for(replace(bp, document_id="another-source"), target)
    wrong_company = sampled()
    wrong_company["documentPatch"]["parties"]["shipper"]["name"] = "CHINA MACHINERY LTD"
    with pytest.raises(ValueError, match="pinned"):
        prepared.validate_for(bp, wrong_company)
    with pytest.raises(ValueError, match="incomplete"):
        prepared.apply_regions([], variable_values={})


def test_present_country_cannot_be_silently_omitted(registry):
    bp, target = blueprint(), sampled()
    prepared = plan(bp, target, registry)
    with pytest.raises(ValueError, match="lacks an owned printed occurrence"):
        prepared.apply_regions([(bp.regions[0], "19 CALLE TEST")], variable_values={})


def test_opaque_country_code_under_literal_caption(registry):
    bp = blueprint(country="HK", address="HONG KONG")
    region = replace(
        bp.regions[0],
        kind="location",
        target_paths=(ROLE + ".country",),
        start=14,
        end=16,
        source="HK",
    )
    bp = replace(bp, source="COUNTRY CODE: HK\n", regions=(region,))
    target = sampled(country="ITALY", address="HONG KONG")
    prepared = plan(bp, target, registry)
    projection = prepared.apply_regions([(region, "It")], variable_values={}, render_casing="title")
    assert projection.regions[0][1] == "IT"
    assert projection.target_values[ROLE + ".country"] == "ITALY"
    assert country_code_caption(bp.source, 14)


def test_unknown_country_and_missing_context_are_errors(registry):
    bp = blueprint()
    with pytest.raises(ValueError, match="unknown"):
        plan(bp, sampled(country="NOT A COUNTRY"), registry)
    with pytest.raises(ValueError, match="explicitly include"):
        prepare_country_rendering(
            bp,
            sampled(),
            registry=registry,
            policy=policy(),
            context_country_codes=(),
            stream=DeterministicStream(42, "test", "sample"),
        )


@pytest.mark.parametrize(
    ("text", "code", "city"),
    [
        ("SAO TOME AND PRINCIPE", "ST", "SAO TOME"),
        ("GUINEA-BISSAU", "GW", "BISSAU"),
    ],
)
def test_country_containing_shorter_city_is_not_excluded(registry, text, code, city):
    assert owned_country_spans(text, code, registry, postal=True, protected_names=(city,))


def test_independent_country_wins_over_equal_locality_literal(registry):
    assert owned_country_spans(
        "Macau-Macao",
        "MO",
        registry,
        protected_names=("Macau",),
    ) == ((6, 11, "Macao"),)


def test_shared_country_city_literal_is_retained_explicitly(registry):
    result, receipt = present_owned_country(
        "DJIBOUTI",
        "DJ",
        registry,
        policy(alias=1.0, family="iso2"),
        random.Random(2),
        alias_opportunity=True,
        protected_names=("DJIBOUTI",),
    )
    assert result == "DJIBOUTI"
    assert receipt["family"] == "retained_protected_literal"


def test_foreign_terminal_country_is_not_masked_by_country_city_word(registry):
    bp = blueprint(country="FRANCE", address="RUE 1 FRANCE")
    target = sampled(
        country="DJIBOUTI", address="RUE MOHAMED DIKI 14 DJIBOUTI FALKLAND ISLANDS (MALVINAS)"
    )
    prepared = plan(bp, target, registry, protected={ROLE: "DJIBOUTI"})
    with pytest.raises(ValueError, match="contradictory foreign country"):
        prepared.apply_regions(
            [(bp.regions[0], target["documentPatch"]["parties"]["shipper"]["addressLine"])],
            variable_values={},
        )


@pytest.mark.parametrize("address", ["SAINT-LEU, RÉUNION FRANCE", "97434, Réunion, France"])
def test_independent_territorial_country_allows_following_affiliation(registry, address):
    assert (
        terminal_country_conflict(address, "RE", registry, protected_names=("SAINT-LEU",)) is None
    )
    bp, target = blueprint(), sampled(country="REUNION", address=address)
    projection = plan(bp, target, registry, protected={ROLE: "SAINT-LEU"}).apply_regions(
        [(bp.regions[0], address)], variable_values={}
    )
    assert projection.target_values[ROLE + ".country"] == "RÉUNION"
    assert projection.regions[0][1].endswith(" FRANCE") or projection.regions[0][1].endswith(
        ", France"
    )


def test_precise_country_declaration_does_not_rewrite_foreign_postal_locality(registry):
    bp = blueprint(country="CHINA", address="12 ROAD HONGKONG")
    source = bp.source + "China(CN)"
    country = RenderRegion(
        "country",
        0,
        len(bp.source.encode()),
        len(source.encode()),
        "China(CN)",
        (ROLE + ".country",),
        "location",
    )
    bp = replace(bp, source=source, regions=(*bp.regions, country))
    target = sampled(country="CHINA", address="12 ROAD HONGKONG")
    projection = plan(bp, target, registry).apply_regions(
        [(bp.regions[0], "12 ROAD HONGKONG"), (country, "China(CN)")], variable_values={}
    )
    assert projection.regions[0][1] == "12 ROAD HONGKONG"
    assert projection.target_values[ROLE + ".country"] == "CHINA"


def test_address_without_party_country_uses_only_existing_coowned_fact(registry):
    bp, target = blueprint(), sampled()
    del bp.target["documentPatch"]["parties"]["shipper"]["country"]
    del target["documentPatch"]["parties"]["shipper"]["country"]
    bp.target["documentPatch"]["placeOfIssue"] = {"country": "FRANCE"}
    target["documentPatch"]["placeOfIssue"] = {"country": "SPAIN"}
    owner = replace(
        bp.regions[0], target_paths=(ROLE + ".addressLine", "documentPatch.placeOfIssue.country")
    )
    bp = replace(bp, regions=(owner,))
    prepared = prepare_country_rendering(
        bp,
        target,
        registry=registry,
        policy=policy(alias=1.0, family="iso2"),
        context_country_codes=("ES",),
        stream=DeterministicStream(42, "test", "sample"),
    )
    projection = prepared.apply_regions([(owner, "19 CALLE TEST SPAIN")], variable_values={})
    assert projection.target_values[ROLE + ".addressLine"] == "19 CALLE TEST ES"
    assert ROLE + ".country" not in projection.target_values
    assert projection.target_values["documentPatch.placeOfIssue.country"] == "SPAIN"


@pytest.mark.parametrize("mode", ["fixed", "missing", "mutable", "wrong_source"])
def test_reviewed_issuer_context_requires_exact_fixed_source_evidence(registry, mode):
    bp, target = blueprint(), sampled()
    sid = "doc_" + "a" * 64
    literal = "CMA CGM ITALY"
    bp = replace(bp, document_id=sid)
    if mode == "fixed":
        bp = replace(bp, source=bp.source + literal)
    elif mode == "mutable":
        start = len(bp.source.encode())
        region = RenderRegion("issuer", 0, start, start + len(literal), literal, (), "other_text")
        bp = replace(bp, source=bp.source + literal, regions=(*bp.regions, region))
    declaration = CountryLanguageContext(
        source_id=("doc_" + "b" * 64) if mode == "wrong_source" else sid,
        literal=literal,
        country_code="IT",
    )
    selected = policy().model_copy(update={"reviewed_contexts": (declaration,)})
    if mode in {"missing", "mutable"}:
        with pytest.raises(ValueError, match=r"absent|mutable"):
            plan(bp, target, registry, selected=selected)
    else:
        prepared = plan(bp, target, registry, selected=selected)
        assert ("IT" in prepared.context_country_codes) == (mode == "fixed")
        evidence = [item for item in prepared.language_evidence if item.get("literal") == literal]
        assert bool(evidence) == (mode == "fixed")
        if evidence:
            assert (
                bp.source.encode()[
                    evidence[0]["fixedOccurrences"][0]["sourceByteStart"] : evidence[0][
                        "fixedOccurrences"
                    ][0]["sourceByteEnd"]
                ]
                == literal.encode()
            )


def test_native_renderer_projects_country_and_address_with_exact_receipts(registry):
    bp, target = blueprint(), sampled()
    prepared = plan(bp, target, registry, selected=policy(alias=1.0, family="iso2"))
    text, projected, proof = render_sampling_blueprint(
        bp,
        target,
        {},
        surface_values={"postal": "19 CALLE TEST SPAIN"},
        country_rendering=prepared,
    )
    assert text == "SHIPPER 0\nEXAMPLE COMPANY\n19 CALLE TEST ES\nEND\n"
    assert projected["documentPatch"]["parties"]["shipper"]["country"] == "SPAIN"
    assert projected["documentPatch"]["parties"]["shipper"]["addressLine"] == "19 CALLE TEST ES"
    assert (
        proof["countryPresentations"]["inputValues"][ROLE + ".addressLine"] == "19 CALLE TEST SPAIN"
    )
    assert proof["edits"][0]["before"] == bp.regions[0].source
    assert proof["edits"][0]["after"] == "19 CALLE TEST ES"


def test_native_no_plan_source_identity_is_unchanged(registry):
    bp = blueprint()
    text, target, proof = render_sampling_blueprint(bp, bp.target, {})
    assert text == bp.source
    assert target == bp.target
    assert proof["edits"] == []


def test_split_country_frame_retains_name_and_opaque_code_sides(registry):
    bp = blueprint()
    source = "France(FR)"
    left = RenderRegion("name", 0, 0, 6, "France", (ROLE + ".country",), "location")
    right = RenderRegion("code", 0, 7, 9, "FR", (ROLE + ".country",), "location")
    bp = replace(bp, source=source, regions=(left, right), historical_bindings={})
    target = sampled()
    prepared = plan(bp, target, registry)
    projection = prepared.apply_regions([(left, "Spain"), (right, "ES")], variable_values={})
    assert [text for _, text in projection.regions] == ["Spain", "ES"]
    assert projection.receipt["owners"][1]["edits"][0]["selection"]["family"] == "iso2"


@pytest.mark.parametrize("aliases", [False, True])
def test_address_literal_frame_uses_its_separately_owned_name_and_code(registry, aliases):
    bp = blueprint(address="12 RUE TEST FRANCE(FR)")
    source = "12 RUE TEST France(FR)"
    postal = RenderRegion("postal", 0, 0, 11, "12 RUE TEST", (ROLE + ".addressLine",), "postal")
    name = RenderRegion("country_name", 0, 12, 18, "France", (ROLE + ".country",), "location")
    code = RenderRegion("country_code", 0, 19, 21, "FR", (ROLE + ".country",), "location")
    contract = bp.contract.model_copy(
        update={
            "variables": [
                *bp.contract.variables,
                Variable(
                    key="postal_core",
                    kind="postal",
                    value="12 RUE TEST",
                    meaning="Owned address before separately printed country frame",
                    required_literals=[],
                    occurrences=[Occurrence(text="12 RUE TEST", occurrence=1, presentation="text")],
                ),
            ],
            "targets": [
                *bp.contract.targets,
                TargetBinding(path=ROLE + ".addressLine", expression="{postal_core} FRANCE(FR)"),
            ],
        }
    )
    bp = replace(
        bp, source=source, regions=(postal, name, code), contract=contract, historical_bindings={}
    )
    target = sampled(address="19 CALLE TEST SPAIN(ES)")
    prepared = plan(bp, target, registry, selected=policy(alias=float(aliases)))
    projection = prepared.apply_regions(
        [(postal, "19 CALLE TEST"), (name, "Spain"), (code, "ES")],
        variable_values={"postal_core": "19 CALLE TEST"},
    )
    printed_name, printed_code = projection.regions[1][1], projection.regions[2][1]
    assert (
        projection.target_values[ROLE + ".addressLine"]
        == ("19 CALLE TEST " + printed_name + "(" + printed_code + ")").upper()
    )
    frames = projection.receipt["addressProjections"][0]["splitCountryFrames"]
    assert frames[0]["owners"] == [["country_name", 0], ["country_code", 0]]
    assert frames[0]["decision"] == "first_complete_source_frame"


def test_joint_literal_target_protects_only_its_owned_country_occurrence(registry):
    bp, target = blueprint(), sampled()
    path = "documentPatch.goodsItemDetails[0].handlingInstructions[0]"
    source_literal = "21 DAYS FREETIME AT FRANCE"
    sampled_literal = "21 DAYS FREETIME AT SPAIN"
    bp.target["documentPatch"]["goodsItemDetails"] = [{"handlingInstructions": [source_literal]}]
    target["documentPatch"]["goodsItemDetails"] = [{"handlingInstructions": [sampled_literal]}]
    source = bp.source + source_literal
    clause = RenderRegion(
        "freetime",
        0,
        len(bp.source.encode()),
        len(source.encode()),
        source_literal,
        (path, ROLE + ".country"),
        "other_text",
    )
    bp = replace(bp, source=source, regions=(*bp.regions, clause))
    prepared = plan(bp, target, registry, selected=policy(alias=1.0, family="iso2"))
    projection = prepared.apply_regions(
        [(bp.regions[0], "19 CALLE TEST SPAIN"), (clause, sampled_literal)], variable_values={}
    )
    assert projection.regions[0][1] == "19 CALLE TEST ES"
    assert projection.regions[1][1] == sampled_literal
    assert path not in projection.target_values
    assert projection.receipt["protectedLiteralValues"] == {path: sampled_literal}
    assert projection.receipt["owners"][1]["edits"][0]["selection"]["protectedTargetPaths"] == [
        path
    ]
    wrong = deepcopy(target)
    wrong["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"][0] = "OTHER CLAUSE SPAIN"
    with pytest.raises(ValueError, match="pinned"):
        prepared.validate_for(bp, wrong)
    with pytest.raises(ValueError, match="joint literal target"):
        prepared.apply_regions(
            [(bp.regions[0], "19 CALLE TEST SPAIN"), (clause, "OTHER CLAUSE SPAIN")],
            variable_values={},
        )


def test_unrelated_literal_country_name_does_not_freeze_other_occurrences(registry):
    bp, target = blueprint(), sampled()
    bp.target["documentPatch"]["goodsItemDetails"] = [{"origin": {"name": "FRANCE"}}]
    target["documentPatch"]["goodsItemDetails"] = [{"origin": {"name": "SPAIN"}}]
    prepared = plan(bp, target, registry, selected=policy(alias=1.0, family="iso2"))
    projection = prepared.apply_regions(
        [(bp.regions[0], "19 CALLE TEST SPAIN")], variable_values={}
    )
    assert projection.regions[0][1] == "19 CALLE TEST ES"
    assert projection.receipt["protectedLiteralValues"] == {}


def test_contradictory_coupling_requires_separate_printed_country(registry):
    bp, target = blueprint(), sampled()
    bp.target["documentPatch"]["placeOfIssue"] = {"country": "FRANCE"}
    target["documentPatch"]["placeOfIssue"] = {"country": "PORTUGAL"}
    postal = replace(
        bp.regions[0],
        target_paths=(*bp.regions[0].target_paths, "documentPatch.placeOfIssue.country"),
    )
    source = bp.source + "PORTUGAL"
    issue = RenderRegion(
        "issue",
        0,
        len(bp.source.encode()),
        len(source.encode()),
        "PORTUGAL",
        ("documentPatch.placeOfIssue.country",),
        "location",
    )
    bp = replace(bp, source=source, regions=(postal, issue))
    prepared = prepare_country_rendering(
        bp,
        target,
        registry=registry,
        policy=policy(),
        context_country_codes=("ES", "PT"),
        stream=DeterministicStream(42, "test", "sample"),
    )
    projection = prepared.apply_regions(
        [(postal, "19 CALLE TEST SPAIN"), (issue, "Portugal")], variable_values={}
    )
    assert projection.receipt["owners"][0]["ignoredCouplingPaths"] == [
        "documentPatch.placeOfIssue.country"
    ]
    without = replace(bp, regions=(postal,))
    held = prepare_country_rendering(
        without,
        target,
        registry=registry,
        policy=policy(),
        context_country_codes=("ES", "PT"),
        stream=DeterministicStream(42, "test", "sample"),
    )
    with pytest.raises(ValueError, match="lacks an owned printed occurrence"):
        held.apply_regions([(postal, "19 CALLE TEST SPAIN")], variable_values={})


@pytest.fixture(scope="module")
def reviewed_route_blueprints():
    root = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
    ownership = yaml.safe_load((root / "ownership.yaml").read_bytes())
    auxiliary = load_auxiliary_contract(root / "auxiliary.yaml")
    result = {}
    for prefix in ("a5265d59", "978a3990", "ebb856"):
        case = next((root / "cases").glob("doc_" + prefix + "*"))
        row = {
            "documentId": case.name,
            "joinedRawText": (case / "source.txt").read_text(),
            "target": json.loads((case / "target.json").read_bytes()),
        }
        contract = SourceContract.model_validate(
            json.loads((case / "contract.json").read_bytes())["contract"]
        )
        bp = build_owned_blueprint(
            row,
            json.loads((case / "template.json").read_bytes()),
            contract,
            ownership["sources"][case.name],
        )
        result[prefix] = augment_auxiliary_blueprint(bp, auxiliary)
    return result


def test_reviewed_freetime_literal_stays_grounded_while_other_countries_vary(
    registry, reviewed_route_blueprints
):
    bp = reviewed_route_blueprints["ebb856"]
    leaves = flat(bp.target)
    prepared = prepare_country_rendering(
        bp,
        bp.target,
        registry=registry,
        policy=policy(alias=1.0),
        context_country_codes={
            registry.require(value) for path, value in leaves.items() if path.endswith(".country")
        },
        stream=DeterministicStream(42, "reviewed-literal-test", "ebb856"),
    )
    text, target, proof = render_sampling_blueprint(bp, bp.target, {}, country_rendering=prepared)
    key = "agent:operational:freetime_clause"
    owner = next(item for item in proof["countryPresentations"]["owners"] if item["key"] == key)
    literal = target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"][0]
    assert owner["before"] == owner["after"] == literal
    assert literal in text
    assert owner["edits"][0]["selection"]["family"] == "retained_literal_target"
    assert any(
        item["before"] != item["after"]
        for item in proof["countryPresentations"]["owners"]
        if item["key"] != key
    )


@pytest.mark.parametrize("prefix", ["a5265d59", "978a3990"])
def test_reviewed_route_country_rebase_keeps_native_source_noop(reviewed_route_blueprints, prefix):
    bp = reviewed_route_blueprints[prefix]
    text, target, proof = render_sampling_blueprint(bp, bp.target, {})
    assert text == bp.source
    assert target == bp.target
    assert proof["edits"] == []


@pytest.mark.parametrize("prefix", ["a5265d59", "978a3990"])
@pytest.mark.parametrize("aliases", [False, True])
def test_reviewed_route_country_uses_own_commercial_alias_not_notify(
    registry, reviewed_route_blueprints, prefix, aliases
):
    bp = reviewed_route_blueprints[prefix]
    target = deepcopy(bp.target)
    patch = target["documentPatch"]
    destination = "AJMAN" if aliases else "SAINT JOHN'S"
    country, code = ("UNITED ARAB EMIRATES", "AE") if aliases else ("ANTIGUA AND BARBUDA", "AG")
    for role in ("portOfDischarge", "placeOfDelivery"):
        if role in patch["route"]:
            patch["route"][role]["name"] = destination
            patch["route"][role]["country"] = country
    notify = patch["parties"]["notifyParties"][0]
    notify["country"] = "FRANCE"
    notify["addressLine"] = notify["addressLine"].replace("EGYPT", "FRANCE")
    leaves = flat(target)
    scenario = SimpleNamespace(
        party_localities={
            path.rsplit(".", 1)[0]: SimpleNamespace(country_code=registry.require(value))
            for path, value in leaves.items()
            if path.startswith("documentPatch.parties.") and path.endswith(".country")
        }
    )
    surfaces = ownership_surfaces(bp, target, scenario)
    prepared = prepare_country_rendering(
        bp,
        target,
        registry=registry,
        policy=policy(alias=float(aliases), family="iso2"),
        context_country_codes={
            registry.require(value) for path, value in leaves.items() if path.endswith(".country")
        },
        stream=DeterministicStream(42, "route-country-test", prefix + str(aliases)),
        protected_localities={"documentPatch.route.portOfDischarge": destination},
    )
    rendered, projected, proof = render_sampling_blueprint(
        bp,
        target,
        {},
        surface_values=surfaces,
        country_rendering=prepared,
        certified_country_codes={
            path: registry.require(value)
            for path, value in leaves.items()
            if path.endswith(".country")
        },
    )
    key = (
        "agent:auxiliary:delivery_term"
        if prefix == "a5265d59"
        else "anchor:documentPatch.cargoGroups[0].marksAndNumbers[2]"
    )
    owners = [owner for owner in proof["countryPresentations"]["owners"] if owner["key"] == key]
    assert len(owners) == 1
    owner = owners[0]
    assert owner["code"] == code
    assert owner["countryPaths"] == sorted(
        "documentPatch.route." + role + ".country"
        for role in ("portOfDischarge", "placeOfDelivery")
        if role in patch["route"]
    )
    assert owner["after"] == ("CIP " if prefix == "a5265d59" else "") + destination + ", " + (
        code if aliases else country
    )
    assert owner["after"] in rendered
    assert projected["documentPatch"]["route"]["portOfDischarge"]["country"] == country
    assert projected["documentPatch"]["parties"]["notifyParties"][0]["country"] == "FRANCE"
    postal_owners = [
        item for item in proof["countryPresentations"]["owners"] if item["postalOwner"]
    ]
    assert not any(
        "documentPatch.route.portOfDischarge.country" in item["countryPaths"]
        for item in postal_owners
    )
