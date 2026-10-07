from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime

import pytest
from test_curated_templates import source_fixture

from document_ocr.synthesis.curated_auxiliary import (
    _render_recipe,
    _validate_recipe,
    augment_auxiliary_blueprint,
    auxiliary_surfaces,
    load_auxiliary_contract,
)
from document_ocr.synthesis.curated_scenarios import ScenarioLocation, ShipmentScenario
from document_ocr.synthesis.curated_templates import (
    compile_sampling_blueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.generators import DeterministicStream


def test_repeated_precarriage_leg_varies_coherently_and_is_distinct_from_main_vessel():
    recipe = {"transport_leg": "precarriage", "source_voyage": "941 S"}
    _validate_recipe(recipe)
    target = {"documentPatch": {"transport": {"vesselName": "MAIN SHIP"}}}
    source = "BIANCA RAMBOW 941 S"
    stream = DeterministicStream(11, "test", "source")
    args = (recipe, scenario(), target, stream, source, {30})
    vessels = ("BIANCA RAMBOW", "MAIN SHIP", "NEW FEEDER")
    result = _render_recipe(*args, vessels)
    assert result == _render_recipe(*args, vessels)
    assert result.startswith("NEW FEEDER ") and not result.endswith("941 S")
    with pytest.raises(ValueError, match="vessel support"):
        _render_recipe(*args)
    with pytest.raises(ValueError, match="declared voyage"):
        _render_recipe(recipe, scenario(), target, stream, "BIANCA RAMBOW 999 X", {30}, vessels)
    with pytest.raises(ValueError, match="exact source voyage"):
        _validate_recipe({"transport_leg": "precarriage"})


def test_transshipment_auxiliary_uses_the_same_registered_route_location():
    sampled = scenario()
    sampled = sampled.model_copy(update={"route_locations": {"transshipmentPort": sampled.origin}})
    recipe = {"text": "VIA {transshipment_port}, {transshipment_country} ({transshipment_code})"}
    args = (
        recipe,
        sampled,
        {"documentPatch": {}},
        DeterministicStream(1, "test", "id"),
        "OLD HUB",
        set(),
    )
    assert _render_recipe(*args) == "VIA MUMBAI, INDIA (IN)"
    with pytest.raises(ValueError, match="unknown auxiliary interpolation"):
        _render_recipe(recipe, scenario(), args[2], args[3], "OLD HUB", set())


def scenario():
    origin = ScenarioLocation(
        name="MUMBAI",
        country="INDIA",
        country_code="IN",
        registry="unlocode_wpi",
        registry_id="INBOM",
    )
    destination = ScenarioLocation(
        name="ROTTERDAM",
        country="NETHERLANDS",
        country_code="NL",
        registry="unlocode_wpi",
        registry_id="NLRTM",
    )
    return ShipmentScenario(
        source_id="sample",
        variant=1,
        origin=origin,
        destination=destination,
        party_localities={"documentPatch.parties.shipper": origin},
        goods_identities=(),
        replacements={},
        cargo={},
        provenance={},
    )


@pytest.mark.parametrize("field", ["text", "prefix", "suffix"])
@pytest.mark.parametrize(
    "invalid", ["FIRST\\nSECOND", "FIRST\\rSECOND", "FIRST\\tSECOND", "FIRST\rSECOND"]
)
def test_auxiliary_document_text_requires_real_line_breaks(tmp_path, field, invalid):
    import yaml

    recipe = {"text": "FIRST\nSECOND", field: invalid}
    path = tmp_path / "auxiliary.yaml"
    path.write_text(
        yaml.safe_dump({"version": 1, "sources": {"sample": {"bindings": {"x": recipe}}}})
    )
    with pytest.raises(ValueError, match="real LF line breaks"):
        load_auxiliary_contract(path)
    recipe[field] = "FIRST\nSECOND"
    path.write_text(
        yaml.safe_dump({"version": 1, "sources": {"sample": {"bindings": {"x": recipe}}}})
    )
    assert (
        load_auxiliary_contract(path)["sources"]["sample"]["bindings"]["x"][field]
        == "FIRST\nSECOND"
    )


def test_caption_dependencies_are_exact_owned_edits_with_replay():
    row, contract, history = source_fixture()
    row["joinedRawText"] += (
        "EGYPTIAN IMPORTER VAT NUMBER: 12-345\nACID-Advance Cargo information declaration: 9999\n"
        "GOODS: BENZOIC ACID\nGOODS: ACID 98 PERCENT\n"
    )
    blueprint = compile_sampling_blueprint(row, history, contract)
    augmented = augment_auxiliary_blueprint(blueprint, {"sources": {"sample": {}}})
    values, receipts = auxiliary_surfaces(
        augmented, scenario(), row["target"], DeterministicStream(3, "test", "sample")
    )
    rendered, labels, proof = render_sampling_blueprint(
        augmented, row["target"], {}, surface_values=values
    )
    assert "IMPORTER REGISTRATION NUMBER: 12-345" in rendered
    assert "IMPORT REFERENCE: 9999" in rendered
    assert "ÉDITÉ" in rendered and "TAX: 12345" in rendered
    assert "GOODS: BENZOIC ACID" in rendered
    assert "GOODS: ACID 98 PERCENT" in rendered
    assert labels == row["target"]
    assert len(receipts) == 2 and proof["unchangedBytesPreserved"]


def test_auxiliary_quote_cannot_consume_lexical_address_or_silently_move():
    row, contract, history = source_fixture()
    blueprint = compile_sampling_blueprint(row, history, contract)
    with pytest.raises(ValueError, match="occurrence mismatch"):
        augment_auxiliary_blueprint(
            blueprint,
            {
                "sources": {
                    "sample": {
                        "spans": [{"quote": "NOT PRESENT", "count": 1, "value": {"text": "NEW"}}]
                    }
                }
            },
        )
    with pytest.raises(ValueError, match="intersects generated lexical"):
        augment_auxiliary_blueprint(
            blueprint,
            {
                "sources": {
                    "sample": {
                        "spans": [{"quote": "12 ROAD", "count": 1, "value": {"text": "NEW"}}]
                    }
                }
            },
        )


def test_repeated_identifiers_share_one_value_and_auxiliary_dates_share_chronology():
    row, contract, history = source_fixture()
    row["joinedRawText"] += "EXPORTER 12-345\nCOPY 12 345\nINVOICE 2024-10-01\nSHIPPED 12/10/2024\n"
    for key, text in [
        ("exporter", "12-345"),
        ("copy", "12 345"),
        ("invoice", "2024-10-01"),
        ("shipped", "12/10/2024"),
    ]:
        source = row["joinedRawText"].encode()
        start = source.index(text.encode())
        history["bindings"].append(
            {
                "logical_key": key,
                "target_paths": [],
                "value_kind": "other_text",
                "occurrences": [
                    {"byte_start": start, "byte_end": start + len(text), "source_text": text}
                ],
            }
        )
    blueprint = compile_sampling_blueprint(row, history, contract)
    blueprint = augment_auxiliary_blueprint(
        blueprint,
        {
            "sources": {
                "sample": {
                    "bindings": {
                        "exporter": {"identifier": "exporter"},
                        "copy": {"identifier": "exporter"},
                        "invoice": {"date": "2024-10-01"},
                        "shipped": {"date": "2024-10-12", "format": "%d/%m/%Y"},
                    }
                }
            }
        },
    )
    new = deepcopy(row["target"])
    new["documentPatch"]["issueDate"] = "2024-11-17"
    values, _ = auxiliary_surfaces(
        blueprint, scenario(), new, DeterministicStream(3, "test", "sample")
    )
    assert values["exporter"] == values["copy"]
    assert values["exporter"] != "12345"
    assert values["invoice"] == "2024-11-01"
    assert values["shipped"] == "12/11/2024"

    # No public date label is necessary to keep source-only dates coherent.
    private_target = deepcopy(row["target"])
    private_target["documentPatch"].pop("issueDate")
    values, _ = auxiliary_surfaces(
        replace(blueprint, target=private_target),
        scenario(),
        private_target,
        DeterministicStream(3, "test", "sample"),
    )
    assert (
        datetime.strptime(values["shipped"], "%d/%m/%Y").date()
        - date.fromisoformat(values["invoice"])
    ).days == 11
    assert "issueDate" not in private_target["documentPatch"]


def test_auxiliary_surface_collision_is_an_error_not_silent_overwrite():
    row, contract, history = source_fixture()
    blueprint = compile_sampling_blueprint(row, history, contract)
    history_copy = deepcopy(dict(blueprint.historical_bindings))
    history_copy["loading"]["auxiliary_recipe"] = {"text": "MUMBAI"}
    blueprint = replace(blueprint, historical_bindings=history_copy)
    with pytest.raises(ValueError, match="two semantic stages"):
        auxiliary_surfaces(
            blueprint,
            scenario(),
            row["target"],
            DeterministicStream(3, "test", "sample"),
            existing={"loading": "ROME"},
        )


def test_auxiliary_public_leaf_requires_agreement_not_just_binding_coverage():
    row, contract, history = source_fixture()
    blueprint = compile_sampling_blueprint(row, history, contract)
    bindings = deepcopy(dict(blueprint.historical_bindings))
    bindings["loading"]["auxiliary_recipe"] = {"text": "ROTTERDAM"}
    blueprint = replace(blueprint, historical_bindings=bindings)
    stream = DeterministicStream(3, "test", "sample")
    with pytest.raises(ValueError, match="disagrees with current public target"):
        auxiliary_surfaces(blueprint, scenario(), row["target"], stream)
    corrected = deepcopy(row["target"])
    corrected["documentPatch"]["route"]["portOfLoading"]["name"] = "Rotterdam"
    values, _ = auxiliary_surfaces(blueprint, scenario(), corrected, stream)
    assert values["loading"] == "ROTTERDAM"
    bindings["loading"]["auxiliary_recipe"] = {"identifier": "private"}
    blueprint = replace(blueprint, historical_bindings=bindings)
    with pytest.raises(ValueError, match="source-only auxiliary generator owns public"):
        auxiliary_surfaces(blueprint, scenario(), corrected, stream)


def test_audited_date_publication_proves_source_and_preserves_identity_before_sampling():
    row, contract, history = source_fixture()
    original = deepcopy(row)
    row["target"]["documentPatch"].pop("issueDate")
    blueprint = compile_sampling_blueprint(row, history, contract)
    date_key = next(
        key
        for key, binding in blueprint.historical_bindings.items()
        if any(o["source_text"] == "2024.10.17" for o in binding["occurrences"])
    )
    declaration = {
        "sources": {
            "sample": {
                "bindings": {
                    date_key: {
                        "date": "2024-10-17",
                        "format": "%Y.%m.%d",
                        "publish_target": "documentPatch.issueDate",
                    }
                }
            }
        }
    }
    augmented = augment_auxiliary_blueprint(blueprint, declaration)
    text, target, proof = render_sampling_blueprint(augmented, augmented.target, {})
    assert text == original["joinedRawText"]
    assert target == original["target"]
    assert proof["unchangedBytesPreserved"]
    assert "issueDate" not in row["target"]["documentPatch"]
    sampled = deepcopy(target)
    sampled["documentPatch"]["issueDate"] = "2024-11-17"
    surfaces, receipts = auxiliary_surfaces(
        augmented, scenario(), sampled, DeterministicStream(3, "test", "sample")
    )
    text, target, _ = render_sampling_blueprint(augmented, sampled, {}, surface_values=surfaces)
    assert "ISSUED: 2024.11.17" in text
    assert target["documentPatch"]["issueDate"] == "2024-11-17"
    assert receipts[0]["recipe"]["publish_target"] == "documentPatch.issueDate"
    with pytest.raises(ValueError, match="already exists"):
        augment_auxiliary_blueprint(augmented, declaration)
    wrong_date = deepcopy(declaration)
    wrong_date["sources"]["sample"]["bindings"][date_key]["date"] = "2024-10-18"
    with pytest.raises(ValueError, match="differs from source"):
        augment_auxiliary_blueprint(blueprint, wrong_date)
    no_owner = replace(blueprint, regions=tuple(r for r in blueprint.regions if r.key != date_key))
    with pytest.raises(ValueError, match="exact source-only owner"):
        augment_auxiliary_blueprint(no_owner, declaration)
    invalid = deepcopy(declaration)
    invalid["sources"]["sample"]["bindings"][date_key]["publish_target"] = "notAField"
    with pytest.raises(ValueError, match="unsupported audited date publication target"):
        augment_auxiliary_blueprint(blueprint, invalid)
    span_publication = {
        "sources": {
            "sample": {
                "spans": [
                    {
                        "quote": "2024.10.17",
                        "count": 1,
                        "value": declaration["sources"]["sample"]["bindings"][date_key],
                    }
                ]
            }
        }
    }
    with pytest.raises(ValueError, match="historical date binding"):
        augment_auxiliary_blueprint(blueprint, span_publication)
    sampled["documentPatch"]["issueDate"] = "2024-11-18"
    # The public target drives one chronology; stale explicit surface values
    # cannot override the audited recipe through another semantic stage.
    with pytest.raises(ValueError, match="two semantic stages"):
        auxiliary_surfaces(
            augmented,
            scenario(),
            sampled,
            DeterministicStream(3, "test", "sample"),
            existing={date_key: "2024.11.17"},
        )
