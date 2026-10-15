from copy import deepcopy
from dataclasses import replace

import pytest

from document_ocr.synthesis.curated import SourceContract
from document_ocr.synthesis.curated_container_ids import (
    format_container_identifier,
    prepare_container_identifier_rendering,
    project_container_identifier,
)
from document_ocr.synthesis.curated_templates import (
    RenderRegion,
    SamplingBlueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.generators import iso6346_check_digit


def identifier(body):
    return body + iso6346_check_digit(body)


OLD = identifier("MCLU507449")
NEW = identifier("MCLU568841")
OTHER = identifier("TEMU123456")
PATH = "documentPatch.containerInformation[0].equipmentIdentifier"
PLACEMENT = "documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].equipmentIdentifier"


def fixture(surfaces=None, *, curated=True):
    surfaces = surfaces or [OLD[:4] + " " + OLD[4:10] + "-" + OLD[-1]]
    raw = "ÉQUIPMENT\n"
    regions = []
    for occurrence, source in enumerate(surfaces):
        raw += "CONTAINER: "
        start = len(raw.encode())
        raw += source
        regions.append(
            RenderRegion(
                "container",
                occurrence,
                start,
                len(raw.encode()),
                source,
                (PATH,),
                "container",
                "container" if curated else None,
                baseline=OLD,
            )
        )
        raw += "/HC40 SEAL: SEAL-0007\n"
    target = {
        "documentPatch": {
            "containerInformation": [
                {
                    "equipmentIdentifier": OLD,
                    "sealNumbers": ["SEAL-0007"],
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }
    contract = SourceContract(
        variables=[
            {
                "key": "container",
                "kind": "container",
                "value": OLD,
                "meaning": "Container identifier",
                "required_literals": [],
                "occurrences": [
                    {"text": surface, "occurrence": 1, "presentation": "text"}
                    for surface in surfaces
                ],
            }
        ],
        targets=[{"path": PATH, "expression": "{container}"}],
        fixed_context="Exact container ID presentation fixture",
    )
    bindings = {"container": {"occurrences": [{"source_text": value} for value in surfaces]}}
    blueprint = SamplingBlueprint("sample", raw, target, contract, tuple(regions), bindings, {}, ())
    sampled = deepcopy(target)
    sampled["documentPatch"]["containerInformation"][0]["equipmentIdentifier"] = NEW
    return blueprint, sampled


@pytest.mark.parametrize("separator", ["", " ", ".", "-", "/", "\n", "\r\n", "\t"])
def test_pure_mask_preserves_separators_and_round_trips(separator):
    source = OLD[:4] + separator + OLD[4:10] + separator + OLD[-1]
    rendered = format_container_identifier(source, OLD, NEW)
    assert rendered == NEW[:4] + separator + NEW[4:10] + separator + NEW[-1]
    assert format_container_identifier(source, OLD, OLD) == source
    assert format_container_identifier(rendered, NEW, OLD) == source


@pytest.mark.parametrize("curated", [False, True])
def test_native_renderer_uses_each_source_format_and_keeps_target_and_neighbors(curated):
    surfaces = [
        OLD[:4] + " " + OLD[4:10] + "-" + OLD[-1],
        OLD,
        OLD[:4] + " " + OLD[4:7] + " " + OLD[7:10] + "." + OLD[-1],
    ]
    blueprint, sampled = fixture(surfaces, curated=curated)
    expected = blueprint.source
    for source in surfaces:
        expected = expected.replace(
            source + "/HC40", format_container_identifier(source, OLD, NEW) + "/HC40", 1
        )
    text, target, proof = render_sampling_blueprint(blueprint, sampled, {})
    assert text == expected and target == sampled
    assert text.count("/HC40 SEAL: SEAL-0007") == 3
    assert proof["changedTargetPaths"] == proof["coveredTargetPaths"] == [PATH]
    assert len(proof["containerIdentifierPresentations"]["records"]) == 3
    assert render_sampling_blueprint(blueprint, sampled, {}) == (text, target, proof)


def test_historical_explicit_surface_is_projected_instead_of_bypassing_mask():
    blueprint, sampled = fixture(curated=False)
    text, target, proof = render_sampling_blueprint(
        blueprint, sampled, {}, surface_values={"container": NEW + " / SOURCE TYPE"}
    )
    styled = format_container_identifier(blueprint.regions[0].source, OLD, NEW)
    assert styled + " / SOURCE TYPE/HC40" in text and target == sampled
    assert proof["containerIdentifierPresentations"]["records"][0]["rendered"] == styled


def test_scoped_composite_projection_preserves_unrelated_type_and_seal_bytes():
    source = OLD[:4] + " " + OLD[4:10] + "/" + OLD[-1]
    frame = "TYPE=40HC | ID=" + NEW + " | SEAL=SEAL-0007 | HC40"
    actual = project_container_identifier(
        frame, source_surface=source, source_identifier=OLD, sampled_identifier=NEW
    )
    assert actual == frame.replace(NEW, format_container_identifier(source, OLD, NEW))
    assert (
        project_container_identifier(
            actual, source_surface=source, source_identifier=OLD, sampled_identifier=NEW
        )
        == actual
    )


def test_source_digit_wrap_and_positions_line_count_survive_native_render():
    source = OLD[:4] + " " + OLD[4:8] + "\n" + OLD[8:]
    blueprint, sampled = fixture([source])
    text, target, proof = render_sampling_blueprint(blueprint, sampled, {})
    assert text.count("\n") == blueprint.source.count("\n")
    assert proof["edits"][0]["after"].count("\n") == source.count("\n")
    assert target == sampled


def test_current_contract_beats_stale_historical_index_co_bindings():
    blueprint, sampled = fixture()
    target = deepcopy(blueprint.target)
    target["documentPatch"]["containerInformation"].append({"equipmentIdentifier": OTHER})
    sampled["documentPatch"]["containerInformation"].append({"equipmentIdentifier": OTHER})
    stale = PATH.replace("[0]", "[1]")
    blueprint = replace(
        blueprint,
        target=target,
        regions=(replace(blueprint.regions[0], target_paths=(PATH, stale)),),
    )
    text, actual, proof = render_sampling_blueprint(blueprint, sampled, {})
    assert actual == sampled and OTHER not in text
    assert proof["containerIdentifierPresentations"]["records"][0]["targetPaths"] == [PATH]


def test_canonical_placement_identity_is_not_projected_to_formatted_text():
    blueprint, sampled = fixture()
    source = deepcopy(blueprint.target)
    source["documentPatch"]["goodsItemDetails"] = [
        {
            "splitGoodsPlacement": [
                {
                    "equipmentIdentifier": OLD,
                    "packageQuantity": 8,
                }
            ]
        }
    ]
    sampled["documentPatch"]["goodsItemDetails"] = [
        {
            "splitGoodsPlacement": [
                {
                    "equipmentIdentifier": NEW,
                    "packageQuantity": 8,
                }
            ]
        }
    ]
    contract = blueprint.contract.model_copy(
        update={
            "targets": [
                *blueprint.contract.targets,
                blueprint.contract.targets[0].model_copy(update={"path": PLACEMENT}),
            ]
        }
    )
    blueprint = replace(blueprint, target=source, contract=contract)
    text, target, proof = render_sampling_blueprint(blueprint, sampled, {})
    assert target == sampled
    assert (
        target["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"][0][
            "equipmentIdentifier"
        ]
        == NEW
    )
    assert set(proof["containerIdentifierPresentations"]["records"][0]["targetPaths"]) == {
        PATH,
        PLACEMENT,
    }
    assert text != blueprint.source


def test_declared_handling_instruction_uses_canonical_id_without_reformatting_target():
    blueprint, sampled = fixture()
    handling_path = "documentPatch.goodsItemDetails[0].handlingInstructions[0]"
    clause = "RELEASE CONTAINER NO. {container} AFTER ALL BILLS ARE COMPLETED."
    original = deepcopy(blueprint.target)
    original["documentPatch"]["goodsItemDetails"] = [
        {"handlingInstructions": [clause.format(container=OLD)]}
    ]
    sampled["documentPatch"]["goodsItemDetails"] = [
        {"handlingInstructions": [clause.format(container=NEW)]}
    ]
    contract = blueprint.contract.model_copy(
        update={
            "targets": [
                *blueprint.contract.targets,
                blueprint.contract.targets[0].model_copy(
                    update={"path": handling_path, "expression": clause}
                ),
            ]
        }
    )
    blueprint = replace(
        blueprint,
        target=original,
        contract=contract,
        regions=(replace(blueprint.regions[0], target_paths=(PATH, handling_path)),),
    )
    text, target, proof = render_sampling_blueprint(blueprint, sampled, {})
    assert format_container_identifier(blueprint.regions[0].source, OLD, NEW) in text
    assert target == sampled
    assert target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        clause.format(container=NEW)
    ]
    assert proof["containerIdentifierPresentations"]["records"][0]["targetPaths"] == [PATH]
    inconsistent = deepcopy(sampled)
    inconsistent["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = [
        clause.format(container=OLD)
    ]
    with pytest.raises(ValueError, match="expression"):
        render_sampling_blueprint(blueprint, inconsistent, {})


def test_explicit_reviewed_character_alias_stays_separate_from_formatting():
    blueprint, sampled = fixture([OTHER], curated=False)
    alias = identifier("TESU123456")
    blueprint = replace(
        blueprint,
        source=blueprint.source.replace(OTHER, alias),
        regions=(replace(blueprint.regions[0], source=alias),),
        ownership_data={"surfaces": {"container": {"expression": "{" + PATH + "}"}}},
    )
    # OLD is the reviewed target identity, while its physical source is an alias.
    text, target, proof = render_sampling_blueprint(
        blueprint, sampled, {}, surface_values={"container": NEW}
    )
    assert target == sampled and NEW in text
    assert (
        proof["containerIdentifierPresentations"]["records"][0]["basis"]
        == "reviewed_identifier_alias"
    )
    with pytest.raises(ValueError, match="not reversible"):
        format_container_identifier(alias, OLD, NEW)
    with pytest.raises(ValueError, match="reviewed alias policy"):
        prepare_container_identifier_rendering(replace(blueprint, ownership_data={}), sampled)


@pytest.mark.parametrize(
    "surface,old,new",
    [
        (OLD[:-1], OLD, NEW),
        ("CONTAINER " + OLD, OLD, NEW),
        (OLD + " SEAL7", OLD, NEW),
        (OLD[:4] + ":" + OLD[4:], OLD, NEW),
        (OLD, OLD[:3] + "X" + OLD[4:], NEW),
        (OTHER, OLD, NEW),
        (OLD, OLD, NEW[:-1] + str((int(NEW[-1]) + 1) % 10)),
    ],
)
def test_seven_invalid_or_contaminated_masks_fail_closed(surface, old, new):
    with pytest.raises(ValueError):
        format_container_identifier(surface, old, new)


@pytest.mark.parametrize("owner", ["NO CONTAINER", NEW + " SEAL " + NEW, "X" + NEW, NEW + "5"])
def test_ambiguous_missing_or_embedded_identity_is_not_rewritten(owner):
    with pytest.raises(ValueError, match="exactly one"):
        project_container_identifier(
            owner, source_surface=OLD, source_identifier=OLD, sampled_identifier=NEW
        )


def test_identity_replay_is_unchanged_and_plan_context_is_pinned():
    blueprint, sampled = fixture()
    text, target, proof = render_sampling_blueprint(blueprint, blueprint.target, {})
    assert text == blueprint.source and target == blueprint.target and proof["edits"] == []
    assert "containerIdentifierPresentations" not in proof
    plan = prepare_container_identifier_rendering(blueprint, sampled)
    plan.validate_for(blueprint, sampled)
    with pytest.raises(ValueError, match="pinned source/input"):
        plan.validate_for(blueprint, blueprint.target)
    broken = deepcopy(sampled)
    broken["documentPatch"]["containerInformation"].append({"equipmentIdentifier": OTHER})
    with pytest.raises(ValueError, match="topology"):
        prepare_container_identifier_rendering(blueprint, broken)
