"""Regression coverage for reviewed route and package-context repairs."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from document_ocr.synthesis.curated import SourceContract
from document_ocr.synthesis.curated_ownership import apply_dependent_text, load_owned_blueprint
from document_ocr.synthesis.curated_templates import render_sampling_blueprint

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
DATASET = CATALOG / "source-authorities/larger-controlled-legacy-closure-20261020/source-dataset"
OWNERSHIP = CATALOG / "ownership.yaml"
SOURCE_00 = "doc_00a165a2adf7156bdecd557c885de531a2a69a98a920a4e029d491bd5d50b3ed"
SOURCE_11 = "doc_11da8a8c9f0e33daafa39cf0a8c9d216edb9caa1b170e201c1f12559fed234d8"
SOURCE_65 = "doc_65a0b16080db9f6815d5315041fdcc185cb1165f4742a299e9c4961591b990df"
SOURCE_301 = "doc_30116fcc85be6af6fe3d2ccb2e360110396cf7dea36f7322b2b5ae1ba51eef15"
SOURCE_612 = "doc_612b6a124061f19b321ec49a68453e4a1460fa84e6c95ae0a71c7e13f83320dc"
SOURCE_0BA = "doc_0ba3ad4fcd4e713df5520786bd64916559b17258db1ba013ef90ec0d38e45a9f"
SOURCE_2B = "doc_2b88d69f06717c214c31562ab8a6bf6fce245d11525aa7f492539172e2513498"
SOURCE_37 = "doc_37c9635406d2aa9b1fd0b710d4c975609ca479400f6a43d0b1fe7fe5d40281f7"
SOURCE_069 = "doc_0690f5b97c27e36615d1be2063865ff50b6eb5bfaf98aeca96d45f07c5607d88"


@pytest.fixture(scope="module")
def rows() -> dict[str, dict]:
    return {
        row["documentId"]: row
        for row in map(json.loads, (DATASET / "train.jsonl").read_text().splitlines())
        if row["documentId"]
        in {
            SOURCE_00, SOURCE_11, SOURCE_65, SOURCE_301,
            SOURCE_612, SOURCE_0BA, SOURCE_2B, SOURCE_37, SOURCE_069,
        }
    }


def _blueprint(rows: dict[str, dict], source_id: str):
    case = CATALOG / "cases" / source_id
    contract = SourceContract.model_validate(
        json.loads((case / "contract.json").read_text())["contract"]
    )
    historical = json.loads((case / "template.json").read_text())
    return load_owned_blueprint(rows[source_id], historical, contract, OWNERSHIP)


def test_egypt_specific_customs_warehouse_follows_the_sampled_destination(rows):
    blueprint = _blueprint(rows, SOURCE_00)
    target = deepcopy(rows[SOURCE_00]["target"])
    target["documentPatch"]["route"]["portOfDischarge"]["name"] = "SAN FERNANDO, LUZON"

    updated = apply_dependent_text(blueprint, target, phase="scenario")
    expected = (
        "CARGO IN TRANSIT TO (SAN FERNANDO, LUZON) CUSTOMS WAREHOUSE ON RECEIVER'S ACCOUNT & RISK"
    )
    assert updated["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [expected]
    assert target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"][0].startswith(
        "CARGO IN TRANSIT TO (BANHA-"
    )
    assert apply_dependent_text(blueprint, updated, phase="scenario") == updated

    text, rendered, proof = render_sampling_blueprint(
        blueprint,
        updated,
        {},
        surface_values={
            "anchor:documentPatch.route.portOfDischarge.name": "SAN FERNANDO, LUZON SEAPORT",
            "reviewed:route_owned_handling": expected,
        },
    )
    assert expected in " ".join(text.upper().split())
    assert "BANHA" not in text
    assert rendered == updated
    assert (
        "documentPatch.goodsItemDetails[0].handlingInstructions[0]" in proof["changedTargetPaths"]
    )


def test_source_only_wooden_package_status_is_removed_as_one_complete_claim(rows):
    blueprint = _blueprint(rows, SOURCE_11)
    omission = next(
        region
        for region in blueprint.regions
        if region.key == "reviewed:obsolete_wooden_package_status"
    )
    assert omission.source == "WOODEN PACKAGE: NOT APPLICABLE"
    assert omission.target_paths == ()
    assert omission.allow_empty

    text, rendered, proof = render_sampling_blueprint(
        blueprint,
        rows[SOURCE_11]["target"],
        {},
        surface_values={"reviewed:obsolete_wooden_package_status": [""]},
    )
    assert "WOODEN PACKAGE" not in text
    assert "NOT APPLICABLE" not in text
    assert rendered == rows[SOURCE_11]["target"]
    assert {
        (edit["before"], edit["after"])
        for edit in proof["edits"]
        if edit["key"] == "reviewed:obsolete_wooden_package_status"
    } == {("WOODEN PACKAGE: NOT APPLICABLE", "")}


def test_two_container_equipment_cells_are_not_one_shared_receipt(rows):
    from document_ocr.synthesis.curated_equipment_inventory import compile_equipment_inventory

    blueprint = _blueprint(rows, SOURCE_65)
    inventory = compile_equipment_inventory(blueprint)
    spans = {span.key: span for span in inventory.spans}
    assert spans["equipment_row_0"].indices == (0,)
    assert spans["equipment_row_1"].indices == (1,)
    assert spans["equipment"].indices == (0, 1)
    assert spans["equipment"].source == "2 x 40HC"
    assert inventory.uniform_aggregate is None
    assert inventory.row_count == 2
    assert blueprint.source == rows[SOURCE_65]["joinedRawText"]


def test_complete_inline_locality_frames_preserve_source_and_both_roles(rows):
    blueprint = _blueprint(rows, SOURCE_301)
    assert blueprint.source == rows[SOURCE_301]["joinedRawText"]
    assert blueprint.target == rows[SOURCE_301]["target"]
    for suffix, role in (
        ("loading", "portOfLoading"),
        ("receipt", "placeOfReceipt"),
        ("discharge", "portOfDischarge"),
    ):
        key = "locality_pair_" + suffix
        paths = {f"documentPatch.route.{role}.name", f"documentPatch.route.{role}.country"}
        region = next(r for r in blueprint.regions if r.key == key)
        assert set(region.target_paths) == paths
        assert region.source in {"LIANYUNGANG, CHINA", "ALEXANDRIA TMT, EGYPT"}
        assert "expression" in blueprint.ownership_data["surfaces"][key]


def test_package_type_owns_shipment_total_and_all_three_container_rows(rows):
    blueprint = _blueprint(rows, SOURCE_2B)
    regions = [region for region in blueprint.regions if region.key == "package_type"]
    assert len(regions) == 4
    assert all(region.source == "DRUMS" for region in regions)
    path = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].typeCategory"
    assert all(region.target_paths == (path,) for region in regions)


def test_three_page_inline_localities_have_complete_occurrence_ownership(rows):
    blueprint = _blueprint(rows, SOURCE_0BA)
    assert blueprint.source == rows[SOURCE_0BA]["joinedRawText"]
    assert blueprint.target == rows[SOURCE_0BA]["target"]
    for suffix, role, quote, count in (
        ("receipt", "placeOfReceipt", "KUTNA HORA, CZECH REPUBLIC", 3),
        ("loading", "portOfLoading", "HAMBURG, GERMANY", 2),
        ("discharge", "portOfDischarge", "PORT SAID WEST, EGYPT", 2),
    ):
        key = "locality_pair_" + suffix
        prefix = "documentPatch.route." + role
        regions = [region for region in blueprint.regions if region.key == key]
        assert len(regions) == count
        assert all(region.source == quote for region in regions)
        assert all(
            set(region.target_paths) == {prefix + ".name", prefix + ".country"}
            for region in regions
        )
        assert blueprint.ownership_data["surfaces"][key] == {
            "expression": "{" + prefix + ".name}, {" + prefix + ".country}"
        }
        assert not any(
            region.key in {"anchor:" + prefix + ".name", "anchor:" + prefix + ".country"}
            for region in blueprint.regions
        )


def test_split_email_source_routes_own_complete_comma_frames(rows):
    blueprint = _blueprint(rows, SOURCE_37)
    assert blueprint.source == rows[SOURCE_37]["joinedRawText"]
    assert blueprint.target == rows[SOURCE_37]["target"]
    for suffix, role, text in (
        ("delivery", "placeOfDelivery", "SOKHNA,EGYPT"),
        ("loading", "portOfLoading", "SHANGHAI,CHINA"),
        ("discharge", "portOfDischarge", "SOKHNA,EGYPT"),
    ):
        regions = [r for r in blueprint.regions if r.key == "locality_pair_" + suffix]
        assert len(regions) == 1
        assert regions[0].source == text
        prefix = "documentPatch.route." + role
        assert set(regions[0].target_paths) == {prefix + ".name", prefix + ".country"}
        assert not any(
            r.key in {"anchor:" + prefix + ".name", "anchor:" + prefix + ".country"}
            for r in blueprint.regions
        )


def test_source_only_stone_remark_does_not_restrict_independent_goods(rows):
    from document_ocr.hashing import sha256_file
    from document_ocr.synthesis.independent_wording import (
        load_certificates,
        phrase_surfaces,
        prepare_blueprint,
    )

    path = CATALOG / "independent-flat-batches02-04-20261021/wording_certificates.json"
    certificate = load_certificates(path, expected_sha256=sha256_file(path))[SOURCE_612]
    original = _blueprint(rows, SOURCE_612)
    blueprint = prepare_blueprint(original, certificate)
    surfaces = phrase_surfaces(blueprint)
    expected = "REMARKS: VESSEL AND CARRIER ARE NOT RESPONSIBLE\nFOR LOSS OR DAMAGE TO GOODS"
    assert surfaces == {"reviewed:generic_optional_damage_disclaimer": expected}
    text, target, proof = render_sampling_blueprint(
        blueprint, original.target, {}, surface_values=surfaces
    )
    assert expected in text
    assert "DUE TO WEAKNESS OF STONES" not in text
    assert "BREAKAGES OF STONE BLOCKS" not in text
    assert target == original.target
    assert blueprint.description_blocks == original.description_blocks
    assert proof["unchangedBytesPreserved"]


def test_yang_ming_routes_own_both_repeated_comma_frames(rows):
    blueprint = _blueprint(rows, SOURCE_069)
    assert blueprint.source == rows[SOURCE_069]["joinedRawText"]
    assert blueprint.target == rows[SOURCE_069]["target"]
    for suffix, role, quote in (
        ("receipt", "placeOfReceipt", "NINGBO, CHINA"),
        ("loading", "portOfLoading", "NINGBO, CHINA"),
        ("discharge", "portOfDischarge", "DAMIETTA, EGYPT"),
        ("delivery", "placeOfDelivery", "DAMIETTA, EGYPT"),
    ):
        region = next(r for r in blueprint.regions if r.key == "locality_pair_" + suffix)
        prefix = "documentPatch.route." + role
        assert region.source == quote
        assert set(region.target_paths) == {prefix + ".name", prefix + ".country"}
        assert blueprint.ownership_data["surfaces"][region.key] == {
            "expression": "{" + prefix + ".name}, {" + prefix + ".country}"
        }
    assert not {f"f{n}" for n in range(14, 22)} & {r.key for r in blueprint.regions}
