"""Private source loads preserve incomplete OCR quantity supervision exactly."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_curated_physical import fixture
from test_curated_scenarios import catalog, source_row

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.curated_campaign import Campaign
from document_ocr.synthesis.curated_physical import prepare_physical_render
from document_ocr.synthesis.curated_scenarios import (
    SourceCapabilities,
    SourceLoadAccounting,
    _pin,
)


def certificate(row, counts=None):
    containers = row["target"]["documentPatch"]["containerInformation"]
    counts = counts or (40, 60)
    return SourceLoadAccounting(
        source_ocr_sha256=sha256_bytes(row["joinedRawText"].encode()),
        source_target_sha256=sha256_bytes(canonical_json_bytes(row["target"])),
        source_pdf={"path": "reviewed.pdf", "sha256": "a" * 64},
        container_package_quantities={
            container["equipmentIdentifier"]: count
            for container, count in zip(containers, counts, strict=True)
        },
        equipment_pair="FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE",
        reviewed_pages=(1,),
        rationale="PDF-review-confirmed complete physical vector; omitted OCR counts stay absent.",
    )


def partial_row(*, omit_placement=False, omit_equipment=False):
    row = source_row()
    row["sourcePdfSha256"] = "a" * 64
    patch = row["target"]["documentPatch"]
    if omit_placement:
        patch["goodsItemDetails"][0]["splitGoodsPlacement"].pop()
    else:
        del patch["goodsItemDetails"][0]["splitGoodsPlacement"][1]["packageQuantity"]
    if omit_equipment:
        for container in patch["containerInformation"]:
            del container["sizeCategory"], container["typeCategory"]
    return row


@pytest.mark.parametrize("omit_placement", [False, True])
@pytest.mark.parametrize("omit_equipment", [False, True])
def test_partial_printed_counts_use_exact_lattice_without_adding_missing_labels(
    omit_placement, omit_equipment
):
    row = partial_row(omit_placement=omit_placement, omit_equipment=omit_equipment)
    snapshot = deepcopy(row)
    pin = certificate(row)
    cap = SourceCapabilities(
        family="ambient", physical_profile="source_bundle", source_load_accounting=pin
    )
    support = catalog([row])
    quantities = set()
    for variant in range(1, 51):
        result = support.sample(row, seed=202610145, variant=variant, capabilities=cap)
        goods = result.cargo["goodsItemDetails"][0]
        quantity = goods["numberAndTypeOfPackages"][0]["packageQuantity"]
        quantities.add(quantity)
        assert quantity % 5 == 0
        assert goods["splitGoodsPlacement"][0]["packageQuantity"] * 100 == quantity * 40
        if not omit_placement:
            assert "packageQuantity" not in goods["splitGoodsPlacement"][1]
        assert len(goods["splitGoodsPlacement"]) == (1 if omit_placement else 2)
        if omit_equipment:
            assert all(
                set(container) == {"equipmentIdentifier"}
                for container in result.cargo["containerInformation"]
            )
        receipt = result.provenance["sourceLoadAccounting"]
        assert receipt["unlabelledQuantity"] * 100 == quantity * 60
        assert receipt["unlabelledEquipmentIndices"] == [1]
        rows = result.provenance["physicalRows"]
        assert [(r["shareNumerator"], r["shareDenominator"]) for r in rows] == [
            (40, 100),
            (60, 100),
        ]
        assert rows[1]["quantity"] is None
        pin.validate_physical_rows(row["target"], {"documentPatch": result.cargo}, rows, receipt)
    assert len(quantities) > 5
    assert row == snapshot


def test_partial_quantity_without_reviewed_certificate_is_still_rejected():
    row = partial_row()
    with pytest.raises(ValueError, match="not a complete exact sum"):
        catalog([row]).sample(
            row, seed=1, variant=1, capabilities=SourceCapabilities(family="ambient")
        )


def test_capability_cannot_use_private_partial_vector_for_a_transferable_donor():
    row = partial_row()
    with pytest.raises(ValidationError, match="source-bundle"):
        SourceCapabilities(family="ambient", source_load_accounting=certificate(row))


@pytest.mark.parametrize(
    "change,reason",
    [
        ("ocr", "source OCR or target"),
        ("target_pin", "source OCR or target"),
        ("pdf", "PDF pin"),
        ("container", "exact source inventory"),
        ("cardinality", "exact source inventory"),
        ("count", "declared source total"),
        ("known_quantity", "printed allocation quantity"),
        ("complete", "complete public allocations"),
        ("none_known", "independently owned"),
        ("pair", "printed source category"),
    ],
)
def test_source_certificate_mutations_fail_before_sampling(change, reason):
    row = partial_row()
    pin = certificate(row)
    changed = pin.model_dump(mode="json")
    if change == "ocr":
        changed["source_ocr_sha256"] = "0" * 64
    elif change == "target_pin":
        changed["source_target_sha256"] = "0" * 64
    elif change == "pdf":
        changed["source_pdf"]["sha256"] = "0" * 64
    elif change == "container":
        changed["container_package_quantities"]["WRONG_ID"] = changed[
            "container_package_quantities"
        ].pop("MSCU7654321")
    elif change == "count":
        changed["container_package_quantities"]["MSCU7654321"] = 59
    elif change == "cardinality":
        row["target"]["documentPatch"]["containerInformation"].pop()
        row["target"]["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"].pop()
        changed["source_target_sha256"] = sha256_bytes(canonical_json_bytes(row["target"]))
    elif change == "known_quantity":
        changed["container_package_quantities"] = {"MSCU1234567": 41, "MSCU7654321": 59}
    elif change in {"complete", "none_known"}:
        placements = row["target"]["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"]
        if change == "complete":
            placements[1]["packageQuantity"] = 60
        else:
            del placements[0]["packageQuantity"]
        changed["source_target_sha256"] = sha256_bytes(canonical_json_bytes(row["target"]))
    else:
        changed["equipment_pair"] = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
    cap = SourceCapabilities(
        family="ambient", physical_profile="source_bundle", source_load_accounting=changed
    )
    with pytest.raises(ValueError, match=reason):
        catalog([row]).sample(row, seed=1, variant=1, capabilities=cap)


def test_pdf_content_hash_is_checked_against_the_live_file(tmp_path):
    pdf = tmp_path / "reviewed.pdf"
    pdf.write_bytes(b"independently reviewed PDF bytes")
    pin = {"path": "reviewed.pdf", "sha256": sha256_bytes(pdf.read_bytes())}
    from document_ocr.synthesis.curated_scenarios import FilePin

    assert _pin(tmp_path, FilePin(**pin)) == pdf
    pdf.write_bytes(b"different PDF")
    with pytest.raises(ValueError, match="registry pin mismatch"):
        _pin(tmp_path, FilePin(**pin))


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_quantity",
        "new_equipment_field",
        "quantity_proportion",
        "off_lattice",
        "wrong_physical_share",
        "remainder",
        "duplicate_equipment_id",
    ],
)
def test_projection_and_physical_receipt_mutations_cannot_leak_unprinted_facts(mutation):
    row = partial_row(omit_equipment=True)
    pin = certificate(row)
    cap = SourceCapabilities(
        family="ambient", physical_profile="source_bundle", source_load_accounting=pin
    )
    result = catalog([row]).sample(row, seed=1, variant=1, capabilities=cap)
    target = {"documentPatch": deepcopy(result.cargo)}
    rows = deepcopy(result.provenance["physicalRows"])
    receipt = deepcopy(result.provenance["sourceLoadAccounting"])
    goods = target["documentPatch"]["goodsItemDetails"][0]
    if mutation == "unknown_quantity":
        goods["splitGoodsPlacement"][1]["packageQuantity"] = 1
    elif mutation == "new_equipment_field":
        target["documentPatch"]["containerInformation"][1]["sizeCategory"] = "FORTY_FOOT_HIGH_CUBE"
    elif mutation == "quantity_proportion":
        goods["splitGoodsPlacement"][0]["packageQuantity"] += 1
    elif mutation == "off_lattice":
        goods["numberAndTypeOfPackages"][0]["packageQuantity"] += 1
    elif mutation == "wrong_physical_share":
        rows[1]["shareNumerator"] = 61
    elif mutation == "duplicate_equipment_id":
        target["documentPatch"]["containerInformation"][1]["equipmentIdentifier"] = target[
            "documentPatch"
        ]["containerInformation"][0]["equipmentIdentifier"]
    else:
        receipt["unlabelledQuantity"] += 1
    with pytest.raises(ValueError):
        pin.validate_physical_rows(row["target"], target, rows, receipt)


def test_private_capacity_cannot_approve_a_changed_public_equipment_pair():
    row = partial_row()
    pin = certificate(row)
    target = deepcopy(row["target"])
    target["documentPatch"]["containerInformation"][0]["sizeCategory"] = (
        "TWENTY_FOOT_STANDARD_HEIGHT"
    )
    with pytest.raises(ValueError, match="reviewed equipment pair"):
        pin.validate_projection(row["target"], target)


def test_stage_boundary_detects_a_changed_reviewed_pdf(tmp_path):
    pdf = tmp_path / "reviewed.pdf"
    pdf.write_bytes(b"pinned reviewed PDF")
    campaign = object.__new__(Campaign)
    campaign.source_ids = []
    campaign.equipment_inventories = {}
    campaign._blueprint_file_pins = {pdf: sha256_bytes(pdf.read_bytes())}
    campaign.validate_source_readiness()
    pdf.write_bytes(b"new PDF bytes")
    with pytest.raises(ValueError, match="source files changed"):
        campaign.validate_source_readiness()


def test_physical_render_replays_partial_context_and_keeps_quantity_presence():
    row, blueprint, scenario, support = fixture()
    old = row["target"]["documentPatch"]
    old["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["packageQuantity"] = 3
    old["goodsItemDetails"][0]["splitGoodsPlacement"] = [
        {
            "equipmentIdentifier": old["containerInformation"][0]["equipmentIdentifier"],
            "packageQuantity": 1,
        },
        {"equipmentIdentifier": old["containerInformation"][1]["equipmentIdentifier"]},
    ]
    blueprint = replace(blueprint, target=deepcopy(row["target"]))
    pin_data = certificate(row, (1, 2)).model_dump(mode="json")
    pin_data["equipment_pair"] = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
    pin = SourceLoadAccounting.model_validate(pin_data)
    cap = SourceCapabilities(
        family="ambient", physical_profile="source_bundle", source_load_accounting=pin
    )
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "packageQuantity"
    ] = 6
    target["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"][0]["packageQuantity"] = 2
    target["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["value"] = 60
    rows = [
        {
            **physical,
            "equipmentIdentifier": container["equipmentIdentifier"],
            "quantity": 2 if index == 0 else None,
            "shareDenominator": 3,
        }
        for index, (physical, container) in enumerate(
            zip(scenario.provenance["physicalRows"], old["containerInformation"], strict=True)
        )
    ]
    scenario.provenance.update(
        capabilities=cap.model_dump(mode="json"),
        physicalRows=rows,
        sourceLoadAccounting={
            "certificate": pin.model_dump(mode="json"),
            "sourceQuantity": 3,
            "sampledQuantity": 6,
            "unlabelledQuantity": 4,
            "unlabelledEquipmentIndices": [1],
        },
    )
    plan = prepare_physical_render(blueprint, scenario, target, support=support)
    assert plan.variable_values == {"total": "60", "first": "20", "second": "40"}
    assert (
        "packageQuantity"
        not in plan.target["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"][1]
    )
    changed = deepcopy(scenario)
    changed.provenance["physicalRows"][1]["shareNumerator"] = 3
    with pytest.raises(ValueError, match="reviewed source share"):
        prepare_physical_render(blueprint, changed, target, support=support)


def test_private_equipment_alias_path_preserves_source_only_types_without_category_labels(
    monkeypatch,
):
    row = partial_row(omit_equipment=True)
    pin = certificate(row)
    cap = SourceCapabilities(
        family="ambient", physical_profile="source_bundle", source_load_accounting=pin
    )
    scenario = catalog([row]).sample(row, seed=1, variant=1, capabilities=cap)
    from document_ocr.synthesis.curated_physical import PhysicalRenderPlan

    plan = PhysicalRenderPlan(
        target={"documentPatch": scenario.cargo}, variable_values={}, surface_values={}, receipt={}
    )
    campaign = object.__new__(Campaign)
    campaign.equipment_inventories = {}
    campaign.equipment_compatibility = {}
    campaign.physical_support = None
    campaign.config = {"seed": 1, "aliases": {}}
    campaign.equipment_alias_registry = object()
    campaign.alias_policy = SimpleNamespace(package_mode="source")
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.prepare_physical_render",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.compile_package_count_context", lambda *args: []
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.project_package_aliases_with_context",
        lambda blueprint, alias_plan, context, **kwargs: alias_plan.surface_values,
    )
    output = campaign.physical_render(SimpleNamespace(), scenario, plan.target)
    assert output.receipt["equipmentAlias"] == {
        "mode": "reviewed_private_source_pair_preserved",
        "pair": pin.equipment_pair,
        "publicEquipmentFieldsAdded": False,
    }
    assert output.target == plan.target
    assert output.surface_values == {}
