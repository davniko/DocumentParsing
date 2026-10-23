"""Independent campaign admission guards without provider calls."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import Campaign
from document_ocr.synthesis.curated_equipment import PrintedEquipmentMix
from document_ocr.synthesis.curated_scenarios import CandidateIncompatibility, FilePin
from document_ocr.synthesis.curated_templates import RenderRegion
from document_ocr.synthesis.curated_wording import (
    WordingField,
    WordingRequest,
    wording_output_type,
)
from document_ocr.synthesis.independent_campaign import (
    IndependentCampaign,
    _complete_optional_package_suffix,
    _equipment_authority,
    _fixed_equipment_slots,
    _validate_row_capacity,
    source_equipment_readiness,
)


@dataclass(frozen=True)
class BlueprintFixture:
    source: str
    target: dict
    regions: tuple[RenderRegion, ...] = ()
    historical_bindings: dict | None = None
    ownership_data: dict | None = None


def test_hardware_owner_rejects_unavailable_regime_envelope() -> None:
    for source_type, regime in (
        ("PRESSURIZED_TANK", "chilled"),
        ("GENERAL_PURPOSE", "chilled"),
    ):
        blueprint = BlueprintFixture(
            source="CONTAINER",
            target={"documentPatch": {"containerInformation": [{"typeCategory": source_type}]}},
        )
        with pytest.raises(ValueError, match="source equipment type lacks"):
            _equipment_authority(blueprint, None, (), regime)


def test_hardware_tare_domain_is_constrained_to_requested_regime(monkeypatch) -> None:
    blueprint = BlueprintFixture(
        source="CONTAINER",
        target={"documentPatch": {"containerInformation": [{"typeCategory": "GENERAL_PURPOSE"}]}},
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.printed_tare_equipment_domain",
        lambda *args: {
            "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE",
            "FORTY_FOOT_HIGH_CUBE|REFRIGERATED",
        },
    )
    domain, _ = _equipment_authority(blueprint, None, (), "ambient")
    assert domain == [
        {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}
    ]


@pytest.mark.parametrize("individuals,passes", [([0, 1, 2, 3], True), ([0, 2], False)])
def test_untyped_tares_require_complete_explicit_row_coverage(monkeypatch, individuals, passes):
    blueprint = BlueprintFixture(source="TARE", target={"documentPatch": {
        "containerInformation": [{"equipmentIdentifier": str(i)} for i in range(4)],
    }})
    before = copy.deepcopy(blueprint.target)
    receipts = [{"owner": i, "pair": None, "kg": "3700"} for i in individuals]
    receipts.append({"owners": [0, 1, 2, 3], "kg": "14800",
                     "ownershipBasis": "explicit_aggregate_or_declared_sum"})
    monkeypatch.setattr("document_ocr.synthesis.independent_campaign._active_tare_bindings",
                        lambda *args: {"reviewed-tare": {}})
    monkeypatch.setattr("document_ocr.synthesis.independent_campaign._tare_surfaces",
                        lambda *args: ({}, receipts))
    if passes:
        assert _equipment_authority(blueprint, None, (), "ambient") == (None, receipts)
        assert blueprint.target == before  # No hardware category inferred from tare.
    else:
        with pytest.raises(ValueError, match="untyped tare lacks"):
            _equipment_authority(blueprint, None, (), "ambient")


def test_open_top_source_authority_preserves_printed_size_and_type(monkeypatch):
    row = {"sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT", "typeCategory": "OPEN_TOP"}
    blueprint = BlueprintFixture(source="1X40'OT", target={"documentPatch": {
        "containerInformation": [row],
    }})
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.printed_tare_equipment_domain",
        lambda *args: None,
    )
    assert _equipment_authority(blueprint, None, (), "ambient") == ([row], [])
    with pytest.raises(ValueError, match="no eligible equipment pair"):
        _equipment_authority(blueprint, None,
                             ("FORTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE",), "ambient")
    blueprint.target["documentPatch"]["containerInformation"][0].pop("sizeCategory")
    with pytest.raises(ValueError, match="explicit equipment size ownership"):
        _equipment_authority(blueprint, None, (), "ambient")


def test_ambient_nor_authority_retains_reefer_without_inventing_cold_chain(monkeypatch):
    row = {"sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT", "typeCategory": "REFRIGERATED"}
    blueprint = BlueprintFixture(source="40' / NOR", target={"documentPatch": {
        "containerInformation": [row],
    }})
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.printed_tare_equipment_domain",
        lambda *args: None,
    )
    assert _equipment_authority(blueprint, None, (), "ambient") == ([row], [])
    row["temperatureSetpoint"] = {"value": -18, "unit": "celsius"}
    with pytest.raises(ValueError, match="thermal setpoint"):
        _equipment_authority(blueprint, None, (), "ambient")
    row.pop("temperatureSetpoint")
    row.pop("sizeCategory")
    with pytest.raises(ValueError, match="explicit equipment size"):
        _equipment_authority(blueprint, None, (), "ambient")


def test_platform_source_authority_preserves_actual_hardware(monkeypatch):
    row = {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "PLATFORM"}
    blueprint = BlueprintFixture(source="40 FLAT 9'6", target={"documentPatch": {
        "containerInformation": [row],
    }})
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.printed_tare_equipment_domain",
        lambda *args: None,
    )
    assert _equipment_authority(blueprint, None, (), "ambient") == ([row], [])
    with pytest.raises(ValueError, match="no eligible equipment pair"):
        _equipment_authority(blueprint, None,
                             ("FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE",), "ambient")
    row.pop("sizeCategory")
    with pytest.raises(ValueError, match="explicit equipment size"):
        _equipment_authority(blueprint, None, (), "ambient")


@pytest.mark.parametrize("gross,cube,payload,family,passes", [
    ("100", "24", "28000", "forty_open_top", True),
    ("100", "24", "28000", "twenty_open_top", True),
    ("100", "24.01", "28000", "forty_open_top", False),
    ("101", "1", "100", "forty_open_top", False),
    ("100", "1", "0", "forty_open_top", False),
    ("100", "1", "28000", "unclassified_open_top", False),
    ("100", "1", "28000", "forty_standard", False),
    ("100", "24", "28000", "out_of_gauge", True),
    ("100", "24.01", "28000", "out_of_gauge", False),
    ("101", "1", "100", "out_of_gauge", False),
])
def test_open_top_capacity_uses_explicit_planning_budget_not_invented_enclosed_cube(
    gross, cube, payload, family, passes
):
    draw = {"package": {"packageQuantity": 1}, "physical": {
        "unitNetKg": "1", "unitGrossKg": gross, "unitVolumeM3": cube,
    }, "allocations": [{"packageQuantity": 1, "netKg": "1", "grossKg": gross,
                        "volumeM3": cube}]}
    rows = [{"quantity": 1, "capacity": {
        "payloadKg": payload, "volumeM3": None, "family": family,
    }}]
    if passes:
        _validate_row_capacity(draw, rows)
    else:
        with pytest.raises((ValueError, CandidateIncompatibility)):
            _validate_row_capacity(draw, rows)


def test_certified_unassigned_equipment_mix_becomes_exact_slots(monkeypatch) -> None:
    source = "1X20 DR\n3X40 HC"
    mix = PrintedEquipmentMix.model_validate(
        {
            "source_sha256": sha256_bytes(source.encode()),
            "source_text_literals": [source],
            "members": [
                {
                    "count": 1,
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "count": 3,
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ],
        }
    )
    blueprint = BlueprintFixture(
        source=source,
        target={"documentPatch": {"containerInformation": [{}, {}, {}, {}]}},
        historical_bindings={},
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.printed_tare_equipment_domain",
        lambda *args: None,
    )
    slots = _fixed_equipment_slots(blueprint, SimpleNamespace(equipment_mix=mix))
    assert [slot["sizeCategory"] for slot in slots] == [
        "TWENTY_FOOT_STANDARD_HEIGHT",
        "FORTY_FOOT_HIGH_CUBE",
        "FORTY_FOOT_HIGH_CUBE",
        "FORTY_FOOT_HIGH_CUBE",
    ]
    with pytest.raises(ValueError, match="regime-compatible domain"):
        source_equipment_readiness(
            blueprint,
            None,
            SimpleNamespace(equipment_mix=mix, allowed_equipment_pairs=()),
            "chilled",
        )
    with pytest.raises(ValueError, match="count differs"):
        _fixed_equipment_slots(
            BlueprintFixture(
                source=source,
                target={"documentPatch": {"containerInformation": [{}, {}, {}]}},
            ),
            SimpleNamespace(equipment_mix=mix),
        )


def test_optional_plural_expands_only_its_exact_package_type_owner() -> None:
    source = "100 CARTON(S)"
    region = RenderRegion(
        key="package_type",
        occurrence=0,
        start=4,
        end=10,
        source="CARTON",
        target_paths=("documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].typeCategory",),
        kind="historical",
    )
    blueprint = BlueprintFixture(
        source=source,
        target={},
        regions=(region,),
        historical_bindings={
            "package_type": {
                "occurrences": [{"byte_start": 4, "byte_end": 10, "source_text": "CARTON"}]
            }
        },
        ownership_data={},
    )
    prepared = _complete_optional_package_suffix(blueprint)
    assert prepared.regions[0].source == "CARTON(S)"
    assert prepared.regions[0].end == len(source)
    assert (
        prepared.historical_bindings["package_type"]["occurrences"][0]["source_text"] == "CARTON(S)"
    )
    assert blueprint.regions[0].source == "CARTON"


def test_equal_load_reallocation_rechecks_every_hardware_row() -> None:
    draw = {
        "package": {"packageQuantity": 10},
        "physical": {"unitNetKg": "2", "unitGrossKg": "2", "unitVolumeM3": "1"},
        "allocations": [
            {"packageQuantity": 2, "netKg": "4", "grossKg": "4", "volumeM3": "2"},
            {"packageQuantity": 8, "netKg": "16", "grossKg": "16", "volumeM3": "8"},
        ],
    }
    physical_rows = [
        {"quantity": 2, "capacity": {"payloadKg": "5", "volumeM3": "5"}},
        {"quantity": 8, "capacity": {"payloadKg": "100", "volumeM3": "100"}},
    ]
    _validate_row_capacity(draw, physical_rows)
    IndependentCampaign._equal_container_loads(draw)
    for physical in physical_rows:
        physical["quantity"] = 5
    with pytest.raises(CandidateIncompatibility, match="realized row exceeds"):
        _validate_row_capacity(draw, physical_rows)


def test_independent_readiness_recompiles_inventory_without_observed_fit(monkeypatch) -> None:
    sid = "doc_" + "4" * 64
    rows = [
        {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"},
        {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"},
    ]
    blueprint = BlueprintFixture(
        source="source", target={"documentPatch": {"containerInformation": rows}}
    )
    inventory = SimpleNamespace(validate_rows=lambda actual: actual == rows)
    campaign = IndependentCampaign.__new__(IndependentCampaign)
    campaign.source_ids = [sid]
    campaign.variant_counts = {sid: 1}
    campaign.requests = {f"{sid}:1": {"regime": "ambient"}}
    campaign._blueprint_file_pins = {}
    campaign.equipment_inventories = {sid: inventory}
    campaign.physical_support = object()
    campaign.catalog = SimpleNamespace(
        config=SimpleNamespace(equipment_inventory=SimpleNamespace(source_ids=(sid,))),
        validate_equipment_inventory_support=lambda *_: pytest.fail("observed fit was used"),
    )
    campaign.blueprint = lambda _: blueprint
    campaign._planning_capability = lambda *_: SimpleNamespace(family="ambient")
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.validate_private_person_coverage",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.validate_identifier_privacy", lambda *_: None
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.compile_equipment_inventory",
        lambda bp, *, allow_uniform_aggregate: inventory
        if bp is blueprint and not allow_uniform_aggregate
        else pytest.fail("inventory was not compiled from the exact source"),
    )
    checked = []
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.source_equipment_readiness",
        lambda *args: checked.append(args),
    )
    campaign.validate_source_readiness()
    assert checked == [
        (blueprint, campaign.physical_support, campaign._planning_capability(), "ambient")
    ]

    campaign.equipment_inventories[sid] = object()
    with pytest.raises(ValueError, match="inventory differs from source owners"):
        campaign.validate_source_readiness()


def test_blueprint_keeps_donor_checks_only_in_observed_mode(monkeypatch, tmp_path) -> None:
    sid = "doc_" + "5" * 64
    row = {"joinedRawText": "SOURCE", "target": {"documentPatch": {"containerInformation": []}}}
    contract = tmp_path / "contracts" / sid / "contract.json"
    history = tmp_path / "historical" / sid / "template.json"
    ownership = tmp_path / "ownership.json"
    pdf = tmp_path / "source.pdf"
    contract.parent.mkdir(parents=True)
    history.parent.mkdir(parents=True)
    contract.write_text(
        json.dumps(
            {
                "sourceSha256": digest(row["joinedRawText"].encode()),
                "targetSha256": digest(row["target"]),
                "contract": {},
            }
        )
    )
    history.write_text("{}")
    ownership.write_text("{}")
    pdf.write_bytes(b"source")
    blueprint = SimpleNamespace(
        source=row["joinedRawText"],
        target=row["target"],
        description_blocks=(),
        regions=(),
        ownership_data={},
    )
    checks = []
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.SourceContract.model_validate",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.load_owned_blueprint",
        lambda *_: blueprint,
    )
    for name in ("bind_postal_country_alias_owners", "augment_auxiliary_blueprint"):
        monkeypatch.setattr(
            f"document_ocr.synthesis.curated_campaign.{name}", lambda bp, *_: bp
        )
    for name in (
        "validate_phone_source_completeness",
        "validate_lexical_owners",
        "certified_identity_kinds",
        "validate_description_regions",
    ):
        monkeypatch.setattr(f"document_ocr.synthesis.curated_campaign.{name}", lambda *_: None)
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.related_actor_contexts", lambda *_: {}
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.validate_source_physical_requirements",
        lambda *_: checks.append("physical"),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.validate_contained_quantity_sampling",
        lambda *_, **__: checks.append("contained"),
    )
    load = SimpleNamespace(
        validate_source=lambda *_: checks.append("source_load"),
        source_pdf=FilePin(path=pdf.name, sha256=sha256_file(pdf)),
    )
    capability = SimpleNamespace(
        source_load_accounting=load,
        party_locality_groups=(),
        equipment_mix=None,
        quantity_multiple=2,
        fixed_package_quantity=None,
    )

    def campaign(cls):
        item = cls.__new__(cls)
        item.root = tmp_path
        item.config = {
            "source_contracts": "contracts",
            "historical_catalog": "historical",
            "ownership": ownership.name,
        }
        item.rows = {sid: row}
        item.capabilities = {sid: capability}
        item._blueprints = {}
        item._blueprint_file_pins = {}
        item.country_registry = object()
        item.auxiliary = object()
        item.catalog = SimpleNamespace(config=SimpleNamespace(equipment_inventory=None))
        return item

    Campaign.blueprint(campaign(IndependentCampaign), sid)
    assert checks == []
    Campaign.blueprint(campaign(Campaign), sid)
    assert checks == ["source_load", "physical", "contained"]


class _Scenario:
    def __init__(self, provenance):
        self.provenance = provenance
        self.party_localities = {}

    def model_copy(self, *, update):
        result = copy.copy(self)
        result.provenance = update["provenance"]
        return result


def test_plan_uses_frozen_request_without_observed_sampler(monkeypatch) -> None:
    """The donor sampler is a tripwire; exact mix slots reach the new sampler."""
    sid = "doc_" + "1" * 64
    sample = IndependentCampaign.__new__(IndependentCampaign)
    sample._planned = {}
    sample._plans_by_sample_id = {}
    sample._plan_lock_guard = threading.Lock()
    sample._plan_locks = {}
    sample._plan_overrides = {}
    sample.variant_counts = {sid: 1}
    sample.config = {"seed": 7}
    sample.requests = {
        f"{sid}:1": {
            "sourceDocumentId": sid,
            "variant": 1,
            "regime": "ambient",
            "equipmentMode": "uniform",
            "selectedCategory": "PACKAGE_CARTON",
            "countedUnitContext": None,
        }
    }
    sample.private_plans = {f"{sid}:1": {"privatePackingRelation": "NONE"}}
    sample.rows = {sid: {}}
    sample.physical_support = object()
    sample.phone_policy = object()
    sample.vessels = ("NEW VESSEL", "OLD VESSEL")
    blueprint = SimpleNamespace(
        document_id=sid,
        source="source",
        target={"documentPatch": {"containerInformation": [{}]}},
    )
    sample.blueprint = lambda _: blueprint
    sample._planning_capability = lambda *_: SimpleNamespace(
        family="ambient", allowed_equipment_pairs=(), quantity_multiple=1, equipment_mix=None
    )

    def forbidden_observed_sample(*args, **kwargs):
        raise AssertionError("observed cargo sampler was called")

    sample.catalog = SimpleNamespace(
        sample=forbidden_observed_sample,
        config=SimpleNamespace(maximum_candidates=2),
        _geography=lambda *args: ({}, {}, {}, {}, {}),
    )
    draws = []

    def draw(**kwargs):
        draws.append(kwargs)
        return {"equipment": [], "allocations": [], "provenance": {}}

    sample.sampler = SimpleNamespace(draw=draw)
    sample.numeric = SimpleNamespace(
        prepare_draw=lambda _, draw, *__: draw,
        build_ledger=lambda *_: {"surfaces": {}},
        validate_composed_capacity=lambda *_: None,
    )
    sample.wording_certificates = {}
    scenario = _Scenario({"physicalRows": []})
    sample._scenario_from_draw = lambda *args: (
        blueprint,
        scenario,
        "syn_full_v7_test",
        {"documentPatch": {"transport": {"vesselName": "OLD VESSEL"}}},
    )
    sample.physical_render = lambda *args: None
    sample.source_phone_country_codes = lambda _: {}
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.prepare_phone_rendering",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign._equipment_authority",
        lambda *args: (None, []),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign._fixed_equipment_slots",
        lambda *args: [
            {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}
        ],
    )
    result = sample.plan(sid, 1)
    assert draws[0]["request"] is sample.requests[f"{sid}:1"]
    assert draws[0]["counted_unit_context"] is None
    assert draws[0]["equipment_slots"] == [
        {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}
    ]
    assert result[3]["documentPatch"]["transport"]["vesselName"] == "NEW VESSEL"
    assert sample.plan(sid, 1) is result
    assert len(draws) == 1


def test_generated_wording_holds_reviewed_owner_error_before_receipt(monkeypatch) -> None:
    sid = "doc_" + "2" * 64
    sample_id = "syn_full_v7_pre_save"
    campaign = IndependentCampaign.__new__(IndependentCampaign)
    request = WordingRequest(sample_id=sample_id, context="", fields=())
    blueprint = SimpleNamespace(document_id=sid)
    scenario = _Scenario({"freshPrivatePacking": {}})
    campaign._plans_by_sample_id = {sample_id: (sid, 1)}
    campaign.plan = lambda *_: (blueprint, scenario, sample_id, {})
    campaign.wording_request = lambda *_: request
    campaign.numeric = SimpleNamespace(project_wording=lambda values, _: (dict(values), {}))
    campaign.geography = SimpleNamespace(validate_values=lambda *_: ["postal owner conflict"])
    campaign.locality_owner_reviews = SimpleNamespace(for_document=lambda _: None)
    campaign.distinct_actor_reviews = SimpleNamespace(for_document=lambda _: None)
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.validate_wording",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.resolve_reviewed_locality_assemblies",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.resolve_reviewed_distinct_actors",
        lambda *args: (),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.validate_locality_owners",
        lambda *args: [],
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.validate_distinct_actors",
        lambda *args: [],
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.postal_wording_errors",
        lambda *_: {},
    )
    values = {sample_id: {"goods": "fresh goods"}}
    with pytest.raises(ValueError, match="postal owner conflict"):
        campaign.validate_generated_wording([request], values)
    assert values == {sample_id: {"goods": "fresh goods"}}


def test_generated_wording_holds_ordinary_postal_error_before_receipt(monkeypatch) -> None:
    sid = "doc_" + "6" * 64
    sample_id = "syn_full_v7_postal_gate"
    campaign = IndependentCampaign.__new__(IndependentCampaign)
    request = WordingRequest(sample_id=sample_id, context="", fields=())
    blueprint = SimpleNamespace(document_id=sid)
    scenario = _Scenario({"freshPrivatePacking": {}})
    campaign._plans_by_sample_id = {sample_id: (sid, 1)}
    campaign.plan = lambda *_: (blueprint, scenario, sample_id, {})
    campaign.wording_request = lambda *_: request
    campaign._check_independent_wording_values = lambda *args: (dict(args[-1]), [])
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.validate_wording",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.postal_wording_errors",
        lambda *_: {"documentPatch.parties.consignee": ["locality is repeated"]},
    )
    values = {sample_id: {"postal": "DUPLICATE LOCALITY"}}
    with pytest.raises(ValueError, match="postalSourceOwnership"):
        campaign.validate_generated_wording([request], values)
    assert values == {sample_id: {"postal": "DUPLICATE LOCALITY"}}


def test_wording_diagnostics_collect_base_and_postal_failures_by_shipment(monkeypatch) -> None:
    requests = [
        WordingRequest(
            sample_id=f"sample-{index}",
            context="",
            fields=(
                WordingField(
                    "postal",
                    "consignee postal",
                    "OLD ROAD",
                    "Generate a fictional postal component.",
                ),
            ),
        )
        for index in range(3)
    ]
    output = wording_output_type(requests).model_validate(
        {
            "s0": {"postal": "\u0421YRILLIC ROAD"},
            "s1": {"postal": "CLEAN ROAD"},
            "s2": {"postal": "DUPLICATE LOCALITY"},
        }
    )
    campaign = Campaign.__new__(Campaign)
    inspected = []

    def diagnostics(request, _values):
        inspected.append(request.sample_id)
        if request.sample_id == "sample-0":
            return {
                "postalSourceOwnership": {
                    "documentPatch.parties.shipper": ["locality is repeated in the same variant"]
                }
            }
        if request.sample_id == "sample-2":
            return {
                "postalSourceOwnership": {
                    "documentPatch.parties.consignee": ["locality is repeated"]
                }
            }
        return {}

    campaign._generated_wording_diagnostics = diagnostics
    with pytest.raises(ValueError, match="wording validation failed by shipment") as raised:
        campaign._unpack_generated_wording(output, requests)
    failure = str(raised.value)
    assert '"shipmentKey": "s0"' in failure
    assert '"sampleId": "sample-0"' in failure
    assert "Latin script" in failure
    assert "locality is repeated in the same variant" in failure
    assert '"shipmentKey": "s2"' in failure
    assert '"sampleId": "sample-2"' in failure
    assert "locality is repeated" in failure
    assert "sample-1" not in failure
    # Provider-native field coverage is already exact, so task-specific checks
    # diagnose the same variant and every sibling despite a lexical failure.
    assert inspected == ["sample-0", "sample-1", "sample-2"]


def test_wording_batch_structural_failure_is_not_reclassified(monkeypatch) -> None:
    requests = [
        WordingRequest(
            sample_id=f"sample-{index}",
            context="",
            fields=(WordingField("postal", "consignee postal", "OLD", "Generate postal."),),
        )
        for index in range(2)
    ]
    incomplete = wording_output_type([requests[0]]).model_validate(
        {"s0": {"postal": "CLEAN ROAD"}}
    )
    campaign = Campaign.__new__(Campaign)
    called = []
    campaign._generated_wording_diagnostics = lambda *_: called.append(True) or {}
    with pytest.raises(ValueError, match="shipment schema differs") as raised:
        campaign._unpack_generated_wording(incomplete, requests)
    assert "wording validation failed by shipment" not in str(raised.value)
    assert called == []


def test_batch_repair_receives_all_failures_and_second_failure_publishes_nothing(
    tmp_path,
) -> None:
    requests = [
        WordingRequest(
            sample_id=f"sample-{index}",
            context="",
            fields=(
                WordingField(
                    "postal",
                    "consignee postal",
                    "OLD ROAD",
                    "Generate a fictional postal component.",
                ),
            ),
        )
        for index in range(3)
    ]
    request_by_id = {request.sample_id: request for request in requests}
    campaign = Campaign.__new__(Campaign)
    campaign.output = tmp_path
    campaign.config = {"variants_per_source": 3}
    campaign.plans = lambda _: [
        (SimpleNamespace(), SimpleNamespace(), request.sample_id, {})
        for request in requests
    ]
    campaign.preflight = lambda _: None
    campaign.wording_request = lambda _bp, _scenario, sample_id, _target: request_by_id[
        sample_id
    ]

    def diagnostics(request, _values):
        if request.sample_id == "sample-2":
            return {
                "postalSourceOwnership": {
                    "documentPatch.parties.consignee": ["locality is repeated"]
                }
            }
        return {}

    campaign._generated_wording_diagnostics = diagnostics
    stages = []

    async def call(stage, _sid, output_type, _system, prompt):
        stages.append(stage)
        if stage == "wording-validation-repair":
            assert '"shipmentKey": "s0"' in prompt
            assert '"sampleId": "sample-0"' in prompt
            assert '"shipmentKey": "s2"' in prompt
            assert '"sampleId": "sample-2"' in prompt
            assert "Latin script" in prompt
            assert "locality is repeated" in prompt
        return output_type.model_validate(
            {
                "s0": {
                    "postal": "\u0421YRILLIC ROAD" if stage == "exact-wording" else "FIXED ROAD"
                },
                "s1": {"postal": "CLEAN ROAD"},
                "s2": {"postal": "DUPLICATE LOCALITY"},
            }
        )

    campaign.calls = SimpleNamespace(call=call)
    contacts = []

    async def generate_contacts(_sid):
        contacts.append(True)
        return {}

    campaign.generate_contacts = generate_contacts
    with pytest.raises(ValueError, match=r'"shipmentKey": "s2"') as raised:
        asyncio.run(campaign.generate("source"))
    assert '"shipmentKey": "s0"' not in str(raised.value)
    assert stages == ["exact-wording", "wording-validation-repair"]
    assert contacts == []
    assert list(tmp_path.rglob("*.json")) == []


def test_failed_postal_correction_preserves_original_receipt(monkeypatch, tmp_path) -> None:
    sid = "doc_" + "7" * 64
    sample_id = "syn_full_v7_postal_correction"
    original_values = {"postal": "BAD OLD LOCALITY"}
    receipt_path = tmp_path / "wording" / f"{sample_id}.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps({"values": original_values}))
    original_bytes = receipt_path.read_bytes()
    address_path = "documentPatch.parties.consignee.addressLine"
    target = {"documentPatch": {"parties": {"consignee": {"addressLine": "BAD OLD LOCALITY"}}}}
    blueprint = SimpleNamespace(
        contract=SimpleNamespace(
            targets=(SimpleNamespace(path=address_path, expression="{postal}"),),
            variables=(SimpleNamespace(key="postal", kind="postal"),),
        )
    )
    scenario = object()
    campaign = Campaign.__new__(Campaign)
    campaign.output = tmp_path
    campaign.config = {"variants_per_source": 1}
    campaign.plans = lambda *_: [(blueprint, scenario, sample_id, target)]
    campaign.wording_request = lambda *_: WordingRequest(
        sample_id,
        "context",
        (WordingField("postal", "consignee postal", "BAD OLD LOCALITY", "postal"),),
    )
    campaign._current_wording = lambda receipt, _: receipt["values"]

    async def correction(*args):
        return object()

    campaign.calls = SimpleNamespace(call=correction)
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.assemble_lexical_target", lambda *_: target
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.host_product_wording", lambda *_: {}
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.postal_wording_errors",
        lambda *_: {"documentPatch.parties.consignee": ["locality is repeated"]},
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign._fixed_postal_country_components",
        lambda *_: set(),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.postal_assembly_context", lambda *_, **__: ""
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.wording_output_type",
        lambda *_: SimpleNamespace(model_json_schema=lambda: {}),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.native_wording_prompt", lambda *_: "prompt"
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.unpack_wording",
        lambda *_: {sample_id: {"postal": "BAD NEW LOCALITY"}},
    )
    with pytest.raises(ValueError, match="postal correction still requires review"):
        asyncio.run(campaign.correct_postal(sid))
    assert receipt_path.read_bytes() == original_bytes


def test_native_evidence_uses_runner_canonical_provider_not_raw_yaml(monkeypatch, tmp_path) -> None:
    sid = "doc_" + "3" * 64
    pins = {}
    for key in (
        "numeric_certificates",
        "wording_certificates",
        "geography_certificates",
        "requests",
        "private_plans",
        "locality_owner_reviews",
        "distinct_actor_reviews",
    ):
        path = tmp_path / (key + ".json")
        if key == "requests":
            value = {
                f"{sid}:1": {
                    "sourceDocumentId": sid,
                    "variant": 1,
                    "selectedCategory": "PACKAGE_CARTON",
                    "planningStage": "native_package_choice_accepted",
                }
            }
        elif key == "private_plans":
            value = {f"{sid}:1": {"privatePackingRelation": "NONE"}}
        else:
            value = {}
        path.write_text(json.dumps(value))
        pins[key] = {"path": path.name, "sha256": sha256_file(path)}
    raw_provider = {"max_prompt_price": 0.25, "request_timeout_seconds": 300}
    canonical_provider = {"max_prompt_price": "0.25", "request_timeout_seconds": 300.0}
    captured = []
    readiness_holds = []
    numeric = SimpleNamespace(
        has_certificate=lambda *_: True,
        source_readiness_holds=lambda *_: tuple(readiness_holds),
        allowed_private_relations=lambda *_: ("NONE",),
        certificate=lambda *_: {},
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.IndependentSampler.from_config",
        lambda *args: SimpleNamespace(phrases={}, validate_request=lambda *_: None),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.IndependentNumericCertificates.from_path",
        lambda *args, **kwargs: numeric,
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.load_wording_certificates",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.load_independent_geography_certificates",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.IndependentGeographyGate",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.load_locality_reviews",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.load_distinct_actor_reviews",
        lambda *args, **kwargs: object(),
    )

    def base_init(self, root, config):
        self.source_ids = [sid]
        self.variant_counts = {sid: 1}
        self.calls = SimpleNamespace(config={"provider": canonical_provider})

    monkeypatch.setattr("document_ocr.synthesis.independent_campaign.Campaign.__init__", base_init)

    def choice(*args, provider, **kwargs):
        captured.append(provider)

    def private(*args, provider, **kwargs):
        captured.append(provider)
        return {"privatePackingRelation": "NONE", "relatedPackageType": None}

    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.verify_native_choice_evidence", choice
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.verify_native_plan_evidence", private
    )
    IndependentCampaign(
        tmp_path, {"independent": {"registry": {}, **pins}, "provider": raw_provider}
    )
    assert captured == [canonical_provider, canonical_provider]
    assert raw_provider != canonical_provider

    captured.clear()
    readiness_holds.append(
        SimpleNamespace(
            category="source_owner_review",
            reason="missing ownership certificate",
            evidence=FilePin(**pins["numeric_certificates"]),
        )
    )
    with pytest.raises(ValueError, match=r"source_readiness_hold.*source_owner_review"):
        IndependentCampaign(
            tmp_path, {"independent": {"registry": {}, **pins}, "provider": raw_provider}
        )
    assert captured == []


def test_review_private_packing_uses_current_independent_ledger() -> None:
    sid = "doc_" + "8" * 64
    sample_id = "syn_full_v7_private_review"
    ledger = {
        "sourceDocumentId": sid,
        "certificateFileSha256": "a" * 64,
        "certificateSha256": "b" * 64,
        "countedCategory": "PACKAGE_DRUM_STEEL",
        "publicQuantity": 40,
        "relation": "CONTAINS",
        "relatedCategory": "PACKAGE_BAG_PLASTIC",
        "relatedQuantity": 160,
        "surfaces": {"held-numeric:0": ["160 BAGS IN 40 DRUMS"]},
    }
    candidate = {
        "documentId": sample_id,
        "variant": 1,
        "scenario": {"provenance": {"freshPrivatePacking": copy.deepcopy(ledger)}},
        "physical": {"freshPrivatePacking": copy.deepcopy(ledger)},
        "surfaceValues": {"held-numeric:0": ["160 BAGS IN 40 DRUMS"]},
        "target": {
            "documentPatch": {
                "goodsItemDetails": [
                    {
                        "numberAndTypeOfPackages": [
                            {"typeCategory": "PACKAGE_DRUM_STEEL", "packageQuantity": 40}
                        ]
                    }
                ]
            }
        },
    }
    campaign = IndependentCampaign.__new__(IndependentCampaign)
    campaign.numeric = SimpleNamespace(file_sha256="a" * 64)
    campaign.requests = {
        f"{sid}:1": {
            "selectedCategory": "PACKAGE_DRUM_STEEL",
            "countedUnitContext": None,
            "nativeChoiceEvidence": {"packetSha256": "c" * 64},
        }
    }
    campaign.private_plans = {
        f"{sid}:1": {
            "origin": "native_typed_private_plan",
            "privatePackingRelation": "CONTAINS",
            "relatedPackageType": "PACKAGE_BAG_PLASTIC",
            "nativePlanPacketSha256": "d" * 64,
        }
    }
    campaign.plan = lambda *_: (
        None,
        SimpleNamespace(provenance={"freshPrivatePacking": ledger}),
        sample_id,
        None,
    )
    context = campaign.review_private_packing_context(
        SimpleNamespace(document_id=sid), candidate
    )
    assert context["declaredCountedUnit"]["quantity"] == 40
    assert context["privatePacking"] == {
        "relation": "CONTAINS",
        "relatedCategory": "PACKAGE_BAG_PLASTIC",
        "relatedQuantity": 160,
    }
    candidate["scenario"]["provenance"]["freshPrivatePacking"]["relatedQuantity"] = 159
    with pytest.raises(ValueError, match="current certified plan"):
        campaign.review_private_packing_context(SimpleNamespace(document_id=sid), candidate)
