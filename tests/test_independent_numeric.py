from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from document_ocr.synthesis.curated import (
    Occurrence,
    SourceContract,
    TargetBinding,
    Variable,
)
from document_ocr.synthesis.curated_templates import (
    RenderRegion,
    SamplingBlueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.independent_numeric import (
    PRIVATE_PARTITION_KEY,
    IndependentNumericCertificates,
    project_handling_targets,
    project_wording,
)


class FakeContract:
    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {"contract": "sanitized"}


@dataclass(frozen=True)
class FakeBlueprint:
    document_id: str
    source: str
    target: dict[str, Any]
    contract: FakeContract = field(default_factory=FakeContract)
    regions: tuple[Any, ...] = ()
    historical_bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    nested_bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    blocked_bindings: tuple[dict[str, Any], ...] = ()
    ownership_data: dict[str, Any] = field(default_factory=dict)
    description_blocks: None = None


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def target(*, containers: int = 0, allocation_rows: int = 0) -> dict[str, Any]:
    return {
        "documentPatch": {
            "containerInformation": [{} for _ in range(containers)],
            "goodsItemDetails": [
                {"splitGoodsPlacement": [{"packageQuantity": 1} for _ in range(allocation_rows)]}
            ],
        }
    }


def certificate(
    blueprint: FakeBlueprint,
    *,
    regions: list[dict[str, Any]] | None = None,
    wording_owners: list[dict[str, Any]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    hold_reasons: list[str] | None = None,
) -> dict[str, Any]:
    patch = blueprint.target["documentPatch"]
    return {
        "reviewStatus": "reviewed",
        "sourceSha256": digest(blueprint.source.encode()),
        "sourceContractSha256": digest(blueprint.contract.model_dump(mode="json")),
        "sourcePublicTargetSha256": digest(blueprint.target),
        "sourceContainerCount": len(patch["containerInformation"]),
        "sourceAllocationPresence": [
            "packageQuantity" in row
            for goods in patch["goodsItemDetails"]
            for row in goods["splitGoodsPlacement"]
        ],
        "regions": regions or [],
        "wordingOwners": wording_owners or [],
        "constraints": constraints or [],
        "holdReasons": hold_reasons or [],
    }


def write_inventory(
    path: Path, blueprint: FakeBlueprint, value: dict[str, Any]
) -> IndependentNumericCertificates:
    payload = {
        "version": 2,
        "selectedSources": [blueprint.document_id],
        "sources": {blueprint.document_id: value},
    }
    path.write_text(json.dumps(payload))
    return IndependentNumericCertificates.from_path(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def quoted(source: str, quote: str, role: str, **extra: Any) -> dict[str, Any]:
    assert source.count(quote) == 1
    start = len(source[: source.index(quote)].encode())
    return {
        "byte_start": start,
        "byte_end": start + len(quote.encode()),
        "source_text": quote,
        "role": role,
        **extra,
    }


def draw(
    category: str = "PACKAGE_DRUM_STEEL",
    quantities: tuple[int, ...] = (4,),
) -> dict[str, Any]:
    quantity = sum(quantities)
    return {
        "internalCategory": category,
        "package": {"packageQuantity": quantity},
        "physical": {
            "unitNetKg": "180",
            "unitGrossKg": "192",
            "unitVolumeM3": "0.28",
            "grossWeight": {"value": str(192 * quantity), "unit": "kilogram"},
        },
        "allocations": [
            {
                "packageQuantity": count,
                "netKg": str(180 * count),
                "grossKg": str(192 * count),
                "volumeM3": str(count * 0.28),
            }
            for count in quantities
        ],
        "provenance": {},
    }


def renderable_blueprint(source_id: str = "doc-optional-range") -> SamplingBlueprint:
    source = "RANGE\nBL1"
    path = "documentPatch.billOfLadingNumber"
    target_value = target()
    target_value["documentPatch"]["billOfLadingNumber"] = "BL1"
    contract = SourceContract(
        variables=[
            Variable(
                key="bill_number",
                kind="identifier",
                value="BL1",
                meaning="Bill number retained outside the optional numeric clause",
                required_literals=[],
                occurrences=[Occurrence(text="BL1", occurrence=1, presentation="text")],
            )
        ],
        targets=[TargetBinding(path=path, expression="{bill_number}")],
        fixed_context="Optional private numeric clause",
    )
    region = RenderRegion(
        "bill_number",
        0,
        len(b"RANGE\n"),
        len(source.encode()),
        "BL1",
        (path,),
        "identifier",
        "bill_number",
        "text",
        "BL1",
    )
    return SamplingBlueprint(
        source_id,
        source,
        target_value,
        contract,
        (region,),
        {
            "bill_number": {
                "target_paths": [path],
                "occurrences": [
                    {
                        "source_text": "BL1",
                        "byte_start": len(b"RANGE\n"),
                        "byte_end": len(source.encode()),
                    }
                ],
            }
        },
        {},
        (),
    )


def test_inventory_path_pin_and_complete_coverage_fail_closed(tmp_path: Path) -> None:
    blueprint = FakeBlueprint(
        "doc-one",
        "SOURCE",
        target={"documentPatch": {"containerInformation": [], "goodsItemDetails": []}},
    )
    value = certificate(blueprint)
    inventory = tmp_path / "certificates.json"
    authority = write_inventory(inventory, blueprint, value)
    assert authority.has_certificate("doc-one")
    with pytest.raises(ValueError, match="lacks a complete"):
        authority.has_certificate("missing")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        IndependentNumericCertificates.from_path(inventory, expected_sha256="0" * 64)
    payload = json.loads(inventory.read_text())
    payload["selectedSources"].append("missing")
    inventory.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="coverage differs"):
        IndependentNumericCertificates.from_path(inventory)


def test_source_readiness_hold_is_typed_separate_from_numeric_role_support(tmp_path: Path):
    blueprint = FakeBlueprint("doc-one", "SOURCE", target())
    value = certificate(blueprint)
    value["sourceReadinessHolds"] = [
        {
            "category": "privacy",
            "reason": "Shipment-appointed agency has no current generator",
            "evidence": {"path": "reviews/source.json", "sha256": "a" * 64},
        }
    ]
    authority = write_inventory(tmp_path / "inventory.json", blueprint, value)
    assert authority.certificate("doc-one")["holdReasons"] == []
    assert authority.allowed_private_relations("doc-one") == ("NONE",)
    (hold,) = authority.source_readiness_holds("doc-one")
    assert hold.category == "privacy" and hold.evidence.sha256 == "a" * 64
    # Numeric application remains a numeric check; campaign/preparation gates
    # independently refuse whole-document readiness before native calls.
    authority.apply_blueprint(blueprint)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        ["untyped reason"],
        [{"category": "privacy", "reason": "missing evidence"}],
        [{"category": "privacy", "reason": " ", "evidence": {"path": "x", "sha256": "a" * 64}}],
        [{"category": "privacy", "reason": "hold", "evidence": {"path": "x", "sha256": "bad"}}],
        [
            {
                "category": "privacy",
                "reason": "hold",
                "evidence": {"path": "../x", "sha256": "a" * 64},
            }
        ],
        [{"category": "privacy", "reason": "hold", "evidence": {"path": "/x", "sha256": "a" * 64}}],
    ],
)
def test_malformed_readiness_authority_is_fatal(tmp_path: Path, bad):
    blueprint = FakeBlueprint("doc-one", "SOURCE", target())
    value = certificate(blueprint)
    value["sourceReadinessHolds"] = bad
    with pytest.raises(ValueError):
        write_inventory(tmp_path / "inventory.json", blueprint, value)


def test_invalid_executable_certificate_fields_fail_during_load(tmp_path: Path) -> None:
    blueprint = FakeBlueprint("doc-invalid", "COUNT", target())
    malformed = certificate(
        blueprint,
        regions=[quoted("COUNT", "COUNT", "private_ranges")],
    )
    payload = {
        "version": 2,
        "selectedSources": [blueprint.document_id],
        "sources": {blueprint.document_id: malformed},
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid private range"):
        IndependentNumericCertificates.from_path(path)
    malformed["regions"][0]["byte_end"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid region span"):
        IndependentNumericCertificates.from_path(path)


def test_reviewed_optional_related_range_can_render_an_exact_omission(
    tmp_path: Path,
) -> None:
    blueprint = renderable_blueprint()
    region = quoted(
        blueprint.source,
        "RANGE",
        "private_ranges",
        countRole="related",
        parts=1,
        factGroup="shipment",
        digits=3,
        prefixPattern="[A-Z]",
        optionalWithoutRelatedUnit=True,
    )
    authority = write_inventory(
        tmp_path / "optional.json",
        blueprint,  # type: ignore[arg-type]
        certificate(blueprint, regions=[region]),  # type: ignore[arg-type]
    )
    prepared = authority.apply_blueprint(blueprint)
    numeric_region = next(item for item in prepared.regions if item.key == "held-numeric:0")
    assert numeric_region.allow_empty is True

    ledger = authority.build_ledger(
        prepared,
        draw(),
        {"privatePackingRelation": "NONE", "relatedPackageType": None},
        7,
        0,
    )
    assert ledger["surfaces"] == {"held-numeric:0": [""]}
    rendered, rendered_target, _ = render_sampling_blueprint(
        prepared,
        prepared.target,
        {},
        surface_values=ledger["surfaces"],
    )
    assert rendered == "\nBL1"
    assert rendered_target == prepared.target


def test_nonoptional_complete_region_still_rejects_a_blank_surface(tmp_path: Path) -> None:
    blueprint = renderable_blueprint("doc-required-range")
    region = quoted(
        blueprint.source,
        "RANGE",
        "private_ranges",
        countRole="counted",
        parts=1,
        factGroup="shipment",
        digits=3,
        prefixPattern="[A-Z]",
    )
    authority = write_inventory(
        tmp_path / "required.json",
        blueprint,  # type: ignore[arg-type]
        certificate(blueprint, regions=[region]),  # type: ignore[arg-type]
    )
    prepared = authority.apply_blueprint(blueprint)
    numeric_region = next(item for item in prepared.regions if item.key == "held-numeric:0")
    assert numeric_region.allow_empty is False
    with pytest.raises(ValueError, match="explicit surface cannot be blank"):
        render_sampling_blueprint(
            prepared,
            prepared.target,
            {},
            surface_values={"held-numeric:0": [""]},
        )


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"optionalWithoutRelatedUnit": "true"}, "invalid optional-region flag"),
        (
            {"role": "inclusive_range", "optionalWithoutRelatedUnit": True},
            "invalid optional-region role",
        ),
        (
            {"countRole": "counted", "optionalWithoutRelatedUnit": True},
            "invalid optional-region role",
        ),
    ],
)
def test_optional_omission_authority_is_strictly_scoped(
    tmp_path: Path, updates: dict[str, Any], message: str
) -> None:
    blueprint = FakeBlueprint("doc-optional-invalid", "RANGE", target())
    region = quoted(
        blueprint.source,
        "RANGE",
        "private_ranges",
        countRole="related",
        parts=1,
        factGroup="shipment",
        digits=3,
        prefixPattern="[A-Z]",
    )
    region.update(updates)
    with pytest.raises(ValueError, match=message):
        write_inventory(
            tmp_path / "invalid-optional.json",
            blueprint,
            certificate(blueprint, regions=[region]),
        )


def test_repeated_counts_units_relations_and_capacity_are_current(tmp_path: Path) -> None:
    source = "PACKING CAPACITY"
    blueprint = FakeBlueprint("doc-repeat", source, target())
    regions = [
        quoted(source, "PACKING", "packing_clause", repeats=3),
        quoted(source, "CAPACITY", "receptacle_statement"),
    ]
    authority = write_inventory(
        tmp_path / "repeat.json", blueprint, certificate(blueprint, regions=regions)
    )
    prepared = authority.apply_blueprint(blueprint)  # type: ignore[arg-type]
    current = draw()
    relation = {
        "privatePackingRelation": "CONTAINS",
        "relatedPackageType": "PACKAGE_BOX",
    }
    ledger = authority.build_ledger(
        prepared,  # type: ignore[arg-type]
        current,
        relation,
        17,
        0,
        nominal_capacity_litres="200",
        capacity_authority="explicit_test_registry",
    )
    assert ledger["surfaces"]["held-numeric:0"][0].count(" IN ") == 3
    assert "NOMINAL CAPACITY 200 LITRES" in ledger["surfaces"]["held-numeric:1"][0]
    assert ledger["facts"]["nominalCapacity"]["authority"] == "explicit_test_registry"
    assert ledger["invariants"]["sourceRatiosConsumed"] is False
    with pytest.raises(ValueError, match="supplied together"):
        authority.build_ledger(
            prepared,
            current,
            relation,
            17,
            0,
            nominal_capacity_litres="200",  # type: ignore[arg-type]
        )


def test_shipment_fact_groups_share_the_shipment_packing_authority(tmp_path: Path) -> None:
    source = "PACKING CARTON RANGE PALLET RANGE"
    blueprint = FakeBlueprint("doc-shipment-groups", source, target())
    regions = [
        quoted(source, "PACKING", "packing_clause"),
        quoted(
            source,
            "CARTON RANGE",
            "private_ranges",
            countRole="related",
            parts=1,
            factGroup="innerCartonRange",
            digits=4,
            prefixPattern="",
            optionalWithoutRelatedUnit=True,
        ),
        quoted(
            source,
            "PALLET RANGE",
            "private_ranges",
            countRole="counted",
            parts=1,
            factGroup="countedPackageRange",
            digits=4,
            prefixPattern="A",
        ),
    ]
    authority = write_inventory(
        tmp_path / "shipment-groups.json",
        blueprint,
        certificate(blueprint, regions=regions),
    )
    prepared = authority.apply_blueprint(blueprint)  # type: ignore[arg-type]

    composed = authority.build_ledger(
        prepared,
        draw("PACKAGE_DRUM"),
        {
            "privatePackingRelation": "CONTAINS",
            "relatedPackageType": "PACKAGE_CARTON",
        },
        23,
        1,
    )
    assert composed["relation"] == "CONTAINS"
    assert composed["facts"]["innerCartonRange"]["quantity"] == composed["relatedQuantity"]
    assert all(composed["surfaces"][f"held-numeric:{index}"][0] for index in range(3))
    counted_scope = str(("shipment", None, "countedPackageRange"))
    assert set(composed["scopeLedgers"]) == {counted_scope}
    assert composed["scopeLedgers"][counted_scope]["relation"] == "NONE"
    assert composed["scopeLedgers"][counted_scope]["facts"]["countedPackageRange"]["quantity"] == 4

    uncomposed = authority.build_ledger(
        prepared,
        draw("PACKAGE_DRUM"),
        {"privatePackingRelation": "NONE", "relatedPackageType": None},
        23,
        2,
    )
    assert uncomposed["surfaces"]["held-numeric:1"] == [""]
    assert (
        uncomposed["scopeLedgers"][counted_scope]["facts"]["countedPackageRange"]["quantity"] == 4
    )


def test_composition_bearing_container_scope_consumes_its_own_load(tmp_path: Path) -> None:
    source = "CONTAINER PACKING"
    blueprint = FakeBlueprint(
        "doc-container-composition",
        source,
        target(containers=1, allocation_rows=1),
    )
    region = quoted(
        source,
        "CONTAINER PACKING",
        "packing_clause",
        scope={"kind": "container", "index": 0},
    )
    authority = write_inventory(
        tmp_path / "container-composition.json",
        blueprint,
        certificate(blueprint, regions=[region]),
    )
    prepared = authority.apply_blueprint(blueprint)  # type: ignore[arg-type]
    current = draw("PACKAGE_PALLET")
    current["physical"]["unitGrossKg"] = "205"
    current["physical"]["grossWeight"]["value"] = "820"
    current["allocations"][0]["grossKg"] = "820"
    ledger = authority.build_ledger(
        prepared,
        current,
        {"privatePackingRelation": "SUPPORTS", "relatedPackageType": "PACKAGE_CARTON"},
        23,
        1,
    )
    assert "ON 4 PALLETS" in ledger["surfaces"]["held-numeric:0"][0]
    child = next(iter(ledger["scopeLedgers"].values()))
    assert child["relatedQuantity"] == ledger["relatedQuantity"]
    assert ledger["facts"]["supportEnvelope"]["totalTareKg"] == "100"


@pytest.mark.parametrize("parts", [2, 4])
def test_additive_components_share_one_positive_partition(tmp_path: Path, parts: int):
    source = "\n".join(f"COMPONENT {i}" for i in range(parts))
    blueprint = FakeBlueprint("doc-components", source, target())
    regions = [
        quoted(
            source,
            f"COMPONENT {i}",
            "component_package",
            factGroup="parts",
            componentCount=parts,
            componentIndex=i,
        )
        for i in range(parts)
    ]
    authority = write_inventory(
        tmp_path / "components.json", blueprint, certificate(blueprint, regions=regions)
    )
    prepared = authority.apply_blueprint(blueprint)
    for seed in range(20):
        ledger = authority.build_ledger(
            prepared,
            draw("PACKAGE_BALE", (31,)),
            {"privatePackingRelation": "NONE", "relatedPackageType": None},
            seed,
            1,
        )
        values = [int(v[0].split()[0]) for v in ledger["surfaces"].values()]
        assert len(values) == parts and min(values) > 0 and sum(values) == 31
        assert blueprint.target == prepared.target
    missing = certificate(blueprint, regions=regions[:-1])
    with pytest.raises(ValueError, match="own every component"):
        write_inventory(tmp_path / "missing.json", blueprint, missing)


def test_mixed_equation_has_independent_components_and_conserves(tmp_path: Path):
    source = "22PKGS=19PLTS/728CTNS+3CTNS=731CTNS"
    blueprint = FakeBlueprint("doc-mixed", source, target())
    value = certificate(
        blueprint,
        regions=[quoted(source, source, "mixed_package_equation")],
        constraints=[{"kind": "mixed_package_conservation"}],
    )
    authority = write_inventory(tmp_path / "mixed.json", blueprint, value)
    prepared = authority.apply_blueprint(blueprint)
    counts = set()
    for seed in range(30):
        ledger = authority.build_ledger(
            prepared,
            draw("PACKAGE_PACKAGE", (83,)),
            {
                "privatePackingRelation": "MIXED",
                "supportPackageType": "PACKAGE_PALLET",
                "relatedPackageType": "PACKAGE_CARTON",
            },
            seed,
            1,
        )
        facts = ledger["facts"]["mixedAccounting"]
        assert facts["supportQuantity"] + facts["looseQuantity"] == 83
        assert facts["packedQuantity"] + facts["looseQuantity"] == ledger["relatedQuantity"]
        assert (
            facts["supportEnvelope"]["totalTareKg"] <= facts["supportEnvelope"]["packingBudgetKg"]
        )
        counts.add((facts["supportQuantity"], facts["packedQuantity"]))
        assert ledger["surfaces"]["held-numeric:0"][0].startswith("83 PACKAGES=")
    assert len(counts) > 5
    assert prepared.target == blueprint.target


def test_unequal_container_nesting_reconciles_shipment_and_preserves_visibility(tmp_path: Path):
    source = "TOTAL PACKING\nFIRST PACKING\nSECOND PACKING"
    blueprint = FakeBlueprint("doc-unequal", source, target(containers=2, allocation_rows=1))
    regions = [
        quoted(source, "TOTAL PACKING", "packing_clause"),
        quoted(source, "FIRST PACKING", "packing_clause", scope={"kind": "container", "index": 0}),
        quoted(source, "SECOND PACKING", "packing_clause", scope={"kind": "container", "index": 1}),
    ]
    authority = write_inventory(
        tmp_path / "rows.json", blueprint, certificate(blueprint, regions=regions)
    )
    prepared = authority.apply_blueprint(blueprint)
    current = draw("PACKAGE_PALLET", (7, 19))
    current["physical"]["unitGrossKg"] = "205"
    current["physical"]["grossWeight"]["value"] = str(205 * 26)
    for row in current["allocations"]:
        row["grossKg"] = str(205 * row["packageQuantity"])
    ledger = authority.build_ledger(
        prepared,
        current,
        {"privatePackingRelation": "SUPPORTS", "relatedPackageType": "PACKAGE_CARTON"},
        61,
        1,
    )
    children = list(ledger["scopeLedgers"].values())
    assert [row["publicQuantity"] for row in children] == [7, 19]
    assert sum(row["relatedQuantity"] for row in children) == ledger["relatedQuantity"]
    assert "ON 7 PALLETS" in ledger["surfaces"]["held-numeric:1"][0]
    assert "ON 19 PALLETS" in ledger["surfaces"]["held-numeric:2"][0]
    assert len(prepared.target["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"]) == 1


def test_component_measures_consume_the_same_count_partition(tmp_path: Path):
    source = "8 CARTONS 1,600\n12 CARTONS 2,400"
    blueprint = FakeBlueprint("doc-component-measures", source, target())
    regions = []
    for index, (count, mass) in enumerate((("8 CARTONS", "1,600"), ("12 CARTONS", "2,400"))):
        common = {"factGroup": "parts", "componentCount": 2, "componentIndex": index}
        regions.extend(
            [
                quoted(
                    source,
                    mass,
                    "component_measurement",
                    **common,
                    fact="grossKg",
                    baseline=mass.replace(",", ""),
                ),
                quoted(source, count, "component_package", **common),
            ]
        )
    authority = write_inventory(
        tmp_path / "measure-parts.json", blueprint, certificate(blueprint, regions=regions)
    )
    prepared = authority.apply_blueprint(blueprint)
    current = draw("PACKAGE_DRUM", (31,))
    current["physical"]["unitVolumeM3"] = "0.28"
    ledger = authority.build_ledger(
        prepared, current, {"privatePackingRelation": "NONE", "relatedPackageType": None}, 71, 1
    )
    masses = [int(ledger["surfaces"][f"held-numeric:{i}"][0].replace(",", "")) for i in (0, 2)]
    counts = [int(ledger["surfaces"][f"held-numeric:{i}"][0].split()[0]) for i in (1, 3)]
    assert masses == [count * 192 for count in counts]
    assert sum(masses) == 31 * 192 and sum(counts) == 31


def test_complete_handling_target_tracks_fresh_packing_not_old_source(tmp_path):
    path = "documentPatch.goodsItemDetails[0].handlingInstructions[0]"
    original = target()
    original["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = ["502 PACKAGES"]
    source = "502 PACKAGES"
    bp = FakeBlueprint("doc-handling", source, original, regions=(
        RenderRegion("old-handling", 0, 0, len(source), source, (path,), "auxiliary"),
    ))
    region = quoted(source, source, "packing_clause", handlingTargetPath=path,
                    handlingTargetSuffix=")", prefix="(", suffix=" ON DECK")
    authority = write_inventory(tmp_path / "handling.json", bp, certificate(bp, regions=[region]))
    prepared = authority.apply_blueprint(bp)
    ledger = {"surfaces": {"held-numeric:0": ["(16 WOODEN CASES ON DECK"]}}
    projected, changes = project_handling_targets(prepared, original, ledger)
    assert projected["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        "(16 WOODEN CASES ON DECK)"
    ]
    assert changes == [{"path": path, "before": "502 PACKAGES",
                        "after": "(16 WOODEN CASES ON DECK)"}]
    assert original["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        "502 PACKAGES"
    ]
    prepared = replace(prepared, regions=())
    with pytest.raises(ValueError, match="existing exact target owner"):
        project_handling_targets(prepared, original, ledger)


def test_complete_handling_projection_cannot_own_a_quantity_target(tmp_path):
    bp = FakeBlueprint("doc-invalid-handling", "COUNT", target())
    region = quoted("COUNT", "COUNT", "packing_clause",
                    handlingTargetPath="documentPatch.goodsItemDetails[0].packageQuantity")
    with pytest.raises(ValueError, match="invalid target declaration"):
        write_inventory(tmp_path / "invalid-handling.json", bp, certificate(bp, regions=[region]))


def test_numeric_handling_replacement_cannot_silently_leave_stale_target(tmp_path):
    path = "documentPatch.goodsItemDetails[0].handlingInstructions[0]"
    bp = FakeBlueprint("doc-stale-handling", "502 PACKAGES", target(), regions=(
        RenderRegion("handling", 0, 0, 12, "502 PACKAGES", (path,), "auxiliary"),
    ))
    authority = write_inventory(tmp_path / "stale-handling.json", bp, certificate(bp, regions=[
        quoted(bp.source, bp.source, "packing_clause")
    ]))
    with pytest.raises(ValueError, match="requires explicit target projection"):
        authority.apply_blueprint(bp)


def test_numeric_ledger_rejects_numbered_bulk_liquid_inner_packages(tmp_path: Path):
    source = "PACKING"
    blueprint = FakeBlueprint("doc-direct-fill", source, target())
    authority = write_inventory(
        tmp_path / "direct.json",
        blueprint,
        certificate(blueprint, regions=[quoted(source, source, "packing_clause")]),
    )
    prepared = authority.apply_blueprint(blueprint)
    with pytest.raises(ValueError, match="countable packages"):
        authority.build_ledger(
            prepared,
            draw("PACKAGE_DRUM", (82,)),
            {"privatePackingRelation": "CONTAINS", "relatedPackageType": "PACKAGE_BULK_LIQUID"},
            7,
            1,
        )
    ledger = authority.build_ledger(
        prepared,
        draw("PACKAGE_DRUM", (82,)),
        {"privatePackingRelation": "NONE", "relatedPackageType": None},
        7,
        1,
    )
    assert ledger["surfaces"]["held-numeric:0"] == ["82 DRUMS"]


def test_discrete_article_support_is_scoped_and_counted_once_per_container(tmp_path: Path):
    source = "FIRST\nSECOND"
    blueprint = FakeBlueprint("doc-discrete-rows", source, target(containers=2, allocation_rows=2))
    regions = [
        quoted(source, word, "packing_clause", scope={"kind": "container", "index": i})
        for i, word in enumerate(("FIRST", "SECOND"))
    ]
    authority = write_inventory(
        tmp_path / "article-rows.json", blueprint, certificate(blueprint, regions=regions)
    )
    prepared = authority.apply_blueprint(blueprint)
    current = draw("PACKAGE_UNPACKED_OR_UNPACKAGED", (2, 5))
    current["physical"]["unitGrossKg"] = "180"
    current["physical"]["grossWeight"]["value"] = str(180 * 7)
    for row in current["allocations"]:
        row["grossKg"] = str(180 * row["packageQuantity"])
    plan = {"privatePackingRelation": "ON_SUPPORT", "relatedPackageType": "PACKAGE_PALLET"}
    current = authority.prepare_draw(
        prepared, current, plan, {"countedUnitContext": "discrete_article"}
    )
    ledger = authority.build_ledger(prepared, current, plan, 71, 1)
    assert ledger["relatedQuantity"] == 7
    children = list(ledger["scopeLedgers"].values())
    assert [child["relatedQuantity"] for child in children] == [2, 5]
    assert ledger["facts"]["supportEnvelope"]["totalTareKg"] == "175"
    assert current["physical"]["grossWeight"]["value"] == "1435"


def test_support_relation_and_composed_discrete_capacity_conserve(tmp_path: Path) -> None:
    source = "SUPPORT"
    blueprint = FakeBlueprint("doc-support", source, target())
    value = certificate(
        blueprint,
        regions=[quoted(source, "SUPPORT", "support_mass_breakdown")],
    )
    authority = write_inventory(tmp_path / "support.json", blueprint, value)
    prepared = authority.apply_blueprint(blueprint)  # type: ignore[arg-type]
    pallet_draw = draw("PACKAGE_PALLET", (2,))
    pallet_draw["physical"]["unitGrossKg"] = "205"
    pallet_draw["physical"]["grossWeight"]["value"] = "410"
    pallet_draw["allocations"][0]["grossKg"] = "410"
    ledger = authority.build_ledger(
        prepared,  # type: ignore[arg-type]
        pallet_draw,
        {
            "privatePackingRelation": "SUPPORTS",
            "relatedPackageType": "PACKAGE_CARTON",
        },
        11,
        0,
    )
    support = ledger["facts"]["supportEnvelope"]
    assert support["quantity"] == 2
    assert support["totalTareKg"] == "50"
    assert ledger["facts"]["support_mass_breakdown"]["conserved"] is True

    article = draw("PACKAGE_UNPACKED_OR_UNPACKAGED", (2,))
    article["physical"].update({"unitNetKg": "180", "unitGrossKg": "180"})
    article["physical"]["grossWeight"]["value"] = "360"
    article["allocations"][0]["grossKg"] = "360"
    composed = authority.prepare_draw(
        prepared,  # type: ignore[arg-type]
        article,
        {
            "privatePackingRelation": "ON_SUPPORT",
            "relatedPackageType": "PACKAGE_PALLET",
        },
        {"countedUnitContext": "discrete_article"},
    )
    assert composed["physical"]["unitGrossKg"] == "205"
    authority.validate_composed_capacity(
        composed,
        [{"quantity": 2, "capacity": {"payloadKg": "500", "volumeM3": "3"}}],
    )
    with pytest.raises(ValueError, match="payload"):
        authority.validate_composed_capacity(
            composed,
            [{"quantity": 2, "capacity": {"payloadKg": "400", "volumeM3": "3"}}],
        )


def test_partial_shared_visibility_uses_mass_partition_not_public_counts(tmp_path: Path) -> None:
    source = "LOT A 1.0 LOT B 1.0 RANGE"
    blueprint = FakeBlueprint("doc-shared", source, target(containers=2, allocation_rows=2))
    regions = [
        quoted(
            source,
            "LOT A 1.0",
            "private_measurement",
            scope={"kind": "shared_container"},
            fact="partitionNetKg",
            factGroup="lots",
            partitionGroup="lots",
            partitionIndex=0,
            baseline="1.0",
            unit="kilogram",
        ),
        quoted(
            source,
            "LOT B 1.0",
            "private_measurement",
            scope={"kind": "shared_container"},
            fact="partitionNetKg",
            factGroup="lots",
            partitionGroup="lots",
            partitionIndex=1,
            baseline="1.0",
            unit="kilogram",
        ),
        quoted(
            source,
            "RANGE",
            "shared_container_range",
            scope={"kind": "shared_container"},
            factGroup="sharedRange",
            rangePrefix="LOT-",
        ),
    ]
    value = certificate(
        blueprint,
        regions=regions,
        constraints=[{"kind": "equal_container_loads"}],
    )
    authority = write_inventory(tmp_path / "shared.json", blueprint, value)
    prepared = authority.apply_blueprint(blueprint)  # type: ignore[arg-type]
    current = draw("PACKAGE_BULK_LIQUID", (1, 1))
    current["physical"].update({"unitNetKg": "8000", "unitGrossKg": "8000", "unitVolumeM3": "16"})
    current["physical"]["grossWeight"] = {"value": "16000", "unit": "kilogram"}
    for row in current["allocations"]:
        row.update({"netKg": "8000", "grossKg": "8000", "volumeM3": "16"})
    composed = authority.prepare_draw(
        prepared,  # type: ignore[arg-type]
        current,
        {"privatePackingRelation": "NONE", "relatedPackageType": None},
        {"countedUnitContext": "bulk_loading_lot"},
    )
    assert composed["provenance"][PRIVATE_PARTITION_KEY]["publicPackageCountsInferred"] is False
    ledger = authority.build_ledger(
        prepared,  # type: ignore[arg-type]
        composed,
        {"privatePackingRelation": "NONE", "relatedPackageType": None},
        19,
        0,
    )
    facts = ledger["facts"]["sharedContainerFacts"]
    assert facts["publicAllocationsInferred"] is False
    assert sum(map(float, facts["positiveLotMassPartitionsKg"]["lots"])) == 8000


def test_prepare_draw_requires_the_canonical_counting_context_key(tmp_path: Path) -> None:
    blueprint = FakeBlueprint("doc-context", "SOURCE", target())
    authority = write_inventory(tmp_path / "context.json", blueprint, certificate(blueprint))
    prepared = authority.apply_blueprint(blueprint)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="countedUnitContext"):
        authority.prepare_draw(
            prepared,  # type: ignore[arg-type]
            draw("PACKAGE_BULK_LIQUID"),
            {"privatePackingRelation": "NONE", "relatedPackageType": None},
            {"declaredUnitContext": "bulk_loading_lot"},
        )


def test_source_pins_hold_and_allocation_mask_are_enforced(tmp_path: Path) -> None:
    blueprint = FakeBlueprint("doc-pins", "SOURCE", target(containers=1, allocation_rows=1))
    value = certificate(blueprint)
    authority = write_inventory(tmp_path / "pins.json", blueprint, value)
    changed = copy.deepcopy(blueprint)
    object.__setattr__(changed, "source", "CHANGED")
    with pytest.raises(ValueError, match="source certificate hash changed"):
        authority.apply_blueprint(changed)  # type: ignore[arg-type]
    held = certificate(blueprint, hold_reasons=["requires independent review"])
    held_authority = write_inventory(tmp_path / "held.json", blueprint, held)
    with pytest.raises(ValueError, match="certificate held"):
        held_authority.apply_blueprint(blueprint)  # type: ignore[arg-type]
    masked = copy.deepcopy(blueprint)
    masked.target["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"][0].clear()
    with pytest.raises(ValueError, match="allocation visibility"):
        authority.apply_blueprint(masked)  # type: ignore[arg-type]


def test_wording_projection_is_idempotent_and_rejects_duplicates() -> None:
    ledger = {"wordingAppends": {"goods": "12 BOXES; 10 KG NET PER BOX."}}
    projected, receipt = project_wording({"goods": "WIDGETS"}, ledger)
    assert projected["goods"].endswith("12 BOXES; 10 KG NET PER BOX.")
    assert receipt["unchangedOtherFields"] is True
    again, second = project_wording(projected, ledger)
    assert again == projected
    assert second["retainedClauses"] == ledger["wordingAppends"]
    duplicate = {"goods": projected["goods"] + "\n" + ledger["wordingAppends"]["goods"]}
    with pytest.raises(ValueError, match="more than once"):
        project_wording(duplicate, ledger)
