from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.locality_registry import LocalityRecord
from document_ocr.synthesis.package_registry import (
    LoadedPackageRegistry,
    PackageRegistryEntry,
    PackageRegistryPayload,
)
from document_ocr.synthesis.raw_text_certification_host import (
    apply_deterministic_line_repairs,
    complete_deterministic_repair_set,
)
from document_ocr.synthesis.raw_text_certification_invariants import (
    CertificationInvariantAudit,
    DeterministicCertificationFinding,
    DeterministicLineRepair,
    InvariantCheck,
    InvariantEvidence,
    audit_certification_invariants,
    compile_certification_invariant_envelope,
)
from document_ocr.synthesis.raw_text_certification_references import (
    CertificationReferenceIndex,
    compile_certification_reference_index,
)
from document_ocr.synthesis.raw_text_inventory import (
    DeterministicInventoryEdit,
    InventoryCandidate,
)
from document_ocr.synthesis.transport_capacity import TransportCapacityLimits

_DOCUMENT_ID = "doc_" + "a" * 64
_SOURCE = (
    "--- PAGE 1 ---\n"
    "Signed on behalf of the Carrier: OLD OCEAN LINE\n"
    "VAT NO: AB123456\n"
    "VAT NO: AB123456\n"
    "SEAL: OLD12345\n"
)
_CANDIDATE = (
    "--- PAGE 1 ---\n"
    "Signed on behalf of the Carrier: NEW OCEAN LINE\n"
    "VAT NO: CD654321\n"
    "VAT NO: CD654321\n"
    "SEAL: ZX987654\n"
)


def _package_registry() -> LoadedPackageRegistry:
    payload = PackageRegistryPayload(
        schemaVersion=1,
        registryKind="package",
        sourceAuthority="test",
        sourceRevision="1",
        sourcePath="test.json",
        sourceSha256="4" * 64,
        entries=(
            PackageRegistryEntry(
                categoryToken="PACKAGE_VEHICLE",
                applicationCode="VN",
                displayName="Vehicle",
            ),
            PackageRegistryEntry(
                categoryToken="PACKAGE_CARTON",
                applicationCode="CT",
                displayName="Carton",
            ),
            PackageRegistryEntry(
                categoryToken="PACKAGE_PIECE",
                applicationCode="PC",
                displayName="Piece",
            ),
        ),
    )
    return LoadedPackageRegistry(path="test.json", sha256="5" * 64, payload=payload)


def _reference_index() -> CertificationReferenceIndex:
    reserved = {"AX", "CN", "EG", "ES", "FI", "IN", "JP", "MU", "SE", "US"}
    countries = [
        {"alpha_2": "EG", "name": "Egypt"},
        {"alpha_2": "SE", "name": "Sweden"},
        {"alpha_2": "FI", "name": "Finland"},
        {"alpha_2": "AX", "name": "Åland Islands"},
        {"alpha_2": "CN", "name": "China"},
        {"alpha_2": "IN", "name": "India"},
        {"alpha_2": "ES", "name": "Spain"},
        {"alpha_2": "JP", "name": "Japan"},
        {"alpha_2": "MU", "name": "Mauritius"},
        {"alpha_2": "US", "name": "United States"},
    ]
    for first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for second in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            code = first + second
            if code not in reserved:
                countries.append({"alpha_2": code, "name": f"Test Country {code}"})
            if len(countries) == 200:
                break
        if len(countries) == 200:
            break
    iso_payload = json.dumps({"3166-1": countries}, ensure_ascii=False).encode("utf-8")
    localities = (
        LocalityRecord(
            geoname_id=1,
            canonical_name="Damietta",
            ascii_name="Damietta",
            country_code="EG",
            feature_code="PPLA",
            population=200_000,
            latitude=Decimal("31.4165"),
            longitude=Decimal("31.8133"),
            timezone="Africa/Cairo",
        ),
        LocalityRecord(
            geoname_id=2,
            canonical_name="Kalmar",
            ascii_name="Kalmar",
            country_code="SE",
            feature_code="PPLA",
            population=41_000,
            latitude=Decimal("56.6616"),
            longitude=Decimal("16.3616"),
            timezone="Europe/Stockholm",
        ),
        LocalityRecord(
            geoname_id=3,
            canonical_name="Ystad",
            ascii_name="Ystad",
            country_code="SE",
            feature_code="PPLA3",
            population=20_000,
            latitude=Decimal("55.4297"),
            longitude=Decimal("13.8204"),
            timezone="Europe/Stockholm",
        ),
    )
    return compile_certification_reference_index(
        iso3166_payload=iso_payload,
        expected_iso3166_sha256=sha256_bytes(iso_payload),
        locality_records=localities,
        geonames_registry_receipt_sha256="1" * 64,
        geonames_registry_content_sha256="2" * 64,
        geonames_jsonl_sha256="3" * 64,
        expected_phonenumberslite_version="9.0.38",
        package_registry=_package_registry(),
    )


_REFERENCES = _reference_index()
_CAPACITY_LIMITS = TransportCapacityLimits(
    policy="source_type_aware_maersk_upper_bounds_v1",
    published_reference_margin_fraction=Decimal("0.05"),
    twenty_standard_payload_kg=Decimal("28300"),
    twenty_standard_volume_m3=Decimal("33.2"),
    forty_standard_payload_kg=Decimal("28870"),
    forty_standard_volume_m3=Decimal("67.7"),
    forty_high_cube_payload_kg=Decimal("28690"),
    forty_high_cube_volume_m3=Decimal("76.4"),
    forty_five_high_cube_payload_kg=Decimal("27650"),
    forty_five_high_cube_volume_m3=Decimal("86"),
    out_of_gauge_payload_kg=Decimal("47300"),
    unclassified_payload_kg=Decimal("47300"),
    unclassified_volume_m3=Decimal("86"),
)


def _repair_audit(
    *repairs: DeterministicLineRepair,
    repairable: bool = True,
) -> CertificationInvariantAudit:
    evidence = tuple(
        InvariantEvidence(lineId=line_id, currentLine="placeholder")
        for line_id in sorted({row.lineId for row in repairs})
    ) or (InvariantEvidence(lineId="L00001", currentLine="placeholder"),)
    finding = DeterministicCertificationFinding(
        invariantId="D" + "1" * 16,
        findingKind="target_fact_mismatch",
        dimension="target_fact_fidelity",
        evidence=evidence,
        targetPaths=("documentPatch.billOfLadingNumber",),
        problem="Exact compiler-owned scalar differs.",
        repairs=repairs if repairable else (),
    )
    return CertificationInvariantAudit(
        schemaVersion=1,
        envelopeSha256="2" * 64,
        passed=False,
        checks=(InvariantCheck(checkId="target_literals", findings=1, passed=False),),
        findings=(finding,),
    )


def test_complete_deterministic_repair_set_applies_atomically_without_topology_change() -> None:
    candidate = "--- PAGE 1 ---\r\nB/L: OLD-123 / PORT: OLDPORT\r\n"
    current_line = "B/L: OLD-123 / PORT: OLDPORT"
    line_hash = sha256_bytes(current_line.encode())
    repairs = (
        DeterministicLineRepair(
            method="replace_exact_fragment_v1",
            lineId="L00002",
            expectedCurrentLineSha256=line_hash,
            oldFragment="OLD-123",
            newFragment="NEW-456",
        ),
        DeterministicLineRepair(
            method="replace_exact_fragment_v1",
            lineId="L00002",
            expectedCurrentLineSha256=line_hash,
            oldFragment="OLDPORT",
            newFragment="NEWPORT",
        ),
    )
    complete = complete_deterministic_repair_set(_repair_audit(*repairs))

    assert complete == repairs
    assert apply_deterministic_line_repairs(candidate, complete) == (
        "--- PAGE 1 ---\r\nB/L: NEW-456 / PORT: NEWPORT\r\n"
    )


def test_deterministic_repair_requires_complete_authority_and_exact_line_hash() -> None:
    repair = DeterministicLineRepair(
        method="replace_exact_fragment_v1",
        lineId="L00002",
        expectedCurrentLineSha256=sha256_bytes(b"B/L: OLD-123"),
        oldFragment="OLD-123",
        newFragment="NEW-456",
    )

    assert complete_deterministic_repair_set(_repair_audit(repair, repairable=False)) is None
    with pytest.raises(ValueError, match="line identity differs"):
        apply_deterministic_line_repairs("--- PAGE 1 ---\nB/L: OTHER\n", (repair,))


def test_deterministic_repair_rejects_overlapping_fragment_claims() -> None:
    current = "ABCDEF"
    line_hash = sha256_bytes(current.encode())
    repairs = (
        DeterministicLineRepair(
            method="replace_exact_fragment_v1",
            lineId="L00001",
            expectedCurrentLineSha256=line_hash,
            oldFragment="ABCD",
            newFragment="LEFT",
        ),
        DeterministicLineRepair(
            method="replace_exact_fragment_v1",
            lineId="L00001",
            expectedCurrentLineSha256=line_hash,
            oldFragment="CDEF",
            newFragment="RIGHT",
        ),
    )

    complete = complete_deterministic_repair_set(_repair_audit(*repairs))
    assert complete is not None
    with pytest.raises(ValueError, match="overlap"):
        apply_deterministic_line_repairs(current, complete)


def _contract() -> dict[str, Any]:
    return {
        "deterministicPrefills": [],
        "sourceLabel": {
            "documentPatch": {
                "parties": {"carrier": {"name": "OLD OCEAN LINE"}},
                "containers": [{"sealNumber": "OLD12345"}],
            }
        },
        "targetLabel": {
            "documentPatch": {
                "parties": {"carrier": {"name": "NEW OCEAN LINE"}},
                "containers": [{"sealNumber": "ZX987654"}],
            }
        },
        "surfaceRenderingRequirements": [
            {
                "kind": "carrier_principal_identity",
                "targetPath": "documentPatch.parties.carrier.name",
                "sourceSurface": "OLD OCEAN LINE",
                "targetSurface": "NEW OCEAN LINE",
                "sourceOccurrences": 1,
                "contextEvidence": "Signed on behalf of the Carrier: OLD OCEAN LINE",
                "sourceLineIds": ["L00002"],
            }
        ],
        "targetLiteralRequirements": [],
        "targetValueOccurrenceRequirements": [],
        "anchoredScalarReplacementRequirements": [
            {
                "targetPaths": ["documentPatch.containers[0].sealNumber"],
                "sourceLineIds": ["L00005"],
                "sourceSurface": "OLD12345",
                "targetSurface": "ZX987654",
                "surfaceKind": "scalar",
            }
        ],
        "sourceSemanticRoleHints": [],
        "inlineSlotTopologyRequirements": [],
        "sourceStatusPreservationRequirements": [],
        "operationalFlavorRequirements": [],
        "cargoFlavorRewriteRequirements": [],
        "jurisdictionalSurfaceRequirements": [],
    }


def _inventory() -> tuple[InventoryCandidate, ...]:
    return (
        InventoryCandidate(
            candidateId="I0001",
            category="changed_source_occurrence",
            sourceSurface="OLD OCEAN LINE",
            lineIds=("L00002",),
            targetPaths=("documentPatch.parties.carrier.name",),
            targetSemantics=[
                {
                    "path": "documentPatch.parties.carrier.name",
                    "target": "NEW OCEAN LINE",
                }
            ],
            disposition="model_residual",
            rationale="Exact source carrier slot.",
        ),
        InventoryCandidate(
            candidateId="I0002",
            category="changed_source_occurrence",
            sourceSurface="OLD12345",
            lineIds=("L00005",),
            targetPaths=("documentPatch.containers[0].sealNumber",),
            targetSemantics=[
                {
                    "path": "documentPatch.containers[0].sealNumber",
                    "target": "ZX987654",
                }
            ],
            disposition="model_residual",
            rationale="Exact source seal slot.",
        ),
    )


def _edits() -> tuple[DeterministicInventoryEdit, ...]:
    return (
        DeterministicInventoryEdit(
            sourceSurface="AB123456",
            targetSurface="CD654321",
            lineIds=("L00003", "L00004"),
            method="hmac_shape_preserving_auxiliary_v1",
        ),
    )


def _audit(candidate: str = _CANDIDATE) -> CertificationInvariantAudit:
    contract = _contract()
    inventory = _inventory()
    edits = _edits()
    envelope = compile_certification_invariant_envelope(
        document_id=_DOCUMENT_ID,
        source=_SOURCE,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=edits,
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )
    return audit_certification_invariants(
        envelope=envelope,
        document_id=_DOCUMENT_ID,
        source=_SOURCE,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=edits,
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )


def _minimal_contract(
    *, source_label: dict[str, Any], target_label: dict[str, Any]
) -> dict[str, Any]:
    return {
        "deterministicPrefills": [],
        "sourceLabel": source_label,
        "targetLabel": target_label,
        "surfaceRenderingRequirements": [],
        "targetLiteralRequirements": [],
        "targetValueOccurrenceRequirements": [],
        "anchoredScalarReplacementRequirements": [],
        "sourceSemanticRoleHints": [],
        "inlineSlotTopologyRequirements": [],
        "sourceStatusPreservationRequirements": [],
        "operationalFlavorRequirements": [],
        "cargoFlavorRewriteRequirements": [],
        "jurisdictionalSurfaceRequirements": [],
    }


def _audit_minimal(
    *,
    source: str,
    candidate: str,
    source_label: dict[str, Any],
    target_label: dict[str, Any],
    inventory: tuple[InventoryCandidate, ...] = (),
    deterministic_edits: tuple[DeterministicInventoryEdit, ...] = (),
    deterministic_prefills: tuple[dict[str, Any], ...] = (),
    anchored_replacements: tuple[dict[str, Any], ...] = (),
    jurisdictional_requirements: tuple[dict[str, Any], ...] = (),
) -> CertificationInvariantAudit:
    contract = _minimal_contract(source_label=source_label, target_label=target_label)
    contract["deterministicPrefills"] = list(deterministic_prefills)
    contract["anchoredScalarReplacementRequirements"] = list(anchored_replacements)
    contract["jurisdictionalSurfaceRequirements"] = list(jurisdictional_requirements)
    envelope = compile_certification_invariant_envelope(
        document_id=_DOCUMENT_ID,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )
    return audit_certification_invariants(
        envelope=envelope,
        document_id=_DOCUMENT_ID,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )


def test_complete_deterministic_invariant_contract_passes() -> None:
    audit = _audit()

    assert audit.passed
    assert not audit.findings
    assert all(row.passed for row in audit.checks)


def test_pinned_vehicle_package_cannot_describe_parts_for_another_article() -> None:
    source = "--- PAGE 1 ---\n1886 CARTONS\nCONTAIN 1,886 CARTONS\nCOTTON COMBED YARN\n"
    candidate = (
        "--- PAGE 1 ---\n2495 VEHICLES\nCONTAIN 2,495 VEHICLES\nPARTS FOR HARVESTING MACHINERY\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "COTTON COMBED YARN"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 1886,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "PARTS FOR HARVESTING MACHINERY",
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 2495,
                    "typeCategory": "PACKAGE_VEHICLE",
                }
            ],
        }
    }
    contract = _minimal_contract(source_label=source_label, target_label=target_label)
    contract["anchoredScalarReplacementRequirements"] = [
        {
            "targetPaths": [
                "documentPatch.cargoPackages[0].quantity",
                "documentPatch.cargoPackages[0].typeCategory",
            ],
            "sourceLineIds": ["L00002"],
            "sourceSurface": "1886 CARTONS",
            "targetSurface": "2495 VEHICLES",
            "surfaceKind": "scalar",
        },
        {
            "targetPaths": [
                "documentPatch.cargoPackages[0].quantity",
                "documentPatch.cargoPackages[0].typeCategory",
            ],
            "sourceLineIds": ["L00003"],
            "sourceSurface": "1,886 CARTONS",
            "targetSurface": "2,495 VEHICLES",
            "surfaceKind": "scalar",
        },
    ]
    contract["cargoFlavorRewriteRequirements"] = [
        {
            "requirementId": "cargo-group-1-span-1",
            "targetPath": "documentPatch.cargoGroups[0].description",
            "targetDescription": "PARTS FOR HARVESTING MACHINERY",
            "sourceLineIds": ["L00004"],
            "sourceSurfaces": ["COTTON COMBED YARN"],
            "enforcePackageSurfaceGuard": True,
            "allowedPackageSurfaces": ["VEHICLE", "VEHICLES"],
        }
    ]
    envelope = compile_certification_invariant_envelope(
        document_id=_DOCUMENT_ID,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=(),
        deterministic_edits=(),
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )

    audit = audit_certification_invariants(
        envelope=envelope,
        document_id=_DOCUMENT_ID,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=(),
        deterministic_edits=(),
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )

    package_finding = next(
        row for row in audit.findings if "internally contradictory" in row.problem
    )
    assert package_finding.findingKind == "cargo_or_dangerous_goods_mismatch"
    assert [row.lineId for row in package_finding.evidence] == [
        "L00002",
        "L00003",
        "L00004",
    ]
    assert package_finding.repairs == ()


def test_container_local_package_row_cannot_claim_shipment_wide_package_totals() -> None:
    source = "--- PAGE 1 ---\nOLDU1234567\n40 ROLL(S) OF PULP\n160 ROLLS (80 PACKS)\n"
    candidate = (
        "--- PAGE 1 ---\n"
        "NEWU1234567\n"
        "132 CARTON(S) PACKED IN 527 CARTONS AND 264 PIECES\n"
        "527 CARTONS AND 264 PIECES\n"
    )
    source_label: dict[str, Any] = {"documentPatch": {"containers": []}}
    target_label: dict[str, Any] = {
        "documentPatch": {
            "containers": [{"containerNumber": "NEWU1234567"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 527,
                    "typeCategory": "PACKAGE_CARTON",
                },
                {
                    "groupId": "g1",
                    "packageId": "p2",
                    "quantity": 264,
                    "typeCategory": "PACKAGE_PIECE",
                },
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [{"containerNumber": "NEWU1234567", "packageQuantity": 132}],
                }
            ],
        }
    }
    contract = _minimal_contract(source_label=source_label, target_label=target_label)
    contract["operationalFlavorRequirements"] = [
        {
            "requirementId": "operational-L00003-package_quantity",
            "kind": "package_quantity",
            "sourceLineId": "L00003",
            "sourceEvidence": "OLDU1234567\n40 ROLL(S) OF PULP",
            "sourceCanonicalValue": "40",
            "sourceValueSurface": "40",
            "sourceMeasurementStartColumn": 0,
            "sourceGrammar": "labeled_measurement",
            "targetCanonicalValue": "132",
            "targetValueSurface": "132",
            "sameLineFollowingContainerNumber": None,
            "targetContainerNumber": "NEWU1234567",
            "targetEquipmentFamily": "twenty_standard",
            "consistencyGroupId": "NEWU1234567-package_quantity",
            "samplingMethod": "target_package_allocation_v1",
            "maximumValue": None,
            "empiricalProfileDocumentId": None,
        }
    ]

    def run(text: str) -> CertificationInvariantAudit:
        envelope = compile_certification_invariant_envelope(
            document_id=_DOCUMENT_ID,
            source=source,
            candidate=text,
            contract=contract,
            inventory=(),
            deterministic_edits=(),
            references=_REFERENCES,
            capacity_limits=_CAPACITY_LIMITS,
        )
        return audit_certification_invariants(
            envelope=envelope,
            document_id=_DOCUMENT_ID,
            source=source,
            candidate=text,
            contract=contract,
            inventory=(),
            deterministic_edits=(),
            references=_REFERENCES,
            capacity_limits=_CAPACITY_LIMITS,
        )

    audit = run(candidate)

    finding = next(row for row in audit.findings if "shipment-wide package facts" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00003"]
    assert finding.targetPaths == (
        "documentPatch.cargoPackages[0]",
        "documentPatch.cargoPackages[1]",
    )
    corrected = candidate.replace(
        "132 CARTON(S) PACKED IN 527 CARTONS AND 264 PIECES",
        "132 CARTON(S) SHIPPER'S LOAD AND COUNT",
    )
    assert not [row for row in run(corrected).findings if "shipment-wide" in row.problem]


def test_anchored_source_value_yields_one_exact_host_repair() -> None:
    candidate = _CANDIDATE.replace("ZX987654", "OLD12345")

    audit = _audit(candidate)

    finding = next(row for row in audit.findings if row.targetPaths)
    assert finding.findingKind == "target_fact_mismatch"
    assert [row.lineId for row in finding.repairs] == ["L00005"]
    assert finding.repairs[0].oldFragment == "OLD12345"
    assert finding.repairs[0].newFragment == "ZX987654"


def test_carrier_principal_cannot_delete_its_static_legal_relation() -> None:
    candidate = _CANDIDATE.replace(
        "Signed on behalf of the Carrier: NEW OCEAN LINE",
        "Signed on behalf of NEW OCEAN LINE",
    )

    audit = _audit(candidate)

    template_findings = tuple(
        row for row in audit.findings if row.dimension == "template_format_and_model_artifacts"
    )
    assert len(template_findings) == 1
    finding = template_findings[0]
    assert [row.lineId for row in finding.evidence] == ["L00002"]
    assert "Static punctuation/label" in finding.problem


def test_changed_carrier_retires_source_branded_office_clauses() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Carrier: Maersk A/S\n"
        "Signed for the Carrier Maersk A/S\n"
        "Place of issue is the Maersk line India office.\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "Carrier: Bluehaven Ocean Transit PLC\n"
        "Signed for the Carrier Bluehaven Ocean Transit PLC\n"
        "Place of issue is the Maersk line India office.\n"
    )
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "Maersk A/S"}}}}
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Bluehaven Ocean Transit PLC"}}}
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    finding = next(
        row for row in audit.findings if row.invariantId and "carrier-branded" in row.problem
    )
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00003", "L00004"]
    assert finding.targetPaths == ("documentPatch.parties.carrier.name",)
    assert not finding.repairs


def test_changed_carrier_retires_carrier_reference_identifier_prefix() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Carrier: Hapag-Lloyd Aktiengesellschaft\n"
        "Carrier's Reference: SWB-No.: 72930692 HLCUMTR230512758 Page: 2 / 6\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "Carrier: Rivage Continental Carriers\n"
        "Carrier's Reference: SWB-No.: 97829556 HLCUMTR250319561 Page: 2 / 6\n"
    )
    source_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Hapag-Lloyd Aktiengesellschaft"}}}
    }
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Rivage Continental Carriers"}}}
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    finding = next(row for row in audit.findings if "alphabetic stem" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00003"]
    assert not finding.repairs


def test_changed_carrier_accepts_new_carrier_reference_identifier_prefix() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Carrier: Hapag-Lloyd Aktiengesellschaft\n"
        "Carrier's Reference: SWB-No.: 72930692 HLCUMTR230512758 Page: 2 / 6\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "Carrier: Rivage Continental Carriers\n"
        "Carrier's Reference: SWB-No.: 97829556 RIVC250319561 Page: 2 / 6\n"
    )
    source_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Hapag-Lloyd Aktiengesellschaft"}}}
    }
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Rivage Continental Carriers"}}}
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    assert audit.passed


def test_deterministic_carrier_reference_owner_is_not_double_counted() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Carrier: Hapag-Lloyd Aktiengesellschaft\n"
        "Carrier's Reference: HLCUMTR230512758\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "Carrier: Rivage Continental Carriers\n"
        "Carrier's Reference: HLCUMTR230512758\n"
    )
    source_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Hapag-Lloyd Aktiengesellschaft"}}}
    }
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Rivage Continental Carriers"}}}
    }
    edit = DeterministicInventoryEdit(
        sourceSurface="HLCUMTR230512758",
        targetSurface="WSJBOKZ258707141",
        lineIds=("L00003",),
        method="hmac_shape_preserving_auxiliary_v1",
    )

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
        deterministic_edits=(edit,),
    )

    assert len(audit.findings) == 1
    assert "Host-generated auxiliary replacement" in audit.findings[0].problem
    assert [row.lineId for row in audit.findings[0].evidence] == ["L00003"]


def test_changed_carrier_does_not_claim_static_branded_legal_prose() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Signed for Carrier Sealand Europe A/S\n"
        "\n"
        "Declared Value Charges (see clause 7.3 of the Sealand Bill of Lading).\n"
        "The U.S. Treasury Department Office list and https://sealand.example/policy apply.\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "Signed for Carrier Northwave Meridian Lines A/S\n"
        "\n"
        "Declared Value Charges (see clause 7.3 of the Sealand Bill of Lading).\n"
        "The U.S. Treasury Department Office list and https://sealand.example/policy apply.\n"
    )
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "Sealand Europe A/S"}}}}
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Northwave Meridian Lines A/S"}}}
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    assert audit.passed


def test_changed_carrier_preserves_static_carrier_role_template() -> None:
    source = "--- PAGE 1 ---\nSigned on behalf of the Carrier: TRANSGLORY\n"
    candidate = "--- PAGE 1 ---\nSigned on behalf of Aureline Oceanic Carriers\n"
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "TRANSGLORY S.A."}}}}
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "Aureline Oceanic Carriers S.A."}}}
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    finding = next(row for row in audit.findings if "carrier-role line" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002"]
    assert finding.dimension == "template_format_and_model_artifacts"


def test_source_carrier_registration_residue_includes_changed_carrier_context() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "CARRIER: CMA CGM Société Anonyme au Capital de 234 988 330 Euros\n"
        "Head Office: 4, quai d'Arenc - 13002 Marseille - France\n"
        "562 024 422 R.C.S. Marseille\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "CARRIER: Valmeris Oceanic AG au Capital de 2 150 000 Euros\n"
        "Head Office: Avenue des Mélèzes 18 Martigny-Ville - Switzerland\n"
        "562 024 422 Reg. No. Martigny-Ville\n"
    )
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "CMA CGM Société Anonyme"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "Valmeris Oceanic AG"}}}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    finding = next(row for row in audit.findings if "'r.c.s'" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00004"]


def test_changed_party_retires_source_address_fragments_in_its_owned_block() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "FOREIGN EXPORTER NAME:\n"
        "UNICHARM CORPORATION\n"
        "COUNTRY: JAPAN\n"
        "ADDRESS: 182 SHIMOBUN,\n"
        "KINSEI-CHO,\n"
        "SHIKOKUCHUO-CITY\n"
        "799-0111\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "FOREIGN EXPORTER NAME:\n"
        "DELTA HORIZON TRADING COMPANY\n"
        "COUNTRY: EGYPT\n"
        "ADDRESS: 182 SHIMOBUN,\n"
        "KINSEI-CHO,\n"
        "SHUBRA AL KHAYMAH\n"
        "799-0111\n"
    )
    source_label = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "name": "UNICHARM CORPORATION",
                        "address": "182 SHIMOBUN, KINSEI-CHO, EHIME-PREF. 799-0111",
                        "city": "SHIKOKUCHUO-CITY",
                        "country": "JAPAN",
                    }
                ]
            }
        }
    }
    target_label = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "name": "DELTA HORIZON TRADING COMPANY",
                        "address": "18 Al Mashtal Street, Industrial District",
                        "city": "Shubra al Khaymah",
                        "country": "Egypt",
                    }
                ]
            }
        }
    }
    inventory = tuple(
        InventoryCandidate(
            candidateId=f"I{index:04d}",
            category="changed_source_occurrence",
            sourceSurface=source_surface,
            lineIds=(line_id,),
            targetPaths=(path,),
            targetSemantics=[{"path": path, "target": target_surface}],
            disposition="model_residual",
            rationale="Exact changed party owner.",
        )
        for index, (source_surface, target_surface, line_id, field) in enumerate(
            (
                ("UNICHARM CORPORATION", "DELTA HORIZON TRADING COMPANY", "L00003", "name"),
                ("JAPAN", "Egypt", "L00004", "country"),
                ("SHIKOKUCHUO-CITY", "Shubra al Khaymah", "L00007", "city"),
            ),
            start=1,
        )
        for path in (f"documentPatch.parties.notifyParties[0].{field}",)
    )

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
        inventory=inventory,
    )

    finding = next(row for row in audit.findings if "Source address fragments" in row.problem)
    assert [row.lineId for row in finding.evidence] == [
        "L00003",
        "L00004",
        "L00005",
        "L00006",
        "L00007",
        "L00008",
    ]
    assert finding.targetPaths == ("documentPatch.parties.notifyParties[0].address",)
    assert not finding.repairs


def test_foreign_exporter_fields_must_match_the_resolved_target_party() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "SPORTKING INDIA LTD.\n"
        "\n"
        "FOREIGN EXPORTER REG NO.: C15130584\n"
        "COUNTRY OF FOREIGN EXPORTER: MAURITIUS\n"
        "FOREIGN EXPORTER NAME-SONVIGO INTERNATIONAL LIMITED.\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "Kaviron Metalworks Private Limited\n"
        "\n"
        "FOREIGN EXPORTER REG NO.: C93482761\n"
        "COUNTRY OF FOREIGN EXPORTER: SPAIN\n"
        "FOREIGN EXPORTER NAME-VELAMAR COMERCIAL EXTERIOR, S.A.\n"
    )
    source_label = {
        "documentPatch": {
            "parties": {"shipper": {"name": "SPORTKING INDIA LTD.", "country": "India"}}
        }
    }
    target_label = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "Kaviron Metalworks Private Limited",
                    "country": "India",
                }
            }
        }
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    finding = next(row for row in audit.findings if "Foreign-exporter identity" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00005", "L00006"]
    assert "explicit country codes ['ES'] do not equal target ['IN']" in finding.problem
    assert "explicit name does not render target" in finding.problem
    assert finding.targetPaths == (
        "documentPatch.parties.shipper.name",
        "documentPatch.parties.shipper.country",
    )


def test_foreign_exporter_address_country_must_match_its_explicit_country() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "VARDHMAN TEXTILES LTD\n"
        "\n"
        "FOREIGN EXPORTER\n"
        "COUNTRY: MAURITIUS\n"
        "CODE: MU\n"
        "SONVIGO INTERNATIONAL LTD\n"
        "PORT-LOUIS 11602, MAURITIUS\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "Yulin Everfield Machinery Components Co., Ltd.\n"
        "\n"
        "FOREIGN EXPORTER\n"
        "COUNTRY: CHINA\n"
        "CODE: CN\n"
        "HARBORLIGHT EXPORT SERVICES LTD\n"
        "GRAND HARBOUR 11004, MAURITIUS\n"
    )
    source_label = {
        "documentPatch": {
            "parties": {"shipper": {"name": "VARDHMAN TEXTILES LTD", "country": "India"}}
        }
    }
    target_label = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "Yulin Everfield Machinery Components Co., Ltd.",
                    "country": "China",
                }
            }
        }
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    finding = next(row for row in audit.findings if "Foreign-exporter identity" in row.problem)
    assert [row.lineId for row in finding.evidence] == [
        "L00002",
        "L00005",
        "L00006",
        "L00007",
        "L00008",
    ]
    assert "conflicting ISO country on lines (8,)" in finding.problem


def test_high_entropy_target_is_rejected_on_an_unowned_line() -> None:
    candidate = _CANDIDATE.replace("VAT NO: CD654321\n", "VAT NO: CD654321 ZX987654\n", 1)

    audit = _audit(candidate)

    finding = next(row for row in audit.findings if "inventory-owned" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00003"]
    assert finding.targetPaths == ("documentPatch.containers[0].sealNumber",)


def test_role_owned_vessel_cannot_escape_into_a_cargo_slot() -> None:
    source = '--- PAGE 1 ---\nVESSEL: OLD SHIP\n"STEEL COILS"\nSTEEL COILS\n'
    candidate = '--- PAGE 1 ---\nVESSEL: NEW SHIP\n"NEW SHIP"\nCOPPER CATHODES\n'
    source_label: dict[str, Any] = {
        "documentPatch": {
            "containers": [],
            "transport": {"vesselName": "OLD SHIP"},
            "cargoGroups": [{"description": "STEEL COILS"}],
        }
    }
    target_label: dict[str, Any] = {
        "documentPatch": {
            "containers": [],
            "transport": {"vesselName": "NEW SHIP"},
            "cargoGroups": [{"description": "COPPER CATHODES"}],
        }
    }
    inventory = (
        InventoryCandidate(
            candidateId="I0001",
            category="changed_source_occurrence",
            sourceSurface="OLD SHIP",
            lineIds=("L00002",),
            targetPaths=("documentPatch.transport.vesselName",),
            targetSemantics=[{"path": "documentPatch.transport.vesselName", "target": "NEW SHIP"}],
            disposition="model_residual",
            rationale="Exact source vessel slot.",
        ),
    )
    contract = _minimal_contract(source_label=source_label, target_label=target_label)
    envelope = compile_certification_invariant_envelope(
        document_id=_DOCUMENT_ID,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=(),
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )

    audit = audit_certification_invariants(
        envelope=envelope,
        document_id=_DOCUMENT_ID,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=(),
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )

    finding = next(row for row in audit.findings if "semantic roles collide" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00003", "L00004"]
    assert finding.dimension == "cargo_packages_and_dangerous_goods"


def test_repeated_source_field_requires_consistent_candidate_copies() -> None:
    source = "--- PAGE 1 ---\nVAT NO: AB123456\nVAT NO: AB123456\n"
    candidate = "--- PAGE 1 ---\nVAT NO: CD654321\nVAT NO: EF111222\n"
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
    )

    finding = next(row for row in audit.findings if "candidate copies disagree" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00003"]


def test_repeated_deterministic_edit_failures_form_one_provenance_finding() -> None:
    source = "--- PAGE 1 ---\nVAT NO: AB123456\nVAT NO: AB123456\n"
    candidate = source
    label: dict[str, Any] = {"documentPatch": {"containers": []}}
    edit = DeterministicInventoryEdit(
        sourceSurface="AB123456",
        targetSurface="CD654321",
        lineIds=("L00002", "L00003"),
        method="hmac_shape_preserving_auxiliary_v1",
    )

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
    )

    findings = tuple(
        row for row in audit.findings if "Host-generated auxiliary replacement" in row.problem
    )
    assert len(findings) == 1
    assert [row.lineId for row in findings[0].evidence] == ["L00002", "L00003"]
    assert [row.lineId for row in findings[0].repairs] == ["L00002", "L00003"]


def test_same_field_multi_value_edit_failures_form_one_provenance_finding() -> None:
    source = "--- PAGE 1 ---\nTHERMOGRAPHS: MHPYN060EV / MG3YN046XV\n"
    candidate = source
    label: dict[str, Any] = {"documentPatch": {"containers": []}}
    edits = (
        DeterministicInventoryEdit(
            sourceSurface="MHPYN060EV",
            targetSurface="OHSDW684FB",
            lineIds=("L00002",),
            method="hmac_shape_preserving_auxiliary_v1",
        ),
        DeterministicInventoryEdit(
            sourceSurface="MG3YN046XV",
            targetSurface="CN2XS595UF",
            lineIds=("L00002",),
            method="hmac_shape_preserving_auxiliary_v1",
        ),
    )

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=edits,
    )

    findings = tuple(
        row for row in audit.findings if "Host-generated auxiliary replacement" in row.problem
    )
    assert len(findings) == 1
    assert "OHSDW684FB" in findings[0].problem
    assert "CN2XS595UF" in findings[0].problem
    assert [row.lineId for row in findings[0].evidence] == ["L00002"]


def test_different_field_edit_failures_on_one_line_remain_independent() -> None:
    source = "--- PAGE 1 ---\nVAT NO: AB123456 REF NO: XY987654\n"
    candidate = source
    label: dict[str, Any] = {"documentPatch": {"containers": []}}
    edits = (
        DeterministicInventoryEdit(
            sourceSurface="AB123456",
            targetSurface="CD654321",
            lineIds=("L00002",),
            method="hmac_shape_preserving_auxiliary_v1",
        ),
        DeterministicInventoryEdit(
            sourceSurface="XY987654",
            targetSurface="QZ246810",
            lineIds=("L00002",),
            method="hmac_shape_preserving_auxiliary_v1",
        ),
    )

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=edits,
    )

    findings = tuple(
        row for row in audit.findings if "Host-generated auxiliary replacement" in row.problem
    )
    assert len(findings) == 2
    assert {row.problem for row in findings} == {
        "Host-generated auxiliary replacement 'CD654321' is missing from one or more exact "
        "source-owned lines or the source value remains.",
        "Host-generated auxiliary replacement 'QZ246810' is missing from one or more exact "
        "source-owned lines or the source value remains.",
    }


def test_source_auxiliary_substring_inside_target_owned_identifier_is_not_leakage() -> None:
    source = "--- PAGE 1 ---\nSCAC CODE: MSCU\nCONTAINER: MSCU7477141\n"
    candidate = "--- PAGE 1 ---\nSCAC CODE: VRBU\nCONTAINER: MSCU9504539\n"
    source_label = {"documentPatch": {"containers": [{"containerNumber": "MSCU7477141"}]}}
    target_label = {"documentPatch": {"containers": [{"containerNumber": "MSCU9504539"}]}}
    edits = (
        DeterministicInventoryEdit(
            sourceSurface="MSCU",
            targetSurface="VRBU",
            lineIds=("L00002",),
            method="hmac_shape_preserving_auxiliary_v1",
        ),
    )

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
        deterministic_edits=edits,
        anchored_replacements=(
            {
                "targetPaths": ["documentPatch.containers[0].containerNumber"],
                "sourceLineIds": ["L00003"],
                "sourceSurface": "MSCU7477141",
                "targetSurface": "MSCU9504539",
                "surfaceKind": "scalar",
            },
        ),
    )

    assert audit.passed


def test_target_owned_identifier_span_does_not_hide_a_second_source_auxiliary() -> None:
    source = "--- PAGE 1 ---\nSCAC CODE: MSCU\nCONTAINER: MSCU7477141\n"
    candidate = "--- PAGE 1 ---\nSCAC CODE: VRBU\nCONTAINER: MSCU9504539 SCAC CODE: MSCU\n"
    source_label = {"documentPatch": {"containers": [{"containerNumber": "MSCU7477141"}]}}
    target_label = {"documentPatch": {"containers": [{"containerNumber": "MSCU9504539"}]}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
        deterministic_edits=(
            DeterministicInventoryEdit(
                sourceSurface="MSCU",
                targetSurface="VRBU",
                lineIds=("L00002",),
                method="hmac_shape_preserving_auxiliary_v1",
            ),
        ),
        anchored_replacements=(
            {
                "targetPaths": ["documentPatch.containers[0].containerNumber"],
                "sourceLineIds": ["L00003"],
                "sourceSurface": "MSCU7477141",
                "targetSurface": "MSCU9504539",
                "surfaceKind": "scalar",
            },
        ),
    )

    finding = next(row for row in audit.findings if "'scac code' value 'MSCU'" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00003"]


def test_deterministic_edit_cannot_mutate_its_field_caption() -> None:
    source = "--- PAGE 1 ---\nFMC-OTI NO. 024004N\n"
    candidate = "--- PAGE 1 ---\nFMC-OTI EM. 538244T\n"
    edit = DeterministicInventoryEdit(
        sourceSurface="024004N",
        targetSurface="861935E",
        lineIds=("L00002",),
        method="hmac_shape_preserving_auxiliary_v1",
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
    )

    finding = next(
        row for row in audit.findings if "changed its source field caption" in row.problem
    )
    assert [row.lineId for row in finding.evidence] == ["L00002"]
    assert any("Host-generated auxiliary replacement" in row.problem for row in audit.findings)


def test_explicit_party_jurisdiction_requirement_authorizes_caption_generalization() -> None:
    source = "--- PAGE 1 ---\nEgyptian Importer Tax ID: 100693490\n"
    candidate = "--- PAGE 1 ---\nImporter Tax ID: 867614375\n"
    edit = DeterministicInventoryEdit(
        sourceSurface="100693490",
        targetSurface="867614375",
        lineIds=("L00002",),
        method="hmac_shape_preserving_auxiliary_v1",
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
        jurisdictional_requirements=(
            {
                "requirementId": "jurisdiction-egypt-importer-tax-id",
                "programId": "egypt_advance_cargo_information",
                "tradeDirection": "import",
                "programJurisdictionCountryCode": "EG",
                "targetRouteCountryCode": "EG",
                "sourceLineIds": ["L00002"],
                "sourceSurface": "EGYPTIAN IMPORTER TAX ID",
                "targetSurface": "IMPORTER TAX ID",
                "alternativeTargetSurfaces": ["CONSIGNEE IMPORTER TAX ID"],
                "sourceOccurrences": 1,
                "authority": "NAFEZA",
                "officialSourceUrl": "https://www.nafeza.gov.eg/en/pages/15",
                "rewriteBasis": "target_party_country_changed",
                "targetPartyRole": "consignee",
                "targetPartyCountryCode": "PK",
            },
        ),
    )

    assert audit.passed


def test_explicit_caption_rewrite_does_not_authorize_a_new_leading_qualifier() -> None:
    source = "--- PAGE 1 ---\nEgyptian Importer Tax ID: 100693490\n"
    candidate = "--- PAGE 1 ---\nForeign Importer Tax ID: 867614375\n"
    edit = DeterministicInventoryEdit(
        sourceSurface="100693490",
        targetSurface="867614375",
        lineIds=("L00002",),
        method="hmac_shape_preserving_auxiliary_v1",
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
        jurisdictional_requirements=(
            {
                "requirementId": "jurisdiction-egypt-importer-tax-id",
                "programId": "egypt_advance_cargo_information",
                "tradeDirection": "import",
                "programJurisdictionCountryCode": "EG",
                "targetRouteCountryCode": "EG",
                "sourceLineIds": ["L00002"],
                "sourceSurface": "EGYPTIAN IMPORTER TAX ID",
                "targetSurface": "IMPORTER TAX ID",
                "alternativeTargetSurfaces": ["CONSIGNEE IMPORTER TAX ID"],
                "sourceOccurrences": 1,
                "authority": "NAFEZA",
                "officialSourceUrl": "https://www.nafeza.gov.eg/en/pages/15",
                "rewriteBasis": "target_party_country_changed",
                "targetPartyRole": "consignee",
                "targetPartyCountryCode": "PK",
            },
        ),
    )

    finding = next(
        row for row in audit.findings if row.findingKind == "route_or_jurisdiction_mismatch"
    )
    assert [row.lineId for row in finding.evidence] == ["L00002"]
    assert not any("changed its source field caption" in row.problem for row in audit.findings)


def test_explicit_role_prefixed_jurisdiction_caption_is_authorized() -> None:
    source = "--- PAGE 1 ---\nEGYPTIAN IMPORTER TAX ID: 212563378\n"
    candidate = "--- PAGE 1 ---\nCONSIGNEE IMPORTER TAX ID: 144421400\n"
    edit = DeterministicInventoryEdit(
        sourceSurface="212563378",
        targetSurface="144421400",
        lineIds=("L00002",),
        method="hmac_shape_preserving_auxiliary_v1",
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
        jurisdictional_requirements=(
            {
                "requirementId": "jurisdiction-egypt-importer-tax-id",
                "programId": "egypt_advance_cargo_information",
                "tradeDirection": "import",
                "programJurisdictionCountryCode": "EG",
                "targetRouteCountryCode": "CN",
                "sourceLineIds": ["L00002"],
                "sourceSurface": "EGYPTIAN IMPORTER TAX ID",
                "targetSurface": "IMPORTER TAX ID",
                "alternativeTargetSurfaces": ["CONSIGNEE IMPORTER TAX ID"],
                "sourceOccurrences": 1,
                "authority": "NAFEZA",
                "officialSourceUrl": "https://www.nafeza.gov.eg/en/pages/15",
                "targetPartyRole": "consignee",
            },
        ),
    )

    assert audit.passed


@pytest.mark.parametrize("caption", ("Reg. No.", "COMPANY REG."))
def test_party_registry_authorizes_only_its_explicit_neutral_captions(
    caption: str,
) -> None:
    source = "--- PAGE 1 ---\n562 024 422 R.C.S. Marseille\n"
    candidate = f"--- PAGE 1 ---\n687 764 867 {caption} Marseille\n"
    edit = DeterministicInventoryEdit(
        sourceSurface="562 024 422",
        targetSurface="687 764 867",
        lineIds=("L00002",),
        method="hmac_shape_preserving_auxiliary_v1",
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
        jurisdictional_requirements=(
            {
                "requirementId": "jurisdiction-france-rcs",
                "programId": "france_register_of_commerce_and_companies",
                "tradeDirection": "party",
                "programJurisdictionCountryCode": "FR",
                "targetRouteCountryCode": None,
                "sourceLineIds": ["L00002"],
                "sourceSurface": "R.C.S.",
                "targetSurface": "REG. NO.",
                "alternativeTargetSurfaces": ["COMPANY REG."],
                "sourceOccurrences": 1,
                "authority": "Direction de l'information légale et administrative",
                "officialSourceUrl": (
                    "https://entreprendre.service-public.gouv.fr/vosdroits/F31190"
                ),
                "rewriteBasis": "target_party_registry_country_changed",
                "targetPartyRole": "carrier",
                "sourcePartyCountryCode": "FR",
                "targetPartyCountryCode": "CH",
            },
        ),
    )

    assert audit.passed


def test_party_registry_caption_does_not_hide_a_retained_source_identifier() -> None:
    source = "--- PAGE 1 ---\n562 024 422 R.C.S. Marseille\n"
    candidate = "--- PAGE 1 ---\n562 024 422 Reg. No. Marseille\n"
    edit = DeterministicInventoryEdit(
        sourceSurface="562 024 422",
        targetSurface="687 764 867",
        lineIds=("L00002",),
        method="hmac_shape_preserving_auxiliary_v1",
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
        jurisdictional_requirements=(
            {
                "requirementId": "jurisdiction-france-rcs",
                "programId": "france_register_of_commerce_and_companies",
                "tradeDirection": "party",
                "programJurisdictionCountryCode": "FR",
                "targetRouteCountryCode": None,
                "sourceLineIds": ["L00002"],
                "sourceSurface": "R.C.S.",
                "targetSurface": "REG. NO.",
                "alternativeTargetSurfaces": ["COMPANY REG."],
                "sourceOccurrences": 1,
                "authority": "Direction de l'information légale et administrative",
                "officialSourceUrl": (
                    "https://entreprendre.service-public.gouv.fr/vosdroits/F31190"
                ),
                "rewriteBasis": "target_party_registry_country_changed",
                "targetPartyRole": "carrier",
                "sourcePartyCountryCode": "FR",
                "targetPartyCountryCode": "CH",
            },
        ),
    )

    assert not audit.passed
    assert len(audit.findings) == 1
    assert "Host-generated auxiliary replacement" in audit.findings[0].problem


def test_standalone_jurisdiction_heading_authorizes_its_bound_value_line_caption() -> None:
    source = "--- PAGE 1 ---\nACID NUMBER\n1004977722024020145\n"
    candidate = "--- PAGE 1 ---\nCUSTOMS REFERENCE\n1081049893218861713\n"
    edit = DeterministicInventoryEdit(
        sourceSurface="1004977722024020145",
        targetSurface="1081049893218861713",
        lineIds=("L00003",),
        method="hmac_shape_preserving_auxiliary_v1",
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
        jurisdictional_requirements=(
            {
                "requirementId": "jurisdiction-egypt-acid-number",
                "programId": "egypt_advance_cargo_information",
                "tradeDirection": "import",
                "programJurisdictionCountryCode": "EG",
                "targetRouteCountryCode": "JP",
                "sourceLineIds": ["L00002"],
                "sourceSurface": "ACID NUMBER",
                "targetSurface": "CUSTOMS REFERENCE",
                "sourceOccurrences": 1,
                "authority": "NAFEZA",
                "officialSourceUrl": "https://www.nafeza.gov.eg/en/pages/15",
            },
        ),
    )

    assert audit.passed


def test_jurisdiction_finding_cites_rendered_target_party_context() -> None:
    source = (
        "--- PAGE 1 ---\nDONATELLA ECUADOR DONATECUA S.A.\nGUAYAQUIL-ECUADOR\nRUC: 1368076554717\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "Merriton Coastal Exchange LLC\n"
        "BRENTWOOD-UNITED STATES\n"
        "RUC: 1368076554717\n"
    )
    source_label: dict[str, Any] = {"documentPatch": {"containers": []}}
    target_label: dict[str, Any] = {"documentPatch": {"containers": []}}
    inventory = (
        InventoryCandidate(
            candidateId="I0001",
            category="changed_source_occurrence",
            sourceSurface="DONATELLA ECUADOR DONATECUA S.A.",
            lineIds=("L00002",),
            targetPaths=("documentPatch.parties.shipper.name",),
            targetSemantics=[
                {
                    "path": "documentPatch.parties.shipper.name",
                    "target": "Merriton Coastal Exchange LLC",
                }
            ],
            disposition="model_residual",
            rationale="Exact exporter name slot.",
        ),
        InventoryCandidate(
            candidateId="I0002",
            category="changed_source_occurrence",
            sourceSurface="GUAYAQUIL",
            lineIds=("L00003",),
            targetPaths=("documentPatch.parties.shipper.city",),
            targetSemantics=[
                {
                    "path": "documentPatch.parties.shipper.city",
                    "target": "Brentwood",
                }
            ],
            disposition="model_residual",
            rationale="Exact exporter city slot.",
        ),
    )

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
        inventory=inventory,
        jurisdictional_requirements=(
            {
                "requirementId": "jurisdiction-ecuador-ruc-test",
                "programId": "ecuador_unique_taxpayer_registry",
                "tradeDirection": "export",
                "programJurisdictionCountryCode": "EC",
                "targetRouteCountryCode": "US",
                "sourceLineIds": ["L00004"],
                "sourceSurface": "RUC:",
                "targetSurface": "TAX ID",
                "sourceOccurrences": 1,
                "authority": "Servicio de Rentas Internas",
                "officialSourceUrl": "https://example.test/ruc",
            },
        ),
    )

    finding = next(row for row in audit.findings if "ecuador_unique" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00003", "L00004"]
    assert finding.targetPaths == (
        "documentPatch.parties.shipper.city",
        "documentPatch.parties.shipper.name",
    )


def test_repeated_identifier_cannot_change_field_kind_between_copies() -> None:
    source = "--- PAGE 1 ---\nACID: 1111111111111\nACID: 1111111111111\n"
    candidate = "--- PAGE 1 ---\nACID: 2222222222222\nCUSTOMS REFERENCE: 2222222222222\n"
    edit = DeterministicInventoryEdit(
        sourceSurface="1111111111111",
        targetSurface="2222222222222",
        lineIds=("L00002", "L00003"),
        method="hmac_shape_preserving_auxiliary_v1",
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
        deterministic_edits=(edit,),
    )

    finding = next(row for row in audit.findings if "inconsistent field kinds" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00003"]


def test_envelope_cannot_be_replayed_with_different_candidate_bytes() -> None:
    contract = _contract()
    envelope = compile_certification_invariant_envelope(
        document_id=_DOCUMENT_ID,
        source=_SOURCE,
        candidate=_CANDIDATE,
        contract=contract,
        inventory=_inventory(),
        deterministic_edits=_edits(),
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )

    with pytest.raises(ValueError, match="envelope differs"):
        audit_certification_invariants(
            envelope=envelope,
            document_id=_DOCUMENT_ID,
            source=_SOURCE,
            candidate=_CANDIDATE.replace("NEW OCEAN LINE", "OTHER OCEAN LINE"),
            contract=contract,
            inventory=_inventory(),
            deterministic_edits=_edits(),
            references=_REFERENCES,
            capacity_limits=_CAPACITY_LIMITS,
        )


def test_source_proven_container_component_total_rejects_factor_scale_error() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Container Packages Weight Measure\n"
        "ABCU1234567 10 BOXES 12.500 2.000 KGM MTQ\n"
        "\n"
        "DEFU7654321 10 BOXES 7.500 3.000 KGM MTQ\n"
        "================ 20.000 5.000\n"
        "KGM MTQ\n"
        "TOTAL ADMT: 20.000\n"
    )
    candidate = (
        source.replace("12.500", "0.012")
        .replace("7.500", "0.008")
        .replace("TOTAL ADMT: 20.000", "TOTAL ADMT: 0.020")
    )
    label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "ABCU1234567"},
                {"containerNumber": "DEFU7654321"},
            ]
        }
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
    )

    findings = [row for row in audit.findings if "container_component_weight" in row.problem]
    assert len(findings) == 1
    assert [row.lineId for row in findings[0].evidence] == [
        "L00003",
        "L00005",
        "L00006",
        "L00007",
        "L00008",
    ]


def test_target_absent_source_volume_and_tare_identities_must_retire() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Measurement\n"
        "44,020\n"
        "VOL 15,120\n"
        "TOTAL 2 9050,000 44,020\n"
        "1 CONT TOT. TARE : 3,910\n"
        "CONT TARE 3910\n"
    )
    candidate = source.replace("9050,000", "9603,300")
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
    )

    measurement_findings = [
        row for row in audit.findings if "source-owned physical measurement slots" in row.problem
    ]
    assert {
        tuple(evidence.lineId for evidence in row.evidence) for row in measurement_findings
    } == {
        ("L00004",),
        ("L00003", "L00005"),
        ("L00006", "L00007"),
    }
    assert {row.findingKind for row in measurement_findings} == {
        "source_only_private_or_auxiliary_fact"
    }


def test_source_volume_retirement_does_not_activate_when_target_owns_volume() -> None:
    source = "--- PAGE 1 ---\nMeasurement\n44,020\n"
    label: dict[str, Any] = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "volume": {"unit": "cubic_metre", "value": 44.02},
                }
            ],
        }
    }

    audit = _audit_minimal(
        source=source,
        candidate=source,
        source_label=label,
        target_label=label,
    )

    assert not [
        row for row in audit.findings if "source-owned physical measurement slots" in row.problem
    ]


def test_missing_local_equipment_assignment_cites_aggregate_equipment_prefill() -> None:
    source = "--- PAGE 1 ---\nABCU1234567\nDEFU7654321\nSAY: TWO (20DRX2) CONTAINERS ONLY.\n"
    candidate = (
        "--- PAGE 1 ---\n"
        "ABCU7654321\n"
        "DEFU1234567\n"
        "SAY: TWO (40' HIGH CUBE GENERAL PURPOSEX1 + "
        "20' STANDARD HEIGHT GENERAL PURPOSEX1) CONTAINERS ONLY.\n"
    )
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "ABCU1234567"},
                {"containerNumber": "DEFU7654321"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "ABCU7654321",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "DEFU1234567",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }
    aggregate = candidate.splitlines()[3]

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
        deterministic_prefills=(
            {
                "lineId": "L00004",
                "targetPaths": [
                    "documentPatch.containers[0].printedEquipmentSurface",
                    "documentPatch.containers[1].printedEquipmentSurface",
                ],
                "sourceSurface": "SAY: TWO (20DRX2) CONTAINERS ONLY.",
                "targetSurface": aggregate,
                "beforeLine": "SAY: TWO (20DRX2) CONTAINERS ONLY.",
                "afterLine": aggregate,
            },
        ),
    )

    finding = next(row for row in audit.findings if "not locally recoverable" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00003", "L00004"]


def test_source_proven_vertical_tuple_rejects_only_broken_derived_columns() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "GROSS WEIGHT TARE MEASUREMENT CBM\n"
        "Cargo\n"
        "KGS\n"
        "10.000\n"
        "KGS\n"
        "2.000\n"
        "CBM\n"
        "3.000\n"
        "KGS\n"
        "2.500\n"
        "CBM\n"
        "4.000\n"
        "Weight in Kgs Total: 2 CONTAINER(S)\n"
        "Continued From Previous Sheet Sheet 2 of 3 20.000 4.500 7.000\n"
    )
    candidate = source.replace("20.000 4.500 7.000", "20.000 4.000 8.000")
    label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "ABCU1234567"},
                {"containerNumber": "DEFU7654321"},
            ]
        }
    }

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
    )

    relation_findings = [row for row in audit.findings if "aggregate-shipment" in row.problem]
    assert [row.problem.split("'")[1] for row in relation_findings] == [
        "aggregate-shipment_tare_weight-L00015",
        "aggregate-shipment_volume-L00015",
    ]


def test_source_proven_serial_ranges_must_still_total_target_packages() -> None:
    source_label = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "marksAndNumbers": ["A100 - A149", "B200 - B249"],
                }
            ],
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 100}],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "marksAndNumbers": ["C300 - C348", "D400 - D448"],
                }
            ],
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 100}],
        }
    }
    source = "--- PAGE 1 ---\nA100 - A149\nB200 - B249\n100 CARTONS\n"
    candidate = "--- PAGE 1 ---\nC300 - C348\nD400 - D448\n100 CARTONS\n"

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
    )

    finding = next(row for row in audit.findings if "inclusive serial-range" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00003", "L00004"]
    assert "target ranges total 98" in finding.problem


def test_shipment_equipment_is_forbidden_when_target_has_no_containers() -> None:
    label: dict[str, Any] = {"documentPatch": {"containers": []}}
    text = "--- PAGE 1 ---\n1x40HC CONTAINER:\n1 FCL :\n"

    audit = _audit_minimal(
        source=text,
        candidate=text,
        source_label=label,
        target_label=label,
    )

    findings = [row for row in audit.findings if "no container inventory" in row.problem]
    assert len(findings) == 1
    assert [row.lineId for row in findings[0].evidence] == ["L00002", "L00003"]


def test_counted_equipment_cannot_exceed_exact_target_family_inventory() -> None:
    source_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "ABCU1234567",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "DEFU7654321",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "ABCU1234567",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "DEFU7654321",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }
    text = "--- PAGE 1 ---\n2 X 40HC\n"

    audit = _audit_minimal(
        source=text,
        candidate=text,
        source_label=source_label,
        target_label=target_label,
    )

    finding = next(row for row in audit.findings if "target inventory contains only" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002"]


def test_source_proven_equipment_payload_cannot_exceed_pinned_capacity_policy() -> None:
    label: dict[str, Any] = {"documentPatch": {"containers": []}}
    source = "--- PAGE 1 ---\n1x40HC CONTAINER:\nTOTAL\n26027.000KGS\n"
    candidate = source.replace("26027.000KGS", "42810.000KGS")

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
    )

    finding = next(
        row for row in audit.findings if "receipt-bound equipment ceiling" in row.problem
    )
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00004"]
    assert "42810.000" in finding.problem
    assert "30124.50" in finding.problem


def test_heterogeneous_container_requires_one_local_nonduplicated_family() -> None:
    label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "ABCU1234567",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "DEFU7654321",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }
    source = (
        "--- PAGE 1 ---\n"
        "ABCU1234567 40' HIGH CUBE GENERAL PURPOSE\n"
        "\n"
        "DEFU7654321 20' STANDARD HEIGHT GENERAL PURPOSE\n"
        "NOTE\n"
    )
    candidate = source.replace("NOTE", "20' STANDARD HEIGHT GENERAL PURPOSE")

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
    )

    finding = next(row for row in audit.findings if "requires exactly one local" in row.problem)
    assert "DEFU7654321" in finding.problem
    assert [row.lineId for row in finding.evidence] == ["L00004", "L00005"]


def test_negotiable_bill_requires_positive_to_order_realization() -> None:
    label: dict[str, Any] = {"documentPatch": {"containers": [], "negotiability": "negotiable"}}
    source = "--- PAGE 1 ---\nCONSIGNEE (NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER)\nTO ORDER\n"
    candidate = source.removesuffix("TO ORDER\n") + "NAMED TRADING COMPANY\n"

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
    )

    finding = next(row for row in audit.findings if "positive TO ORDER" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00003"]


def test_non_negotiable_bill_cannot_populate_to_order_block() -> None:
    label: dict[str, Any] = {"documentPatch": {"containers": [], "negotiability": "non_negotiable"}}
    text = (
        "--- PAGE 1 ---\nConsigned to order of\nNAMED TRADING COMPANY\nSEAWAYBILL\nnon negotiable\n"
    )

    audit = _audit_minimal(
        source=text,
        candidate=text,
        source_label=label,
        target_label=label,
    )

    finding = next(row for row in audit.findings if "Non-negotiable sea-waybill" in row.problem)
    assert [row.lineId for row in finding.evidence] == [
        "L00002",
        "L00003",
        "L00004",
        "L00005",
    ]


def test_original_bill_field_and_signed_clause_must_agree() -> None:
    label: dict[str, Any] = {"documentPatch": {"containers": [], "negotiability": "non_negotiable"}}
    text = (
        "--- PAGE 1 ---\n"
        "IN WITNESS WHEREOF THE AGENT HAS SIGNED THREE(3) BILLS OF LADING.\n"
        "NO OF ORIGINAL B/L\n"
        "ZERO(0)\n"
    )

    audit = _audit_minimal(
        source=text,
        candidate=text,
        source_label=label,
        target_label=label,
    )

    finding = next(row for row in audit.findings if "original-bill field counts" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00002", "L00004"]
    assert not finding.repairs


def _office_inventory() -> tuple[InventoryCandidate, ...]:
    return (
        InventoryCandidate(
            candidateId="I0001",
            category="changed_source_auxiliary_copy",
            sourceSurface="Damietta",
            lineIds=("L00004", "L00005", "L00008"),
            targetPaths=("documentPatch.route.placeOfDelivery.name",),
            targetSemantics={
                "policy": "synthesize_distinct_context_compatible_auxiliary_value",
                "mustDifferFromSource": True,
                "mustNotDuplicateAnyLabeledTarget": True,
            },
            disposition="model_residual",
            rationale="Repeated source destination-office locality.",
        ),
        InventoryCandidate(
            candidateId="I0002",
            category="changed_source_auxiliary_copy",
            sourceSurface="Egypt",
            lineIds=("L00005",),
            targetPaths=("documentPatch.parties.consignee.country",),
            targetSemantics={
                "policy": "synthesize_distinct_context_compatible_auxiliary_value",
                "mustDifferFromSource": True,
                "mustNotDuplicateAnyLabeledTarget": True,
            },
            disposition="model_residual",
            rationale="Source destination-office country.",
        ),
        InventoryCandidate(
            candidateId="I0003",
            category="source_only_contact_identity",
            sourceSurface="+20572292230",
            lineIds=("L00006",),
            targetPaths=(),
            targetSemantics={
                "policy": "synthesize_realistic_fictional_contact_value",
                "preservePrintedDelimiters": True,
                "preserveRepeatedIdentity": True,
                "semanticField": "tel",
            },
            disposition="model_residual",
            rationale="Source-only destination-office telephone.",
        ),
        InventoryCandidate(
            candidateId="I0004",
            category="source_only_contact_identity",
            sourceSurface="+20572292206",
            lineIds=("L00007",),
            targetPaths=(),
            targetSemantics={
                "policy": "synthesize_realistic_fictional_contact_value",
                "preservePrintedDelimiters": True,
                "preserveRepeatedIdentity": True,
                "semanticField": "fax",
            },
            disposition="model_residual",
            rationale="Source-only destination-office fax.",
        ),
    )


def _audit_office(candidate: str) -> CertificationInvariantAudit:
    source = (
        "--- PAGE 1 ---\n"
        "DESTINATION OFFICE\n"
        "EGYPTIAN GLOBAL LOGISTICS\n"
        "Damietta Port, West of Damietta Port Area\n"
        "Damietta; Egypt\n"
        "Tel : +20572292230\n"
        "Fax : +20572292206\n"
        "Email damietta@example.test\n"
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}
    contract = _minimal_contract(source_label=label, target_label=label)
    inventory = _office_inventory()
    envelope = compile_certification_invariant_envelope(
        document_id=_DOCUMENT_ID,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=(),
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )
    return audit_certification_invariants(
        envelope=envelope,
        document_id=_DOCUMENT_ID,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=(),
        references=_REFERENCES,
        capacity_limits=_CAPACITY_LIMITS,
    )


def test_source_proven_office_requires_one_locality_and_matching_calling_code() -> None:
    candidate = (
        "--- PAGE 1 ---\n"
        "DESTINATION OFFICE\n"
        "YSTAD SHIPPING AND LOGISTICS\n"
        "Kalmar Port, East of Kalmar Port Area\n"
        "Ystad; Sweden\n"
        "Tel : +35819412288\n"
        "Fax : +35819412264\n"
        "Email kalmar@example.test\n"
    )

    audit = _audit_office(candidate)

    locality = next(row for row in audit.findings if "inconsistent GeoNames" in row.problem)
    phone = next(row for row in audit.findings if "calling-code regions" in row.problem)
    assert [row.lineId for row in locality.evidence] == [
        "L00003",
        "L00004",
        "L00005",
        "L00008",
    ]
    assert [row.lineId for row in phone.evidence] == [
        "L00003",
        "L00005",
        "L00006",
        "L00007",
    ]
    assert "['AX', 'FI']" in phone.problem
    assert "'SE'" in phone.problem


def test_source_proven_office_accepts_consistent_locality_and_country_calling_code() -> None:
    candidate = (
        "--- PAGE 1 ---\n"
        "DESTINATION OFFICE\n"
        "KALMAR SHIPPING AND LOGISTICS\n"
        "Kalmar Port, East of Kalmar Port Area\n"
        "Kalmar; Sweden\n"
        "Tel : +4640123456\n"
        "Fax : +4640654321\n"
        "Email kalmar@example.test\n"
    )

    audit = _audit_office(candidate)

    assert not [
        row
        for row in audit.findings
        if row.invariantId.startswith("D") and "office" in row.problem.lower()
    ]


def test_source_proven_freight_schedule_must_retire_private_money_surfaces() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "FREIGHT & CHARGES\n"
        "Ocean Freight\n"
        "1 USD 100.00\n"
        "USD 100.00\n"
        "Security Fee\n"
        "1 USD 20.00\n"
        "USD 20.00\n"
        "Declared Value: TOTAL FREIGHT & CHARGES USD 120.00\n"
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=source,
        source_label=label,
        target_label=label,
    )

    finding = next(row for row in audit.findings if "exact source monetary" in row.problem)
    assert [row.lineId for row in finding.evidence] == [
        "L00004",
        "L00005",
        "L00007",
        "L00008",
        "L00009",
    ]


def test_rewritten_freight_schedule_must_remain_additive() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "FREIGHT & CHARGES\n"
        "Ocean Freight\n"
        "1 USD 100.00\n"
        "USD 100.00\n"
        "Security Fee\n"
        "1 USD 20.00\n"
        "USD 20.00\n"
        "Declared Value: TOTAL FREIGHT & CHARGES USD 120.00\n"
    )
    candidate = (
        "--- PAGE 1 ---\n"
        "FREIGHT & CHARGES\n"
        "Ocean Freight\n"
        "1 USD 110.00\n"
        "USD 110.00\n"
        "Security Fee\n"
        "1 USD 15.00\n"
        "USD 15.00\n"
        "Declared Value: TOTAL FREIGHT & CHARGES USD 130.00\n"
    )
    label: dict[str, Any] = {"documentPatch": {"containers": []}}

    audit = _audit_minimal(
        source=source,
        candidate=candidate,
        source_label=label,
        target_label=label,
    )

    finding = next(row for row in audit.findings if "no longer internally additive" in row.problem)
    assert [row.lineId for row in finding.evidence] == ["L00005", "L00008", "L00009"]
