from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from document_ocr.synthesis.anchors import build_document_anchors
from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.domain import RelationalTables, rows_for_document
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_container_number,
    generate_from_surface_pattern,
    generate_quantity,
    generate_seal_identifier,
    generate_unique_container_numbers,
    generate_unique_seal_identifiers,
    iso6346_check_digit,
    largest_remainder_allocation,
    reconcile_allocation_group,
    shift_document_dates,
    surface_pattern,
    validate_container_number,
    validate_mass_order,
)
from document_ocr.synthesis.preparation import (
    _normal_path_aliases,
    _resolve_recorded_artifact_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-package-categories-v2"
)


def _records() -> list[dict[str, object]]:
    return [json.loads(line) for line in (DATASET / "records.jsonl").read_text().splitlines()]


def test_recorded_artifact_path_relocates_to_explicit_project_root(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/labels/doc.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")
    recorded = "/mnt/d/Projects/DocumentParsing/artifacts/labels/doc.json"
    assert _resolve_recorded_artifact_file(tmp_path, recorded, "annotation") == artifact
    with pytest.raises(ValueError, match="one project artifacts component"):
        _resolve_recorded_artifact_file(tmp_path, "/outside/doc.json", "annotation")


def test_normal_path_aliases_requires_one_explicit_annotation_schema() -> None:
    package_metadata: dict[str, object] = {
        "groupDiagnoses": [],
        "sourceNormalTarget": {"documentPatch": {}},
    }
    with pytest.raises(ValueError, match="exactly one"):
        _normal_path_aliases(
            annotation={"normalLabel": {}, "label": {}, "evidence": []},
            package_metadata=package_metadata,
        )
    with pytest.raises(ValueError, match="exactly one"):
        _normal_path_aliases(
            annotation={"evidence": []},
            package_metadata=package_metadata,
        )
    assert (
        _normal_path_aliases(
            annotation={"normalLabel": {"documentPatch": {}}, "evidence": []},
            package_metadata=package_metadata,
        )
        == {}
    )


def test_bill_of_lading_domain_projection_inverts_all_1157_targets() -> None:
    tables = RelationalTables({name: [] for name in ADAPTER.table_order})
    targets: dict[str, dict[str, object]] = {}
    for row_index, row in enumerate(_records()):
        document_id = str(row["documentId"])
        target = row["target"]
        assert isinstance(target, dict)
        targets[document_id] = target
        tables.extend(
            ADAPTER.project(
                document_id=document_id,
                source_row_index=row_index,
                target=target,
            )
        )
    assert len(tables.rows["documents"]) == 1157
    assert len(tables.rows["allocations"]) == 1853
    for document_id, target in targets.items():
        assert (
            ADAPTER.reconstruct(
                document_id=document_id,
                tables=rows_for_document(tables.rows, document_id),
            )
            == target
        )


def test_bill_of_lading_sdv_metadata_covers_every_domain_table() -> None:
    metadata = ADAPTER.sdv_metadata()
    assert tuple(metadata["tables"]) == ADAPTER.table_order
    for table_name in ADAPTER.table_order:
        table = metadata["tables"][table_name]
        primary_key = table["primary_key"]
        assert primary_key in table["columns"]
        assert table["columns"][primary_key]["sdtype"] == "id"


@pytest.mark.parametrize(
    ("body", "digit"),
    (("TGHU123456", "7"), ("MSCU663987", "0"), ("ABCU000000", "1")),
)
def test_iso6346_vectors_and_generation(body: str, digit: str) -> None:
    assert iso6346_check_digit(body) == digit
    assert validate_container_number(body + digit)
    assert not validate_container_number(body + str((int(digit) + 1) % 10))
    stream = DeterministicStream(7, "test", body)
    generated = generate_container_number(owner_and_category=body[:4], stream=stream)
    assert generated == generate_container_number(owner_and_category=body[:4], stream=stream)
    assert validate_container_number(generated)


def test_batch_identifier_allocation_is_unique_and_input_order_independent() -> None:
    owners = {f"doc-{index:05d}": "MSCU" for index in range(1_000)}
    reversed_owners = dict(reversed(tuple(owners.items())))
    stream = DeterministicStream(7, "containers", "pilot")
    first = generate_unique_container_numbers(owner_by_identity=owners, stream=stream)
    second = generate_unique_container_numbers(owner_by_identity=reversed_owners, stream=stream)
    assert first == second
    assert len(set(first.values())) == len(first)
    assert all(validate_container_number(value) for value in first.values())
    replacement = generate_unique_container_numbers(
        owner_by_identity=owners,
        stream=stream,
        excluded=set(first.values()),
    )
    assert set(replacement.values()).isdisjoint(first.values())

    sources = {f"doc-{index:05d}": "ML-TH0184180" for index in range(1_000)}
    seals = generate_unique_seal_identifiers(
        source_by_identity=sources,
        stream=DeterministicStream(7, "seals", "pilot"),
    )
    assert len(set(seals.values())) == len(seals)
    assert "ML-TH0184180" not in seals.values()


def test_surface_generator_preserves_shape_and_avoids_source() -> None:
    source = "ML-TH0184180"
    pattern = surface_pattern(source)
    generated = generate_from_surface_pattern(
        pattern=pattern,
        stream=DeterministicStream(11, "seal", "doc:path"),
        excluded=frozenset({source}),
    )
    assert generated != source
    assert surface_pattern(generated) == pattern
    assert generate_seal_identifier(
        source=source,
        stream=DeterministicStream(11, "seal", "doc:path"),
    ) == generate_seal_identifier(
        source=source,
        stream=DeterministicStream(11, "seal", "doc:path"),
    )
    with pytest.raises(ValueError, match="whitespace-free"):
        generate_seal_identifier(
            source="IN028435 TO IN028444",
            stream=DeterministicStream(11, "seal", "doc:path"),
        )


def test_date_quantity_mass_and_allocation_invariants() -> None:
    shifted = shift_document_dates(
        issue_date=date(2024, 4, 8),
        shipped_on_board_date=date(2024, 4, 6),
        minimum=date(2020, 1, 1),
        maximum=date(2030, 12, 31),
        stream=DeterministicStream(5, "date", "doc"),
    )
    assert shifted[0] is not None and shifted[1] is not None
    assert (shifted[0] - shifted[1]).days == 2
    assert (
        generate_quantity(
            minimum=1,
            maximum=3,
            stream=DeterministicStream(1, "quantity", "p1"),
            excluded=frozenset({1, 2}),
        )
        == 3
    )
    assert largest_remainder_allocation(11, (4, 6)) == (4, 7)
    packages = ({"packageId": "p1", "quantity": 11},)
    allocation = {
        "groupId": "g1",
        "coverage": "single_package_level",
        "packageIds": ["p1"],
        "allocations": [
            {"containerNumber": "TGHU1234567", "packageQuantity": 4},
            {"containerNumber": "MSCU6639870", "packageQuantity": 6},
        ],
    }
    reconciled = reconcile_allocation_group(packages=packages, allocation_group=allocation)
    assert [row["packageQuantity"] for row in reconciled["allocations"]] == [4, 7]
    with pytest.raises(ValueError, match="erase a positive container membership"):
        reconcile_allocation_group(
            packages=({"packageId": "p1", "quantity": 1},),
            allocation_group={
                **allocation,
                "allocations": [
                    {"containerNumber": "TGHU1234567", "packageQuantity": 1},
                    {"containerNumber": "MSCU6639870", "packageQuantity": 1},
                ],
            },
        )
    validate_mass_order(
        gross_weight={"value": 134865, "unit": "kilogram"},
        net_weight={"value": 134.865, "unit": "metric_tonne"},
    )
    with pytest.raises(ValueError, match="below net"):
        validate_mass_order(
            gross_weight={"value": 999, "unit": "kilogram"},
            net_weight={"value": 1, "unit": "metric_tonne"},
        )


def test_audited_evidence_maps_to_task_facing_category_target() -> None:
    row = _records()[0]
    document_id = str(row["documentId"])
    annotation_path = (
        PROJECT_ROOT
        / "artifacts/kie-labels/mpci-bl-semantic-v2-pilot150-r1/validated"
        / f"{document_id}.json"
    )
    annotation = json.loads(annotation_path.read_text())
    target = row["target"]
    normal = row["normalTarget"]
    assert isinstance(target, dict) and isinstance(normal, dict)
    anchors, summary = build_document_anchors(
        document_id=document_id,
        joined_raw_text=str(row["joinedRawText"]),
        target=target,
        normal_target=normal,
        annotation=annotation,
    )
    assert summary["evidence_coverage"] == 1.0
    category = [
        anchor
        for anchor in anchors
        if anchor["relation_target_path"] == "documentPatch.cargoPackages[0].typeCategory"
    ]
    assert category
    assert category[0]["normal_target_path"].endswith("goodsItems[0].packages[0].type")
    assert category[0]["raw_value"] == "FIBRE DRUMS"
