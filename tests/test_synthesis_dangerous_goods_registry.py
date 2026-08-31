from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.dangerous_goods_registry import (
    CompiledDangerousGoodsRegistry,
    DangerousGoodsGenerationDisabled,
    DangerousGoodsLicenseReceipt,
    DangerousGoodsProvider,
    DangerousGoodsRegistry,
    DangerousGoodsRegistryError,
    DangerousGoodsRegistryReceipt,
    DangerousGoodsRegulatoryTuple,
    DangerousGoodsShipmentSelection,
    DangerousGoodsSourceReceipt,
    ShipmentFlashpoint,
    build_dangerous_goods_regulatory_tuple,
    compile_dangerous_goods_registry,
    load_pinned_dangerous_goods_registry,
)
from document_ocr.synthesis.run_safety import StagedRunError


def _license(
    *,
    automated: bool = True,
    compiled: bool = True,
    synthetic: bool = True,
    basis: str = "public_license",
    evidence: bool = False,
) -> DangerousGoodsLicenseReceipt:
    values: dict[str, object] = {
        "basis": basis,
        "license_name": "Fixture machine-use terms",
        "license_url": "https://authority.example/terms/v1",
        "terms_sha256": "1" * 64,
        "attribution": "Fixture maritime authority",
        "automated_processing_permitted": automated,
        "compiled_artifact_permitted": compiled,
        "synthetic_generation_permitted": synthetic,
    }
    if evidence:
        values.update(
            permission_evidence_sha256="2" * 64,
            permission_evidence_reference="Contract fixture, clause 4",
        )
    return DangerousGoodsLicenseReceipt.model_validate(values, strict=True)


def _source(
    *,
    regime: str = "IMDG",
    records: int = 2,
    license_receipt: DangerousGoodsLicenseReceipt | None = None,
) -> DangerousGoodsSourceReceipt:
    return DangerousGoodsSourceReceipt.model_validate(
        {
            "provider": "fixture-normalizer-v1",
            "authority": "Fixture regulatory authority",
            "dataset": "Fixture dangerous goods list",
            "regime": regime,
            "edition": "Fixture edition 2026",
            "source_url": "https://authority.example/dangerous-goods/2026",
            "source_bytes": 1234,
            "source_sha256": "3" * 64,
            "source_records": records,
            "parser_contract": "normalized_dangerous_goods_provider_v1",
            "maritime_authoritative": regime == "IMDG",
            "license": license_receipt or _license(),
        },
        strict=True,
    )


def _rows(source: DangerousGoodsSourceReceipt) -> tuple[DangerousGoodsRegulatoryTuple, ...]:
    return (
        build_dangerous_goods_regulatory_tuple(
            source=source,
            provider_record_id="fixture:UN1993:PGIII",
            un_number="1993",
            proper_shipping_name="FLAMMABLE LIQUID, N.O.S.",
            primary_class_division="3",
            compatibility_group=None,
            subsidiary_risks=("8",),
            packing_group="III",
            technical_name_required=True,
        ),
        build_dangerous_goods_regulatory_tuple(
            source=source,
            provider_record_id="fixture:UN2556:PGII",
            un_number="2556",
            proper_shipping_name=(
                "NITROCELLULOSE WITH ALCOHOL, not less than 25% alcohol, by mass"
            ),
            primary_class_division="4.1",
            compatibility_group=None,
            subsidiary_risks=(),
            packing_group="II",
            technical_name_required=False,
        ),
    )


class _Provider(DangerousGoodsProvider):
    def __init__(
        self,
        source: DangerousGoodsSourceReceipt,
        rows: Iterable[DangerousGoodsRegulatoryTuple],
    ) -> None:
        self._source = source
        self._rows = tuple(rows)
        self.iteration_count = 0

    def source_receipt(self) -> DangerousGoodsSourceReceipt:
        return self._source

    def iter_regulatory_tuples(self) -> Iterable[DangerousGoodsRegulatoryTuple]:
        self.iteration_count += 1
        return iter(self._rows)


class _FixedIndex:
    def __init__(self, index: object) -> None:
        self._index = index

    def randrange(self, stop: int) -> int:
        del stop
        return self._index  # type: ignore[return-value]


def _compile(
    tmp_path: Path,
    *,
    source: DangerousGoodsSourceReceipt | None = None,
    purpose: str = "production_generation",
    run_name: str = "dangerous-goods",
) -> CompiledDangerousGoodsRegistry:
    selected_source = source or _source()
    provider = _Provider(selected_source, _rows(selected_source))
    return compile_dangerous_goods_registry(
        provider=provider,
        output_parent=tmp_path / "artifacts",
        run_name=run_name,
        purpose=purpose,  # type: ignore[arg-type]
    )


def test_regulatory_tuple_preserves_divisions_subsidiary_order_and_independent_pg() -> None:
    source = _source(records=1)
    row = build_dangerous_goods_regulatory_tuple(
        source=source,
        provider_record_id="source-row-17",
        un_number="0332",
        proper_shipping_name="EXPLOSIVE, BLASTING, TYPE E",
        primary_class_division="1.5",
        compatibility_group="D",
        subsidiary_risks=("6.1", "8"),
        packing_group=None,
        technical_name_required=False,
    )

    assert row.primary_class_division == "1.5"
    assert row.compatibility_group == "D"
    assert row.subsidiary_risks == ("6.1", "8")
    assert row.packing_group is None
    assert row.entry_id.startswith("dg_")

    with pytest.raises(ValidationError, match="primary_class_division"):
        DangerousGoodsRegulatoryTuple.model_validate(
            {**row.model_dump(), "primary_class_division": "4"}, strict=True
        )
    with pytest.raises(ValidationError, match="compatibility group"):
        DangerousGoodsRegulatoryTuple.model_validate(
            {**row.model_dump(), "compatibility_group": None}, strict=True
        )
    with pytest.raises(ValidationError, match="subsidiary risks must be unique"):
        DangerousGoodsRegulatoryTuple.model_validate(
            {**row.model_dump(), "subsidiary_risks": ("8", "8")},
            strict=True,
        )


def test_flashpoint_is_shipment_specific_and_forbidden_in_regulatory_tuple() -> None:
    source = _source()
    row = _rows(source)[0]
    flashpoint = ShipmentFlashpoint(value=23.5, unit="celsius", test_method="Closed cup")
    selection = DangerousGoodsShipmentSelection(regulatory=row, flashpoint=flashpoint)

    assert selection.regulatory == row
    assert selection.flashpoint == flashpoint
    assert "flashpoint" not in row.model_dump(mode="json")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DangerousGoodsRegulatoryTuple.model_validate(
            {**row.model_dump(), "flashpoint": 23.5}, strict=True
        )
    with pytest.raises(ValidationError, match="absolute zero"):
        ShipmentFlashpoint(value=-460.0, unit="fahrenheit")


def test_normalized_provider_must_state_every_optional_regulatory_field() -> None:
    row = _rows(_source())[0]
    for field in ("compatibility_group", "subsidiary_risks", "packing_group"):
        incomplete = row.model_dump()
        del incomplete[field]
        with pytest.raises(ValidationError, match=field):
            DangerousGoodsRegulatoryTuple.model_validate(incomplete, strict=True)


def test_production_sampling_returns_one_atomic_immutable_source_row() -> None:
    source = _source()
    rows = tuple(sorted(_rows(source), key=lambda row: row.un_number))
    registry = DangerousGoodsRegistry(source=source, entries=rows)

    first = registry.sample_for_production(_FixedIndex(0))
    second = registry.sample_for_production(_FixedIndex(1))

    assert first is registry.entries[0]
    assert second is registry.entries[1]
    assert (
        first.un_number,
        first.proper_shipping_name,
        first.primary_class_division,
        first.subsidiary_risks,
        first.packing_group,
    ) == ("1993", "FLAMMABLE LIQUID, N.O.S.", "3", ("8",), "III")
    assert (
        second.un_number,
        second.primary_class_division,
        second.subsidiary_risks,
        second.packing_group,
    ) == ("2556", "4.1", (), "II")
    with pytest.raises(ValidationError, match="frozen"):
        first.packing_group = "I"
    for invalid in (-1, len(rows), True, "0"):
        with pytest.raises(DangerousGoodsRegistryError, match="invalid index"):
            registry.sample_for_production(_FixedIndex(invalid))


@pytest.mark.parametrize(
    ("license_receipt", "error_type", "match"),
    [
        (
            _license(automated=False),
            DangerousGoodsRegistryError,
            "does not permit automated compiled-artifact creation",
        ),
        (
            _license(compiled=False),
            DangerousGoodsRegistryError,
            "does not permit automated compiled-artifact creation",
        ),
        (
            _license(synthetic=False),
            DangerousGoodsGenerationDisabled,
            "licensed maritime-authoritative IMDG",
        ),
    ],
)
def test_production_compiler_rejects_unlicensed_source_before_reading_rows(
    tmp_path: Path,
    license_receipt: DangerousGoodsLicenseReceipt,
    error_type: type[Exception],
    match: str,
) -> None:
    source = _source(records=1, license_receipt=license_receipt)
    provider = _Provider(source, ())

    with pytest.raises(error_type, match=match):
        compile_dangerous_goods_registry(
            provider=provider,
            output_parent=tmp_path,
            run_name="must-not-exist",
        )

    assert provider.iteration_count == 0
    assert not (tmp_path / "must-not-exist").exists()


def test_non_maritime_registry_can_be_compiled_for_evaluation_but_never_generate(
    tmp_path: Path,
) -> None:
    source = _source(regime="TDG_CANADA")
    rows = _rows(source)
    blocked_provider = _Provider(source, rows)
    with pytest.raises(DangerousGoodsGenerationDisabled, match="maritime-authoritative IMDG"):
        compile_dangerous_goods_registry(
            provider=blocked_provider,
            output_parent=tmp_path / "production",
            run_name="blocked",
        )
    assert blocked_provider.iteration_count == 0

    evaluation_provider = _Provider(source, rows)
    result = compile_dangerous_goods_registry(
        provider=evaluation_provider,
        output_parent=tmp_path / "evaluation",
        run_name="allowed-for-evaluation",
        purpose="evaluation_only",
    )
    assert evaluation_provider.iteration_count == 1
    with pytest.raises(DangerousGoodsGenerationDisabled, match="licensed maritime-authoritative"):
        result.registry.sample_for_production(_FixedIndex(0))


def test_private_license_requires_complete_pinned_permission_evidence() -> None:
    with pytest.raises(ValidationError, match="pinned permission evidence"):
        _license(basis="written_permission")
    assert _license(basis="commercial_contract", evidence=True).permits_production_generation
    with pytest.raises(ValidationError, match="must not claim private permission evidence"):
        _license(evidence=True)


def test_source_receipt_rejects_false_maritime_authority_claim() -> None:
    values = _source(regime="TDG_CANADA").model_dump(mode="json")
    values["maritime_authoritative"] = True
    with pytest.raises(ValidationError, match="only an IMDG source"):
        DangerousGoodsSourceReceipt.model_validate(values, strict=True)


def test_registry_rejects_recombined_or_duplicate_source_rows() -> None:
    source = _source()
    rows = tuple(sorted(_rows(source), key=lambda row: row.un_number))
    recombined = rows[0].model_copy(update={"packing_group": rows[1].packing_group})
    with pytest.raises(DangerousGoodsRegistryError, match="identity is not bound"):
        DangerousGoodsRegistry(source=source, entries=(recombined, rows[1]))

    duplicate_provider_id = build_dangerous_goods_regulatory_tuple(
        source=source,
        provider_record_id=rows[0].provider_record_id,
        un_number="0004",
        proper_shipping_name="AMMONIUM PICRATE, dry or wetted",
        primary_class_division="1.1",
        compatibility_group="D",
        subsidiary_risks=(),
        packing_group=None,
        technical_name_required=False,
    )
    duplicate_rows = tuple(sorted((duplicate_provider_id, rows[0]), key=lambda row: row.un_number))
    with pytest.raises(DangerousGoodsRegistryError, match="duplicate stable/provider"):
        DangerousGoodsRegistry(source=source, entries=duplicate_rows)


def test_compiler_emits_canonical_self_hashed_receipt_and_is_idempotent(
    tmp_path: Path,
) -> None:
    first = _compile(tmp_path)
    payload = (first.root / "dangerous-goods.jsonl").read_bytes()
    lines = payload.splitlines()
    assert payload.endswith(b"\n")
    assert all(line == canonical_json_bytes(json.loads(line)) for line in lines)
    assert first.receipt.source.regime == "IMDG"
    assert first.receipt.source.edition == "Fixture edition 2026"
    assert first.receipt.source.license.license_name == "Fixture machine-use terms"

    receipt_payload = (first.root / "registry-receipt.json").read_bytes()
    assert receipt_payload == canonical_json_bytes(json.loads(receipt_payload)) + b"\n"
    assert (
        DangerousGoodsRegistryReceipt.model_validate_json(receipt_payload, strict=True)
        == first.receipt
    )
    second = _compile(tmp_path)
    assert second.created is False
    assert second.commit_receipt == first.commit_receipt

    (first.root / "dangerous-goods.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(StagedRunError, match="artifact inventory differs"):
        _compile(tmp_path)


def test_pinned_loader_checks_external_receipt_pin_and_artifact_bytes(tmp_path: Path) -> None:
    result = _compile(tmp_path)
    receipt_payload = (result.root / "registry-receipt.json").read_bytes()
    receipt_sha256 = sha256_bytes(receipt_payload)
    loaded = load_pinned_dangerous_goods_registry(
        result.root,
        expected_receipt_sha256=receipt_sha256,
    )
    assert loaded.source == result.registry.source
    assert loaded.entries == result.registry.entries

    with pytest.raises(DangerousGoodsRegistryError, match="receipt SHA-256 mismatch"):
        load_pinned_dangerous_goods_registry(
            result.root,
            expected_receipt_sha256="0" * 64,
        )
    (result.root / "dangerous-goods.jsonl").write_bytes(
        (result.root / "dangerous-goods.jsonl").read_bytes() + b"\n"
    )
    with pytest.raises(DangerousGoodsRegistryError, match="blank or unterminated"):
        load_pinned_dangerous_goods_registry(
            result.root,
            expected_receipt_sha256=receipt_sha256,
        )


def test_provider_count_mismatch_fails_without_publishing(tmp_path: Path) -> None:
    source = _source(records=1)
    provider = _Provider(source, _rows(source))
    with pytest.raises(DangerousGoodsRegistryError, match="provider count mismatch"):
        compile_dangerous_goods_registry(
            provider=provider,
            output_parent=tmp_path,
            run_name="count-mismatch",
        )
    assert provider.iteration_count == 1
    assert not (tmp_path / "count-mismatch").exists()
