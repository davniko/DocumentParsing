from __future__ import annotations

from dataclasses import replace

from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    normalize_plain_container_count_receipts,
    normalize_printed_freight_amount_total,
    validate_draft_source_alignment,
)


def _draft(
    raw: str,
    surface: str,
    *,
    key: str,
    occurrence: int = 0,
    kind: str = "decimal_measurement",
    group_kind: str = "commercial",
    group_key: str = "commercial:freight",
    derivation: str | None = None,
    dependency_paths: tuple[str, ...] = (),
) -> SpanDraft:
    start = -1
    for _ in range(occurrence + 1):
        start = raw.index(surface, start + 1)
    return SpanDraft(
        draft_id=key,
        logical_key=key,
        render_mode="deterministic_derived" if derivation else "deterministic_auxiliary",
        value_kind=kind,
        group_kind=group_kind,
        group_key=group_key,
        target_paths=(),
        derivation=derivation,
        dependency_paths=dependency_paths,
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(surface),
        source_text=surface,
        evidence_origin="host_verified_agent_proposal",
        render_policy="numeric_surface" if kind == "decimal_measurement" else "derived_surface",
        rationale="Exact source-owned amount or count.",
    )


def test_printed_prepaid_total_derives_from_distinct_source_amounts() -> None:
    raw = (
        "Basic Ocean Freight\nBAS Premium Package\nTotal USD\n"
        "Rate\n2940.00\n20.00\nUnit\nPer Container\nPer Container\n"
        "Currency\nUSD\nUSD\nUSD\n\nPrepaid\n8820.00\n60.00\n8880.00\n"
    )
    drafts = (
        _draft(raw, "2940.00", key="agent:freight:rate:basic"),
        _draft(raw, "20.00", key="agent:freight:rate:premium"),
        _draft(raw, "8820.00", key="agent:freight:amount:basic_prepaid"),
        _draft(raw, "60.00", key="agent:freight:amount:premium_prepaid"),
        _draft(raw, "8880.00", key="agent:freight:amount:total"),
    )

    result = normalize_printed_freight_amount_total(raw=raw, drafts=drafts)

    validate_draft_source_alignment(raw=raw, drafts=result)
    total = next(row for row in result if row.logical_key.endswith(":total"))
    assert total.derivation == "sum_monetary_amounts"
    assert total.render_mode == "deterministic_derived"
    assert total.dependency_bindings == (
        "agent:freight:amount:basic_prepaid",
        "agent:freight:amount:premium_prepaid",
    )
    assert all(row.derivation is None for row in result if row.logical_key != total.logical_key)


def test_freight_total_does_not_derive_from_accidental_or_wrong_sum() -> None:
    raw = "Total USD\nPrepaid\n8820.00\n60.00\n9000.00\n"
    drafts = (
        _draft(raw, "8820.00", key="agent:freight:amount:basic_prepaid"),
        _draft(raw, "60.00", key="agent:freight:amount:premium_prepaid"),
        _draft(raw, "9000.00", key="agent:freight:amount:total"),
    )
    assert normalize_printed_freight_amount_total(raw=raw, drafts=drafts) == drafts


def test_plain_one_container_row_is_count_not_equipment_type() -> None:
    raw = (
        "1 Container Said to Contain 1834 CARTONS PALLETIZED\n"
        "MNBU4015403 40 REEF 9'6 1834 CARTONS\n"
        "1 Container Said to Contain 1834 CARTONS PALLETIZED\n"
        "SUDU6082480 40 REEF 9'6 1834 CARTONS\n"
    )
    drafts = (
        _draft(
            raw,
            "1 Container",
            key="agent:equipment_receipt:container:0",
            kind="equipment",
            group_kind="equipment",
            group_key="equipment:container:0",
            derivation="equipment_receipt",
            dependency_paths=("documentPatch.containers[0]",),
        ),
        _draft(
            raw,
            "1 Container",
            key="agent:equipment_receipt:container:1",
            occurrence=1,
            kind="equipment",
            group_kind="equipment",
            group_key="equipment:container:1",
            derivation="equipment_receipt",
            dependency_paths=("documentPatch.containers[1]",),
        ),
    )
    source_target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "MNBU4015403"},
                {"containerNumber": "SUDU6082480"},
            ]
        }
    }

    result = normalize_plain_container_count_receipts(
        raw=raw, drafts=drafts, source_target=source_target
    )

    validate_draft_source_alignment(raw=raw, drafts=result)
    for index, row in enumerate(result):
        assert row.derivation == "container_count"
        assert row.value_kind == "integer"
        assert row.target_paths == (f"documentPatch.containers[{index}]",)
        assert row.dependency_paths == row.target_paths


def test_plain_container_count_does_not_erase_typed_equipment_receipt() -> None:
    raw = "1 X 40 REEF CONTAINER\n"
    draft = _draft(
        raw,
        "1 X 40 REEF CONTAINER",
        key="agent:equipment_receipt:container:0",
        kind="equipment",
        group_kind="equipment",
        group_key="equipment:container:0",
        derivation="equipment_receipt",
        dependency_paths=("documentPatch.containers[0]",),
    )
    source_target = {"documentPatch": {"containers": [{"containerNumber": "MNBU4015403"}]}}
    assert normalize_plain_container_count_receipts(
        raw=raw, drafts=(draft,), source_target=source_target
    ) == (draft,)
    assert normalize_plain_container_count_receipts(
        raw=raw,
        drafts=(replace(draft, source_text="2 Containers"),),
        source_target=source_target,
    ) == (replace(draft, source_text="2 Containers"),)
