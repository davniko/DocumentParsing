import asyncio
import json
import threading
from copy import deepcopy
from dataclasses import replace
from datetime import date
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.template_compiler import complete_pipeline as pipeline
from document_ocr.synthesis.template_compiler import descendant as render
from document_ocr.synthesis.template_compiler.complete_targets import propose_goods


def test_composite_measurement_does_not_imply_whole_target_unit_precision():
    from document_ocr.synthesis.template_compiler import complete_targets as targets

    source = NS(template=NS(bindings=()))
    assert targets._numeric_quantum(
        source, "documentPatch.cargoGroups[0].grossWeight.value", 48.751
    ) == Decimal("0.001")
    assert targets._numeric_quantum(
        source, "documentPatch.cargoGroups[0].grossWeight.value", 100
    ) == Decimal(1)


def test_measurement_quantum_uses_proven_printed_precision_across_adapters():
    from document_ocr.synthesis.template_compiler import complete_targets as targets

    path = "documentPatch.cargoGroups[0].grossWeight.value"
    target = {"documentPatch": {"cargoGroups": [{"grossWeight": {
        "value": 7740.0, "unit": "kilogram",
    }}]}}
    source = b"GROSS 7.740 KG"
    binding = NS(
        target_paths=(path,),
        realization=NS(adapter="agent"),
        occurrences=(NS(source_text="7.740", byte_start=6, byte_end=11),),
    )
    example = NS(source=source, target=target, template=NS(bindings=(binding,)))
    assert targets._numeric_quantum(example, path, 7740.0) == Decimal(1)

    source = b"GROSS 7740 KG\nEXACT 7740.25 KG"
    target["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] = 7740.25
    rounded = NS(
        source_text="7740", byte_start=source.index(b"7740"),
        byte_end=source.index(b"7740") + 4,
    )
    precise = NS(
        source_text="7740.25", byte_start=source.index(b"7740.25"),
        byte_end=source.index(b"7740.25") + 7,
    )
    binding.occurrences = (rounded, precise)
    example = NS(source=source, target=target, template=NS(bindings=(binding,)))
    assert targets._numeric_quantum(example, path, 7740.25) == Decimal("0.01")

    source = b"GROSS 11200.000KGS"
    target["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] = 11200.0
    binding.occurrences = (
        NS(source_text="11200", byte_start=6, byte_end=11),
        NS(source_text="000", byte_start=12, byte_end=15),
    )
    example = NS(source=source, target=target, template=NS(bindings=(binding,)))
    assert targets._numeric_quantum(example, path, 11200.0) == Decimal("0.001")


def _sampled_tare_fixture():
    source_target = {
        "schemaVersion": "5.0.0",
        "documentPatch": {
            "parties": {"carrier": {"name": "Carrier Ltd."}},
            "references": {"billOfLadingNumber": "SOURCE"},
        },
    }
    target = deepcopy(source_target)
    target["documentPatch"]["references"]["billOfLadingNumber"] = "SYNTHETIC"
    binding = NS(
        logical_key="tare_0",
        target_paths=(),
        value_kind="decimal_measurement",
        occurrences=(NS(slot_id="tare_slot", source_text="3,700 KGS"),),
    )
    contract = pipeline.numeric.NumericContract(
        mode="sampled_equipment_tare",
        role="tare",
        source_value="3700",
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Audited container tare",
    )
    source = NS(
        document_id="source_1",
        source=b"TARE 3,700 KGS",
        source_target=source_target,
        target=source_target,
        template=NS(
            carrier=NS(canonical_name="Carrier Ltd."),
            source_sha256="a" * 64,
            bindings=(),
        ),
    )
    scenario = NS(receipt={"sampledEquipmentTares": {"tare_0": "3850"}})
    return source, target, binding, contract, scenario


def test_sampled_tare_receipt_survives_real_prepare_freeze_and_render():
    source, target, binding, contract, scenario = _sampled_tare_fixture()
    sampled_values = pipeline._scenario_equipment_tares(scenario)
    assert sampled_values == {"tare_0": Decimal(3850)}
    prepared = pipeline.numeric.prepare(
        (binding,),
        {binding.logical_key: contract},
        source_target=source.target,
        target=target,
        scale=Decimal(1),
        equipment_tare_values=sampled_values,
    )
    case = pipeline._prepared(
        source,
        target,
        {},
        sample_id="synthetic_1",
        seed=7,
        numeric_values=prepared,
        equipment_tare_values=sampled_values,
    )

    render._require_frozen_target(case)
    assert case.target_receipt.equipment_tare_values_sha256 == sha256_bytes(
        canonical_json_bytes(scenario.receipt["sampledEquipmentTares"])
    )
    assert pipeline.numeric.render_prepared(
        (binding,),
        case.numeric_auxiliary,
        source_target=case.topology_reference_target,
        target=case.target,
        equipment_tare_values=case.equipment_tare_values,
    ) == {"tare_0": {"tare_slot": "3,850 KGS"}}
    with pytest.raises(ValueError, match="changed after generation"):
        render._require_frozen_target(
            replace(case, equipment_tare_values={"tare_0": Decimal(3900)})
        )
    with pytest.raises(ValueError, match="dependency contract"):
        pipeline.numeric.render_prepared(
            (binding,),
            case.numeric_auxiliary,
            source_target=case.topology_reference_target,
            target=case.target,
            equipment_tare_values={"tare_0": Decimal(3900)},
        )


@pytest.mark.parametrize(
    "receipt",
    (
        {},
        {"sampledEquipmentTares": {"tare_0": 3850}},
        {"sampledEquipmentTares": {"tare_0": "NaN"}},
        {"sampledEquipmentTares": {"tare_0": "03850"}},
        {"sampledEquipmentTares": {"tare_0": "-1"}},
    ),
)
def test_scenario_tare_receipt_must_be_explicit_canonical_and_positive(receipt):
    with pytest.raises(ValueError, match="sampled equipment tare"):
        pipeline._scenario_equipment_tares(NS(receipt=receipt))


@pytest.mark.parametrize("values", ({}, {"tare_0": Decimal(3850), "extra": Decimal(3900)}))
def test_scenario_tare_receipt_must_cover_exactly_the_sampled_bindings(values):
    source, target, binding, contract, _scenario = _sampled_tare_fixture()
    with pytest.raises(ValueError, match="cover exactly"):
        pipeline.numeric.prepare(
            (binding,),
            {binding.logical_key: contract},
            source_target=source.target,
            target=target,
            scale=Decimal(1),
            equipment_tare_values=values,
        )


def test_descendant_plan_receives_independent_tare_receipt(monkeypatch):
    from document_ocr.synthesis.template_compiler import geographic_context, package_prose

    source, target, binding, contract, scenario = _sampled_tare_fixture()
    values = pipeline._scenario_equipment_tares(scenario)
    prepared = pipeline.numeric.prepare(
        (binding,), {binding.logical_key: contract},
        source_target=source.target, target=target, scale=Decimal(1),
        equipment_tare_values=values,
    )
    case = pipeline._prepared(
        source, target, {}, sample_id="synthetic_1", seed=7,
        numeric_values=prepared, equipment_tare_values=values,
    )
    monkeypatch.setattr(render, "numeric_bindings", lambda _template: (binding,))
    monkeypatch.setattr(geographic_context, "source_entity_conflicts", lambda *_args: ())
    monkeypatch.setattr(package_prose, "validate_package_masses", lambda *_args: None)

    class RenderReached(Exception):
        pass

    def checked_render(bindings, numeric_values, **kwargs):
        assert bindings == (binding,)
        assert numeric_values == prepared
        assert kwargs["equipment_tare_values"] is case.equipment_tare_values
        assert pipeline.numeric.render_prepared(bindings, numeric_values, **kwargs) == {
            "tare_0": {"tare_slot": "3,850 KGS"}
        }
        raise RenderReached

    monkeypatch.setattr(render, "render_prepared", checked_render)
    with pytest.raises(RenderReached):
        render._build_initial_plan(case, seed=7, country_codes={})


def test_residual_checkpoint_cannot_replace_independent_tare_receipt():
    source, target, binding, contract, scenario = _sampled_tare_fixture()
    values = pipeline._scenario_equipment_tares(scenario)
    prepared = pipeline.numeric.prepare(
        (binding,), {binding.logical_key: contract},
        source_target=source.target, target=target, scale=Decimal(1),
        equipment_tare_values=values,
    )
    case = pipeline._prepared(
        source, target, {}, sample_id="synthetic_1", seed=7,
        numeric_values=prepared, equipment_tare_values=values,
    )
    checkpoint = {
        "sampleId": case.document_id,
        "sourceDocumentId": case.source_document_id,
        "target": case.target,
        "targetSha256": sha256_bytes(canonical_json_bytes(case.target)),
        "targetReceipt": case.target_receipt.model_dump(mode="json"),
        "auxiliaryValues": {},
        "numericAuxiliary": {
            key: row.model_dump(mode="json") for key, row in case.numeric_auxiliary.items()
        },
        "cargoScenario": {"sampledEquipmentTares": {"tare_0": "3900"}},
        "residualResponse": {"output": {}},
    }
    with pytest.raises(ValueError, match="sampled equipment tares differ"):
        pipeline._reusable_residual(checkpoint, case, NS(), "prompt")


def test_equal_container_partitions_constrain_generation_before_rendering(monkeypatch):
    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.template_compiler import complete_targets as targets

    path = "documentPatch.cargoGroups[0].volume.value"
    original = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "volume": {"value": 92.476, "unit": "cubic_metre"}}]
        }
    }
    source = NS(source=b"", target=original, template=NS(bindings=()))
    monkeypatch.setattr(targets.count_aliases, "fixed_quantities", lambda *a: {})
    for index in range(100):
        target = deepcopy(original)
        targets._scale_numbers(
            source,
            target,
            DeterministicStream(1, "partition", str(index)),
            {},
            {},
            {path: Decimal(".002")},
        )
        volume = Decimal(str(target["documentPatch"]["cargoGroups"][0]["volume"]["value"]))
        assert volume % Decimal(".002") == 0
        assert volume / 2 == (volume / 2).quantize(Decimal(".001"))
        assert 0 < volume <= Decimal("92.476")


@pytest.mark.parametrize("gross_unit", ["metric_tonne", "kilogram"])
def test_scaled_equal_net_gross_keep_equality_across_different_adapters(monkeypatch, gross_unit):
    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.template_compiler import complete_targets as targets

    patch = {
        "cargoGroups": [
            {
                "groupId": "g1",
                "grossWeight": {
                    "value": 50.729 if gross_unit == "metric_tonne" else 50729.0,
                    "unit": gross_unit,
                },
                "netWeight": {"value": 50.729, "unit": "metric_tonne"},
            }
        ]
    }
    net_binding = NS(
        target_paths=("documentPatch.cargoGroups[0].netWeight.value",),
        realization=NS(adapter="numeric"),
        occurrences=(NS(source_text="50.729"),),
    )
    source = NS(source=b"", target={"documentPatch": patch}, template=NS(bindings=(net_binding,)))
    monkeypatch.setattr(targets.count_aliases, "fixed_quantities", lambda *a: {})
    for index in range(30):
        target = deepcopy(source.target)
        targets._scale_numbers(source, target, DeterministicStream(1, "probe", str(index)), {}, {})
        group = target["documentPatch"]["cargoGroups"][0]
        assert Decimal(str(group["grossWeight"]["value"])) == Decimal(
            str(group["netWeight"]["value"])
        ) * (1 if gross_unit == "metric_tonne" else 1000)
        assert Decimal(str(group["grossWeight"]["value"])) % Decimal("0.001") == 0


def test_residual_exhaustion_keeps_frozen_target_and_all_costs(tmp_path, monkeypatch):
    """Failed insertion cannot start regenerating already accepted facts."""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("test prompt")
    config = NS(
        template_run=None,
        sample_plan_run=None,
        sample_plan=NS(records=1),
        documents=1,
        iso3166_snapshot=None,
        vessel_registry=NS(sha256="x", records=0),
        hs_manifest=None,
        hs_metadata=None,
        hs_report=None,
        generation_prompt=None,
        residual_prompt=NS(sha256="0" * 64),
        numeric_prompt=None,
        seed=1,
        output_dir="runs",
        run_name="test",
        response_cache_dir="cache",
        environment_file=None,
        provider=NS(pricing=NS(model_dump=lambda **kw: {"price": 1})),
        max_concurrent_requests=1,
        numeric_contract_run=NS(),
        generation_attempts=2,
        provider_launch_authorized=True,
        spending=NS(ledger_path="budget.sqlite3", maximum_estimated_cost_usd=Decimal(1)),
        request_batch_size=2,
        route_scenarios=None,
        route_scenarios_run=None,
        customs_program_registry=None,
        cargo_sampling=None,
        cargo_lexical_contracts=None,
        reviewed_generation=None,
    )
    monkeypatch.setattr(pipeline, "project_root_from_config", lambda _: tmp_path)
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline, "_pinned", lambda *a: prompt)
    monkeypatch.setattr(pipeline.render, "_validate_committed_run", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(
        pipeline.render,
        "_read_jsonl",
        lambda *a, **kw: [
            dict(
                sampleId="sample",
                sourceDocumentId="source",
                targetGeneration="complete_latest_schema_targets_v1",
            )
        ],
    )
    monkeypatch.setattr(pipeline.render, "_country_code_map", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_vessel_name_registry", lambda *a, **kw: NS(records=()))
    monkeypatch.setattr(pipeline, "load_ukgt_source_pin", lambda *a: None)
    monkeypatch.setattr(pipeline, "compile_uk_global_tariff_registry", lambda **kw: None)
    source = NS(
        source=b"original",
        source_target={},
        target={},
        template=NS(coherence_constraints=[], bindings=[], auxiliary_semantic_plan=NS(entities=[])),
    )
    monkeypatch.setattr(pipeline.targets, "load_source", lambda *a: source)
    monkeypatch.setattr(pipeline.targets, "reserve_identifiers", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline.targets, "lexical_contract", lambda *a: [])
    monkeypatch.setattr(pipeline.targets, "structured_proposal", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline.targets, "propose_goods", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline.targets, "scenario_scale", lambda *a: 1)
    for name in ("minimum_quantities", "quantity_multiples", "generated_measurements", "prepare"):
        monkeypatch.setattr(pipeline.numeric, name, lambda *a, **kw: {})
    monkeypatch.setattr(pipeline.numeric, "numeric_bindings", lambda *a: ())
    monkeypatch.setattr(pipeline, "_load_numeric_contracts", lambda *a: {"source": {}})
    monkeypatch.setattr(pipeline.render, "_provider_model", lambda **kw: None)
    monkeypatch.setattr(pipeline.render, "_validate_target_compatibility", lambda **kw: None)
    calls = []

    async def request(**kwargs):
        receipt = dict(
            output={},
            completedAt="test",
            wallSeconds=0,
            messages=[],
            usage=dict(requests=1, estimatedCostUsd="0.1"),
            cacheReused=False,
        )
        calls.append(receipt)
        return receipt

    proposals = []
    event_loop_thread = threading.get_ident()

    def proposal(*a, **kw):
        assert threading.get_ident() != event_loop_thread
        proposals.append(1)
        if len(proposals) > 1:
            raise ValueError("invalid second proposal")
        return {}, {}, {"retained": {}}

    monkeypatch.setattr(pipeline, "_cached_fields", request)
    monkeypatch.setattr(pipeline.targets, "complete_proposal", proposal)
    monkeypatch.setattr(
        pipeline,
        "_prepared",
        lambda *a, **kw: NS(
            target_receipt=NS(model_dump=lambda **kw: {}), dangerous_goods_facts=()
        ),
    )
    monkeypatch.setattr(
        pipeline.render,
        "_build_initial_plan",
        lambda *a, **kw: NS(residual_bindings=[NS(logical_key="test", target_paths=())], routes=[]),
    )
    monkeypatch.setattr(pipeline.render, "_residual_payload", lambda **kw: {})
    monkeypatch.setattr(
        pipeline.render, "_residual_output_type", lambda *a: NS(model_json_schema=lambda: {})
    )
    monkeypatch.setattr(
        pipeline,
        "ResidualStageReceipt",
        NS(model_validate=lambda *a: NS(model_dump=lambda **kw: {})),
    )
    result = NS(
        status="host_rejected", error_message="invalid residual", model_dump=lambda **kw: {}
    )

    def materialize(**kw):
        assert threading.get_ident() != event_loop_thread
        return NS(result=result, rendered=None)

    monkeypatch.setattr(pipeline.render, "_materialize_case", materialize)
    summary = asyncio.run(pipeline.run(prompt))
    assert summary["status"] == "incomplete"
    assert summary["failed"] == 1 and summary["trainingPublished"] is False
    assert len(calls) == summary["providerRequests"] == summary["newProviderRequests"] == 3
    assert summary["estimatedCostUsd"] == pytest.approx(0.3)
    assert len(proposals) == 1
    saved = json.loads((tmp_path / "runs/.work/test/sample.json").read_bytes())
    assert len(saved["attempts"]) == 1 and len(saved["residualAttempts"]) == 2


def test_cached_receipt_io_does_not_block_provider_event_loop(tmp_path, monkeypatch):
    from pydantic import create_model

    schema = create_model("CachedFields", value=(str, ...))
    contract = dict(system="test", prompt="{}", schema=schema.model_json_schema(), provider={})
    digest = sha256_bytes(canonical_json_bytes(contract))
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir()
    output = {"value": "accepted"}
    path.write_bytes(
        canonical_json_bytes(
            dict(
                requestSha256=digest,
                output=output,
                outputSha256=sha256_bytes(canonical_json_bytes(output)),
            )
        )
    )
    original = pipeline._read_receipt
    loop_thread = threading.get_ident()
    reads = []

    def read(path):
        reads.append(threading.get_ident())
        assert threading.get_ident() != loop_thread
        return original(path)

    monkeypatch.setattr(pipeline, "_read_receipt", read)

    async def run():
        result = await pipeline._cached_fields(
            cache=tmp_path,
            model=None,
            provider=NS(model_dump=lambda **kw: {}),
            system_prompt="test",
            payload={},
            output_type=schema,
            limiter=asyncio.Semaphore(1),
            authorized=False,
        )
        assert result["cacheReused"] and result["output"] == output

    asyncio.run(run())
    assert len(reads) == 1


def test_compilation_does_not_inherit_lightweight_realization_reasoning():
    from pathlib import Path

    from document_ocr.synthesis.template_compiler.models import ExtractionConfig
    from document_ocr.synthesis.template_compiler.pipeline import load_config

    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/synthesis/production/mpci_bl_template_compilation_remaining1420_v1_luna.yaml"
    )
    raw = config.model_dump(mode="json")
    raw["compiler_provider"]["reasoning_effort"] = "low"
    with pytest.raises(ValueError, match=r"high|max"):
        ExtractionConfig.model_validate_json(json.dumps(raw))


def test_numeric_extension_requests_only_missing_keys_and_keeps_verified_contracts(
    tmp_path, monkeypatch
):
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.numeric_auxiliary import NumericContract

    members = [
        NS(
            logical_key=k,
            value_kind="package",
            target_paths=(),
            derivation=None,
            occurrences=(NS(slot_id=k, source_text=f"{v} BOXES"),),
        )
        for k, v in (("first", 5), ("second", 1))
    ]

    def contract(value):
        return NumericContract(
            mode="target_share",
            role="cargo_quantity",
            source_value=str(value),
            target_paths=["documentPatch.quantity"],
            multiplier="1",
            divisor=1,
            reason="Source box constituents sum to the shipment total.",
        )

    source = NS(
        document_id="source",
        source=b"5 BOXES + 1 BOX = 6 BOXES",
        target={"documentPatch": {"quantity": 6}},
        template=NS(bindings=()),
    )
    monkeypatch.setattr(pipeline.numeric, "numeric_bindings", lambda _: tuple(members))
    monkeypatch.setattr(
        pipeline.numeric,
        "numeric_payload",
        lambda **kw: {"keys": [b.logical_key for b in kw["bindings"]]},
    )
    calls = []

    async def request(**kwargs):
        calls.append(kwargs["payload"])
        return {"output": {"numeric_0000": contract(1).model_dump(mode="json")}}

    monkeypatch.setattr(pipeline, "_cached_fields", request)
    prior = contract(5)
    result, receipts, failures = asyncio.run(
        pipeline._numeric_contracts(
            sources={"source": source},
            cache=tmp_path / "cache",
            work=tmp_path,
            model=None,
            config=NS(
                provider=None,
                provider_launch_authorized=True,
                generation_attempts=1,
                max_concurrent_requests=1,
            ),
            prompt="test",
            limiter=asyncio.Semaphore(1),
            existing_contracts={"source": {"first": prior}},
        )
    )
    assert not failures and len(receipts) == 1
    assert calls[0]["keys"] == ["second"]
    assert result["source"]["first"] == prior
    prepared = pipeline.numeric.prepare(
        members,
        result["source"],
        source_target=source.target,
        target={"documentPatch": {"quantity": 4}},
        scale=Decimal("0.7"),
    )
    assert [prepared[k].value for k in ("first", "second")] == ["3", "1"]


def test_pinned_numeric_catalog_has_exact_source_identity_and_complete_coverage(
    tmp_path, monkeypatch
):
    template = NS(bindings=(), model_dump=lambda **kwargs: {"testTemplate": 1})
    source = NS(source=b"original", target={"documentPatch": {}}, template=template)
    row = dict(
        sourceDocumentId="source",
        sourceSha256=sha256_bytes(source.source),
        sourceTargetSha256=sha256_bytes(canonical_json_bytes(source.target)),
        effectiveTemplateSha256=sha256_bytes(canonical_json_bytes(template.model_dump())),
        contracts={},
    )
    path = tmp_path / "contracts.jsonl"
    monkeypatch.setattr(pipeline.render, "_validate_committed_run", lambda *args, **kw: tmp_path)
    monkeypatch.setattr(pipeline.numeric, "numeric_bindings", lambda *args: ())
    path.write_text(json.dumps(row) + "\n")
    assert pipeline._load_numeric_contracts(tmp_path, None, {"source": source}) == {"source": {}}
    with pytest.raises(ValueError, match="does not cover"):
        pipeline._load_numeric_contracts(tmp_path, None, {"missing": source})
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        pipeline._load_numeric_contracts(tmp_path, None, {"source": source})
    row["sourceSha256"] = "0" * 64
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="differs from pinned"):
        pipeline._load_numeric_contracts(tmp_path, None, {"source": source})


@pytest.mark.parametrize(
    "code,issue_date", [("52010020", "2024-01-01"), ("520100", "2015-01-01"), ("520100", None)]
)
def test_tariff_generation_does_not_invent_jurisdiction_or_historical_support(code, issue_date):
    registry = NS(
        global_codes=("520100",),
        receipt=NS(snapshot_date=date(2026, 8, 31)),
        require_global=lambda *args, **kwargs: NS(valid_from=date(2022, 1, 1)),
    )
    target = {"documentPatch": {"cargoGroups": [{"groupId": "g1", "hsCodes": [code]}]}}
    if issue_date:
        target["documentPatch"]["issueDate"] = issue_date
    ledger = propose_goods(target, registry=registry, sample_id="sample", seed=1)
    assert target["documentPatch"]["cargoGroups"][0]["hsCodes"] == [code]
    assert ledger["g1"][0]["reason"]


def test_tariff_substitutions_are_unique_and_repeated_identities_stay_equal(monkeypatch):
    from document_ocr.synthesis.template_compiler import complete_targets

    monkeypatch.setattr(complete_targets, "classify_thermal_hs", lambda value: None)
    registry = NS(
        global_codes=("520100", "520110", "520120", "520130"),
        receipt=NS(snapshot_date=date(2026, 8, 31)),
        require_global=lambda *a, **kw: NS(
            valid_from=date(2022, 1, 1), heading_description="heading", description="variant"
        ),
    )
    for seed in range(20):
        target = {
            "documentPatch": {
                "issueDate": "2024-01-01",
                "cargoGroups": [
                    {"groupId": "g1", "hsCodes": ["520100", "520110"]},
                    {"groupId": "g2", "hsCodes": ["520100"]},
                ],
            }
        }
        propose_goods(target, registry=registry, sample_id="sample", seed=seed)
        groups = target["documentPatch"]["cargoGroups"]
        assert len(set(groups[0]["hsCodes"])) == 2
        assert not set(groups[0]["hsCodes"]) & {"520100", "520110"}
        assert groups[0]["hsCodes"][0] == groups[1]["hsCodes"][0]


def test_global_tariff_identity_cannot_diverge_from_its_fixed_national_extension():
    registry = NS(
        global_codes=("020210", "020220", "020230"),
        receipt=NS(snapshot_date=date(2026, 8, 31)),
        require_global=lambda *a, **kw: NS(
            valid_from=date(2022, 1, 1),
            heading_description="Frozen bovine meat",
            description="Boneless",
        ),
    )
    target = {
        "documentPatch": {
            "issueDate": "2024-01-01",
            "cargoGroups": [{"groupId": "g1", "hsCodes": ["020230", "02023000"]}],
        }
    }
    ledger = propose_goods(target, registry=registry, sample_id="sample", seed=42)
    assert target["documentPatch"]["cargoGroups"][0]["hsCodes"] == ["020230", "02023000"]
    assert ledger["g1"][0]["description"] == "Boneless"
    assert "shared" in ledger["g1"][0]["reason"]


def test_quantities_respect_subrow_support_and_unit_product_divisibility():
    from copy import deepcopy

    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.template_compiler.complete_targets import _scale_numbers

    original = {
        "documentPatch": {
            "cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": 16}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": str(i), "packageQuantity": 4} for i in range(4)
                    ],
                }
            ],
        }
    }
    minima = {
        f"documentPatch.cargoAllocationGroups[0].allocations[{i}].packageQuantity": 3
        for i in range(4)
    }
    for seed in range(20):
        target = deepcopy(original)
        _scale_numbers(
            NS(source=b"quantity 16", target=original, template=NS(bindings=[])),
            target,
            DeterministicStream(seed, "test", "sample"),
            minima,
            {"documentPatch.cargoPackages[0].quantity": 2},
        )
        patch = target["documentPatch"]
        total = patch["cargoPackages"][0]["quantity"]
        rows = patch["cargoAllocationGroups"][0]["allocations"]
        assert total % 2 == 0 and total <= 16
        assert all(r["packageQuantity"] >= 3 for r in rows)
        assert sum(r["packageQuantity"] for r in rows) == total


@pytest.mark.parametrize("has_imo", [True, False])
def test_vessel_name_is_not_detached_from_supplied_imo_identity(monkeypatch, has_imo):
    from document_ocr.synthesis.template_compiler import complete_targets as module

    monkeypatch.setattr(module, "apply_identifier_plan", lambda **kwargs: None)
    transport = {"vesselName": "ORIGINAL VESSEL"}
    if has_imo:
        transport["vesselImoNumber"] = "9897028"
    source = NS(
        source=b"ORIGINAL VESSEL",
        target={"documentPatch": {"transport": transport}},
        template=NS(bindings=[], coherence_constraints=[]),
    )
    target = module.structured_proposal(
        source, sample_id="sample", seed=42, allocations={}, vessel_names=["NEW VESSEL"]
    )
    assert target["documentPatch"]["transport"]["vesselName"] == (
        "ORIGINAL VESSEL" if has_imo else "NEW VESSEL"
    )


def test_composite_reference_block_does_not_merge_distinct_facts():
    from document_ocr.synthesis.template_compiler.complete_targets import lexical_contract

    paths = tuple(f"documentPatch.forwardingAndExportReferences[{i}]" for i in range(2))
    binding = NS(
        logical_key="invoice_block",
        target_paths=paths,
        derivation=None,
        dependency_paths=(),
        source_relationships=(),
        dependency_bindings=(),
        occurrences=(
            NS(
                source_text="A123, B456 DT: 2025-01-01",
                render_policy="natural_text",
                format_envelope=NS(newline_sequence=()),
            ),
        ),
        realization=NS(mode="agent_required", adapter="agent", slots=()),
    )
    source = NS(
        target={
            "documentPatch": {
                "forwardingAndExportReferences": ["A123 2025-01-01", "B456 2025-01-01"]
            }
        },
        template=NS(
            bindings=(binding,), coherence_constraints=(), auxiliary_semantic_plan=NS(entities=())
        ),
    )
    requests = lexical_contract(source)
    assert [row["paths"] for row in requests] == [[paths[0]], [paths[1]]]
