import asyncio
import json
from copy import deepcopy
from decimal import Decimal

import pytest
from pydantic import ConfigDict, create_model

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.template_compiler.request_batches import (
    RequestBatcher,
    allocated_usage,
    lexical_payload,
    run_template_waves,
    semantic_aliases,
    shared_context,
    validate_batch_member,
)
from document_ocr.synthesis.usage_receipt import LinguisticUsageReceipt


def usage():
    return dict(
        requests=1,
        providerResponseIds=["provider-id"],
        finishReasons=["stop"],
        inputTokens=13,
        cacheReadTokens=2,
        cacheWriteTokens=9,
        outputTokens=17,
        reasoningTokens=12,
        visibleOutputTokens=5,
        estimatedCostUsd="0.003001000001",
        providerReportedCostUsd=None,
        downstreamProviders=[],
        providerTokenAccountingAnomaly=False,
    )


@pytest.mark.parametrize("count", [1, 7, 16, 33])
@pytest.mark.parametrize("provider_failure", [False, True])
def test_wave_readiness_batches_uneven_preparation_and_drains_partial_stages(
    count, provider_failure
):
    async def run():
        calls = []
        finished = []
        schema = create_model("Fields", slot=(str, ...))

        async def submit(**kwargs):
            calls.append(len(kwargs["payload"]["cases"]))
            if provider_failure and len(calls) == 1:
                raise ValueError("explicit provider failure")
            output = {case: {"slot": "new"} for case in kwargs["payload"]["cases"]}
            return dict(
                output=output,
                outputSha256=sha256_bytes(canonical_json_bytes(output)),
                requestSha256="a" * 64,
                usage=usage(),
            )

        batcher = RequestBatcher(size=16, submit=submit)

        async def execute(row):
            for stage in range(2):
                if stage and int(row["sampleId"]) % 2:
                    break  # A cached/deterministic second stage makes no request.
                await asyncio.sleep(0.0002 * (int(row["sampleId"]) % 16))
                try:
                    await batcher.request(
                        group=f"one:{stage}",
                        payload={
                            "sampleId": row["sampleId"],
                            "residualBindings": [
                                {
                                    "logicalKey": "one",
                                    "slots": [
                                        {
                                            "slotId": "slot",
                                            "sourceText": "old",
                                            "caseProfile": "mixed",
                                            "lineCount": 1,
                                        }
                                    ],
                                }
                            ],
                        },
                        output_type=schema,
                        system_prompt="test",
                    )
                except ValueError as error:
                    assert provider_failure and str(error) == "explicit provider failure"
            finished.append(row["sampleId"])

        await asyncio.wait_for(
            run_template_waves(
                [{"sourceDocumentId": "one", "sampleId": str(i)} for i in range(count)],
                batch_size=16,
                workers=1,
                execute=execute,
                should_stop=lambda: False,
            ),
            timeout=3,
        )
        await batcher.drain()
        assert len(finished) == count
        assert len(calls) == 2 * ((count + 15) // 16)
        assert max(calls) <= 16
        assert sum(calls) == count + (count + 1) // 2

    asyncio.run(run())


def test_template_waves_preserve_siblings_after_uneven_completions():
    async def run():
        admitted = []
        live = peak = 0
        samples = [
            {"sampleId": f"{source}-{i}", "sourceDocumentId": str(source)}
            for source, count in enumerate((7, 19, 8, 23, 4, 11))
            for i in range(count)
        ]

        async def execute(row):
            nonlocal live, peak
            admitted.append(row)
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.001 * (1 + int(row["sourceDocumentId"]) % 3))
            live -= 1

        await asyncio.wait_for(
            run_template_waves(
                samples, batch_size=8, workers=2, execute=execute, should_stop=lambda: False
            ),
            timeout=3,
        )
        assert peak <= 16 and live == 0
        assert {r["sampleId"] for r in admitted} == {r["sampleId"] for r in samples}
        assert len(admitted) == len(samples)
        # All members of an admitted wave start contiguously, including partial tails.
        expected = []
        for source in range(6):
            members = [r for r in samples if r["sourceDocumentId"] == str(source)]
            expected.extend(members[i : i + 8] for i in range(0, len(members), 8))
        offset = 0
        for wave in expected:
            assert admitted[offset : offset + len(wave)] == wave
            offset += len(wave)

    asyncio.run(run())


def test_template_wave_stop_drains_admitted_work_without_starting_next_wave():
    async def run():
        stopped = False
        completed = []
        samples = [{"sourceDocumentId": str(i // 3), "sampleId": str(i)} for i in range(9)]

        async def execute(row):
            nonlocal stopped
            stopped = True
            await asyncio.sleep(0)
            completed.append(row["sampleId"])

        await run_template_waves(
            samples, batch_size=3, workers=1, execute=execute, should_stop=lambda: stopped
        )
        assert completed == ["0", "1", "2"]

    asyncio.run(run())


@pytest.mark.parametrize("size,workers", [(0, 1), (1, 0), (-1, 2)])
def test_template_waves_reject_invalid_bounds(size, workers):
    async def execute(row):
        raise AssertionError("invalid input must never execute")

    with pytest.raises(ValueError, match="positive"):
        asyncio.run(
            run_template_waves(
                [], batch_size=size, workers=workers, execute=execute, should_stop=lambda: False
            )
        )


def test_compaction_preserves_nested_paths_including_punctuation():
    first = {"x": {"a.b[0]": [{"value": 2}, "same"]}, "empty": []}
    second = deepcopy(first)
    second["x"]["a.b[0]"][0]["value"] = 9
    result = shared_context([first, second])
    assert result["varyingPaths"] == [["x", "a.b[0]", 0, "value"]]
    assert result["sharedContext"] == first
    assert result["cases"]["s1"]["values"] == [9]
    assert first["x"]["a.b[0]"][0]["value"] == 2


def test_residual_compaction_uses_one_value_only_for_identical_owned_occurrences():
    def slot(identity, text="Same clause", profile="mixed"):
        return dict(slotId=identity, sourceText=text, caseProfile=profile, lineCount=1)

    payload = dict(
        residualBindings=[
            dict(logicalKey="one", slots=[slot("s1"), slot("s2"), slot("s3", profile="upper")]),
            dict(logicalKey="two", slots=[slot("s4")]),
        ]
    )
    aliases = semantic_aliases(payload, ("s1", "s2", "s3", "s4"))
    assert aliases == dict(s1="s1", s2="s1", s3="s3", s4="s4")
    compact = lexical_payload(payload, aliases)
    assert [s["slotId"] for b in compact["residualBindings"] for s in b["slots"]] == [
        "s1",
        "s3",
        "s4",
    ]
    assert len(payload["residualBindings"][0]["slots"]) == 3
    data = batch_member_fixture()
    data["provenance"]["fieldAliases"]["repeated"] = "slot"
    data["output"]["repeated"] = "Second"
    validate_batch_member(**data)
    data["output"]["repeated"] = "Different"
    with pytest.raises(ValueError, match="output differs"):
        validate_batch_member(**data)


def test_lexical_view_keeps_required_facts_without_duplicate_source_or_audit_text():
    field = {
        "key": "f0",
        "paths": ["documentPatch.parties.shipper.name"],
        "source": "Old Co",
        "constraints": [
            {
                "minimumWords": 2,
                "fixedLiteralTokens": [["CO"]],
                "completeTokenPartition": None,
                "surfaces": ["Old Co"],
            }
        ],
    }
    payload = {
        "requestedFields": [field],
        "structuredScenario": {
            "documentPatch": {
                "parties": {"shipper": {"name": "Old Co", "country": "CHINA"}},
                "cargoPackages": [{"quantity": 17}],
            }
        },
        "numericAuxiliary": {
            "tare": {
                "contract": {
                    "role": "tare",
                    "mode": "source_fixed",
                    "source_value": "3700",
                    "target_paths": [],
                    "reason": "lengthy audit rationale",
                },
                "value": "3700",
                "scenario_scale": ".8",
            }
        },
    }
    original = deepcopy(payload)
    output = lexical_payload(payload, {"f0": "shipper_name"})
    assert payload == original
    scenario = output["structuredScenario"]["documentPatch"]
    assert scenario["parties"]["shipper"] == {
        "name": {"generate": "shipper_name"},
        "country": "CHINA",
    }
    assert scenario["cargoPackages"] == [{"quantity": 17}]
    constraints = output["requestedFields"][0]["constraints"][0]
    assert constraints["minimumWords"] == 2 and constraints["fixedLiteralTokens"] == [["CO"]]
    assert "completeTokenPartition" not in constraints
    assert output["numericAuxiliary"]["tare"]["value"] == "3700"
    assert "lengthy audit rationale" not in json.dumps(output)


def batch_member_fixture():
    parent = {"s0": {"slot": "First"}, "s1": {"slot": "Second"}}
    provenance = dict(
        requestSha256="a" * 64,
        outputSha256=sha256_bytes(canonical_json_bytes(parent)),
        member="s1",
        members=2,
        usageAllocation="equal_share",
        fieldAliases={"slot": "slot"},
        parentUsage=usage(),
    )
    messages = [
        {
            "kind": "response",
            "provider_response_id": "provider-id",
            "parts": [{"part_kind": "text", "content": json.dumps(parent)}],
        }
    ]
    return dict(
        provenance=provenance,
        output=parent["s1"],
        messages=messages,
        usage=allocated_usage(usage(), 1, 2),
    )


def test_shared_request_zero_count_is_verified_not_invented():
    from document_ocr.synthesis.template_compiler.descendant_models import ResidualStageReceipt

    data = batch_member_fixture()
    validate_batch_member(**data)
    stage = ResidualStageReceipt.model_validate(
        dict(
            schema_version=1,
            document_id="sample",
            status="success",
            started_at=None,
            completed_at=None,
            duration_seconds=0,
            system_prompt_sha256="a" * 64,
            user_prompt_sha256=None,
            output_schema_sha256=None,
            output=data["output"],
            error_type=None,
            error_message=None,
            messages=data["messages"],
            usage=data["usage"],
            batch_provenance=data["provenance"],
        )
    )
    assert stage.usage["requests"] == 0


@pytest.mark.parametrize("corruption", ["member", "output", "hash", "usage", "request", "missing"])
def test_batch_replay_rejects_tampered_attribution(corruption):
    data = batch_member_fixture()
    if corruption == "member":
        data["provenance"]["member"] = "s0"
    if corruption == "output":
        data["output"]["slot"] = "Wrong"
    if corruption == "hash":
        data["provenance"]["outputSha256"] = "a" * 64
    if corruption == "usage":
        data["usage"]["requests"] = 1
    if corruption == "request":
        data["messages"][0]["provider_response_id"] = "Wrong"
    if corruption == "missing":
        data["provenance"].pop("parentUsage")
    with pytest.raises(ValueError):
        validate_batch_member(**data)


@pytest.mark.parametrize("rows", [[], [{"a": 1}, {"b": 1}], [{"a": []}, {"a": [1]}]])
def test_incompatible_batch_inputs_raise(rows):
    with pytest.raises(ValueError):
        shared_context(rows)


@pytest.mark.parametrize("count", [2, 3, 16, 32])
def test_allocated_billing_and_tokens_sum_to_exact_provider_receipt(count):
    original = usage()
    pieces = [allocated_usage(original, i, count) for i in range(count)]
    for piece in pieces:
        LinguisticUsageReceipt.model_validate_json(canonical_json_bytes(piece), strict=True)
    for field in (
        "requests",
        "inputTokens",
        "cacheReadTokens",
        "cacheWriteTokens",
        "outputTokens",
        "reasoningTokens",
        "visibleOutputTokens",
    ):
        assert sum(piece[field] for piece in pieces) == original[field]
    assert sum(Decimal(piece["estimatedCostUsd"]) for piece in pieces) == Decimal(
        original["estimatedCostUsd"]
    )


def test_partial_batches_drain_without_deadlock_and_preserve_named_ownership():
    async def run():
        calls = []

        async def submit(**kwargs):
            calls.append(kwargs)
            fields = kwargs["payload"]["sharedContext"]["requestedFields"]
            assert fields[0]["key"] == "shipper_name" and fields[1]["key"] == "consignee_name"
            output = {
                case: {
                    fields[0]["key"]: "New Shipper " + case,
                    fields[1]["key"]: "New Consignee " + case,
                }
                for case in kwargs["payload"]["cases"]
            }
            return dict(
                output=output,
                outputSha256=sha256_bytes(canonical_json_bytes(output)),
                requestSha256="a" * 64,
                usage=usage(),
                cacheReused=False,
                messages=[],
            )

        schema = create_model(
            "Values",
            __config__=ConfigDict(extra="forbid", strict=True),
            f0=(str, ...),
            f1=(str, ...),
        )
        batcher = RequestBatcher(size=16, submit=submit)
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    batcher.request(
                        group="one-source",
                        payload={
                            "sampleId": str(i),
                            "structuredScenario": {
                                "documentPatch": {
                                    "parties": {
                                        "shipper": {"name": "Old Shipper"},
                                        "consignee": {"name": "Old Consignee"},
                                    }
                                }
                            },
                            "requestedFields": [
                                {
                                    "key": "f0",
                                    "paths": ["documentPatch.parties.shipper.name"],
                                    "constraints": [],
                                },
                                {
                                    "key": "f1",
                                    "paths": ["documentPatch.parties.consignee.name"],
                                    "constraints": [],
                                },
                            ],
                        },
                        output_type=schema,
                        system_prompt="test",
                    )
                    for i in range(5)
                )
            ),
            timeout=2,
        )
        await batcher.drain()
        assert len(calls) == 1 and len(results) == 5
        assert [r["output"]["f0"] for r in results] == ["New Shipper s" + str(i) for i in range(5)]
        assert sum(r["usage"]["requests"] for r in results) == 1

    asyncio.run(run())


def test_budget_failure_reaches_every_waiting_member():
    from document_ocr.synthesis.template_compiler.spending_guard import SpendingLimitExceeded

    async def run():
        async def submit(**kwargs):
            raise SpendingLimitExceeded("limit")

        schema = create_model("Values", f0=(str, ...))
        batcher = RequestBatcher(size=2, submit=submit)
        results = await asyncio.gather(
            *(
                batcher.request(
                    group="same", payload={"id": i}, output_type=schema, system_prompt="test"
                )
                for i in range(2)
            ),
            return_exceptions=True,
        )
        await batcher.drain()
        assert all(isinstance(r, SpendingLimitExceeded) for r in results)

    asyncio.run(run())
