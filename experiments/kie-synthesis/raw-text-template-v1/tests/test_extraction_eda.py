from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest
from document_ocr.atomic import json_artifact_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.run_safety import StagedArtifactRun

from raw_text_template_experiment.extraction_eda import analyze_template_extraction

_PRICING = {
    "input_usd_per_million": "1",
    "cached_input_usd_per_million": "0.1",
    "cache_write_multiplier": "1.25",
    "output_usd_per_million": "2",
}


def _usage(*, input_tokens: int, cache_read: int, output_tokens: int) -> dict[str, object]:
    cost = (
        Decimal(input_tokens - cache_read)
        + Decimal(cache_read) * Decimal("0.1")
        + Decimal(output_tokens) * 2
    ) / Decimal(1_000_000)
    return {
        "requests": 1,
        "inputTokens": input_tokens,
        "cacheReadTokens": cache_read,
        "cacheWriteTokens": 0,
        "outputTokens": output_tokens,
        "reasoningTokens": output_tokens // 2,
        "visibleOutputTokens": output_tokens - output_tokens // 2,
        "estimatedCostUsd": str(cost),
    }


def _stage(
    *,
    role: str,
    pass_number: int,
    input_tokens: int,
    cache_read: int,
    output_tokens: int,
    status: str = "success",
    verdict: str | None = None,
) -> dict[str, object]:
    output: dict[str, object] = {}
    if verdict is not None:
        output = {"verdict": verdict, "findings": []}
    return {
        "role": role,
        "pass_number": pass_number,
        "status": status,
        "duration_seconds": float(input_tokens) / 100,
        "usage": _usage(
            input_tokens=input_tokens,
            cache_read=cache_read,
            output_tokens=output_tokens,
        ),
        "output": output,
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _build_run(parent: Path, *, name: str, token_multiplier: int = 1) -> Path:
    selected = []
    preflight = []
    results = []
    catalog = []
    template_payloads: dict[str, dict[str, object]] = {}
    for ordinal in range(1, 4):
        document_id = f"doc_{ordinal}"
        source_sha256 = str(ordinal) * 64
        selected.append(
            {
                "ordinal": ordinal,
                "document_id": document_id,
                "carrier_name": f"CARRIER {ordinal}",
                "carrier_family": f"FAMILY {ordinal % 2}",
                "document_type": "bill_of_lading" if ordinal != 2 else "sea_waybill",
                "template_proxy_id": f"template_{ordinal}",
                "page_count": ordinal,
                "ocr_lines": 10 * ordinal,
                "ocr_characters": 100 * ordinal,
                "container_count": ordinal,
                "cargo_group_count": ordinal,
                "package_count": ordinal,
                "dangerous_goods_count": int(ordinal == 3),
                "temperature_count": int(ordinal == 2),
                "source_sha256": source_sha256,
            }
        )
        preflight.append(
            {
                "ordinal": ordinal,
                "documentId": document_id,
                "sourceSha256": source_sha256,
                "sourceBytes": 110 * ordinal,
                "sourceLines": 10 * ordinal,
                "compilerRequestBytes": 500 * ordinal,
                "acceptedAnchorBindings": 4 * ordinal,
                "riskCandidates": 3 * ordinal,
                "requiredTargetCoBindings": ordinal,
            }
        )
    results.append(
        {
            "document_id": "doc_1",
            "status": "certified",
            "elapsed_seconds": 3.0,
            "compiler_stages": [
                _stage(
                    role="compiler",
                    pass_number=1,
                    input_tokens=100 * token_multiplier,
                    cache_read=20 * token_multiplier,
                    output_tokens=10 * token_multiplier,
                )
            ],
            "critic_stages": [
                _stage(
                    role="critic",
                    pass_number=1,
                    input_tokens=80 * token_multiplier,
                    cache_read=0,
                    output_tokens=8 * token_multiplier,
                    verdict="pass",
                )
            ],
        }
    )
    results.append(
        {
            "document_id": "doc_2",
            "status": "certified",
            "elapsed_seconds": 6.0,
            "compiler_stages": [
                _stage(
                    role="compiler",
                    pass_number=1,
                    input_tokens=120 * token_multiplier,
                    cache_read=0,
                    output_tokens=12 * token_multiplier,
                    status="host_rejected",
                ),
                _stage(
                    role="compiler",
                    pass_number=2,
                    input_tokens=130 * token_multiplier,
                    cache_read=30 * token_multiplier,
                    output_tokens=14 * token_multiplier,
                ),
            ],
            "critic_stages": [
                _stage(
                    role="critic",
                    pass_number=1,
                    input_tokens=90 * token_multiplier,
                    cache_read=10 * token_multiplier,
                    output_tokens=10 * token_multiplier,
                    verdict="pass",
                )
            ],
        }
    )
    results.append(
        {
            "document_id": "doc_3",
            "status": "rejected",
            "elapsed_seconds": 9.0,
            "compiler_stages": [
                _stage(
                    role="compiler",
                    pass_number=1,
                    input_tokens=140 * token_multiplier,
                    cache_read=0,
                    output_tokens=14 * token_multiplier,
                )
            ],
            "critic_stages": [
                _stage(
                    role="critic",
                    pass_number=1,
                    input_tokens=100 * token_multiplier,
                    cache_read=0,
                    output_tokens=10 * token_multiplier,
                    verdict="revise",
                )
            ],
        }
    )
    for ordinal, bindings, deterministic, agent in ((1, 10, 9, 1), (2, 20, 18, 2)):
        document_id = f"doc_{ordinal}"
        catalog.append(
            {
                "documentId": document_id,
                "sourceSha256": str(ordinal) * 64,
                "certified": True,
                "pages": ordinal,
                "lines": 10 * ordinal,
                "characters": 100 * ordinal,
                "documentType": "bill_of_lading" if ordinal != 2 else "sea_waybill",
                "bindings": bindings,
                "occurrences": bindings + 2,
                "deterministicBindings": deterministic,
                "agentAssistedBindings": agent,
                "agentResidualBindings": agent,
                "semanticOnlyTargetFacts": ordinal - 1,
                "renderModes": {
                    "target_binding": deterministic,
                    "agent_residual": agent,
                },
                "realizationModes": {
                    "single_surface": deterministic,
                    "agent_required": agent,
                },
                "valueKinds": {"identifier": bindings},
                "coherenceContracts": {"numeric_values": 1},
                "coherenceBindings": 1,
            }
        )
        template_payloads[document_id] = {
            "document_id": document_id,
            "coherence_constraints": [
                {
                    "kind": "numeric_values",
                    "member_logical_keys": [f"binding:{document_id}"],
                }
            ],
            "certification": {
                "source_hash_valid": True,
                "source_round_trip": True,
                "final_critic_pass": True,
            },
            "literal_certification": {
                "final_critic_pass": True,
                "remaining_unowned_risk_candidates": 0,
            },
        }

    all_stages = [
        stage
        for result in results
        for stage in [*result["compiler_stages"], *result["critic_stages"]]
    ]
    usage = {
        "stages": len(all_stages),
        "requests": sum(stage["usage"]["requests"] for stage in all_stages),
        "inputTokens": sum(stage["usage"]["inputTokens"] for stage in all_stages),
        "cacheReadTokens": sum(stage["usage"]["cacheReadTokens"] for stage in all_stages),
        "outputTokens": sum(stage["usage"]["outputTokens"] for stage in all_stages),
        "reasoningTokens": sum(stage["usage"]["reasoningTokens"] for stage in all_stages),
        "visibleOutputTokens": sum(stage["usage"]["visibleOutputTokens"] for stage in all_stages),
        "estimatedCostUsd": str(
            sum(
                (Decimal(str(stage["usage"]["estimatedCostUsd"])) for stage in all_stages),
                Decimal(0),
            )
        ),
    }
    summary = {
        "documents": 3,
        "statusCounts": {"certified": 2, "rejected": 1},
        "acceptanceGatePassed": False,
        "bindings": 30,
        "occurrences": 34,
        "deterministicBindings": 27,
        "agentAssistedBindings": 3,
        "agentResidualBindings": 3,
        "semanticOnlyTargetFacts": 1,
        "renderModeCounts": {"target_binding": 27, "agent_residual": 3},
        "realizationModeCounts": {"single_surface": 27, "agent_required": 3},
        "usage": {"all": usage},
    }
    config = {
        "inputs": {"source_corpus": {"records": 30}},
        "compiler_provider": {"pricing": _PRICING},
        "critic_provider": {"pricing": _PRICING},
    }
    artifacts: dict[str, bytes] = {
        "config.json": json_artifact_bytes(config),
        "summary.json": json_artifact_bytes(summary),
        "selection-manifest.json": json_artifact_bytes({"rows": selected}),
        "preflight.json": json_artifact_bytes({"cases": preflight}),
        "results.jsonl": _jsonl(results),
        "catalog.jsonl": _jsonl(catalog),
    }
    for document_id, template in template_payloads.items():
        artifacts[f"cases/{document_id}/template.json"] = json_artifact_bytes(template)
    transaction = sha256_bytes(
        canonical_json_bytes({"name": name, "tokenMultiplier": token_multiplier})
    )
    staged = StagedArtifactRun(
        output_parent=parent,
        run_name=name,
        transaction_sha256=transaction,
    )
    for relative, payload in artifacts.items():
        staged.publish_bytes(relative, payload)
    staged.commit(
        expected_artifacts=artifacts,
        metadata={"documents": 3, "schemaVersion": 1},
    )
    return staged.final_root


def test_success_eda_reconciles_cost_quality_coverage_and_exact_baseline(
    tmp_path: Path,
) -> None:
    current = _build_run(tmp_path / "inputs", name="current", token_multiplier=1)
    baseline = _build_run(tmp_path / "inputs", name="baseline", token_multiplier=2)

    output = analyze_template_extraction(
        run_dir=current,
        baseline_run_dir=baseline,
        output_parent=tmp_path / "analysis",
        run_name="eda",
        expected_documents=3,
        outlier_limit=2,
    )

    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["certifiedDocuments"] == 2
    assert summary["firstPassCertifiedDocuments"] == 1
    assert summary["providerRequests"] == 7
    assert summary["deterministicBindings"] == 27
    assert summary["agentAssistedBindings"] == 3
    assert summary["coherenceBindings"] == 2
    assert summary["coherenceContracts"] == 2
    assert summary["coherenceContractCounts"] == {"numeric_values": 2}
    assert summary["templateQualityGateFailures"] == 0
    assert Decimal(summary["coldNormalizedCostUsd"]) > Decimal(summary["observedCostUsd"])
    assert Decimal(summary["baseline"]["observedCostReductionFraction"]) == Decimal("0.5")
    assert summary["integrityChecks"]["summaryUsageReconciledToStages"] is True
    with (output / "documents.csv").open(newline="", encoding="utf-8") as stream:
        document_rows = list(csv.DictReader(stream))
    assert [row["document_id"] for row in document_rows] == ["doc_1", "doc_2", "doc_3"]
    assert document_rows[0]["first_pass_certified"] == "True"
    assert document_rows[1]["additional_stage_count"] == "1"
    assert len(list(output.glob("[0-9][0-9]-*.png"))) == 8
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in output.glob("*.png"))
    assert (output / "_COMMIT.json").is_file()


def test_success_eda_fails_closed_when_committed_input_is_tampered(tmp_path: Path) -> None:
    current = _build_run(tmp_path / "inputs", name="current")
    with (current / "summary.json").open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(ValueError, match="committed artifact byte count differs"):
        analyze_template_extraction(
            run_dir=current,
            output_parent=tmp_path / "analysis",
            run_name="eda",
            expected_documents=3,
        )
