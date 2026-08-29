from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from document_ocr.training.metrics import PredictionAssessment


def _load_tool() -> ModuleType:
    tools = Path(__file__).parents[1] / "tools"
    sys.path.insert(0, str(tools))
    path = tools / "analyze_kie_embedding_comparison.py"
    spec = importlib.util.spec_from_file_location("analyze_kie_embedding_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TOOL = _load_tool()


def _diagnostic(document_id: str, *, predicted: bool, reference_value: str = '"A"') -> object:
    field = "$.documentPatch.billOfLadingNumber"
    reference = frozenset({(field, reference_value)})
    predicted_values = reference if predicted else frozenset()
    assessment = PredictionAssessment(
        generated_text="{}",
        reference_text="{}",
        json_valid=True,
        schema_valid=True,
        canonical_exact_match=predicted,
        predicted_field_values=predicted_values,
        reference_field_values=reference,
        predicted_cargo_relation_facts=frozenset(),
        reference_cargo_relation_facts=frozenset(),
        predicted_category_values=frozenset(),
        reference_category_values=frozenset(),
    )
    record = _TOOL.Record(
        document_id=document_id,
        split="validation",
        raw_text="A",
        target={"documentPatch": {"billOfLadingNumber": "A"}},
        source_corpus="test",
        source_path="test.pdf",
        page_count=1,
        input_tokens=1,
        target_tokens=1,
        carrier_name="<MISSING>",
        carrier_family="<UNAVAILABLE>",
        template_id="<UNAVAILABLE>",
        template_seen_in_train="unknown",
        container_count=0,
        cargo_group_count=0,
        package_count=0,
        allocation_group_count=0,
        target_leaves=1,
    )
    return _TOOL.Diagnostic(
        record=record,
        assessment=assessment,
        true_positive=len(predicted_values & reference),
        predicted=len(predicted_values),
        reference=1,
        compared_paths=1,
        index_true_positive=len(predicted_values & reference),
        schema_failure_class="none",
        schema_failure_detail="",
        generated_characters=2,
        reference_characters=2,
    )


def _audit(key: str, diagnostics: tuple[object, ...]) -> object:
    spec = _TOOL.RunSpec(
        key=key,
        label=key,
        run_dir=Path("run"),
        evaluation_dir=Path("evaluation"),
        adapter_path=Path("adapter"),
        rank=32,
        alpha=64,
        epochs=1,
        embeddings_adapted=key == "embedding",
    )
    return _TOOL.RunAudit(
        spec=spec,
        diagnostics=diagnostics,
        schema_rows=(),
        documents=(),
        fields=(),
        sections=(),
        relations=(),
        category_facts=(),
        categories=(),
        groups=(),
        errors=(),
        mechanisms=(),
        history=(),
        manifest_metrics={},
        adapter={},
    )


def test_paired_bootstrap_uses_pooled_counts_and_reports_document_directions() -> None:
    embedding = _audit(
        "embedding",
        (_diagnostic("doc-1", predicted=True), _diagnostic("doc-2", predicted=False)),
    )
    baseline = _audit(
        "r32",
        (_diagnostic("doc-1", predicted=True), _diagnostic("doc-2", predicted=True)),
    )

    result = _TOOL._bootstrap_delta(embedding, baseline)

    assert result["point_delta"] == pytest.approx(-1 / 3)
    assert result["document_wins"] == 0
    assert result["document_ties"] == 1
    assert result["document_losses"] == 1
    assert result["upper_95"] <= 0


def test_prediction_parity_rejects_reference_drift() -> None:
    first = _audit("embedding", (_diagnostic("doc-1", predicted=True),))
    second = _audit(
        "r32",
        (_diagnostic("doc-1", predicted=True, reference_value='"DIFFERENT"'),),
    )

    with pytest.raises(ValueError, match="identical documents and references"):
        _TOOL._assert_prediction_parity((first, second))
