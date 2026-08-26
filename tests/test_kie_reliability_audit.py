from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from document_ocr.training.tasks import get_training_task


def _load_audit_module() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "audit_kie_reliability.py"
    spec = importlib.util.spec_from_file_location("kie_reliability_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load audit module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_AUDIT = _load_audit_module()
DatasetRecord = _AUDIT.DatasetRecord
PredictionRow = _AUDIT.PredictionRow
_counterfactual_fix = _AUDIT._counterfactual_fix
_make_shingles = _AUDIT._make_shingles
_schema_annotation_count = _AUDIT._schema_annotation_count
_structure_features = _AUDIT._structure_features
classify_reference_match = _AUDIT.classify_reference_match
raw_grounding = _AUDIT.raw_grounding
value_similarity = _AUDIT.value_similarity


def test_reference_match_distinguishes_wrong_index_from_wrong_value() -> None:
    path = "$.documentPatch.goodsItems[0].packages[0].type"
    predicted = [
        ("$.documentPatch.goodsItems[0].packages[0].type", '"CARTONS"', "CARTONS"),
        ("$.documentPatch.goodsItems[0].packages[1].type", '"BAGS"', "BAGS"),
    ]

    assert classify_reference_match(path, '"BAGS"', "BAGS", predicted) == (
        "exact_value_other_index",
        "$.documentPatch.goodsItems[0].packages[1].type",
        "BAGS",
    )


def test_similarity_retains_exact_metric_but_exposes_near_copy() -> None:
    result = value_similarity("Prime SPVC Resin", "Prime SPVC Resin 1091")

    assert result["space_normalized_exact"] is False
    assert result["prediction_contains_reference"] is True
    assert 0.8 < result["sequence_similarity"] < 1.0


def test_raw_grounding_handles_joined_fragments_without_accepting_absent_text() -> None:
    raw = "Zona Industrial de Oiã - Lote 34\nApartado 90\n3770-908 Oiã (OBR)"

    assert (
        raw_grounding("Zona Industrial de Oiã - Lote 34 Apartado 90 3770-908 (OBR)", raw)
        == "ordered_token_subsequence"
    )
    assert raw_grounding("UNPRINTED COMPANY", raw) == "not_grounded"


def test_schema_annotation_counter_ignores_property_named_description() -> None:
    schema = {
        "properties": {
            "description": {"type": "string", "description": "Cargo value"},
        },
        "title": "Root",
    }

    assert _schema_annotation_count(schema) == 2


def test_structure_features_identify_coupled_goods_complexity() -> None:
    record = DatasetRecord(
        document_id="doc_test",
        cohort="test",
        split="train",
        source_path="train.jsonl",
        raw_text="--- PAGE 1 ---\ntext",
        raw_text_sha256="0" * 64,
        target={
            "schemaVersion": "2.0.0",
            "documentPatch": {
                "containers": [
                    {"containerNumber": "MSCU6639870"},
                    {"containerNumber": "MSCU1234566"},
                ],
                "goodsItems": [
                    {
                        "description": "CARGO A",
                        "packages": [
                            {"quantity": 10, "type": "PALLETS"},
                            {"quantity": 100, "type": "CARTONS"},
                        ],
                    },
                    {"description": "CARGO B"},
                ],
            },
        },
        page_count=3,
    )

    features = _structure_features(record)
    assert "multiple_goods_items" in features
    assert "nested_package_levels" in features
    assert "multiple_containers" in features
    assert "complex_goods_structure" in features
    assert "three_or_more_pages" in features


def test_counterfactual_perfect_document_replaces_both_fp_and_fn() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    reference = '{"documentPatch":{"billOfLadingNumber":"BL-A"},"schemaVersion":"2.0.0"}'
    generated = '{"documentPatch":{"billOfLadingNumber":"BL-B"},"schemaVersion":"2.0.0"}'
    assessment = _AUDIT.assess_prediction(generated, reference, task)
    predictions = {
        "doc_test": PredictionRow(
            document_id="doc_test",
            generated_text=generated,
            reference_text=reference,
            assessment=assessment,
        )
    }

    baseline = _counterfactual_fix(predictions)
    repaired = _counterfactual_fix(predictions, document_filter={"doc_test"})

    assert baseline.true_positive == 0
    assert baseline.predicted == baseline.reference == 1
    assert repaired.true_positive == repaired.predicted == repaired.reference == 1


def test_template_shingles_ignore_identifier_digits_but_preserve_words() -> None:
    first = _make_shingles("ONE TWO THREE FOUR BL123 FIVE SIX")
    second = _make_shingles("ONE TWO THREE FOUR BL999 FIVE SIX")
    different = _make_shingles("ALPHA BETA GAMMA DELTA EPSILON ZETA ETA")

    assert first == second
    assert first != different
