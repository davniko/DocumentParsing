"""Contract tests for the paired KIE training-result analysis."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
_tool = importlib.import_module("tools.analyze_kie_synth_training_comparison")
common_fields = _tool.common_fields
leaves = _tool.leaves
paired_bootstrap = _tool.paired_bootstrap
score = _tool.score


def _prediction(reference: dict[str, object]) -> dict[str, str]:
    return {"reference_text": json.dumps({"documentPatch": reference})}


def test_common_fields_excludes_changed_target_family() -> None:
    runs = {
        "v5": {
            "a": _prediction({"number": "123", "containers": [{"sizeCategory": "FORTY"}]}),
            "b": _prediction({"number": "456"}),
        },
        "v3": {
            "a": _prediction({"number": "123", "containers": [{"typeDescription": "40 HC"}]}),
            "b": _prediction({"number": "456"}),
        },
    }
    families, report = common_fields(runs, "v5")
    assert families == {"$.documentPatch.number"}
    assert report["stable_families"] == 1


def test_exact_leaf_scoring_preserves_array_positions() -> None:
    reference = leaves({"packages": [{"quantity": 2}, {"quantity": 4}]})
    predicted = leaves({"packages": [{"quantity": 4}, {"quantity": 2}]})
    metrics = score([predicted], [reference])
    assert metrics["tp"] == 0
    assert metrics["f1"] == 0


def test_paired_bootstrap_is_deterministic_and_keeps_document_pairs() -> None:
    reference = [{("a", "1")}, {("b", "2")}]
    first = [set(reference[0]), set(reference[1])]
    second = [set(reference[0]), set()]
    a = paired_bootstrap(first, second, reference, replicates=200)
    b = paired_bootstrap(first, second, reference, replicates=200)
    assert a == b
    assert a["delta_f1"] > 0
    assert a["bootstrap_low_95"] >= 0
