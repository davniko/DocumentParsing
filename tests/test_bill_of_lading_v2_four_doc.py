from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/bill_of_lading_v2_four_doc.py"
_SPEC = importlib.util.spec_from_file_location("bill_of_lading_v2_four_doc", _TOOL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to import semantic-v2 four-document runner")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_page_splitter_preserves_text_and_order() -> None:
    joined = "--- PAGE 1 ---\nFIRST\n\n--- PAGE 2 ---\nSECOND"

    assert _MODULE._page_texts(joined) == {1: "FIRST", 2: "SECOND"}


def test_duplicate_comparison_allows_source_supported_additive_fields() -> None:
    first = {"documentPatch": {"billOfLadingNumber": "A", "goodsItems": [{"description": "X"}]}}
    second = {
        "documentPatch": {
            "billOfLadingNumber": "A",
            "goodsItems": [{"description": "X", "volume": {"value": 60.0}}],
        }
    }

    comparison = _MODULE._duplicate_comparison("first", first, "second", second)

    assert comparison["contradictions"] == []
    assert comparison["firstOnlyPaths"] == []
    assert comparison["secondOnlyPaths"] == ["documentPatch.goodsItems[0].volume.value"]


def test_duplicate_comparison_surfaces_shared_path_contradictions() -> None:
    first = {"documentPatch": {"goodsItems": [{"description": "FERRO MOLYBDENUM"}]}}
    second = {"documentPatch": {"goodsItems": [{"description": "COPPER"}]}}

    comparison = _MODULE._duplicate_comparison("first", first, "second", second)

    assert comparison["contradictions"] == [
        {
            "path": "documentPatch.goodsItems[0].description",
            "first": "FERRO MOLYBDENUM",
            "second": "COPPER",
        }
    ]


def test_semantic_contamination_gate_checks_all_target_strings_and_address_overlap() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "name": "EXAMPLE LTD",
                    "address": "11 MAIN STREET CAIRO TAX ID: 123",
                    "city": "CAIRO",
                }
            },
            "goodsItems": [{"description": "SHIPPER'S LOAD & COUNT"}],
        }
    }

    findings = _MODULE._semantic_contamination_findings(target)

    assert any("prohibited tax/boilerplate" in finding for finding in findings)
    assert any("address repeats separately emitted city" in finding for finding in findings)


def test_projection_benchmark_exercises_every_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = _MODULE.BillOfLadingLabel.model_validate(
        {
            "schemaVersion": "2.0.0",
            "documentPatch": {"billOfLadingNumber": "BOL-1"},
        },
        strict=True,
    )
    monkeypatch.setattr(_MODULE, "project_bill_of_lading_to_mpci", lambda value, **_: value)

    benchmark = _MODULE._benchmark_projection([label])

    assert benchmark["acceptedLabels"] == 1
    assert benchmark["projectionCount"] == 2_000
    assert benchmark["checksum"] == 4_000
    assert benchmark["projectionsPerSecond"] > 0
    assert benchmark["peakTracedBytes"] >= benchmark["residentOutputBytes"] > 0


def test_manifest_digest_publication_recovers_after_two_artifact_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    (tmp_path / "payload.json").write_text('{"ok":true}\n', encoding="utf-8")

    _MODULE._manifest()
    (tmp_path / "manifest.json.sha256").unlink()
    _MODULE._recover_manifest_digest()

    digest_line = (tmp_path / "manifest.json.sha256").read_text(encoding="utf-8")
    assert digest_line.endswith("  manifest.json\n")
