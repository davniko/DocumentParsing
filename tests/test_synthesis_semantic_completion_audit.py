from __future__ import annotations

import pytest

from document_ocr.synthesis.semantic_completion_audit import _source_feature_statistics


def test_source_feature_statistics_use_the_correct_document_and_class3_denominators() -> None:
    rows = [
        {
            "target": {
                "documentPatch": {
                    "containers": [{"temperatureSetpoint": {"value": -20, "unit": "celsius"}}],
                    "transport": {"vesselImoNumber": "9262704", "vesselFlagCountry": "France"},
                    "cargoGroups": [
                        {
                            "dangerousGoods": [
                                {
                                    "hazardCategory": "FLAMMABLE_LIQUIDS",
                                    "flashPoint": {"temperature": {"value": 10, "unit": "celsius"}},
                                }
                            ]
                        }
                    ],
                }
            }
        },
        {
            "target": {
                "documentPatch": {
                    "cargoGroups": [
                        {
                            "dangerousGoods": [
                                {
                                    "hazardCategory": "TOXIC_AND_INFECTIOUS_SUBSTANCES",
                                    "subsidiaryHazardCategories": ["FLAMMABLE_LIQUIDS"],
                                }
                            ]
                        }
                    ]
                }
            }
        },
    ]

    result = _source_feature_statistics(rows)

    assert result["counts"] == {
        "documents": 2,
        "thermal_documents": 1,
        "imo_documents": 1,
        "flag_documents": 1,
        "dangerous_goods_documents": 2,
        "class3_rows": 2,
        "class3_rows_with_flashpoint": 1,
    }
    assert result["ratesPercent"] == {
        "thermal document": 50.0,
        "IMO number": 50.0,
        "vessel flag": 50.0,
        "DG document": 100.0,
        "class-3 flashpoint": 50.0,
    }


def test_source_feature_statistics_fail_closed_on_missing_target() -> None:
    with pytest.raises(TypeError, match="no target object"):
        _source_feature_statistics([{"notTarget": {}}])
