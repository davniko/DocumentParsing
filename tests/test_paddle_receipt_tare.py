import copy
from pathlib import Path

import pytest

from document_ocr.synthesis.paddle_render import extra_value


def test_aggregate_tare_can_use_explicit_sampled_surface_without_an_old_text_edit():
    """A sampled unprinted aggregate is still authoritative for newly visible rows."""
    pair = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
    candidate = {"surfaceValues": {"tare-total": ["4400"]}}
    current = {
        "target": {
            "containerInformation": [
                {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}
            ]
            * 2
        }
    }
    spec = {
        "index": 0,
        "minPlaces": 0,
        "aggregate": {"indices": [0, 1], "surfaceKey": "tare-total"},
        "support": {"pairs": {pair: [{"kg": "2200", "sourceIds": ["reviewed"]}]}},
    }
    recipe = {"owner": "tare-row", "equipmentTare": spec}
    assert extra_value(recipe, candidate, current, "test", None, project_root=Path.cwd()) == (
        "2200",
        None,
    )
    for aggregate in (
        {"indices": [0, 1]},
        {"indices": [0, 1], "surfaceKey": "tare-total", "fromKey": "other"},
        {"indices": [0, 1], "surfaceKey": "tare-total", "unknown": True},
    ):
        bad = copy.deepcopy(recipe)
        bad["equipmentTare"]["aggregate"] = aggregate
        with pytest.raises(ValueError, match="explicit sampled-value authority"):
            extra_value(bad, candidate, current, "test", None, project_root=Path.cwd())
    for values in (["4400", "4500"], ["0"], ["NaN"], ["4300"]):
        with pytest.raises(ValueError):
            extra_value(
                recipe,
                {"surfaceValues": {"tare-total": values}},
                current,
                "test",
                None,
                project_root=Path.cwd(),
            )


def test_receipt_tare_preserves_sampled_equipment_fact_without_new_draw():
    row = {
        "owner": 0,
        "kg": "2250.000",
        "observationSources": ["reviewed-source"],
        "pair": "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE",
    }
    candidate = {"physical": {"tares": [row]}}
    current = {
        "target": {
            "containerInformation": [
                {
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }
    recipe = {"receiptEquipmentTare": {"index": 0, "minPlaces": 0}}

    def render(value):
        return extra_value(recipe, value, current, "test", None, project_root=Path.cwd())

    assert render(candidate) == ("2250", None)
    assert candidate["physical"]["tares"][0] == row
    for kg in ("0", "-1", "NaN", "Infinity"):
        bad = copy.deepcopy(candidate)
        bad["physical"]["tares"][0]["kg"] = kg
        with pytest.raises(ValueError, match="positive and finite"):
            render(bad)
    for rows in (
        [],
        [row, row],
        [{**row, "observationSources": []}],
        [{**row, "pair": "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"}],
    ):
        with pytest.raises(ValueError):
            render({"physical": {"tares": rows}})
