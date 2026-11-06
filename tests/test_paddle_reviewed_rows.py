import copy
import json
from pathlib import Path

import numpy as np
import pytest

from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion
from document_ocr.synthesis.curated_reflow_geometry import layout_page, validate_page
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.paddle_reflow import PaddleBlock, PaddleReflow


@pytest.fixture(scope="module")
def engine():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/synthesis/mpci_bl_curated_v7_multirow.json").read_text())
    return PaddleReflow(root, ReflowPolicy.model_validate(config["positions"]["reflow"]))


def case():
    boxes = np.array(
        [
            [10, 50, 100, 60],
            [10, 100, 100, 120],
            [300, 105, 390, 125],
            [500, 20, 600, 30],
            [500, 40, 600, 50],
        ],
        dtype=float,
    )
    replacements = [
        {
            "id": "grow",
            "source": boxes[0].tolist(),
            "lineHeight": 10,
            "height": 100,
            "width": 90,
            "ownedBoxKeys": [tuple(boxes[0])],
        }
    ]
    return boxes, replacements


def layout(boxes, replacements, groups=()):
    return layout_page(
        boxes,
        replacements,
        gap_ratio=0.1,
        row_lock_fraction=1 / 3,
        bottom_guard_fraction=1,
        row_groups=groups,
    )


def test_reviewed_caption_value_row_moves_together_without_text_fusion():
    boxes, replacements = case()
    old, *_ = layout(boxes, replacements)
    assert old[0]["box"][1] > 100 and old[1]["box"][1] == 105
    nodes, _, _, scale = layout(boxes, replacements, ((tuple(boxes[1]), tuple(boxes[2])),))
    caption, value = nodes[:2]
    assert value["box"][1] - caption["box"][1] == 5
    assert value["box"][0] == 300 and caption["box"][0] == 10
    assert not validate_page(nodes, scale, minimum_gap=0)
    # The independent check must not rely on the solver's band assignment.
    corrupt = copy.deepcopy(nodes)
    corrupt[1]["band"] = -1
    corrupt[1]["box"][1] -= 20
    corrupt[1]["box"][3] -= 20
    assert any(
        f[0] == "reviewed_row_alignment" for f in validate_page(corrupt, scale, minimum_gap=0)
    )


@pytest.mark.parametrize("indices", [(1,), (1, 1), (0, 2)])
def test_invalid_reviewed_rows_fail_closed(indices):
    boxes, replacements = case()
    with pytest.raises(ValueError, match="reviewed row"):
        layout(boxes, replacements, (tuple(tuple(boxes[i]) for i in indices),))


def test_absent_reviewed_rectangle_fails_closed():
    boxes, replacements = case()
    with pytest.raises(ValueError, match="reviewed row"):
        layout(boxes, replacements, ((tuple(boxes[1]), (200, 100, 220, 120)),))


def test_reviewed_row_captures_replacement_original_box():
    boxes, replacements = case()
    replacements.append(
        {
            "id": "date",
            "source": boxes[2].tolist(),
            "lineHeight": 20,
            "height": 20,
            "width": 80,
            "ownedBoxKeys": [tuple(boxes[2])],
        }
    )
    nodes, _, _, scale = layout(boxes, replacements, ((tuple(boxes[1]), tuple(boxes[2])),))
    caption = next(n for n in nodes if n["id"] == "source:1")
    date = next(n for n in nodes if n["id"] == "date")
    assert date["box"][1] - caption["box"][1] == 5
    assert not validate_page(nodes, scale, minimum_gap=0)


def test_engine_replays_reviewed_row_and_explicit_omission(engine):
    boxes, _ = case()
    texts = ["GOODS", "ISSUE DATE", "01/02/2025", "TEXT", "TEXT"]
    pages = (
        PaddlePage(
            0,
            1000,
            1000,
            tuple(
                PaddleRegion(i, text, tuple(box))
                for i, (text, box) in enumerate(zip(texts, boxes, strict=True))
            ),
        ),
    )
    block = PaddleBlock("goods", 0, (0,), ("GOODS",), "\n".join(["OTHER"] * 10))
    kwargs = dict(identity="row-control", row_groups=((0, (1, 2)),))
    output, receipt = engine.render(pages, (block,), **kwargs)
    assert (output, receipt) == engine.render(pages, (block,), **kwargs)
    by_text = {r.text: r for r in output[0].regions}
    assert by_text["01/02/2025"].box[1] - by_text["ISSUE DATE"].box[1] == pytest.approx(5)
    assert receipt["reviewedRowGroups"] == ((0, (1, 2)),)
    omit = PaddleBlock("date", 0, (2,), ("01/02/2025",), "", "omit")
    omitted, _ = engine.render(pages, (block, omit), **kwargs)
    assert all(r.text != "01/02/2025" for r in omitted[0].regions)
    with pytest.raises(ValueError, match="invalid reviewed row"):
        engine.render(pages, (), identity="invalid", row_groups=((0, (1, 9)),))
    with pytest.raises(ValueError, match="do not share"):
        engine.render(pages, (), identity="invalid", row_groups=((0, (0, 2)),))
