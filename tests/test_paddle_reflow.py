import json
from pathlib import Path

import pytest

from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion, serialize
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.paddle_reflow import (
    PaddleBlock,
    PaddleReflow,
    physical_lines,
    rectangle_order,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", ["66", "1,234", "1234.500", "14,331000"])
def test_numeric_scalar_cannot_be_split_into_ambiguous_rows(value):
    class Metrics:
        @staticmethod
        def length(text):
            return len(text)

    with pytest.raises(ValueError, match="numeric scalar cannot wrap"):
        physical_lines(value, len(value) - 1, 1, Metrics())
    assert physical_lines(value, len(value), 1, Metrics()) == [value]


def test_textual_identifier_wrapping_is_unchanged():
    class Metrics:
        @staticmethod
        def length(text):
            return len(text)

    assert physical_lines("ABCD1234", 4, 1, Metrics()) == ["ABCD", "1234"]


def test_numeric_logical_line_remains_indivisible_in_compound_block():
    class Metrics:
        @staticmethod
        def length(text):
            return len(text)

    with pytest.raises(ValueError, match="numeric scalar cannot wrap"):
        physical_lines("123456\nPKGS", 5, 1, Metrics())
    assert physical_lines("12345\nPKGS", 5, 1, Metrics()) == ["12345", "PKGS"]


@pytest.fixture(scope="module")
def engine():
    cfg = json.loads((ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json").read_text())
    return PaddleReflow(ROOT, ReflowPolicy.model_validate(cfg["positions"]["reflow"]))


def pages():
    return (
        PaddlePage(
            0,
            1000,
            1000,
            (
                PaddleRegion(0, "PRODUCT", (10, 30, 80, 45)),
                PaddleRegion(1, "WEIGHT", (300, 30, 380, 45)),
                PaddleRegion(2, "AFTER", (10, 120, 70, 135)),
            ),
        ),
        PaddlePage(1, 1000, 1000, ()),
    )


def test_full_physical_fragments_and_replay(engine):
    value = "WOVEN COTTON SHIRTS WITH BUTTONS AND LONG SLEEVES " * 4
    block = PaddleBlock("goods", 0, (0,), ("PRODUCT",), value)
    rendered, receipt = engine.render(pages(), (block,), identity="control")
    assert (rendered, receipt) == engine.render(pages(), (block,), identity="control")
    fragments = receipt["pages"][0]["blocks"][0]["physicalFragments"]
    assert len(fragments) > 1
    assert " ".join(f["text"] for f in fragments) == value.strip()
    assert all(f["box"][2] < 300 for f in fragments)
    assert all(r.text != value for r in rendered[0].regions)
    assert not receipt["pages"][0]["geometryFailures"]
    assert rendered[1] == pages()[1]
    assert " || " in serialize(rendered)


def test_noop_and_explicit_omission(engine):
    assert engine.render(pages(), (), identity="unchanged")[0] == pages()
    block = PaddleBlock("removed_private", 0, (0,), ("PRODUCT",), "", "omit")
    output, receipt = engine.render(pages(), (block,), identity="omit")
    assert [r.text for r in output[0].regions] == ["WEIGHT", "AFTER"]
    assert receipt["pages"][0]["omittedRegions"] == [0]


def test_native_small_text_is_preserved_but_cannot_be_shrunk(engine):
    height = engine.calibration.minimum_page_median_height / 2
    source = (
        PaddlePage(
            0,
            1000,
            1000,
            (
                PaddleRegion(0, "ALPHA", (10, 30, 40, 30 + height)),
                PaddleRegion(1, "BOTTOM", (10, 980, 50, 980 + height)),
            ),
        ),
    )
    block = PaddleBlock("name", 0, (0,), ("ALPHA",), "OMEGA")
    rendered, _ = engine.render(source, (block,), identity="small-native")
    assert rendered[0].regions[0].box[3] - rendered[0].regions[0].box[1] == pytest.approx(height)
    # Force vertical growth that cannot fit without reducing the native size.
    growth = PaddleBlock("name", 0, (0,), ("ALPHA",), "\n".join(["OMEGA"] * 300))
    with pytest.raises(ValueError, match="unsafe Paddle page geometry"):
        engine.render(source, (growth,), identity="small-native")


def test_wrong_or_overlapping_owners_are_errors(engine):
    block = PaddleBlock("goods", 0, (0,), ("WRONG",), "SHIRTS")
    with pytest.raises(ValueError, match="differs"):
        engine.render(pages(), (block,), identity="x")
    a = PaddleBlock("a", 0, (0,), ("PRODUCT",), "SHIRTS")
    b = PaddleBlock("b", 0, (0,), ("PRODUCT",), "SHIRTS")
    with pytest.raises(ValueError, match="overlapping"):
        engine.render(pages(), (a, b), identity="x")


def test_foreign_column_inside_a_block_cannot_be_swallowed(engine):
    block = PaddleBlock("both", 0, (0, 2), ("PRODUCT", "AFTER"), "SHIRTS")
    page = PaddlePage(
        0,
        1000,
        1000,
        (
            pages()[0].regions[0],
            PaddleRegion(1, "FOREIGN", (10, 70, 70, 85)),
            pages()[0].regions[2],
        ),
    )
    with pytest.raises(ValueError, match="foreign text"):
        engine.render((page,), (block,), identity="x")


def test_sorter_uses_pixel_threshold_not_normalized_centroids():
    rows = [
        PaddleRegion(0, "right", (50, 20, 70, 30)),
        PaddleRegion(1, "left", (10, 25, 30, 35)),
        PaddleRegion(2, "lower", (1, 40, 20, 50)),
    ]
    assert [r.text for r in rectangle_order(rows)] == ["left", "right", "lower"]


def test_privacy_preserves_native_order_not_rectangle_sort(engine):
    source = (
        PaddlePage(
            0,
            1000,
            1000,
            (
                PaddleRegion(0, "RIGHT", (300, 30, 350, 45)),
                PaddleRegion(1, "LEFT", (10, 30, 55, 45)),
                PaddleRegion(2, "AFTER", (10, 120, 70, 135)),
            ),
        ),
    )
    block = PaddleBlock("private", 0, (2,), ("AFTER",), "OTHER")
    native, receipt = engine.render(source, (block,), identity="order", reading_order="source")
    generated, _ = engine.render(source, (block,), identity="order")
    assert [r.text for r in native[0].regions] == ["RIGHT", "LEFT", "OTHER"]
    assert [r.text for r in generated[0].regions] == ["LEFT", "RIGHT", "OTHER"]
    assert sorted((r.text, r.box) for r in native[0].regions) == sorted(
        (r.text, r.box) for r in generated[0].regions
    )
    assert receipt["readingOrder"] == "native_source_anchors_with_in_place_physical_fragments"
    omitted = PaddleBlock("private", 0, (2,), ("AFTER",), "", "omit")
    output, _ = engine.render(source, (omitted,), identity="order", reading_order="source")
    assert [r.text for r in output[0].regions] == ["RIGHT", "LEFT"]
    with pytest.raises(ValueError, match="unknown Paddle reading order"):
        engine.render(source, (), identity="order", reading_order="guess")


def test_native_order_keeps_new_fragments_at_their_owned_anchor(engine):
    block = PaddleBlock("goods", 0, (0,), ("PRODUCT",), "SHIRTS\nTROUSERS")
    output, receipt = engine.render(pages(), (block,), identity="wrap", reading_order="source")
    assert [r.text for r in output[0].regions] == ["SHIRTS", "TROUSERS", "WEIGHT", "AFTER"]
    assert len(receipt["pages"][0]["blocks"][0]["physicalFragments"]) == 2


def test_existing_caption_overlap_is_not_new_ownership():
    from document_ocr.synthesis.paddle_reflow import foreign_in_new_envelope

    own = [(93, 395, 576, 421), (93, 424, 234, 448)]
    envelope = (93, 395, 576, 448)
    assert not foreign_in_new_envelope(envelope, own, (92, 375, 744, 398))
    assert foreign_in_new_envelope(envelope, own, (300, 423, 500, 440))
    # Touching one owned line cannot excuse also occupying the unowned gap.
    assert foreign_in_new_envelope(envelope, own, (300, 417, 500, 430))


def test_wrapping_does_not_normalize_internal_phone_spaces(engine):
    from document_ocr.synthesis.paddle_reflow import physical_lines

    value = "TEL: +20  3  1234567"
    assert physical_lines(value, 1000, 0.01, engine.metrics) == [value]


def test_reviewed_background_watermark_is_retained_not_consumed(engine):
    page = PaddlePage(
        0,
        1000,
        1000,
        (
            PaddleRegion(0, "PRODUCT", (10, 30, 80, 45)),
            PaddleRegion(1, "COPY", (15, 25, 300, 100)),
            PaddleRegion(2, "DETAIL", (10, 50, 70, 65)),
        ),
    )
    block = PaddleBlock("goods", 0, (0, 2), ("PRODUCT", "DETAIL"), "NEW PRODUCT\nDETAIL")
    with pytest.raises(ValueError, match="foreign text"):
        engine.render((page,), (block,), identity="watermark")
    rendered, receipt = engine.render(
        (page,), (block,), identity="watermark", watermark_regions=((0, 1),)
    )
    assert sum(r.text == "COPY" for r in rendered[0].regions) == 1
    assert receipt["reviewedWatermarkRegions"] == [(0, 1)]
    with pytest.raises(ValueError, match="cannot own"):
        engine.render((page,), (block,), identity="watermark", watermark_regions=((0, 0),))


def test_reviewed_watermark_does_not_limit_foreground_column_or_move(engine):
    page = PaddlePage(0, 1000, 1000, (
        PaddleRegion(0, "PRODUCT", (10, 100, 80, 115)),
        PaddleRegion(1, "CERTIFIED", (82, 80, 800, 250)),
        PaddleRegion(2, "MASS", (600, 100, 700, 115)),
        PaddleRegion(3, "FOOTER", (10, 300, 80, 315)),
    ))
    value = "HYDRAULIC CYLINDERS FOR INDUSTRIAL MACHINERY"
    block = PaddleBlock("goods", 0, (0,), ("PRODUCT",), value)
    ordinary, old = engine.render((page,), (block,), identity="background-control")
    rendered, receipt = engine.render(
        (page,), (block,), identity="background-control", watermark_regions=((0, 1),),
        reading_order="source",
    )
    fragments = receipt["pages"][0]["blocks"][0]["physicalFragments"]
    assert len(fragments) < len(old["pages"][0]["blocks"][0]["physicalFragments"])
    assert all(f["box"][2] < 600 for f in fragments)
    watermark, = (r for r in rendered[0].regions if r.text == "CERTIFIED")
    assert watermark.box == page.regions[1].box
    assert " ".join(f["text"] for f in fragments) == value
    assert sum(r.text == "CERTIFIED" for r in ordinary[0].regions) == 1


def test_omitting_all_foreground_retains_reviewed_background(engine):
    page = PaddlePage(0, 1000, 1000, (
        PaddleRegion(0, "PRIVATE", (10, 100, 80, 115)),
        PaddleRegion(1, "COPY", (82, 80, 800, 250)),
    ))
    block = PaddleBlock("private", 0, (0,), ("PRIVATE",), "", "omit")
    output, _ = engine.render((page,), (block,), identity="omit", watermark_regions=((0, 1),))
    assert output[0].regions == (PaddleRegion(0, "COPY", page.regions[1].box),)


def test_reviewed_right_aligned_count_grows_left_without_wrapping(engine):
    page = PaddlePage(
        0,
        1000,
        1000,
        (
            PaddleRegion(0, "MARKS", (20, 30, 90, 50)),
            PaddleRegion(1, "1", (190, 30, 200, 50)),
            PaddleRegion(2, "VEHICLE", (205, 30, 300, 50)),
        ),
    )
    block = PaddleBlock("count", 0, (1,), ("1",), "12")
    output, receipt = engine.render(
        (page,), (block,), identity="aligned", right_aligned_numeric_regions=((0, 1),)
    )
    numeric = [r for r in output[0].regions if r.text == "12"]
    assert len(numeric) == 1
    assert numeric[0].box == pytest.approx((180, 30, 200, 50))
    assert output[0].regions[0].text == "MARKS"
    assert output[0].regions[2].text == "VEHICLE"
    assert receipt["reviewedRightAlignedNumericRegions"] == [(0, 1)]
    unchanged, _ = engine.render(
        (page,), (), identity="same-count", right_aligned_numeric_regions=((0, 1),)
    )
    assert unchanged == (page,)
    huge = PaddleBlock("count", 0, (1,), ("1",), "1" * 20)
    with pytest.raises(ValueError, match="cannot fit"):
        engine.render((page,), (huge,), identity="aligned", right_aligned_numeric_regions=((0, 1),))
    for invalid in [((0, 0),), ((0, 1), (0, 1)), ((5, 0),)]:
        with pytest.raises(ValueError, match="numeric owners"):
            engine.render(
                (page,), (block,), identity="aligned", right_aligned_numeric_regions=invalid
            )


def test_right_alignment_rejects_non_numeric_or_multi_region_owners(engine):
    block = PaddleBlock("name", 0, (0,), ("PRODUCT",), "12")
    with pytest.raises(ValueError, match="numeric owners"):
        engine.render(pages(), (block,), identity="x", right_aligned_numeric_regions=((0, 0),))
    block = PaddleBlock("multiple", 0, (0, 2), ("PRODUCT", "AFTER"), "12")
    with pytest.raises(ValueError, match="numeric owners"):
        engine.render(pages(), (block,), identity="x", right_aligned_numeric_regions=((0, 0),))


@pytest.mark.parametrize("compound", [False, True])
def test_numeric_slot_alignment_does_not_restrict_complete_package_lists(engine, compound):
    page = PaddlePage(
        0,
        1000,
        1000,
        (
            PaddleRegion(0, "1", (190, 30, 200, 50)),
            PaddleRegion(1, "PACKAGE", (220, 30, 300, 50)),
            PaddleRegion(2, "WEIGHT", (700, 30, 780, 50)),
        ),
    )
    regions = (0, 1) if compound else (0,)
    expected = ("1", "PACKAGE") if compound else ("1",)
    block = PaddleBlock("declaration", 0, regions, expected, "2 DRUMS + 3 PALLETS")
    normal, _ = engine.render((page,), (block,), identity="list")
    aligned, receipt = engine.render(
        (page,), (block,), identity="list", right_aligned_numeric_regions=((0, 0),)
    )
    assert aligned == normal
    assert receipt["reviewedRightAlignedNumericRegions"] == [(0, 0)]
