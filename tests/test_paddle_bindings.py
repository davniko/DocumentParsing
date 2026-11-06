import pytest

from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion, RegionSpan
from document_ocr.synthesis.paddle_bindings import (
    PaddleReplacement,
    compile_blocks,
    project_replacements,
    validate_existing_repetition,
)
from document_ocr.synthesis.paddle_render import source_passage_projection


def test_reviewed_postal_caption_is_removed_from_target_not_input():
    text = "NO. 35 ROAD, POST CODE: 00124"
    pages = (PaddlePage(0, 1000, 1000, (PaddleRegion(0, text, (10, 10, 500, 30)),)),)
    action = {"regions": [[0, 0]], "target": "/parties/shipper/addressLine",
              "stripPostalCaptions": ["POST CODE:"]}
    assert source_passage_projection(pages, (), action) == "NO. 35 ROAD, 00124"
    assert pages[0].regions[0].text == text
    assert source_passage_projection(pages, (), {"regions": [[0, 0]]}) == text
    with pytest.raises(ValueError, match="restricted to addressLine"):
        source_passage_projection(pages, (), {**action, "target": "/description"})


def test_complete_passage_repetition_preserves_different_sampled_counts():
    text = "12 CARTONS WITH 120 KG OF FISH"
    pages = (
        PaddlePage(
            0,
            1000,
            1000,
            tuple(PaddleRegion(i, text, (10, 10 + i * 30, 500, 30 + i * 30)) for i in range(3)),
        ),
    )
    action = {
        "groups": [{"regions": [[0, i]]} for i in range(3)],
        "targetCase": "upper",
        "deduplicatePassages": True,
    }
    assert source_passage_projection(pages, (), action) == text
    replacement = PaddleReplacement(
        "third", (RegionSpan(0, 2, 0, len(text), text),), "8 CARTONS WITH 80 KG OF FISH"
    )
    assert source_passage_projection(pages, (replacement,), action) == (
        text + " 8 CARTONS WITH 80 KG OF FISH"
    )
    assert source_passage_projection(pages, (), {"regions": [[0, 0]]}) == text
    with pytest.raises(ValueError, match="choose one passage"):
        source_passage_projection(pages, (), {**action, "regions": [[0, 0]]})
    with pytest.raises(ValueError, match="nonempty ordered list"):
        source_passage_projection(pages, (), {"groups": []})
    with pytest.raises(ValueError, match="explicit boolean"):
        source_passage_projection(pages, (), {**action, "deduplicatePassages": "false"})


def fixture():
    return (
        PaddlePage(
            0,
            1000,
            1000,
            (
                PaddleRegion(0, "ABCU1234567 / SEAL1 / 9 PALLETS", (10, 10, 500, 30)),
                PaddleRegion(1, "COMPANY", (10, 40, 100, 60)),
                PaddleRegion(2, "LIMITED", (10, 70, 100, 90)),
            ),
        ),
    )


def test_existing_repetition_tracks_logical_owner_before_frame_partition():
    from document_ocr.synthesis.paddle_bindings import partition_region_frames

    spans = (RegionSpan(0, 1, 0, 7, "COMPANY"), RegionSpan(0, 2, 0, 7, "LIMITED"))
    owner = PaddleReplacement("equipment@1:9", spans, "40HC")
    frames = partition_region_frames(owner, len, allow_empty_continuations=True)
    assert [frame.text for frame in frames] == ["40HC", ""]
    validate_existing_repetition((owner,), spans, semantic_text="40HC")
    with pytest.raises(ValueError, match="does not cover"):
        validate_existing_repetition((), spans)
    with pytest.raises(ValueError, match="does not cover"):
        validate_existing_repetition((owner,), spans[:1])
    with pytest.raises(ValueError, match="disagrees"):
        validate_existing_repetition((owner,), spans, semantic_text="45HC")
    # Independently repeated complete owners must each agree, not concatenate.
    other = PaddleReplacement("equipment@10:18", (spans[1],), "40HC")
    first = PaddleReplacement("equipment@1:9", (spans[0],), "40HC")
    validate_existing_repetition((first, other), spans, semantic_text="40HC")


def test_disjoint_subfields_share_one_physical_block():
    ps = fixture()
    items = (
        PaddleReplacement("container", (RegionSpan(0, 0, 0, 11, "ABCU1234567"),), "XYZU7654321"),
        PaddleReplacement("seal", (RegionSpan(0, 0, 14, 19, "SEAL1"),), "SEAL2"),
    )
    blocks = compile_blocks(ps, items)
    assert len(blocks) == 1
    assert blocks[0].text == "XYZU7654321 / SEAL2 / 9 PALLETS"


def test_literal_projection_handles_alternative_package_owners():
    pages = fixture()
    count = RegionSpan(0, 0, 22, 23, "9")
    noun = RegionSpan(0, 0, 24, 31, "PALLETS")
    whole = RegionSpan(0, 0, 22, 31, "9 PALLETS")
    # Full source passage, not a stale pre-formatting target or fixed owner name.
    assert (
        project_replacements(
            pages,
            (whole,),
            (
                PaddleReplacement("count", (count,), "12"),
                PaddleReplacement("noun", (noun,), "DRUMS"),
            ),
        )
        == "12 DRUMS"
    )
    assert (
        project_replacements(
            pages, (whole,), (PaddleReplacement("multi", (whole,), "12 IBCS + 8 PALLETS"),)
        )
        == "12 IBCS + 8 PALLETS"
    )
    with pytest.raises(ValueError, match="part of"):
        project_replacements(
            pages, (noun,), (PaddleReplacement("multi", (whole,), "12 IBCS + 8 PALLETS"),)
        )
    with pytest.raises(ValueError, match="overlapping replacement"):
        project_replacements(
            pages,
            (whole,),
            (PaddleReplacement("a", (whole,), "X"), PaddleReplacement("b", (noun,), "Y")),
        )


def test_multi_region_owner_and_overlap_guard():
    ps = fixture()
    name = PaddleReplacement(
        "name",
        (RegionSpan(0, 1, 0, 7, "COMPANY"), RegionSpan(0, 2, 0, 7, "LIMITED")),
        "NEW COMPANY LIMITED",
    )
    assert compile_blocks(ps, (name,))[0].text == "NEW COMPANY LIMITED"
    with pytest.raises(ValueError, match="overlapping character"):
        compile_blocks(ps, (name, PaddleReplacement("other", name.spans[:1], "X")))


def test_explicit_whole_region_omission():
    name = PaddleReplacement("omit", (RegionSpan(0, 1, 0, 7, "COMPANY"),), "")
    assert compile_blocks(fixture(), (name,))[0].action == "omit"


def test_separate_region_frames_preserve_words_without_merging_columns():
    from document_ocr.synthesis.paddle_bindings import partition_region_frames

    spans = (RegionSpan(0, 1, 0, 7, "COMPANY"), RegionSpan(0, 2, 0, 7, "LIMITED"))
    replacement = PaddleReplacement("address", spans, "NEW STREET 12 CAIRO EGYPT")
    parts = partition_region_frames(replacement, len)
    assert " ".join(p.text for p in parts) == replacement.text
    assert tuple(p.spans[0] for p in parts) == spans
    assert len(compile_blocks(fixture(), parts)) == 2
    assert project_replacements(fixture(), spans, parts) == replacement.text
    for invalid in (
        spans[::-1],
        spans[:1],
        (spans[0], spans[0]),
        (spans[0], RegionSpan(1, 2, 0, 7, "LIMITED")),
    ):
        with pytest.raises(ValueError):
            partition_region_frames(PaddleReplacement("a", invalid, replacement.text), len)
    with pytest.raises(ValueError, match="insufficient words"):
        partition_region_frames(PaddleReplacement("a", spans, "ONE"), len)
    with pytest.raises(ValueError, match="positive source"):
        partition_region_frames(replacement, lambda _: 0)


def test_reviewed_physical_group_retains_static_text_without_semantic_reassignment():
    name = PaddleReplacement("name", (RegionSpan(0, 1, 0, 7, "COMPANY"),), "NEW COMPANY")
    blocks = compile_blocks(fixture(), (name,), physical_groups=((0, (1, 2)),))
    assert len(blocks) == 1
    assert blocks[0].text == "NEW COMPANY\nLIMITED"
    with pytest.raises(ValueError, match="distinct ordered"):
        compile_blocks(fixture(), (name,), physical_groups=((0, (2, 1)),))


def test_explicit_short_name_preserves_text_and_omits_only_owned_continuation():
    from document_ocr.synthesis.paddle_bindings import partition_region_frames

    spans = (RegionSpan(0, 1, 0, 7, "COMPANY"), RegionSpan(0, 2, 0, 7, "LIMITED"))
    replacement = PaddleReplacement("name", spans, "Tetrahydrofuran")
    parts = partition_region_frames(replacement, len, allow_empty_continuations=True)
    assert [p.text for p in parts] == ["Tetrahydrofuran", ""]
    assert project_replacements(fixture(), spans, parts) == "Tetrahydrofuran"
    blocks = compile_blocks(fixture(), parts)
    assert [(b.regions, b.action) for b in blocks] == [((1,), "replace"), ((2,), "omit")]
    with pytest.raises(ValueError, match="nonempty generated"):
        partition_region_frames(
            PaddleReplacement("name", spans, ""), len, allow_empty_continuations=True
        )
    with pytest.raises(ValueError, match="positive source"):
        partition_region_frames(replacement, lambda _: 0, allow_empty_continuations=True)
