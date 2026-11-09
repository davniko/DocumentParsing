from dataclasses import replace
from copy import deepcopy
import hashlib

import pytest

from document_ocr.spatial_inputs.paddle import (
    PaddlePage,
    PaddleRegion,
    RegionSpan,
    fingerprint,
    project,
    serialize,
    text_view_digests,
)


def test_text_view_digests_replace_inherited_metadata_without_changing_payload():
    row = {"joinedRawText": "New naïve name", "positionedText": "New naïve name || 1,2",
           "joinedRawTextSha256": "old raw", "positionedTextSha256": "old centroid",
           "target": {"name": "New naïve name"}}
    original = deepcopy(row)
    hashes = text_view_digests(row)
    assert row == original
    assert hashes == {field + "Sha256": hashlib.sha256(row[field].encode()).hexdigest()
                      for field in ("joinedRawText", "positionedText")}
    row.update(hashes)
    assert row["target"] == original["target"]
    assert text_view_digests(row) == hashes


def test_text_view_digests_reject_missing_or_nontext_views():
    with pytest.raises(KeyError):
        text_view_digests({"joinedRawText": "x"})
    with pytest.raises(TypeError, match="positionedText"):
        text_view_digests({"joinedRawText": "x", "positionedText": None})


def fixture():
    return (
        PaddlePage(
            0,
            200,
            400,
            (
                PaddleRegion(0, "3270 PACKAGE", (50, 100, 90, 120)),
                PaddleRegion(1, "TCLU", (10, 101, 20, 121)),
                PaddleRegion(2, "5473302", (22, 101, 40, 121)),
            ),
        ),
        PaddlePage(1, 200, 400, ()),
    )


def test_native_order_empty_pages_and_shared_geometry():
    pages = fixture()
    assert (
        serialize(pages) == "--- PAGE 1 ---\n3270 PACKAGE || 350,275\nTCLU || 75,278\n"
        "5473302 || 155,278\n\n--- PAGE 2 ---\n"
    )
    assert serialize(pages, geometry="bbox").splitlines()[1] == "3270 PACKAGE || 250,250,450,300"
    assert serialize(pages, geometry="none").splitlines()[1] == "3270 PACKAGE"
    assert fingerprint(pages) == fingerprint(pages)


def test_explicit_column_owners_and_identifier_join():
    spans = (RegionSpan(0, 1, 0, 4, "TCLU"), RegionSpan(0, 2, 0, 7, "5473302"))
    assert project(fixture(), spans, separator="") == "TCLU5473302"
    with pytest.raises(ValueError, match="overlapping"):
        project(fixture(), spans + spans, separator="")
    with pytest.raises(ValueError, match="differs"):
        project(fixture(), (replace(spans[0], expected="TCIU"),), separator="")


@pytest.mark.parametrize(
    "region",
    [
        PaddleRegion(1, "X", (0, 0, 1, 1)),
        PaddleRegion(0, "X\nY", (0, 0, 1, 1)),
        PaddleRegion(0, "X", (0, 0, float("nan"), 1)),
        PaddleRegion(0, "X", (0, 0, 201, 1)),
        PaddleRegion(0, "X", (0, 0, 0, 1)),
    ],
)
def test_invalid_evidence_is_not_repaired(region):
    with pytest.raises(ValueError):
        serialize((PaddlePage(0, 200, 400, (region,)),))


def test_page_identity_and_mode_fail_closed():
    with pytest.raises(ValueError):
        serialize(())
    with pytest.raises(ValueError):
        serialize((replace(fixture()[0], index=1),))
    with pytest.raises(ValueError):
        serialize(fixture(), geometry="guess")


def test_empty_native_recognition_preserves_region_identity():
    page = PaddlePage(0, 200, 400, (PaddleRegion(0, "", (0, 0, 1, 1)),))
    assert serialize((page,)) == "--- PAGE 1 ---\n || 2,1"
