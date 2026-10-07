import json
from copy import deepcopy
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from document_ocr.spatial_inputs.alignment import enrich, parse_lines
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_layout import PositionPolicy, augment_page, validate_transform
from document_ocr.synthesis.curated_position_regions import place_region
from document_ocr.synthesis.curated_positions import (
    Anchor,
    _geometry_rows,
    _line_coordinates,
    augment_positions,
    load_page_geometry,
    position_campaign,
    transfer_positions,
)


def example(after="NEW SITE\nNEW DISTRICT\nNEW CITY", *, source_separator="\n"):
    source = (
        f"--- PAGE 1 ---\nSHIPPER\nÉTÉ STREET{source_separator}OLD CITY\nTEL: 123456789\n"
        "\n--- PAGE 2 ---\nOLD CITY\n"
    )
    before = f"ÉTÉ STREET{source_separator}OLD CITY"
    start = source.encode().index(before.encode())
    edit = {
        "key": "address",
        "occurrence": 0,
        "byteStart": start,
        "byteEnd": start + len(before.encode()),
        "before": before,
        "after": after,
    }
    rendered = source.replace(before, after, 1)
    proof = {
        "sourceSha256": digest(source.encode()),
        "renderedSha256": digest(rendered.encode()),
        "edits": [edit],
    }
    alignment = {
        "original_input_sha256": digest(source.encode()),
        "lines": [
            {
                "line_number": line.number,
                "page_index": line.page,
                "original_text": line.text,
                "xy": [100, line.number * 20] if line.page == 0 else [700, 100],
            }
            for line in parse_lines(source, 2)
        ],
    }
    alignment["positioned_input_sha256"] = digest(
        enrich(source, {line["line_number"]: line["xy"] for line in alignment["lines"]}).encode()
    )
    for record in alignment["lines"]:
        x, y = record["xy"]
        record["bbox"] = [x - 10, y - 4, x + 10, y + 4]
        record["regions"] = [{"bbox": record["bbox"]}]
    return source, rendered, proof, alignment


def measured(alignment):
    return {
        "page_boxes": {
            page: np.array([r["bbox"] for r in alignment["lines"] if r["page_index"] == page])
            for page in {r["page_index"] for r in alignment["lines"]}
        },
        "page_dimensions": {"0": [1000, 1000], "1": [1000, 1000]},
    }


def test_line_expansion_does_not_shift_later_fields_or_match_repeated_words():
    source, rendered, proof, alignment = example()
    text, result = transfer_positions(source, rendered, proof, alignment, **measured(alignment))
    assert "NEW SITE || 100,60\nNEW DISTRICT || 100,70\nNEW CITY || 100,80" in text
    assert "TEL: 123456789 || 100,100" in text
    assert "--- PAGE 2 ---\nOLD CITY || 700,100" in text
    assert result["measuredSyntheticGeometry"] is False
    telephone = next(line for line in result["lines"] if line["text"].startswith("TEL"))
    assert telephone["source_lines"] == [5]


@pytest.mark.parametrize(
    "after", ["NEW SITE\nNEW CITY", "NEW SITE\nNEW DISTRICT\nNEW CITY", "NEW SITE\n\nNEW CITY"]
)
def test_owned_paragraph_separator_is_not_a_page_boundary_or_coordinate_source(after):
    source, rendered, proof, alignment = example(after, source_separator="\n\n")
    text, receipt = transfer_positions(source, rendered, proof, alignment, **measured(alignment))
    assert "TEL: 123456789 || 100,120" in text
    assert "--- PAGE 2 ---\nOLD CITY || 700,100" in text
    assert receipt["placements"][0]["sourceLines"] == 2
    assert all(4 not in line["source_lines"] for line in receipt["lines"])
    assert all(set(line["source_lines"]) <= {3, 5} for line in receipt["lines"] if line["edits"])
    assert all(line["xy"] is not None for line in receipt["lines"] if line["edits"])
    assert receipt["renderedSha256"] == digest(rendered.encode())


@pytest.mark.parametrize("region", ["page_crossing", "blank_only"])
def test_blank_separator_support_does_not_license_page_crossings_or_anchorless_edits(region):
    source, _, _, alignment = example(source_separator="\n\n")
    raw = source.encode()
    if region == "page_crossing":
        start, end = raw.index(b"OLD CITY"), len(raw) - 1
    else:
        start = raw.index(b"\n\n") + 1
        end = start + 1
    replacement = b"REPLACEMENT"
    rendered = (raw[:start] + replacement + raw[end:]).decode()
    proof = {
        "sourceSha256": digest(raw),
        "renderedSha256": digest(rendered.encode()),
        "edits": [
            {
                "key": region,
                "occurrence": 0,
                "byteStart": start,
                "byteEnd": end,
                "before": raw[start:end].decode(),
                "after": replacement.decode(),
            }
        ],
    }
    with pytest.raises(ValueError, match="structural line or page boundary"):
        transfer_positions(source, rendered, proof, alignment, **measured(alignment))


@pytest.mark.parametrize(
    "after", ["ONE LINE", "FOUR\nNEW\nADDRESS\nLINES", "ÉTÉ STREET\nOLD CITY", "\nNEW CITY", ""]
)
def test_contraction_expansion_unicode_and_identity_keep_text_and_page_ownership(after):
    source, rendered, proof, alignment = example(after)
    text, receipt = transfer_positions(source, rendered, proof, alignment, **measured(alignment))
    assert "TEL: 123456789 || 100,100" in text
    assert receipt["renderedSha256"] == digest(rendered.encode())
    assert all(
        line["page_index"] == (1 if line["xy"][0] == 700 else 0)
        for line in receipt["lines"]
        if line["xy"] is not None
    )


def test_missing_anchor_stays_unknown_instead_of_neighbor_borrowing():
    source, rendered, proof, alignment = example()
    alignment["lines"][2]["xy"] = None
    alignment["positioned_input_sha256"] = digest(
        enrich(source, {line["line_number"]: line["xy"] for line in alignment["lines"]}).encode()
    )
    text, receipt = transfer_positions(source, rendered, proof, alignment)
    assert "NEW SITE ||\nNEW DISTRICT ||\nNEW CITY ||\n" in text
    assert receipt["placements"][0]["reason"] == "missing_source_anchor"


@pytest.mark.parametrize("peer_edit", [None, "quantity:0", "package_type:0"])
@pytest.mark.parametrize("moved", [(100, 80), (100, 120), None])
def test_inline_caption_and_numeric_peers_follow_reflow_without_inventing_unknowns(
    peer_edit, moved
):
    peers = [Anchor((3,), (100, 100), peer_edit), Anchor((3,), moved, "address:0", True)]
    assert _line_coordinates(peers) == moved
    # An unrelated line fragment remains relevant; it is not suppressed merely
    # because this output line also contains a relocated address or description.
    assert _line_coordinates([*peers, Anchor((4,), None)]) is None
    if moved is not None:
        assert _line_coordinates([*peers, Anchor((4,), (300, 200))]) == (
            200,
            round((moved[1] + 200) / 2),
        )


def test_source_line_peers_follow_expanded_region_and_competing_reflows_abstain():
    source, _, _, alignment = example()
    edits = []
    for key, before, after in [
        ("caption", "ÉTÉ", "NEW"),
        ("product", "STREET", "ITEM\nGRADE"),
    ]:
        start = source.encode().index(before.encode())
        edits.append(
            dict(
                key=key,
                occurrence=0,
                byteStart=start,
                byteEnd=start + len(before.encode()),
                before=before,
                after=after,
            )
        )

    def render(edits):
        raw = source.encode()
        for edit in reversed(edits):
            raw = raw[: edit["byteStart"]] + edit["after"].encode() + raw[edit["byteEnd"] :]
        rendered = raw.decode()
        proof = dict(sourceSha256=digest(source.encode()), renderedSha256=digest(raw), edits=edits)
        return transfer_positions(source, rendered, proof, alignment, **measured(alignment))

    text, receipt = render(edits)
    product = [line for line in receipt["lines"] if "product:0" in line["edits"]]
    assert all(line["xy"] is not None for line in product)
    assert all(b["xy"][1] - a["xy"][1] >= 8 for a, b in pairwise(product))
    assert "TEL: 123456789 || 100,100" in text
    edits[0]["after"] = "OTHER\nRENDERED\nREGION"
    _, receipt = render(edits)
    assert all(p["reason"] == "overlapping_reflow_owners" for p in receipt["placements"])
    assert all(line["xy"] is None for line in receipt["lines"] if line["edits"])


@pytest.mark.parametrize(
    "fault", ["hash", "text", "page", "coordinate", "duplicate", "order", "output", "structure"]
)
def test_corrupted_authorities_and_structural_edits_fail_closed(fault):
    source, rendered, proof, alignment = example()
    if fault == "hash":
        alignment["original_input_sha256"] = "bad"
    elif fault == "text":
        alignment["lines"][0]["original_text"] = "WRONG"
    elif fault == "page":
        alignment["lines"][0]["page_index"] = 9
    elif fault == "coordinate":
        alignment["lines"][0]["xy"] = [900, 900]
    elif fault == "duplicate":
        alignment["lines"].append(deepcopy(alignment["lines"][0]))
    elif fault == "order":
        proof["edits"].append(deepcopy(proof["edits"][0]))
    elif fault == "output":
        rendered += "INVENTED"
    elif fault == "structure":
        proof["edits"][0]["after"] = "--- PAGE 3 ---"
    with pytest.raises(ValueError):
        transfer_positions(source, rendered, proof, alignment)


def test_expanded_regions_use_measured_clearance_or_explicitly_abstain():
    record = {
        "xy": [100, 100],
        "bbox": [90, 96, 110, 104],
        "regions": [{"bbox": [90, 96, 110, 104]}],
    }
    boxes = np.array([[90, 96, 110, 104], [90, 140, 110, 148]])
    points, receipt = place_region([record], 3, boxes=boxes, dimensions=[1000, 1000])
    assert receipt["method"] == "local_space_reflow"
    assert len(set(points)) == 3
    assert all(y + 4 <= 122 for x, y in points)
    points, receipt = place_region([record], 22, boxes=boxes, dimensions=[1000, 1000])
    assert points == [None] * 22 and receipt["reason"] == "insufficient_local_space"
    assert place_region([record], 3)[1]["reason"] == "missing_measured_geometry"
    crowded = np.concatenate((boxes, [[95, 98, 115, 106]]))
    assert (
        place_region([record], 3, boxes=crowded, dimensions=[1000, 1000])[1]["reason"]
        == "unowned_text_intersects_source_region"
    )
    # Source-owned boxes must remain identifiable after real-pixel normalization;
    # a different arithmetic order used to misclassify these as foreign obstacles.
    dimensions = [1654, 2339]
    pixels = np.array([[601, 498, 768, 525], [601, 560, 768, 587]], dtype=float)
    normalized = pixels * 1000 / np.array([*dimensions, *dimensions])
    measured_record = {
        "xy": np.rint((normalized[0, :2] + normalized[0, 2:]) / 2).astype(int).tolist(),
        "bbox": pixels[0].tolist(),
        "regions": [{"bbox": pixels[0].tolist()}],
    }
    points, receipt = place_region([measured_record], 2, boxes=normalized, dimensions=dimensions)
    assert all(point is not None for point in points)
    assert receipt["method"] == "local_space_reflow"


def policy(**changes):
    return PositionPolicy(
        **dict(
            output_subdirectory="augmented",
            seed=4,
            scale_min=0.95,
            scale_max=1.05,
            max_translation=20,
            scale_attempts=32,
        )
        | changes
    )


def nearest(points):
    """Independent scalar reference, including all ties and repeated anchors."""
    result = []
    for i, (x, y) in enumerate(points):
        distances = {j: (x - a) ** 2 + (y - b) ** 2 for j, (a, b) in enumerate(points) if i != j}
        result.append({j for j, d in distances.items() if d == min(distances.values())})
    return result


@pytest.mark.parametrize(
    "points,boxes",
    [
        ([[100, 100], [100, 100], [200, 100], [200, 200]], [[90, 90, 210, 210]]),
        ([[0, 1000]], [[0, 998, 2, 1000]]),
        ([[100, 100], [101, 100], [100, 101]], [[90, 90, 110, 110]]),
    ],
)
def test_page_augmentation_is_reproducible_preserves_geometry_and_varies(points, boxes):
    seen = set()
    for seed in range(20):
        p = policy(seed=seed)
        output, receipt = augment_page(points, boxes, policy=p, identity="sample:0")
        again, replay = augment_page(points, boxes, policy=p, identity="sample:0")
        assert np.array_equal(output, again) and receipt == replay
        assert nearest(points) == nearest(output.tolist())
        assert np.all((output >= 0) & (output <= 1000))
        for i, a in enumerate(points):
            for j, b in enumerate(points):
                for axis in (0, 1):
                    assert np.sign(a[axis] - b[axis]) == np.sign(output[i, axis] - output[j, axis])
        seen.add(tuple(map(tuple, output)))
    assert len(seen) > 1


def test_dense_grid_exhaustion_is_explicit_and_no_geometry_is_invented():
    points = np.column_stack((np.arange(1001), np.full(1001, 500)))
    output, receipt = augment_page(
        points, [[0, 0, 1000, 1000]], policy=policy(scale_attempts=0), identity="dense"
    )
    assert np.array_equal(output, points)
    assert receipt["mode"] == "integer_translation_only" and receipt["changed_points"] == 0
    empty = np.empty((0, 2))
    output, receipt = augment_page(empty, np.empty((0, 4)), policy=policy(), identity="unknown")
    assert output.shape == (0, 2) and receipt["mode"] == "no_known_anchors"


@pytest.mark.parametrize(
    "points,boxes",
    [
        ([[float("nan"), 0]], [[0, 0, 10, 10]]),
        ([[1001, 0]], [[0, 0, 10, 10]]),
        ([[0.5, 0]], [[0, 0, 10, 10]]),
        ([[0, 0]], [[10, 0, 0, 10]]),
        ([[0, 0]], np.empty((0, 4))),
    ],
)
def test_invalid_geometry_is_rejected_not_clamped(points, boxes):
    with pytest.raises(ValueError):
        augment_page(points, boxes, policy=policy(), identity="broken")


def test_final_quantization_must_not_change_neighbour_ties():
    before = np.array([[100, 100], [110, 100], [100, 110]])
    after = np.rint((before - 500) * 0.95 + 500 + [0.1, 0.8])
    with pytest.raises(ValueError, match="nearest_neighbour"):
        validate_transform(before, after, [[90, 90, 120, 120]], scale=0.95, dx=0.1, dy=0.8)
    for bad in (float("nan"), -1):
        with pytest.raises(ValueError):
            validate_transform(before, before, [[90, 90, 120, 120]], scale=bad, dx=0, dy=0)


@pytest.mark.parametrize("fault", ["individual_jitter", "off_page", "collapsed_columns"])
def test_geometric_negative_controls_are_rejected(fault):
    before = np.array([[100, 100], [101, 100], [200, 200]])
    boxes = [[90, 90, 210, 210]]
    scale, dx, dy = 1.0, 0, 0
    after = before.copy()
    if fault == "individual_jitter":
        after[0, 0] += 20
    elif fault == "off_page":
        dx = 950
        after[:, 0] += dx
    else:
        scale = 0.1
        after = np.rint((before - 500) * scale + 500)
    with pytest.raises(ValueError):
        validate_transform(before, after, boxes, scale=scale, dx=dx, dy=dy)


def test_augmented_document_keeps_unknowns_source_pages_text_and_is_label_blind():
    source, rendered, proof, alignment = example()
    _, inherited = transfer_positions(source, rendered, proof, alignment)
    inherited["lines"][1]["xy"] = None
    geometry = {0: np.array([[50, 20, 800, 400]]), 1: np.array([[650, 50, 750, 150]])}
    original = deepcopy(inherited)
    text, receipt = augment_positions(rendered, inherited, geometry, policy(), "sample")
    assert inherited == original
    assert receipt["lines"][1]["xy"] is None
    assert receipt["lines"][1]["anchor_xy"] is None
    assert [r["page_index"] for r in receipt["lines"]] == [
        r["page_index"] for r in inherited["lines"]
    ]
    inherited["unused_labels"] = {"shipper": "UNRELATED"}
    assert augment_positions(rendered, inherited, geometry, policy(), "sample")[0] == text
    assert text.count("--- PAGE") == 2


@pytest.fixture
def positioned_campaign(tmp_path):
    source, rendered, proof, alignment = example()
    data, paddle, output = [tmp_path / s for s in ("data", "paddle", "pilot")]
    for p in (data / "alignments", paddle, output / "candidates"):
        p.mkdir(parents=True)
    sid, sample = "doc_source", "syn_sample"
    alignment.update(document_id=sid, pdf_id="pdf_abc", source_sha256="abc")
    regions = []
    for i, line in enumerate(alignment["lines"]):
        x, y = line["xy"]
        box = [x - 1, y - 1, x + 1, y + 1]
        region = dict(index=i, bbox=box, text=line["original_text"], recognition_score=1.0)
        line.update(bbox=box, region_indices=[i], regions=[region])
        regions.append(
            dict(
                document_id="pdf_abc",
                page_index=line["page_index"],
                line_index=i,
                text=line["original_text"],
                recognition_score=1.0,
                **dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)),
            )
        )
    pages = [
        dict(
            document_id="pdf_abc",
            page_index=p,
            source_sha256="abc",
            width=1000,
            height=1000,
            recognized_lines=sum(r["page_index"] == p for r in regions),
            coordinate_frame="rendered_page",
        )
        for p in (0, 1)
    ]
    pq.write_table(pa.Table.from_pylist(pages), paddle / "pages.parquet")
    pq.write_table(pa.Table.from_pylist(regions), paddle / "lines.parquet")
    positioned = enrich(source, {r["line_number"]: r["xy"] for r in alignment["lines"]})
    source_row = dict(documentId=sid, joinedRawText=source, positionedText=positioned)
    (data / "train.jsonl").write_text(json.dumps(source_row) + "\n")
    manifest = dict(
        config=dict(alignment=dict(grid_size=1000), paddle_run="paddle"),
        output_sha256={"train.jsonl": digest((data / "train.jsonl").read_bytes())},
        source_sha256={
            str(paddle / n): digest((paddle / n).read_bytes())
            for n in ("pages.parquet", "lines.parquet")
        },
    )
    (data / "manifest.json").write_text(json.dumps(manifest))
    (data / "alignments" / f"{sid}.json").write_text(json.dumps(alignment))
    row = dict(
        documentId=sample,
        sourceDocumentId=sid,
        joinedRawText=rendered,
        target={"goods": "UNCHANGED"},
    )
    (output / "dataset.jsonl").write_text(json.dumps(row) + "\n")
    (output / "manifest.json").write_text(
        json.dumps({"files": {"dataset.jsonl": digest((output / "dataset.jsonl").read_bytes())}})
    )
    (output / "candidates" / f"{sample}.json").write_text(json.dumps(dict(row, proof=proof)))
    calls = []
    campaign = SimpleNamespace(
        root=tmp_path,
        output=output,
        rows={sid: source_row},
        config={"dataset": "data", "positions": policy().model_dump()},
        replay_candidate=lambda candidate: calls.append(candidate),
    )
    return campaign, alignment, row, calls


def test_production_publication_is_complete_replayable_and_cannot_overwrite_other_policy(
    positioned_campaign,
):
    c, _, original, calls = positioned_campaign
    result = position_campaign(c)
    output = c.output / "augmented"
    row = json.loads((output / "dataset.jsonl").read_text())
    assert {k: row[k] for k in original} == original
    assert len(calls) == 1 and result["records"] == 1
    assert result["datasetSha256"] == digest((output / "dataset.jsonl").read_bytes())
    assert position_campaign(c)["datasetSha256"] == result["datasetSha256"]
    c.config["positions"]["seed"] += 1
    with pytest.raises((ValueError, FileExistsError)):
        position_campaign(c)


def test_source_geometry_provenance_and_alignment_are_both_checked(positioned_campaign):
    c, alignment, _, _ = positioned_campaign
    data = c.root / "data"
    load_page_geometry(c.root, data, {"doc_source": alignment})
    broken = deepcopy(alignment)
    broken["lines"][0]["xy"][0] += 1
    with pytest.raises(ValueError, match="coordinate"):
        load_page_geometry(c.root, data, {"doc_source": broken})
    with (c.root / "paddle" / "lines.parquet").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="geometry hash"):
        load_page_geometry(c.root, data, {"doc_source": alignment})


@pytest.mark.parametrize("statistics", [True, False])
def test_geometry_stream_selects_exact_rows_with_or_without_statistics(tmp_path, statistics):
    rows = [dict(document_id=f"pdf_{i:03}", value=i) for i in range(9)]
    # Deliberately unsorted, with wanted/unwanted IDs sharing row groups.
    rows = rows[::2] + rows[1::2]
    path = tmp_path / "regions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2, write_statistics=statistics)
    ids = ["pdf_000", "pdf_003", "pdf_008", "pdf_999"]
    assert list(_geometry_rows(path, ids)) == [r for r in rows if r["document_id"] in ids]
    assert list(_geometry_rows(path, [])) == []
