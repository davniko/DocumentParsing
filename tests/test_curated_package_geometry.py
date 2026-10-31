from types import SimpleNamespace

import numpy as np
import pytest

from document_ocr.synthesis.curated_package_geometry import package_regions
from document_ocr.synthesis.curated_reflow import ReflowContext, run_layout


class Metrics:
    def length(self, text):
        return len(text)


def fixture(after="CTNS", unknown=False):
    source = "10 CARTONS\nNEXT ROW"
    rendered = f"20 {after}\nNEXT ROW"
    proof = {
        "edits": [
            {
                "key": "count",
                "occurrence": 0,
                "byteStart": 0,
                "byteEnd": 2,
                "before": "10",
                "after": "20",
            },
            {
                "key": "noun",
                "occurrence": 0,
                "byteStart": 3,
                "byteEnd": 10,
                "before": "CARTONS",
                "after": after,
            },
        ],
        "packagePresentations": {
            "slots": [{"key": "noun", "occurrence": 0, "before": "CARTONS", "after": after}],
            "nonScalarOwners": [],
        },
    }
    lines = []
    boxes = []
    for i, text in enumerate(source.splitlines()):
        box = [100, 100 + i * 30, 200, 110 + i * 30]
        boxes.append(box)
        lines.append(
            {
                "line_number": i + 1,
                "page_index": 0,
                "original_text": text,
                "xy": None if unknown and i == 0 else [150, 105 + i * 30],
                "region_indices": [i],
                "regions": [{"index": i, "text": text, "bbox": box}],
            }
        )
    alignment = {"lines": lines}
    receipt = {
        "lines": [
            {
                "line_number": r["line_number"],
                "page_index": 0,
                "text": text,
                "xy": r["xy"],
                "source_lines": [r["line_number"]],
            }
            for r, text in zip(lines, rendered.splitlines(), strict=True)
        ]
    }
    context = ReflowContext(
        "example",
        rendered,
        receipt,
        alignment,
        {0: np.array(boxes, dtype=float)},
        {"0": [1000, 1000]},
        [],
        source,
        proof,
    )
    return context


def test_complete_region_includes_quantity_and_alias():
    c = fixture()
    rows, reasons = package_regions(c.source, c.package_proof, c.alignment)
    assert not reasons
    assert rows[0]["after"] == "20 CTNS"


@pytest.mark.parametrize(
    "source_text,regions",
    [
        (
            "S.T.C 1 PALLET",
            [
                ("S.T.C", [100, 100, 150, 110]),
                ("1", [160, 100, 166, 110]),
                ("PALLET", [180, 100, 220, 110]),
            ],
        ),
        ("1", [("1", [100, 100, 106, 110])]),
    ],
)
def test_complete_declared_list_uses_owned_line_and_real_column_space(source_text, regions):
    from document_ocr.synthesis.curated_reflow import groups_for

    c = fixture()
    c.source = source_text + "\nNEXT ROW"
    replacement = "14 PALLETS + 92 FIBREBOARD BOXES"
    before = "1 PALLET" if "PALLET" in source_text else "1"
    start = source_text.index(before)
    c.rendered = source_text.replace(before, replacement) + "\nNEXT ROW"
    c.receipt["lines"][0]["text"] = c.rendered.splitlines()[0]
    line = c.alignment["lines"][0]
    line.update(
        original_text=source_text,
        method="spatial_group",
        regions=[{"index": 10 + i, "text": t, "bbox": b} for i, (t, b) in enumerate(regions)],
        region_indices=list(range(10, 10 + len(regions))),
    )
    c.boxes[0] = np.array([b for _, b in regions] + [[100, 130, 200, 140], [350, 100, 450, 110]])
    # A foreign column remains a hard boundary even when the count/type split
    # is reflowed as one complete logical line.
    c.alignment["lines"].append(
        {
            "line_number": 3,
            "page_index": 0,
            "original_text": "WEIGHT",
            "xy": [400, 105],
            "region_indices": [20],
            "regions": [{"index": 20, "text": "WEIGHT", "bbox": [350, 100, 450, 110]}],
        }
    )
    c.receipt["lines"].append(
        {"line_number": 3, "page_index": 0, "text": "WEIGHT", "xy": [400, 105], "source_lines": [3]}
    )
    c.source += "\nWEIGHT"
    c.rendered += "\nWEIGHT"
    c.package_proof = {
        "declaredPackageLists": {"verified": True},
        "edits": [
            {
                "key": "declared-packages:0",
                "occurrence": 0,
                "byteStart": start,
                "byteEnd": start + len(before),
                "before": before,
                "after": replacement,
            }
        ],
        "packagePresentations": {
            "slots": [],
            "nonScalarOwners": [
                {"key": "declared-packages:0", "occurrence": 0, "mode": "compound"}
            ],
        },
    }
    c.groups = groups_for(c.source, c.package_proof, c.receipt, c.alignment)
    calibration = SimpleNamespace(
        pitch_ratio_range=(1.1, 1.3), clearance_height_ratio=0.2, minimum_page_median_height=1
    )
    policy = SimpleNamespace(
        seed=1,
        row_lock_height_fraction=1 / 3,
        bottom_guard_height_fraction=1,
        column_guard_height_fraction=0.5,
    )
    coords, pages, text = run_layout(c, calibration, policy, Metrics())
    node = next(n for n in pages[0]["nodes"] if "lineBoxes" in n)
    assert node["regionIndices"] == list(range(10, 10 + len(regions)))
    assert node["box"][2] < 350
    assert coords[1] is not None and coords[3] is not None
    assert replacement in text


def test_repeated_digits_are_resolved_by_owned_line_offset():
    c = fixture()
    c.alignment["lines"][0]["original_text"] = "10 CARTONS 10 KG"
    c.alignment["lines"][0]["regions"][0]["text"] = "10 CARTONS 10 KG"
    rows, reasons = package_regions(c.source, c.package_proof, c.alignment)
    assert not reasons
    assert rows[0]["after"] == "20 CTNS 10 KG"


@pytest.mark.parametrize(
    "neighbour,accepted",
    [
        ([100, 90, 200, 102], True),
        ([100, 108, 200, 120], True),
        ([100, 98, 200, 107], False),
        ([120, 101, 180, 109], False),
    ],
)
def test_declared_list_preserves_edge_neighbour_not_interior_text(neighbour, accepted):
    from document_ocr.synthesis.curated_reflow import groups_for, propose_groups

    c = fixture()
    c.package_proof["declaredPackageLists"] = {"verified": True}
    c.package_proof["edits"] = [
        {
            "key": "declared-packages:0",
            "occurrence": 0,
            "byteStart": 0,
            "byteEnd": 10,
            "before": "10 CARTONS",
            "after": "4 CTNS + 16 BAGS",
        }
    ]
    c.rendered = "4 CTNS + 16 BAGS\nNEXT ROW"
    c.receipt["lines"][0]["text"] = "4 CTNS + 16 BAGS"
    c.boxes[0] = np.vstack([c.boxes[0], neighbour])
    c.groups = groups_for(c.source, c.package_proof, c.receipt, c.alignment)
    calibration = SimpleNamespace(pitch_ratio_range=(1.1, 1.3))
    policy = SimpleNamespace(seed=1, column_guard_height_fraction=0.5)
    replacements, reasons = propose_groups(c, 0, calibration, policy, Metrics())
    assert bool(replacements) is accepted
    if accepted:
        assert replacements[0]["regionIndices"] == [0]
    else:
        assert any(r["reason"] == "foreign_text_inside_owner_envelope" for r in reasons)


def test_unknown_source_anchor_is_not_invented():
    c = fixture(unknown=True)
    rows, reasons = package_regions(c.source, c.package_proof, c.alignment)
    assert rows == []
    assert reasons and all(not r["blocking"] for r in reasons)


@pytest.mark.parametrize(
    "original,printed,old,offset,expected",
    [
        ("40.0000 CBM", "40,0000 CBM", "40.0000", 0, "40,0000"),
        ("(38 PACKAGES)", "(38 PACKAGES", "(38", 0, "(38"),
        ("10 PKGS 1.000 KG 10 PKGS 2.000 KG", "10 PKGS 2,000 KG", "10", 17, "10"),
    ],
)
def test_full_region_punctuation_join_preserves_subspan_ownership(
    original, printed, old, offset, expected
):
    from document_ocr.synthesis.curated_package_geometry import fragment_matches

    line = {"original_text": original, "regions": [{"text": printed, "index": 8}]}
    region, start, end = fragment_matches(line, old, offset)[0]
    assert region["text"][start:end] == expected


def test_complete_punctuation_unique_line_keeps_its_certified_region():
    from document_ocr.synthesis.curated_package_geometry import fragment_matches

    line = {
        "method": "punctuation_unique",
        "original_text": "PACKING: 800 BAGS.",
        "regions": [{"text": "PACKING:800 BAGS", "index": 8}],
    }
    assert fragment_matches(line, "PACKING: 800 BAGS.", 0) == [(line["regions"][0], 0, 16)]
    line["regions"][0]["text"] = "PACKING:900 BAGS"
    assert fragment_matches(line, "PACKING: 800 BAGS.", 0) == []


def test_package_count_does_not_capture_same_digits_inside_equipment_region():
    from document_ocr.synthesis.curated_package_geometry import fragment_matches

    line = {
        "original_text": "20 BAGS /FCL/FCL /20GP/18300.000KGS;26.600CBM",
        "regions": [
            {"text": "20", "index": 1},
            {"text": "BAGS", "index": 2},
            {"text": "/FCL/FCL/20GP/18300.000KGS26.600CBM", "index": 3},
        ],
    }
    assert fragment_matches(line, "20", 0) == [(line["regions"][0], 0, 2)]


@pytest.mark.parametrize("printed", ["40'HC", "40 HC", "40`HC"])
def test_certified_equipment_quote_variants_join_without_changing_digits(printed):
    from document_ocr.synthesis.curated_package_geometry import fragment_matches

    line = {
        "method": "punctuation_unique",
        "original_text": "1 X 40\u2019HC 20 PACKAGES",
        "regions": [{"text": f"1 X {printed} 20 PACKAGES", "index": 8}],
    }
    region, start, end = fragment_matches(line, "40\u2019HC", 4)[0]
    assert region["text"][start:end] == printed


@pytest.mark.parametrize("after", ["CTNS", "FIBREBOARD CARTON(S)"])
def test_same_line_alias_projects_center_and_wraps_in_source_column(after):
    c = fixture(after)
    calibration = SimpleNamespace(
        pitch_ratio_range=(1.1, 1.3), clearance_height_ratio=0.2, minimum_page_median_height=1
    )
    policy = SimpleNamespace(row_lock_height_fraction=1 / 3, bottom_guard_height_fraction=1)
    coords, pages, text = run_layout(c, calibration, policy, Metrics())
    assert pages[0]["status"] == "accepted"
    node = next(n for n in pages[0]["nodes"] if "horizontalRegion" in n)
    assert node["box"][0] == 100 and node["box"][2] <= 200
    assert coords[1] != [150, 105]
    assert after in text
    assert (node["virtualLines"] > 1) == (len(after) > 7)


def test_ambiguous_partial_region_does_not_guess():
    c = fixture()
    line = c.alignment["lines"][0]
    line["regions"][0]["text"] = "XX 10 CARTONS 10"
    _, reasons = package_regions(c.source, c.package_proof, c.alignment)
    assert any(
        r["blocking"] and r["reason"] == "ambiguous_repeated_package_region" for r in reasons
    )


def test_stacked_ocr_rows_in_one_logical_line_use_complete_owned_block():
    from document_ocr.synthesis.curated_package_geometry import merged_package_lines
    from document_ocr.synthesis.curated_reflow import groups_for

    c = fixture()
    line = c.alignment["lines"][0]
    line["method"] = "spatial_group"
    line["regions"] = [
        {"index": 0, "text": "10 CARTONS", "bbox": [100, 100, 200, 110]},
        {"index": 2, "text": "PARTS", "bbox": [100, 112, 150, 122]},
    ]
    line["region_indices"] = [0, 2]
    line["original_text"] = "10 CARTONS PARTS"
    c.source = "10 CARTONS PARTS\nNEXT ROW"
    c.rendered = "20 CTNS MACHINE PARTS\nNEXT ROW"
    c.receipt["lines"][0]["text"] = "20 CTNS MACHINE PARTS"
    c.package_proof["edits"].append(
        {
            "key": "product",
            "occurrence": 0,
            "byteStart": 11,
            "byteEnd": 16,
            "before": "PARTS",
            "after": "MACHINE PARTS",
        }
    )
    c.boxes[0] = np.array([[100, 100, 200, 110], [100, 130, 200, 140], [100, 112, 150, 122]])
    c.groups = groups_for(c.source, c.package_proof, c.receipt, c.alignment)
    assert merged_package_lines(c.source, c.package_proof, c.alignment) == {1}
    calibration = SimpleNamespace(
        pitch_ratio_range=(1.1, 1.3), clearance_height_ratio=0.2, minimum_page_median_height=1
    )
    policy = SimpleNamespace(
        seed=1,
        row_lock_height_fraction=1 / 3,
        bottom_guard_height_fraction=1,
        column_guard_height_fraction=0.5,
    )
    coords, pages, _ = run_layout(c, calibration, policy, Metrics())
    node = next(n for n in pages[0]["nodes"] if "lineBoxes" in n)
    assert node["regionIndices"] == [0, 2]
    assert node["box"][2] <= 200
    assert coords[1] != [150, 105]


def test_owned_suffix_with_missing_paddle_closing_parenthesis():
    from document_ocr.synthesis.curated_package_geometry import fragment_matches

    region = {"text": "TOTAL 252 PKGS (TWO HUNDRED FIFTY"}
    line = {
        "method": "punctuation_unique",
        "original_text": "TOTAL 252 PKGS (TWO HUNDRED FIFTY)",
        "regions": [region],
    }
    start = line["original_text"].index("(")
    assert fragment_matches(line, "(TWO HUNDRED FIFTY)", start) == [
        (region, start, len(region["text"]))
    ]
