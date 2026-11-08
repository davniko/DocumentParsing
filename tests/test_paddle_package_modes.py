import pytest

from document_ocr.spatial_inputs.paddle import RegionSpan, pages_from_record
from document_ocr.synthesis.paddle_bindings import PaddleReplacement
from document_ocr.synthesis.paddle_package_modes import validate_package_mode_owners
from document_ocr.synthesis.paddle_render import consumed_native_extra, project_sampled_lines


def test_nested_projection_keeps_every_group_not_only_first():
    assert project_sampled_lines("a\nb\nc", 1, stride=3) == "b"
    assert project_sampled_lines("a\nb\nc\nd\ne\nf", 1, stride=3) == "b\ne"
    assert project_sampled_lines("a\nb\nc\nd\ne\nf", 0, stride=3) == "a\nd"
    with pytest.raises(ValueError, match="stride"):
        project_sampled_lines("a\nb\nc\nd", 1, stride=3)
    with pytest.raises(ValueError, match="stride"):
        project_sampled_lines("a\nb\nc", 3, stride=3)


def test_compound_admission_rejects_only_scalar_coverage():
    contract = {"declarations": [{"start": 0, "end": 10, "source": "24 PALLETS"}]}
    template = {"overlay": {"overrides": []}}
    transfer = {"slots": [{"owner": "count@0:2", "status": "exact_measured_owner_candidate"}]}
    with pytest.raises(ValueError, match="complete owner"):
        validate_package_mode_owners("source", contract, template, transfer)
    template["overlay"]["overrides"].append(
        {"key": "declared-packages:0", "byteStart": 0, "byteEnd": 10}
    )
    validate_package_mode_owners("source", contract, template, transfer)


@pytest.mark.parametrize("alternative", [False, True])
def test_exact_absence_is_not_a_missing_binding(alternative):
    member = {"key": "declared-packages:0", "byteStart": 0, "before": "24 PALLETS"}
    decision = (
        {
            "alternatives": [[member], [{"key": "count", "byteStart": 0, "before": "24"}]],
            "reason": "missing Paddle occurrence",
        }
        if alternative
        else {**member, "reason": "missing Paddle occurrence"}
    )
    validate_package_mode_owners(
        "source",
        {"declarations": [{"start": 0, "end": 10, "source": "24 PALLETS"}]},
        {"overlay": {"overrides": [], "absentEdits": [decision]}},
        {"slots": []},
    )


def setup_owner():
    pages = pages_from_record(
        {
            "pages": [
                {
                    "index": 0,
                    "width": 100,
                    "height": 100,
                    "regions": [{"index": 0, "text": "24 PALLETS OTHER", "box": [0, 0, 90, 10]}],
                }
            ]
        }
    )
    owner = "declared-packages:0@0:10"
    logical = {
        owner: PaddleReplacement(
            owner, (RegionSpan(0, 0, 0, 10, "24 PALLETS"),), "13 PALLETS + 16 WOODEN PALLETS"
        )
    }
    extra = {"owner": "count", "spans": [[0, 0, 0, 2]], "consumedBy": [owner]}
    return pages, logical, extra


def test_consumed_scalar_extra_is_explicit_and_mode_scoped():
    pages, logical, extra = setup_owner()
    assert consumed_native_extra(extra, logical, pages)
    assert not consumed_native_extra(extra, {}, pages)
    assert not consumed_native_extra({"spans": [[0, 0, 0, 2]]}, logical, pages)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"target": "/a"}, "target projection"),
        ({"targets": []}, "target projection"),
        ({"spans": [[0, 0, 11, 16]]}, "does not contain"),
        ({"consumedBy": ["declared-packages:0@0:10"] * 2}, "distinct"),
        ({"consumedBy": [{}]}, "distinct"),
        ({"consumedBy": [""]}, "distinct"),
    ],
)
def test_consumption_never_suppresses_unowned_text_or_labels(change, message):
    pages, logical, extra = setup_owner()
    with pytest.raises(ValueError, match=message):
        consumed_native_extra({**extra, **change}, logical, pages)
