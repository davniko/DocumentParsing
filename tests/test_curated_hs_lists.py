from copy import deepcopy
from dataclasses import replace

import pytest

from document_ocr.synthesis.curated import Occurrence, SourceContract, TargetBinding, Variable
from document_ocr.synthesis.curated_hs_lists import blocks, sample
from document_ocr.synthesis.curated_templates import (
    RenderRegion,
    SamplingBlueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.generators import DeterministicStream

PATH = "documentPatch.goodsItemDetails[0].hsCodes"


def blueprint():
    codes = ["8528.59.00.00.11", "6109.10.00.00.22", "5205:22", "290522"]
    a, b = "\n".join(codes[:2]), "; ".join(codes[2:])
    text = "PAGE 1\nHS:\n" + a + "\nPAGE 2\nHS CONTINUED:\n" + b + "\nBL123\nEND"
    regions, bindings = [], {}
    for key, value, indices in (("first", a, [0, 1]), ("second", b, [2, 3])):
        start = len(text[: text.index(value)].encode())
        paths = tuple(f"{PATH}[{i}]" for i in indices)
        regions.append(
            RenderRegion(key, 0, start, start + len(value.encode()), value, paths, "identifier")
        )
        bindings[key] = {"occurrences": [{"source_text": value}], "target_paths": list(paths)}
    start = text.index("BL123")
    regions.append(
        RenderRegion(
            "bill",
            0,
            start,
            start + 5,
            "BL123",
            ("documentPatch.billOfLadingNumber",),
            "identifier",
            "bill",
            baseline="BL123",
        )
    )
    return SamplingBlueprint(
        "hs-list",
        text,
        {
            "documentPatch": {
                "billOfLadingNumber": "BL123",
                "goodsItemDetails": [{"hsCodes": codes}],
            }
        },
        SourceContract(
            variables=[
                Variable(
                    key="bill",
                    kind="identifier",
                    value="BL123",
                    meaning="bill",
                    required_literals=[],
                    occurrences=[Occurrence(text="BL123", occurrence=1, presentation="text")],
                )
            ],
            targets=[TargetBinding(path="documentPatch.billOfLadingNumber", expression="{bill}")],
            fixed_context="Reviewed HS list",
        ),
        tuple(regions),
        bindings,
        {},
        (),
        {
            "hs_list_blocks": [
                {"key": "first", "sourceIndices": [0, 1]},
                {"key": "second", "sourceIndices": [2, 3]},
            ]
        },
    )


@pytest.mark.parametrize("count", [2, 3, 4, 6, 12])
def test_variable_lists_render_every_target_and_preserve_other_bytes(count):
    bp = blueprint()
    stream = DeterministicStream(1, "hs-lists", "test")
    codes = sample(bp, [str(610910 + i) for i in range(count)], stream)
    assert codes == sample(bp, [str(610910 + i) for i in range(count)], stream)
    target = deepcopy(bp.target)
    target["documentPatch"]["goodsItemDetails"][0]["hsCodes"] = codes
    text, output, proof = render_sampling_blueprint(bp, target, {})
    assert output == target
    assert len(codes) == count and all(code in text for code in codes)
    assert (
        text.startswith("PAGE 1\nHS:\n")
        and "\nPAGE 2\nHS CONTINUED:\n" in text
        and text.endswith("\nEND")
    )
    assert proof["changedTargetPaths"] == proof["coveredTargetPaths"]
    assert proof["unchangedBytesPreserved"]


def test_source_replay_is_exact():
    bp = blueprint()
    text, target, proof = render_sampling_blueprint(bp, bp.target, {})
    assert text == bp.source and target == bp.target and proof["edits"] == []


def test_cannot_drop_printed_blocks_or_invent_unrelated_fields():
    bp = blueprint()
    with pytest.raises(ValueError, match="every printed page"):
        sample(bp, ["610910"], DeterministicStream(1, "hs", "test"))
    target = deepcopy(bp.target)
    target["documentPatch"]["unexpected"] = "X"
    with pytest.raises(ValueError, match="topology"):
        render_sampling_blueprint(bp, target, {})


def test_incomplete_ownership_and_wrong_masks_fail():
    bp = blueprint()
    bad = replace(
        bp, ownership_data={"hs_list_blocks": [{"key": "first", "sourceIndices": [0, 1]}]}
    )
    with pytest.raises(ValueError, match="every source code"):
        blocks(bad)
    target = deepcopy(bp.target)
    target["documentPatch"]["goodsItemDetails"][0]["hsCodes"][0] = "852859000011"
    with pytest.raises(ValueError, match="code mask"):
        render_sampling_blueprint(bp, target, {})


def test_ordinary_scalar_templates_do_not_gain_variable_arrays():
    bp = replace(blueprint(), ownership_data={})
    target = deepcopy(bp.target)
    target["documentPatch"]["goodsItemDetails"][0]["hsCodes"].pop()
    with pytest.raises(ValueError, match="topology"):
        render_sampling_blueprint(bp, target, {})
