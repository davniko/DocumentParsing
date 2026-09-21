from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import lexical_partitions as parts
from document_ocr.synthesis.template_compiler.descendant import _render_target_binding
from document_ocr.synthesis.template_compiler.request_batches import (
    lexical_payload,
    semantic_aliases,
)

PATH = "documentPatch.cargoGroups[0].description"


def binding(mode="token_projected_surface"):
    words = "GENSET SWEK KIT MODEL C1675 OPEN".lower().split()
    intervals = ((0, 1), (1, 3), (3, 6), (0, 1), (1, 3), (3, 6))
    if mode == "segmented_surface":
        intervals = intervals[:3]
    return NS(
        logical_key="cargo",
        target_paths=(PATH,),
        realization=NS(
            mode=mode,
            target_values=(NS(source_value="GENSET SWEK KIT MODEL:C1675 OPEN"),),
            slots=tuple(
                NS(required_target_prefix_tokens=words[:a], required_target_suffix_tokens=words[b:])
                for a, b in intervals
            ),
        ),
        occurrences=tuple(
            NS(slot_id=f"s{i}", source_text=" ".join(words[a:b]).upper())
            for i, (a, b) in enumerate(intervals)
        ),
    )


@pytest.mark.parametrize("mode", ["token_projected_surface", "segmented_surface"])
def test_item_names_are_owned_fragments_not_weighted_words(mode):
    b = binding(mode)
    plan = parts.partition(b)
    aux = dict(
        zip(
            (plan.key(i) for i in sorted(plan.mutable)),
            ("INDUSTRIAL GENERATOR", "ENGINE SERVICE KIT", "MODEL:ZR6840 OPEN"),
            strict=True,
        )
    )
    target = parts.assembled_targets(NS(bindings=[b]), aux)[PATH]
    assert target == "INDUSTRIAL GENERATOR ENGINE SERVICE KIT MODEL:ZR6840 OPEN"
    output = _render_target_binding(
        b, {"documentPatch": {"cargoGroups": [{"description": target}]}}, auxiliary_values=aux
    )
    assert output.replacements["s0"] == "INDUSTRIAL GENERATOR"
    assert output.replacements["s1"] == "ENGINE SERVICE KIT"
    assert output.replacements["s2"] == "MODEL:ZR6840 OPEN"
    if mode.startswith("token"):
        assert output.replacements["s3"] == output.replacements["s0"]
    with pytest.raises(ValueError, match="weighted splitting is forbidden"):
        parts.render_parts(plan, target, {})
    with pytest.raises(ValueError, match="frozen target"):
        parts.render_parts(plan, target + " UNREQUESTED", aux)


def test_proven_unowned_gap_is_retained_once():
    b = binding()
    b.realization.target_values = (NS(source_value="GENSET P/N MODEL"),)
    b.realization.slots = (
        NS(required_target_prefix_tokens=(), required_target_suffix_tokens=("p", "n", "model")),
        NS(required_target_prefix_tokens=("genset", "p", "n"), required_target_suffix_tokens=()),
    )
    b.occurrences = (NS(slot_id="s0", source_text="GENSET"), NS(slot_id="s1", source_text="MODEL"))
    p = parts.partition(b)
    assert p.mutable == {0, 2}
    aux = {p.key(0): "NEW DIESEL GENERATOR", p.key(2): "XR900"}
    assert p.assemble(aux) == "NEW DIESEL GENERATOR P/N XR900"
    assert parts.render_parts(p, p.assemble(aux), aux) == {
        "s0": "NEW DIESEL GENERATOR",
        "s1": "XR900",
    }


def test_unproven_segment_partition_is_rejected():
    b = binding("segmented_surface")
    b.occurrences[0].source_text = "UNRELATED ITEM"
    with pytest.raises(ValueError, match="source-token partition"):
        parts.partition(b)


def test_labelled_part_numbers_repeat_by_identity_and_preserve_printed_labels():
    b = binding("agent_required")
    original = "PANEL; PART.NO:A100; C/P.NO:B200"
    b.realization.target_values = (NS(source_value=original),)
    raw = b"PANEL\nPART.NO:A100\nC/P.NO:B200\nPART.NO:A100\n"
    spans = ((0, 5, "PANEL"), (14, 18, "A100"), (26, 30, "B200"), (39, 43, "A100"))
    b.occurrences = tuple(
        NS(slot_id=f"s{i}", source_text=value, byte_start=start, byte_end=end)
        for i, (start, end, value) in enumerate(spans)
    )
    # Compute fixture offsets from exact occurrences, including repeated values.
    cursor = 0
    for slot in b.occurrences:
        slot.byte_start = raw.index(slot.source_text.encode(), cursor)
        slot.byte_end = slot.byte_start + len(slot.source_text)
        cursor = slot.byte_end
    plan = parts.partition(b)
    assert plan is not None
    aux = {plan.key(i): v for i, v in enumerate(("LCD PANEL", "R301", "S402"))}
    target = plan.assemble(aux)
    assert target == "LCD PANEL; PART.NO:R301; C/P.NO:S402"
    assert parts.render_parts(plan, target, aux, source=raw) == {
        "s0": "LCD PANEL",
        "s1": "R301",
        "s2": "S402",
        "s3": "R301",
    }
    with pytest.raises(ValueError, match="pinned source"):
        parts.render_parts(plan, target, aux)
    with pytest.raises(ValueError, match="label is not printed"):
        parts.render_parts(plan, target, aux, source=raw.replace(b"PART.NO:", b"WRONGXX:"))


def test_reordered_repeated_fragments_have_exact_shared_ownership():
    b = binding("agent_required")
    original = "DISPLAY PARTS; OPEN CELL X100; OPEN CELL Y200"
    b.realization.target_values = (NS(source_value=original),)
    source_parts = ("OPEN CELL", "X100", "OPEN CELL", "Y200", "DISPLAY PARTS", "X100")
    b.occurrences = tuple(NS(slot_id=f"s{i}", source_text=v) for i, v in enumerate(source_parts))
    plan = parts.partition(b)
    assert plan is not None
    assert plan.key(1) == plan.key(3)
    source_aux = {plan.key(i): value for i, value in enumerate(plan.source_parts)}
    assert plan.assemble(source_aux) == original
    aux = {
        plan.key(0): "LCD COMPONENTS",
        plan.key(1): "PANEL MODULE",
        plan.key(2): "R400",
        plan.key(4): "S500",
    }
    target = "LCD COMPONENTS; PANEL MODULE R400; PANEL MODULE S500"
    assert plan.assemble(aux) == target
    assert parts.render_parts(plan, target, aux) == {
        "s0": "PANEL MODULE",
        "s1": "R400",
        "s2": "PANEL MODULE",
        "s3": "S500",
        "s4": "LCD COMPONENTS",
        "s5": "R400",
    }
    with pytest.raises(ValueError, match="frozen target"):
        parts.render_parts(plan, target.replace("S500", "MISSING"), aux)


@pytest.mark.parametrize(
    "surfaces",
    [
        ("DISPLAY", "OPEN CELL"),  # Missing lexical coverage.
        ("DISPLAY PARTS", "PARTS OPEN CELL", "OPEN CELL"),  # Overlap.
        ("DISPLAY PARTS", "UNRELATED"),
    ],
)
def test_unproven_agent_fragment_graph_remains_residual(surfaces):
    b = binding("agent_required")
    b.realization.target_values = (NS(source_value="DISPLAY PARTS OPEN CELL"),)
    b.occurrences = tuple(NS(slot_id=f"s{i}", source_text=v) for i, v in enumerate(surfaces))
    assert parts.partition(b) is None


def test_ocr_wrap_within_a_word_does_not_break_item_ownership():
    b = binding("segmented_surface")
    b.occurrences[0].source_text = "GEN\nSET"
    assert parts.partition(b).intervals == ((0, 1), (1, 3), (3, 6))
    b.occurrences[0].source_text = "GEN"
    b.occurrences[1].source_text = "SET SWEK KIT"
    with pytest.raises(ValueError, match="semantic token boundary"):
        parts.partition(b)


def test_static_closing_bracket_is_present_in_label_but_not_duplicated_in_slot():
    b = binding("segmented_surface")
    b.realization.target_values = (NS(source_value="GENSET SWEK KIT (MODEL:C1675 OPEN)"),)
    b.occurrences[2].source_text = "(MODEL:C1675 OPEN"
    p = parts.partition(b)
    source = b"GENSET\nSWEK KIT\n(MODEL:C1675 OPEN)"
    b.occurrences[0].byte_start = 0
    b.occurrences[0].byte_end = 6
    b.occurrences[1].byte_start = 7
    b.occurrences[1].byte_end = 15
    b.occurrences[2].byte_start = 16
    b.occurrences[2].byte_end = len(source) - 1
    assert parts.boundary_punctuation(p, 2, source) == ("", ")")
    aux = {p.key(0): "NEW GENERATOR", p.key(1): "SERVICE KIT", p.key(2): "(MODEL:ZR6840 OPEN)"}
    rendered = parts.render_parts(p, p.assemble(aux), aux, source=source)
    assert rendered["s2"] == "(MODEL:ZR6840 OPEN"
    assert rendered["s2"] + ")" == "(MODEL:ZR6840 OPEN)"
    aux[p.key(2)] = "(MODEL:ZR6840 OPEN"
    with pytest.raises(ValueError, match="boundary punctuation"):
        parts.render_parts(p, p.assemble(aux), aux, source=source)


def test_batch_payload_exposes_fragment_ownership_without_label_fields():
    fields = [
        dict(
            key=f"f{i}",
            paths=[],
            auxiliaryKey=f"part{i}",
            constraints=[],
            cargoFragment=dict(targetPaths=[PATH], index=i),
        )
        for i in range(2)
    ]
    payload = dict(
        requestedFields=fields,
        structuredScenario=dict(documentPatch=dict(cargoGroups=[dict(description="SOURCE")])),
    )
    aliases = semantic_aliases(payload, [f["key"] for f in fields])
    compact = lexical_payload(payload, aliases)
    assert compact["structuredScenario"]["documentPatch"]["cargoGroups"][0]["description"] == {
        "generateParts": list(aliases.values())
    }
    assert (
        payload["structuredScenario"]["documentPatch"]["cargoGroups"][0]["description"] == "SOURCE"
    )
