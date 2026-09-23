from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import transport_derivations as transport
from document_ocr.synthesis.template_compiler.host import SpanDraft


def binding(key="feeder"):
    return NS(
        logical_key=key,
        derivation="sampled_auxiliary_vessel",
        target_paths=(),
        dependency_paths=("documentPatch.transport.vesselName",),
        dependency_bindings=(),
        render_mode="deterministic_derived",
        group_kind="transport",
        value_kind="equipment",
        occurrences=(NS(source_text="Old Feeder"),),
    )


def test_registry_sampling_is_repeatable_distinct_and_does_not_extend_labels():
    target = {"documentPatch": {"transport": {"vesselName": "Main Vessel"}}}
    arguments = dict(
        vessel_names=["Main Vessel", "Old Feeder", "New Dawn", "New Breeze"],
        target=target,
        seed=42,
        sample_id="s",
    )
    values = transport.values([binding("one"), binding("two")], **arguments)
    assert set(values.values()) == {"New Dawn", "New Breeze"}
    assert values == transport.values([binding("two"), binding("one")], **arguments)
    assert target == {"documentPatch": {"transport": {"vesselName": "Main Vessel"}}}
    with pytest.raises(ValueError, match="lacks distinct"):
        transport.values(
            [binding()], **{**arguments, "vessel_names": ["Main Vessel", "Old Feeder"]}
        )


def test_untyped_or_target_backed_vessels_are_not_treated_as_independent():
    b = binding()
    b.target_paths = ("documentPatch.transport.vesselName",)
    with pytest.raises(ValueError, match="explicit source-only"):
        transport.validate(b)


def test_printed_private_leg_can_have_only_main_voyage_in_extraction_labels():
    b = binding()
    b.dependency_paths = ("documentPatch.transport",)
    target = {"documentPatch": {"transport": {"voyageNumber": "093W"}}}
    transport.validate_source([b], target)
    arguments = dict(vessel_names=["Old Feeder", "New Dawn"], target=target, seed=42, sample_id="s")
    assert transport.values([b], **arguments) == {"feeder": "New Dawn"}
    assert target == {"documentPatch": {"transport": {"voyageNumber": "093W"}}}
    with pytest.raises(ValueError, match="transport owner"):
        transport.validate_source([b], {"documentPatch": {"transport": {}}})
    with pytest.raises(ValueError, match="main vessel is invalid"):
        transport.validate_source([b], {"documentPatch": {"transport": {"vesselName": ""}}})


def test_source_contract_cannot_hide_a_missing_or_aliased_main_vessel():
    with pytest.raises(ValueError, match="target-backed main"):
        transport.validate_source([binding()], {"documentPatch": {}})
    with pytest.raises(ValueError, match="alias the main"):
        transport.validate_source(
            [binding()], {"documentPatch": {"transport": {"vesselName": "Old Feeder"}}}
        )
    with pytest.raises(ValueError, match="target-backed main"):
        transport.values(
            [binding()], vessel_names=["New"], target={"documentPatch": {}}, seed=1, sample_id="s"
        )
    b = binding()
    b.group_kind = "route"
    with pytest.raises(ValueError, match="explicit source-only"):
        transport.validate(b)


def draft(raw, name):
    start = raw.index(name)
    return SpanDraft(
        draft_id="d",
        logical_key="feeder",
        render_mode="agent_required",
        value_kind="equipment",
        group_kind="transport",
        group_key="precarriage",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(name),
        source_text=name,
        evidence_origin="compiler",
        render_policy="natural_text",
        rationale="source",
    )


@pytest.mark.parametrize("caption", ["Feeder Vessel\n", "First Leg Vessel ", "INTENDED VESSEL : "])
def test_explicit_independent_caption_compiles_to_registry_contract(caption):
    raw = caption + "NEW DAWN V.013W"
    source = {"documentPatch": {"transport": {"vesselName": "MAIN SHIP"}}}
    (value,) = transport.normalize_explicit_names(
        raw=raw, drafts=[draft(raw, "NEW DAWN")], source_target=source
    )
    assert value.derivation == "sampled_auxiliary_vessel"
    assert value.target_paths == ()
    transport.validate(value)
    assert transport.normalize_explicit_names(raw=raw, drafts=[value], source_target=source) == (
        value,
    )


@pytest.mark.parametrize(
    "raw,name,main",
    [
        ("Pre-Carriage by\nNEW DAWN 013W", "NEW DAWN 013W", "MAIN SHIP"),
        ("Pre-Carriage by\nNEW DAWN(EG)", "NEW DAWN(EG)", "MAIN SHIP"),
        ("Place of Receipt\nNEW DAWN", "NEW DAWN", "MAIN SHIP"),
        ("Pre-Carriage by\nNEW DAWN", "NEW DAWN", "MAIN SHIP"),
        ("Pre-Carriage by\nNEW DAWN", "NEW DAWN", "New Dawn"),
        ("Pre-Carriage by\nNEW DAWN", "NEW DAWN", None),
    ],
)
def test_incomplete_or_ambiguous_vessel_evidence_is_not_promoted(raw, name, main):
    original = draft(raw, name)
    assert transport.normalize_explicit_names(
        raw=raw,
        drafts=[original],
        source_target={"documentPatch": {"transport": {"vesselName": main}}},
    ) == (original,)
