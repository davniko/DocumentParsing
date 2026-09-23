from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as Node

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.template_compiler import complete_targets as targets
from document_ocr.synthesis.template_compiler import descendant as render
from document_ocr.synthesis.template_compiler import mixed_inventory as mixed
from document_ocr.synthesis.template_compiler.host import SpanDraft, validate_binding_realizations

SMALL = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
LARGE = "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"


def fixture(*, first="1X20ST", second="5X40HC", count=6, owned=True):
    raw = (first + "\n" + second).encode()
    bindings = []
    start = 0
    for key, text in (("small", first), ("large", second)):
        bindings.append(
            Node(
                logical_key=key,
                value_kind="equipment",
                target_paths=(),
                render_mode="deterministic_auxiliary",
                derivation=None,
                dependency_paths=("documentPatch.containers",) if owned else (),
                dependency_bindings=(),
                occurrences=(
                    Node(
                        source_text=text, byte_start=start, byte_end=start + len(text), slot_id=key
                    ),
                ),
            )
        )
        start += len(text) + 1
    target = {
        "documentPatch": {"containers": [dict(containerNumber=f"ID-{i}") for i in range(count)]}
    }
    return targets.SourceTemplate(
        "test", raw, deepcopy(target), target, Node(bindings=tuple(bindings)), "pin"
    )


def test_mixed_source_is_a_multiset_not_original_row_assignment():
    source = fixture()
    original = deepcopy(source.target)
    contract = mixed.compile_inventory(source)
    assert contract.equipment_counts == tuple(sorted(((SMALL, 1), (LARGE, 5))))
    assert source.target == original
    domains = {i: frozenset({SMALL, LARGE}) for i in range(6)}
    domains[0] = frozenset({LARGE})
    domains[5] = frozenset({SMALL})
    chosen = contract.sample(domains, DeterministicStream(1, "test", "sample"))
    assert chosen == {i: SMALL if i == 5 else LARGE for i in range(6)}
    audit = contract.audit(chosen)
    assert audit["originalTypeToIdentifierMappingClaimed"] is False
    assert audit["labelVisibilityUnchanged"] is True
    assert source.target == original


def test_assignment_can_differ_across_synthetic_samples_without_source_claim():
    contract = mixed.compile_inventory(fixture(first="1X20ST", second="1X40HC", count=2))
    domains = {0: frozenset({SMALL, LARGE}), 1: frozenset({SMALL, LARGE})}
    choices = {
        tuple(contract.sample(domains, DeterministicStream(seed, "test", "x"))[i] for i in range(2))
        for seed in range(12)
    }
    assert choices == {(SMALL, LARGE), (LARGE, SMALL)}


def test_no_matching_physical_assignment_is_an_explicit_failure():
    contract = mixed.compile_inventory(fixture())
    domains = {i: frozenset({LARGE}) for i in range(6)}
    assert contract.witness(domains) is None
    with pytest.raises(ValueError, match="no joint row-domain assignment"):
        contract.sample(domains, DeterministicStream(1, "x", "y"))


def test_aggregate_cannot_assume_missing_inventory_ownership():
    assert mixed.compile_inventory(fixture(owned=False)) is None


@pytest.mark.parametrize(
    "kwargs", [dict(count=5), dict(first="0X20ST"), dict(first="1X20FT"), dict(first="5X40HC")]
)
def test_ambiguous_or_contradictory_inventory_is_not_admitted(kwargs):
    with pytest.raises(ValueError):
        mixed.compile_inventory(fixture(**kwargs))


def test_missing_row_domains_cannot_be_treated_as_unconstrained():
    contract = mixed.compile_inventory(fixture())
    with pytest.raises(ValueError, match="every container exactly"):
        contract.witness({0: frozenset({SMALL, LARGE})})


def test_audit_rejects_changed_multiset():
    contract = mixed.compile_inventory(fixture())
    with pytest.raises(ValueError, match="contradicts its printed multiset"):
        contract.audit({i: LARGE for i in range(6)})


def test_stale_source_span_rejected():
    source = fixture()
    source = replace(source, source=source.source.replace(b"1X20ST", b"2X20ST"))
    with pytest.raises(ValueError, match="source evidence is stale"):
        mixed.compile_inventory(source)


def test_known_individual_types_stay_on_existing_path():
    source = fixture()
    source.target["documentPatch"]["containers"][0]["typeDescription"] = "20ST"
    assert mixed.compile_inventory(source) is None


def test_explicit_repeated_term_does_not_double_count_inventory():
    source = fixture()
    raw = source.source + b"\n40HCX5"
    repeat = Node(
        logical_key="large-repeat",
        value_kind="equipment",
        target_paths=(),
        render_mode="deterministic_derived",
        derivation="same_as_binding",
        dependency_paths=(),
        dependency_bindings=("large",),
        occurrences=(
            Node(source_text="40HCX5", byte_start=len(source.source) + 1, byte_end=len(raw)),
        ),
    )
    source = replace(
        source, source=raw, template=Node(bindings=(*source.template.bindings, repeat))
    )
    contract = mixed.compile_inventory(source)
    assert contract.container_count == 6
    assert len(contract.binding_keys) == 3


def test_duplicate_root_term_cannot_silently_be_assumed_repeated():
    source = fixture()
    raw = source.source + b"\n40HCX5"
    repeat = Node(
        logical_key="large-duplicate",
        value_kind="equipment",
        target_paths=(),
        render_mode="deterministic_auxiliary",
        derivation=None,
        dependency_paths=("documentPatch.containers",),
        dependency_bindings=(),
        occurrences=(
            Node(source_text="40HCX5", byte_start=len(source.source) + 1, byte_end=len(raw)),
        ),
    )
    source = replace(
        source, source=raw, template=Node(bindings=(*source.template.bindings, repeat))
    )
    with pytest.raises(ValueError, match="repeated-versus-additive"):
        mixed.compile_inventory(source)


def test_public_renderer_retains_proved_aggregate_without_adding_type_labels():
    source = fixture()
    proof = mixed.compile_inventory(source)
    target = deepcopy(source.target)
    for i, row in enumerate(target["documentPatch"]["containers"]):
        row["containerNumber"] = f"NEW-{i}"
    for binding in source.template.bindings:
        output = render._render_equipment_receipt_binding(
            binding,
            source_target=source.target,
            target=target,
            aggregate_inventory=proof,
        )
        assert output.replacements[binding.logical_key] == binding.occurrences[0].source_text
    assert all(set(row) == {"containerNumber"} for row in target["documentPatch"]["containers"])


@pytest.mark.parametrize("change", ["count", "type", "duplicate"])
def test_renderer_rejects_changed_multiset_topology_or_label_visibility(change):
    source = fixture()
    proof = mixed.compile_inventory(source)
    target = deepcopy(source.target)
    if change == "count":
        target["documentPatch"]["containers"].pop()
    elif change == "type":
        target["documentPatch"]["containers"][0]["typeCategory"] = "GENERAL_PURPOSE"
    else:
        target["documentPatch"]["containers"][0]["containerNumber"] = "ID-1"
    with pytest.raises(ValueError, match="inventory or label visibility"):
        render._render_equipment_receipt_binding(
            source.template.bindings[0],
            source_target=source.target,
            target=target,
            aggregate_inventory=proof,
        )


def test_compiler_proves_all_mixed_terms_together_before_accepting_subset_receipts():
    source = fixture()
    drafts = tuple(
        SpanDraft(
            draft_id=b.logical_key,
            logical_key=b.logical_key,
            render_mode="deterministic_derived",
            value_kind="equipment",
            group_kind="equipment",
            group_key="equipment:aggregate",
            target_paths=(),
            derivation="equipment_receipt",
            dependency_paths=b.dependency_paths,
            dependency_bindings=(),
            char_start=b.occurrences[0].byte_start,
            char_end=b.occurrences[0].byte_end,
            source_text=b.occurrences[0].source_text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Whole-shipment count/type inventory without observed row assignment.",
        )
        for b in source.template.bindings
    )
    validate_binding_realizations(
        raw=source.source.decode(), drafts=drafts, source_target=source.target
    )
    contradictory = deepcopy(source.target)
    contradictory["documentPatch"]["containers"].pop()
    with pytest.raises(ValueError, match="count differs"):
        validate_binding_realizations(
            raw=source.source.decode(), drafts=drafts, source_target=contradictory
        )
