import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_phones import (
    PhoneFamilyWeight,
    PhoneRenderingPolicy,
    grounded_phone_surface,
    prepare_phone_rendering,
    validate_phone_source_completeness,
)
from document_ocr.synthesis.curated_templates import (
    RenderRegion,
    SamplingBlueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.template_compiler.contact_values import phone

ROLE = "documentPatch.parties.consignee"
PATH = ROLE + ".contactDetails.phoneNumbers[0]"
STREAM = DeterministicStream(411, "phone-render-test", "sample")


def fixture(value="0020 2 2575 2002", *, literal_plus=False, curated=False, repeat=False):
    source_piece = value[1:] if literal_plus else value
    raw = "CONSIGNEE\nTEL: " + ("+" if literal_plus else "") + source_piece + "\n"
    if repeat:
        raw += "COPY\nTEL: " + source_piece + "\n"
    target = {
        "documentPatch": {"parties": {"consignee": {"contactDetails": {"phoneNumbers": [value]}}}}
    }
    variables = [
        {
            "key": "phone",
            "kind": "identifier",
            "value": value,
            "meaning": "consignee phone",
            "required_literals": [],
            "occurrences": [{"text": source_piece, "occurrence": 1, "presentation": "text"}],
        }
    ]
    contract = SourceContract(
        variables=variables,
        targets=[{"path": PATH, "expression": "{phone}"}],
        fixed_context="Explicit test phone ownership",
    )
    starts = [raw.index(source_piece)]
    if repeat:
        starts.append(raw.index(source_piece, starts[0] + len(source_piece)))
    regions = tuple(
        RenderRegion(
            "phone",
            index,
            start,
            start + len(source_piece),
            source_piece,
            (PATH,),
            "phone",
            "phone" if curated else None,
            baseline=value if curated else None,
        )
        for index, start in enumerate(starts)
    )
    bindings = {"phone": {"occurrences": [{"source_text": source_piece} for _ in starts]}}
    return SamplingBlueprint("sample", raw, target, contract, regions, bindings, {}, ()), target


def plan_for(blueprint, target, *, policy=None):
    return prepare_phone_rendering(
        blueprint,
        target,
        source_country_codes={ROLE: "EG"},
        target_country_codes={ROLE: "GB"},
        policy=policy,
        stream=STREAM,
    )


@pytest.mark.parametrize("source_country_codes", [{ROLE: "EG"}, {}])
def test_bare_phone_target_does_not_leave_old_dialing_prefix_in_complete_fax_alias(source_country_codes):
    blueprint, target = fixture("4863088")
    alias = "FAX: (+203) 4863088"
    source = blueprint.source + alias + "\n"
    bindings = deepcopy(blueprint.historical_bindings)
    bindings["fax_alias"] = {"occurrences": [{"source_text": alias}],
                             "auxiliary_recipe": {"target": PATH, "prefix": "FAX: "}}
    blueprint = replace(blueprint, source=source, historical_bindings=bindings,
                        regions=(*blueprint.regions, RenderRegion(
                            "fax_alias", 0, len(blueprint.source), len(source) - 1,
                            alias, (), "auxiliary")))
    sampled = deepcopy(target)
    canonical = "+44 7700-123456"
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [canonical]
    plan = prepare_phone_rendering(
        blueprint, sampled, source_country_codes=source_country_codes,
        target_country_codes={ROLE: "GB"}, stream=STREAM)
    text, actual, _ = render_sampling_blueprint(
        blueprint, sampled, {}, surface_values={"fax_alias": "FAX: " + canonical}, phone_rendering=plan)
    fax = text.split("FAX: ", 1)[1].strip()
    assert re.sub(r"\D", "", fax) == re.sub(r"\D", "", canonical)
    assert "(+203)" not in fax
    assert actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"][0] in text


@pytest.mark.parametrize("curated", [False, True])
def test_owned_source_presentation_and_target_projection_are_reproducible(curated):
    blueprint, target = fixture(curated=curated, repeat=True)
    sampled = deepcopy(target)
    canonical = phone(STREAM, country_code="GB")
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [canonical]
    plan = plan_for(blueprint, sampled)
    text, actual, proof = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    styled = actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"][0]
    assert styled.startswith("0044") and not styled.startswith("+")
    assert text.count(styled) == 2
    assert proof["phonePresentations"]["inputValues"] == {PATH: canonical}
    assert proof["phonePresentations"]["targetValues"] == {PATH: styled}
    assert proof["changedTargetPaths"] == proof["coveredTargetPaths"] == [PATH]
    assert render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan) == (
        text,
        actual,
        proof,
    )


@pytest.mark.parametrize("tail", ["/5", "/1/2", "/ 469"])
@pytest.mark.parametrize("partial_owner", [False, True])
def test_baseline_cannot_omit_compact_alternate_even_when_unchanged(tail, partial_owner):
    blueprint, target = fixture("20 57 290114" + tail)
    target["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        "20 57 290114"
    ]
    if partial_owner:
        region = blueprint.regions[0]
        blueprint = replace(
            blueprint,
            regions=(replace(region, source="20 57 290114", end=region.end - len(tail)),),
        )
    with pytest.raises(ValueError, match="omits a printed compact alternate"):
        validate_phone_source_completeness(blueprint)
    with pytest.raises(ValueError, match="omits a printed compact alternate"):
        plan_for(blueprint, target)


def test_complete_compact_alternate_has_identical_printed_and_target_values():
    blueprint, target = fixture("20 57 290114/5")
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    plan = plan_for(blueprint, sampled)
    text, actual, _ = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    value = actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"][0]
    assert "/" in value and "TEL: " + value + "\n" in text


@pytest.mark.parametrize(
    "tail", ["-7", "-73", " - 73", "\u20137", "~73", "-123456", "-12345 6789", "-123-456"]
)
@pytest.mark.parametrize("partial_owner", [False, True])
def test_incomplete_numeric_hyphen_tail_rejected_without_reclassifying_phone(tail, partial_owner):
    base = "20 66 3244235"
    blueprint, target = fixture(base + tail)
    target["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [base]
    region = blueprint.regions[0]
    blueprint = replace(
        blueprint,
        regions=(
            replace(
                region,
                source=base if partial_owner else region.source,
                end=region.start + len((base if partial_owner else region.source).encode()),
            ),
        ),
    )
    with pytest.raises(ValueError, match="printed numeric continuation"):
        validate_phone_source_completeness(blueprint)
    with pytest.raises(ValueError, match="printed numeric continuation"):
        plan_for(blueprint, target)


@pytest.mark.parametrize("value", ["20 66 3244235-7", "+420 596 270 052-58", "+44-20-7946-1234"])
def test_complete_hyphen_envelope_remains_fully_printed_and_labelled(value):
    blueprint, target = fixture(value)
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    plan = plan_for(blueprint, sampled)
    text, actual, _ = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    labelled = actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"][0]
    assert "-" in labelled and "TEL: " + labelled + "\n" in text


_NEGATIVES = json.loads(
    (
        Path(__file__).parents[1]
        / "docs/analysis/r28-legacy-closure-20261019/phones/negative-controls.json"
    ).read_bytes()
)
_HYPHEN_NEGATIVES = json.loads(
    (
        Path(__file__).parents[1]
        / "docs/analysis/r28-held-branch-repair-20261020/phones/negative-controls.json"
    ).read_bytes()
)


@pytest.mark.parametrize("control", _NEGATIVES + _HYPHEN_NEGATIVES, ids=lambda c: c["documentId"])
def test_complete_second_phone_after_separator_is_not_an_abbreviated_alternate(control):
    blueprint, target = fixture()
    raw = control["line"] + "\n"
    target["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = control[
        "phones"
    ]
    spans = {}
    for index, value in enumerate(control["phones"]):
        pattern = r"[ \t]*".join(re.escape(t) for t in value.split())
        for match in re.finditer(pattern, raw):
            start, end = len(raw[: match.start()].encode()), len(raw[: match.end()].encode())
            spans[(start, end)] = (index, match[0])
    regions = tuple(
        RenderRegion(
            f"phone-{index}",
            0,
            start,
            end,
            source,
            (ROLE + f".contactDetails.phoneNumbers[{index}]",),
            "phone",
        )
        for (start, end), (index, source) in sorted(spans.items())
    )
    blueprint = replace(
        blueprint,
        source=raw,
        target=target,
        regions=regions,
    )
    validate_phone_source_completeness(blueprint)
    assert plan_for(blueprint, target).target_values == {}


def test_campaign_rejects_bad_phone_authority_before_planning_or_paid_work(tmp_path, monkeypatch):
    from document_ocr.synthesis import curated_campaign as campaign_module

    blueprint, target = fixture("20 57 290114/5")
    target["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        "20 57 290114"
    ]
    case = tmp_path / "cases" / "sample"
    case.mkdir(parents=True)
    (case / "contract.json").write_text(
        json.dumps(
            {
                "sourceSha256": digest(blueprint.source.encode()),
                "targetSha256": digest(target),
                "contract": blueprint.contract.model_dump(mode="json"),
            }
        )
    )
    (case / "template.json").write_text("{}")
    (tmp_path / "ownership.yaml").write_text("version: 1\nsources: {}\n")
    campaign = campaign_module.Campaign.__new__(campaign_module.Campaign)
    campaign.root, campaign.config = (
        tmp_path,
        {"source_contracts": "cases", "historical_catalog": "cases", "ownership": "ownership.yaml"},
    )
    campaign.rows = {"sample": {"joinedRawText": blueprint.source, "target": target}}
    campaign._blueprints, campaign._blueprint_file_pins = {}, {}
    campaign.country_registry, campaign.auxiliary = None, {}
    monkeypatch.setattr(campaign_module, "load_owned_blueprint", lambda *args: blueprint)
    monkeypatch.setattr(campaign_module, "bind_postal_country_alias_owners", lambda bp, _: bp)
    monkeypatch.setattr(campaign_module, "augment_auxiliary_blueprint", lambda bp, _: bp)
    with pytest.raises(ValueError, match="omits a printed compact alternate"):
        campaign.blueprint("sample")


def test_external_literal_plus_is_preserved_once_without_changing_frame():
    blueprint, target = fixture("+20 2 2575 2002", literal_plus=True)
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    plan = plan_for(blueprint, sampled)
    text, actual, proof = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    assert "TEL: +44" in text and "++" not in text
    assert actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"][
        0
    ].startswith("+44")
    assert proof["edits"][0]["before"] == "20 2 2575 2002"
    assert not proof["edits"][0]["after"].startswith("+")


def test_identity_replay_stays_byte_exact_and_changed_phone_requires_plan():
    blueprint, target = fixture()
    text, actual, proof = render_sampling_blueprint(blueprint, target, {})
    assert text == blueprint.source and actual == target and proof["edits"] == []
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    with pytest.raises(ValueError, match="explicit source-conditioned"):
        render_sampling_blueprint(blueprint, sampled, {})
    plan = plan_for(blueprint, sampled)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM.derive("other"), country_code="GB")
    ]
    with pytest.raises(ValueError, match="pinned source or input"):
        render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)


def test_policy_weights_are_explicit_and_literal_boundary_constrains_family():
    policy = PhoneRenderingPolicy(family_weights=(PhoneFamilyWeight(family="national", weight=1),))
    assert policy.select(STREAM, ("",)) == "national"
    with pytest.raises(ValueError, match="unchanged dialing prefix"):
        policy.select(STREAM, ("+",))
    assert PhoneRenderingPolicy().select(STREAM, ("+", "00")) == "source"
    with pytest.raises(ValidationError, match="nonempty and distinct"):
        PhoneRenderingPolicy(family_weights=())
    with pytest.raises(ValidationError, match="nonempty and distinct"):
        PhoneRenderingPolicy(family_weights=(PhoneFamilyWeight(family="source", weight=1),) * 2)


def test_missing_country_and_wrong_identity_country_surface_errors():
    blueprint, target = fixture()
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    with pytest.raises(ValueError, match="explicit country authority"):
        prepare_phone_rendering(
            blueprint, sampled, source_country_codes={}, target_country_codes={}, stream=STREAM
        )
    with pytest.raises(ValueError, match="valid for the target country"):
        prepare_phone_rendering(
            blueprint,
            sampled,
            source_country_codes={ROLE: "EG"},
            target_country_codes={ROLE: "AE"},
            stream=STREAM,
        )


def test_partial_owner_cannot_expand_into_two_complete_numbers():
    blueprint, target = fixture()
    original = blueprint.regions[0]
    partial = replace(original, source="002", end=original.start + 3)
    blueprint = replace(blueprint, regions=(partial,))
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    with pytest.raises(ValueError, match="partial or unrelated number"):
        plan_for(blueprint, sampled)


def test_two_distinct_phone_fields_share_one_owned_slash_surface_without_mixing():
    blueprint, target = fixture("+20 3 4853388")
    second_path = ROLE + ".contactDetails.phoneNumbers[1]"
    target["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"].append(
        "4812336"
    )
    source = "+20 3 4853388/4812336"
    original = blueprint.regions[0]
    raw = blueprint.source.replace(original.source, source)
    blueprint = replace(
        blueprint,
        source=raw,
        target=target,
        regions=(
            replace(
                original,
                source=source,
                end=original.start + len(source),
                target_paths=(PATH, second_path),
            ),
        ),
    )
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM.derive("one"), country_code="GB"),
        phone(STREAM.derive("two"), country_code="GB"),
    ]
    plan = plan_for(blueprint, sampled)
    text, actual, proof = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    values = actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"]
    assert values[0].startswith("+44") and not values[1].startswith("+")
    assert "TEL: " + "/".join(values) + "\n" in text
    assert len(proof["phonePresentations"]["regions"]) == 2
    assert "4812336" not in text


def test_repeated_phone_identity_across_distinct_party_roles_projects_consistently():
    blueprint, target = fixture(repeat=True)
    notify_role = "documentPatch.parties.notifyParties[0]"
    notify_path = notify_role + ".contactDetails.phoneNumbers[0]"
    target["documentPatch"]["parties"]["notifyParties"] = [
        deepcopy(target["documentPatch"]["parties"]["consignee"])
    ]
    blueprint = replace(
        blueprint,
        target=target,
        regions=(blueprint.regions[0], replace(blueprint.regions[1], target_paths=(notify_path,))),
    )
    sampled = deepcopy(target)
    canonical = phone(STREAM, country_code="GB")
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [canonical]
    sampled["documentPatch"]["parties"]["notifyParties"][0]["contactDetails"]["phoneNumbers"] = [
        canonical
    ]
    plan = prepare_phone_rendering(
        blueprint,
        sampled,
        source_country_codes={ROLE: "EG", notify_role: "EG"},
        target_country_codes={ROLE: "GB", notify_role: "GB"},
        stream=STREAM,
    )
    text, actual, proof = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    assert len(set(plan.target_values.values())) == 1
    assert text.count(next(iter(plan.target_values.values()))) == 2
    assert set(proof["phonePresentations"]["targetValues"]) == {PATH, notify_path}
    assert (
        actual["documentPatch"]["parties"]["consignee"]["contactDetails"]
        == actual["documentPatch"]["parties"]["notifyParties"][0]["contactDetails"]
    )


def test_non_ascii_character_adjacent_to_phone_does_not_require_byte_decoding():
    blueprint, target = fixture()
    raw = blueprint.source.replace("TEL: ", "TEL:é")
    original = blueprint.regions[0]
    start = raw.encode().index(original.source.encode())
    blueprint = replace(
        blueprint,
        source=raw,
        regions=(replace(original, start=start, end=start + len(original.source.encode())),),
    )
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    plan = plan_for(blueprint, sampled)
    text, _, _ = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    assert "TEL:é0044" in text


def test_reviewed_brace_envelope_is_required_and_projects_literal_printed_target():
    blueprint, target = fixture("+20 2 24801010")
    old = blueprint.regions[0]
    literal = "{+}20 2 24801010"
    blueprint = replace(
        blueprint,
        source=blueprint.source.replace(old.source, literal),
        regions=(replace(old, source=literal, end=old.start + len(literal)),),
    )
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    with pytest.raises(ValueError, match="no exact source-certified printed occurrence"):
        plan_for(blueprint, sampled)
    blueprint = replace(
        blueprint,
        ownership_data={
            "phone_target_presentations": {
                PATH: {"key": "phone", "occurrence": 1, "value": literal}
            }
        },
    )
    plan = plan_for(blueprint, sampled)
    text, actual, proof = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    value = actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"][0]
    assert value.startswith("{+}44") and "TEL: " + value + "\n" in text
    assert proof["phonePresentations"]["regions"][0]["targetSourceValue"] == literal
    invalid = replace(
        blueprint,
        ownership_data={
            "phone_target_presentations": {
                PATH: {"key": "phone", "occurrence": 2, "value": literal}
            }
        },
    )
    with pytest.raises(ValueError, match="lacks its exact owned source"):
        plan_for(invalid, sampled)


def test_immutable_external_opening_parenthesis_is_part_of_grounding_not_replaced():
    blueprint, target = fixture("(20-57)290221")
    old = blueprint.regions[0]
    blueprint = replace(
        blueprint, regions=(replace(old, start=old.start + 1, source=old.source[1:]),)
    )
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    plan = plan_for(blueprint, sampled)
    text, actual, proof = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    value = actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"][0]
    assert value.startswith("(") and "TEL: " + value + "\n" in text
    assert proof["phonePresentations"]["regions"][0]["framePrefix"] == "("
    assert not proof["edits"][0]["after"].startswith("(")


def test_only_source_attested_digit_run_newlines_are_joined_in_target_grounding():
    blueprint, target = fixture("+20122738114")
    old = blueprint.regions[0]
    printed = "+201227381\n14"
    blueprint = replace(
        blueprint,
        source=blueprint.source.replace(old.source, printed),
        regions=(replace(old, source=printed, end=old.start + len(printed)),),
    )
    sampled = deepcopy(target)
    sampled["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"] = [
        phone(STREAM, country_code="GB")
    ]
    plan = plan_for(blueprint, sampled)
    text, actual, proof = render_sampling_blueprint(blueprint, sampled, {}, phone_rendering=plan)
    record = proof["phonePresentations"]["regions"][0]
    value = actual["documentPatch"]["parties"]["consignee"]["contactDetails"]["phoneNumbers"][0]
    assert record["joinedNewlineOrdinals"] == [0]
    assert record["rendered"].count("\n") == 1
    assert grounded_phone_surface(record["rendered"], (0,)) == value
    assert " ".join(record["rendered"].split()) != value
    assert text.count("\n") == blueprint.source.count("\n")
