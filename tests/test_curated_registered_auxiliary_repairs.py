"""Real-source replay for reviewed identifiers and complete auxiliary omissions."""

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from document_ocr.synthesis import curated_campaign as campaign_module
from document_ocr.synthesis import curated_wording as wording_module
from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_auxiliary import load_auxiliary_contract
from document_ocr.synthesis.curated_campaign import Campaign
from document_ocr.synthesis.template_compiler.models import (
    AuxiliarySemanticPlan,
    CertifiedSemanticTemplate,
)
from document_ocr.synthesis.template_compiler.semantic_plan import (
    build_auxiliary_semantic_plan,
    validate_auxiliary_semantic_plan,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
AUDIT = ROOT / "docs/analysis/mixed-equipment-production-20261012"
PROPOSAL = (
    ROOT / "docs/analysis/production-equipment-inventory-20261012/auxiliary-repair-proposals.yaml"
)
OMISSIONS = {
    "35136": ("reviewed:omitted_unresolved_notify_owner", "Owner: Al Ghurair Centre LLC"),
    "654e": ("reviewed:omitted_detached_source_brand", "BRAND: ASTUZINC ELECTRO 99.995%"),
}
CURRENT_SYSTEM = campaign_module.WORDING_PROMPT
CURRENT_REQUEST = campaign_module.wording_request
CURRENT_FIELD_REQUIREMENT = wording_module.wording_field_requirement


@pytest.fixture(scope="module")
def campaign():
    config = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_mixed_inventory_validation.yaml").read_text()
    )
    sources = [
        sid
        for sid in config["source_ids"]
        if sid.startswith(tuple("doc_" + p for p in ("00a165a2", "428aa480", "f36409", *OMISSIONS)))
    ]
    assert len(sources) == 5
    config.update(
        source_ids=sources,
        capabilities={sid: config["capabilities"][sid] for sid in sources},
    )
    config.pop("template_sampling", None)
    config["sampling"]["equipment_inventory"]["source_ids"] = sources
    # This fixture proves previously committed artifacts, not new generation.
    # Bind the immutable provider system prompt explicitly; production must
    # still reject these caches under the current instruction contract.
    instance = Campaign(ROOT, config)
    historical_prompt = None
    for file in sorted((instance.output / "calls").glob("exact-wording-*.json")):
        call = json.loads(file.read_text())
        if call.get("error") is not None:
            continue
        for message in call["messages"]:
            for part in message.get("parts", []):
                content = part.get("content")
                if isinstance(content, str) and content.startswith(
                    "Write fictional Bill of Lading"
                ):
                    historical_prompt = content
                    break
        if historical_prompt:
            break
    assert historical_prompt
    current_request = campaign_module.wording_request

    def historical_request(*args, **kwargs):
        request = current_request(*args, **kwargs)
        return replace(
            request,
            fields=tuple(
                replace(
                    field,
                    requirement=field.requirement.replace(
                        "has that detail. Incidental per-unit capacity/fill can be product detail. "
                        "Shipment counts, weights, transport settings and codes "
                        "are separate host facts.",
                        "has that detail. Packing, fills, quantities, weights, "
                        "transport settings and "
                        "codes are separate host-rendered facts.",
                    ),
                )
                for field in request.fields
            ),
        )

    def historical_field_requirement(field):
        # Match the archived native schema as well as its system/user prompts.
        # This is read-only historical replay, never approval under new rules.
        requirement = field.requirement
        if field.identity_kind is not None:
            requirement += " " + wording_module.IDENTITY_KIND_REQUIREMENTS[field.identity_kind]
        return requirement

    with (
        patch.object(campaign_module, "WORDING_PROMPT", historical_prompt),
        patch.object(campaign_module, "wording_request", historical_request),
        patch.object(wording_module, "wording_field_requirement", historical_field_requirement),
    ):
        yield instance


def test_identifier_recipes_are_registered_without_changing_role_country_rules():
    proposed = load_auxiliary_contract(PROPOSAL)
    registry = load_auxiliary_contract(REGISTRY / "auxiliary.yaml")
    campaign = load_auxiliary_contract(AUDIT / "campaign-auxiliary.yaml")
    previous = load_auxiliary_contract(AUDIT / "auxiliary-registry-before.yaml")
    for sid, spec in proposed["sources"].items():
        assert registry["sources"][sid]["bindings"] == spec["bindings"]
        assert campaign["sources"][sid]["bindings"] == spec["bindings"]
        key = "agent:exporter_registration_country"
        assert (
            registry["sources"][sid]["bindings"][key] == (previous["sources"][sid]["bindings"][key])
        )
    metadata = json.loads((AUDIT / "auxiliary-registry-before-metadata.json").read_text())
    assert (
        sha256((AUDIT / "auxiliary-registry-before.yaml").read_bytes()).hexdigest()
        == metadata["registryFileSha256"]
    )
    assert (
        sha256((ROOT / metadata["campaignAuxiliaryPath"]).read_bytes()).hexdigest()
        == metadata["campaignAuxiliarySha256"]
    )


def test_current_instructions_reject_historical_wording_cache(campaign):
    sid = next(s for s in campaign.source_ids if s.startswith("doc_f36409"))
    with (
        patch.object(campaign_module, "WORDING_PROMPT", CURRENT_SYSTEM),
        patch.object(campaign_module, "wording_request", CURRENT_REQUEST),
        patch.object(wording_module, "wording_field_requirement", CURRENT_FIELD_REQUIREMENT),
        pytest.raises(ValueError, match="wording request changed"),
    ):
        campaign.render(sid, 1, persist=False)


def test_unresolved_owner_metadata_removes_only_the_unsupported_equality():
    before = CertifiedSemanticTemplate.model_validate_json(
        (AUDIT / "template-35136-before-auxiliary-repair.json").read_bytes()
    )
    after_path = REGISTRY / "cases" / before.document_id / "template.json"
    after = CertifiedSemanticTemplate.model_validate_json(after_path.read_bytes())
    assert before.bindings == after.bindings
    assert before.source_sha256 == after.source_sha256
    assert len(before.auxiliary_semantic_plan.entities) == (
        len(after.auxiliary_semantic_plan.entities) + 1
    )
    assert not any(
        member.logical_key == "agent:notify_party_owner"
        for entity in after.auxiliary_semantic_plan.entities
        for member in entity.members
    )
    decision = next(
        row
        for row in after.auxiliary_semantic_plan.dispositions
        if row.logical_key == "agent:notify_party_owner"
    )
    assert decision.disposition == "independent_fact" and decision.semantic_id is None
    assert "unresolved" in decision.rationale
    validate_auxiliary_semantic_plan(
        plan=after.auxiliary_semantic_plan,
        bindings=after.bindings,
        source_target=json.loads((after_path.parent / "target.json").read_text()),
    )


def test_owner_metadata_repair_rejects_a_dangling_former_entity_reference():
    original = json.loads((AUDIT / "template-35136-before-auxiliary-repair.json").read_text())
    plan = deepcopy(original["auxiliary_semantic_plan"])
    plan["entities"] = [
        entity
        for entity in plan["entities"]
        if not any(m["logical_key"] == "agent:notify_party_owner" for m in entity["members"])
    ]
    with pytest.raises(ValidationError, match="unknown semantic facts"):
        AuxiliarySemanticPlan.model_validate_json(json.dumps(plan))


def test_actual_original_owner_recompilation_cannot_restore_the_false_notify_alias():
    original = CertifiedSemanticTemplate.model_validate_json(
        (AUDIT / "template-35136-before-auxiliary-repair.json").read_bytes()
    )
    folder = REGISTRY / "cases" / original.document_id
    plan = build_auxiliary_semantic_plan(
        raw=(folder / "source.txt").read_text(),
        bindings=original.bindings,
        source_target=json.loads((folder / "target.json").read_text()),
    )
    owner = next(
        entity
        for entity in plan.entities
        if any(member.logical_key == "agent:notify_party_owner" for member in entity.members)
    )
    assert owner.relationship == "independent" and owner.target_party_path is None
    assert "name conflicts" in owner.rationale


def test_active_carrier_signer_metadata_is_independent_of_the_forwarder():
    original = CertifiedSemanticTemplate.model_validate_json(
        (AUDIT / "template-f364-before-name-guard.json").read_bytes()
    )
    folder = REGISTRY / "cases" / original.document_id
    current = CertifiedSemanticTemplate.model_validate_json((folder / "template.json").read_bytes())
    assert current.bindings == original.bindings
    entity = next(
        e
        for e in current.auxiliary_semantic_plan.entities
        if any(m.logical_key == "agent:forwarding_agent_public_surface" for m in e.members)
    )
    assert entity.relationship == "independent" and entity.target_party_path is None
    assert "Signed for the Carrier" in entity.rationale
    assert "Forwarder block" in entity.rationale
    validate_auxiliary_semantic_plan(
        plan=current.auxiliary_semantic_plan,
        bindings=current.bindings,
        source_target=json.loads((folder / "target.json").read_text()),
    )


@pytest.mark.parametrize("variant", [1, 2])
def test_carrier_signer_metadata_change_preserves_current_render_and_labels(campaign, variant):
    sid = next(s for s in campaign.source_ids if s.startswith("doc_f36409"))
    sample = "syn_full_v7_" + digest([sid, campaign.config["seed"], variant])[:24]
    cached = json.loads((campaign.output / "candidates" / f"{sample}.json").read_text())
    result = campaign.render(sid, variant, persist=False)
    assert result["target"] == cached["target"]
    assert result["joinedRawText"] == cached["joinedRawText"]
    assert result["renderValues"] == cached["renderValues"]
    assert result["validation"]["exactReplay"] and result["validation"]["currentSchema"]


@pytest.mark.parametrize(
    "prefix,variant", [(p, v) for p in ("00a165a2", "428aa480") for v in (1, 2)]
)
def test_actual_identifier_rerender_preserves_every_public_target(campaign, prefix, variant):
    sid = next(s for s in campaign.source_ids if s.startswith("doc_" + prefix))
    sample = "syn_full_v7_" + digest([sid, campaign.config["seed"], variant])[:24]
    cached = json.loads((campaign.output / "candidates" / f"{sample}.json").read_text())
    result = campaign.render(sid, variant, persist=False)
    assert result["target"] == cached["target"]
    assert result["renderValues"] == cached["renderValues"]
    assert result["countryCodes"] == cached["countryCodes"]
    assert result["validation"]["exactReplay"] and result["validation"]["currentSchema"]
    blueprint = campaign.blueprint(sid)
    proposal = load_auxiliary_contract(PROPOSAL)["sources"][sid]["bindings"]
    for key, recipe in proposal.items():
        if "identifier" not in recipe:
            continue
        originals = [o["source_text"] for o in blueprint.historical_bindings[key]["occurrences"]]
        value = result["surfaceValues"][key]
        values = value if isinstance(value, list) else [value] * len(originals)
        assert len(set(values)) == 1
        for original in originals:
            assert original not in result["joinedRawText"]
        assert all(a != b for a, b in zip(originals, values, strict=True))


@pytest.mark.parametrize("prefix,variant", [(p, v) for p in OMISSIONS for v in (1, 2)])
def test_actual_whole_caption_omission_preserves_public_targets(campaign, prefix, variant):
    sid = next(s for s in campaign.source_ids if s.startswith("doc_" + prefix))
    sample = "syn_full_v7_" + digest([sid, campaign.config["seed"], variant])[:24]
    cached = json.loads((campaign.output / "candidates" / f"{sample}.json").read_text())
    blueprint = campaign.blueprint(sid)
    key, quote = OMISSIONS[prefix]
    regions = [r for r in blueprint.regions if r.key == key]
    assert len(regions) == 1 and regions[0].source == quote and regions[0].allow_empty
    assert regions[0].target_paths == ()
    result = campaign.render(sid, variant, persist=False)
    if prefix == "654e":
        path = "documentPatch.goodsItemDetails[0].handlingInstructions[0]"
        before, after = flat(cached["target"]), flat(result["target"])
        assert {p for p in set(before) | set(after) if before.get(p) != after.get(p)} <= {path}
        projection = result["proof"]["handlingProjection"]["fields"]
        assert len(projection) == 1 and projection[0]["path"] == path
        assert projection[0]["before"] == flat(blueprint.target)[path]
        assert projection[0]["value"] == after[path]
        for span in projection[0]["spans"]:
            printed = (
                result["joinedRawText"]
                .encode()[span["renderedByteStart"] : span["renderedByteEnd"]]
                .decode()
            )
            assert printed == span["text"]
        assert " ".join(span["text"] for span in projection[0]["spans"]).upper() == after[path]
    else:
        assert result["target"] == cached["target"]
    assert quote not in result["joinedRawText"]
    edits = [e for e in result["proof"]["edits"] if e["key"] == key]
    assert len(edits) == 1 and edits[0]["before"] == quote and edits[0]["after"] == ""
    assert result["validation"]["exactReplay"] and result["validation"]["currentSchema"]
