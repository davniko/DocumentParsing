"""Real-source replay for reviewed identifiers and complete auxiliary omissions."""

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis import curated_campaign as campaign_module
from document_ocr.synthesis import curated_contacts as contacts_module
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
from document_ocr.training.tasks import get_training_task

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
HISTORICAL_REGISTRY = (
    ROOT / "docs/analysis/synthesis-private-contacts-repair-20261013/before/registry"
)
AUDIT = ROOT / "docs/analysis/mixed-equipment-production-20261012"
PROPOSAL = (
    ROOT / "docs/analysis/production-equipment-inventory-20261012/auxiliary-repair-proposals.yaml"
)
OMISSIONS = {
    "35136": ("reviewed:omitted_unresolved_notify_owner", "Owner: Al Ghurair Centre LLC"),
    "654e": ("reviewed:omitted_detached_source_brand", "BRAND: ASTUZINC ELECTRO 99.995%"),
}


@pytest.fixture(scope="module")
def historical_output():
    config = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_mixed_inventory_validation.yaml").read_text()
    )
    return ROOT / config["output"]


@pytest.fixture(scope="module")
def campaign(tmp_path_factory, historical_output):
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
        # Replay one complete pre-privacy source version. Current contracts now
        # own additional private contacts and correctly invalidate old receipts.
        dataset=str(
            (HISTORICAL_REGISTRY / "source-authorities/reaudit-20261012/real660").relative_to(ROOT)
        ),
        source_contracts=str((HISTORICAL_REGISTRY / "cases").relative_to(ROOT)),
        historical_catalog=str((HISTORICAL_REGISTRY / "cases").relative_to(ROOT)),
        ownership=str((HISTORICAL_REGISTRY / "ownership.yaml").relative_to(ROOT)),
        auxiliary=str((HISTORICAL_REGISTRY / "auxiliary.yaml").relative_to(ROOT)),
        output=str(tmp_path_factory.mktemp("revalidated-auxiliary-cache")),
    )
    config.pop("template_sampling", None)
    config["sampling"]["equipment_inventory"]["source_ids"] = sources
    # The archived vocabulary remains fixed; schema pins bind current code.
    # This does not relax production's stale-constraint rejection.
    original = json.loads((ROOT / config["task_constraints"]).read_text())
    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    original.update(
        basePromptSchemaSha256=task.base_prompt_schema_sha256(),
        targetSchemaSha256=sha256_bytes(canonical_json_bytes(task.target_schema())),
    )
    constraint = tmp_path_factory.mktemp("historical-auxiliary-vocabulary") / "constraints.json"
    constraint.write_bytes(canonical_json_bytes(original))
    config["task_constraints"] = str(constraint)
    config["task_constraints_sha256"] = sha256_bytes(constraint.read_bytes())
    # Revalidate immutable historical values under current contracts, then bind
    # only temporary copied receipts. Production caches and guards stay intact.
    instance = Campaign(ROOT, config)
    for sid in sources:
        for blueprint, scenario, sample_id, target in instance.plans(sid):
            source = historical_output / "wording" / f"{sample_id}.json"
            receipt = json.loads(source.read_text())
            request = instance.wording_request(blueprint, scenario, sample_id, target)
            model = wording_module.wording_output_type([request]).model_validate(
                {"s0": receipt["values"]}
            )
            checked = wording_module.unpack_wording(model, [request])[sample_id]
            assert checked == receipt["values"]
            assert not campaign_module.postal_wording_errors(blueprint, scenario, target, checked)
            receipt["testRevalidation"] = {
                "historicalFileSha256": digest(source.read_bytes()),
                "historicalRequestSha256": receipt["requestSha256"],
                "currentNativeSchemaAndSemanticValidation": True,
                "postalValidation": True,
                "valuesUnchanged": True,
            }
            receipt["requestSha256"] = campaign_module.wording_request_hash(request)
            destination = instance.output / "wording" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(receipt))
            cached = historical_output / "candidates" / f"{sample_id}.json"
            destination = instance.output / "candidates" / cached.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, destination)

        parties = instance.contact_requests(sid)
        if not parties:
            continue
        source = historical_output / "contacts" / f"{sid}.json"
        receipt = json.loads(source.read_text())
        checked = contacts_module.unpack_contacts(
            contacts_module.contact_output_type(parties).model_validate(receipt["output"]),
            parties,
        )
        assert checked == receipt["values"]
        receipt["testRevalidation"] = {
            "historicalFileSha256": digest(source.read_bytes()),
            "historicalRequestSha256": receipt["requestSha256"],
            "currentSchemaAndSemanticValidation": True,
            "valuesUnchanged": True,
        }
        receipt["requestSha256"] = campaign_module.contact_request_hash(parties)
        destination = instance.output / "contacts" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt))
        for sample_id in checked:
            selected = [party for party in parties if party.sample_id == sample_id]
            source = historical_output / "contacts/samples" / f"{sample_id}.json"
            sample = json.loads(source.read_text())
            assert (
                contacts_module.unpack_contacts(
                    contacts_module.contact_output_type(selected).model_validate(sample["output"]),
                    selected,
                )[sample_id]
                == sample["values"]
            )
            sample["testRevalidation"] = {
                "historicalFileSha256": digest(source.read_bytes()),
                "historicalRequestSha256": sample["requestSha256"],
                "valuesUnchanged": True,
            }
            sample["requestSha256"] = campaign_module.contact_request_hash(selected)
            sample["batchReceiptSha256"] = digest(receipt)
            destination = instance.output / "contacts/samples" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(sample))
    return instance


@pytest.fixture(scope="module")
def historical_contact_output(campaign, historical_output, tmp_path_factory):
    """Current checked wording with unmodified old contact receipts: negative control."""
    output = tmp_path_factory.mktemp("stale-contact-negative-control")
    shutil.copytree(campaign.output / "wording", output / "wording")
    for sid in campaign.source_ids:
        parties = campaign.contact_requests(sid)
        if not parties:
            continue
        source = historical_output / "contacts" / f"{sid}.json"
        destination = output / "contacts" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        for sample_id in {party.sample_id for party in parties}:
            source = historical_output / "contacts/samples" / f"{sample_id}.json"
            destination = output / "contacts/samples" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return output


def assert_preserved_phone_identities(before, result):
    old, new = flat(before), flat(result["target"])
    receipt = result["proof"].get("phonePresentations", {})
    changed = {
        p for p in set(old) | set(new)
        if ".phoneNumbers[" in p and old.get(p) != new.get(p)
    }
    for path in changed:
        assert receipt["inputValues"][path] == old[path]
        assert receipt["targetValues"][path] == new[path]
    return changed


def assert_except_documented_presentations_equal(before, result):
    old, new = flat(before), flat(result["target"])
    phones = assert_preserved_phone_identities(before, result)
    assert {p: v for p, v in old.items() if ".hsCodes[" not in p} == {
        p: old[p] if p in phones else v for p, v in new.items() if ".hsCodes[" not in p
    }


def assert_non_hs_edits_equal(cached, result, blueprint):
    hs_keys = {
        region.key
        for region in blueprint.regions
        if any(".hsCodes[" in path for path in region.target_paths)
    }
    hs_keys.update(
        key
        for key, binding in blueprint.historical_bindings.items()
        if ".hsCodes[" in binding.get("auxiliary_recipe", {}).get("target", "")
    )
    assert [e for e in cached["proof"]["edits"] if e["key"] not in hs_keys] == [
        e for e in result["proof"]["edits"] if e["key"] not in hs_keys
    ]


def test_identifier_recipes_are_registered_without_changing_role_country_rules():
    proposed = load_auxiliary_contract(PROPOSAL)
    registry = load_auxiliary_contract(REGISTRY / "auxiliary.yaml")
    campaign = load_auxiliary_contract(AUDIT / "campaign-auxiliary.yaml")
    previous = load_auxiliary_contract(AUDIT / "auxiliary-registry-before.yaml")
    for sid, spec in proposed["sources"].items():
        for key, recipe in spec["bindings"].items():
            active = registry["sources"][sid]["bindings"][key]
            if "identifier" in recipe and active != recipe:
                assert "identifier" in active and active["format"] == "source"
            else:
                assert active == recipe
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


def test_current_instructions_reject_historical_wording_cache(campaign, historical_output):
    sid = next(s for s in campaign.source_ids if s.startswith("doc_f36409"))
    with (
        patch.object(campaign, "output", historical_output),
        pytest.raises(ValueError, match="wording request changed"),
    ):
        campaign.render(sid, 1, persist=False)


@pytest.mark.parametrize("variant", [1, 2])
def test_current_contact_instructions_reject_historical_contact_cache(
    campaign, historical_contact_output, variant
):
    sid = next(s for s in campaign.source_ids if s.startswith("doc_654e"))
    with (
        patch.object(campaign, "output", historical_contact_output),
        pytest.raises(ValueError, match="company contact receipt is stale"),
    ):
        campaign.render(sid, variant, persist=False)


def test_revalidated_contact_replay_still_hashes_company_and_source_values(campaign):
    sid = next(s for s in campaign.source_ids if s.startswith("doc_654e"))
    parties = campaign.contact_requests(sid)
    original = campaign_module.contact_request_hash(parties)
    renamed = [replace(parties[0], name="DIFFERENT FICTIONAL COMPANY"), *parties[1:]]
    changed_email = [
        replace(
            parties[0],
            fields=(replace(parties[0].fields[0], original="other.person@different.example"),),
        ),
        *parties[1:],
    ]
    assert campaign_module.contact_request_hash(renamed) != original
    assert campaign_module.contact_request_hash(changed_email) != original


def test_fixture_revalidation_preserves_immutable_cached_values(campaign, historical_output):
    for folder in ("wording", "contacts", "contacts/samples"):
        receipts = list((campaign.output / folder).glob("*.json"))
        assert receipts
        for copied in receipts:
            receipt = json.loads(copied.read_text())
            historical = historical_output / folder / copied.name
            original = json.loads(historical.read_text())
            proof = receipt["testRevalidation"]
            assert proof["historicalFileSha256"] == digest(historical.read_bytes())
            assert proof["historicalRequestSha256"] == original["requestSha256"]
            assert receipt["values"] == original["values"]
            assert receipt["requestSha256"] != original["requestSha256"]


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
    assert_except_documented_presentations_equal(cached["target"], result)
    assert_non_hs_edits_equal(cached, result, campaign.blueprint(sid))
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
    assert_except_documented_presentations_equal(cached["target"], result)
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
        phone_changes = assert_preserved_phone_identities(cached["target"], result)
        assert {
            p
            for p in set(before) | set(after)
            if before.get(p) != after.get(p) and ".hsCodes[" not in p
        } <= {path} | phone_changes
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
        assert_except_documented_presentations_equal(cached["target"], result)
    assert quote not in result["joinedRawText"]
    edits = [e for e in result["proof"]["edits"] if e["key"] == key]
    assert len(edits) == 1 and edits[0]["before"] == quote and edits[0]["after"] == ""
    assert result["validation"]["exactReplay"] and result["validation"]["currentSchema"]
