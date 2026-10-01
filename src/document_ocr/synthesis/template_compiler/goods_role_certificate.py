"""Fail-closed source-role admission for compiled goods templates."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.raw_text_template import render_compiled_template

from .models import CertifiedSemanticTemplate
from .one_to_one_pallet_bags import certify as certify_pallet_bags
from .source_only_lot_ids import validate_binding as validate_lot_ids

_HASH = re.compile(r"^[0-9a-f]{64}$")
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class OriginalAdditionalInformation(BaseModel):
    model_config = _STRICT

    group_index: int = Field(ge=0)
    value_index: int = Field(ge=0)
    value: str = Field(min_length=1)
    role: Literal[
        "private_pallet_bag_level",
        "private_lot_identifier_list",
        "description_labelled_reference",
        "source_only_three_set_vehicle_logistics",
    ]


class GoodsRoleCertificate(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    document_id: str = Field(min_length=1)
    admission: Literal[
        "source_has_no_additional_information",
        "certified_private_cargo_roles",
        "certified_description_reference",
        "certified_three_set_vehicle_consolidation",
    ]
    source_sha256: str
    source_label_sha256: str
    template_sha256: str
    original_source_label_sha256: str
    original_additional_information: tuple[OriginalAdditionalInformation, ...]
    critic_stage_sha256: str | None
    changed_scenario_proof_sha256: str | None
    fixed_inner_package_category: Literal["PACKAGE_BAG"] | None
    reviewed_lexical_contract_sha256: str | None = None
    compiler_draft_sha256: str | None = None

    @model_validator(mode="after")
    def complete_evidence(self) -> GoodsRoleCertificate:
        for name in (
            "source_sha256",
            "source_label_sha256",
            "template_sha256",
            "original_source_label_sha256",
        ):
            if _HASH.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"role certificate has invalid {name}")
        for name in (
            "critic_stage_sha256",
            "changed_scenario_proof_sha256",
            "reviewed_lexical_contract_sha256",
            "compiler_draft_sha256",
        ):
            value = getattr(self, name)
            if value is not None and _HASH.fullmatch(value) is None:
                raise ValueError(f"role certificate has invalid {name}")
        if self.admission == "source_has_no_additional_information":
            if (
                self.original_additional_information
                or self.critic_stage_sha256 is not None
                or self.changed_scenario_proof_sha256 is not None
                or self.fixed_inner_package_category is not None
                or self.reviewed_lexical_contract_sha256 is not None
                or self.compiler_draft_sha256 is not None
            ):
                raise ValueError("empty goods role certificate contains unreviewed role evidence")
        elif self.admission == "certified_private_cargo_roles":
            if (
                not self.original_additional_information
                or self.critic_stage_sha256 is None
                or self.changed_scenario_proof_sha256 is None
                or self.fixed_inner_package_category != "PACKAGE_BAG"
                or self.reviewed_lexical_contract_sha256 is not None
                or self.compiler_draft_sha256 is not None
                or any(
                    row.role
                    in {"description_labelled_reference", "source_only_three_set_vehicle_logistics"}
                    for row in self.original_additional_information
                )
            ):
                raise ValueError("certified private cargo role lacks complete physical proof")
        elif self.admission == "certified_three_set_vehicle_consolidation":
            if (
                len(self.original_additional_information) != 1
                or self.original_additional_information[0].role
                != "source_only_three_set_vehicle_logistics"
                or self.critic_stage_sha256 is None
                or self.changed_scenario_proof_sha256 is None
                or self.compiler_draft_sha256 is None
                or self.fixed_inner_package_category is not None
                or self.reviewed_lexical_contract_sha256 is not None
            ):
                raise ValueError("three-set vehicle consolidation lacks complete proof")
        elif (
            len(self.original_additional_information) != 1
            or self.original_additional_information[0].role != "description_labelled_reference"
            or self.critic_stage_sha256 is None
            or self.changed_scenario_proof_sha256 is None
            or self.reviewed_lexical_contract_sha256 is None
            or self.fixed_inner_package_category is not None
            or self.compiler_draft_sha256 is not None
        ):
            raise ValueError("description reference role lacks complete source and scenario proof")
        return self


def _additional_information(target: Mapping[str, Any]) -> tuple[tuple[int, int, str], ...]:
    patch = target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source target lacks documentPatch")
    groups = patch.get("cargoGroups", [])
    if not isinstance(groups, list):
        raise ValueError("source target lacks cargoGroups")
    result = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError("source target contains a non-object cargo group")
        values = group.get("additionalInformation", [])
        if not isinstance(values, list):
            raise ValueError("cargo additionalInformation must be a list")
        for value_index, value in enumerate(values):
            if not isinstance(value, str) or not value:
                raise ValueError("cargo additionalInformation contains a non-text value")
            result.append((group_index, value_index, value))
    return tuple(result)


def _validate_description_reference(
    *,
    certificate: GoodsRoleCertificate,
    source: bytes,
    target: dict[str, Any],
    original_target: dict[str, Any],
    template_bytes: bytes,
    critic_stage_bytes: bytes,
    changed_scenario_proof_bytes: bytes,
    reviewed_lexical_contract_bytes: bytes,
) -> None:
    """Prove an AAI product reference belongs to two adjacent goods-description slots.

    This admission is deliberately narrower than a generic AAI-to-description
    migration: a pinned source label, physical split binding, reviewed lexical
    recipe, and reproducible changed shipment must all agree.
    """
    from . import complete_targets as targets
    from . import descendant as render
    from . import lexical_facts, lexical_partitions, realization_contract
    from .cargo_scenarios import CargoScenario
    from .latest_target import latest_target_from_source

    entry = certificate.original_additional_information[0]
    old_groups = original_target["documentPatch"]["cargoGroups"]
    new_groups = target["documentPatch"]["cargoGroups"]
    if (
        len(old_groups) != 1
        or len(new_groups) != 1
        or entry.group_index != 0
        or entry.value_index != 0
    ):
        raise ValueError("description reference requires one source-owned goods item and AAI")
    old_description = old_groups[0].get("description")
    if not isinstance(old_description, str) or not old_description:
        raise ValueError("description reference lacks the source goods description")
    expected = deepcopy(original_target)
    group = expected["documentPatch"]["cargoGroups"][0]
    group.pop("additionalInformation")
    group["description"] = f"{old_description} {entry.value}"
    if target != expected:
        raise ValueError("description reference cleanup changed unrelated source labels")

    critic = json.loads(critic_stage_bytes)
    verdict = critic.get("verdict", {})
    if (
        critic.get("sourceDocumentId") != certificate.document_id
        or critic.get("usage", {}).get("requests") != 1
        or verdict.get("sourceRoleCorrect") is not True
        or verdict.get("changedScenarioCorrect") is not True
        or verdict.get("productReferenceBoundToGoods") is not True
        or verdict.get("remainingConcern") is not None
        or verdict.get("sourceEvidenceQuote") != f"{old_description}\n{entry.value}"
    ):
        raise ValueError("description reference independent review is incomplete")

    proof = json.loads(changed_scenario_proof_bytes)
    sidecar = json.loads(reviewed_lexical_contract_bytes)
    if (
        proof.get("sourceDocumentId") != certificate.document_id
        or proof.get("baseCatalogSourceSha256") != certificate.source_sha256
        or proof.get("candidateSourceLabelSha256") != certificate.source_label_sha256
        or proof.get("candidateTemplateSha256") != certificate.template_sha256
        or proof.get("reviewedLexicalContractSha256")
        != certificate.reviewed_lexical_contract_sha256
        or proof.get("sourceDescription") != group["description"]
        or proof.get("sourceBytesUnchangedOutsideTwoSlots") is not True
        or proof.get("sourceRoleAndTargetCompatibilityPassed") is not True
        or set(sidecar) != {certificate.document_id}
    ):
        raise ValueError("description reference proof does not pin this case")
    sample_id = proof.get("sampleId")
    seed = proof.get("seed")
    identity = proof.get("sampledGoodsIdentity")
    if (
        not isinstance(sample_id, str)
        or not sample_id
        or type(seed) is not int
        or not isinstance(identity, str)
        or not identity
    ):
        raise ValueError("description reference proof lacks replay inputs")

    template = CertifiedSemanticTemplate.model_validate_json(template_bytes, strict=True)
    if (
        template.document_id != certificate.document_id
        or template.source_sha256 != certificate.source_sha256
    ):
        raise ValueError("description reference template identity differs")
    source_case = targets.SourceTemplate(
        document_id=certificate.document_id,
        source=source,
        source_target=target,
        target=latest_target_from_source(target),
        template=realization_contract.effective_realization_template(template),
        original_template_sha256=certificate.template_sha256,
    )
    fields = targets.lexical_contract(source_case)
    fragments = [field for field in fields if "cargoFragment" in field]
    if len(fragments) != 2 or [field["source"] for field in fragments] != [
        old_description,
        entry.value,
    ]:
        raise ValueError("description reference lacks its exact two source fragments")
    contract = lexical_facts.CargoLexicalContract.model_validate_json(
        json.dumps(sidecar[certificate.document_id]), strict=True
    )
    if (
        set(contract.fragments) != {field["key"] for field in fragments}
        or contract.references
        or contract.fragments[fragments[0]["key"]].role != "identity"
        or contract.fragments[fragments[0]["key"]].identity_indices != (0,)
        or contract.fragments[fragments[1]["key"]].role != "labelled_reference"
    ):
        raise ValueError("description reference has an unreviewed lexical role")
    lexical_facts.require_cargo_recipes(fragments, contract)

    (goods,) = source_case.target["documentPatch"]["cargoGroups"]
    scenario = CargoScenario(
        target=deepcopy(source_case.target),
        identities={goods["groupId"]: [{"requiredDescription": identity}]},
        dangerous_goods_facts=(),
        receipt={},
    )
    plan = lexical_facts.prepare(
        source_case,
        fragments,
        scenario=scenario,
        projection=None,
        contract=contract,
        sample_id=sample_id,
        seed=seed,
    )
    bound_target = plan.preview(source_case, scenario.target, fragments)
    new_description = bound_target["documentPatch"]["cargoGroups"][0]["description"]
    if new_description != proof.get("sampledDescription") or entry.value in new_description:
        raise ValueError("description reference replay changed or retained source product ID")
    render._validate_target_compatibility(
        source=source,
        source_target=target,
        target=bound_target,
        template=source_case.template,
        pending_auxiliary_keys=frozenset(),
    )
    bindings = [
        binding
        for binding in source_case.template.bindings
        if tuple(binding.target_paths) == ("documentPatch.cargoGroups[0].description",)
    ]
    if len(bindings) != 1 or len(bindings[0].occurrences) != 2:
        raise ValueError("description reference has no unique two-slot physical binding")
    binding = bindings[0]
    slots = sorted(binding.occurrences, key=lambda item: item.byte_start)
    between = source[slots[0].byte_end : slots[1].byte_start]
    heading = source[max(0, slots[0].byte_start - 1000) : slots[0].byte_start]
    if (
        [slot.source_text for slot in slots] != [old_description, entry.value]
        or between.strip()
        or b"\n" not in between
        or re.search(rb"Description of Goods", heading, re.I) is None
    ):
        raise ValueError("description reference is not adjacent in the printed goods block")
    partition = lexical_partitions.partition(binding)
    if partition is None:
        raise ValueError("description reference lacks the split realization contract")
    auxiliary = {field["auxiliaryKey"]: plan.values[field["key"]] for field in fragments}
    edits = lexical_partitions.render_parts(partition, new_description, auxiliary, source=source)
    if edits != proof.get("changedPhysicalSlots"):
        raise ValueError("description reference replay changed physical slot values")
    changed_source = source
    for slot in reversed(slots):
        if source[slot.byte_start : slot.byte_end].decode() != slot.source_text:
            raise ValueError("description reference source slot differs from its OCR")
        changed_source = (
            changed_source[: slot.byte_start]
            + edits[slot.slot_id].encode()
            + changed_source[slot.byte_end :]
        )
    if sha256_bytes(changed_source) != proof.get("changedOcrSha256"):
        raise ValueError("description reference changed OCR does not replay")


def _validate_three_set_vehicle_consolidation(
    *,
    certificate: GoodsRoleCertificate,
    source: bytes,
    target: dict[str, Any],
    original_target: dict[str, Any],
    template_bytes: bytes,
    critic_stage_bytes: bytes,
    changed_scenario_proof_bytes: bytes,
    compiler_draft_bytes: bytes,
) -> None:
    """Certify one printed three-set vehicle shipment as one accounting item."""
    from .lexical_partitions import partition

    old_patch = original_target["documentPatch"]
    old_groups = old_patch.get("cargoGroups", [])
    old_packages = old_patch.get("cargoPackages", [])
    old_allocations = old_patch.get("cargoAllocationGroups", [])
    if (
        len(old_groups) != 3
        or len(old_packages) != 1
        or len(old_allocations) != 3
        or old_packages[0].get("quantity") != 3
        or old_packages[0].get("groupId") != old_groups[0].get("groupId")
        or old_groups[0].get("additionalInformation") != ["CONSOLIDATED CARGO"]
        or any(g.get("additionalInformation") for g in old_groups[1:])
        or any(g.get("grossWeight") or g.get("volume") for g in old_groups[1:])
        or not old_groups[0].get("grossWeight")
        or not old_groups[0].get("volume")
    ):
        raise ValueError("three-set vehicle source graph lacks one aggregate and three items")
    descriptions = [g.get("description") for g in old_groups]
    marks = [g.get("marksAndNumbers") for g in old_groups]
    if (
        any(not isinstance(text, str) or not text for text in descriptions)
        or any(not isinstance(values, list) or len(values) != 1 for values in marks)
        or len({values[0] for values in marks}) != 3
        or any(g.get("hsCodes") != old_groups[0].get("hsCodes") for g in old_groups)
        or any(g.get("dangerousGoods") != old_groups[0].get("dangerousGoods") for g in old_groups)
    ):
        raise ValueError("three vehicle rows lack distinct marks and common tariff/DG facts")
    container_numbers = []
    for group, allocation in zip(old_groups, old_allocations, strict=True):
        rows = allocation.get("allocations", [])
        if (
            allocation.get("groupId") != group.get("groupId")
            or allocation.get("coverage") != "unlinked_package_quantities"
            or allocation.get("packageIds") != []
            or len(rows) != 1
            or rows[0].get("packageQuantity") != 1
        ):
            raise ValueError("three-set source allocation is not one set per vehicle row")
        container_numbers.append(rows[0].get("containerNumber"))
    if not container_numbers[0] or len(set(container_numbers)) != 1:
        raise ValueError("three-set vehicle rows do not share one printed container")
    source_text = source.decode("utf-8")
    if (
        source_text.count("1 SET OF:") != 3
        or source_text.count("3 SETS") != 1
        or source_text.count("CONSOLIDATED CARGO") != 1
        or source_text.count("DANGEROUS CARGO") != 1
        or any(value[0] not in source_text for value in marks)
    ):
        raise ValueError("three-set vehicle source assertions or marks are absent")
    expected = deepcopy(original_target)
    patch = expected["documentPatch"]
    merged = patch["cargoGroups"][0]
    merged.pop("additionalInformation")
    merged["description"] = "; ".join(descriptions)
    merged["marksAndNumbers"] = [value[0] for value in marks]
    patch["cargoGroups"] = [merged]
    patch["cargoAllocationGroups"] = [
        {
            "groupId": merged["groupId"],
            "packageIds": [old_packages[0]["packageId"]],
            "coverage": "single_package_level",
            "allocations": [
                {"containerNumber": container_numbers[0], "packageQuantity": 3}
            ],
        }
    ]
    if target != expected:
        raise ValueError("three-set vehicle derivative changed unproved source facts")

    critic = json.loads(critic_stage_bytes)
    verdict = critic.get("output", {})
    pins = critic.get("inputPins", {})
    proof = json.loads(changed_scenario_proof_bytes)
    draft = json.loads(compiler_draft_bytes)
    if (
        critic.get("status") != "success"
        or critic.get("usage", {}).get("requests") != 1
        or verdict.get("verdict") != "pass"
        or not all(
            verdict.get(key) is True
            for key in (
                "mixed_goods_accounting_unit_supported",
                "three_vins_owned_without_motorbike_invention",
                "aggregate_mass_volume_owned_once",
                "consolidated_cargo_literal_remains_true",
                "dangerous_cargo_literal_remains_true",
                "changed_scenario_render_coherent",
            )
        )
        or pins.get("sourceOcrSha256") != certificate.source_sha256
        or pins.get("originalSourceLabelSha256") != certificate.original_source_label_sha256
        or pins.get("correctedSourceLabelSha256") != certificate.source_label_sha256
        or pins.get("changedScenarioProofSha256")
        != certificate.changed_scenario_proof_sha256
        or pins.get("candidateTemplateSha256") != certificate.compiler_draft_sha256
        or draft.get("sourceDocumentId") != certificate.document_id
        or draft.get("status") != "HOST_ONLY_PROBE_NOT_POST_EDIT_CRITIC_CERTIFIED"
        or canonical_json_bytes(draft.get("template")) != template_bytes
    ):
        raise ValueError("three-set vehicle independent critic or compiler draft differs")
    template = CertifiedSemanticTemplate.model_validate_json(template_bytes, strict=True)
    if (
        template.document_id != certificate.document_id
        or template.source_sha256 != sha256_bytes(source)
    ):
        raise ValueError("three-set vehicle template source identity differs")
    logistics = [
        binding
        for binding in template.bindings
        if binding.logical_key
        in {"agent:cargo:consolidated_cargo", "agent:cargo:dangerous_cargo_flag"}
    ]
    if (
        len(logistics) != 2
        or any(
            binding.render_mode != "literal_static" or binding.target_paths
            for binding in logistics
        )
        or {slot.source_text for binding in logistics for slot in binding.occurrences}
        != {"CONSOLIDATED CARGO", "DANGEROUS CARGO"}
    ):
        raise ValueError("source-only vehicle logistics statements are mutable")
    description_bindings = [
        binding
        for binding in template.bindings
        if tuple(binding.target_paths) == ("documentPatch.cargoGroups[0].description",)
    ]
    if len(description_bindings) != 1 or (
        plan := partition(description_bindings[0])
    ) is None or len(plan.source_parts) != 6:
        raise ValueError("three-set vehicle description lacks six printed fragments")
    slot_values = {slot.slot_id: slot.source_text for slot in template.byte_template.slots}
    source_replay, source_proof = render_compiled_template(
        source=source, template=template.byte_template, bindings=slot_values
    )
    changes = proof.get("changedSlotOutputs")
    if (
        proof.get("sourceDocumentId") != certificate.document_id
        or proof.get("candidateTemplateSha256") != certificate.compiler_draft_sha256
        or not all(
            proof.get(key) is True
            for key in (
                "sourceRoundTrip",
                "sourceOnlyLogisticsPresent",
                "noUnprintedModelSemicolons",
                "pageMarkersUnchanged",
                "literalRegionsExact",
                "allChangedDescriptionFragmentsPresent",
                "allChangedMarksPresent",
                "allChangedHouseAcidPresent",
            )
        )
        or source_replay != source
        or not source_proof.exact_literal_regions
        or not isinstance(changes, dict)
        or not changes
        or not set(changes).issubset(slot_values)
        or any(not isinstance(value, str) or not value for value in changes.values())
    ):
        raise ValueError("three-set vehicle changed-scenario evidence is incomplete")
    slot_values.update(changes)
    rendered, render_proof = render_compiled_template(
        source=source, template=template.byte_template, bindings=slot_values
    )
    text = rendered.decode("utf-8")
    changed_marks = proof.get("changedMarks")
    changed_acids = proof.get("changedHouseAcid")
    if (
        not render_proof.exact_literal_regions
        or not render_proof.page_markers_unchanged
        or text.count("CONSOLIDATED CARGO") != 1
        or text.count("DANGEROUS CARGO") != 1
        or text.count("1 SET OF:") != 3
        or not isinstance(changed_marks, list)
        or len(changed_marks) != 3
        or any(not isinstance(mark, str) or text.count(mark) != 1 for mark in changed_marks)
        or set(changed_marks) & {value[0] for value in marks}
        or not isinstance(changed_acids, dict)
        or len(changed_acids) != 3
        or any(
            not isinstance(value, str) or text.count(value) != 1
            for value in changed_acids.values()
        )
        or not isinstance(proof.get("renderedCargoExcerpt"), str)
        or proof["renderedCargoExcerpt"] not in text
        or re.search(r"MODEL:[^\n]*;", text)
    ):
        raise ValueError("three-set vehicle changed render contradicts its source role")


def validate_case(
    *,
    document_id: str,
    source: bytes,
    source_label_bytes: bytes,
    template_bytes: bytes,
    certificate_bytes: bytes,
    original_source_label_bytes: bytes | None = None,
    critic_stage_bytes: bytes | None = None,
    changed_scenario_proof_bytes: bytes | None = None,
    reviewed_lexical_contract_bytes: bytes | None = None,
    compiler_draft_bytes: bytes | None = None,
) -> GoodsRoleCertificate:
    certificate = GoodsRoleCertificate.model_validate_json(certificate_bytes, strict=True)
    if (
        certificate.document_id != document_id
        or certificate.source_sha256 != sha256_bytes(source)
        or certificate.source_label_sha256 != sha256_bytes(source_label_bytes)
        or certificate.template_sha256 != sha256_bytes(template_bytes)
    ):
        raise ValueError(f"goods role certificate does not pin case bytes: {document_id}")
    target = json.loads(source_label_bytes)
    if not isinstance(target, dict) or _additional_information(target):
        raise ValueError("admitted goods role source target still contains additionalInformation")
    if certificate.admission == "source_has_no_additional_information":
        if certificate.original_source_label_sha256 != certificate.source_label_sha256:
            raise ValueError("empty goods role certificate does not pin the original label")
        return certificate
    if (
        original_source_label_bytes is None
        or sha256_bytes(original_source_label_bytes) != certificate.original_source_label_sha256
    ):
        raise ValueError("private cargo role lacks its original source label")
    original_target = json.loads(original_source_label_bytes)
    if not isinstance(original_target, dict):
        raise ValueError("original goods role source label is not an object")
    declared = tuple(
        (row.group_index, row.value_index, row.value)
        for row in certificate.original_additional_information
    )
    if _additional_information(original_target) != declared:
        raise ValueError("original goods role inventory differs from its pinned source label")
    if (
        critic_stage_bytes is None
        or changed_scenario_proof_bytes is None
        or sha256_bytes(critic_stage_bytes) != certificate.critic_stage_sha256
        or sha256_bytes(changed_scenario_proof_bytes) != certificate.changed_scenario_proof_sha256
    ):
        raise ValueError("certified goods role lacks its pinned review/proof artifacts")
    if certificate.admission == "certified_description_reference":
        if (
            reviewed_lexical_contract_bytes is None
            or sha256_bytes(reviewed_lexical_contract_bytes)
            != certificate.reviewed_lexical_contract_sha256
        ):
            raise ValueError("description reference lacks its pinned lexical contract")
        _validate_description_reference(
            certificate=certificate,
            source=source,
            target=target,
            original_target=original_target,
            template_bytes=template_bytes,
            critic_stage_bytes=critic_stage_bytes,
            changed_scenario_proof_bytes=changed_scenario_proof_bytes,
            reviewed_lexical_contract_bytes=reviewed_lexical_contract_bytes,
        )
        return certificate
    if certificate.admission == "certified_three_set_vehicle_consolidation":
        if (
            compiler_draft_bytes is None
            or sha256_bytes(compiler_draft_bytes) != certificate.compiler_draft_sha256
        ):
            raise ValueError("three-set vehicle compiler draft hash differs")
        _validate_three_set_vehicle_consolidation(
            certificate=certificate,
            source=source,
            target=target,
            original_target=original_target,
            template_bytes=template_bytes,
            critic_stage_bytes=critic_stage_bytes,
            changed_scenario_proof_bytes=changed_scenario_proof_bytes,
            compiler_draft_bytes=compiler_draft_bytes,
        )
        return certificate
    cleaned = deepcopy(original_target)
    for group in cleaned["documentPatch"]["cargoGroups"]:
        group.pop("additionalInformation", None)
    if cleaned != target:
        raise ValueError("private cargo role cleanup changed unrelated source labels")
    critic = json.loads(critic_stage_bytes)
    proof = json.loads(changed_scenario_proof_bytes)
    if (
        critic.get("status") != "success"
        or critic.get("output", {}).get("verdict") != "pass"
        or critic.get("usage", {}).get("requests") != 1
        or not isinstance(proof, dict)
        or proof.get("sourceRoundTrip") is not True
        or proof.get("changedPackageCount") != {"from": 60, "to": 73}
    ):
        raise ValueError("private cargo role critic or changed-scenario proof is incomplete")
    template = CertifiedSemanticTemplate.model_validate_json(template_bytes, strict=True)
    if template.document_id != document_id or template.source_sha256 != sha256_bytes(source):
        raise ValueError("private cargo role template identity differs")
    constraint = certify_pallet_bags(
        source=source, bindings=template.bindings, source_target=target
    )
    if constraint is None or constraint.inner_category != certificate.fixed_inner_package_category:
        raise ValueError("private cargo role lacks its source-pinned pallet/bag contract")
    lot_bindings = [b for b in template.bindings if b.value_kind == "lot_identifier_list"]
    for binding in lot_bindings:
        validate_lot_ids(binding, source=source, target=target)
    roles = tuple(row.role for row in certificate.original_additional_information)
    if roles.count("private_pallet_bag_level") != 1 or roles.count(
        "private_lot_identifier_list"
    ) != len(lot_bindings):
        raise ValueError("private cargo role inventory does not match physical bindings")
    return certificate
