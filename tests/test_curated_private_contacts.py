from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import Occurrence, SourceContract, TargetBinding, Variable
from document_ocr.synthesis.curated_auxiliary import _render_recipe, _validate_recipe
from document_ocr.synthesis.curated_campaign import Campaign
from document_ocr.synthesis.curated_contacts import (
    ContactField,
    ContactParty,
    apply_contacts,
    contact_output_type,
    contact_parties,
    contact_prompt,
    contact_surfaces,
    request_hash,
    unpack_contacts,
)
from document_ocr.synthesis.curated_private_contacts import (
    private_contact_contract,
    source_contact_parties,
    validate_private_person_coverage,
)
from document_ocr.synthesis.generators import DeterministicStream


def test_unnamed_party_website_uses_form_context_without_original_hostname():
    party = ContactParty(
        "sample",
        None,
        "",
        (ContactField("c0", "website", "WWW.ORIGINALCOMPANY.COM", ("documentPatch.website",)),),
        "shipper",
    )
    prompt = contact_prompt([party])
    assert "ORIGINALCOMPANY.COM" not in prompt
    assert "WWW. prefix" in prompt
    assert "new fictional website" in prompt
    schema = contact_output_type([party]).model_json_schema()
    assert "New fictional website for the unnamed party" in str(schema)


@pytest.mark.parametrize(
    "section",
    ["email_paths", "generic_email_paths", "replace_domain_paths", "provider_domain_paths"],
)
def test_non_string_contact_paths_reject_without_an_unhashable_set_error(section):
    bp = blueprint()
    bp.ownership_data["private_contacts"][section] = [{"bad": "path"}]
    with pytest.raises(ValueError, match="invalid private contact ownership sections"):
        private_contact_contract(bp)


@pytest.mark.parametrize(
    "old,new",
    [
        ("tarek_madkour@old.com", "tarek.madkour@new.com"),
        ("AmrSalah@old.com", "amr.salah@new.com"),
        ("checkpoint577@old.com", "apc.checkpoint577@new.com"),
    ],
)
def test_personal_mailbox_identity_cannot_be_kept_under_a_new_domain(old, new):
    p = ContactParty(
        "sample",
        "NEW COMPANY",
        "",
        (ContactField("c0", "email", old, ("documentPatch.email",), True),),
    )
    with pytest.raises(ValueError, match="source mailbox identity"):
        unpack_contacts(contact_output_type([p]).model_validate({"p0": {"c0": new}}), [p])


def test_generic_mailbox_can_keep_department_with_new_domain():
    p = ContactParty(
        "sample",
        "NEW COMPANY",
        "",
        (ContactField("c0", "email", "info@old.com", ("documentPatch.email",)),),
    )
    assert unpack_contacts(
        contact_output_type([p]).model_validate({"p0": {"c0": "info@new.com"}}), [p]
    )


def test_person_context_is_supplied_and_hash_bound():
    before = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "name": "OLD CO",
                    "contactDetails": {
                        "contactName": "OLD PERSON",
                        "emailAddresses": ["old.person@old.com"],
                    },
                }
            }
        }
    }
    after = deepcopy(before)
    after["documentPatch"]["parties"]["consignee"]["contactDetails"]["contactName"] = "NEW PERSON"
    path = "documentPatch.parties.consignee.contactDetails.emailAddresses[0]"
    parties = contact_parties(before, after, "sample", personal_paths=frozenset({path}))
    assert "NEW CONTACT NAMES: NEW PERSON" in contact_prompt(parties)
    assert parties[0].fields[0].personal
    assert request_hash(parties) != request_hash(
        contact_parties(before, before, "sample", personal_paths=frozenset({path}))
    )
    assert "NONE" not in contact_prompt(
        contact_parties(
            before,
            {
                "documentPatch": {
                    "parties": {
                        "consignee": {"name": "NEW CO", "contactDetails": {"contactName": None}}
                    }
                }
            },
            "sample",
        )
    )
    with pytest.raises(ValueError, match="present email target"):
        contact_parties(before, after, "sample", personal_paths=frozenset({"documentPatch.wrong"}))


def blueprint():
    variable = Variable(
        key="person",
        kind="name",
        value="OLD PERSON",
        meaning="Source-only signatory",
        occurrences=[Occurrence(text="OLD PERSON", occurrence=1, presentation="text")],
        required_literals=[],
    )
    return SimpleNamespace(
        source="PUBLIC CARRIER\nOLD PERSON\nold.person@carrier.com\n",
        target={"documentPatch": {}},
        contract=SourceContract(
            variables=[variable],
            targets=[TargetBinding(path="documentPatch.billOfLadingNumber", expression="B123")],
            fixed_context="Public carrier remains unchanged",
        ),
        regions=[SimpleNamespace(key="mail", curated_key=None)],
        historical_bindings={
            "mail": {"target_paths": [], "occurrences": [{"source_text": "old.person@carrier.com"}]}
        },
        ownership_data={
            "identifier_privacy": {},
            "private_contacts": {
                "names": {"person": {"role": "carrier signatory"}},
                "emails": {
                    "mail": {
                        "company": "PUBLIC CARRIER",
                        "country": "",
                        "role": "carrier signatory",
                        "person_key": "person",
                        "domain_policy": "public_carrier",
                    }
                },
            },
        },
    )


def test_source_only_email_never_creates_a_party_or_target():
    bp = blueprint()
    parties = source_contact_parties(bp, "sample", {"person": "NEW PERSON"})
    output = contact_output_type(parties).model_validate({"p0": {"c0": "new.person@carrier.com"}})
    values = unpack_contacts(output, parties)["sample"]
    assert apply_contacts(bp.target, values, parties) == bp.target
    assert contact_surfaces(values) == {"mail": "new.person@carrier.com"}
    assert "NEW PERSON" in contact_prompt(parties)


def test_private_source_owner_failures_are_not_silently_skipped():
    bp = blueprint()
    with pytest.raises(ValueError, match="generated person"):
        source_contact_parties(bp, "sample", {})
    bp.source += "OLD PERSON\n"
    with pytest.raises(ValueError, match="unowned repeated"):
        private_contact_contract(bp)
    bp = blueprint()
    bp.target = {"documentPatch": {"email": "old.person@carrier.com"}}
    bp.historical_bindings["mail"]["target_paths"] = ["documentPatch.email"]
    with pytest.raises(ValueError, match="target email path"):
        private_contact_contract(bp)


def test_numeric_identifier_preserves_envelope_and_repeat_identity():
    recipe = {"identifier": "individual CPF", "format": "source"}
    stream = DeterministicStream(22, "privacy", "document")
    a = _render_recipe(recipe, None, {}, stream, "133.848.188-65", set())
    b = _render_recipe(recipe, None, {}, stream, "13384818865", set())
    assert a != "133.848.188-65"
    assert "".join(c for c in a if c.isdigit()) == b
    assert [c for c in a if not c.isdigit()] == [".", ".", "-"]
    with pytest.raises(ValueError, match="ASCII letters or digits"):
        _render_recipe(recipe, None, {}, stream, "É13384818865", set())
    with pytest.raises(ValueError, match="identifier formatting"):
        _validate_recipe({"identifier": "CPF", "format": "invented"})


def test_auxiliary_fax_aliases_share_one_subscriber_and_country():
    recipe = {
        "phone": "source",
        "country_code": "EG",
        "channel": "fax",
        "canonical_identity": "20222694915",
    }
    stream = DeterministicStream(22, "privacy", "document")
    a = _render_recipe(recipe, None, {}, stream, "22694915", set())
    b = _render_recipe(recipe, None, {}, stream, "+2 (02) 22694915", set())
    assert a == b and a.startswith("+20 ")
    with pytest.raises(ValueError, match="explicit country"):
        _validate_recipe({"phone": "source"})
    with pytest.raises(ValueError, match="contact channel"):
        _validate_recipe({"phone": "source", "country_code": "EG", "channel": "invented"})


def _validate_campaign_readiness(bp):
    for region in bp.regions:
        binding = bp.historical_bindings.get(region.key, {})
        region.kind = binding.get("value_kind", "other_text")
        region.target_paths = tuple(binding.get("target_paths", ()))
    campaign = SimpleNamespace(
        source_ids=("source",),
        blueprint=lambda _sid: bp,
        _blueprint_file_pins={},
        equipment_inventories={},
    )
    Campaign.validate_source_readiness(campaign)


@pytest.mark.parametrize("kind", ["person", "contact_name"])
def test_carrier_static_private_person_cannot_pass_campaign_readiness(kind):
    bp = blueprint()
    bp.regions.append(SimpleNamespace(key="static_signer", curated_key=None))
    bp.historical_bindings["static_signer"] = {
        "render_mode": "carrier_static",
        "value_kind": kind,
    }
    with pytest.raises(ValueError, match="private carrier person requires"):
        _validate_campaign_readiness(bp)


def test_public_carrier_company_remains_allowed_and_unchanged():
    bp = blueprint()
    bp.regions.append(SimpleNamespace(key="carrier_company", curated_key=None))
    bp.historical_bindings["carrier_company"] = {
        "render_mode": "carrier_static",
        "value_kind": "company",
    }
    before = deepcopy(bp)
    _validate_campaign_readiness(bp)
    assert bp.source == before.source
    assert bp.target == before.target
    assert bp.historical_bindings == before.historical_bindings


def test_explicit_generated_private_name_owner_passes_campaign_readiness():
    bp = blueprint()
    bp.regions.append(SimpleNamespace(key="person", curated_key="person"))
    assert private_contact_contract(bp)["names"] == {"person": {"role": "carrier signatory"}}
    validate_private_person_coverage(bp)
    _validate_campaign_readiness(bp)


@pytest.mark.parametrize("source_only", [False, True])
def test_private_contact_prompt_omits_source_person_but_keeps_generation_context(source_only):
    parties = [
        ContactParty(
            "sample",
            "PUBLIC CARRIER" if source_only else "NEW COMPANY",
            "BRAZIL",
            (
                ContactField(
                    "c0",
                    "email",
                    "julia_ortiz77@public-carrier.com",
                    () if source_only else ("documentPatch.email",),
                    True,
                    ("mail",) if source_only else (),
                ),
            ),
            "carrier signatory" if source_only else "consignee",
            ("NEW FICTIONAL PERSON",),
        )
    ]
    prompt = contact_prompt(parties)
    assert "julia" not in prompt.casefold()
    assert "ortiz" not in prompt.casefold()
    assert "julia_ortiz77@public-carrier.com" not in prompt
    assert "COUNTRY: BRAZIL" in prompt
    assert "NEW CONTACT NAMES: NEW FICTIONAL PERSON" in prompt
    assert "recipient words separated with _" in prompt
    assert "may include synthetic digits" in prompt
    assert "Use a new recipient identity." in prompt
    if source_only:
        assert "COMPANY p0: PUBLIC CARRIER" in prompt
        assert "public company domain public-carrier.com" in prompt
    else:
        assert "COMPANY p0: NEW COMPANY" in prompt
        assert "public-carrier.com" not in prompt


def _email_classification_blueprint():
    bp = blueprint()
    bp.target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "contactDetails": {
                        "emailAddresses": ["source.person@company.com", "info@company.com"]
                    }
                }
            }
        }
    }
    paths = tuple(
        f"documentPatch.parties.shipper.contactDetails.emailAddresses[{index}]" for index in (0, 1)
    )
    return bp, paths


@pytest.mark.parametrize("declare_private", [False, True])
def test_unclassified_source_mailboxes_block_campaign_readiness(declare_private):
    bp, paths = _email_classification_blueprint()
    if declare_private:
        bp.ownership_data["private_contacts"]["email_paths"] = [paths[0]]
    with pytest.raises(ValueError, match="every source email requires explicit private or generic"):
        _validate_campaign_readiness(bp)


def test_private_generic_mailbox_classifications_cannot_overlap():
    bp, paths = _email_classification_blueprint()
    bp.ownership_data["private_contacts"].update(
        email_paths=[paths[0]], generic_email_paths=list(paths)
    )
    with pytest.raises(ValueError, match="distinct existing email targets"):
        _validate_campaign_readiness(bp)


def test_complete_explicit_private_generic_mailbox_partition_passes_readiness():
    bp, paths = _email_classification_blueprint()
    bp.ownership_data["private_contacts"].update(
        email_paths=[paths[0]], generic_email_paths=[paths[1]], replace_domain_paths=list(paths)
    )
    before = deepcopy(bp)
    _validate_campaign_readiness(bp)
    assert bp.target == before.target
    assert bp.ownership_data == before.ownership_data


def test_unnamed_private_contact_gets_explicit_new_domain_instruction():
    party = ContactParty(
        "sample",
        None,
        "EGYPT",
        (
            ContactField(
                "c0", "email", "source.person@oldcompany.com", ("documentPatch.email",), True
            ),
        ),
        "consignee",
        ("NEW FICTIONAL PERSON",),
    )
    prompt = contact_prompt([party])
    assert "PARTY p0: unnamed consignee" in prompt
    assert "Invent a fictional business domain for the new contacts" in prompt
    assert "NEW CONTACT NAMES: NEW FICTIONAL PERSON" in prompt
    assert "source.person" not in prompt
    assert "oldcompany.com" not in prompt


@pytest.mark.parametrize("new", ["new.person@privatecompany.com", "new.person@PRIVATECOMPANY.COM"])
def test_private_company_domain_cannot_survive_a_changed_recipient(new):
    party = ContactParty(
        "sample",
        "NEW COMPANY",
        "",
        (
            ContactField(
                "c0",
                "email",
                "old.person@privatecompany.com",
                ("documentPatch.email",),
                personal=True,
                replace_domain=True,
            ),
        ),
    )
    with pytest.raises(ValueError, match="original private company domain"):
        unpack_contacts(contact_output_type([party]).model_validate({"p0": {"c0": new}}), [party])


def test_private_company_domain_is_replaced_with_a_new_host():
    party = ContactParty(
        "sample",
        "NEW COMPANY",
        "",
        (
            ContactField(
                "c0",
                "email",
                "old.person@privatecompany.com",
                ("documentPatch.email",),
                personal=True,
                replace_domain=True,
            ),
        ),
    )
    value = "new.person@newcompany.com"
    assert unpack_contacts(
        contact_output_type([party]).model_validate({"p0": {"c0": value}}), [party]
    ) == {"sample": {"documentPatch.email": value}}


def test_source_only_public_carrier_domain_is_not_forced_to_change():
    parties = source_contact_parties(blueprint(), "sample", {"person": "NEW PERSON"})
    assert not parties[0].fields[0].replace_domain
    value = "new.person@carrier.com"
    assert unpack_contacts(
        contact_output_type(parties).model_validate({"p0": {"c0": value}}), parties
    ) == {"sample": {"binding:mail": value}}


@pytest.mark.parametrize("policy", [None, "unclassified", False, [], {}])
def test_source_only_mailbox_requires_explicit_supported_domain_policy(policy):
    bp = blueprint()
    if policy is None:
        bp.ownership_data["private_contacts"]["emails"]["mail"].pop("domain_policy")
    else:
        bp.ownership_data["private_contacts"]["emails"]["mail"]["domain_policy"] = policy
    with pytest.raises(ValueError, match="source-only mailbox requires explicit"):
        private_contact_contract(bp)
    with pytest.raises(ValueError, match="source-only mailbox requires explicit"):
        source_contact_parties(bp, "sample", {"person": "NEW PERSON"})


@pytest.mark.parametrize("new", ["new.person@privatecompany.com", "new.person@PRIVATECOMPANY.COM"])
def test_source_only_private_company_mailbox_rejects_retained_domain(new):
    bp = blueprint()
    declaration = bp.ownership_data["private_contacts"]["emails"]["mail"]
    declaration.update(
        company=None, role="outside-block consignee contact", domain_policy="private_company"
    )
    bp.historical_bindings["mail"]["occurrences"][0]["source_text"] = (
        "old.person@privatecompany.com"
    )
    parties = source_contact_parties(bp, "sample", {"person": "NEW PERSON"})
    assert parties[0].fields[0].replace_domain
    prompt = contact_prompt(parties)
    assert "privatecompany.com" not in prompt
    assert "public company domain" not in prompt
    assert "new fictional business domain" in prompt
    with pytest.raises(ValueError, match="original private company domain"):
        unpack_contacts(contact_output_type(parties).model_validate({"p0": {"c0": new}}), parties)


def test_source_only_private_company_mailbox_accepts_complete_new_identity_without_target():
    bp = blueprint()
    declaration = bp.ownership_data["private_contacts"]["emails"]["mail"]
    declaration.update(company=None, domain_policy="private_company")
    parties = source_contact_parties(bp, "sample", {"person": "NEW PERSON"})
    value = "new.person@newcompany.com"
    result = unpack_contacts(
        contact_output_type(parties).model_validate({"p0": {"c0": value}}), parties
    )
    assert result == {"sample": {"binding:mail": value}}
    assert apply_contacts(bp.target, result["sample"], parties) == bp.target
    assert contact_surfaces(result["sample"]) == {"mail": value}


@pytest.mark.parametrize("policy", ["public_provider", "public_carrier"])
def test_source_only_public_mailbox_keeps_domain_but_replaces_recipient(policy):
    bp = blueprint()
    bp.ownership_data["private_contacts"]["emails"]["mail"]["domain_policy"] = policy
    if policy == "public_provider":
        bp.historical_bindings["mail"]["occurrences"][0]["source_text"] = "old.person@gmail.com"
    parties = source_contact_parties(bp, "sample", {"person": "NEW PERSON"})
    assert not parties[0].fields[0].replace_domain
    host = "gmail.com" if policy == "public_provider" else "carrier.com"
    value = "new.person@" + host
    assert unpack_contacts(
        contact_output_type(parties).model_validate({"p0": {"c0": value}}), parties
    ) == {"sample": {"binding:mail": value}}
    with pytest.raises(ValueError, match=r"source identity|source mailbox identity"):
        unpack_contacts(
            contact_output_type(parties).model_validate({"p0": {"c0": "old.person@" + host}}),
            parties,
        )


def test_source_only_domain_policy_changes_contact_request_identity():
    bp = blueprint()
    public = source_contact_parties(bp, "sample", {"person": "NEW PERSON"})
    bp.ownership_data["private_contacts"]["emails"]["mail"]["domain_policy"] = "private_company"
    private = source_contact_parties(bp, "sample", {"person": "NEW PERSON"})
    assert request_hash(public) != request_hash(private)


@pytest.mark.parametrize("host", ["gmail.com", "outlook.com", "hotmail.com"])
def test_public_email_provider_is_allowed_without_domain_replacement_flag(host):
    party = ContactParty(
        "sample",
        "NEW COMPANY",
        "",
        (ContactField("c0", "email", f"old.person@{host}", ("documentPatch.email",), True),),
    )
    value = f"new.person@{host}"
    assert unpack_contacts(
        contact_output_type([party]).model_validate({"p0": {"c0": value}}), [party]
    ) == {"sample": {"documentPatch.email": value}}


def test_generic_department_recipient_can_stay_when_private_company_host_changes():
    party = ContactParty(
        "sample",
        "NEW COMPANY",
        "",
        (
            ContactField(
                "c0",
                "email",
                "info@privatecompany.com",
                ("documentPatch.email",),
                replace_domain=True,
            ),
        ),
    )
    value = "info@newcompany.com"
    assert unpack_contacts(
        contact_output_type([party]).model_validate({"p0": {"c0": value}}), [party]
    ) == {"sample": {"documentPatch.email": value}}


@pytest.mark.parametrize(
    "invalid_path",
    [
        "documentPatch.wrong",
        "documentPatch.parties.shipper.name",
        "documentPatch.parties.shipper.contactDetails.websiteUrls[0]",
    ],
)
def test_domain_replacement_paths_must_be_present_email_targets(invalid_path):
    bp, _ = _email_classification_blueprint()
    shipper = bp.target["documentPatch"]["parties"]["shipper"]
    shipper["name"] = "OLD COMPANY"
    shipper["contactDetails"]["websiteUrls"] = ["www.oldcompany.com"]
    with pytest.raises(ValueError, match="present email target"):
        contact_parties(
            bp.target, bp.target, "sample", replace_domain_paths=frozenset({invalid_path})
        )


def test_domain_replacement_is_explicit_for_private_and_generic_email_targets():
    bp, paths = _email_classification_blueprint()
    parties = contact_parties(
        bp.target,
        bp.target,
        "sample",
        personal_paths=frozenset({paths[0]}),
        replace_domain_paths=frozenset(paths),
    )
    assert len(parties) == 1
    assert all(field.replace_domain for field in parties[0].fields)
    assert [field.personal for field in parties[0].fields] == [True, False]
    assert request_hash(parties) != request_hash(
        contact_parties(bp.target, bp.target, "sample", personal_paths=frozenset({paths[0]}))
    )
    bp.ownership_data["private_contacts"].update(
        email_paths=[paths[0]], generic_email_paths=[paths[1]], replace_domain_paths=list(paths)
    )
    _validate_campaign_readiness(bp)


@pytest.mark.parametrize("domain_section", ["replace_domain_paths", "provider_domain_paths"])
@pytest.mark.parametrize("declaration", ["unclassified", "duplicate", "absent"])
def test_private_domain_paths_require_a_distinct_classified_subset(declaration, domain_section):
    bp, paths = _email_classification_blueprint()
    domain_paths = {
        "unclassified": [paths[1]],
        "duplicate": [paths[0], paths[0]],
        "absent": ["documentPatch.wrong"],
    }[declaration]
    bp.ownership_data["private_contacts"].update(
        {"email_paths": [paths[0]], domain_section: domain_paths}
    )
    with pytest.raises(ValueError, match="distinct classified email target paths"):
        private_contact_contract(bp)


@pytest.mark.parametrize("declare_one", [False, True])
def test_unclassified_domains_block_readiness_with_complete_recipient_policy(declare_one):
    bp, paths = _email_classification_blueprint()
    bp.ownership_data["private_contacts"].update(
        email_paths=[paths[0]],
        generic_email_paths=[paths[1]],
        replace_domain_paths=[paths[0]] if declare_one else [],
    )
    with pytest.raises(ValueError, match="explicit private-company or public-provider domain"):
        _validate_campaign_readiness(bp)


def test_private_company_public_provider_domain_classifications_cannot_overlap():
    bp, paths = _email_classification_blueprint()
    bp.ownership_data["private_contacts"].update(
        email_paths=[paths[0]],
        generic_email_paths=[paths[1]],
        replace_domain_paths=[paths[0]],
        provider_domain_paths=list(paths),
    )
    with pytest.raises(ValueError, match="distinct classified email target paths"):
        _validate_campaign_readiness(bp)


def test_complete_private_company_public_provider_domain_partition_passes_readiness():
    bp, paths = _email_classification_blueprint()
    bp.target["documentPatch"]["parties"]["shipper"]["contactDetails"]["emailAddresses"][1] = (
        "info@gmail.com"
    )
    bp.ownership_data["private_contacts"].update(
        email_paths=[paths[0]],
        generic_email_paths=[paths[1]],
        replace_domain_paths=[paths[0]],
        provider_domain_paths=[paths[1]],
    )
    before = deepcopy(bp)
    _validate_campaign_readiness(bp)
    assert bp.target == before.target
    assert bp.ownership_data == before.ownership_data
