from copy import deepcopy

import pytest

from document_ocr.synthesis.curated_contacts import (
    apply_contacts,
    contact_output_type,
    contact_parties,
    contact_prompt,
    request_hash,
    unpack_contacts,
)


def parties():
    company = {
        "name": "OLD LTD",
        "country": "TURKIYE",
        "contactDetails": {
            "emailAddresses": ["INFO@OLDLIFT.COM"],
            "websiteUrls": ["WWW.OLDLIFT.COM"],
        },
    }
    source = {"documentPatch": {"parties": {"consignee": company, "notifyParties": [company]}}}
    target = deepcopy(source)
    for party in (
        target["documentPatch"]["parties"]["consignee"],
        target["documentPatch"]["parties"]["notifyParties"][0],
    ):
        party.update(name="HUAXIN TEXTILE WORKS", country="CHINA")
    return source, target, contact_parties(source, target, "sample")


def test_company_contacts_preserve_shared_identity_and_exact_scope():
    source, target, requests = parties()
    assert len(requests) == 1
    schema = contact_output_type(requests)
    values = unpack_contacts(
        schema.model_validate(
            {"p0": {"c0": "sales@huaxintextile.com", "c1": "WWW.HUAXINTEXTILE.COM"}}
        ),
        requests,
    )["sample"]
    assert len(values) == 4
    fixed = apply_contacts(target, values, requests)
    assert (
        fixed["documentPatch"]["parties"]["consignee"]
        == fixed["documentPatch"]["parties"]["notifyParties"][0]
    )
    assert (
        target["documentPatch"]["parties"]["consignee"]["contactDetails"]
        == source["documentPatch"]["parties"]["consignee"]["contactDetails"]
    )
    with pytest.raises(ValueError, match="incomplete or unrelated"):
        apply_contacts(target, {**values, "documentPatch.extra": "bad"}, requests)
    assert request_hash(requests) != request_hash(contact_parties(source, source, "sample"))
    title = deepcopy(target)
    title["documentPatch"]["parties"]["consignee"].update(
        name="Huaxin Textile Works", country="China"
    )
    # The other occurrence remains uppercase; identity grouping and the exact
    # contact-generation request must still agree, without editing either label.
    assert request_hash(requests) == request_hash(contact_parties(source, title, "sample"))
    assert title["documentPatch"]["parties"]["consignee"]["name"] == "Huaxin Textile Works"


@pytest.mark.parametrize(
    "email,website",
    [
        ("INFO@OLDLIFT.COM", "WWW.OLDLIFT.COM"),
        ("sales@company-123.example", "www.company-123.example"),
        ("sales@huaxintextile.com\nNEW LINE", "www.huaxintextile.com"),
        ("sales@huaxintextile.com", "javascript:alert(1)"),
    ],
)
def test_bad_contact_outputs_fail_without_rewriting_or_fallback(email, website):
    _, _, requests = parties()
    with pytest.raises(ValueError):
        unpack_contacts(
            contact_output_type(requests).model_validate({"p0": {"c0": email, "c1": website}}),
            requests,
        )


@pytest.mark.parametrize("email", ["sales@huaxintextile.com", "huaxin.sales@gmail.com"])
def test_email_and_website_domains_are_independent(email):
    _, _, requests = parties()
    website = "www.huaxin-works.com"
    output = contact_output_type(requests).model_validate({"p0": {"c0": email, "c1": website}})
    values = unpack_contacts(output, requests)["sample"]
    assert set(values.values()) == {email, website}


def test_contact_only_roles_preserve_absent_identity_and_do_not_merge_unrelated_roles():
    source = {
        "documentPatch": {
            "parties": {
                "consignee": {"name": "NAMED IMPORTER"},
                "notifyParties": [
                    {"name": "NAMED IMPORTER"},
                    {"contactDetails": {"emailAddresses": ["oldnotify@gmail.com"]}},
                    {"contactDetails": {"emailAddresses": ["oldnotify@gmail.com"]}},
                ],
            }
        }
    }
    before = deepcopy(source)
    requests = contact_parties(source, source, "sample")
    assert len(requests) == 2
    assert all(p.name is None for p in requests)
    assert [p.role for p in requests] == [
        "documentPatch.parties.notifyParties[1]",
        "documentPatch.parties.notifyParties[2]",
    ]
    assert "unnamed documentPatch.parties.notifyParties[1]" in contact_prompt(requests)
    assert "company identity is not supplied" in contact_prompt(requests)
    values = unpack_contacts(
        contact_output_type(requests).model_validate(
            {"p0": {"c0": "newnotify@gmail.com"}, "p1": {"c0": "othernotify@gmail.com"}}
        ),
        requests,
    )["sample"]
    result = apply_contacts(source, values, requests)
    assert source == before
    assert result["documentPatch"]["parties"]["consignee"] == {"name": "NAMED IMPORTER"}
    assert all("name" not in p for p in result["documentPatch"]["parties"]["notifyParties"][1:])
    assert len(values) == 2
    assert request_hash(requests) == request_hash(contact_parties(source, source, "sample"))


@pytest.mark.parametrize("reference", ["shipper", "consignee"])
def test_symbolic_notify_contacts_use_referenced_company_without_copying_identity(reference):
    source = {
        "documentPatch": {
            "parties": {
                reference: {"name": "OLD COMPANY", "country": "EGYPT"},
                "notifyParties": [
                    {
                        "sameAs": reference,
                        "contactDetails": {"emailAddresses": ["M.AROUSSY@ITACEGYPT.NET"]},
                    }
                ],
            }
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["parties"][reference].update(
        name="Atlantic Steel Distribution (UK) Limited", country="United Kingdom"
    )
    before = deepcopy(target)
    requests = contact_parties(source, target, "sample")
    assert len(requests) == 1
    party = requests[0]
    assert party.name == "ATLANTIC STEEL DISTRIBUTION (UK) LIMITED"
    assert party.country == "UNITED KINGDOM"
    assert party.role is None
    assert party.fields[0].paths == (
        "documentPatch.parties.notifyParties[0].contactDetails.emailAddresses[0]",
    )
    assert "COMPANY p0: ATLANTIC STEEL DISTRIBUTION (UK) LIMITED" in contact_prompt(requests)
    assert "COUNTRY: UNITED KINGDOM" in contact_prompt(requests)
    assert "identity is not supplied" not in contact_prompt(requests)
    values = unpack_contacts(
        contact_output_type(requests).model_validate({"p0": {"c0": "atlantic.sales@gmail.com"}}),
        requests,
    )["sample"]
    fixed = apply_contacts(target, values, requests)
    assert fixed["documentPatch"]["parties"]["notifyParties"][0] == {
        "sameAs": reference,
        "contactDetails": {"emailAddresses": ["atlantic.sales@gmail.com"]},
    }
    assert target == before
    assert request_hash(requests) != request_hash(contact_parties(source, source, "sample"))
    target["documentPatch"]["parties"][reference]["name"] = party.name
    target["documentPatch"]["parties"][reference]["country"] = party.country
    assert request_hash(requests) == request_hash(contact_parties(source, target, "sample"))


def test_symbolic_notify_groups_repeated_contact_with_named_company_but_keeps_override():
    source, target, _ = parties()
    source["documentPatch"]["parties"]["notifyParties"][0] = {
        "sameAs": "consignee",
        "contactDetails": {"emailAddresses": ["INFO@OLDLIFT.COM", "SECOND@OLDLIFT.COM"]},
    }
    target["documentPatch"]["parties"]["notifyParties"][0] = deepcopy(
        source["documentPatch"]["parties"]["notifyParties"][0]
    )
    requests = contact_parties(source, target, "sample")
    assert len(requests) == 1
    assert len(requests[0].fields) == 3
    assert len(requests[0].fields[0].paths) == 2
    assert requests[0].fields[2].original == "SECOND@OLDLIFT.COM"
    assert len(requests[0].fields[2].paths) == 1


def test_symbolic_contact_reference_does_not_invent_missing_company_name():
    source = {
        "documentPatch": {
            "parties": {
                "consignee": {"addressLine": "12 ROAD", "country": "INDIA"},
                "notifyParties": [
                    {
                        "sameAs": "consignee",
                        "contactDetails": {"emailAddresses": ["oldnotify@gmail.com"]},
                    }
                ],
            }
        }
    }
    requests = contact_parties(source, source, "sample")
    assert requests[0].name is None
    assert requests[0].country == "INDIA"
    assert requests[0].role == "documentPatch.parties.notifyParties[0]"
    source["documentPatch"]["parties"]["consignee"].pop("country")
    assert contact_parties(source, source, "sample")[0].country == ""


@pytest.mark.parametrize(
    "change",
    [
        "unsupported",
        "missing",
        "null",
        "empty",
        "nested",
        "invalid_name",
        "invalid_country",
        "wrong_role",
    ],
)
def test_invalid_explicit_contact_references_fail_without_mutation(change):
    source, target, _ = parties()
    # The shared-company fixture intentionally aliases both party dictionaries;
    # invalid-reference cases need independent concrete and symbolic parties.
    for document in (source, target):
        document["documentPatch"]["parties"]["notifyParties"] = deepcopy(
            document["documentPatch"]["parties"]["notifyParties"]
        )
    source["documentPatch"]["parties"]["consignee"].pop("contactDetails")
    target["documentPatch"]["parties"]["consignee"].pop("contactDetails")
    notify = target["documentPatch"]["parties"]["notifyParties"][0]
    notify.pop("name")
    notify.pop("country")
    notify["sameAs"] = "consignee"
    members = target["documentPatch"]["parties"]
    if change == "unsupported":
        notify["sameAs"] = "deliveryAgent"
    elif change == "missing":
        members.pop("consignee")
    elif change == "null":
        members["consignee"] = None
    elif change == "empty":
        members["consignee"] = {}
    elif change == "nested":
        members["consignee"]["sameAs"] = "shipper"
    elif change == "invalid_name":
        members["consignee"]["name"] = " "
    elif change == "invalid_country":
        members["consignee"]["country"] = 1
    elif change == "wrong_role":
        members["consignee"]["sameAs"] = "shipper"
        source["documentPatch"]["parties"]["consignee"]["contactDetails"] = {
            "emailAddresses": ["wrongrole@example.com"]
        }
    before = deepcopy(target)
    with pytest.raises(ValueError, match=r"contact.*reference"):
        contact_parties(source, target, "sample")
    assert target == before


@pytest.mark.parametrize(
    "email",
    [
        "R..khouri@lebanesemedsupplies.com",
        ".sales@huaxintextile.com",
        "sales.@huaxintextile.com",
        "sales@-huaxintextile.com",
        "sales@huaxintextile..com",
        "sales@huaxintextile.cöm",
        "équipe@huaxintextile.com",
        "a" * 65 + "@huaxintextile.com",
    ],
)
def test_native_contact_output_rejects_invalid_mailbox_before_host_unpack(email):
    _, _, requests = parties()
    output = contact_output_type(requests)
    assert (
        output.model_json_schema()["$defs"]["CompanyContacts0"]["properties"]["c0"]["pattern"]
        != r"^\S+$"
    )
    with pytest.raises(ValueError):
        output.model_validate({"p0": {"c0": email, "c1": "www.huaxintextile.com"}})


def test_ascii_dot_atom_mailbox_accepts_supported_punctuation():
    _, _, requests = parties()
    output = contact_output_type(requests).model_validate(
        {"p0": {"c0": "sales.eu+forwarding@huaxintextile.com", "c1": "www.huaxintextile.com"}}
    )
    assert unpack_contacts(output, requests)["sample"]
