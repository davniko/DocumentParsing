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
