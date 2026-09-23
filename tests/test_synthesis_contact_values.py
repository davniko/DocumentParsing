from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.raw_text_template import format_envelope
from document_ocr.synthesis.template_compiler.contact_values import (
    mailbox,
    phone,
    phone_inside_literal_prefix,
    render_mailbox,
    source_phone_country,
    validate_mailbox,
    validate_phone,
)
from document_ocr.synthesis.template_compiler.descendant import _render_direct_auxiliary
from document_ocr.synthesis.template_compiler.synthetic_values import DeterministicValueFactory


@pytest.mark.parametrize(
    "source,start,value,expected",
    [
        (b"TEL:+201234567890", 5, "+31 20 123 4567", "31 20 123 4567"),
        (b"TEL:+201234567890", 4, "+31 20 123 4567", "+31 20 123 4567"),
        (b"TEL:0201234567", 4, "+31 20 123 4567", "+31 20 123 4567"),
        (b"0201234567", 0, "+31 20 123 4567", "+31 20 123 4567"),
    ],
)
def test_phone_prefix_has_exactly_one_owner(source, start, value, expected):
    assert phone_inside_literal_prefix(source=source, byte_start=start, rendered=value) == expected


@pytest.mark.parametrize(
    "source,region",
    [("(+202)\n23498086", "EG"), ("00447809141862", "GB"), ("+31-10-280-2555", "NL")],
)
def test_standalone_international_phone_keeps_its_proven_country(source, region):
    assert source_phone_country(source) == region
    b = NS(
        logical_key="phone",
        value_kind="phone",
        target_paths=(),
        occurrences=(NS(slot_id="s", source_text=source, render_policy="natural_text"),),
    )
    out = _render_direct_auxiliary(b, DeterministicStream(42, "test", "phone"))
    validate_phone(out.canonical_value, country_code=region)
    validate_phone(out.replacements["s"], country_code=region)


def test_unknown_local_country_is_explicitly_unresolved_not_random_digits():
    with pytest.raises(ValueError, match="explicit international"):
        source_phone_country("0571-471090")


def test_complete_phone_repeat_can_change_width_without_changing_meaning():
    from document_ocr.synthesis.template_compiler.descendant import (
        BindingOutput,
        _render_same_as_binding,
    )

    binding = NS(
        value_kind="phone",
        occurrences=(NS(slot_id="r", source_text="203 393 8888", render_policy="natural_text"),),
    )
    dependency = NS(occurrences=(NS(source_text="+203 393 8888"),))
    value = "+64 21 123 8765"
    out = _render_same_as_binding(binding, dependency, BindingOutput({}, value))
    assert out.canonical_value == value
    assert out.replacements["r"] == value


@pytest.mark.parametrize(
    "region", ["ES", "NL", "DE", "PL", "TR", "KR", "SG", "BE", "US", "CA", "BR", "NZ"]
)
def test_country_aware_phones_are_valid_reproducible_and_varied(region):
    generated = set()
    for i in range(100):
        stream = DeterministicStream(42, "phone-test", str(i))
        value = phone(stream, country_code=region)
        validate_phone(value, country_code=region)
        assert phone(stream, country_code=region) == value
        generated.add(value)
    assert len(generated) > 90


def test_independent_phone_and_address_share_geography_and_phone_members_differ():
    factory = DeterministicValueFactory(seed=42, document_id="s", target={})
    entity = NS(entity_id="delivery-agent", target_party_path=None)
    region = factory.geography_for_identity(entity.entity_id).country_code
    values = [
        factory.entity_textual(
            entity=entity, member=NS(field="phone", logical_key=k), country_codes={}
        )
        for k in ("phone:0", "phone:1")
    ]
    for value in values:
        validate_phone(value, country_code=region)
    assert values[0] != values[1]
    with pytest.raises(ValueError, match="entity country"):
        validate_phone("+70 061 8877 345", country_code=region)


@pytest.mark.parametrize(
    "source",
    [
        "info@agrofruitsarl@gmail.com",
        "HELMY3BDO@GMAIL.",
        "ONELOGISTICSEGYPTIMPORTOPE\nRATIONS\n@BAKERHUGHES.COM",
        "  old@somewhere.com  ",
        "old@some\r\nwhere.com",
    ],
)
def test_malformed_or_wrapped_sources_produce_one_complete_new_mailbox(source):
    stream = DeterministicStream(42, "contacts", "sample")
    b = NS(
        logical_key="email",
        value_kind="email",
        target_paths=(),
        occurrences=(NS(slot_id="s", source_text=source, render_policy="natural_text"),),
    )
    out = _render_direct_auxiliary(b, stream)
    value = "".join(out.replacements["s"].split())
    validate_mailbox(value)
    assert value.casefold() == out.canonical_value.casefold()
    assert value.casefold() != "".join(source.split()).casefold()
    before = format_envelope(source, render_policy="natural_text")
    after = format_envelope(out.replacements["s"], render_policy="natural_text")
    assert before == after


@pytest.mark.parametrize(
    "value",
    [
        "a@b@c.com",
        "a@example.",
        "a..b@example.com",
        ".a@example.com",
        "a@-example.com",
        "a@exa_mple.com",
        "a@localhost",
        "a b@example.com",
        "a@b..com",
        "a" * 65 + "@example.com",
    ],
)
def test_invalid_generated_contacts_fail_explicitly(value):
    with pytest.raises(ValueError, match="email"):
        validate_mailbox(value)
    with pytest.raises(ValueError, match="email"):
        render_mailbox("old@source.com", value)


def test_entity_mailbox_uses_new_party_and_is_deterministic():
    target = {"documentPatch": {"parties": {"shipper": {"name": "New Forest Products"}}}}
    factory = DeterministicValueFactory(seed=42, document_id="s", target=target)
    entity = NS(entity_id="shipper", target_party_path="documentPatch.parties.shipper")
    kwargs = dict(entity=entity, member=NS(field="email"), country_codes={})
    value = factory.entity_textual(**kwargs)
    assert "@new-forest-products-" in value
    assert factory.entity_textual(**kwargs) == value
    target["documentPatch"]["parties"]["shipper"]["contactDetails"] = {
        "emailAddresses": ["known@new.example"]
    }
    assert factory.entity_textual(**kwargs) == "known@new.example"


@pytest.mark.parametrize("organization", ["上海贸易", "Öst Export GmbH", "A" * 200, None])
def test_unicode_long_and_anonymous_organizations_have_valid_synthetic_domains(organization):
    value = mailbox(DeterministicStream(1, "contacts", "identity"), organization=organization)
    validate_mailbox(value)
    assert value.endswith(".example")


@pytest.mark.parametrize("source", ["NA", "N/A", "NOT PROVIDED", "NIL", "NONE", "---"])
def test_explicit_source_only_contact_absence_is_not_an_invented_mailbox(source):
    from document_ocr.synthesis.template_compiler.descendant import _explicit_unknown_placeholder

    b = NS(
        logical_key="absence",
        value_kind="email",
        target_paths=(),
        occurrences=(NS(slot_id="s", source_text=source, render_policy="natural_text"),),
    )
    assert _explicit_unknown_placeholder(b)
    result = _render_direct_auxiliary(b, DeterministicStream(42, "test", "absence"))
    assert result.replacements == {"s": source}
    b.target_paths = ("documentPatch.email",)
    assert not _explicit_unknown_placeholder(b)


def test_shared_prefix_phone_list_can_change_country_without_changing_label_shape():
    from document_ocr.synthesis.template_compiler.contact_values import validate_party_phones

    source = ["+20 3 4853366", "4853377", "4853388"]
    target = ["+31 227 684 510", "684 511", "684 512"]
    validate_party_phones(source, target, country_code="NL")
    with pytest.raises(ValueError, match="contradicts sampled country"):
        validate_party_phones(source, ["+20 3 4853366", *target[1:]], country_code="NL")
    with pytest.raises(ValueError, match="invalid full phone"):
        validate_party_phones([source[0]], ["684 511"], country_code="NL")
