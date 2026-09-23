from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import descendant as r
from document_ocr.synthesis.template_compiler.geographic_context import (
    _terminal_country,
    address_country_context,
    generated_address_country,
)
from document_ocr.synthesis.template_compiler.models import AuxiliaryEntityMember
from document_ocr.synthesis.template_compiler.semantic_plan import resolve_geographic_members


def test_terminal_postal_letters_are_not_country_evidence():
    countries = {"bd": "BD", "egypt": "EG", "us": "US"}
    assert _terminal_country("14 Oosterhaven, 1671 BD", countries) is None
    assert _terminal_country("Main Street, 07020 US", countries) is None
    assert _terminal_country("Main Street, US", countries) == "US"
    assert _terminal_country("Main Street, 11211 Egypt", countries) == "EG"


@pytest.mark.parametrize(
    "address",
    [
        "Korea, Republic of",
        "42 Street, Seoul, KOREA, REPUBLIC\nOF",
        "Seoul; Korea, Republic of",
    ],
)
def test_country_alias_can_span_complete_comma_and_line_components(address):
    assert _terminal_country(address, {"korearepublicof": "KR"}) == "KR"


def test_country_components_cannot_skip_intervening_address_text():
    countries = {"korearepublicof": "KR", "jersey": "JE"}
    assert _terminal_country("Korea, Commercial Street, Republic of", countries) is None
    assert _terminal_country("Country unknown, New Jersey", countries) is None


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("18 Ataturk Industrial Avenue Kayseri Turkey", "TR"),
        ("14 Harbour Road Qurayyat 211 Oman", "OM"),
        ("18 Harbour Road Democratic Republic of the Congo", "CD"),
        ("18 Harbour Road Hong Kong China", "HK"),
        ("18 Harbour Road, Kaohsiung, Taiwan, Province of China", "TW"),
        ("18 Industrial Road New Mexico", None),
        ("18 Industrial Road New Jersey", None),
        ("18 Road, 1671 BD", None),
        ("18 Turkey Road", None),
        ("18 Road, US", "US"),
    ],
)
def test_flattened_generated_country_preserves_long_aliases_and_region_context(address, expected):
    countries = {
        "turkey": "TR",
        "oman": "OM",
        "congo": "CG",
        "republicofthecongo": "CG",
        "democraticrepublicofthecongo": "CD",
        "hongkongchina": "HK",
        "china": "CN",
        "mexico": "MX",
        "jersey": "JE",
        "bd": "BD",
        "us": "US",
        "taiwanprovinceofchina": "TW",
    }
    assert (
        generated_address_country(
            address, countries, region_names=frozenset({"newmexico", "newjersey"})
        )
        == expected
    )


def fixture(country_tail=", U.S.A."):
    raw = f"EDGEWATER, NEW JERSEY 07020{country_tail}\nCOUNTRY: UNITED STATES".encode()
    definitions = [
        ("city", "location", "city", "EDGEWATER"),
        ("address_region_postal", "address", "address", "NEW JERSEY"),
        ("postal_code", "identifier", "other_identifier", "07020"),
        ("country", "location", "country", "UNITED STATES"),
    ]
    bindings = []
    members = []
    for name, kind, field, text in definitions:
        start = raw.index(text.encode())
        key = "agent:buyer_" + name
        bindings.append(
            NS(
                logical_key=key,
                value_kind=kind,
                group_key="buyer",
                occurrences=(
                    NS(
                        source_text=text,
                        byte_start=start,
                        byte_end=start + len(text),
                        render_policy="free_text",
                    ),
                ),
            )
        )
        members.append(AuxiliaryEntityMember(logical_key=key, field=field))
    entity = NS(entity_id="buyer", members=tuple(members))
    entity.model_copy = lambda update: NS(entity_id="buyer", **update)
    plan = NS(entities=(entity,))
    plan.model_copy = lambda update: NS(**update)
    template = NS(bindings=tuple(bindings), auxiliary_semantic_plan=plan)
    return raw, template


def test_literal_country_is_pinned_and_separate_region_is_not_a_street():
    raw, template = fixture()
    countries = {"usa": "US", "unitedstates": "US"}
    plan = resolve_geographic_members(
        template.auxiliary_semantic_plan, template.bindings, countries
    )
    assert {m.logical_key: m.field for m in plan.entities[0].members} == {
        "agent:buyer_city": "city",
        "agent:buyer_address_region_postal": "region",
        "agent:buyer_postal_code": "postal_code",
        "agent:buyer_country": "country",
    }
    assert address_country_context(plan.entities[0], template, raw, countries) == {"US"}
    _, constraints = r._entity_country_code_context(template, countries, raw)
    assert constraints["buyer"].fixed_country_code == "US"


@pytest.mark.parametrize("tail", [", NEW JERSEY", "\nU.S.A.", ", U.S.A. Freight", " U.S.A."])
def test_country_context_does_not_cross_lines_or_match_partial_components(tail):
    raw, template = fixture(tail)
    plan = resolve_geographic_members(template.auxiliary_semantic_plan, template.bindings, {})
    assert not address_country_context(
        plan.entities[0], template, raw, {"usa": "US", "jersey": "JE"}
    )


def test_owned_country_is_mutable_not_literal_context():
    raw, template = fixture()
    start = raw.index(b"U.S.A.")
    template.bindings += (
        NS(logical_key="country_alias", occurrences=(NS(byte_start=start, byte_end=start + 6),)),
    )
    # Use the already correctly typed members, not a second resolver pass.
    template.auxiliary_semantic_plan.entities[0].members = (
        AuxiliaryEntityMember(logical_key="agent:buyer_postal_code", field="postal_code"),
    )
    assert not address_country_context(
        template.auxiliary_semantic_plan.entities[0], template, raw, {"usa": "US"}
    )
