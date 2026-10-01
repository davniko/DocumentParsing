from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.raw_text_template import validate_no_new_duplicate_commas
from document_ocr.synthesis.template_compiler.descendant import (
    _render_target_binding,
    _segment_without_repeated_literal_comma,
)
from document_ocr.synthesis.template_compiler.generation_contract import (
    validate_no_new_adjacent_party_localities,
    validate_party_address_locality_slots,
)


def _address_case(source_text: str, *, city_owns_comma: bool = False):
    source = source_text.encode("utf-8")
    first_text = "OLD HARBOR PARK"
    first = NS(
        slot_id="slot_0001",
        source_text=first_text,
        byte_start=0,
        byte_end=len(first_text),
        render_policy="natural_text",
    )
    region_start = source.index(b"REGION")
    region = NS(
        slot_id="slot_0003",
        source_text="REGION",
        byte_start=region_start,
        byte_end=region_start + len("REGION"),
        render_policy="natural_text",
    )
    city_start = source.index(b"CITY")
    if city_owns_comma:
        city_start -= 1
    city = NS(
        slot_id="slot_0002",
        source_text=source[city_start : city_start + 4 + int(city_owns_comma)].decode(),
        byte_start=city_start,
        byte_end=city_start + 4 + int(city_owns_comma),
    )
    template = NS(byte_template=NS(slots=(first, city, region)))
    binding = NS(
        group_kind="party",
        value_kind="address",
        target_paths=("documentPatch.address",),
        occurrences=(first, region),
        realization=NS(
            mode="segmented_surface",
            adapter="natural_text",
            target_values=(NS(source_value="OLD HARBOR PARK, REGION"),),
        ),
    )
    return source, template, binding


def test_segmented_address_reuses_literal_comma_without_doubling_it():
    source, template, binding = _address_case("OLD HARBOR PARK,CITY\nREGION")
    target = {"documentPatch": {"address": "NEW HARBOR PARK, ZHEJIANG"}}

    output = _render_target_binding(binding, target, source=source, template=template)

    assert output.replacements == {
        "slot_0001": "NEW HARBOR PARK",
        "slot_0003": "ZHEJIANG",
    }
    rendered = (
        output.replacements["slot_0001"] + source[binding.occurrences[0].byte_end :].decode()
    ).replace("REGION", output.replacements["slot_0003"])
    assert "PARK,CITY\nZHEJIANG" in rendered
    assert ",," not in rendered


def test_segmented_address_preserves_internal_and_unowned_comma_grammar():
    target = {"documentPatch": {"address": "NEW, HARBOR PARK, ZHEJIANG"}}
    source, template, binding = _address_case("OLD HARBOR PARK,CITY\nREGION")
    assert (
        _render_target_binding(binding, target, source=source, template=template).replacements[
            "slot_0001"
        ]
        == "NEW, HARBOR PARK"
    )

    source, template, binding = _address_case("OLD HARBOR PARK CITY\nREGION")
    assert (
        _render_target_binding(binding, target, source=source, template=template).replacements[
            "slot_0001"
        ]
        == "NEW, HARBOR PARK,"
    )

    source, template, binding = _address_case("OLD HARBOR PARK,CITY\nREGION", city_owns_comma=True)
    assert (
        _render_target_binding(binding, target, source=source, template=template).replacements[
            "slot_0001"
        ]
        == "NEW, HARBOR PARK,"
    )


def test_segmented_source_authentic_double_comma_is_not_rewritten():
    source = b"OLD HARBOR PARK,,CITY\nREGION"
    slot = NS(source_text="OLD HARBOR PARK,", byte_start=0, byte_end=16)
    template = NS(byte_template=NS(slots=(slot,)))
    assert (
        _segment_without_repeated_literal_comma(
            surface="NEW HARBOR PARK,", slot=slot, source=source, template=template
        )
        == "NEW HARBOR PARK,"
    )


def test_segmented_surface_requires_source_provenance():
    source, template, binding = _address_case("OLD HARBOR PARK,CITY\nREGION")
    target = {"documentPatch": {"address": "NEW HARBOR PARK, ZHEJIANG"}}
    with pytest.raises(ValueError, match="certified source and template"):
        _render_target_binding(binding, target)
    with pytest.raises(ValueError, match="certified source and template"):
        _render_target_binding(binding, target, source=source)
    with pytest.raises(ValueError, match="certified source and template"):
        _render_target_binding(binding, target, template=template)


def test_final_comma_guard_rejects_generated_seams_but_preserves_source_grammar():
    source, template, binding = _address_case("OLD HARBOR PARK,CITY\nREGION")
    slots = binding.occurrences
    with pytest.raises(ValueError, match="template boundary"):
        validate_no_new_duplicate_commas(
            source=source,
            template=template.byte_template,
            bindings={
                slots[0].slot_id: "NEW HARBOR PARK,",
                "slot_0002": "CITY",
                slots[1].slot_id: "ZHEJIANG",
            },
        )
    validate_no_new_duplicate_commas(
        source=source,
        template=template.byte_template,
        bindings={
            slots[0].slot_id: "NEW HARBOR PARK",
            "slot_0002": "CITY",
            slots[1].slot_id: "ZHEJIANG",
        },
    )

    source = b"HS CODE:123,,456"
    validate_no_new_duplicate_commas(source=source, template=NS(slots=()), bindings={})


def test_final_comma_guard_rejects_new_doubles_inside_a_slot():
    source = b"OLD,ITEM"
    template = NS(slots=(NS(slot_id="slot_0001", byte_start=0, byte_end=len(source)),))
    with pytest.raises(ValueError, match="source span"):
        validate_no_new_duplicate_commas(
            source=source,
            template=template,
            bindings={"slot_0001": "NEW,,ITEM"},
        )


def test_party_source_role_guard_rejects_locality_in_region_slot():
    address = NS(
        group_kind="party",
        group_key="party:shipper:0",
        value_kind="address",
        realization=NS(mode="segmented_surface"),
        target_paths=("documentPatch.parties.shipper.address",),
        occurrences=(
            NS(slot_id="a1", source_text="Old Street"),
            NS(slot_id="a2", source_text="ZHEJIANG"),
        ),
    )
    city = NS(
        group_kind="party",
        group_key="party:shipper:0",
        value_kind="city",
        realization=NS(mode="direct"),
        target_paths=("documentPatch.parties.shipper.city",),
        occurrences=(NS(slot_id="city", source_text="SHAOXING"),),
    )
    country = NS(
        group_kind="party",
        group_key="party:shipper:0",
        value_kind="country",
        realization=NS(mode="direct"),
        target_paths=("documentPatch.parties.shipper.country",),
        occurrences=(NS(slot_id="country", source_text="CHINA"),),
    )
    bindings = (address, city, country)
    validate_party_address_locality_slots(
        bindings=bindings,
        slot_bindings={
            "a1": "New Street",
            "a2": "ZHEJIANG",
            "city": "SHAOXING",
            "country": "CHINA",
        },
    )
    with pytest.raises(ValueError, match="different source address role"):
        validate_party_address_locality_slots(
            bindings=bindings,
            slot_bindings={
                "a1": "New Street",
                "a2": "CHINA",
                "city": "SHAOXING",
                "country": "CHINA",
            },
        )


def test_party_source_role_guard_checks_single_surface_with_separate_localities():
    address = NS(
        group_kind="party",
        group_key="party:consignee:0",
        value_kind="address",
        realization=NS(mode="single_surface"),
        target_paths=("documentPatch.parties.consignee.address",),
        occurrences=(NS(slot_id="address", source_text="WHOLE MARKET FOR VEGETABLES & FRUITS"),),
    )
    city = NS(
        group_kind="party",
        group_key="party:consignee:0",
        value_kind="location",
        target_paths=("documentPatch.parties.consignee.city",),
        occurrences=(NS(slot_id="city", source_text="ALEXANDRIA"),),
    )
    country = NS(
        group_kind="party",
        group_key="party:consignee:0",
        value_kind="location",
        target_paths=("documentPatch.parties.consignee.country",),
        occurrences=(NS(slot_id="country", source_text="EGYPT"),),
    )
    with pytest.raises(ValueError, match="different source address role"):
        validate_party_address_locality_slots(
            bindings=(address, city, country),
            slot_bindings={
                "address": "17 EL NASR AVENUE, ALEXANDRIA, EGYPT",
                "city": "ALEXANDRIA",
                "country": "EGYPT",
            },
        )


def test_party_source_role_guard_distinguishes_nested_locality_from_city_component():
    address = NS(
        group_kind="party",
        group_key="party:consignee:0",
        value_kind="address",
        realization=NS(mode="segmented_surface"),
        target_paths=("documentPatch.parties.consignee.address",),
        occurrences=(NS(slot_id="address", source_text="NEW CAIRO"),),
    )
    city = NS(
        group_kind="party",
        group_key="party:consignee:0",
        value_kind="location",
        target_paths=("documentPatch.parties.consignee.city",),
        occurrences=(NS(slot_id="city", source_text="CAIRO"),),
    )
    validate_party_address_locality_slots(
        bindings=(address, city),
        slot_bindings={"address": "NEW CAIRO", "city": "CAIRO"},
    )
    with pytest.raises(ValueError, match="different source address role"):
        validate_party_address_locality_slots(
            bindings=(address, city),
            slot_bindings={"address": "CAIRO", "city": "CAIRO"},
        )


def test_final_party_locality_guard_preserves_source_duplicates_only():
    target = {"documentPatch": {"parties": {"shipper": {"city": "SHAOXING", "country": "CHINA"}}}}
    with pytest.raises(ValueError, match="adjacent party city repetition"):
        validate_no_new_adjacent_party_localities(
            source=b"Old Street, SHAOXING, CHINA",
            rendered=b"New Street, SHAOXING, SHAOXING, CHINA",
            target=target,
        )
    validate_no_new_adjacent_party_localities(
        source=b"Old Street, SHAOXING, SHAOXING, CHINA",
        rendered=b"New Street, SHAOXING, SHAOXING, CHINA",
        target=target,
    )
