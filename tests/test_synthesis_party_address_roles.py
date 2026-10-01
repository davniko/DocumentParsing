"""Source-role contracts protect party address ownership across render modes."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler import descendant as render
from document_ocr.synthesis.template_compiler.complete_pipeline import (
    _duplicate_party_address_slot_aliases,
    _generation_schema,
    _party_slot_generation_fields,
    _validate_shared_party_address_values,
)
from document_ocr.synthesis.template_compiler.party_address_roles import (
    AddressSlotRole,
    EmbeddedLocalityRole,
    FixedAdmin1Frame,
    PartyAddressGroupRole,
    PartyAddressRoleCertificate,
    fixed_source_only_replacements,
    propose_address_label_from_source_projection,
    required_immutable_admin1,
    role_values_cover_repetitions,
    slot_request_fields,
    slot_value_key,
    validate_address_label_context,
    validate_role_certificate,
)
from document_ocr.synthesis.template_compiler.request_batches import (
    lexical_payload,
    semantic_aliases,
)


def _role_fixture() -> tuple[PartyAddressRoleCertificate, SimpleNamespace, bytes, bytes]:
    source = b"ROAD SC"
    template_bytes = b"pinned-template"
    source_slot = SimpleNamespace(slot_id="slot_1", source_text="ROAD", byte_start=0, byte_end=4)
    address_binding = SimpleNamespace(
        binding_id="binding_1",
        logical_key="party:shipper:0:address",
        group_kind="party",
        value_kind="address",
        group_key="party:shipper:0",
        target_paths=("documentPatch.parties.shipper.address",),
        occurrences=(source_slot,),
        realization=SimpleNamespace(slots=(SimpleNamespace(repeat_group_index=None),)),
    )
    template = SimpleNamespace(
        document_id="doc_source",
        source_sha256=hashlib.sha256(source).hexdigest(),
        byte_template=SimpleNamespace(slots=(source_slot,)),
        bindings=(address_binding,),
        auxiliary_semantic_plan=SimpleNamespace(entities=()),
    )
    certificate = PartyAddressRoleCertificate(
        schema_version=1,
        source_document_id="doc_source",
        source_sha256=hashlib.sha256(source).hexdigest(),
        template_sha256=hashlib.sha256(template_bytes).hexdigest(),
        groups=(
            PartyAddressGroupRole(
                group_key="party:shipper:0",
                binding_id="binding_1",
                binding_target_paths=("documentPatch.parties.shipper.address",),
                address_target_path="documentPatch.parties.shipper.address",
                separately_bound_locality_fields=(),
                slots=(
                    AddressSlotRole(
                        slot_id="slot_1",
                        source_text="ROAD",
                        roles=("street_or_site",),
                        evidence="ROAD is the source address span",
                    ),
                ),
                fixed_admin1_frames=(
                    FixedAdmin1Frame(
                        byte_start=5,
                        byte_end=7,
                        source_text="SC",
                        country_code="BR",
                        geonames_admin1_code="BR.26",
                        evidence="SC is fixed source text for Santa Catarina",
                    ),
                ),
            ),
        ),
    )
    return certificate, template, source, template_bytes


def test_source_role_certificate_pins_address_and_immutable_region() -> None:
    certificate, template, source, template_bytes = _role_fixture()
    validate_role_certificate(
        certificate,
        template=template,
        source=source,
        template_bytes=template_bytes,
    )
    assert required_immutable_admin1(certificate, "party:shipper:0") == ("BR", "BR.26")
    with pytest.raises(ValueError, match="different source bytes"):
        validate_role_certificate(
            certificate,
            template=template,
            source=b"ROAD SP",
            template_bytes=template_bytes,
        )
    with pytest.raises(ValueError, match="unknown party group"):
        required_immutable_admin1(certificate, "party:consignee:0")


def test_typed_address_label_retains_only_certified_fixed_region() -> None:
    certificate, _, _, _ = _role_fixture()
    group = certificate.groups[0]
    validate_address_label_context(
        group,
        source_target_address="ROAD SC",
        candidate="LONG AVENUE SC",
        values={"slot_1": "LONG AVENUE"},
    )
    with pytest.raises(ValueError, match="changed source-owned fixed label context"):
        validate_address_label_context(
            group,
            source_target_address="ROAD SC",
            candidate="LONG AVENUE SP",
            values={"slot_1": "LONG AVENUE"},
        )
    with pytest.raises(ValueError, match="unowned text"):
        validate_address_label_context(
            group.model_copy(update={"fixed_admin1_frames": ()}),
            source_target_address="ROAD SC",
            candidate="LONG AVENUE SC",
            values={"slot_1": "LONG AVENUE"},
        )


def test_role_values_reject_wrong_city_country_and_retired_source() -> None:
    group = PartyAddressGroupRole(
        group_key="party:shipper:0",
        binding_id="binding_1",
        binding_target_paths=("documentPatch.parties.shipper.address",),
        address_target_path="documentPatch.parties.shipper.address",
        separately_bound_locality_fields=("city", "country"),
        slots=(
            AddressSlotRole(
                slot_id="street",
                source_text="AJA ROAD",
                roles=("street_or_site",),
                retire_on_geography_change=("AJA",),
                evidence="AJA ROAD is the source street",
            ),
            AddressSlotRole(
                slot_id="region",
                source_text="ZHEJIANG",
                roles=("administrative_region",),
                administrative_level=1,
                evidence="ZHEJIANG is the source province",
            ),
        ),
    )
    valid = {"street": "18 Beacon Street", "region": "Massachusetts"}
    assert (
        role_values_cover_repetitions(
            group,
            valid,
            geography_changed=True,
            sampled_city="Dorchester",
            sampled_country="United States",
            administrative_names_by_level={1: "Massachusetts"},
        )
        == "18 Beacon Street Massachusetts"
    )
    with pytest.raises(ValueError, match="retired source locality"):
        role_values_cover_repetitions(
            group,
            {**valid, "street": "AJA Center"},
            geography_changed=True,
            sampled_city="Dorchester",
            sampled_country="United States",
            administrative_names_by_level={1: "Massachusetts"},
        )
    with pytest.raises(ValueError, match="different source address role"):
        role_values_cover_repetitions(
            group,
            {**valid, "street": "18 Beacon Street, Dorchester"},
            geography_changed=True,
            sampled_city="Dorchester",
            sampled_country="United States",
            administrative_names_by_level={1: "Massachusetts"},
        )
    with pytest.raises(ValueError, match="contradicts sampled region"):
        role_values_cover_repetitions(
            group,
            {**valid, "region": "Nebraska"},
            geography_changed=True,
            sampled_city="Dorchester",
            sampled_country="United States",
            administrative_names_by_level={1: "Massachusetts"},
        )


def test_role_values_reject_spelled_ordinal_and_close_city_aliases() -> None:
    group = PartyAddressGroupRole(
        group_key="party:consignee:0",
        binding_id="binding_address",
        binding_target_paths=("documentPatch.parties.consignee.address",),
        address_target_path="documentPatch.parties.consignee.address",
        separately_bound_locality_fields=("city", "country"),
        slots=(
            AddressSlotRole(
                slot_id="street",
                source_text="ZONE A6 BLOCK 16",
                roles=("street_or_site",),
                evidence="Only the source zone and block occupy this address slot",
            ),
        ),
    )
    for alias in ("TEN RAMADAN CITY", "TENTH OF RAMADHAN CITY"):
        with pytest.raises(ValueError, match="different source address role"):
            role_values_cover_repetitions(
                group,
                {"street": "UNIT 6, FOOD PROCESSING COMPLEX, " + alias},
                geography_changed=True,
                sampled_city="10TH OF RAMADAN CITY",
                sampled_country="EGYPT",
                administrative_names_by_level={},
            )
    assert role_values_cover_repetitions(
        group,
        {"street": "UNIT 6, FOOD PROCESSING COMPLEX"},
        geography_changed=True,
        sampled_city="10TH OF RAMADAN CITY",
        sampled_country="EGYPT",
        administrative_names_by_level={},
    ) == "UNIT 6, FOOD PROCESSING COMPLEX"


def test_repeated_party_address_roles_cannot_diverge() -> None:
    party = {
        "name": "ACME",
        "address": "KOBRY EL KOBBA",
        "city": "CAIRO",
        "country": "EGYPT",
    }
    source = {"documentPatch": {"parties": {"consignee": party, "notifyParties": [party]}}}
    sampled_party = {**party, "address": "18 AL NASR AVENUE, CAIRO"}
    proposed = {
        "documentPatch": {
            "parties": {"consignee": sampled_party, "notifyParties": [sampled_party]}
        }
    }
    paths = (
        "documentPatch.parties.consignee",
        "documentPatch.parties.notifyParties[0]",
    )
    _validate_shared_party_address_values(
        source, proposed, [(paths[0], "HELIOPOLIS"), (paths[1], "heliopolis")]
    )
    with pytest.raises(ValueError, match="repeated party acquired conflicting"):
        _validate_shared_party_address_values(
            source, proposed, [(paths[0], "HELIOPOLIS"), (paths[1], "NASR CITY")]
        )


def test_identical_party_and_role_slots_are_generated_once() -> None:
    party = {"name": "ACME", "address": "KOBRY EL KOBBA", "city": "CAIRO", "country": "EGYPT"}
    source_target = {
        "documentPatch": {"parties": {"consignee": party, "notifyParties": [party]}}
    }
    sampled_party = {**party, "address": "18 AL NASR AVENUE, CAIRO"}
    proposed = {
        "documentPatch": {
            "parties": {"consignee": sampled_party, "notifyParties": [sampled_party]}
        }
    }
    binding_rows = []
    groups = []
    lexical = []
    for name, slot_id in (("consignee", "slot_a"), ("notifyParties[0]", "slot_b")):
        path = f"documentPatch.parties.{name}.address"
        binding_id = "binding_" + slot_id
        group = PartyAddressGroupRole(
            group_key="party:" + name,
            binding_id=binding_id,
            binding_target_paths=(path,),
            address_target_path=path,
            separately_bound_locality_fields=("city", "country"),
            slots=(
                AddressSlotRole(
                    slot_id=slot_id,
                    source_text="KOBRY EL KOBBA",
                    roles=("district_or_neighborhood",),
                    evidence="A separately printed city follows the source district",
                ),
            ),
        )
        groups.append(group)
        binding_rows.append(
            SimpleNamespace(
                binding_id=binding_id,
                logical_key=path,
                target_paths=(path,),
                group_kind="party",
                value_kind="address",
            )
        )
        lexical.append({"key": binding_id, "paths": [path], "partyPath": path[:-8]})
    source = SimpleNamespace(
        target=source_target, template=SimpleNamespace(bindings=binding_rows)
    )
    certificate = PartyAddressRoleCertificate(
        schema_version=1,
        source_document_id="doc_source",
        source_sha256="a" * 64,
        template_sha256="b" * 64,
        groups=tuple(groups),
    )
    expected_alias = {slot_value_key("binding_slot_b", "slot_b"): slot_value_key(
        "binding_slot_a", "slot_a"
    )}
    assert _duplicate_party_address_slot_aliases(
        source, certificate, lexical, proposed
    ) == expected_alias
    fields = _party_slot_generation_fields(
        source, certificate, lexical, SimpleNamespace(), None, proposed
    )
    assert [field["key"] for field in fields] == [slot_value_key("binding_slot_a", "slot_a")]
    changed = {
        "documentPatch": {
            "parties": {
                "consignee": sampled_party,
                "notifyParties": [{**sampled_party, "contactDetails": {"phoneNumbers": ["1"]}}],
            }
        }
    }
    assert _duplicate_party_address_slot_aliases(source, certificate, lexical, changed) == {}


def test_source_only_state_slot_requires_exact_owner_and_span() -> None:
    certificate, template, source, template_bytes = _role_fixture()
    state_slot = SimpleNamespace(slot_id="slot_state", source_text="SC", byte_start=5, byte_end=7)
    state_binding = SimpleNamespace(
        binding_id="binding_state",
        group_kind="party",
        value_kind="location",
        group_key="party:shipper:0",
        target_paths=(),
        occurrences=(state_slot,),
    )
    template.bindings += (state_binding,)
    template.byte_template.slots += (state_slot,)
    frame = (
        certificate.groups[0]
        .fixed_admin1_frames[0]
        .model_copy(
            update={"source_only_binding_id": "binding_state", "source_only_slot_id": "slot_state"}
        )
    )
    group = certificate.groups[0].model_copy(update={"fixed_admin1_frames": (frame,)})
    certificate = certificate.model_copy(update={"groups": (group,)})
    validate_role_certificate(
        certificate, template=template, source=source, template_bytes=template_bytes
    )
    assert fixed_source_only_replacements(certificate) == {"binding_state": {"slot_state": "SC"}}
    invalid = frame.model_copy(update={"source_only_binding_id": "binding_1"})
    invalid_group = group.model_copy(update={"fixed_admin1_frames": (invalid,)})
    invalid_certificate = certificate.model_copy(update={"groups": (invalid_group,)})
    with pytest.raises(ValueError, match="not a location auxiliary"):
        validate_role_certificate(
            invalid_certificate, template=template, source=source, template_bytes=template_bytes
        )


def test_named_district_locality_is_licensed_but_unrelated_city_is_not() -> None:
    district = AddressSlotRole(
        slot_id="address",
        source_text="VILLA 172, NEW CAIRO",
        roles=("street_or_site", "district_or_neighborhood"),
        embedded_locality_roles=(
            EmbeddedLocalityRole(
                field="city",
                source_locality="CAIRO",
                source_phrase="NEW CAIRO",
                prefix="NEW ",
                suffix="",
                semantic_role="district_or_neighborhood",
                replacement_policy="same_locality_only",
                evidence="NEW CAIRO is a named district in the source address",
            ),
        ),
        evidence="The address prints street and named district",
    )
    group = PartyAddressGroupRole(
        group_key="party:shipper:0",
        binding_id="binding_1",
        binding_target_paths=("documentPatch.parties.shipper.address",),
        address_target_path="documentPatch.parties.shipper.address",
        separately_bound_locality_fields=("city",),
        slots=(district,),
    )
    assert (
        role_values_cover_repetitions(
            group,
            {"address": "BUILDING 28 STREET 90 NEW CAIRO"},
            geography_changed=False,
            sampled_city="CAIRO",
            sampled_country=None,
            administrative_names_by_level={},
        )
        == "BUILDING 28 STREET 90 NEW CAIRO"
    )
    with pytest.raises(ValueError, match="different source address role"):
        role_values_cover_repetitions(
            group,
            {"address": "BUILDING 28 STREET 90 NEW CAIRO, CAIRO"},
            geography_changed=False,
            sampled_city="CAIRO",
            sampled_country=None,
            administrative_names_by_level={},
        )


def test_source_projection_keeps_fixed_region_in_target_address() -> None:
    group = PartyAddressGroupRole(
        group_key="party:shipper:0",
        binding_id="binding_1",
        binding_target_paths=("documentPatch.parties.shipper.address",),
        address_target_path="documentPatch.parties.shipper.address",
        separately_bound_locality_fields=("city", "country"),
        slots=(
            AddressSlotRole(
                slot_id="street",
                source_text="RUA BLUMENAU, 658 TERREO SALA 01",
                roles=("street_or_site", "floor_or_unit"),
                evidence="Printed street and room",
            ),
        ),
    )
    assert propose_address_label_from_source_projection(
        group,
        source_target_address="RUA BLUMENAU, 658 TERREO SALA 01 SC",
        values={"street": "RUA DAS FLORES, 20 SALA 4"},
    ) == "RUA DAS FLORES, 20 SALA 4 SC"
    with pytest.raises(ValueError, match="uniquely expose"):
        propose_address_label_from_source_projection(
            group,
            source_target_address=(
                "RUA BLUMENAU, 658 TERREO SALA 01 SC "
                "RUA BLUMENAU, 658 TERREO SALA 01 SC"
            ),
            values={"street": "RUA DAS FLORES, 20 SALA 4"},
        )
    with pytest.raises(ValueError, match="different source address role"):
        role_values_cover_repetitions(
            group,
            {"street": "BUILDING 28 STREET 90 NEW HANGZHOU"},
            geography_changed=True,
            sampled_city="HANGZHOU",
            sampled_country=None,
            administrative_names_by_level={},
        )


def test_typed_party_components_render_to_their_own_source_slots() -> None:
    binding = SimpleNamespace(
        binding_id="binding_1",
        group_kind="party",
        value_kind="address",
        target_paths=("documentPatch.parties.shipper.address",),
        realization=SimpleNamespace(adapter="copy"),
        occurrences=(
            SimpleNamespace(slot_id="slot_street", source_text="PAOJIANG INDUSTRIAL ZONE"),
            SimpleNamespace(slot_id="slot_region", source_text="ZHEJIANG"),
        ),
    )
    target = {"documentPatch": {"parties": {"shipper": {"address": "Eastern Park, Anhui"}}}}
    auxiliary = {
        slot_value_key("binding_1", "slot_street"): "Eastern Park",
        slot_value_key("binding_1", "slot_region"): "Anhui",
    }
    output = render._render_target_binding(binding, target, auxiliary_values=auxiliary)
    assert output.replacements == {
        "slot_street": "EASTERN PARK",
        "slot_region": "ANHUI",
    }
    with pytest.raises(ValueError, match="lacks one or more"):
        render._render_target_binding(
            binding,
            target,
            auxiliary_values={slot_value_key("binding_1", "slot_street"): "Eastern Park"},
        )


def test_party_slot_is_a_unique_provider_field_with_its_own_geography() -> None:
    certificate, template, _, _ = _role_fixture()
    ordinary = {
        "key": "field_0000",
        "paths": ["documentPatch.parties.shipper.address"],
        "source": "ROAD",
        "constraints": [],
    }
    fields = slot_request_fields(certificate, template, (ordinary,))
    assert len(fields) == 1
    schema = _generation_schema(fields, [])
    assert schema.model_validate({fields[0]["key"]: "NEW ROAD"}).model_dump() == {
        fields[0]["key"]: "NEW ROAD"
    }
    payload = {
        "requestedFields": fields,
        "structuredScenario": {
            "documentPatch": {"parties": {"shipper": {"address": "ROAD"}}}
        },
        "partyGeography": {
            "documentPatch.parties.shipper": {
                "country_code": "BR",
                "country_name": "Brazil",
                "city": "Blumenau",
            }
        },
    }
    aliases = semantic_aliases(payload, tuple(schema.model_fields))
    transmitted = lexical_payload(payload, aliases)
    assert transmitted["requestedFields"][0]["expectedGeography"]["city"] == "Blumenau"
    assert transmitted["structuredScenario"]["documentPatch"]["parties"]["shipper"][
        "address"
    ] == {"generateAddressSlots": [aliases[fields[0]["key"]]]}
