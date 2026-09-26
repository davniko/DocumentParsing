from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.template_compiler.complete_pipeline import (
    _address_repair_requirements,
    _generation_schema,
    _validate_address_repairs,
)
from document_ocr.synthesis.template_compiler.descendant import _validate_unrepresented_party_facets
from document_ocr.synthesis.template_compiler.route_projection import (
    AddressCountryConflict,
    AddressPostalConflict,
    PartyGeography,
    RouteProjection,
    _explicit_numeric_postal_party_paths,
    require_route_contract,
    validate_generated_postal_values,
)


def fixture():
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "parties": {"consignee": {"country": "Egypt"}, "carrier": {"name": "FIXED"}},
            "route": {"portOfDischarge": {"name": "Alexandria"}},
        },
    }
    source = NS(document_id="source", target=target)
    row = RouteProjection(
        schema_version=1,
        sample_id="sample",
        source_document_id="source",
        source_target_sha256=sha256_bytes(canonical_json_bytes(target)),
        updates={
            "documentPatch.parties.consignee.country": "Netherlands",
            "documentPatch.route.portOfDischarge.name": "Rotterdam",
        },
        party_geography={},
        auxiliary_values={"consignee-country-code": "NL"},
        scenario={},
        rejected_candidates=(),
    )
    return source, row


def test_compiled_route_capability_uses_owned_target_or_dependency_paths():
    from document_ocr.synthesis.template_compiler.route_derivations import (
        supports_transshipment,
    )

    direct = NS(target_paths=("documentPatch.route.transshipmentPort.name",), dependency_paths=())
    latent = NS(target_paths=(), dependency_paths=("documentPatch.route.transshipmentPort",))
    other = NS(target_paths=("documentPatch.route.portOfLoading.name",), dependency_paths=())
    assert supports_transshipment((direct,))
    assert supports_transshipment((latent,))
    assert not supports_transshipment((other,))


def test_only_declared_shared_name_slots_constrain_route_locations():
    from document_ocr.synthesis.template_compiler.route_projection import _shared_route_locations

    source = NS(
        template=NS(
            bindings=(
                NS(
                    target_relationship="shared_value_equality",
                    target_paths=(
                        "documentPatch.route.portOfDischarge.name",
                        "documentPatch.route.placeOfDelivery.name",
                    ),
                ),
                NS(
                    target_relationship="shared_value_equality",
                    target_paths=(
                        "documentPatch.route.portOfLoading.country",
                        "documentPatch.route.placeOfReceipt.country",
                    ),
                ),
                NS(
                    target_relationship="single_target",
                    target_paths=("documentPatch.route.portOfLoading.name",),
                ),
            )
        )
    )
    assert _shared_route_locations(source) == (("portOfDischarge", "placeOfDelivery"),)


@pytest.mark.parametrize("second_country", ["EG", "CI", None])
def test_shared_phone_geography_is_checked_before_route_publication(second_country):
    from document_ocr.synthesis.template_compiler.route_projection import (
        _validate_shared_phone_context,
    )

    owners = ("documentPatch.parties.consignee", "documentPatch.parties.notifyParties[0]")
    source = NS(template=NS(bindings=(NS(target_paths=tuple(
        owner + ".contactDetails.phoneNumbers[0]" for owner in owners
    )),)))
    context = {owners[0]: PartyGeography(country_code="EG", country_name="Egypt", city="Cairo")}
    if second_country is not None:
        context[owners[1]] = PartyGeography(
            country_code=second_country, country_name="Example", city="Other city"
        )
    if second_country == "EG":
        _validate_shared_phone_context(source, context)
    else:
        with pytest.raises(ValueError, match="shared phone"):
            _validate_shared_phone_context(source, context)


def test_distinct_phone_bindings_do_not_merge_party_geography():
    from document_ocr.synthesis.template_compiler.route_projection import (
        _validate_shared_phone_context,
    )

    owners = ("documentPatch.parties.consignee", "documentPatch.parties.notifyParties[0]")
    source = NS(template=NS(bindings=tuple(
        NS(target_paths=(owner + ".contactDetails.phoneNumbers[0]",)) for owner in owners
    )))
    context = {owner: PartyGeography(country_code=code, country_name=code, city="Test")
               for owner, code in zip(owners, ("EG", "CI"), strict=True)}
    _validate_shared_phone_context(source, context)


@pytest.mark.parametrize(
    "fragment", [("nsw", "2529"), ("mi", "48174"), ("zhejiang",), ("21500",), ("tokyo", "section")]
)
def test_fixed_address_geographic_fragments_are_reviewed_before_route_sampling(
    monkeypatch, fragment
):
    from document_ocr.synthesis.template_compiler import route_projection as module

    binding = NS(
        logical_key="shipper-address",
        derivation=None,
        target_paths=("documentPatch.parties.shipper.address",),
        dependency_paths=(),
        dependency_bindings=(),
        realization=NS(mode="token_projected_surface"),
        occurrences=(),
    )
    source = NS(source=b"", target={}, template=NS(bindings=(binding,)))
    monkeypatch.setattr(
        module, "fixed_projection_ranges", lambda _: ((0, len(fragment), fragment),)
    )
    with pytest.raises(ValueError, match="fixed address fragment"):
        require_route_contract(source)


def test_projected_route_is_immutable_through_generation_and_does_not_extend_labels():
    source, row = fixture()
    target = row.apply(source, source.target)
    assert target["documentPatch"]["parties"]["consignee"]["country"] == "Netherlands"
    assert source.target["documentPatch"]["parties"]["consignee"]["country"] == "Egypt"
    assert target["documentPatch"]["parties"]["carrier"] == {"name": "FIXED"}
    row.validate_final(target, row.auxiliary_values)
    target["documentPatch"]["parties"]["consignee"]["country"] = "Egypt"
    with pytest.raises(ValueError, match="changed after generation"):
        row.validate_final(target, row.auxiliary_values)


def test_projection_rejects_source_drift_carrier_edits_new_label_leaves_and_missing_aux():
    source, row = fixture()
    changed = deepcopy(source.target)
    changed["documentPatch"]["parties"]["consignee"]["country"] = "China"
    with pytest.raises(ValueError, match="pinned source"):
        row.apply(NS(document_id="source", target=changed), changed)
    for path, error in [
        ("documentPatch.parties.carrier.name", "carrier"),
        ("documentPatch.newTrainingField", "introduce training-label leaves"),
    ]:
        invalid = row.model_copy(update={"updates": {path: "BAD"}})
        with pytest.raises(ValueError, match=error):
            invalid.apply(source, source.target)
    with pytest.raises(ValueError, match="source-only geography"):
        row.validate_final(row.apply(source, source.target), {})


def test_generated_addresses_and_phones_cannot_keep_the_old_country():
    source, row = fixture()
    row = row.model_copy(
        update={
            "party_geography": {
                "documentPatch.parties.consignee": PartyGeography(
                    country_code="NL", country_name="Netherlands", city="Rotterdam"
                )
            }
        }
    )
    target = row.apply(source, source.target)
    party = target["documentPatch"]["parties"]["consignee"]
    party["address"] = "New Street 4, EGYPT"
    with pytest.raises(ValueError, match="address contradicts"):
        row.validate_final(target, row.auxiliary_values, country_codes={"egypt": "EG"})
    party["address"] = "New Street 4, Rotterdam, Netherlands"
    party["contactDetails"] = {"phoneNumbers": ["+20 100 123 4567"]}
    with pytest.raises(ValueError, match="phone contradicts"):
        row.validate_final(target, row.auxiliary_values)
    party["contactDetails"]["phoneNumbers"] = ["+31 6 12345678"]
    row.validate_final(target, row.auxiliary_values)


def test_ambiguous_region_country_is_explicitly_repaired_not_ignored():
    source, row = fixture()
    path = "documentPatch.parties.consignee"
    row = row.model_copy(
        update={
            "updates": {},
            "party_geography": {
                path: PartyGeography(
                    country_code="US", country_name="United States", city="Cartersville"
                )
            },
        }
    )
    target = deepcopy(source.target)
    target["documentPatch"]["parties"]["consignee"]["address"] = (
        "287 Maple Ridge Drive, Cartersville, Georgia"
    )
    countries = {"georgia": "GE", "unitedstates": "US"}
    with pytest.raises(AddressCountryConflict) as caught:
        row.validate_final(target, row.auxiliary_values, country_codes=countries)
    fields = [dict(key="address", paths=[path + ".address"])]
    requirements = _address_repair_requirements(fields, [], [], caught.value)
    schema = _generation_schema(fields, requirements)
    with pytest.raises(ValueError, match="exact requested"):
        _validate_address_repairs(
            {"address": "287 Maple Ridge Drive, Cartersville, Georgia"}, requirements
        )
    repaired = "287 Maple Ridge Drive, Cartersville, Georgia, United States"
    assert schema.model_validate({"address": repaired}).address == repaired
    _validate_address_repairs({"address": repaired}, requirements)
    target["documentPatch"]["parties"]["consignee"]["address"] = repaired
    row.validate_final(target, row.auxiliary_values, country_codes=countries)
    assert (
        _address_repair_requirements(fields, [], requirements, ValueError("other field"))
        == requirements
    )


def test_foreign_postcode_is_repaired_before_address_acceptance():
    source, row = fixture()
    path = "documentPatch.parties.consignee"
    row = row.model_copy(
        update={
            "updates": {},
            "party_geography": {
                path: PartyGeography(
                    country_code="US", country_name="United States", city="Birmingham"
                )
            },
        }
    )
    target = deepcopy(source.target)
    target["documentPatch"]["parties"]["consignee"]["address"] = (
        "77 Foundry Road B11 4QJ, Birmingham, United States"
    )
    with pytest.raises(AddressPostalConflict) as caught:
        row.validate_final(target, row.auxiliary_values)
    fields = [dict(key="address", paths=[path + ".address"])]
    requirements = _address_repair_requirements(fields, [], [], caught.value)
    with pytest.raises(ValueError, match="incompatible postal code"):
        _validate_address_repairs(
            {"address": "77 Foundry Road B11 4QJ, Birmingham, United States"},
            requirements,
        )
    corrected = "77 Foundry Road, Birmingham, United States"
    _validate_address_repairs({"address": corrected}, requirements)
    target["documentPatch"]["parties"]["consignee"]["address"] = corrected
    row.validate_final(target, row.auxiliary_values)


def test_generated_postal_placeholders_fail_without_rejecting_japanese_suffixes():
    target = {"documentPatch": {"parties": {"consignee": {
        "country": "United Arab Emirates", "address": "12 Harbour Road, 00000 Dubai"
    }}}}
    with pytest.raises(AddressPostalConflict, match="all-zero placeholder"):
        validate_generated_postal_values(target, {})
    target["documentPatch"]["parties"]["consignee"]["address"] = (
        "12 Harbour Road, 809-0000 Fukuoka"
    )
    validate_generated_postal_values(target, {})
    with pytest.raises(ValueError, match="postal auxiliary"):
        validate_generated_postal_values(target, {"agent:party:postal_code": "AE-00000"})
    target["documentPatch"]["cargoGroups"] = [
        {"groupId": "g1", "marksAndNumbers": ["HK-00000 YUEN LONG"]}
    ]
    with pytest.raises(ValueError, match="cargo mark"):
        validate_generated_postal_values(target, {})


def test_only_explicit_numeric_postal_slots_constrain_postcode_free_routes():
    raw = b"POSTAL CODE: 22713\nP.O. BOX 12345\n"
    source = NS(
        source=raw,
        template=NS(bindings=(
            NS(target_paths=("documentPatch.parties.consignee.address",),
               occurrences=(NS(source_text="22713", byte_start=13, byte_end=18),)),
            NS(target_paths=("documentPatch.parties.shipper.address",),
               occurrences=(NS(source_text="12345", byte_start=28, byte_end=33),)),
        )),
    )
    assert _explicit_numeric_postal_party_paths(source) == frozenset({
        "documentPatch.parties.consignee"
    })


def test_unrepresented_context_can_only_be_pending_during_preliminary_validation():
    source = {"documentPatch": {"parties": {"consignee": {"country": "Egypt"}}}}
    target = {"documentPatch": {"parties": {"consignee": {"country": "Netherlands"}}}}
    template = NS(
        bindings=(),
        auxiliary_semantic_plan=NS(
            entities=[
                NS(
                    target_party_path="documentPatch.parties.consignee",
                    members=[NS(field="city", logical_key="missing-city")],
                )
            ]
        ),
    )
    arguments = dict(source_target=source, target=target, template=template)
    _validate_unrepresented_party_facets(
        **arguments, pending_auxiliary_keys=frozenset({"missing-city"})
    )
    with pytest.raises(ValueError, match="unresolved auxiliary identity"):
        _validate_unrepresented_party_facets(**arguments)
    with pytest.raises(ValueError, match="unknown auxiliary"):
        _validate_unrepresented_party_facets(
            **arguments, pending_auxiliary_keys=frozenset({"anything"})
        )


def test_untyped_auxiliary_routes_cannot_silently_use_independent_random_locations():
    source = NS(
        source=b"",
        template=NS(
            bindings=[
                NS(
                    group_kind="route",
                    value_kind="location",
                    logical_key="unowned-origin",
                    derivation=None,
                    target_paths=(),
                    dependency_paths=(),
                    dependency_bindings=(),
                    occurrences=[NS(source_text="Rotterdam")],
                    realization=NS(mode="generated_auxiliary"),
                )
            ]
        ),
        target={"documentPatch": {}},
    )
    with pytest.raises(ValueError, match="lack dependency contracts"):
        require_route_contract(source)


@pytest.mark.parametrize(
    "path",
    [
        "documentPatch.placeOfIssue.name",
        "documentPatch.placeOfIssue.country",
        "documentPatch.route.portOfLoading.name",
        "documentPatch.freight.paymentPlace.name",
    ],
)
def test_static_geography_is_reviewed_before_sampling_or_provider_calls(path):
    source = NS(
        source=b"",
        target={},
        template=NS(
            bindings=[
                NS(
                    logical_key="fixed-office",
                    derivation=None,
                    target_paths=(path,),
                    realization=NS(mode="static"),
                )
            ]
        ),
    )
    with pytest.raises(ValueError, match="constrained route contract"):
        require_route_contract(source)


@pytest.mark.parametrize(
    "role,mode,accepted",
    [
        ("carrier", "static", True),
        ("carrier", "single_surface", False),
        ("shipper", "static", False),
        ("shipper", "single_surface", False),
    ],
)
def test_city_country_equality_only_exempts_immutable_carrier_geography(role, mode, accepted):
    party = {"city": "Singapore", "country": "Singapore"}
    binding = NS(
        logical_key="shared-place",
        derivation=None,
        target_paths=tuple(f"documentPatch.parties.{role}.{key}" for key in party),
        dependency_paths=(),
        dependency_bindings=(),
        target_relationship="shared_value_equality",
        realization=NS(mode=mode),
        group_kind="carrier" if role == "carrier" else "party",
        value_kind="location",
        occurrences=(NS(source_text="Singapore"),),
    )
    source = NS(
        source=b"Singapore",
        target={"documentPatch": {"parties": {role: party}}},
        template=NS(bindings=(binding,), auxiliary_semantic_plan=NS(entities=())),
    )
    if accepted:
        require_route_contract(source)
        assert party == {"city": "Singapore", "country": "Singapore"}
    else:
        with pytest.raises(ValueError, match="constrained geography contract"):
            require_route_contract(source)


def test_independent_precarriage_cannot_be_reused_for_a_new_route():
    source = NS(
        source=b"",
        target={"documentPatch": {}},
        template=NS(
            bindings=[
                NS(
                    group_kind="transport",
                    value_kind="equipment",
                    logical_key="precarriage",
                    derivation=None,
                    target_paths=(),
                    dependency_paths=(),
                    dependency_bindings=(),
                    occurrences=[NS(source_text="FEEDER VESSEL")],
                    realization=NS(mode="generated_auxiliary"),
                )
            ]
        ),
    )
    with pytest.raises(ValueError, match="transport facts lack route dependency"):
        require_route_contract(source)


def test_city_country_equality_requires_a_joint_geography_contract_before_random_retries():
    source = NS(
        source=b"",
        target={},
        template=NS(
            bindings=[
                NS(
                    target_paths=(
                        "documentPatch.parties.shipper.city",
                        "documentPatch.parties.shipper.country",
                    ),
                    target_relationship="shared_value_equality",
                    derivation=None,
                    occurrences=(),
                    realization=NS(mode="single_surface"),
                )
            ]
        ),
    )
    with pytest.raises(ValueError, match="both city and country"):
        require_route_contract(source)


def test_company_name_token_cannot_also_be_a_sampled_country(monkeypatch):
    from document_ocr.synthesis.template_compiler import route_projection as module

    source = NS(
        source=b"",
        target={"documentPatch": {}},
        template=NS(
            bindings=[],
            auxiliary_semantic_plan=NS(
                entities=[
                    NS(
                        target_party_path="documentPatch.parties.consignee",
                        members=[NS(field="country", logical_key="name-fragment")],
                    )
                ]
            ),
        ),
    )
    monkeypatch.setattr(module, "projected_auxiliary_values", lambda *_: {"name-fragment": "EGYPT"})
    with pytest.raises(ValueError, match="also a lexical projection"):
        require_route_contract(source)


def test_unbound_fixed_width_phone_requires_review_before_route_sampling(monkeypatch):
    from document_ocr.synthesis.template_compiler import route_projection as module

    binding = NS(
        logical_key="contact",
        derivation=None,
        value_kind="identifier",
        group_kind="party",
        target_paths=(),
        dependency_paths=(),
        dependency_bindings=(),
        occurrences=(NS(source_text="TEL18915657802"),),
        realization=NS(mode="generated_auxiliary"),
    )
    source = NS(
        source=b"",
        target={
            "documentPatch": {
                "parties": {"shipper": {"contactDetails": {"phoneNumbers": ["18915657802"]}}}
            }
        },
        template=NS(bindings=(binding,), auxiliary_semantic_plan=NS(entities=())),
    )
    monkeypatch.setattr(module, "fixed_projection_ranges", lambda _: ())
    monkeypatch.setattr(module, "projected_auxiliary_values", lambda *_: {})
    with pytest.raises(ValueError, match="phone realization contract"):
        require_route_contract(source)
    binding.occurrences = (NS(source_text="REF18915657802"),)
    require_route_contract(source)
