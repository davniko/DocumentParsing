from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.descendant import (
    _composite_package_quantity_matches,
    _layout_like_source,
    _render_certified_date_surface,
    _render_target_binding,
    _string_semantics_match,
    _validate_description_volume_units,
)
from document_ocr.synthesis.template_compiler.generation_contract import (
    require_complete_variation,
    validate_compiled_party_contract,
    validate_rendered_party_boundaries,
    validate_repeated_agent_party_pages,
)
from document_ocr.synthesis.template_compiler.host import _exact_repeated_scalar_groups
from document_ocr.synthesis.template_compiler.realization_contract import complete_token_intervals


def test_cargo_volume_cannot_reuse_one_number_under_different_units():
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "FRESH AVOCADOS, VOLUME 43.470 CBM"}]
        }
    }
    with pytest.raises(ValueError, match="same cargo-volume number"):
        _validate_description_volume_units(target, "Gross Cargo Weight 23,677 kg. 43.470 cu. ft.")
    _validate_description_volume_units(target, "Gross Cargo Weight 23,677 kg. 1,535.018 cu. ft.")
    _validate_description_volume_units(target, "Gross Cargo Weight 23,677 kg. 43.470 CBM")


def test_repeated_scalar_groups_require_complete_ordered_source_copies():
    slots = tuple(NS(source_text=value) for value in ("ALPHA ", "STREET 1", "ALPHA ", "STREET 1"))
    assert _exact_repeated_scalar_groups(slots, "ALPHA STREET 1") == (0, 0, 1, 1)
    assert _exact_repeated_scalar_groups(slots[:3], "ALPHA STREET 1") is None
    assert _exact_repeated_scalar_groups(slots[1:], "ALPHA STREET 1") is None


def test_agent_repeated_party_page_must_retain_complete_generated_value():
    path = "documentPatch.parties.shipper.address"
    source = b"--- PAGE 1 ---\nOLD ROAD 1\n--- PAGE 2 ---\nOLD ROAD 1\n"
    binding = NS(
        realization=NS(mode="agent_required"),
        target_paths=(path,),
        occurrences=(NS(byte_start=15), NS(byte_start=51)),
    )
    source_target = {"documentPatch": {"parties": {"shipper": {"address": "OLD ROAD 1"}}}}
    target = {"documentPatch": {"parties": {"shipper": {"address": "NEW HARBOR ROAD"}}}}
    with pytest.raises(ValueError, match="lacks its generated target"):
        validate_repeated_agent_party_pages(
            source=source,
            rendered=b"--- PAGE 1 ---\nNEW ROAD\n--- PAGE 2 ---\nNEW HARBOR ROAD\n",
            source_target=source_target,
            target=target,
            bindings=(binding,),
        )
    validate_repeated_agent_party_pages(
        source=source,
        rendered=b"--- PAGE 1 ---\nNEW HARBOR ROAD\n--- PAGE 2 ---\nNEW HARBOR ROAD\n",
        source_target=source_target,
        target=target,
        bindings=(binding,),
    )


def test_compiled_party_contract_requires_contact_separator_and_name_affiliation_ownership():
    path = "documentPatch.parties.shipper.name"
    source = {"documentPatch": {"parties": {"shipper": {"name": "Alpha ON BEHALF OF Beta"}}}}

    def binding(key, kind, text, start, paths=(), mode="target_binding"):
        return NS(
            logical_key=key,
            group_key="party:shipper:0",
            group_kind="party",
            value_kind=kind,
            render_mode=mode,
            target_paths=paths,
            occurrences=(NS(byte_start=start, byte_end=start + len(text), source_text=text),),
        )

    name = binding("name", "organization", "Alpha", 0, (path,))
    affiliate = binding(
        "affiliate", "organization", "ON BEHALF OF Beta", 20, mode="deterministic_auxiliary"
    )
    phone = binding("phone", "phone", "034840282", 50)
    email = binding("email", "email", "name@example.com", 59)
    entity = NS(
        relationship="same_as_target_party",
        target_party_path="documentPatch.parties.shipper",
        members=(NS(logical_key="affiliate", field="other"),),
    )
    with pytest.raises(ValueError, match="phone/email"):
        validate_compiled_party_contract(
            raw=b" " * 200,
            source_target=source,
            bindings=(name, affiliate, phone, email),
            entities=(entity,),
        )
    email.occurrences = (NS(byte_start=60, byte_end=76, source_text="name@example.com"),)
    with pytest.raises(ValueError, match="affiliation"):
        validate_compiled_party_contract(
            raw=b" " * 200,
            source_target=source,
            bindings=(name, affiliate, phone, email),
            entities=(entity,),
        )
    affiliate.target_paths = (path,)
    validate_compiled_party_contract(
        raw=b" " * 200,
        source_target=source,
        bindings=(name, affiliate, phone, email),
        entities=(entity,),
    )


def test_compiled_party_contract_rejects_a_name_suffix_swallowed_by_the_address():
    party_path = "documentPatch.parties.consignee"
    name_path = party_path + ".name"
    address_path = party_path + ".address"
    source = {
        "documentPatch": {"parties": {"consignee": {"name": "Alpha S.A.E", "address": "Street 8"}}}
    }
    name = NS(
        logical_key="name",
        target_paths=(name_path,),
        group_key="party:consignee:0",
        group_kind="party",
        value_kind="organization",
        render_mode="target_binding",
        occurrences=(NS(byte_start=0, byte_end=5, source_text="Alpha"),),
    )
    address = NS(
        logical_key="address",
        target_paths=(address_path,),
        group_key="party:consignee:0",
        group_kind="party",
        value_kind="address",
        realization=NS(mode="single_surface"),
        render_mode="target_binding",
        occurrences=(NS(byte_start=6, byte_end=21, source_text="S.A.EStreet 8"),),
    )
    with pytest.raises(ValueError, match="suffix is swallowed"):
        validate_compiled_party_contract(
            raw=b" " * 200, source_target=source, bindings=(name, address), entities=()
        )
    name.occurrences = (NS(byte_start=0, byte_end=11, source_text="Alpha S.A.E"),)
    address.occurrences = (NS(byte_start=12, byte_end=20, source_text="Street 8"),)
    validate_compiled_party_contract(
        raw=b" " * 200, source_target=source, bindings=(name, address), entities=()
    )


def test_party_name_and_address_require_a_rendered_boundary():
    source = {
        "documentPatch": {
            "parties": {
                "forwardingAgent": {
                    "name": "Northline Customs Oy",
                    "address": "Kallionkatu 27",
                }
            }
        }
    }
    name = NS(
        logical_key="forwarder_name",
        target_paths=("documentPatch.parties.forwardingAgent.name",),
        group_key="party:forwardingAgent:0",
        group_kind="party",
        value_kind="organization",
        realization=NS(mode="single_surface"),
        occurrences=(NS(byte_start=0, byte_end=20, source_text="Northline Customs Oy"),),
    )
    address = NS(
        logical_key="forwarder_address",
        target_paths=("documentPatch.parties.forwardingAgent.address",),
        group_key="party:forwardingAgent:0",
        group_kind="party",
        value_kind="address",
        realization=NS(mode="single_surface"),
        occurrences=(NS(byte_start=20, byte_end=34, source_text="Kallionkatu 27"),),
    )
    with pytest.raises(ValueError, match="name/address slots lack a literal separator"):
        validate_compiled_party_contract(
            raw=b"Northline Customs OyKallionkatu 27",
            source_target=source,
            bindings=(name, address),
            entities=(),
        )
    address.occurrences = (NS(byte_start=21, byte_end=35, source_text="Kallionkatu 27"),)
    with pytest.raises(ValueError, match="name/address slots lack a literal separator"):
        validate_compiled_party_contract(
            raw=b"Northline Customs Oy.Kallionkatu 27",
            source_target=source,
            bindings=(name, address),
            entities=(),
        )
    validate_compiled_party_contract(
        raw=b"Northline Customs Oy\nKallionkatu 27",
        source_target=source,
        bindings=(name, address),
        entities=(),
    )

    with pytest.raises(ValueError, match="name and address lack a separator"):
        validate_rendered_party_boundaries(source, "Northline Customs OyKallionkatu 27")
    validate_rendered_party_boundaries(source, "Northline Customs Oy\nKallionkatu 27")
    source["documentPatch"]["parties"]["forwardingAgent"]["address"] = "FW> Kallionkatu 27"
    with pytest.raises(ValueError, match="continuation marker"):
        validate_compiled_party_contract(
            raw=b"Northline Customs Oy\nKallionkatu 27",
            source_target=source,
            bindings=(name, address),
            entities=(),
        )


def test_equal_party_address_projections_share_fixed_suffix_coordinates():
    original = "OLD ROAD STREET OFFICE 4 ESTATE NO35"
    words = original.lower().split()

    def projected(key, path, intervals):
        return NS(
            logical_key=key,
            target_paths=("documentPatch." + path,),
            value_kind="address",
            realization=NS(
                mode="token_projected_surface",
                adapter="natural_text",
                target_values=(NS(source_value=original),),
                slots=tuple(
                    NS(
                        required_target_prefix_tokens=tuple(words[:a]),
                        required_target_suffix_tokens=tuple(words[b:]),
                    )
                    for a, b in intervals
                ),
            ),
            occurrences=tuple(
                NS(slot_id=f"{key}{i}", source_text=" ".join(words[a:b]).upper())
                for i, (a, b) in enumerate(intervals)
            ),
        )

    notify = projected("notify", "notify", [(0, 7), (3, 7)])
    agent = projected("agent", "agent", [(0, 3)])
    template = NS(bindings=[notify, agent])
    value = "NEW MUCH LONGER INDUSTRIAL PARK STREET OFFICE 4 ESTATE NO35"
    target = dict(documentPatch=dict(notify=value, agent=value))
    result = _render_target_binding(notify, target, template=template)
    assert result.replacements == {"notify0": value, "notify1": "OFFICE 4 ESTATE NO35"}
    assert _render_target_binding(agent, target, template=template).replacements == {
        "agent0": "NEW MUCH LONGER INDUSTRIAL PARK STREET"
    }
    # Different descendant facts must not inherit the other's fixed anchor.
    target["documentPatch"]["notify"] = "DIFFERENT PLACE SUITE SEVEN"
    assert (
        "OFFICE 4"
        not in _render_target_binding(notify, target, template=template).replacements["notify1"]
    )


def _binding():
    return NS(
        target_paths=("documentPatch.address",),
        occurrences=(
            NS(slot_id="slot_0001", source_text="OLD STREET"),
            NS(slot_id="slot_0002", source_text="1000"),
        ),
        realization=NS(
            mode="token_projected_surface",
            adapter="natural_text",
            target_values=(NS(source_value="OLD STREET 1000"),),
            slots=(
                NS(required_target_prefix_tokens=(), required_target_suffix_tokens=("1000",)),
                NS(
                    required_target_prefix_tokens=("OLD", "STREET"),
                    required_target_suffix_tokens=(),
                ),
            ),
        ),
    )


def test_complete_projection_varies_all_parts_instead_of_pinning_original_words():
    binding = _binding()
    assert complete_token_intervals(binding) == ((0, 2), (2, 3))
    output = _render_target_binding(binding, {"documentPatch": {"address": "NEW HARBOR ROAD 2000"}})
    assert " ".join(output.replacements.values()) == "NEW HARBOR ROAD 2000"


def test_incomplete_projection_is_not_declared_fully_mutable():
    binding = _binding()
    binding.realization.slots = binding.realization.slots[:1]
    binding.occurrences = binding.occurrences[:1]
    assert complete_token_intervals(binding) is None
    with pytest.raises(ValueError, match="token-suffix"):
        _render_target_binding(binding, {"documentPatch": {"address": "NEW ROAD 2000"}})


def test_declared_address_facet_is_not_an_auxiliary_override(monkeypatch):
    from document_ocr.synthesis.template_compiler import descendant

    member = NS(logical_key="address_suffix", field="address")
    template = NS(
        bindings=(),
        auxiliary_semantic_plan=NS(
            entities=[NS(target_party_path="documentPatch.parties.shipper", members=[member])]
        ),
    )
    target = {"documentPatch": {"parties": {"shipper": {"address": "NEW STREET 1000"}}}}
    monkeypatch.setattr(
        descendant, "projected_auxiliary_values", lambda *args: {"address_suffix": "1000"}
    )
    descendant._validate_unrepresented_party_facets(
        source_target=target,
        target=target,
        template=template,
        auxiliary_values={"address_suffix": "1000"},
    )
    with pytest.raises(ValueError, match="cannot override"):
        descendant._validate_unrepresented_party_facets(
            source_target=target,
            target=target,
            template=template,
            auxiliary_values={"address_suffix": "9999"},
        )


def test_unicode_casefold_projection_uses_compiler_tokens_and_original_offsets():
    binding = _binding()
    binding.occurrences = (NS(slot_id="slot_0001", source_text="İç Kap\u0131 No:194"),)
    binding.realization.target_values = (NS(source_value="İç Kap\u0131 No:194 QBZ"),)
    binding.realization.slots = (
        NS(required_target_prefix_tokens=(), required_target_suffix_tokens=("qbz",)),
    )
    output = _render_target_binding(
        binding, {"documentPatch": {"address": "Yeni İç Kap\u0131 No:27 QBZ"}}
    )
    assert output.replacements["slot_0001"] == "Yeni İç Kap\u0131 No:27"


def test_projected_unowned_reference_must_survive_final_document_edits():
    from document_ocr.synthesis.template_compiler.descendant import _validate_projected_context

    binding = _binding()
    binding.realization.target_values = (NS(source_value="OLD STREET QBZ"),)
    binding.realization.slots = (
        NS(required_target_prefix_tokens=(), required_target_suffix_tokens=("qbz",)),
    )
    reference = NS(
        logical_key="reference",
        occurrences=(NS(source_text="QBZ", slot_id="ref"),),
        realization=NS(mode="single_surface"),
    )
    template = NS(bindings=(binding, reference))
    binding.logical_key = "address"
    binding.occurrences = (NS(source_text="OLD STREET", slot_id="address"),)
    _validate_projected_context(template, {"reference": NS(replacements={"ref": "QBZ"})})
    with pytest.raises(ValueError, match="context changed"):
        _validate_projected_context(template, {"reference": NS(replacements={"ref": "LWL"})})


@pytest.mark.parametrize(
    "owner_path",
    [
        "documentPatch.route.portOfLoading.name",
        "documentPatch.placeOfIssue.name",
    ],
)
def test_immutable_address_locality_requires_review_before_resampling_its_place(owner_path):
    from document_ocr.synthesis.template_compiler.descendant import _validate_projected_context
    from document_ocr.synthesis.template_compiler.route_projection import require_route_contract

    binding = _binding()
    binding.logical_key = "address"
    binding.target_paths = ("documentPatch.parties.shipper.address",)
    binding.dependency_bindings = ()
    binding.dependency_paths = ()
    binding.derivation = None
    binding.source_relationships = ()
    binding.realization.target_values = (NS(source_value="OLD STREET BARCELONA"),)
    binding.realization.slots = (
        NS(required_target_prefix_tokens=(), required_target_suffix_tokens=("barcelona",)),
    )
    port = NS(
        derivation=None,
        logical_key="loading_port",
        target_paths=(owner_path,),
        occurrences=(NS(source_text="BARCELONA", slot_id="port"),),
        realization=NS(mode="single_surface"),
    )
    template = NS(bindings=(binding, port))
    output = {"loading_port": NS(replacements={"port": "Valencia"})}
    with pytest.raises(ValueError, match="fixed lexical frame"):
        require_route_contract(NS(source=b"", target={}, template=template))
    with pytest.raises(ValueError, match="context changed"):
        _validate_projected_context(template, output)
    binding.dependency_paths = port.target_paths
    with pytest.raises(ValueError, match="context changed"):
        _validate_projected_context(template, output)


def test_separately_printed_geographic_context_is_derived_before_generation():
    from document_ocr.synthesis.template_compiler.realization_contract import (
        projected_auxiliary_values,
    )

    binding = _binding()
    binding.realization.slots = binding.realization.slots[:1]
    binding.occurrences = binding.occurrences[:1]
    owner = NS(
        logical_key="postcode",
        target_paths=(),
        derivation=None,
        occurrences=(NS(source_text="1000"),),
        realization=NS(mode="generated_auxiliary"),
    )
    binding.logical_key = "address"
    template = NS(bindings=(binding, owner), auxiliary_semantic_plan=NS(entities=()))
    target = {"documentPatch": {"address": "NEW AVENUE 1000"}}
    assert projected_auxiliary_values(template, target) == {"postcode": "1000"}
    assert target["documentPatch"]["address"] == "NEW AVENUE 1000"
    owner.target_paths = ("documentPatch.reference",)
    assert projected_auxiliary_values(template, target) == {}


def test_care_of_facet_uses_only_its_proven_fixed_subrange():
    from document_ocr.synthesis.template_compiler.realization_contract import (
        projected_auxiliary_values,
    )

    binding = _binding()
    binding.logical_key = "shipper_name"
    binding.target_paths = ("documentPatch.parties.shipper.name",)
    binding.occurrences = (NS(source_text="OLD EXPORT"),)
    binding.realization.target_values = (NS(source_value="OLD EXPORT C/O: FIXED PRINCIPAL LLC"),)
    binding.realization.slots = (
        NS(
            required_target_prefix_tokens=(),
            required_target_suffix_tokens=("c", "o", "fixed", "principal", "llc"),
        ),
    )
    owner = NS(
        logical_key="care_of",
        target_paths=(),
        derivation=None,
        occurrences=(NS(source_text="FIXED PRINCIPAL LLC"),),
        realization=NS(mode="generated_auxiliary"),
    )
    entity = NS(
        target_party_path="documentPatch.parties.shipper",
        members=(NS(logical_key="care_of", field="name"),),
    )
    template = NS(bindings=(binding, owner), auxiliary_semantic_plan=NS(entities=(entity,)))
    target = {
        "documentPatch": {"parties": {"shipper": {"name": "NEW EXPORT C/O: FIXED PRINCIPAL LLC"}}}
    }
    assert projected_auxiliary_values(template, target) == {"care_of": "FIXED PRINCIPAL LLC"}
    # An unrelated party or unproven subphrase must not be frozen by coincidence.
    entity.target_party_path = "documentPatch.parties.consignee"
    assert projected_auxiliary_values(template, target) == {}
    entity.target_party_path = "documentPatch.parties.shipper"
    owner.target_paths = ("documentPatch.parties.consignee.name",)
    assert projected_auxiliary_values(template, target) == {}


@pytest.mark.parametrize(
    "quantity,package,text",
    [
        (765, {"typeCategory": "PACKAGE_CARTON"}, "43PLTS=765CTNS=15300PCES"),
        (74, {"typeDescription": "PACKS"}, "147 ROLLS (74 PACKS)"),
    ],
)
def test_compact_and_free_text_package_counts(quantity, package, text):
    path = "documentPatch.cargoPackages[0].quantity"
    target = {"documentPatch": {"cargoPackages": [{"quantity": quantity, **package}]}}
    assert _composite_package_quantity_matches(None, target, path, text)
    target["documentPatch"]["cargoPackages"][0]["quantity"] += 1
    assert not _composite_package_quantity_matches(None, target, path, text)


@pytest.mark.parametrize(
    "surface", ["12 CRATES OF 12 PIECES (CRATES)", "12 Crate(s) of 12 PIECES(CRATES)"]
)
def test_parenthetical_package_description_keeps_quantity_ownership(surface):
    target = {
        "documentPatch": {"cargoPackages": [{"quantity": 12, "typeDescription": "PIECES (CRATES)"}]}
    }
    path = "documentPatch.cargoPackages[0].quantity"
    assert _composite_package_quantity_matches(None, target, path, surface)
    assert not _composite_package_quantity_matches(
        None, target, path, "12 CRATES OF 8 PIECES (CRATES)"
    )
    assert not _composite_package_quantity_matches(None, target, path, "12 PIECES (CARTONS)")


def test_single_allocation_quantity_uses_proven_package_identity_not_unrelated_number():
    path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    package_path = "documentPatch.cargoPackages[0].quantity"
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 12,
                    "typeCategory": "PACKAGE_CRATE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageIds": ["p1"],
                    "allocations": [{"containerNumber": "C1", "packageQuantity": 12}],
                }
            ],
        }
    }
    binding = NS(target_paths=(path, package_path))
    assert _composite_package_quantity_matches(
        binding, target, path, "12 CRATES OF 12 PIECES (CRATES)"
    )
    assert not _composite_package_quantity_matches(binding, target, path, "8 CRATES AND 12 CARTONS")
    target["documentPatch"]["cargoAllocationGroups"][0]["allocations"].append(
        {"containerNumber": "C2", "packageQuantity": 4}
    )
    assert not _composite_package_quantity_matches(binding, target, path, "12 CRATES")


def test_multiline_uppercase_applies_to_new_text_on_old_numeric_line():
    result = _layout_like_source("OLD STREET\n1000", "New Wharf Road Unit Seven")
    assert result == result.upper()


def test_projected_partition_rejects_insufficient_tokens():
    with pytest.raises(ValueError, match="fewer tokens"):
        _render_target_binding(_binding(), {"documentPatch": {"address": "SHORT"}})


def test_projected_mutable_parts_do_not_pin_each_other_across_literal_marker():
    b = _binding()
    b.realization.target_values = (NS(source_value="OLD DISPLAY P/N AB12-100"),)
    b.realization.slots = (
        NS(
            required_target_prefix_tokens=(),
            required_target_suffix_tokens=("p", "n", "ab12", "100"),
        ),
        NS(
            required_target_prefix_tokens=("old", "display", "p", "n"),
            required_target_suffix_tokens=(),
        ),
    )
    b.occurrences = (
        NS(slot_id="slot_0001", source_text="OLD DISPLAY"),
        NS(slot_id="slot_0002", source_text="AB12-100"),
    )
    output = _render_target_binding(
        b, {"documentPatch": {"address": "NEW BRIGHT SCREEN P/N CD34-200"}}
    )
    assert output.replacements == {"slot_0001": "NEW BRIGHT SCREEN", "slot_0002": "CD34-200"}
    with pytest.raises(ValueError, match="literal-gap"):
        _render_target_binding(
            b, {"documentPatch": {"address": "OLD DISPLAY P/N NEW PART P/N AB12-100"}}
        )


@pytest.mark.parametrize(
    "surface,old,new,expected",
    [
        ("29.07.24", "2024-07-29", "2020-07-29", "29.07.20"),
        ("04.04.22", "2022-04-04", "2018-04-04", "04.04.18"),
        ("FEB\n20\n2022", "2022-02-20", "2018-02-20", "FEB\n20\n2018"),
        ("MAY 07,2025", "2025-05-07", "2021-05-07", "MAY 07,2021"),
    ],
)
def test_certified_date_preserves_source_grammar(surface, old, new, expected):
    assert _render_certified_date_surface(surface, old, new) == expected


def test_composite_package_number_is_linked_to_its_category():
    path = "documentPatch.cargoPackages[0].quantity"
    binding = NS(target_paths=(path, path.replace("quantity", "typeCategory")))
    target = {
        "documentPatch": {
            "cargoPackages": [
                {"quantity": 12, "typeCategory": "PACKAGE_INTERMEDIATE_BULK_CONTAINER"}
            ]
        }
    }
    assert _composite_package_quantity_matches(
        binding, target, path, "20 PKGS (12 IBC TANKS + 8 PLTS)"
    )
    assert not _composite_package_quantity_matches(
        binding, target, path, "12 PKGS (4 IBC TANKS + 8 PLTS)"
    )
    assert _string_semantics_match("PACKAGE_INTERMEDIATE_BULK_CONTAINER", "12 IBC TANKS")


def test_residual_partial_value_is_not_complete_semantic_match():
    assert not _string_semantics_match("NEW LARGE TRADING COMPANY", "NEW LARGE")


def test_segmented_email_preserves_all_characters_and_continuation_markers():
    binding = NS(
        value_kind="email",
        target_paths=("documentPatch.email",),
        realization=NS(
            mode="segmented_surface",
            adapter="natural_text",
            target_values=(NS(source_value="HEE.AHMED.ABDELHAMID@HAIER.COM"),),
        ),
        occurrences=(
            NS(slot_id="s1", source_text="HEE.AHMED.**"),
            NS(slot_id="s2", source_text="**ABDELHAMID@HAIER.COM"),
        ),
    )
    value = "salma.nabil@eastgatehome.eg"
    output = _render_target_binding(binding, {"documentPatch": {"email": value}})
    assert output.replacements["s1"].endswith("**")
    assert output.replacements["s2"].startswith("**")
    reconstructed = output.replacements["s1"][:-2] + output.replacements["s2"][2:]
    assert reconstructed.casefold() == value
    binding.occurrences[1].source_text = "UNOWNED COMPANY TEXT"
    with pytest.raises(ValueError, match="source-character partition"):
        _render_target_binding(binding, {"documentPatch": {"email": value}})


def test_composite_package_never_uses_internal_enum_as_printed_text():
    from document_ocr.synthesis.template_compiler.descendant import _composite_target_surface

    with pytest.raises(ValueError, match="no textual surface"):
        _composite_target_surface(NS(value_kind="package"), (243, "PACKAGE_CARTON"))
    with pytest.raises(ValueError, match="human-readable"):
        _composite_target_surface(NS(value_kind="package"), (243, "PACKAGE_CARTON", "42 CARTONS"))
    assert (
        _composite_target_surface(NS(value_kind="package"), (243, "PACKAGE_CARTON", "243 CARTONS"))
        == "243 CARTONS"
    )


@pytest.mark.parametrize(
    "category,quantity,text",
    [
        ("PACKAGE_BOX_PLYWOOD", 1, "1 X PLYWOOD BOX (OVERPACK USED)"),
        ("PACKAGE_BOX_FIBREBOARD", 1, "1\nFIBREBOARD BOX CONTAINING: 2 METAL RECEPTACLES"),
        ("PACKAGE_RECEPTACLE_METAL", 2, "1 FIBREBOARD BOX CONTAINING: 2 METAL RECEPTACLES"),
    ],
)
def test_material_qualified_nested_package_counts(category, quantity, text):
    path = "documentPatch.cargoPackages[0].quantity"
    binding = NS(target_paths=(path, path.replace("quantity", "typeCategory")))
    target = {
        "documentPatch": {"cargoPackages": [{"quantity": quantity, "typeCategory": category}]}
    }
    assert _composite_package_quantity_matches(binding, target, path, text)
    assert _string_semantics_match(category, text)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = quantity + 100
    assert not _composite_package_quantity_matches(binding, target, path, text)


def test_bill_only_change_cannot_be_published_as_complete_synthesis():
    source = {
        "documentPatch": {
            "billOfLadingNumber": "OLD1",
            "parties": {
                "carrier": {"name": "FIXED"},
                "shipper": {"name": "OLD PARTY", "address": "OLD ADDRESS"},
            },
            "cargoGroups": [{"description": "OLD GOODS"}],
            "containers": [{"containerNumber": "OLD2"}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["billOfLadingNumber"] = "NEW1"
    with pytest.raises(ValueError, match="synthesis did not occur"):
        require_complete_variation(source, target)
    target["documentPatch"]["parties"]["shipper"].update(name="NEW PARTY", address="NEW ADDRESS")
    target["documentPatch"]["cargoGroups"][0]["description"] = "NEW GOODS"
    target["documentPatch"]["containers"][0]["containerNumber"] = "NEW2"
    assert len(require_complete_variation(source, target)) == 4


@pytest.mark.parametrize(
    "surface,old,new,expected",
    [
        ("78.280", "78.280", "48.768", "48.768"),
        ("24,510.000 Kgs", "24510", "19020", "19,020.000 Kgs"),
        ("18.35", "18.345", "10.125", "10.13"),
        ("ONE CONTAINER", "1", "2", "TWO CONTAINERS"),
    ],
)
def test_declared_arithmetic_disambiguates_numeric_grammar_and_rounding(
    surface, old, new, expected
):
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import (
        _render_proven_numeric_derivation,
    )

    binding = NS(logical_key="total", occurrences=(NS(slot_id="slot", source_text=surface),))
    output = _render_proven_numeric_derivation(
        binding, source_value=Decimal(old), target_value=Decimal(new)
    )
    assert output.replacements == {"slot": expected}


def test_derived_total_is_not_arbitrarily_rescaled_to_fit_wrong_source():
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import (
        _render_proven_numeric_derivation,
    )

    binding = NS(logical_key="total", occurrences=(NS(slot_id="slot", source_text="78.280"),))
    with pytest.raises(ValueError, match="does not prove"):
        _render_proven_numeric_derivation(
            binding, source_value=Decimal("79.280"), target_value=Decimal("40")
        )


def test_residual_agent_cannot_self_certify_an_incorrect_derived_total(monkeypatch):
    from document_ocr.synthesis.template_compiler import descendant as d

    binding = NS(
        logical_key="total",
        derivation="sum_volume",
        dependency_bindings=(),
        dependency_paths=("documentPatch.volume.value",),
        realization=NS(mode="deterministic_derivation"),
        occurrences=(NS(slot_id="slot", source_text="78.280", byte_end=6),),
    )
    case = NS(
        template=NS(bindings=(binding,)),
        numeric_auxiliary={},
        source=b"78.280",
        source_target={"documentPatch": {"volume": {"value": 78.280, "unit": "cubic_metre"}}},
        target={"documentPatch": {"volume": {"value": 48.768, "unit": "cubic_metre"}}},
    )
    case.template.byte_template = NS()
    outputs = {"total": d.BindingOutput(replacements={"slot": "999.000"}, canonical_value=999)}
    monkeypatch.setattr(d, "_validate_binding_format", lambda **kwargs: None)
    d._render_derivations(
        case=case, outputs=outputs, country_codes={}, residual_bindings=(binding,)
    )
    assert outputs["total"].replacements == {"slot": "48.768"}


def test_mixed_digit_and_number_word_derivation_uses_one_quantity():
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import (
        _render_proven_numeric_derivation,
    )

    binding = NS(
        logical_key="count",
        occurrences=(
            NS(slot_id="n", source_text="501"),
            NS(slot_id="w", source_text="FIVE HUNDRED ONE Package(s)"),
        ),
    )
    output = _render_proven_numeric_derivation(
        binding, source_value=Decimal(501), target_value=Decimal(302)
    )
    assert output.replacements == {"n": "302", "w": "THREE HUNDRED TWO Package(s)"}


def test_gross_weight_sum_does_not_include_net_weight_or_volume():
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import _measurement_dependency_values

    binding = NS(derivation="sum_gross_weight", dependency_paths=("documentPatch.cargoGroups",))
    target = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "grossWeight": {"value": 12, "unit": "kilogram"},
                    "netWeight": {"value": 10, "unit": "kilogram"},
                    "volume": {"value": 2, "unit": "cubic_metre"},
                },
                {"grossWeight": {"value": 8, "unit": "kilogram"}},
            ]
        }
    }
    assert sum(_measurement_dependency_values(binding, target).values()) == Decimal(20)


def test_explicit_mass_conversion_accounts_for_both_printed_rounding_intervals():
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import (
        _derivation_measurement_factor,
        _render_proven_numeric_derivation,
    )

    path = "documentPatch.cargoGroups[0].grossWeight.value"
    binding = NS(
        derivation="sum_gross_weight",
        logical_key="pounds",
        dependency_paths=(path,),
        occurrences=(NS(slot_id="lb", source_text="1800.000", byte_end=8),),
    )
    kg_binding = NS(target_paths=(path,), occurrences=(NS(source_text="816.466"),))
    source = {
        "documentPatch": {"cargoGroups": [{"grossWeight": {"value": 816.466, "unit": "kilogram"}}]}
    }
    target = {
        "documentPatch": {"cargoGroups": [{"grossWeight": {"value": 400.0, "unit": "kilogram"}}]}
    }
    case = NS(
        source_target=source,
        target=target,
        source=b"1800.000Lbs",
        template=NS(bindings=(kg_binding,)),
    )
    factor, uncertainty = _derivation_measurement_factor(binding, case)
    assert factor == Decimal(1) / Decimal("0.45359237") and uncertainty > 0
    output = _render_proven_numeric_derivation(
        binding,
        source_value=Decimal("816.466") * factor,
        target_value=Decimal(400) * factor,
        source_uncertainty=uncertainty,
    )
    assert output.replacements == {"lb": "881.849"}
    with pytest.raises(ValueError, match="does not prove"):
        _render_proven_numeric_derivation(
            binding,
            source_value=Decimal("800") * factor,
            target_value=Decimal(400) * factor,
            source_uncertainty=uncertainty,
        )


def test_same_as_identifier_can_project_one_complete_delimited_reference_component():
    from document_ocr.synthesis.template_compiler.descendant import (
        BindingOutput,
        _render_same_as_binding,
    )

    dependency = NS(occurrences=(NS(source_text="PHO24A1227/656110/R141"),))
    binding = NS(
        value_kind="identifier", occurrences=(NS(slot_id="ref", source_text="EGLHOPHO24A1227"),)
    )
    generated = BindingOutput(
        replacements={"dep": "ABC25B9381/193842/C203"}, canonical_value="ABC25B9381/193842/C203"
    )
    assert _render_same_as_binding(binding, dependency, generated).replacements == {
        "ref": "EGLHOABC25B9381"
    }
    binding.occurrences[0].source_text = "EGLHO99PHO24A1227"
    with pytest.raises(ValueError, match="not uniquely embedded"):
        _render_same_as_binding(binding, dependency, generated)


def test_numeric_grouping_cannot_appear_after_decimal_separator():
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import _numeric_interpretations

    assert _numeric_interpretations("23,952.50 KGS") == (Decimal("23952.50"),)
    assert _numeric_interpretations("23.952,50 KGS") == (Decimal("23952.50"),)


def test_number_word_phrase_includes_british_conjunction_and_preserves_frame():
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import (
        _render_proven_numeric_derivation,
    )

    binding = NS(
        logical_key="words",
        occurrences=(
            NS(slot_id="w", source_text="SAY ONE THOUSAND TWO HUNDRED AND THIRTY-ONE ONLY"),
        ),
    )
    result = _render_proven_numeric_derivation(
        binding, source_value=Decimal(1231), target_value=Decimal(87)
    )
    assert result.replacements == {"w": "SAY EIGHTY SEVEN ONLY"}


def test_numeric_sum_includes_source_only_auxiliary_with_direct_target_path():
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import (
        BindingOutput,
        _derivation_numeric_values,
    )

    binding = NS(
        logical_key="total",
        derivation="sum_decimal_values",
        dependency_paths=("documentPatch.weight",),
        dependency_bindings=("tare",),
    )
    auxiliary = {"tare": NS(contract=NS(source_value="2.000"), value="2.000")}
    assert _derivation_numeric_values(
        binding=binding,
        source_target={"documentPatch": {"weight": 22.080}},
        target={"documentPatch": {"weight": 30.0}},
        bindings={"tare": NS(target_paths=())},
        outputs={"tare": BindingOutput(replacements={}, canonical_value=2.0)},
        numeric_auxiliary=auxiliary,
    ) == (Decimal("24.080"), Decimal("32.000"))


@pytest.mark.parametrize("new_alias", [7, 8])
def test_collection_sum_counts_shared_quantity_alias_once_after_proving_equality(new_alias):
    from decimal import Decimal

    from document_ocr.synthesis.template_compiler.descendant import _derivation_numeric_values

    quantity = "documentPatch.cargoPackages[0].quantity"
    alias = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    binding = NS(
        logical_key="total",
        derivation="sum_package_quantity",
        dependency_paths=("documentPatch.cargoPackages",),
        dependency_bindings=("quantity",),
    )

    def document(value, other):
        return {
            "documentPatch": {
                "cargoPackages": [{"quantity": value}],
                "cargoAllocationGroups": [{"allocations": [{"packageQuantity": other}]}],
            }
        }

    kwargs = dict(
        binding=binding,
        source_target=document(1, 1),
        target=document(7, new_alias),
        bindings={"quantity": NS(target_paths=(quantity, alias))},
        outputs={},
    )
    if new_alias == 7:
        assert _derivation_numeric_values(**kwargs) == (Decimal(1), Decimal(7))
    else:
        with pytest.raises(ValueError, match="not covered"):
            _derivation_numeric_values(**kwargs)


def test_number_words_do_not_ignore_untyped_non_numeric_dependencies():
    from document_ocr.synthesis.template_compiler.descendant import _number_to_words_value

    binding = NS(
        logical_key="ambiguous",
        dependency_paths=("documentPatch.quantity", "documentPatch.unknown"),
        dependency_bindings=(),
    )
    with pytest.raises(ValueError, match="not uniquely numeric"):
        _number_to_words_value(
            binding, {}, {"documentPatch": {"quantity": 3, "unknown": "PACKAGE_PALLET"}}
        )


@pytest.mark.parametrize("category,noun", [("PACKAGE_CARTON", "CARTONS"), ("PACKAGE_BOX", "BOXES")])
@pytest.mark.parametrize("source_noun", ["DRUMS", "DRUM(S)"])
def test_number_word_package_frame_changes_quantity_and_category_together(
    category, noun, source_noun
):
    from document_ocr.synthesis.template_compiler.descendant import _render_one_derivation

    binding = NS(
        logical_key="package_words",
        derivation="number_to_words",
        dependency_bindings=(),
        dependency_paths=(
            "documentPatch.cargoPackages[0].quantity",
            "documentPatch.cargoPackages[0].typeCategory",
        ),
        occurrences=(NS(slot_id="words", source_text="SAY TWO HUNDRED FORTY " + source_noun),),
    )
    case = NS(
        source_target={
            "documentPatch": {"cargoPackages": [{"quantity": 240, "typeCategory": "PACKAGE_DRUM"}]}
        },
        target={"documentPatch": {"cargoPackages": [{"quantity": 137, "typeCategory": category}]}},
    )
    result = _render_one_derivation(
        binding=binding, case=case, outputs={}, bindings={}, country_codes={}
    )
    assert result.replacements["words"] == "SAY ONE HUNDRED THIRTY SEVEN " + noun


@pytest.mark.parametrize("source_noun", ["WOODEN CASE", "CASE WOODEN"])
def test_number_word_material_package_recognizes_printed_word_order(source_noun):
    from document_ocr.synthesis.template_compiler.descendant import _render_one_derivation

    binding = NS(
        logical_key="material_words",
        derivation="number_to_words",
        dependency_bindings=(),
        dependency_paths=(
            "documentPatch.cargoPackages[0].quantity",
            "documentPatch.cargoPackages[0].typeCategory",
        ),
        occurrences=(NS(slot_id="words", source_text="ONE (1) " + source_noun),),
    )
    case = NS(
        source_target={
            "documentPatch": {
                "cargoPackages": [{"quantity": 1, "typeCategory": "PACKAGE_CASE_WOODEN"}]
            }
        },
        target={
            "documentPatch": {"cargoPackages": [{"quantity": 2, "typeCategory": "PACKAGE_CARTON"}]}
        },
    )
    result = _render_one_derivation(
        binding=binding, case=case, outputs={}, bindings={}, country_codes={}
    )
    assert result.replacements["words"] == "TWO (2) CARTONS"
    binding.occurrences = (NS(slot_id="words", source_text="ONE (1) WOODEN CRATE"),)
    with pytest.raises(ValueError, match="noun has no unique source proof"):
        _render_one_derivation(
            binding=binding, case=case, outputs={}, bindings={}, country_codes={}
        )


def test_repeated_measurement_converts_every_occurrence_and_rejects_unconverted_alias():
    from document_ocr.synthesis.template_compiler.descendant import (
        BindingOutput,
        _render_target_measurement,
        _target_binding_semantics_valid,
    )

    path = "documentPatch.cargoGroups[0].grossWeight.value"
    binding = NS(
        logical_key="weight",
        target_paths=(path,),
        derivation=None,
        occurrences=(
            NS(slot_id="tonnes", source_text="22.080", byte_end=6),
            NS(slot_id="kg", source_text="22080.00", byte_end=18),
        ),
    )
    source = {
        "documentPatch": {
            "cargoGroups": [{"grossWeight": {"value": 22.08, "unit": "metric_tonne"}}]
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] = 17.0
    case = NS(
        source_target=source,
        target=target,
        source=b"22.080 MT\n22080.00 KGS",
        dangerous_goods_facts=(),
        template=NS(bindings=(binding,)),
    )
    expected = _render_target_measurement(binding, case)
    assert expected.replacements == {"tonnes": "17.000", "kg": "17000.00"}
    assert _target_binding_semantics_valid(case=case, outputs={"weight": expected}) == (True, ())
    invalid = BindingOutput(replacements={"tonnes": "17.000", "kg": "17.00"}, canonical_value=17.0)
    assert _target_binding_semantics_valid(case=case, outputs={"weight": invalid}) == (
        False,
        ("weight:measurement-occurrence-differs",),
    )


def test_identifier_capacity_fails_before_retrying_an_exhausted_single_digit_domain():
    from document_ocr.synthesis.run_safety import IdentifierSpaceError
    from document_ocr.synthesis.structured_semantics import _validate_generic_identifier_capacity

    inventory = NS(
        generic_requests=tuple(NS(request_key=str(i)) for i in range(9)),
        generic_source_by_request={str(i): "9" for i in range(9)},
    )
    with pytest.raises(
        IdentifierSpaceError, match="capacity insufficient: requested 9, available at most 8"
    ):
        _validate_generic_identifier_capacity(inventory, ["8", "9"])
    _validate_generic_identifier_capacity(inventory, ["9"])
