import json
import re
from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import Occurrence, SourceContract
from document_ocr.synthesis.curated_ownership import (
    _declared_date_surface,
    apply_dependent_text,
    build_owned_blueprint,
    load_owned_blueprint,
    ownership_surfaces,
)
from document_ocr.synthesis.curated_templates import (
    LexicalOwnership,
    _aligned_sample_leaves,
    compile_sampling_blueprint,
    current_path,
    render_sampling_blueprint,
    substitute_expression_literals,
    wrap_owned_text,
)


def source_fixture():
    raw = (
        "ÉDITÉ\nSHIPPER\nACME LTD\n12 ROAD\nMUMBAI, INDIA\n"
        "TAX: 12345\nPORT: MUMBAI\nCOUNTRY: INDIA\n"
        "GOODS: STEEL BOLTS\n24 BOXES\nHS: 731815\nISSUED: 2024.10.17\n"
    )
    row = {
        "documentId": "sample",
        "joinedRawText": raw,
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": "non_negotiable",
                "issueDate": "2024-10-17",
                "route": {"portOfLoading": {"name": "MUMBAI", "country": "INDIA"}},
                "parties": {
                    "shipper": {
                        "name": "ACME LTD",
                        "addressLine": "12 ROAD MUMBAI, INDIA",
                        "country": "INDIA",
                    }
                },
                "goodsItemDetails": [
                    {
                        "description": "STEEL BOLTS",
                        "hsCodes": ["731815"],
                        "numberAndTypeOfPackages": [
                            {"packageQuantity": 24, "typeCategory": "PACKAGE_BOX"}
                        ],
                    }
                ],
            },
        },
    }
    contract = SourceContract(
        variables=[
            dict(
                key="name",
                kind="name",
                value="ACME LTD",
                meaning="shipper",
                required_literals=[],
                occurrences=[dict(text="ACME LTD", occurrence=1, presentation="text")],
            ),
            dict(
                key="postal",
                kind="postal",
                value="12 ROAD MUMBAI, INDIA",
                meaning="postal",
                required_literals=["INDIA"],
                occurrences=[
                    dict(text="12 ROAD\nMUMBAI, INDIA", occurrence=1, presentation="text")
                ],
            ),
            dict(
                key="product",
                kind="product",
                value="STEEL BOLTS",
                meaning="goods",
                required_literals=["STEEL"],
                occurrences=[dict(text="STEEL BOLTS", occurrence=1, presentation="text")],
            ),
            dict(
                key="quantity",
                kind="count",
                value="24",
                meaning="packages",
                required_literals=[],
                occurrences=[dict(text="24", occurrence=1, presentation="number")],
            ),
        ],
        targets=[
            dict(path="documentPatch.parties.shipper.name", expression="{name}"),
            dict(path="documentPatch.parties.shipper.addressLine", expression="{postal}"),
            dict(path="documentPatch.goodsItemDetails[0].description", expression="{product}"),
            dict(
                path="documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity",
                expression="{quantity}",
            ),
        ],
        fixed_context="original lexical pilot constraints superseded by sampled scenario",
    )
    bindings = []

    def binding(key, path, text, kind, occurrence=1):
        offset = -1
        for _ in range(occurrence):
            offset = raw.index(text, offset + 1)
        start = len(raw[:offset].encode())
        bindings.append(
            dict(
                logical_key=key,
                target_paths=[path],
                value_kind=kind,
                occurrences=[
                    dict(byte_start=start, byte_end=start + len(text.encode()), source_text=text)
                ],
            )
        )

    binding("shipper_country", "documentPatch.parties.shipper.country", "INDIA", "location")
    binding("loading", "documentPatch.route.portOfLoading.name", "MUMBAI", "location", 2)
    binding("loading_country", "documentPatch.route.portOfLoading.country", "INDIA", "location", 2)
    binding("package", "documentPatch.cargoPackages[0].typeCategory", "BOXES", "package")
    binding("hs", "documentPatch.cargoGroups[0].hsCodes[0]", "731815", "identifier")
    binding("date", "documentPatch.issueDate", "2024.10.17", "date")
    return row, contract, {"document_id": "sample", "bindings": bindings}


def test_rebase_identity_and_full_scalar_sampling_preserve_unrelated_bytes():
    row, contract, old = source_fixture()
    blueprint = compile_sampling_blueprint(row, old, contract)
    inventory = blueprint.inventory()
    assert inventory["nestedBindings"]
    assert json.loads(json.dumps(inventory)) == inventory
    raw, target, proof = render_sampling_blueprint(blueprint, row["target"], {})
    assert raw == row["joinedRawText"] and target == row["target"] and proof["edits"] == []
    new = deepcopy(row["target"])
    patch = new["documentPatch"]
    patch["issueDate"] = "2025-09-12"
    patch["route"]["portOfLoading"] = dict(name="ROTTERDAM", country="NETHERLANDS")
    patch["parties"]["shipper"].update(
        name="NORTH BV", addressLine="8 CANAL ROAD ROTTERDAM, NETHERLANDS", country="NETHERLANDS"
    )
    patch["goodsItemDetails"][0].update(description="CERAMIC TILES", hsCodes=["690721"])
    patch["goodsItemDetails"][0]["numberAndTypeOfPackages"][0].update(
        packageQuantity=36, typeCategory="PACKAGE_CARTON"
    )
    raw, target, proof = render_sampling_blueprint(blueprint, new, {})
    assert target == new
    assert "ÉDITÉ" in raw and "TAX: 12345" in raw
    assert "36 CARTONS" in raw and "2025.09.12" in raw and "690721" in raw
    assert "INDIA" not in raw and "STEEL" not in raw
    assert proof["changedTargetPaths"] == proof["coveredTargetPaths"]
    assert row["target"] != new


@pytest.mark.parametrize("named_consignee", [True, False])
def test_order_instructions_remain_fixed_while_named_party_regions_change(named_consignee):
    from document_ocr.synthesis.curated_sampling import validate_instruction_inheritance

    role = "consignee" if named_consignee else "notifyParties[0]"
    name_path = f"documentPatch.parties.{role}.name"
    parties = (
        {"consignee": {"name": "OLD BANK"}, "notifyParties": [{"sameAs": "consignee"}]}
        if named_consignee
        else {"notifyParties": [{"name": "OLD BANK", "sameAs": None}]}
    )
    raw = (
        "CONSIGNEE\nTO THE ORDER OF OLD BANK\nNOTIFY\nSAME AS CONSIGNEE\n"
        if named_consignee
        else "CONSIGNEE\nTO ORDER\nNOTIFY\nOLD BANK\n"
    )
    row = {
        "documentId": "order",
        "joinedRawText": raw,
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": "negotiable",
                "parties": parties,
            },
        },
    }
    contract = SourceContract(
        variables=[
            dict(
                key="name",
                kind="name",
                value="OLD BANK",
                meaning="printed party name",
                required_literals=[],
                occurrences=[dict(text="OLD BANK", occurrence=1, presentation="text")],
            )
        ],
        targets=[dict(path=name_path, expression="{name}")],
        fixed_context="Order and notify instructions.",
    )
    blueprint = compile_sampling_blueprint(row, {"document_id": "order", "bindings": []}, contract)
    target = deepcopy(row["target"])
    owner = target["documentPatch"]["parties"]
    (owner["consignee"] if named_consignee else owner["notifyParties"][0])["name"] = "NEW COMPANY"
    text, actual, proof = render_sampling_blueprint(blueprint, target, {})
    assert text == raw.replace("OLD BANK", "NEW COMPANY") and actual == target
    assert proof["changedTargetPaths"] == [name_path]
    validate_instruction_inheritance(row["target"], actual)
    if not named_consignee:
        assert "consignee" not in actual["documentPatch"]["parties"]


@pytest.mark.parametrize("style", ["preserve", "uppercase", "title"])
def test_casing_is_owned_presentation_not_a_label_or_technical_text_edit(style):
    from document_ocr.synthesis.curated_casing import CasingPolicy, case_owned_text

    row, contract, old = source_fixture()
    blueprint = compile_sampling_blueprint(row, old, contract)
    text, target, proof = render_sampling_blueprint(
        blueprint, row["target"], {}, render_casing=style
    )
    assert target == row["target"]
    assert "GOODS: STEEL BOLTS\n24 BOXES\nHS: 731815" in text
    assert "ÉDITÉ\nSHIPPER\n" in text and "TAX: 12345" in text
    if style == "title":
        assert "Acme Ltd\n12 Road\nMumbai, India" in text
        assert "PORT: Mumbai\nCOUNTRY: India" in text
    raw = row["joinedRawText"].encode()
    for edit in reversed(proof["edits"]):
        assert edit["after"].upper() == edit["before"].upper()
        raw = raw[: edit["byteStart"]] + edit["after"].encode() + raw[edit["byteEnd"] :]
    assert raw.decode() == text
    protected = "AB12Cd µA 5mL Contact@Example.com https://Example.com/Path"
    for path in (
        "documentPatch.goodsItemDetails[0].description",
        "documentPatch.parties.shipper.contactDetails.emailAddresses[0]",
        "documentPatch.containerInformation[0].equipmentIdentifier",
    ):
        assert case_owned_text(protected, (path,), style) == protected
    postal = "ROAD AB12Cd Contact@Example.com https://Example.com/Path"
    assert case_owned_text(postal, ("documentPatch.parties.shipper.addressLine",), "title") == (
        "Road AB12Cd Contact@Example.com https://Example.com/Path"
    )
    policy = CasingPolicy(render_styles=("uppercase", "title"))
    first = {str(i): policy.select(7, str(i)) for i in range(64)}
    assert first == {str(i): policy.select(7, str(i)) for i in reversed(range(64))}
    assert set(first.values()) == {"uppercase", "title"}


@pytest.mark.parametrize(
    "source", ["MAKİNA TİCARET LTD. ŞTİ.", "ÉTÉ A\u0300 PARIS", "Straße \u0131STANBUL"]
)
@pytest.mark.parametrize("style", ["uppercase", "title"])
def test_owned_casing_preserves_canonically_equivalent_unicode_letters(source, style):
    import unicodedata

    from document_ocr.synthesis.curated_casing import case_owned_text

    result = case_owned_text(source, ("documentPatch.parties.shipper.name",), style)
    expected = source.upper() if style == "uppercase" else source.title()
    assert result == expected
    assert unicodedata.normalize("NFC", result.upper()) == unicodedata.normalize(
        "NFC", source.upper()
    )


def test_changed_package_requires_printed_type_not_declared_or_product_overlap_only():
    row, contract, old = source_fixture()
    target = deepcopy(row["target"])
    goods = target["documentPatch"]["goodsItemDetails"][0]
    goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_PALLET"
    blueprint = compile_sampling_blueprint(row, old, contract)
    with pytest.raises(ValueError, match="lacks an owned printed noun"):
        render_sampling_blueprint(blueprint, target, {}, surface_values={"package": "BOXES"})
    package = next(b for b in old["bindings"] if b["logical_key"] == "package")
    start = len(row["joinedRawText"].split("STEEL BOLTS")[0].encode())
    package["occurrences"] = [
        dict(byte_start=start, byte_end=start + len("STEEL BOLTS"), source_text="STEEL BOLTS")
    ]
    blueprint = compile_sampling_blueprint(row, old, contract)
    goods["description"] = "PALLET STACKERS"
    with pytest.raises(ValueError, match="lacks an owned printed noun"):
        render_sampling_blueprint(blueprint, target, {})


@pytest.mark.parametrize("quantity,expected", [(1, "PACKAGE"), (5440, "PACKAGES")])
def test_reviewed_package_recipe_reconciles_source_noun_with_unchanged_current_category(
    quantity, expected
):
    row, contract, old = source_fixture()
    path = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].typeCategory"
    # A rebased current target can disagree with the historical noun even when
    # synthesis does not change its category; the explicit recipe owns this repair.
    row["target"]["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "typeCategory"
    ] = "PACKAGE_PACKAGE"
    blueprint = build_owned_blueprint(
        row, old, contract, {"surfaces": {"package": {"package_path": path}}}
    )
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "packageQuantity"
    ] = quantity
    surfaces = ownership_surfaces(
        blueprint,
        target,
        SimpleNamespace(
            party_localities={"documentPatch.parties.shipper": SimpleNamespace(country_code="IN")}
        ),
    )
    assert surfaces["package"] == [expected]
    text, rendered, proof = render_sampling_blueprint(
        blueprint, target, {}, surface_values=surfaces
    )
    assert expected in text and "BOXES" not in text
    assert rendered == target and path not in proof["changedTargetPaths"]
    assert any(edit["key"] == "package" and edit["after"] == expected for edit in proof["edits"])


def test_country_cannot_be_claimed_by_ownership_if_not_generated():
    row, contract, old = source_fixture()
    blueprint = compile_sampling_blueprint(row, old, contract)
    target = deepcopy(row["target"])
    target["documentPatch"]["parties"]["shipper"]["country"] = "SPAIN"
    with pytest.raises(ValueError, match="absent from owned postal"):
        render_sampling_blueprint(blueprint, target, {})
    target["documentPatch"]["parties"]["shipper"].update(
        country="INDIA", addressLine="55 NEW ROAD MUMBAI"
    )
    with pytest.raises(ValueError, match="absent from owned postal"):
        render_sampling_blueprint(blueprint, target, {})


def test_composite_country_replacement_is_atomic_and_keeps_variable_names():
    assert (
        substitute_expression_literals(
            "{shipper_postal} CHINA(CN)",
            {"CHINA(CN)": "GUINEA-BISSAU", "CHINA": "GUINEA-BISSAU", "CN": "GW"},
        )
        == "{shipper_postal} GUINEA-BISSAU"
    )
    assert (
        substitute_expression_literals("{country_ir} CHINA IR", {"CHINA": "IRAN", "IR": "FI"})
        == "{country_ir} IRAN FI"
    )


def test_country_codes_require_explicit_authority_and_terminal_postal_position():
    row, contract, old = source_fixture()
    blueprint = compile_sampling_blueprint(row, old, contract)
    target = deepcopy(row["target"])
    path = "documentPatch.parties.shipper.country"
    party = target["documentPatch"]["parties"]["shipper"]
    party.update(country="SPAIN", addressLine="12 CALLE MAYOR MADRID ES")
    with pytest.raises(ValueError, match="absent from owned postal"):
        render_sampling_blueprint(blueprint, target, {})
    text, _, proof = render_sampling_blueprint(
        blueprint, target, {}, certified_country_codes={path: "ES"}
    )
    assert "MADRID ES" in " ".join(text.split())
    assert proof["certifiedCountryCodes"] == {path: "ES"}
    party["addressLine"] = "12 ES STREET MADRID"
    with pytest.raises(ValueError, match="absent from owned postal"):
        render_sampling_blueprint(blueprint, target, {}, certified_country_codes={path: "ES"})
    with pytest.raises(ValueError, match="invalid certified"):
        render_sampling_blueprint(blueprint, target, {}, certified_country_codes={path: "ESP"})


def test_multiword_country_remains_grounded_across_physical_line_wrap():
    row, contract, old = source_fixture()
    blueprint = compile_sampling_blueprint(row, old, contract)
    target = deepcopy(row["target"])
    target["documentPatch"]["parties"]["shipper"].update(
        country="TURKS AND CAICOS ISLANDS",
        addressLine="1 ROAD TURKS AND CAICOS ISLANDS",
    )
    text, _, _ = render_sampling_blueprint(
        blueprint, target, {"postal": "1 ROAD TURKS AND\nCAICOS ISLANDS"}
    )
    assert "1 ROAD TURKS AND\nCAICOS ISLANDS" in text


def test_reviewed_date_format_must_match_source_date():
    row, contract, old = source_fixture()
    blueprint = build_owned_blueprint(
        row,
        old,
        contract,
        {"surfaces": {"date": {"date_path": "documentPatch.issueDate", "format": "%Y.%m.%d"}}},
    )
    target = deepcopy(row["target"])
    target["documentPatch"]["issueDate"] = "2025-09-01"
    scenario = SimpleNamespace(
        party_localities={"documentPatch.parties.shipper": SimpleNamespace(country_code="IN")}
    )
    surfaces = ownership_surfaces(blueprint, target, scenario)
    assert surfaces["date"] == ["2025.09.01"]
    blueprint.ownership_data["surfaces"]["date"]["format"] = "%Y.%d.%m"
    with pytest.raises(ValueError):
        ownership_surfaces(blueprint, target, scenario)


def test_worded_date_profiles_cover_every_calendar_day():
    rendered = [
        _declared_date_surface(date(2026, 1, n), "%B {day_ordinal}, %Y") for n in range(1, 32)
    ]
    assert len(set(rendered)) == 31
    assert rendered[0] == "January FIRST, 2026"
    assert rendered[19:22] == [
        "January TWENTIETH, 2026",
        "January TWENTY-FIRST, 2026",
        "January TWENTY-SECOND, 2026",
    ]
    assert rendered[-2:] == ["January THIRTIETH, 2026", "January THIRTY-FIRST, 2026"]


def test_postal_order_override_cannot_introduce_or_drop_facts():
    row, contract, old = source_fixture()
    with pytest.raises(ValueError, match="changes facts rather than order"):
        build_owned_blueprint(
            row,
            old,
            contract,
            {"render_expressions": {"documentPatch.parties.shipper.addressLine": "{postal} INDIA"}},
        )


def test_ownership_cache_is_content_bound_and_does_not_share_mutable_declarations(tmp_path):
    row, contract, old = source_fixture()
    path = tmp_path / "ownership.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {"sample": {"delete": {"date": "Reviewed deletion"}}},
            }
        )
    )
    first = load_owned_blueprint(row, old, contract, path)
    assert next(r for r in first.regions if r.key == "date").allow_empty
    first.ownership_data["delete"].clear()
    second = load_owned_blueprint(row, old, contract, path)
    assert second.ownership_data["delete"] == {"date": "Reviewed deletion"}
    path.write_text(json.dumps({"version": 1, "sources": {"sample": {}}}))
    third = load_owned_blueprint(row, old, contract, path)
    assert not next(r for r in third.regions if r.key == "date").allow_empty
    path.write_text(json.dumps({"version": 1, "sources": {}}))
    with pytest.raises(ValueError, match="does not cover"):
        load_owned_blueprint(row, old, contract, path)
    path.write_text(json.dumps({"version": 2, "sources": {"sample": {}}}))
    with pytest.raises(ValueError, match="unsupported"):
        load_owned_blueprint(row, old, contract, path)


def test_split_postal_country_ownership_distinguishes_named_site_from_country_component():
    role = "documentPatch.parties.shipper"
    row = {
        "documentId": "split-postal",
        "joinedRawText": "SHIPPER\nACME\nINDIA INDUSTRIAL PARK\nMUMBAI INDIA\n",
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": "non_negotiable",
                "parties": {
                    "shipper": {
                        "name": "ACME",
                        "country": "INDIA",
                        "addressLine": "INDIA INDUSTRIAL PARK MUMBAI INDIA",
                    }
                },
            },
        },
    }
    contract = SourceContract(
        variables=[
            dict(
                key=key,
                kind=kind,
                value=value,
                meaning=key,
                required_literals=[],
                occurrences=[dict(text=value, occurrence=1, presentation="text")],
            )
            for key, kind, value in [
                ("name", "name", "ACME"),
                ("site", "postal", "INDIA INDUSTRIAL PARK"),
                ("locality", "postal", "MUMBAI INDIA"),
            ]
        ],
        targets=[
            dict(path=role + ".name", expression="{name}"),
            dict(path=role + ".addressLine", expression="{site} {locality}"),
        ],
        fixed_context="Split address with a country word inside a named site.",
    )
    old = {"document_id": row["documentId"], "bindings": []}
    country_path = role + ".country"
    declaration = {"postal_country_owners": {country_path: ["locality"]}}
    blueprint = build_owned_blueprint(row, old, contract, declaration)
    assert country_path not in next(r for r in blueprint.regions if r.key == "site").target_paths
    assert country_path in next(r for r in blueprint.regions if r.key == "locality").target_paths
    target = deepcopy(row["target"])
    target["documentPatch"]["parties"]["shipper"].update(
        addressLine="8 INDUSTRIAL ESTATE MADRID SPAIN", country="SPAIN"
    )
    text, _, _ = render_sampling_blueprint(
        blueprint, target, {"site": "8 INDUSTRIAL ESTATE", "locality": "MADRID SPAIN"}
    )
    assert "8 INDUSTRIAL ESTATE\nMADRID SPAIN" in text
    with pytest.raises(ValueError, match="absent from owned postal"):
        render_sampling_blueprint(
            blueprint, target, {"site": "8 INDUSTRIAL ESTATE", "locality": "MADRID"}
        )
    for owners in [[], ["name"], ["missing"], ["locality", "locality"]]:
        with pytest.raises(ValueError, match="postal country"):
            build_owned_blueprint(
                row, old, contract, {"postal_country_owners": {country_path: owners}}
            )
    with pytest.raises(ValueError, match="invalid postal country"):
        build_owned_blueprint(
            row, old, contract, {"postal_country_owners": {role + ".name": ["name"]}}
        )
    with pytest.raises(ValueError, match="lacks same-party source evidence"):
        build_owned_blueprint(
            row,
            old,
            contract,
            {
                **declaration,
                "delete": {"locality": "Cannot delete the country owner"},
            },
        )


def test_declared_contact_expansion_prints_owned_contact_in_party_and_syncs_repeat():
    raw = "SHIPPER\nACME\nTEL: 5551234\nGOODS\nCONTACT ALICE\n"
    role = "documentPatch.parties.shipper"
    phone_path, name_path = (
        role + ".contactDetails.phoneNumbers[0]",
        role + ".contactDetails.contactName",
    )
    row = {
        "documentId": "contact-example",
        "joinedRawText": raw,
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": "non_negotiable",
                "parties": {
                    "shipper": {
                        "name": "ACME",
                        "contactDetails": {"phoneNumbers": ["5551234"], "contactName": "ALICE"},
                    }
                },
            },
        },
    }
    contract = SourceContract.model_validate(
        {
            "variables": [
                {
                    "key": "name",
                    "kind": "name",
                    "value": "ACME",
                    "meaning": "shipper",
                    "required_literals": [],
                    "occurrences": [{"text": "ACME", "occurrence": 1, "presentation": "text"}],
                }
            ],
            "targets": [{"path": role + ".name", "expression": "{name}"}],
            "fixed_context": "Contact source topology test.",
        }
    )
    historical = {"document_id": row["documentId"], "bindings": []}
    for key, path, value in [("phone", phone_path, "5551234"), ("contact", name_path, "ALICE")]:
        start = raw.index(value)
        historical["bindings"].append(
            {
                "logical_key": key,
                "target_paths": [path],
                "value_kind": "other_text",
                "occurrences": [
                    {"byte_start": start, "byte_end": start + len(value), "source_text": value}
                ],
            }
        )
    declaration = {
        "bindings": {
            "phone": {"paths": [phone_path, name_path], "occurrences": [{"text": "TEL: 5551234"}]}
        },
        "surfaces": {
            "phone": {"expression": "CONTACT: {" + name_path + "}\nTEL: {" + phone_path + "}"}
        },
    }
    blueprint = build_owned_blueprint(row, historical, contract, declaration)
    target = deepcopy(row["target"])
    target["documentPatch"]["parties"]["shipper"]["contactDetails"] = {
        "phoneNumbers": ["7777777"],
        "contactName": "BOB",
    }
    surfaces = ownership_surfaces(blueprint, target, SimpleNamespace(party_localities={}))
    text, _, proof = render_sampling_blueprint(blueprint, target, {}, surface_values=surfaces)
    assert text == "SHIPPER\nACME\nCONTACT: BOB\nTEL: 7777777\nGOODS\nCONTACT BOB\n"
    assert len(proof["edits"]) == 2


def test_reject_stale_offsets_unknown_values_missing_owners_and_topology_changes():
    row, contract, old = source_fixture()
    broken = deepcopy(old)
    broken["bindings"][0]["occurrences"][0]["byte_start"] += 1
    with pytest.raises(ValueError, match="differs from current OCR"):
        compile_sampling_blueprint(row, broken, contract)
    blueprint = compile_sampling_blueprint(row, old, contract)
    with pytest.raises(ValueError, match="unknown render keys"):
        render_sampling_blueprint(blueprint, row["target"], {"imagined": "X"})
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["hsCodes"].append("690721")
    with pytest.raises(ValueError, match="duplicate-HS6 collapse"):
        render_sampling_blueprint(blueprint, target, {})
    target = deepcopy(row["target"])
    target["schemaVersion"] = "999"
    with pytest.raises(ValueError, match="no rendered owner"):
        render_sampling_blueprint(blueprint, target, {})


def test_explicit_composite_surface_required_and_receipted():
    row, contract, old = source_fixture()
    old["bindings"][-1]["value_kind"] = "identifier"
    blueprint = compile_sampling_blueprint(row, old, contract)
    target = deepcopy(row["target"])
    target["documentPatch"]["issueDate"] = "2025-09-12"
    with pytest.raises(ValueError, match="explicit composite"):
        render_sampling_blueprint(blueprint, target, {})
    raw, _, proof = render_sampling_blueprint(
        blueprint, target, {}, surface_values={"date": "12/SEP/2025"}
    )
    assert "12/SEP/2025" in raw
    assert proof["edits"][-1]["after"] == "12/SEP/2025"


def test_whole_product_override_replaces_old_identity_not_just_one_fragment():
    row, contract, old = source_fixture()
    override = LexicalOwnership(
        key="whole_goods",
        target_paths=("documentPatch.goodsItemDetails[0].description",),
        occurrences=(Occurrence(text="STEEL BOLTS", occurrence=1, presentation="text"),),
    )
    blueprint = compile_sampling_blueprint(row, old, contract, ownership_overrides=(override,))
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["description"] = "POLYESTER FABRIC"
    raw, _, _ = render_sampling_blueprint(blueprint, target, {})
    assert "POLYESTER FABRIC" in raw and "STEEL BOLTS" not in raw
    incomplete = LexicalOwnership(
        key="partial",
        target_paths=override.target_paths,
        occurrences=(Occurrence(text="STEEL", occurrence=1, presentation="text"),),
    )
    with pytest.raises(ValueError, match="partially intersects"):
        compile_sampling_blueprint(row, old, contract, ownership_overrides=(incomplete,))


def test_hs_collapse_retains_all_printed_surfaces_but_not_duplicate_targets():
    before = {
        "g.hsCodes[0]": "020622",
        "g.hsCodes[1]": "02062200",
        "g.hsCodes[2]": "0206220000",
        "other": 1,
    }
    after = {"g.hsCodes[0]": "020714", "other": 1}
    aligned, aliases = _aligned_sample_leaves(before, after)
    assert [aligned[f"g.hsCodes[{i}]"] for i in range(3)] == ["020714"] * 3
    assert set(aliases.values()) == {"g.hsCodes[0]"}
    with pytest.raises(ValueError, match="not a duplicate-HS6 collapse"):
        _aligned_sample_leaves({**before, "g.hsCodes[2]": "123456"}, after)


@pytest.mark.parametrize(
    "value",
    [
        "22 SULTAN HUSSEIN ST., ANTOUKHY , ALEXANDRIA , EGYPT",
        "EGYPT, SUEZ , DREAM MALL , SECOND FLOOR , NO.14",
    ],
)
def test_wrapping_keeps_punctuation_with_text_without_dropping_tokens(value):
    text = wrap_owned_text("39 Banni El-Abbas, Bab Sharqi ,\nAlexandria , Egypt", value)
    assert " ".join(text.split()) == value
    assert all(not line.startswith((",", ";", ":")) for line in text.splitlines())
    with pytest.raises(ValueError, match="detached delimiter"):
        wrap_owned_text("x\ny", "CITY\n, COUNTRY")
    # Goods have no inferred character quota; short source fragments are not
    # evidence of narrow columns. Explicit product-list boundaries also survive.
    assert wrap_owned_text("PARTS", value, product=True) == value
    listed = "ALLOY STEEL PIPE\nHIGH PRESSURE VALVE ASSEMBLY"
    assert wrap_owned_text("METAL PARTS", listed, product=True) == listed


def test_current_paths_translate_only_structural_fields():
    assert (
        current_path("documentPatch.containers[0].containerNumber")
        == "documentPatch.containerInformation[0].equipmentIdentifier"
    )
    assert (
        current_path("documentPatch.cargoAllocationGroups[0].allocations[1].containerNumber")
        == "documentPatch.goodsItemDetails[0].splitGoodsPlacement[1].equipmentIdentifier"
    )
    assert (
        current_path("documentPatch.parties.shipper.city") == "documentPatch.parties.shipper.city"
    )


def test_lexical_ownership_changes_the_wording_contract_and_checks_repeat_coverage():
    row, contract, old = source_fixture()
    declaration = {
        "lexical": [
            {
                "key": "whole_goods",
                "kind": "product",
                "paths": ["documentPatch.goodsItemDetails[0].description"],
                "occurrences": [{"text": "STEEL BOLTS"}],
            }
        ],
    }
    blueprint = build_owned_blueprint(row, old, contract, declaration)
    assert "product" not in {v.key for v in blueprint.contract.variables}
    assert "whole_goods" in {v.key for v in blueprint.contract.variables}
    assert blueprint.contract.targets[-1].expression == "{whole_goods}"
    assert render_sampling_blueprint(blueprint, row["target"], {})[0] == row["joinedRawText"]
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["description"] = "CERAMIC TILES"
    assert "CERAMIC TILES" in " ".join(
        render_sampling_blueprint(blueprint, target, {"whole_goods": "CERAMIC TILES"})[0].split()
    )


def test_reviewed_postal_owner_consumes_repeated_suffix_and_phone_prefix_without_relabeling():
    address = "12 ROAD\nCAIRO\n12573"
    complete_address = address + " / 12573\nEgypt"
    source_phone = "+2\n02 2269 3193"
    raw = ("CONSIGNEE\n" + complete_address + "\nTEL : " + source_phone + "\n") * 2
    path = "documentPatch.parties.consignee"
    row = {
        "documentId": "postal-boundary",
        "joinedRawText": raw,
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": "non_negotiable",
                "parties": {
                    "consignee": {
                        "addressLine": "12 ROAD CAIRO 12573",
                        "country": "EGYPT",
                        "contactDetails": {"phoneNumbers": ["+2 02 2269 3193"]},
                    }
                },
            },
        },
    }
    contract = SourceContract(
        variables=[
            dict(
                key="postal",
                kind="postal",
                value="12 ROAD CAIRO 12573",
                meaning="consignee postal address",
                required_literals=[],
                occurrences=[dict(text=address, occurrence=i, presentation="text") for i in (1, 2)],
            )
        ],
        targets=[dict(path=path + ".addressLine", expression="{postal}")],
        fixed_context="Reviewed real label excludes duplicated printed postal suffix.",
    )
    historical = {"document_id": row["documentId"], "bindings": []}
    for key, field, text, kind in (
        ("country", ".country", "Egypt", "location"),
        ("phone", ".contactDetails.phoneNumbers[0]", source_phone[1:], "phone"),
    ):
        starts = [match.start() for match in re.finditer(re.escape(text), raw)]
        historical["bindings"].append(
            dict(
                logical_key=key,
                target_paths=[path + field],
                value_kind=kind,
                occurrences=[
                    dict(byte_start=p, byte_end=p + len(text), source_text=text) for p in starts
                ],
            )
        )
    declaration = {
        "bindings": {"phone": {"occurrences": [{"text": source_phone, "all": True}]}},
        "lexical": [
            {
                "key": "postal",
                "kind": "postal",
                "paths": [path + ".addressLine"],
                "occurrences": [{"text": complete_address, "all": True}],
            }
        ],
    }
    blueprint = build_owned_blueprint(row, historical, contract, declaration)
    assert blueprint.target == row["target"]
    assert blueprint.contract.variables[0].value == "12 ROAD CAIRO 12573"
    assert render_sampling_blueprint(blueprint, row["target"], {})[0] == raw
    target = deepcopy(row["target"])
    target["documentPatch"]["parties"]["consignee"].update(
        addressLine="47 NEW ROAD LAMPANG 52000 THAILAND",
        country="THAILAND",
        contactDetails={"phoneNumbers": ["+66 123 4567"]},
    )
    text, actual, proof = render_sampling_blueprint(blueprint, target, {})
    assert actual == target
    assert "12573" not in text and "Egypt" not in text and "++" not in text
    assert text.count("THAILAND") == " ".join(text.split()).count("TEL : +66 123 4567") == 2
    assert proof["changedTargetPaths"] == proof["coveredTargetPaths"]


def test_reviewed_literal_deletion_is_explicit_and_other_blank_surfaces_fail():
    row, contract, old = source_fixture()
    declaration = {
        "add": [{"key": "old_tax_caption", "occurrences": [{"text": "TAX:"}]}],
        "delete": {"old_tax_caption": "Test reviewed removal of an obsolete caption."},
    }
    blueprint = build_owned_blueprint(row, old, contract, declaration)
    text, _, proof = render_sampling_blueprint(
        blueprint, row["target"], {}, surface_values={"old_tax_caption": ""}
    )
    assert "12345" in text and "TAX:" not in text
    assert any(edit["before"] == "TAX:" and edit["after"] == "" for edit in proof["edits"])
    with pytest.raises(ValueError, match="cannot be blank"):
        render_sampling_blueprint(blueprint, row["target"], {}, surface_values={"date": ""})


def test_numeric_constant_is_unfrozen_only_after_exact_historical_ownership():
    row, contract, old = source_fixture()
    path = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"
    contract = contract.model_copy(
        update={
            "variables": [v for v in contract.variables if v.key != "quantity"],
            "targets": [
                t.model_copy(update={"expression": "24"}) if t.path == path else t
                for t in contract.targets
            ],
        }
    )
    unowned = build_owned_blueprint(row, old, contract, {})
    assert any(t.path == path and t.expression == "24" for t in unowned.contract.targets)
    start = len(row["joinedRawText"].split("24 BOXES")[0].encode())
    old["bindings"].append(
        dict(
            logical_key="printed_count",
            target_paths=[path],
            value_kind="number",
            occurrences=[dict(byte_start=start, byte_end=start + 2, source_text="24")],
        )
    )
    owned = build_owned_blueprint(row, old, contract, {})
    assert not any(t.path == path for t in owned.contract.targets)
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "packageQuantity"
    ] = 36
    assert "36 BOXES" in render_sampling_blueprint(owned, target, {})[0]
    with pytest.raises(ValueError, match="no rendered owner"):
        render_sampling_blueprint(unowned, target, {})


def handling_dependency_fixture():
    row, contract, old = source_fixture()
    paths = [f"documentPatch.goodsItemDetails[0].handlingInstructions[{i}]" for i in range(2)]
    clauses = ["21 DAYS DEMURRAGE FREETIME IN BOMBAY", "CARGO IN TRANSIT TO MUMBAI WAREHOUSE"]
    row["target"]["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = clauses
    text = "\n".join(clauses)
    row["joinedRawText"] += text + "\n"
    declaration = {
        "dependent_text": [
            {
                "path": paths[0],
                "source": clauses[0],
                "expression": (
                    "21 DAYS DEMURRAGE FREETIME IN {documentPatch.route.portOfLoading.name}"
                ),
            },
            {
                "path": paths[1],
                "source": clauses[1],
                "expression": (
                    "CARGO IN TRANSIT TO {documentPatch.route.portOfLoading.name} WAREHOUSE"
                ),
            },
        ],
        "add": [
            {
                "key": "handling",
                "paths": paths,
                "occurrences": [{"text": text}],
            }
        ],
        "surfaces": {"handling": {"expression": "{" + paths[0] + "}\n{" + paths[1] + "}"}},
    }
    return row, build_owned_blueprint(row, old, contract, declaration)


def test_handling_dependencies_rebind_reviewed_route_aliases_and_render_all_clauses():
    row, blueprint = handling_dependency_fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["route"]["portOfLoading"]["name"] = "ROTTERDAM"
    before = deepcopy(target)
    updated = apply_dependent_text(blueprint, target)
    clauses = updated["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"]
    assert clauses == [
        "21 DAYS DEMURRAGE FREETIME IN ROTTERDAM",
        "CARGO IN TRANSIT TO ROTTERDAM WAREHOUSE",
    ]
    assert target == before and blueprint.target == row["target"]
    assert apply_dependent_text(blueprint, updated) == updated
    text, rendered, proof = render_sampling_blueprint(
        blueprint,
        updated,
        {},
        surface_values=ownership_surfaces(
            blueprint,
            updated,
            SimpleNamespace(
                party_localities={
                    "documentPatch.parties.shipper": SimpleNamespace(country_code="IN")
                }
            ),
        ),
    )
    assert all(clause in " ".join(text.split()) for clause in clauses)
    assert rendered == updated
    assert set(proof["changedTargetPaths"]) <= set(proof["coveredTargetPaths"])
    blueprint.ownership_data.pop("dependent_text")
    assert apply_dependent_text(blueprint, target) is target


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("not_list", "must be a list"),
        ("extra_key", "exactly path, source and expression"),
        ("blank", "nonempty strings"),
        ("non_handling_path", "owned handling instruction"),
        ("unowned_path", "owned handling instruction"),
        ("duplicate", "duplicate dependent text path"),
        ("stale_source", "differs from pinned target"),
        ("literal_only", "invalid placeholders"),
        ("malformed_expression", "invalid placeholders"),
        ("non_route_name", "only accepts route name"),
        ("missing_source_route", "route name is absent or invalid"),
        ("missing_sampled_route", "route name is absent or invalid"),
        ("invalid_sampled_route", "route name is absent or invalid"),
        ("missing_handling", "absent or conflicts"),
        ("conflicting_handling", "absent or conflicts"),
    ],
)
def test_handling_dependencies_fail_closed_without_partial_target_changes(change, error):
    row, blueprint = handling_dependency_fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["route"]["portOfLoading"]["name"] = "ROTTERDAM"
    declarations = blueprint.ownership_data["dependent_text"]
    item = declarations[1]
    if change == "not_list":
        blueprint.ownership_data["dependent_text"] = None
    elif change == "extra_key":
        item["unexpected"] = True
    elif change == "blank":
        item["expression"] = " "
    elif change == "non_handling_path":
        item["path"] = "documentPatch.parties.shipper.name"
    elif change == "unowned_path":
        item["path"] = "documentPatch.goodsItemDetails[0].handlingInstructions[2]"
    elif change == "duplicate":
        declarations.append(deepcopy(declarations[0]))
    elif change == "stale_source":
        item["source"] += " STALE"
    elif change == "literal_only":
        item["expression"] = "CARGO IN TRANSIT"
    elif change == "malformed_expression":
        item["expression"] = "CARGO IN TRANSIT TO {{documentPatch.route.portOfLoading.name}"
    elif change == "non_route_name":
        item["expression"] = "CARGO IN TRANSIT TO {documentPatch.route.portOfLoading.country}"
    elif change == "missing_source_route":
        del blueprint.target["documentPatch"]["route"]["portOfLoading"]["name"]
    elif change == "missing_sampled_route":
        del target["documentPatch"]["route"]["portOfLoading"]["name"]
    elif change == "invalid_sampled_route":
        target["documentPatch"]["route"]["portOfLoading"]["name"] = 42
    elif change == "missing_handling":
        target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"].pop()
    elif change == "conflicting_handling":
        target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"][1] = "OTHER CLAUSE"
    before = deepcopy(target)
    with pytest.raises(ValueError, match=error):
        apply_dependent_text(blueprint, target)
    assert target == before
