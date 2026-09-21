from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.fixed_vocabulary import fixed_vocabulary


def test_source_only_route_is_fixed_only_for_the_complete_unchanged_scenario():
    from document_ocr.synthesis.template_compiler.fixed_vocabulary import fixed_context

    b = binding(
        "LUDHIANA - CHAWAPAIL",
        group_kind="route",
        value_kind="location",
        logical_key="pre_carriage",
    )
    source = {"documentPatch": {"route": {"portOfLoading": {"country": "INDIA", "name": "MUNDRA"}}}}
    assert fixed_context(b, source, source)
    assert not fixed_context(
        b, source, {"documentPatch": {"route": {"portOfLoading": {"name": "SAVANNAH"}}}}
    )
    assert not fixed_context(b, {"documentPatch": {}}, {"documentPatch": {}})
    b.dependency_paths = ("documentPatch.route.portOfLoading",)
    assert not fixed_context(b, source, source)


def test_unlabeled_equipment_receipt_is_fixed_scenario_context_not_an_identifier():
    from document_ocr.synthesis.template_compiler.fixed_vocabulary import fixed_context

    b = binding(
        "5X40'HC", group_kind="equipment", value_kind="equipment", logical_key="equipment:receipt"
    )
    blank = {"documentPatch": {}}
    assert fixed_context(b, blank, blank)
    assert not fixed_context(
        b, blank, {"documentPatch": {"containers": [{"containerNumber": "NEW"}]}}
    )
    b.occurrences = (NS(source_text="5O50'GS"),)
    assert not fixed_context(b, blank, blank)


@pytest.mark.parametrize(
    "text",
    [
        "COC",
        "SEAL",
        "40' HC",
        "02 X 40`HC FCL CONTAINERS",
        "1 x 20 FT ISO TANK CONTAINER(S)",
        "2 CONTAINER (2x40')",
        "40HQ*4",
        "HC45",
        "45G1",
        "5X40' CNTR(S)",
    ],
)
def test_closed_equipment_context_requires_unchanged_physical_equipment(text):
    from document_ocr.synthesis.template_compiler.fixed_vocabulary import fixed_context

    b = binding(
        text, group_kind="equipment", value_kind="equipment", logical_key="equipment:receipt"
    )
    source = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "OLD",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "NEW",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }
    assert fixed_context(b, source, target)
    target["documentPatch"]["containers"][0]["sizeCategory"] = "TWENTY_FOOT"
    assert not fixed_context(b, source, target)


def test_independent_feeder_is_fixed_but_main_vessel_alias_is_not():
    from document_ocr.synthesis.template_compiler.fixed_vocabulary import fixed_context

    b = binding(
        "HMM RAON",
        group_kind="transport",
        value_kind="equipment",
        logical_key="pre_carriage_vessel",
    )
    source = {
        "documentPatch": {
            "route": {"portOfLoading": {"name": "BUSAN"}},
            "transport": {"vesselName": "HMM GARAM"},
        }
    }
    target = {
        "documentPatch": {
            "route": {"portOfLoading": {"name": "BUSAN"}},
            "transport": {"vesselName": "NEW MAIN VESSEL"},
        }
    }
    assert fixed_context(b, source, target)
    source["documentPatch"]["transport"]["vesselName"] = "HMM RAON BUSAN"
    assert not fixed_context(b, source, target)


@pytest.mark.parametrize("kind", ["equipment", "dangerous_goods", "other_text", "integer"])
def test_format_policy_never_grants_permission_to_randomize_non_identifiers(kind):
    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.template_compiler.descendant import _render_direct_auxiliary

    b = binding("5X40'HC", group_kind="equipment", value_kind=kind, logical_key="equipment:receipt")
    b.occurrences = (NS(source_text="5X40'HC", render_policy="opaque_identifier"),)
    with pytest.raises(ValueError, match="no deterministic auxiliary renderer"):
        _render_direct_auxiliary(b, DeterministicStream(42, "test", "equipment"))


def binding(text, **changes):
    return NS(
        **{
            "target_paths": (),
            "dependency_paths": (),
            "dependency_bindings": (),
            "source_relationships": (),
            "derivation": None,
            "value_kind": "operational_text",
            "group_kind": "transport",
            "occurrences": (NS(source_text=text),),
            **changes,
        }
    )


@pytest.mark.parametrize(
    "text",
    [
        "CY / CY",
        "FCL/FCL",
        "Shipped on\nBoard",
        "Prepaid",
        "Basic Ocean Freight",
        "INCOTERMS 2020",
        "FOB",
        "21 days detention free at destination",
    ],
)
def test_complete_fixed_context_is_recognized(text):
    assert fixed_vocabulary(binding(text))


@pytest.mark.parametrize(
    "text",
    [
        "CY/CY 12 CONTAINERS",
        "SHIPPED ON BOARD 2026-01-02",
        "Mercantile Shipping (Jeddah)",
        "20 days",
        "PG III",
        "PREPAID BY ACME",
        "",
        "FCL UN 2903",
        "21 days detention free at destination FOR ACME",
    ],
)
def test_unknown_or_shipment_dependent_text_is_never_copied(text):
    assert not fixed_vocabulary(binding(text))


@pytest.mark.parametrize(
    "changes",
    [
        {"target_paths": ("documentPatch.freightPayment",)},
        {"dependency_paths": ("documentPatch.parties.shipper.name",)},
        {"dependency_bindings": ("party",)},
        {"source_relationships": ("relation",)},
        {"group_kind": "party"},
        {"group_kind": "cargo"},
        {"value_kind": "organization"},
        {"derivation": "same_as"},
    ],
)
def test_identity_and_dependency_contracts_take_precedence(changes):
    assert not fixed_vocabulary(binding("PREPAID", **changes))


def test_every_occurrence_must_qualify():
    assert not fixed_vocabulary(
        binding("CY/CY", occurrences=(NS(source_text="CY/CY"), NS(source_text="CY/CY ACME")))
    )


def test_cargo_qualifiers_do_not_require_rewriting_their_meaning():
    assert fixed_vocabulary(binding("SLAC*", group_kind="cargo"))
    assert not fixed_vocabulary(binding("SLAC* 30 BOXES", group_kind="cargo"))
