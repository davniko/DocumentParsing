from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import route_derivations as route


def binding(derivation="sampled_route_country", owner="portOfDischarge"):
    return NS(
        derivation=derivation,
        target_paths=(),
        dependency_bindings=(),
        dependency_paths=("documentPatch.route." + owner,),
        group_kind="route",
        value_kind="location",
        render_mode="deterministic_derived",
        logical_key="country",
    )


@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("name", "Rotterdam"),
        ("country", "Netherlands"),
        ("country_code", "NL"),
        ("location", "Rotterdam, Netherlands"),
        ("locode", "NLRTM"),
    ],
)
def test_auxiliary_route_uses_pinned_endpoint_without_adding_labels(suffix, expected):
    scenario = {
        "dischargePort": dict(
            name="Rotterdam", countryName="Netherlands", countryCode="NL", locode="NLRTM"
        )
    }
    assert route.values((binding("sampled_route_" + suffix),), scenario) == {"country": expected}
    assert set(scenario) == {"dischargePort"}


@pytest.mark.parametrize("owner", ["vessel", "parties.consignee", "missing"])
def test_unsupported_owner_fails_explicitly(owner):
    with pytest.raises(ValueError, match="physical endpoint"):
        route.endpoint(binding(owner=owner))


def test_private_transshipment_surface_uses_the_same_sampled_chain():
    item = binding("sampled_route_name", "transshipmentPort")
    item.occurrences = (NS(source_text="Mersin"),)
    source = {"documentPatch": {"route": {"portOfLoading": {"name": "Norfolk"}}}}
    private = route.physical_source(source, (item,))
    assert private["documentPatch"]["route"]["transshipmentPort"] == {"name": "Mersin"}
    assert "transshipmentPort" not in source["documentPatch"]["route"]
    scenario = {"routeLocations": {"transshipmentPort": {"name": "Aliaga"}}}
    assert route.values((item,), scenario) == {"country": "Aliaga"}


def test_missing_latent_fact_and_unowned_endpoint_are_not_source_copied():
    with pytest.raises(KeyError):
        route.values((binding(),), {"dischargePort": {"name": "Rotterdam"}})
    item = binding()
    item.target_paths = ("documentPatch.route.portOfDischarge.country",)
    with pytest.raises(ValueError, match="one route owner"):
        route.endpoint(item)


def test_unlabelled_receipt_stays_private_and_requires_its_own_name_evidence():
    source = {"documentPatch": {"route": {"portOfLoading": {"name": "Shanghai"}}}}
    item = binding("sampled_route_name", "placeOfReceipt")
    item.occurrences = (NS(source_text="Ningbo"),)
    private = route.physical_source(source, (item,))
    assert private["documentPatch"]["route"]["placeOfReceipt"] == {"name": "Ningbo"}
    assert "placeOfReceipt" not in source["documentPatch"]["route"]
    country = binding("sampled_route_country", "placeOfReceipt")
    country.occurrences = (NS(source_text="China"),)
    with pytest.raises(ValueError, match="lacks a printed name"):
        route.physical_source(source, (country,))
    country.occurrences = (NS(source_text="China"), NS(source_text="Japan"))
    with pytest.raises(ValueError, match="non-equivalent"):
        route.physical_source(source, (item, country))


def test_state_abbreviation_cannot_pass_as_combined_port_country_even_after_critic_pass():
    target = {
        "documentPatch": {"route": {"portOfLoading": {"name": "Paranagua", "country": "Brazil"}}}
    }
    item = binding("sampled_route_location", "portOfLoading")
    item.occurrences = (NS(source_text="PR"),)
    with pytest.raises(ValueError, match=r"does not prove.*location"):
        route.validate_source(target, (item,))
    item.occurrences = (NS(source_text="PARANAGUA, BRAZIL"),)
    route.validate_source(target, (item,))
    item.derivation = "sampled_route_name"
    with pytest.raises(ValueError, match=r"does not prove.*name"):
        route.validate_source(target, (item,))


def test_source_only_country_can_complete_a_private_endpoint_without_changing_labels():
    source = {"documentPatch": {"route": {"portOfLoading": {"name": "Ningbo"}}}}
    item = binding("sampled_route_country", "portOfLoading")
    item.occurrences = (NS(source_text="China"),)
    physical = route.physical_source(source, (item,))
    assert physical["documentPatch"]["route"]["portOfLoading"]["country"] == "China"
    assert "country" not in source["documentPatch"]["route"]["portOfLoading"]
    route.validate_source(source, (item,))


@pytest.mark.parametrize(
    "text,valid",
    [
        ("BUSAN SEAPORT IN SOUTH KOREA", True),
        ("BUSAN IN SOUTH KOREA", False),
        ("BUSAN SEAPORT IN NORTH KOREA", False),
        ("BUSAN SEAPORT VIA SHANGHAI IN SOUTH KOREA", False),
    ],
)
def test_location_connector_preserves_both_complete_source_facts(text, valid):
    source = {
        "documentPatch": {
            "route": {
                "portOfLoading": {
                    "name": "BUSAN SEAPORT",
                    "country": "SOUTH KOREA",
                }
            }
        }
    }
    item = binding("sampled_route_location", "portOfLoading")
    item.occurrences = (NS(source_text=text),)
    if valid:
        route.validate_source(source, (item,))
    else:
        with pytest.raises(ValueError, match="does not prove"):
            route.validate_source(source, (item,))


@pytest.mark.parametrize(
    "surface,expected",
    [
        ("MAHER TERMINAL", "Rotterdam Terminal"),
        ("LONG BEACH CONTAINER TERM", "Rotterdam Container Term"),
        ("SSA TERMINAL 18", "Rotterdam Terminal 18"),
    ],
)
def test_synthetic_terminal_stays_owned_by_sampled_port(surface, expected):
    item = binding("sampled_route_terminal")
    item.occurrences = (NS(source_text=surface),)
    source = {"documentPatch": {"route": {"portOfDischarge": {"name": "New York"}}}}
    route.validate_source(source, (item,))
    assert route.values((item,), {"dischargePort": {"name": "Rotterdam"}}) == {"country": expected}
    assert route.physical_source(source, (item,)) == source


@pytest.mark.parametrize("surface", ["CHARLOTTE", "ORIGINALS TO BE RELEASED AT", "TX", "TERMINAL"])
def test_terminal_derivation_cannot_hide_other_route_facts(surface):
    item = binding("sampled_route_terminal")
    item.occurrences = (NS(source_text=surface),)
    source = {"documentPatch": {"route": {"portOfDischarge": {"name": "New York"}}}}
    with pytest.raises(ValueError, match="typed terminal"):
        route.validate_source(source, (item,))
    item = binding("sampled_route_terminal", "placeOfReceipt")
    with pytest.raises(ValueError, match="port owner"):
        route.endpoint(item)


@pytest.mark.parametrize("role", ["portOfLoading", "portOfDischarge"])
def test_explicit_printed_port_can_be_private_without_adding_training_fields(role):
    source = {"documentPatch": {}}
    item = binding("sampled_route_name", role)
    item.occurrences = (NS(source_text="Shanghai"),)
    physical = route.physical_source(source, (item,))
    assert physical["documentPatch"]["route"][role] == {"name": "Shanghai"}
    assert source == {"documentPatch": {}}
    route.validate_source(source, (item,))
    item.derivation = "sampled_route_country"
    item.occurrences = (NS(source_text="China"),)
    with pytest.raises(ValueError, match="lacks a printed name"):
        route.physical_source(source, (item,))


def test_subdivision_is_private_pinned_port_context_and_missing_codes_fail_closed():
    item = binding("sampled_route_subdivision_code", "portOfLoading")
    item.occurrences = (NS(source_text="PR"),)
    target = {"documentPatch": {"route": {"portOfLoading": {"name": "Paranagua", "country": "BR"}}}}
    support = NS(
        maritime_by_country={"BR": (NS(locode="BRPNG", name="Paranagua", subdivision_code="PR"),)}
    )
    countries = NS(resolve=lambda value: value if value == "BR" else None)
    route.validate_source(target, (item,))
    route.validate_subdivision_evidence(target, (item,), support=support, countries=countries)
    assert route.physical_source(target, (item,)) == target
    assert route.values((item,), {"loadingPort": {"subdivisionCode": "TX"}}) == {"country": "TX"}
    with pytest.raises(ValueError, match="required auxiliary fact"):
        route.values((item,), {"loadingPort": {"subdivisionCode": None}})
    item.occurrences = (NS(source_text="SP"),)
    with pytest.raises(ValueError, match="unique pinned registry identity"):
        route.validate_subdivision_evidence(target, (item,), support=support, countries=countries)
    item.occurrences = (NS(source_text="Parana"),)
    with pytest.raises(ValueError, match="does not prove"):
        route.validate_source(target, (item,))


def test_subdivision_evidence_rejects_ambiguous_ports_and_inland_role():
    item = binding("sampled_route_subdivision_code", "portOfLoading")
    item.occurrences = (NS(source_text="PR"),)
    target = {"documentPatch": {"route": {"portOfLoading": {"name": "Paranagua PR"}}}}
    ports = tuple(
        NS(locode=code, name="Paranagua", subdivision_code="PR") for code in ("BRPNG", "BRPRG")
    )
    with pytest.raises(ValueError, match="unique pinned registry identity"):
        route.validate_subdivision_evidence(
            target, (item,), support=NS(maritime_by_country={"BR": ports}), countries=NS()
        )
    item.dependency_paths = ("documentPatch.route.placeOfReceipt",)
    with pytest.raises(ValueError, match="port owner"):
        route.endpoint(item)
