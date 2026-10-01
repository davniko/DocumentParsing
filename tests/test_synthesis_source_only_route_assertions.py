"""Source-owned route claims cannot silently disagree with sampled endpoints."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.source_only_route_assertions import validate

SOURCE = {
    "documentPatch": {
        "route": {"portOfDischarge": {"name": "ALEXANDRIA - OLD PORT", "country": "EGYPT"}},
        "parties": {"consignee": {"country": "EGYPT"}},
    }
}
TARGET = {
    "documentPatch": {
        "route": {"portOfDischarge": {"name": "Tacoma", "country": "United States"}},
        "parties": {"consignee": {"country": "United States"}},
    }
}
EMPTY_TEMPLATE = SimpleNamespace(bindings=())


def check(source: str, rendered: str, *, target: dict | None = None) -> None:
    validate(
        source=source.encode(),
        rendered=rendered.encode(),
        source_target=SOURCE,
        target=TARGET if target is None else target,
        template=EMPTY_TEMPLATE,
    )


def test_every_repeated_free_time_place_must_have_same_discharge_owner() -> None:
    source = "21 DAYS DEMMURAGE FREETIME IN ALEXANDRIA\n" * 2
    good = "24 DAYS DEMURRAGE FREETIME IN TACOMA\n" * 2
    check(source, good)
    check(source, "")  # Whole optional statement may be omitted.
    with pytest.raises(ValueError, match="partly realized"):
        check(source, "24 DAYS DEMURRAGE FREETIME IN TACOMA\n")
    with pytest.raises(ValueError, match="contradicts sampled discharge"):
        check(source, "24 DAYS DEMURRAGE FREETIME IN TACOMA\n24 DAYS DEMURRAGE FREETIME IN DUBAI\n")


def test_unowned_liner_requires_omission_when_route_changes() -> None:
    source = "LINER IN ISTANBUL - FREE OUT ALEXANDRIA\n" * 2
    check(source, "")
    with pytest.raises(ValueError, match="no compiled two-endpoint contract"):
        check(source, "LINER IN SEATTLE - FREE OUT TACOMA\n" * 2)
    with_loading = deepcopy(TARGET)
    with_loading["documentPatch"]["route"]["portOfLoading"] = {"name": "Changshu Pt"}
    check(source, "LINER IN CHANGSHU - FREE OUT TACOMA\n" * 2, target=with_loading)
    with pytest.raises(ValueError, match="contradicts sampled endpoints"):
        check(source, "LINER IN SEATTLE - FREE OUT TACOMA\n" * 2, target=with_loading)


def test_explicit_country_and_customs_jurisdiction() -> None:
    source = "Free Time in\nEgypt: 21 days\nACID · 1000376232023060230\n"
    check(source, "Free Time in\nUnited States: 21 days\n")
    with pytest.raises(ValueError, match="free-time country contradicts"):
        check(source, source)
    with pytest.raises(ValueError, match="ACID declaration contradicts"):
        check(source, "Free Time in\nUnited States: 21 days\nACID · 1000376232023060230\n")
    egypt = deepcopy(TARGET)
    egypt["documentPatch"]["route"]["portOfDischarge"] = {
        "name": "ALEXANDRIA - OLD PORT",
        "country": "EGYPT",
    }
    egypt["documentPatch"]["parties"]["consignee"]["country"] = "EGYPT"
    check(source, source, target=egypt)


def test_optional_alexandria_terminal_and_egyptian_law_are_route_dependent() -> None:
    source = (
        "Goods. Any transportation expss of containers at Alexandria terminal "
        "to be collected. AS PER EGYPTIAN LAWS."
    )
    check(source, "Goods.")
    with pytest.raises(ValueError, match="Alexandria terminal clause"):
        check(source, source)
    with pytest.raises(ValueError, match="Egyptian-law clause"):
        check(source, "Goods. AS PER EGYPTIAN LAWS.")


def test_unrelated_free_time_place_is_not_falsely_owned() -> None:
    check("15 DAYS DEMURRAGE FREETIME IN ISTANBUL", "15 DAYS DEMURRAGE FREETIME IN ISTANBUL")


def test_port_terminal_facility_cannot_be_invented_by_changing_port_slot() -> None:
    source = (
        "Port of discharge ALEXANDRIA - OLD PORT\n"
        "ALL LANDING, DISCHARGING AD RELOADING OPERATIONS TO BE EFFECTED BY THE "
        "ALEXANDRIA PORT CONTAINERS TERMINAL AT RISK AND EXPENSE OF THE MERCHANTS."
    )
    check(source, "Port of discharge TACOMA\n")
    with pytest.raises(ValueError, match="port-terminal facility was not omitted"):
        check(
            source,
            "Port of discharge TACOMA\n"
            "ALL LANDING, DISCHARGING AD RELOADING OPERATIONS TO BE EFFECTED BY THE "
            "TACOMA PORT CONTAINERS TERMINAL AT RISK AND EXPENSE OF THE MERCHANTS.",
        )


def test_fixed_egypt_agent_heading_must_not_survive_changed_agent_locality() -> None:
    source = "NAME AND FULL ADDRESS OF SHIPPING AGENT IN EGYPT\nALEXANDRIA\nAGENT CO\n"
    check(source, source)
    check(source, "NAME AND FULL ADDRESS OF SHIPPING AGENT\nAUCKLAND\nAGENT CO\n")
    with pytest.raises(ValueError, match="Egypt-specific shipping-agent heading"):
        check(source, "NAME AND FULL ADDRESS OF SHIPPING AGENT IN EGYPT\nAUCKLAND\nAGENT CO\n")
