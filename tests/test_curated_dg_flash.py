"""Official property support is pinned, exactly joined and faithfully rendered."""

import importlib.util
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_curated_dg_presentations import record
from test_curated_physical import PREFIX, fixture
from test_curated_scenarios import catalog, source_row

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_dg_flash import (
    DgFlashPoint,
    DgFlashPointRegistry,
    load_flash_points,
)
from document_ocr.synthesis.curated_physical import _operational
from document_ocr.synthesis.curated_scenarios import SourceCapabilities


def property_record():
    return DgFlashPoint(
        hmt_record_id="hmt_" + "a" * 64,
        cas_number="75-07-0",
        un_number="1089",
        chemical_name="Acetaldehyde",
        value_fahrenheit="-36",
        source_url="https://www.cdc.gov/niosh/npg/npgd0001.html",
        source_sha256="b" * 64,
        source_literal="-36°F",
        method="closed_cup",
    )


def registry():
    return DgFlashPointRegistry(
        schema_version=1,
        source_index_url="https://www.cdc.gov/niosh/npg/npgdcas.html",
        source_index_sha256="c" * 64,
        hmt_sha256="d" * 64,
        ecics_sha256="e" * 64,
        records=(property_record(),),
    )


def identity_authority():
    return SimpleNamespace(
        hmt_sha256="d" * 64,
        ecics_sha256="e" * 64,
        hmt_records=(
            SimpleNamespace(
                record_id=property_record().hmt_record_id,
                un_number="1089",
                nos_entry=False,
                technical_name_required=False,
            ),
        ),
        ecics_links=(
            SimpleNamespace(
                disposition="eligible_unique_maritime_hmt",
                cas_numbers=("75-07-0",),
                hmt_record_id=property_record().hmt_record_id,
                un_number="1089",
            ),
        ),
    )


def test_official_property_join_and_source_unit_conversion():
    value = property_record()
    assert value.value("fahrenheit") == Decimal(-36)
    assert value.value("celsius") == Decimal(-68) * 5 / 9
    assert registry().identity_index(identity_authority()) == {value.hmt_record_id: value}
    with pytest.raises(ValueError, match="unsupported flash-point"):
        value.value("kelvin")


@pytest.mark.parametrize("fault", ["hash", "cas", "un", "nos", "technical", "duplicate"])
def test_corrupt_or_formulation_specific_authority_never_passes(fault):
    authority = identity_authority()
    value = registry()
    if fault == "hash":
        authority.hmt_sha256 = "f" * 64
    elif fault == "cas":
        authority.ecics_links[0].cas_numbers = ("64-17-5",)
    elif fault == "un":
        authority.hmt_records[0].un_number = "1170"
    elif fault == "nos":
        authority.hmt_records[0].nos_entry = True
    elif fault == "technical":
        authority.hmt_records[0].technical_name_required = True
    else:
        value = value.model_copy(update={"records": value.records * 2})
    with pytest.raises(ValueError):
        value.identity_index(authority)


def test_registry_content_and_count_pins(tmp_path):
    path = tmp_path / "properties.json"
    path.write_text(registry().model_dump_json())
    assert load_flash_points(path, sha256_file(path), 1) == registry()
    with pytest.raises(ValueError, match="count"):
        load_flash_points(path, sha256_file(path), 2)
    with pytest.raises(ValueError, match="content"):
        load_flash_points(path, "0" * 64, 1)


def test_sampler_uses_matching_official_fact_only_for_printed_flash_slots():
    row = source_row()
    declarations = [
        {
            "unNumber": "1824",
            "hazardCategory": "CORROSIVE_SUBSTANCES",
            "flashPoint": {"temperature": {"value": 12, "unit": "celsius"}},
        }
    ]
    row["target"]["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = declarations
    support = catalog([row])
    selected = record()
    selected.record_id, selected.un_number = property_record().hmt_record_id, "1089"
    support.chemical_candidates = ((record(), "520512"), (selected, "520512"))
    support.flash_points = {selected.record_id: property_record()}
    result = support.sample(
        row, seed=19, variant=1, capabilities=SourceCapabilities(family="dg_chemical")
    )
    actual = result.cargo["goodsItemDetails"][0]["dangerousGoods"][0]
    assert actual["unNumber"] == "1089"
    assert actual["flashPoint"]["temperature"]["value"] == float(Decimal(-68) * 5 / 9)
    assert result.provenance["dgPrintedFacts"]["flashPointAuthority"]["cas_number"] == "75-07-0"
    del row["target"]["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"][0]["flashPoint"]
    result = support.sample(
        row, seed=19, variant=1, capabilities=SourceCapabilities(family="dg_chemical")
    )
    assert "flashPoint" not in result.cargo["goodsItemDetails"][0]["dangerousGoods"][0]
    assert "flashPointAuthority" not in result.provenance["dgPrintedFacts"]


@pytest.mark.parametrize(
    ("text", "baseline", "converted", "expected", "number"),
    [
        ("12 DEG C", 12, "-37.777777777", "-38 DEG C", -38),
        ("FP 110.0`C", 110, "32.777777777", "FP 32.8`C", 32.8),
        ("FLASH -36°F", -36, "44", "FLASH 44°F", 44),
    ],
)
def test_flash_rendering_rounds_label_to_the_actual_printed_source_precision(
    text, baseline, converted, expected, number
):
    row, blueprint, scenario, _ = fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = [
        {"flashPoint": {"temperature": {"value": baseline, "unit": "celsius"}}}
    ]
    path = f"{PREFIX}.dangerousGoods[0].flashPoint.temperature.value"
    blueprint = replace(
        blueprint,
        target=deepcopy(target),
        historical_bindings={
            "flash": {
                "target_paths": [path],
                "occurrences": [
                    {"source_text": text},
                    {"source_text": text},
                ],
            }
        },
        regions=(*blueprint.regions, SimpleNamespace(key="flash", curated_key=None)),
    )
    target["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"][0]["flashPoint"][
        "temperature"
    ]["value"] = float(converted)
    scenario.provenance["dgPrintedFacts"] = {"flashPointAuthority": {"source": "NIOSH"}}
    surfaces, receipt = {}, {}
    _operational(blueprint, scenario, target, surfaces, receipt)
    assert surfaces == {"flash": [expected] * 2}
    assert (
        target["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"][0]["flashPoint"][
            "temperature"
        ]["value"]
        == number
    )
    assert receipt["flashPointRounding"][0]["printedValue"] == str(Decimal(str(number)))
    scenario.provenance["dgPrintedFacts"] = {"class": "3"}
    with pytest.raises(ValueError, match="property authority"):
        _operational(blueprint, scenario, target, {}, {})


def compiler():
    path = Path(__file__).resolve().parents[1] / "scripts/synthesis/compile_dg_flashpoints.py"
    spec = importlib.util.spec_from_file_location("niosh_flash_compiler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("literal", ["NA", "?", "(oc) 42°F", "36-39°F", "<4°F", "100-162°F"])
def test_official_non_scalar_or_open_cup_facts_are_not_guessed(literal):
    html = "".join(
        f'<div class="card-title">{title}</div><div class="card-text">{text}</div>'
        for title, text in [
            ("CAS No.", "75-07-0"),
            ("DOT ID & Guide", "1089 129"),
            ("Flash Point", literal),
        ]
    )
    assert compiler().scalar_flash_point(html)[1] is None


def test_card_parser_does_not_read_explanatory_flammability_thresholds():
    html = "<p>Liquid with a flash point below 140°F.</p>" + "".join(
        f'<div class="card-title">{title}</div><div class="card-text">{text}</div>'
        for title, text in [
            ("CAS No.", "75-07-0"),
            ("DOT ID & Guide", "1089 129"),
            ("Flash Point", "\u221236°F"),
        ]
    )
    assert compiler().scalar_flash_point(html)[1] == Decimal(-36)
    with pytest.raises(ValueError, match="structure changed"):
        compiler().scalar_flash_point("<p>Flash Point -36°F</p>")
    with pytest.raises(ValueError, match="duplicate"):
        compiler().scalar_flash_point(html + html)
