"""Dependent unit aliases must replay and remain tied to one sampled measure."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import pytest

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_measurements import measured_number, printed_quantum
from document_ocr.synthesis.curated_ownership import build_owned_blueprint, ownership_surfaces
from document_ocr.synthesis.curated_physical import _owners
from document_ocr.synthesis.curated_templates import render_sampling_blueprint

PREFIX = "documentPatch.goodsItemDetails[0]"


def converted_fixture(*, pounds="1351.434", pound_unit="Lbs"):
    raw = f"GROSS {pounds} {pound_unit}\n613.000 Kgs\nVOLUME 216.126 Cbf\n6.120 Cbm\n"
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "goodsItemDetails": [
                {
                    "grossWeight": {"value": 613.0, "unit": "kilogram"},
                    "volume": {"value": 6.12, "unit": "cubic_metre"},
                }
            ]
        },
    }
    row = {"documentId": "dual_units", "joinedRawText": raw, "target": target}
    variables, targets, bindings, recipes = [], [], [], {}
    for key, measure, kind, value, printed, alias, unit in (
        ("mass", "grossWeight", "mass", "613", "613.000", pounds, "pound"),
        ("volume", "volume", "volume", "6.12", "6.120", "216.126", "cubic_foot"),
    ):
        variables.append(
            dict(
                key=key,
                kind=kind,
                value=value,
                meaning=PREFIX + "." + measure + ".value",
                required_literals=[],
                occurrences=[dict(text=printed, occurrence=1, presentation="number")],
            )
        )
        targets.append(dict(path=PREFIX + "." + measure + ".value", expression="{" + key + "}"))
        start = len(raw[: raw.index(alias)].encode())
        bindings.append(
            dict(
                logical_key=key + "_alias",
                target_paths=[],
                group_key="cargo:" + ("gross_weight" if key == "mass" else "volume"),
                value_kind="decimal_measurement",
                occurrences=[
                    dict(byte_start=start, byte_end=start + len(alias.encode()), source_text=alias)
                ],
            )
        )
        recipes[key + "_alias"] = dict(measure_path=PREFIX + "." + measure, render_unit=unit)
    contract = SourceContract(
        variables=variables, targets=targets, fixed_context="Explicit dual-unit printouts."
    )
    historical = dict(
        document_id=row["documentId"],
        source_sha256=digest(raw.encode()),
        bindings=bindings,
    )
    return row, build_owned_blueprint(row, historical, contract, {"surfaces": recipes})


def test_declared_unit_aliases_preserve_baseline_and_render_canonical_values_only():
    row, blueprint = converted_fixture()
    snapshot = deepcopy(row)
    baseline = ownership_surfaces(blueprint, row["target"], None)
    assert baseline == {"mass_alias": ["1351.434"], "volume_alias": ["216.126"]}
    assert [owner.key for owner in _owners(blueprint)] == ["mass", "volume"]
    changed = deepcopy(row["target"])
    goods = changed["documentPatch"]["goodsItemDetails"][0]
    goods["grossWeight"]["value"] = 1000
    goods["volume"]["value"] = 10
    surfaces = ownership_surfaces(blueprint, changed, None)
    assert surfaces == {"mass_alias": ["2204.623"], "volume_alias": ["353.147"]}
    rendered, _, _ = render_sampling_blueprint(
        blueprint, changed, {"mass": "1000", "volume": "10"}, surface_values=surfaces
    )
    assert rendered == "GROSS 2204.623 Lbs\n1000.000 Kgs\nVOLUME 353.147 Cbf\n10.000 Cbm\n"
    assert row == snapshot


@pytest.mark.parametrize(
    ("pounds", "unit", "error"),
    [("1351.444", "Lbs", "printed interval"), ("1351.434", "Kgs", "unit is not printed")],
)
def test_converted_alias_rejects_incompatible_number_or_unit(pounds, unit, error):
    row, blueprint = converted_fixture(pounds=pounds, pound_unit=unit)
    with pytest.raises(ValueError, match=error):
        ownership_surfaces(blueprint, row["target"], None)


def test_converted_alias_rejects_dimension_and_undeclared_ownership():
    row, blueprint = converted_fixture()
    declarations = deepcopy(blueprint.ownership_data)
    declarations["surfaces"]["mass_alias"]["render_unit"] = "cubic_foot"
    with pytest.raises(ValueError, match="unit dimension"):
        ownership_surfaces(replace(blueprint, ownership_data=declarations), row["target"], None)
    with pytest.raises(ValueError, match="active historical surface owner"):
        ownership_surfaces(replace(blueprint, regions=()), row["target"], None)


def test_grouped_measurement_precision_is_proven_not_guessed():
    assert measured_number("25 123,96 KG", Decimal("25123.96")) == (
        "25 123,96",
        Decimal("25123.96"),
    )
    assert printed_quantum("25 123,96", Decimal("25123.96")) == Decimal(".01")
    for text in ("25 123 96", "25 KG 123 KG", "25 12,396"):
        with pytest.raises(ValueError):
            measured_number(text, Decimal("25123.96"))
