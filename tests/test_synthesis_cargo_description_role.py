"""Product-only source slots cannot acquire redundant shipment declarations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.cargo_description_role import (
    product_only_source_groups,
    requested_product_only_paths,
    structured_declarations,
    validate,
)


def _target(description: str) -> dict[str, object]:
    return {"documentPatch": {"cargoGroups": [{"description": description}]}}


def _template(
    surface: str,
    *,
    render_mode: str = "target_binding",
    target_paths: tuple[str, ...] = ("documentPatch.cargoGroups[0].description",),
) -> SimpleNamespace:
    occurrence = SimpleNamespace(source_text=surface, target_paths=target_paths)
    binding = SimpleNamespace(
        render_mode=render_mode,
        target_paths=target_paths,
        occurrences=(occurrence,),
    )
    return SimpleNamespace(bindings=(binding,))


@pytest.mark.parametrize(
    ("description", "category"),
    [
        ("Frozen beef; 630 cartons", "package_count"),
        ("Thirty pallets of frozen beef", "package_count"),
        ("Frozen beef in 630 cartons", "package_count"),
        ("Fresh Blushgold apricots in cartons", "outer_packaging"),
        ("LLDPE resin packed in bags", "outer_packaging"),
        ("Steel coils, seven coils", "package_count"),
        ("Process oil, ONE TANK", "package_count"),
        ("Process oil packed in 41 woven bags", "package_count"),
        ("Lubricant packed in 164 intermediate bulk containers", "package_count"),
        ("Glass-fibre batts, 426 packs and 1208 rolls", "package_count"),
        ("LCD panels, 48 palletized units", "package_count"),
        ("Frozen beef, gross weight 12,500 kg", "shipment_mass"),
        ("Frozen beef, 12,500 kg, 28.2 CBM", "shipment_mass"),
        ("Frozen beef; volume 28.2 m3", "shipment_volume"),
        ("Frozen beef, HS CODE 020230", "tariff_or_dangerous_goods_code"),
        ("Frozen beef, HS codes 020230 and 020220", "tariff_or_dangerous_goods_code"),
        ("Frozen beef, UN 1203", "tariff_or_dangerous_goods_code"),
        ("Frozen beef loaded in one 40-foot refrigerated container", "equipment_placement"),
        ("Frozen beef, temperature -18 C", "temperature_setpoint"),
    ],
)
def test_generated_structured_padding_is_rejected(description: str, category: str) -> None:
    source = _target("Frozen beef")
    template = _template("Frozen beef")
    assert category in structured_declarations(description)
    with pytest.raises(ValueError, match=category):
        validate(source_target=source, template=template, target=_target(description))
    validate(source_target=source, template=template, target=_target("Chilled lamb cuts"))


@pytest.mark.parametrize(
    "description",
    [
        "Hand-operated mechanical appliances, 10 kg or less, for preparing food",
        "Normal white garlic, 10KG CARTON",
        "Pumps designed for 0.25 m3 each",
        "Pumps, volume 0.25 m3 each",
        "Garlic, gross weight 10 kg per carton",
        "Drums, gross weight 30 kg / package",
        "Bottles, unit gross weight 12 kg",
        "Cotton fabric for 20-foot awnings",
        "Brand Five Bags organic flour",
        "Uncoated paperboard in rolls or rectangular sheets",
    ],
)
def test_product_specification_is_not_a_shipment_declaration(description: str) -> None:
    source = _target("Frozen beef")
    assert not structured_declarations(description)
    validate(
        source_target=source,
        template=_template("Frozen beef"),
        target=_target(description),
    )


def test_source_role_requires_exact_complete_physical_binding() -> None:
    source = _target("Frozen beef")
    assert product_only_source_groups(source, _template("Frozen beef")) == {0}
    assert not product_only_source_groups(source, _template("Frozen beef cartons"))
    assert not product_only_source_groups(source, _template("Frozen beef", render_mode="literal"))
    assert not product_only_source_groups(
        source,
        _template(
            "Frozen beef",
            target_paths=(
                "documentPatch.cargoGroups[0].description",
                "documentPatch.cargoGroups[0].additionalInformation",
            ),
        ),
    )


def test_source_with_structural_description_is_not_product_only() -> None:
    source = _target("Frozen beef, 630 cartons")
    assert not product_only_source_groups(source, _template("Frozen beef, 630 cartons"))
    source = _target("Fresh apricots in cartons")
    assert not product_only_source_groups(source, _template("Fresh apricots in cartons"))


def test_group_cardinality_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="cardinality"):
        validate(
            source_target=_target("Frozen beef"),
            template=_template("Frozen beef"),
            target={"documentPatch": {"cargoGroups": []}},
        )


def test_prompt_receives_only_requested_source_proven_paths() -> None:
    source = _target("Frozen beef")
    template = _template("Frozen beef")
    path = "documentPatch.cargoGroups[0].description"
    assert requested_product_only_paths(
        source, template, ({"paths": [path, "documentPatch.parties.shipper.name"]},)
    ) == (path,)
    assert not requested_product_only_paths(
        source, template, ({"paths": ["documentPatch.parties.shipper.name"]},)
    )
    with pytest.raises(ValueError, match="paths"):
        requested_product_only_paths(source, template, ({"paths": None},))
