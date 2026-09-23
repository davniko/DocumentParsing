from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.dangerous_goods_registry import DangerousGoodsHmtRecord
from document_ocr.synthesis.template_compiler.complete_pipeline import _prepared
from document_ocr.synthesis.template_compiler.complete_targets import load_source
from document_ocr.synthesis.template_compiler.dangerous_goods_realization import (
    DangerousGoodsFact,
    compile_surfaces,
    render_facts,
    validate_un_references,
)
from document_ocr.synthesis.template_compiler.descendant import (
    _dangerous_goods_outputs,
    _require_frozen_target,
)

CATALOG = Path(
    "artifacts/kie-synthesis-production/template-base/catalogs/"
    "mpci-bl-production-template-catalog1510-v6"
).resolve()
SOURCE_ID = "doc_fec3f655ad2f77e0ef63ba1ca98a8e122c340a0c33eb57d608c46b15d1beedb5"
DG_PATH = "documentPatch.cargoGroups[0].dangerousGoods[0]"


def test_whole_rendered_text_cannot_retain_a_stale_un_in_an_unowned_region():
    validate_un_references("Hydrochloric acid, UN: 1789", {"1789"})
    with pytest.raises(ValueError, match="UN number"):
        validate_un_references("Hydrochloric acid, UN: 1789\nOld marks UN NO. 1057", {"1789"})


def fact():
    record = DangerousGoodsHmtRecord(
        record_id="hmt_" + "a" * 64,
        source_row=2,
        un_number="1789",
        proper_shipping_name="HYDROCHLORIC ACID",
        proper_shipping_name_markup="HYDROCHLORIC ACID",
        optional_qualifiers=(),
        exact_hazard_class="8",
        hazard_category="CORROSIVE_SUBSTANCES",
        exact_label_codes=("8",),
        exact_subsidiary_hazards=(),
        subsidiary_hazard_categories=(),
        packing_group_code="II",
        packing_group_category="MEDIUM_DANGER",
        symbols=(),
        technical_name_required=False,
        nos_entry=False,
        vessel_stowage_location="A",
        vessel_stowage_other=(),
        maritime_eligible=True,
        maritime_disposition="eligible",
        source_fields={"UN ID Number": "UN1789"},
    )
    return DangerousGoodsFact(target_path=DG_PATH, record=record)


def prepared():
    source = load_source(CATALOG, SOURCE_ID)
    target = deepcopy(source.target)
    target["documentPatch"]["cargoGroups"][0]["dangerousGoods"][0].update(
        unNumber="1789", hazardCategory="CORROSIVE_SUBSTANCES"
    )
    return source, _prepared(
        source,
        target,
        {},
        sample_id="synthetic",
        seed=7,
        numeric_values={},
        dangerous_goods_facts=(fact(),),
    )


def test_registry_tuple_renders_exact_classes_and_un_into_real_compiled_slots():
    source, case = prepared()
    _require_frozen_target(case)
    outputs = _dangerous_goods_outputs(case)
    surfaces = compile_surfaces(source.template)
    assert {s.field for s in surfaces} == {"un_number", "primary_class"}
    for surface in surfaces:
        expected = "1789" if surface.field == "un_number" else "8"
        assert outputs[surface.logical_key].canonical_value == expected
        assert all(expected in v for v in outputs[surface.logical_key].replacements.values())


def test_changed_dg_cannot_render_without_a_registry_tuple():
    source, case = prepared()
    missing = _prepared(source, case.target, {}, sample_id="synthetic", seed=7, numeric_values={})
    with pytest.raises(ValueError, match="require complete registry"):
        _require_frozen_target(missing)


def test_registry_context_is_frozen_with_the_target():
    _, case = prepared()
    with pytest.raises(ValueError, match="facts changed"):
        _require_frozen_target(replace(case, dangerous_goods_facts=()))


@pytest.mark.parametrize("bad", [(), (fact(), fact())])
def test_missing_and_duplicate_declaration_context_is_rejected(bad):
    source, case = prepared()
    with pytest.raises(ValueError, match=r"cover|duplicate"):
        render_facts(source=source.source, template=source.template, target=case.target, facts=bad)


def test_inconsistent_target_and_registry_tuple_is_rejected():
    source, case = prepared()
    target = deepcopy(case.target)
    target["documentPatch"]["cargoGroups"][0]["dangerousGoods"][0]["unNumber"] = "1993"
    with pytest.raises(ValueError, match="differs from its registry"):
        render_facts(source=source.source, template=source.template, target=target, facts=(fact(),))


def test_pure_chemical_flashpoint_is_not_invented_from_a_hazard_class():
    _, case = prepared()
    target = deepcopy(case.target)
    target["documentPatch"]["cargoGroups"][0]["dangerousGoods"][0]["flashPoint"] = {
        "temperature": {"value": 12.0, "unit": "celsius"}
    }
    with pytest.raises(ValueError, match="formulation-property contract"):
        fact().validate_target(target)


def caption_template(text, *, mode="literal_static", dependencies=()):
    return SimpleNamespace(
        bindings=(
            SimpleNamespace(
                logical_key="dg-caption",
                group_key="declaration:0",
                group_kind="dangerous_goods",
                render_mode=mode,
                target_paths=(),
                dependency_paths=dependencies,
                dependency_bindings=(),
                occurrences=(SimpleNamespace(source_text=text),),
            ),
        )
    )


@pytest.mark.parametrize("text", ["UN", "CLASS:", "P.G.", "Packing Group", " class. "])
def test_explicit_value_free_static_dg_captions_are_not_chemical_facts(text):
    assert compile_surfaces(caption_template(text)) == ()


@pytest.mark.parametrize("text", ["UN 1993", "CLASS 3", "P.G. III", "FLAMMABLE", "3", ""])
def test_static_dg_values_cannot_escape_fact_ownership_as_captions(text):
    with pytest.raises(ValueError, match="unique declaration owner"):
        compile_surfaces(caption_template(text))


@pytest.mark.parametrize(
    "kwargs", [dict(mode="deterministic_auxiliary"), dict(dependencies=(DG_PATH,))]
)
def test_caption_text_does_not_override_a_nonstatic_or_dependent_contract(kwargs):
    with pytest.raises(ValueError, match="unique declaration owner"):
        compile_surfaces(caption_template("UN", **kwargs))


def private_fact_template(field, text, *, owner=DG_PATH, **updates):
    anchored = SimpleNamespace(
        logical_key="un",
        group_key="target-declaration",
        group_kind="dangerous_goods",
        render_mode="target_binding",
        target_paths=(DG_PATH + ".unNumber",),
        dependency_paths=(),
        dependency_bindings=(),
        derivation=None,
        occurrences=(SimpleNamespace(source_text="1993"),),
    )
    private = SimpleNamespace(
        logical_key="private",
        group_key="separately-named-compiler-group",
        group_kind="dangerous_goods",
        value_kind="dangerous_goods",
        render_mode="deterministic_derived",
        target_paths=(),
        dependency_paths=(owner,),
        dependency_bindings=(),
        derivation="sampled_dg_" + field,
        occurrences=(SimpleNamespace(source_text=text, slot_id="private-slot"),),
    )
    for name, value in updates.items():
        setattr(private, name, value)
    return SimpleNamespace(bindings=(anchored, private))


@pytest.mark.parametrize(
    "field,text,expected",
    [
        ("packing_group", "P.G. III", "P.G. II"),
        ("packing_group", "GR III", "GR II"),
        ("packing_group", "Packing Group 3", "Packing Group 2"),
        ("shipping_name", "PROPIONIC ACID", "HYDROCHLORIC ACID"),
        ("primary_class", "CLASS 3", "CLASS 8"),
        ("un_number", "UN: 1993", "UN: 1789"),
    ],
)
def test_explicit_source_only_dg_owners_do_not_depend_on_group_key(field, text, expected):
    from document_ocr.synthesis.template_compiler.dangerous_goods_realization import _render_field

    template = private_fact_template(field, text)
    before = deepcopy(template)
    surface = compile_surfaces(template)[1]
    assert surface.target_path == DG_PATH and surface.field == field
    output = _render_field(template.bindings[1], surface, fact())
    assert output.replacements == {"private-slot": expected}
    assert template == before


@pytest.mark.parametrize(
    "updates",
    [
        dict(owner=DG_PATH + ".unNumber"),
        dict(owner=DG_PATH.replace("[0]", "[1]", 1)),
        dict(dependency_paths=(DG_PATH, DG_PATH)),
        dict(dependency_bindings=("un",)),
        dict(target_paths=(DG_PATH + ".hazardCategory",)),
        dict(render_mode="literal_static"),
        dict(group_kind="cargo"),
        dict(value_kind="location"),
    ],
)
def test_explicit_dg_contract_rejects_mixed_missing_or_malformed_owners(updates):
    with pytest.raises(ValueError, match="DG"):
        compile_surfaces(private_fact_template("packing_group", "III", **updates))


@pytest.mark.parametrize(
    "field,text",
    [
        ("packing_group", "III FLASHPOINT 20C"),
        ("primary_class", "CLASS 3 PG III"),
        ("un_number", "UN 1993 UN 1789"),
        ("shipping_name", "PROPIONIC ACID FLASHPOINT 20C"),
    ],
)
def test_explicit_dg_property_cannot_swallow_another_chemical_fact(field, text):
    with pytest.raises(ValueError, match="DG"):
        compile_surfaces(private_fact_template(field, text))


def test_explicit_private_dg_render_keeps_the_extraction_schema_unchanged():
    from document_ocr.synthesis.template_compiler.dangerous_goods_realization import _render_field

    target = {
        "documentPatch": {
            "cargoGroups": [
                {"dangerousGoods": [{"unNumber": "1789", "hazardCategory": "CORROSIVE_SUBSTANCES"}]}
            ]
        }
    }
    original = deepcopy(target)
    template = private_fact_template("packing_group", "III")
    fact().validate_target(target)
    _render_field(template.bindings[1], compile_surfaces(template)[1], fact())
    assert target == original
