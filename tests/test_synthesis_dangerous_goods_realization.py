import json
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.dangerous_goods_registry import DangerousGoodsHmtRecord
from document_ocr.synthesis.template_compiler.complete_pipeline import _prepared
from document_ocr.synthesis.template_compiler.complete_targets import load_source as _load_source
from document_ocr.synthesis.template_compiler.dangerous_goods_realization import (
    DangerousGoodsFact,
    compile_surfaces,
    render_facts,
    shared_declaration_representatives,
    validate_explicit_un_source_coverage,
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
REVIEWED_CATALOG = Path(
    "artifacts/kie-synthesis-production/template-base/catalogs/"
    "mpci-bl-production-template-catalog1507-v26-dg-grounding"
).resolve()
REVIEWED_SOURCE_ID = "doc_3941a8dea9764dfbbf0bc4adc915dc075063df47406afba05f1d90984757c484"

_HISTORICAL_CASE_HASHES = {
    (REVIEWED_CATALOG, REVIEWED_SOURCE_ID): (
        "8576457a326818f0532eaeb2c3bf75e366079b97442d1a145b883fc56b31915c",
        "df5db178bbe05d1bb48dbf5dc4271f94248182df513c7010ef18b5fd5a8f2151",
        "67f0cb87bd4b0d45d839a61f407c6584469c74ebbef91a748eadddd748a286be",
    ),
    (CATALOG, SOURCE_ID): (
        "db2e9ae2f3b8b454d5a316441263f120fcff1faa25c3b43e36c5da04598ee032",
        "0e55a15e64775c673ed93314c021b9d20487365f25797e6069558779b0d35214",
        "870f913feb88b142f1d5fb55ae24b7094865d4aa2245e2819ec2081534546f44",
    ),
}


@lru_cache(maxsize=2)
def load_source(catalog: Path, document_id: str):
    """Adapt exact pre-role-certificate DG fixtures without changing their catalogs."""
    source_case = catalog / "cases" / document_id
    files = ("source.txt", "source-label.json", "template.json")
    payloads = tuple((source_case / name).read_bytes() for name in files)
    hashes = tuple(sha256_bytes(payload) for payload in payloads)
    assert hashes == _HISTORICAL_CASE_HASHES[(catalog, document_id)]
    certificate = {
        "schema_version": 1,
        "document_id": document_id,
        "admission": "source_has_no_additional_information",
        "source_sha256": hashes[0],
        "source_label_sha256": hashes[1],
        "template_sha256": hashes[2],
        "original_source_label_sha256": hashes[1],
        "original_additional_information": [],
        "critic_stage_sha256": None,
        "changed_scenario_proof_sha256": None,
        "fixed_inner_package_category": None,
    }
    with TemporaryDirectory(prefix="dg-historical-fixture-") as staging:
        root = Path(staging)
        case = root / "cases" / document_id
        case.mkdir(parents=True)
        for name, payload in zip(files, payloads, strict=True):
            (case / name).write_bytes(payload)
        (case / "goods-role-certificate.json").write_text(json.dumps(certificate))
        return _load_source(root, document_id)


def test_whole_rendered_text_cannot_retain_a_stale_un_in_an_unowned_region():
    validate_un_references("Hydrochloric acid, UN: 1789", {"1789"})
    with pytest.raises(ValueError, match="UN number"):
        validate_un_references("Hydrochloric acid, UN: 1789\nOld marks UN NO. 1057", {"1789"})


def test_printed_positive_un_declaration_requires_a_source_target() -> None:
    target = {"documentPatch": {"cargoGroups": [{"dangerousGoods": []}]}}
    validate_explicit_un_source_coverage("UN Number\nClass\nNo dangerous goods", target)
    with pytest.raises(ValueError, match="2556"):
        validate_explicit_un_source_coverage("UN Number: 2556 - IMDG Class: 4.1", target)
    target["documentPatch"]["cargoGroups"][0]["dangerousGoods"] = [{"unNumber": "2556"}]
    validate_explicit_un_source_coverage("UN Number: 2556 - IMDG Class: 4.1", target)


def test_reviewed_shared_dg_template_uses_one_tuple_across_two_cargo_groups() -> None:
    source = load_source(REVIEWED_CATALOG, REVIEWED_SOURCE_ID)
    assert source.source_target["schemaVersion"] == "5.0.0-experimental"
    assert all(
        group["dangerousGoods"][0]["packingGroupCategory"] == "MEDIUM_DANGER"
        for group in source.target["documentPatch"]["cargoGroups"]
    )
    owners = shared_declaration_representatives(source.template, source.source_target)
    expected = {f"documentPatch.cargoGroups[{i}].dangerousGoods[0]" for i in (0, 1)}
    assert set(owners) == expected
    assert len(set(owners.values())) == 1
    surfaces = compile_surfaces(source.template)
    assert {surface.field for surface in surfaces} == {
        "un_number",
        "primary_class",
        "packing_group",
        "shipping_name",
    }
    assert all(
        set(surface.shared_target_paths) == expected - {surface.target_path}
        for surface in surfaces
        if surface.field != "shipping_name"
    )
    target = deepcopy(source.target)
    target["documentPatch"]["cargoGroups"][1]["dangerousGoods"][0]["unNumber"] = "2555"
    with pytest.raises(ValueError, match="same regulatory tuple"):
        shared_declaration_representatives(source.template, target)


def test_shared_dg_slots_reject_two_independently_sampled_registry_rows() -> None:
    source = load_source(REVIEWED_CATALOG, REVIEWED_SOURCE_ID)
    target = deepcopy(source.target)
    for group in target["documentPatch"]["cargoGroups"]:
        group["dangerousGoods"][0] = {
            "unNumber": "1789",
            "hazardCategory": "CORROSIVE_SUBSTANCES",
            "packingGroupCategory": "MEDIUM_DANGER",
        }
    first = fact()
    second = DangerousGoodsFact(
        target_path="documentPatch.cargoGroups[1].dangerousGoods[0]",
        record=first.record.model_copy(update={"record_id": "hmt_" + "b" * 64}),
    )
    with pytest.raises(ValueError, match="one complete regulatory tuple"):
        render_facts(
            source=source.source,
            template=source.template,
            target=target,
            facts=(first, second),
        )


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


def test_source_only_dg_property_requires_the_declaration_owner_group() -> None:
    owner = "dangerous_goods:0:0"
    declaration = SimpleNamespace(
        logical_key="dg-un",
        group_key=owner,
        group_kind="dangerous_goods",
        render_mode="target_binding",
        target_paths=(DG_PATH + ".unNumber",),
        dependency_paths=(),
        dependency_bindings=(),
        derivation=None,
        occurrences=(SimpleNamespace(source_text="1641"),),
    )
    packing_group = SimpleNamespace(
        logical_key="dg-packing-group",
        group_key="dangerous_goods:other:0",
        group_kind="dangerous_goods",
        render_mode="deterministic_auxiliary",
        target_paths=(),
        dependency_paths=(),
        dependency_bindings=(),
        derivation=None,
        occurrences=(SimpleNamespace(source_text="PACKING GROUP III"),),
    )
    template = SimpleNamespace(bindings=(declaration, packing_group))
    with pytest.raises(ValueError, match="no unique declaration owner"):
        compile_surfaces(template)

    packing_group.group_key = owner
    surfaces = compile_surfaces(template)
    assert [(surface.logical_key, surface.target_path, surface.field) for surface in surfaces] == [
        ("dg-un", DG_PATH, "un_number"),
        ("dg-packing-group", DG_PATH, "packing_group"),
    ]


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
