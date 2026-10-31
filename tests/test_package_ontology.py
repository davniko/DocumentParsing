from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from document_ocr.package_ontology import (
    PackageComponent,
    PackageMeasurement,
    definition_includes,
    load_package_ontology,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("configs/ontology/package-properties-v1/manifest.json")


@pytest.fixture(scope="module")
def ontology():
    return load_package_ontology(ROOT / REFERENCE, project_root=ROOT)


@pytest.fixture
def editable_reference(tmp_path):
    source = ROOT / REFERENCE
    manifest = json.loads(source.read_text())
    shutil.copytree(source.parent, tmp_path / REFERENCE.parent)
    for key in ("sourceRegistry", "requiredActiveTask"):
        relative = manifest[key]["path"]
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def _save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _rewrite_shard(root, mutate, *, index=0):
    manifest_path = root / REFERENCE
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / manifest["entryFiles"][index]["path"]
    rows = json.loads(path.read_text())
    mutate(rows)
    _save(path, rows)
    manifest["entryFiles"][index]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _save(manifest_path, manifest)


def _load(root):
    return load_package_ontology(root / REFERENCE, project_root=root)


def test_full_authority_and_required_meanings_are_preserved(ontology):
    manifest = json.loads((ROOT / REFERENCE).read_text())
    registry = json.loads((ROOT / manifest["sourceRegistry"]["path"]).read_text())
    assert len(ontology.by_code) == 405
    assert len(ontology.by_category) == 407
    for row in registry["entries"]:
        definition = ontology.by_code[row["applicationCode"]]
        assert definition.category_token == row["categoryToken"]
        assert definition.display_name == row["displayName"]
        assert ontology.by_category[row["categoryToken"]] is definition
    constraints = json.loads((ROOT / manifest["requiredActiveTask"]["path"]).read_text())
    assert len(constraints["packageCategoryTokens"]) == 48
    assert set(constraints["packageCategoryTokens"]) <= ontology.by_category.keys()
    assert {"BARREL", "JERRICAN", "BUCKET", "TUB", "KIT"} <= {
        d.properties.family for d in ontology.by_category.values()
    }
    assert ontology.by_category["PACKAGE_SET"].code is None
    assert ontology.by_category["PACKAGE_BALE"].code is None
    assert ontology.by_code["CY"].properties.family == "CYLINDER"
    assert ontology.by_code["GB"].properties.family == "GAS_BOTTLE"


@pytest.mark.parametrize("key", ["dr", "DRUM", "STEEL DRUMS", "1A1", "ZZZ"])
def test_code_lookup_never_guesses_an_alias_or_a_different_namespace(ontology, key):
    with pytest.raises(KeyError):
        ontology.by_code[key]


@pytest.mark.parametrize("code", ["DR", "1A", "5M", "BG", "WA", "NE"])
def test_broader_definitions_do_not_invent_missing_attributes(ontology, code):
    assert ontology.by_code[code].properties.attributes == {}


@pytest.mark.parametrize(
    ("specific", "general", "expected"),
    [
        ("1A", "DR", True),
        ("QA", "1A", True),
        ("QB", "1A", True),
        ("1A", "QA", False),
        ("1A", "QB", False),
        ("DR", "1A", False),
        ("1A", "IH", False),
        ("QB", "QA", False),
        ("XJ", "5M", True),
        ("5M", "XJ", False),
        ("XK", "XJ", True),
        ("XJ", "XK", False),
        ("YA", "6H", True),
        ("YN", "6H", False),
        ("YN", "6P", True),
        ("YA", "1A", False),
        ("ZS", "WA", True),
        ("ZL", "ZS", True),
        ("ZS", "ZL", False),
        ("ZL", "ZM", False),
        ("WQ", "WN", False),
        ("WT", "WW", False),
        ("NF", "NE", True),
        ("NE", "NF", False),
        ("44", "EC", True),
        ("EC", "44", True),
        ("NA", "ZZ", False),
    ],
)
def test_assertion_relation_distinguishes_broader_from_unsupported_child(
    ontology, specific, general, expected
):
    assert (
        definition_includes(
            ontology.by_code[specific].properties, ontology.by_code[general].properties
        )
        is expected
    )


def test_generic_bale_is_valid_without_choosing_compression(ontology):
    generic = ontology.by_category["PACKAGE_BALE"].properties
    assert generic.attributes == {}
    assert definition_includes(ontology.by_code["BL"].properties, generic)
    assert definition_includes(ontology.by_code["BN"].properties, generic)
    assert not definition_includes(generic, ontology.by_code["BL"].properties)
    assert not definition_includes(generic, ontology.by_code["BN"].properties)


def test_component_material_cannot_supply_body_or_other_component_material(ontology):
    plastic_body_unknown_liner = ontology.by_code["WQ"].properties
    plastic_liner_required = replace(
        plastic_body_unknown_liner,
        components=(PackageComponent("LINER", None, "PLASTIC", {}),),
    )
    assert not definition_includes(plastic_body_unknown_liner, plastic_liner_required)
    composite = ontology.by_code["YA"].properties
    assert composite.material is None
    assert {c.role: c.material for c in composite.components} == {
        "INNER_RECEPTACLE": "PLASTIC",
        "OUTER_ENCLOSURE": "STEEL",
    }


def test_material_alternatives_remain_disjunctive(ontology):
    liftvan = ontology.by_code["LV"].properties
    assert liftvan.material is None
    assert set(liftvan.material_alternatives) == {"WOOD", "METAL"}
    assert not definition_includes(
        liftvan, replace(liftvan, material="WOOD", material_alternatives=())
    )
    wood_liftvan = replace(liftvan, material="WOOD", material_alternatives=())
    assert definition_includes(wood_liftvan, liftvan)
    assert not definition_includes(replace(liftvan, material_alternatives=()), liftvan)


def test_measurement_uncertainty_and_ownership_are_retained(ontology):
    assert ontology.by_code["WJ"].properties.measurements[0].relation == "UNSPECIFIED"
    assert ontology.by_code["WG"].properties.measurements[0].relation == "GT"
    assert ontology.by_code["OX"].properties.measurements[0].unit is None
    capacity = ontology.by_code["GL"].properties.measurements[0]
    assert capacity.values == (1,)
    assert capacity.unit == "GALLON_UNSPECIFIED"
    assert ontology.by_code["TG"].properties.measurements == ()
    assert ontology.by_code["CY"].properties.material is None
    assert ontology.by_code["GB"].properties.material == "METAL"
    # A formal code lookup is not an ordinary-language recognition path.
    with pytest.raises(KeyError):
        ontology.by_code["GAS BOTTLE"]


def test_all_definitions_include_themselves(ontology):
    for definition in ontology.by_category.values():
        assert definition_includes(definition.properties, definition.properties)


def test_definitions_are_immutable(ontology):
    with pytest.raises(TypeError):
        ontology.by_code["NEW"] = ontology.by_code["DR"]
    with pytest.raises(TypeError):
        ontology.by_code["QA"].properties.attributes["CLOSURE"] = "LID"
    with pytest.raises(TypeError):
        ontology.by_code["ZL"].properties.components[0].attributes["RIGIDITY"] = "FLEXIBLE"
    with pytest.raises(FrozenInstanceError):
        ontology.by_code["1A"].properties.material = "PLASTIC"


def test_concurrent_loads_are_independent_and_consistent():
    def counts(_):
        value = load_package_ontology(ROOT / REFERENCE, project_root=ROOT)
        return len(value.by_code), value.manifest_sha256

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(counts, range(8)))
    assert len(set(results)) == 1
    assert results[0][0] == 405


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "coverage missing"),
        (lambda rows: rows.append(rows[0]), "duplicate crosswalk code"),
        (lambda rows: rows[0].update(code="00"), "unknown or duplicate"),
        (lambda rows: rows[0].update(material="steel"), "String should match pattern"),
        (lambda rows: rows[0].update(attributes={"CLOSURE": "FAIRY"}), "unreviewed"),
        (lambda rows: rows[0].update(attributes={"UNKNOWN": "YES"}), "unreviewed"),
        (lambda rows: rows[0].update(definitionSources=["unknown"]), "undeclared primary"),
        (lambda rows: rows[0].update(materialAlternatives=["STEEL", "WOOD"]), "cannot also"),
        (lambda rows: rows[0].update(material=None, materialAlternatives=["WOOD"]), "multiple"),
        (lambda rows: rows[0].update(packageQuantity=1), "Extra inputs"),
        (
            lambda rows: rows[0].update(components=[{"role": "LINER"}, {"role": "LINER"}]),
            "roles must be unique",
        ),
        (
            lambda rows: rows[0].update(
                attributes={"LINING": "ABSENT"}, components=[{"role": "LINER"}]
            ),
            "absent liner",
        ),
        (
            lambda rows: rows[0].update(attributes={"CONSTRUCTION": "COMPOSITE"}),
            "component ownership",
        ),
        (lambda rows: rows[0].update(kind="ADMINISTRATIVE"), "cannot invent physical"),
    ],
)
def test_semantically_malformed_or_incomplete_reference_fails(editable_reference, mutate, message):
    _rewrite_shard(editable_reference, mutate)
    with pytest.raises(ValueError, match=message):
        _load(editable_reference)


def test_missing_semantic_extension_is_not_silently_dropped(editable_reference):
    manifest_path = editable_reference / REFERENCE
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / manifest["semanticExtensions"]["path"]
    _save(path, [])
    manifest["semanticExtensions"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _save(manifest_path, manifest)
    with pytest.raises(ValueError, match="active package category"):
        _load(editable_reference)


def test_missing_required_extra_family_fails(editable_reference):
    def remove_kit(rows):
        row = next(r for r in rows if r["code"] == "KI")
        row["family"] = "OTHER"

    _rewrite_shard(editable_reference, remove_kit, index=1)
    with pytest.raises(ValueError, match="additional pilot family"):
        _load(editable_reference)


def test_stale_hash_fails(editable_reference):
    path = editable_reference / REFERENCE.parent / "entries-001-137.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        _load(editable_reference)


def test_duplicate_json_keys_fail_before_last_value_can_win(editable_reference):
    path = editable_reference / REFERENCE.parent / "entries-001-137.json"
    path.write_text('[{"code":"1A","code":"DR"}]')
    with pytest.raises(ValueError, match="duplicate JSON property"):
        _load(editable_reference)


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json"])
def test_reference_paths_cannot_escape_their_root(editable_reference, path):
    manifest_path = editable_reference / REFERENCE
    manifest = json.loads(manifest_path.read_text())
    manifest["sourceRegistry"]["path"] = path
    _save(manifest_path, manifest)
    with pytest.raises(ValueError, match="inside its declared root"):
        _load(editable_reference)


@pytest.mark.parametrize(
    "measurement",
    [
        dict(property="DIMENSIONS", values=[1], unit="CENTIMETRE", relation="EQ"),
        dict(property="DIMENSIONS", values=[1, -2], unit="CENTIMETRE", relation="EQ"),
        dict(property="PRESSURE", values=[1, 2], unit="KILOPASCAL", relation="GT"),
        dict(property="PRESSURE", values=[0], unit="KILOPASCAL", relation="EQ"),
        dict(property="PRESSURE", values=[1], unit="CELSIUS", relation="EQ"),
        dict(property="CAPACITY", values=[-1], unit="GALLON_UNSPECIFIED", relation="EQ"),
        dict(property="TEMPERATURE", values=[float("nan")], unit="CELSIUS", relation="EQ"),
        dict(property="TEMPERATURE", values=[True], unit="CELSIUS", relation="EQ"),
    ],
)
def test_invalid_definition_measurements_are_rejected(measurement):
    with pytest.raises(ValueError):
        PackageMeasurement.model_validate_json(json.dumps(measurement))
