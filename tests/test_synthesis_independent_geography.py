from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from document_ocr.hashing import canonical_json_sha256, sha256_bytes
from document_ocr.synthesis.country_presentations import Registry
from document_ocr.synthesis.curated import (
    Occurrence,
    SourceContract,
    TargetBinding,
    Variable,
    digest,
)
from document_ocr.synthesis.curated_auxiliary import (
    augment_auxiliary_blueprint,
    load_auxiliary_contract,
)
from document_ocr.synthesis.curated_countries import bind_postal_country_alias_owners
from document_ocr.synthesis.curated_equipment_inventory import compile_equipment_inventory
from document_ocr.synthesis.curated_ownership import load_owned_blueprint
from document_ocr.synthesis.curated_scenarios import ScenarioLocation
from document_ocr.synthesis.curated_templates import (
    RenderRegion,
    SamplingBlueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate,
    load_independent_geography_certificates,
    project_composite_locality_homonyms,
    recompile_projected_equipment_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
AUTHORITY = REGISTRY / "independent-geography-v5"
CERTIFICATE_PATH = AUTHORITY / "geography_certificates.json"
CERTIFICATE_SHA256 = "527ae4a3c67c3b3da41172904489b245e42db83d67d4a405a08850e592e375f9"
ACTIVE_CONFIG = REGISTRY / "independent-flat-full-registry-20261021/working-config.json"
SOURCE_DATASET = (
    REGISTRY / "source-authorities/larger-controlled-legacy-closure-20261020/source-dataset"
)


@pytest.fixture(scope="module")
def certificates():
    active_pin = json.loads(ACTIVE_CONFIG.read_bytes())["independent"]["geography_certificates"]
    return load_independent_geography_certificates(
        (ROOT / active_pin["path"]).read_bytes(), expected_sha256=active_pin["sha256"]
    )


@pytest.fixture(scope="module")
def gate(certificates):
    return IndependentGeographyGate(certificates)


@pytest.fixture(scope="module")
def blueprints(certificates):
    rows = {
        row["documentId"]: row
        for row in map(json.loads, (SOURCE_DATASET / "train.jsonl").read_text().splitlines())
    }
    countries = Registry.load(ROOT)
    auxiliary = load_auxiliary_contract(REGISTRY / "auxiliary.yaml")
    result = {}
    for source_id in certificates.sources:
        case = REGISTRY / "cases" / source_id
        contract = SourceContract.model_validate(
            json.loads((case / "contract.json").read_bytes())["contract"]
        )
        blueprint = load_owned_blueprint(
            rows[source_id],
            json.loads((case / "template.json").read_bytes()),
            contract,
            REGISTRY / "ownership.yaml",
        )
        result[source_id] = augment_auxiliary_blueprint(
            bind_postal_country_alias_owners(blueprint, countries), auxiliary
        )
    return result


def _scenario(blueprint, *, name="ROTTERDAM", country="NETHERLANDS", code="NL"):
    location = ScenarioLocation(
        name=name,
        country=country,
        country_code=code,
        registry="unlocode_wpi",
        registry_id="NLRTM",
    )
    return SimpleNamespace(
        party_localities={
            actor["context_party"]: location
            for actor in blueprint.ownership_data["related_actors"].values()
        }
    )


def _good_values(blueprint, scenario, gate):
    values = {variable.key: variable.value for variable in blueprint.contract.variables}
    for assembly in gate.certificates[blueprint.document_id].assemblies:
        location = gate.location_for(blueprint, assembly, scenario)
        for index, key in enumerate(assembly.postal_keys):
            text = f"MERIDIAN HOUSE {31 + index}, TRADE STREET"
            if assembly.locality is not None and assembly.locality.component == key:
                text += "\n" + location.name
            if (
                assembly.country is not None
                and assembly.country.component == key
                and not (
                    assembly.locality is not None
                    and assembly.locality.component == key
                    and location.name == location.country
                )
            ):
                text += ", " + location.country
            values[key] = text
        if assembly.scope == "country_only":
            values[assembly.postal_keys[0]] = location.country
        elif assembly.scope == "locality_country":
            values[assembly.postal_keys[0]] = (
                location.name
                if location.name == location.country
                else location.name + ", " + location.country
            )
    return values


def test_certificate_loader_is_explicit_and_pinned(certificates):
    assert len(certificates.sources) == 29
    assert sum(len(value.assemblies) for value in certificates.sources.values()) == 30
    assert certificates.inventory_sha256 == (
        "ce6cc701f7fd59993d12e6c57846b71ff08cc14e62b919b008c0a57aa0eb52ff"
    )
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        load_independent_geography_certificates(
            CERTIFICATE_PATH.read_bytes(), expected_sha256="0" * 64
        )
    duplicated = (
        b'{"schemaVersion":1,"schemaVersion":1,"inventorySha256":"'
        + (b"0" * 64)
        + b'","sources":{}}'
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_independent_geography_certificates(
            duplicated, expected_sha256=sha256_bytes(duplicated)
        )


def test_reviewed_metadata_successors_change_only_ownership_pins(blueprints):
    phase_changes = {
        "doc_97e9b467b37f213e5f9566b74b2bb392fa0239b30a0681bf157d763740802eae": (
            "b79785d0aa760a7b123181a053edd8921f76b9db4c8ebaf01bc84b2705a37593",
            "f8df91527e3a7f0d5578e9598f289d7b0f8d02055dd06724aab8dd4d2cd465b2",
        ),
        "doc_baf806435410f5474705794f489b31e103a654b9ca3a5853524a1feb46955dd4": (
            "682c8dace2435812f029ed8567395b876ce9cb84ecdfa50ecf1f4db4a1098d31",
            "46d1d0fff6c39fe96278982910c258a28eb70e5089d9572c3a6cdb8c5236300f",
        ),
    }
    route_change = {
        "doc_00a165a2adf7156bdecd557c885de531a2a69a98a920a4e029d491bd5d50b3ed": (
            "14bac438ff7f116547369cd09f4aa8ce9907176103b2a5bd0f6ade0ce77b3609",
            "2d2cabc83fc6c60dd95a6e5497d5ad2d40d7e55c871f6be4f7b565564c2f1415",
        )
    }
    predecessor_path = REGISTRY / "independent-geography-v2/geography_certificates.json"
    predecessor_bytes = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_bytes)
    successor = json.loads(
        (REGISTRY / "independent-geography-v4/geography_certificates.json").read_bytes()
    )
    for source_id, (old_pin, new_pin) in (phase_changes | route_change).items():
        old_certificate = predecessor["sources"][source_id]
        new_certificate = successor["sources"][source_id]
        assert (
            old_certificate["effective_contract_sha256"]
            == new_certificate["effective_contract_sha256"]
        )
        assert old_certificate["source_sha256"] == new_certificate["source_sha256"]
        assert old_certificate["ownership_sha256"] == old_pin
        assert new_certificate["ownership_sha256"] == new_pin
        predecessor["sources"][source_id]["ownership_sha256"] = new_pin
    assert predecessor == successor

    # The predecessor gate itself proves that adding the required execution
    # phase did not change any certified postal/name/component owner.
    old_certificates = load_independent_geography_certificates(
        predecessor_bytes,
        expected_sha256="9a3669cdcedf09e2cb737718c8cbdd0af41d1653341fc2c61ce4ec1c36571e1b",
    )
    gate = IndependentGeographyGate(old_certificates)
    for source_id in phase_changes:
        current = blueprints[source_id]
        ownership = copy.deepcopy(current.ownership_data)
        declarations = ownership["dependent_text"]
        assert declarations and all(item["phase"] == "scenario" for item in declarations)
        for declaration in declarations:
            del declaration["phase"]
        predecessor_blueprint = replace(current, ownership_data=ownership)
        gate.bind_blueprint(predecessor_blueprint, current)
        scenario = _scenario(current)
        assembly = old_certificates.sources[source_id].assemblies[0]
        assert gate.location_for(current, assembly, scenario).name == "ROTTERDAM"


def test_three_page_route_rebase_preserves_all_postal_assemblies():
    before = json.loads(
        (REGISTRY / "independent-geography-v4/geography_certificates.json").read_bytes()
    )
    after = json.loads(CERTIFICATE_PATH.read_bytes())
    sid = "doc_0ba3ad4fcd4e713df5520786bd64916559b17258db1ba013ef90ec0d38e45a9f"
    assert before["sources"][sid]["ownership_sha256"] != after["sources"][sid]["ownership_sha256"]
    before["sources"][sid]["ownership_sha256"] = after["sources"][sid]["ownership_sha256"]
    before["inventorySha256"] = sha256_bytes((AUTHORITY / "owner-rebase-review.json").read_bytes())
    assert before == after


def test_all_reviewed_sources_accept_exact_partial_scope_and_render(blueprints, gate):
    for blueprint in blueprints.values():
        scenario = _scenario(blueprint)
        values = _good_values(blueprint, scenario, gate)
        assert gate.validate_values(blueprint, values, scenario) == []
        rendered, target, proof = render_sampling_blueprint(blueprint, blueprint.target, values)
        candidate = {
            "joinedRawText": rendered,
            "renderValues": values,
            "target": target,
            "proof": proof,
        }
        assert target == blueprint.target
        assert gate.validate_rendered(blueprint, candidate, scenario) == []


def test_omitted_geography_is_not_inferred(blueprints, gate):
    blueprint = next(value for key, value in blueprints.items() if key.startswith("doc_e18486c6"))
    scenario = _scenario(blueprint)
    values = _good_values(blueprint, scenario, gate)
    assert "ROTTERDAM" not in values["f14_0"] + values["f14_1"]
    assert gate.validate_values(blueprint, values, scenario) == []
    bad = {**values, "f14_0": values["f14_0"] + ", ROTTERDAM, NETHERLANDS"}
    assert gate.validate_values(blueprint, bad, scenario)


def test_stale_contract_and_missing_values_fail_closed(blueprints, gate):
    blueprint = next(iter(blueprints.values()))
    with pytest.raises(ValueError, match="differs from active blueprint"):
        gate.validate_values(
            replace(blueprint, source=blueprint.source + " "), {}, _scenario(blueprint)
        )
    with pytest.raises(ValueError, match="missing generated values"):
        gate.validate_values(blueprint, {}, _scenario(blueprint))


def test_prepared_blueprint_binding_is_exact_and_idempotent(blueprints, certificates):
    blueprint = next(iter(blueprints.values()))
    local_gate = IndependentGeographyGate(certificates)
    prepared = replace(
        blueprint,
        contract=blueprint.contract.model_copy(
            update={"fixed_context": blueprint.contract.fixed_context + "\nPrepared."}
        ),
    )
    local_gate.bind_blueprint(blueprint, prepared)
    local_gate.bind_blueprint(blueprint, prepared)
    conflicting = replace(
        prepared,
        contract=prepared.contract.model_copy(
            update={"fixed_context": prepared.contract.fixed_context + " Different."}
        ),
    )
    with pytest.raises(ValueError, match="already binds a different"):
        local_gate.bind_blueprint(blueprint, conflicting)


NAME = "documentPatch.route.finalDestination.name"
COUNTRY = "documentPatch.route.finalDestination.country"


def _certified_three_page_projection(blueprints, role):
    sid = "doc_0ba3ad4fcd4e713df5520786bd64916559b17258db1ba013ef90ec0d38e45a9f"
    blueprint = blueprints[sid]
    target = copy.deepcopy(blueprint.target)
    target["documentPatch"]["route"][role] = {"name": "SINGAPORE", "country": "SINGAPORE"}
    country_path = "documentPatch.route." + role + ".country"
    regions = [region for region in blueprint.regions if country_path in region.target_paths]
    candidate = {
        "target": target,
        "proof": {
            "sourceSha256": digest(blueprint.source.encode()),
            "edits": [
                {
                    "key": region.key,
                    "occurrence": region.occurrence,
                    "byteStart": region.start,
                    "byteEnd": region.end,
                    "before": region.source,
                    "after": "SINGAPORE, SINGAPORE",
                    "targetPaths": list(region.target_paths),
                }
                for region in regions
            ],
        },
    }
    plan = (blueprint, _Scenario(replacements={}), "sample", target)
    projected, receipt = project_composite_locality_homonyms(plan, candidate)
    assert receipt is not None
    return blueprint, projected[0]


def test_locality_projection_postal_gate_is_isolated_and_exact(blueprints, certificates):
    original, receipt_projection = _certified_three_page_projection(blueprints, "placeOfReceipt")
    _, loading_projection = _certified_three_page_projection(blueprints, "portOfLoading")
    gate = IndependentGeographyGate(certificates)
    gate.bind_blueprint(original, original)
    before = dict(gate._prepared)
    receipt_view = gate.locality_projection_view(original, receipt_projection)
    loading_view = gate.locality_projection_view(original, loading_projection)
    assert gate._prepared == before
    assert receipt_view is not gate and loading_view is not gate
    assert receipt_view._certificate(receipt_projection) is not None
    assert loading_view._certificate(loading_projection) is not None
    for wrong_gate, blueprint in ((gate, receipt_projection), (receipt_view, loading_projection)):
        with pytest.raises(ValueError, match="differs from active blueprint"):
            wrong_gate._certificate(blueprint)


@pytest.mark.parametrize("mutation", ["source", "document_id", "postal", "party_target"])
def test_locality_projection_cannot_rebind_changed_postal_ownership(
    blueprints, certificates, mutation
):
    original, projected = _certified_three_page_projection(blueprints, "placeOfReceipt")
    gate = IndependentGeographyGate(certificates)
    if mutation == "source":
        projected = replace(projected, source=projected.source + "changed")
    elif mutation == "document_id":
        projected = replace(projected, document_id="different-source")
    elif mutation == "party_target":
        target = copy.deepcopy(projected.target)
        target["documentPatch"]["parties"]["shipper"]["name"] = "UNREVIEWED COMPANY"
        projected = replace(projected, target=target)
    else:
        region = next(r for r in projected.regions if r.curated_key == "ob_related_postal")
        projected = replace(
            projected,
            regions=tuple(
                replace(r, source=r.source + "changed") if r is region else r
                for r in projected.regions
            ),
        )
    with pytest.raises(ValueError, match="locality projection changed"):
        gate.locality_projection_view(original, projected)


class _Contract(BaseModel):
    targets: list[TargetBinding]


class _Scenario(BaseModel):
    replacements: dict


@dataclass(frozen=True)
class _ProjectionBlueprint:
    source: str
    target: dict
    regions: tuple
    historical_bindings: dict
    ownership_data: dict
    contract: _Contract


def _projection_fixture(*, printed="Djibouti, Djibouti", country="DJIBOUTI"):
    source = "++EL DEKHEILA, EGYPT|EL DEKHEILA|OTHER EGYPT"
    region = RenderRegion("long", 0, 2, 20, "EL DEKHEILA, EGYPT", (NAME, COUNTRY), "location")
    short = RenderRegion("short", 0, 21, 32, "EL DEKHEILA", (NAME,), "location")
    target = {
        "documentPatch": {
            "route": {
                "finalDestination": {"name": "DJIBOUTI", "country": country},
                "portOfDischarge": {"name": "DJIBOUTI", "country": "DJIBOUTI"},
            }
        }
    }
    binding = {
        "target_paths": [NAME, COUNTRY],
        "occurrences": [{"target_paths": [NAME, COUNTRY], "source_text": region.source}],
    }
    blueprint = _ProjectionBlueprint(
        source,
        copy.deepcopy(target),
        (region, short),
        {"long": binding, "short": {"target_paths": [NAME], "occurrences": [{}]}},
        {"surfaces": {"long": {"expression": "{" + NAME + "}, {" + COUNTRY + "}"}}},
        _Contract(targets=[TargetBinding(path=COUNTRY, expression="country")]),
    )
    scenario = _Scenario(replacements={NAME: "DJIBOUTI", COUNTRY: country, "other": "same"})
    candidate = {
        "target": target,
        "proof": {
            "sourceSha256": digest(source.encode()),
            "edits": [
                {
                    "key": "long",
                    "occurrence": 0,
                    "byteStart": 2,
                    "byteEnd": 20,
                    "before": region.source,
                    "after": printed,
                    "targetPaths": [NAME, COUNTRY],
                }
            ],
        },
    }
    return (blueprint, scenario, "sample", copy.deepcopy(target)), candidate


def test_homonym_projection_is_narrow_and_nonmutating():
    plan, candidate = _projection_fixture()
    witness = copy.deepcopy((plan, candidate))
    projected, receipt = project_composite_locality_homonyms(plan, candidate)
    assert projected[3]["documentPatch"]["route"]["finalDestination"] == {"name": "DJIBOUTI"}
    assert projected[3]["documentPatch"]["route"]["portOfDischarge"]["country"] == ("DJIBOUTI")
    assert receipt is not None and receipt["fields"][0]["countryPath"] == COUNTRY
    assert (plan, candidate) == witness
    for printed in ("Djibouti, DJ", "Djibouti, Republic of Djibouti"):
        unchanged, receipt = project_composite_locality_homonyms(
            *_projection_fixture(printed=printed)
        )
        assert receipt is None
        assert unchanged[0] is not None


def test_split_inline_homonym_cannot_silently_bypass_projection():
    plan, candidate = _projection_fixture()
    source = "SHANGHAI, CHINA"
    blueprint = replace(plan[0], source=source)
    candidate["proof"] = {
        "sourceSha256": digest(source.encode()),
        "edits": [
            {
                "key": "city",
                "occurrence": 0,
                "byteStart": 0,
                "byteEnd": 8,
                "before": "SHANGHAI",
                "after": "DJIBOUTI",
                "targetPaths": [NAME],
            },
            {
                "key": "country",
                "occurrence": 0,
                "byteStart": 10,
                "byteEnd": 15,
                "before": "CHINA",
                "after": "DJIBOUTI",
                "targetPaths": [COUNTRY],
            },
        ],
    }
    with pytest.raises(ValueError, match="complete frame ownership"):
        project_composite_locality_homonyms((blueprint, *plan[1:]), candidate)


def _equipment_blueprint():
    identifier = "ABCU0000000"
    source = identifier + " 20GP\nEGYPT"
    identifier_path = "documentPatch.containerInformation[0].equipmentIdentifier"
    country_path = "documentPatch.route.portOfDischarge.country"
    rows = [
        {
            "equipmentIdentifier": identifier,
            "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
            "typeCategory": "GENERAL_PURPOSE",
        }
    ]
    target = {
        "documentPatch": {
            "containerInformation": rows,
            "route": {"portOfDischarge": {"name": "EGYPT", "country": "EGYPT"}},
        }
    }
    identifier_end = len(identifier)
    equipment_start = identifier_end + 1
    country_start = source.rindex("EGYPT")
    return SamplingBlueprint(
        "source-test",
        source,
        target,
        SourceContract(
            variables=[
                Variable(
                    key="identity",
                    kind="container",
                    value=identifier,
                    meaning="Container identity",
                    required_literals=[],
                    occurrences=[Occurrence(text=identifier, occurrence=1, presentation="text")],
                )
            ],
            targets=[TargetBinding(path=identifier_path, expression="{identity}")],
            fixed_context="Equipment projection test",
        ),
        (
            RenderRegion(
                "identity",
                0,
                0,
                identifier_end,
                identifier,
                (identifier_path,),
                "identifier",
                "identity",
            ),
            RenderRegion(
                "equipment",
                0,
                equipment_start,
                equipment_start + 4,
                "20GP",
                (
                    "documentPatch.containerInformation[0].sizeCategory",
                    "documentPatch.containerInformation[0].typeCategory",
                ),
                "equipment",
            ),
            RenderRegion(
                "country",
                0,
                country_start,
                country_start + 5,
                "EGYPT",
                (country_path,),
                "location",
            ),
        ),
        {
            "equipment": {
                "target_paths": ["documentPatch.containers[0].typeDescription"],
                "occurrences": [
                    {
                        "byte_start": equipment_start,
                        "byte_end": equipment_start + 4,
                        "source_text": "20GP",
                    }
                ],
                "target_relationship": "single_target",
            }
        },
        {},
        (),
    )


def test_locality_projection_recompiles_identical_equipment_ownership():
    ordinary = _equipment_blueprint()
    original_target = copy.deepcopy(ordinary.target)
    sampled = copy.deepcopy(original_target)
    sampled["documentPatch"]["route"]["portOfDischarge"] = {
        "name": "HONG KONG",
        "country": "HONG KONG",
    }
    projected_source = copy.deepcopy(original_target)
    projected_target = copy.deepcopy(sampled)
    del projected_source["documentPatch"]["route"]["portOfDischarge"]["country"]
    del projected_target["documentPatch"]["route"]["portOfDischarge"]["country"]
    projected = replace(ordinary, target=projected_source)
    original_inventory = compile_equipment_inventory(ordinary)
    result = recompile_projected_equipment_inventory(
        ordinary, projected, sampled, projected_target, original_inventory
    )
    assert result.source_target_sha256 == canonical_json_sha256(projected.target)
    with pytest.raises(ValueError, match="locality countries"):
        recompile_projected_equipment_inventory(
            ordinary, ordinary, sampled, sampled, original_inventory
        )
