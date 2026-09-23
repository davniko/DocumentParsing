import asyncio
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.template_compiler import complete_pipeline as pipeline
from document_ocr.synthesis.template_compiler import lexical_facts as host
from document_ocr.synthesis.template_compiler.cargo_scenarios import CargoScenario
from document_ocr.synthesis.template_compiler.contact_values import validate_phone


def cargo():
    target = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "NATURAL LATEX",
                    "hsCodes": ["400219"],
                }
            ],
            "cargoPackages": [],
        }
    }
    source = NS(
        target=deepcopy(target), original_template_sha256="a" * 64, template=NS(bindings=[])
    )
    scenario = CargoScenario(target, {"g1": [{"requiredDescription": "Synthetic rubber"}]}, (), {})
    field = dict(
        key="f0",
        paths=["documentPatch.cargoGroups[0].description"],
        source="NATURAL LATEX",
        constraints=[{"minimumWords": 1}],
    )
    return source, scenario, field


def test_missing_cargo_recipe_is_rejected_before_sampling():
    _, _, field = cargo()
    fragment = {**field, "paths": [], "cargoFragment": {"targetPaths": field["paths"]}}
    with pytest.raises(ValueError, match="lacks a reviewed lexical recipe"):
        host.require_cargo_recipes([fragment], None)
    contract = host.CargoLexicalContract(
        source_template_sha256="a" * 64,
        lexical_contract_sha256="b" * 64,
        fragments={
            "f0": host.FragmentRecipe(role="item", source=field["source"], identity_indices=(0,))
        },
    )
    host.require_cargo_recipes([fragment], contract)
    with pytest.raises(ValueError, match="recipe source changed"):
        host.require_cargo_recipes([{**fragment, "source": "OTHER"}], contract)
    with pytest.raises(ValueError, match="fixed commodity literal"):
        host.require_cargo_recipes([{**field, "hostAssembly": {}}], None)
    # Simple descriptions and unrelated linguistic fields still need no recipe.
    host.require_cargo_recipes([field], None)
    host.require_cargo_recipes(
        [{**field, "paths": ["documentPatch.parties.shipper.name"], "hostAssembly": {}}], None
    )


def test_registry_owns_commodity_no_source_contamination_or_model_override():
    source, scenario, field = cargo()
    result = host.prepare(
        source, [field], scenario=scenario, projection=None, contract=None, sample_id="s", seed=42
    )
    assert result.values["f0"].startswith("Synthetic rubber;")
    assert "LATEX" not in result.values["f0"].upper()
    assert result.merge({"name": "New Trading"})["name"] == "New Trading"
    with pytest.raises(ValueError, match="overwrite"):
        result.merge({"f0": "Synthetic rubber and natural latex"})
    assert source.target["documentPatch"]["cargoGroups"][0]["description"] == "NATURAL LATEX"


@pytest.mark.parametrize("surface", ["680DTEX", "ON SPOOL (F1635)", 'AA" GRADE)', "NATURAL LATEX"])
def test_model_approved_literal_cannot_retain_product_specific_facts(surface, monkeypatch):
    source, scenario, field = cargo()
    field.update(
        source=surface,
        paths=[],
        cargoFragment={
            "targetPaths": ["documentPatch.cargoGroups[0].description"],
            "contexts": [{"sourceSlot": surface, "after": ""}],
        },
    )
    monkeypatch.setattr(host.targets, "lexical_contract", lambda _: (field,))
    contract = host.CargoLexicalContract(
        source_template_sha256=source.original_template_sha256,
        lexical_contract_sha256=sha256_bytes(canonical_json_bytes((field,))),
        fragments={"f0": host.FragmentRecipe(role="literal", source=surface)},
    )
    with pytest.raises(ValueError, match="retained cargo literal"):
        host.require_cargo_recipes([field], contract)
    with pytest.raises(ValueError, match="retained cargo literal"):
        host.prepare(
            source,
            [field],
            scenario=scenario,
            projection=None,
            contract=contract,
            sample_id="s",
            seed=42,
        )


@pytest.mark.parametrize(
    "contexts,valid",
    [
        ([{"sourceSlot": "PART", "after": " NO.:123"}], True),
        (
            [
                {"sourceSlot": "PART", "after": " NO.:123"},
                {"sourceSlot": "PART", "after": " NO.:456"},
            ],
            True,
        ),
        ([], False),
        ([{"sourceSlot": "PART", "after": " OF A MACHINE"}], False),
        (
            [
                {"sourceSlot": "PART", "after": " NO.:123"},
                {"sourceSlot": "PART", "after": " OF A MACHINE"},
            ],
            False,
        ),
    ],
)
def test_literal_caption_requires_its_own_frame_at_every_occurrence(contexts, valid):
    _, _, field = cargo()
    field.update(
        source="PART",
        paths=[],
        cargoFragment={
            "targetPaths": ["documentPatch.cargoGroups[0].description"],
            "contexts": contexts,
        },
    )
    contract = host.CargoLexicalContract(
        source_template_sha256="a" * 64,
        lexical_contract_sha256="b" * 64,
        fragments={"f0": host.FragmentRecipe(role="literal", source="PART")},
    )
    if valid:
        host.require_cargo_recipes([field], contract)
    else:
        with pytest.raises(ValueError, match="retained cargo literal"):
            host.require_cargo_recipes([field], contract)


def test_compiled_origin_mark_cannot_be_replaced_with_shipping_country():
    source, scenario, _ = cargo()
    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    origin = "documentPatch.cargoGroups[0].origin.name"
    for target in (source.target, scenario.target):
        target["documentPatch"]["cargoGroups"][0].update(
            marksAndNumbers=["Made in Taiwan"], origin={"name": "Taiwan"}
        )
    source.template.bindings = [NS(target_paths=(path, origin), derivation=None)]
    field = dict(key="mark", paths=[path], source="Made in Taiwan", constraints=[])
    plan = host.prepare(
        source, [field], scenario=scenario, projection=None, contract=None, sample_id="s", seed=42
    )
    assert plan.values == {"mark": "Made in Taiwan"}
    assert plan.evidence["mark"]["kind"] == "compiled_manufacturing_origin"
    with pytest.raises(ValueError, match="overwrite"):
        plan.merge({"mark": "Made in Mexico"})
    scenario.target["documentPatch"]["cargoGroups"][0]["origin"]["name"] = "Mexico"
    plan = host.prepare(
        source, [field], scenario=scenario, projection=None, contract=None, sample_id="s", seed=42
    )
    assert plan.values == {"mark": "Made in Mexico"}


@pytest.mark.parametrize(
    "role,original",
    [
        ("source_shaped_code", "Y25T1083"),
        ("source_shaped_code", "3104635208"),
        ("shipment_mark", "SSAM-SEEG-VD(IN DIA)"),
    ],
)
def test_reviewed_reference_is_host_owned_with_exact_pinned_role(monkeypatch, role, original):
    source, scenario, _ = cargo()
    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    field = dict(key="mark", paths=[path], source=original, constraints=[])
    monkeypatch.setattr(host.targets, "lexical_contract", lambda _: (field,))
    contract = host.CargoLexicalContract(
        source_template_sha256=source.original_template_sha256,
        lexical_contract_sha256=sha256_bytes(canonical_json_bytes((field,))),
        fragments={},
        references={"mark": host.ReferenceRecipe(role=role, source=original)},
    )
    args = dict(scenario=scenario, projection=None, contract=contract, sample_id="s", seed=42)
    plan = host.prepare(source, [field], **args)
    value = plan.values["mark"]
    assert value != original
    assert host.prepare(source, [field], **args).values == plan.values
    if role == "source_shaped_code":
        from document_ocr.synthesis.generators import surface_pattern

        assert surface_pattern(value) == surface_pattern(original)
    else:
        assert value.startswith("MARK-")
    with pytest.raises(ValueError, match="overwrite"):
        plan.merge({"mark": "Some unrelated cargo description"})
    with pytest.raises(ValueError, match="absent field"):
        host.prepare(source, [], **args)
    with pytest.raises(ValueError, match="ownership"):
        host.prepare(source, [{**field, "source": "changed"}], **args)


def test_reviewed_continuation_list_is_generated_before_linguistic_requests(monkeypatch):
    source, scenario, _ = cargo()
    original = "3903-001118,3903-001130,"
    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    field = dict(key="mark", paths=[path], source=original, constraints=[])
    source.target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [original]
    scenario.target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [original]
    source.template.bindings = [
        NS(target_paths=[path], realization=NS(mode="single_surface"), derivation=None)
    ]
    monkeypatch.setattr(host.targets, "lexical_contract", lambda _: (field,))
    contract = host.CargoLexicalContract(
        source_template_sha256=source.original_template_sha256,
        lexical_contract_sha256=sha256_bytes(canonical_json_bytes((field,))),
        fragments={},
        references={"mark": host.ReferenceRecipe(role="source_shaped_code_list", source=original)},
    )
    plan = host.prepare(
        source,
        [field],
        scenario=scenario,
        projection=None,
        contract=contract,
        sample_id="s",
        seed=42,
    )
    assert plan.values["mark"] != original
    assert len(plan.values["mark"]) == len(original)
    assert plan.values["mark"].endswith(",")
    assert plan.evidence["mark"]["role"] == "source_shaped_code_list"
    with pytest.raises(ValueError, match="overwrite"):
        plan.merge({"mark": "copied source"})


def test_item_fragments_need_a_source_pinned_recipe(monkeypatch):
    source, scenario, field = cargo()
    field.update(
        paths=[],
        auxiliaryKey="part:0",
        cargoFragment=dict(
            targetPaths=["documentPatch.cargoGroups[0].description"],
            index=0,
            sourceParts=["NATURAL LATEX", "MODEL R200"],
        ),
    )
    args = dict(scenario=scenario, projection=None, sample_id="s", seed=42)
    with pytest.raises(ValueError, match="reviewed lexical recipe"):
        host.prepare(source, [field], contract=None, **args)
    monkeypatch.setattr(host.targets, "lexical_contract", lambda _: (field,))
    contract = host.CargoLexicalContract(
        source_template_sha256=source.original_template_sha256,
        lexical_contract_sha256=sha256_bytes(canonical_json_bytes((field,))),
        fragments={
            "f0": host.FragmentRecipe(role="item", source="NATURAL LATEX", identity_indices=(0,))
        },
    )
    assert (
        "Synthetic rubber" in host.prepare(source, [field], contract=contract, **args).values["f0"]
    )
    stale = contract.model_copy(update={"source_template_sha256": "b" * 64})
    with pytest.raises(ValueError, match="pinned source"):
        host.prepare(source, [field], contract=stale, **args)


def test_reviewed_detail_supplies_four_real_words_without_repeating_source(monkeypatch):
    source, scenario, field = cargo()
    field.update(
        paths=[],
        constraints=[{"minimumWords": 4}],
        auxiliaryKey="part:0",
        cargoFragment=dict(targetPaths=["documentPatch.cargoGroups[0].description"]),
    )
    monkeypatch.setattr(host.targets, "lexical_contract", lambda _: (field,))
    contract = host.CargoLexicalContract(
        source_template_sha256=source.original_template_sha256,
        lexical_contract_sha256=sha256_bytes(canonical_json_bytes((field,))),
        fragments={"f0": host.FragmentRecipe(role="detail", source=field["source"])},
    )
    plan = host.prepare(
        source,
        [field],
        scenario=scenario,
        projection=None,
        contract=contract,
        sample_id="s",
        seed=42,
    )
    assert plan.values["f0"].startswith("Commercial product reference ")
    assert len(plan.values["f0"].split()) == 4
    assert field["source"] not in plan.values["f0"]


@pytest.mark.parametrize("country", ["HN", "EG", "NL", "CN", "US"])
def test_structured_phone_uses_sampled_country_without_a_provider(country):
    field = dict(
        key="phone",
        paths=["documentPatch.parties.consignee.contactDetails.phoneNumbers[0]"],
        source="old",
        constraints=[],
    )
    projection = NS(party_geography={"documentPatch.parties.consignee": NS(country_code=country)})
    result = host.prepare(
        NS(template=NS(bindings=[])),
        [field],
        scenario=None,
        projection=projection,
        contract=None,
        sample_id="s",
        seed=42,
    )
    validate_phone(result.values["phone"], country_code=country)


def test_shared_phone_cannot_cross_country_boundaries():
    field = dict(
        key="phone",
        paths=[
            f"documentPatch.parties.{role}.contactDetails.phoneNumbers[0]"
            for role in ("shipper", "consignee")
        ],
        source="old",
        constraints=[],
    )
    projection = NS(
        party_geography={
            f"documentPatch.parties.{role}": NS(country_code=code)
            for role, code in [("shipper", "US"), ("consignee", "CN")]
        }
    )
    with pytest.raises(ValueError, match="inconsistent sampled countries"):
        host.prepare(
            NS(template=NS(bindings=[])),
            [field],
            scenario=None,
            projection=projection,
            contract=None,
            sample_id="s",
            seed=42,
        )


def test_empty_linguistic_request_has_zero_calls_and_zero_cost(tmp_path):
    result = asyncio.run(
        pipeline._cached_fields(
            cache=tmp_path,
            model=None,
            provider=NS(model_dump=lambda **_: {}),
            system_prompt="unused",
            payload={},
            output_type=pipeline._generation_schema([], []),
            limiter=asyncio.Semaphore(1),
            authorized=False,
        )
    )
    assert result["output"] == {}
    assert result["hostOnly"] is True
    assert result["usage"]["requests"] == 0
    assert float(result["usage"]["estimatedCostUsd"]) == 0
    assert not list(tmp_path.iterdir())


def test_shipping_range_mark_is_fully_host_owned_not_a_model_arithmetic_request():
    field = dict(
        key="mark",
        paths=["documentPatch.cargoGroups[0].marksAndNumbers[0]"],
        source="OLD C/NO. 1-20",
        constraints=[],
        hostRangeIdentity=True,
        hostAssembly=dict(prefix="", suffix=" C/NO. 1-13", mutableSource="OLD", minimumWords=1),
    )
    source = NS(template=NS(bindings=[]))
    plan = host.prepare(
        source, [field], scenario=None, projection=None, contract=None, sample_id="s", seed=42
    )
    assert plan.values["mark"].startswith("MARK-")
    assert plan.evidence["mark"]["kind"] == "host_shipping_mark_and_range"
    with pytest.raises(ValueError, match="overwrite"):
        plan.merge(dict(mark="NEW C/NO. 1-20"))


def test_repeated_numeric_export_reference_is_host_owned_with_exact_source_shape():
    path = "documentPatch.forwardingAndExportReferences[0]"
    source = NS(
        template=NS(
            bindings=[
                NS(
                    logical_key="owner",
                    target_paths=[path],
                    derivation=None,
                    value_kind="other_text",
                ),
                NS(
                    logical_key="copy",
                    target_paths=[],
                    derivation="same_as_binding",
                    value_kind="identifier",
                    dependency_bindings=["owner"],
                ),
            ]
        )
    )
    field = dict(key="ref", paths=[path], source="31106024020018", constraints=[])
    args = dict(scenario=None, projection=None, contract=None, sample_id="s", seed=42)
    value = host.prepare(source, [field], **args).values["ref"]
    assert value.isdigit() and len(value) == 14 and value != field["source"]
    assert host.prepare(source, [field], **args).values["ref"] == value
    source.template.bindings.pop()
    assert not host.prepare(source, [field], **args).values
