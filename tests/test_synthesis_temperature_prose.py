from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import lexical_facts as host
from document_ocr.synthesis.template_compiler import temperature_prose as prose
from document_ocr.synthesis.template_compiler.cargo_scenarios import CargoScenario


@pytest.mark.parametrize(
    "text",
    [
        "SET TEMP: -18 C",
        "Cargo at requested carrying temperature of -20 degrees Celsius",
        "Reefer Settings Cont. Nos.: 1°C--2°C",
    ],
)
def test_source_only_carrying_setting_cannot_silently_be_fixed(text):
    with pytest.raises(ValueError, match="lacks physical ownership"):
        prose.require_source_setting_owners(text.encode(), NS(bindings=()), {"documentPatch": {}})


def test_explicit_summary_dependency_owns_utf8_source_number():
    text = "Température\nSET TEMP: -18 C".encode()
    start = text.index(b"-18")
    owner = NS(
        target_paths=(),
        dependency_paths=("documentPatch.containers[0].temperatureSetpoint.value",),
        occurrences=(NS(byte_start=start, byte_end=start + 3),),
    )
    prose.require_source_setting_owners(text, NS(bindings=(owner,)), {"documentPatch": {}})


def test_gate_measurement_and_flashpoint_are_not_carrying_instructions():
    prose.require_source_setting_owners(
        b"GATE IN TEMPERATURE: -6 C; FLASH POINT: 18 C", NS(bindings=()), {"documentPatch": {}}
    )


def test_source_screen_accepts_existing_proved_cargo_instruction_owner():
    text = "Cargo carrying temperature of -20 degrees Celsius."
    path = "documentPatch.cargoGroups[0].handlingInstructions[0]"
    base = "documentPatch.containers[0].temperatureSetpoint"
    instruction = NS(
        target_paths=(path,),
        dependency_paths=(),
        derivation=None,
        occurrences=(NS(byte_start=0, byte_end=len(text.encode())),),
    )
    setpoint = NS(
        target_paths=(base + ".value", base + ".unit"),
        dependency_paths=(),
        derivation=None,
        occurrences=(),
    )
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "handlingInstructions": [text]}],
            "containers": [{"temperatureSetpoint": {"value": -20.0, "unit": "celsius"}}],
        }
    }
    prose.require_source_setting_owners(text.encode(), NS(bindings=(instruction, setpoint)), target)


@pytest.mark.parametrize(
    "surface,new_value,expected",
    [
        ("TEMPERATURE IS 4 DEGREES CENTIGRADE", 5, "TEMPERATURE IS 5 DEGREES CENTIGRADE"),
        ("TEMPERATURE TO BE SET AT +21.0 C", 5, "TEMPERATURE TO BE SET AT +5.0 C"),
    ],
)
def test_explicit_cargo_temperature_is_one_deterministic_text_and_setpoint_surface(
    surface, new_value, expected
):
    paths = (
        "documentPatch.cargoGroups[0].handlingInstructions[0]",
        "documentPatch.containers[0].temperatureSetpoint.value",
        "documentPatch.containers[0].temperatureSetpoint.unit",
    )
    old_value = 4 if "CENTIGRADE" in surface else 21
    original = {paths[0]: surface, paths[1]: old_value, paths[2]: "celsius"}
    changed = {paths[0]: expected, paths[1]: new_value, paths[2]: "celsius"}
    assert prose.composite_instruction_contract(paths, surface, original) == paths
    assert prose.render_composite_instruction(surface, paths, original, changed) == expected
    changed[paths[0]] = surface
    with pytest.raises(ValueError, match="contradicts"):
        prose.render_composite_instruction(surface, paths, original, changed)


def test_composite_temperature_is_prepared_before_linguistic_generation():
    text = "TEMPERATURE IS 4 DEGREES CENTIGRADE"
    path = "documentPatch.cargoGroups[0].handlingInstructions[0]"
    paths = (
        path,
        "documentPatch.containers[0].temperatureSetpoint.value",
        "documentPatch.containers[0].temperatureSetpoint.unit",
    )
    template = NS(bindings=(NS(target_paths=paths, derivation=None),))
    old = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "handlingInstructions": [text]}],
            "containers": [{"temperatureSetpoint": {"value": 4, "unit": "celsius"}}],
        }
    }
    new = deepcopy(old)
    new["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"] = 5
    field = dict(key="instruction", paths=[path], source=text, constraints=[])
    plan = host.prepare(
        NS(target=old, template=template),
        [field],
        scenario=CargoScenario(new, {}, (), {}),
        projection=None,
        contract=None,
        sample_id="s",
        seed=1,
    )
    assert plan.values["instruction"] == "TEMPERATURE IS 5 DEGREES CENTIGRADE"
    assert plan.evidence["instruction"]["kind"] == "compiled_temperature_dependency"


def test_single_printed_reefer_setting_must_not_be_missing_from_source_labels():
    source = {
        "documentPatch": {
            "containers": [{"typeDescription": "20RF"}],
            "cargoGroups": [{"handlingInstructions": ["TEMPERATURE IS 4 DEGREES CENTIGRADE"]}],
        }
    }
    with pytest.raises(ValueError, match="without a temperatureSetpoint label"):
        prose.require_singleton_printed_setpoint(source)
    source["documentPatch"]["containers"][0]["temperatureSetpoint"] = {
        "unit": "celsius",
        "value": 4,
    }
    prose.require_singleton_printed_setpoint(source)


def case(text="Cargo carrying temperature of -20 degrees Celsius."):
    path = "documentPatch.cargoGroups[0].handlingInstructions[0]"
    base = "documentPatch.containers[0].temperatureSetpoint"
    binding = NS(target_paths=(path, base + ".value", base + ".unit"), derivation=None)
    template = NS(bindings=(binding,))
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "handlingInstructions": [text]}],
            "containers": [{"temperatureSetpoint": {"value": -20.0, "unit": "celsius"}}],
        }
    }
    new = deepcopy(target)
    new["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"] = -1.5
    return template, target, new, path


@pytest.mark.parametrize(
    "text", ["Maintain -20 degrees Celsius.", "SET AT -20 C", "-20°C, maintain -20 Celsius."]
)
def test_temperature_instruction_uses_compiled_numeric_owner(text):
    template, old, new, path = case(text)
    expected = text.replace("-20", "-1.5")
    assert prose.generate(template, old, new) == {path: expected}
    with pytest.raises(ValueError, match="contradicts"):
        prose.validate(template, old, new)
    new["documentPatch"]["cargoGroups"][0]["handlingInstructions"][0] = expected
    prose.validate(template, old, new)


@pytest.mark.parametrize(
    "text", ["Set at -19 Celsius.", "Set at -20 Fahrenheit.", "Maintain safe conditions."]
)
def test_unproven_temperature_text_is_reviewed_not_guessed(text):
    template, old, new, _ = case(text)
    with pytest.raises(ValueError, match="source disagrees"):
        prose.generate(template, old, new)


def test_lexical_preparation_owns_temperature_before_target_acceptance():
    template, old, new, path = case()
    source = NS(target=old, template=template)
    scenario = CargoScenario(new, {}, (), {})
    field = dict(
        key="instruction",
        paths=[path],
        source=old["documentPatch"]["cargoGroups"][0]["handlingInstructions"][0],
        constraints=[],
    )
    plan = host.prepare(
        source, [field], scenario=scenario, projection=None, contract=None, sample_id="s", seed=1
    )
    assert "-1.5 degrees Celsius" in plan.values["instruction"]
    assert plan.evidence["instruction"]["kind"] == "compiled_temperature_dependency"
    with pytest.raises(ValueError, match="overwrite"):
        plan.merge({"instruction": field["source"]})


def test_missing_unit_owner_and_changed_unit_are_explicit_errors():
    template, old, new, _ = case()
    new["documentPatch"]["containers"][0]["temperatureSetpoint"]["unit"] = "fahrenheit"
    with pytest.raises(ValueError, match="source unit"):
        prose.generate(template, old, new)
    template.bindings[0].target_paths = template.bindings[0].target_paths[:-1]
    with pytest.raises(ValueError, match="complete compiled"):
        prose.generate(template, old, new)


@pytest.mark.parametrize("text", ["TEMPERATURE: +2 - +8", "Maintain at 4C."])
def test_unowned_printed_temperature_is_rejected_before_goods_sampling(text):
    template, old, _, path = case(text)
    template.bindings[0].target_paths = (path,)
    old["documentPatch"]["containers"] = []
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.require_contract(template, old)


def test_proven_temperature_and_nonnumeric_boilerplate_remain_supported():
    template, old, _, _ = case()
    prose.require_contract(template, old)
    old["documentPatch"]["cargoGroups"][0]["additionalInformation"] = [
        "Shipper responsible for appropriate temperature control.",
        "Temperature controlled cargo; invoice ABC123.",
    ]
    prose.require_contract(template, old)


def test_printed_storage_interval_is_retained_and_constrains_all_owned_setpoints():
    template, old, new, path = case("TEMPERATURE: +2 - +8")
    template.bindings[0].target_paths = (path,)
    old["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"] = 4
    new = deepcopy(old)
    new["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"] = 3
    assert prose.celsius_bounds(template, old) == {0: (2, 8)}
    assert prose.generate(template, old, new) == {path: "TEMPERATURE: +2 - +8"}
    prose.require_contract(template, old)
    prose.validate(template, old, new)
    new["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"] = -2
    with pytest.raises(ValueError, match="contradicts its printed temperature interval"):
        prose.generate(template, old, new)


def test_temperature_interval_never_infers_missing_or_conflicting_source_setpoints():
    template, old, _, path = case("TEMPERATURE: +2 - +8")
    template.bindings[0].target_paths = (path,)
    with pytest.raises(ValueError, match="source setpoint contradicts"):
        prose.intervals(template, old)
    old["documentPatch"]["containers"][0].pop("temperatureSetpoint")
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.intervals(template, old)


@pytest.mark.parametrize(
    "text,setting,bounds",
    [
        ("STOWED IN REEFER CONTAINER AT TEMPERATURE OF PLUS 2C TILL PLUS 2C", 2, (2, 2)),
        ("TEMP MINUS 20 C TO MINUS 18 CELSIUS", -19, (-20, -18)),
        ("TEMPERATURE: MINUS 2,5 C TILL PLUS 1 C", 0, (-2.5, 1)),
    ],
)
def test_worded_temperature_bounds_are_physical_constraints_not_random_prose(text, setting, bounds):
    template, old, _, path = case(text)
    template.bindings[0].target_paths = (path,)
    old["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"] = setting
    assert prose.celsius_bounds(template, old) == {0: bounds}
    assert prose.generate(template, old, old) == {path: text}
    prose.require_contract(template, old)
    new = deepcopy(old)
    new["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"] = bounds[1] + 1
    with pytest.raises(ValueError, match="sampled setpoint contradicts"):
        prose.generate(template, old, new)


@pytest.mark.parametrize(
    "text,error",
    [
        ("TEMPERATURE PLUS 2 C TILL PLUS 2 F", "different units"),
        ("TEMPERATURE PLUS 8 C TILL PLUS 2 C", "source setpoint contradicts"),
    ],
)
def test_worded_temperature_ranges_do_not_hide_contradictory_bounds(text, error):
    template, old, _, path = case(text)
    template.bindings[0].target_paths = (path,)
    with pytest.raises(ValueError, match=error):
        prose.intervals(template, old)


def test_worded_interval_does_not_infer_an_absent_setpoint_or_accept_extra_clauses():
    text = "TEMPERATURE PLUS 2 C TILL PLUS 2 C"
    template, old, _, path = case(text)
    template.bindings[0].target_paths = (path,)
    old["documentPatch"]["containers"][0].pop("temperatureSetpoint")
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.intervals(template, old)
    assert prose._interval_values(text + " UNLESS INSTRUCTED OTHERWISE") is None


def test_source_only_temperature_cannot_be_randomized_as_operational_code():
    binding = NS(
        target_paths=(),
        group_kind="cargo",
        value_kind="operational_text",
        source_text="4C",
        logical_key="handling-code",
    )
    with pytest.raises(ValueError, match="typed physical review"):
        prose.require_temperature_classification([binding])
    binding.value_kind = "temperature"
    prose.require_temperature_classification([binding])
    binding.value_kind = "identifier"
    binding.target_paths = ("documentPatch.cargoGroups[0].marksAndNumbers[0]",)
    prose.require_temperature_classification([binding])


def test_yarn_construction_is_not_fahrenheit_but_a_real_setting_still_is():
    template, old, _, path = case("DTY 75D/72F (83DTEX/72F) SIM SD RW AA GRADE")
    template.bindings[0].target_paths = (path,)
    old["documentPatch"]["containers"] = []
    prose.require_contract(template, old)
    old["documentPatch"]["cargoGroups"][0]["handlingInstructions"][0] += " TEMP 72F"
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.require_contract(template, old)


def test_separate_cargo_instruction_joins_only_complete_printed_setpoints():
    template, old, new, path = case("HUMIDITY 85% TEMP -20C VENT 15")
    template.bindings = (
        NS(target_paths=(path,), derivation=None),
        NS(
            target_paths=(
                "documentPatch.containers[0].temperatureSetpoint.value",
                "documentPatch.containers[0].temperatureSetpoint.unit",
            ),
            derivation=None,
        ),
    )
    assert prose.generate(template, old, new) == {path: "HUMIDITY 85% TEMP -1.5C VENT 15"}
    prose.require_contract(template, old)
    template.bindings = template.bindings[:1]
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.require_contract(template, old)


def test_joint_cargo_instruction_requires_every_container_to_agree():
    template, old, new, path = case("TEMP -20C")
    template.bindings = (
        NS(target_paths=(path,), derivation=None),
        *(
            NS(
                target_paths=(
                    f"documentPatch.containers[{i}].temperatureSetpoint.value",
                    f"documentPatch.containers[{i}].temperatureSetpoint.unit",
                ),
                derivation=None,
            )
            for i in range(2)
        ),
    )
    old["documentPatch"]["containers"].append(deepcopy(old["documentPatch"]["containers"][0]))
    new["documentPatch"]["containers"].append(deepcopy(new["documentPatch"]["containers"][0]))
    assert prose.shared_setpoint_owners(template, old) == {0: (0, 1), 1: (0, 1)}
    assert prose.generate(template, old, new) == {path: "TEMP -1.5C"}
    new["documentPatch"]["containers"][1]["temperatureSetpoint"]["value"] = -3
    with pytest.raises(ValueError, match="one shared"):
        prose.generate(template, old, new)
    old["documentPatch"]["containers"][1].pop("temperatureSetpoint")
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.require_contract(template, old)


def summary_case():
    template, old, new, _ = case("Refrigerated cargo")
    for target in (old, new):
        target["documentPatch"]["containers"].append(
            deepcopy(target["documentPatch"]["containers"][0])
        )
    paths = tuple(f"documentPatch.containers[{i}].temperatureSetpoint.value" for i in range(2))
    template.bindings = (
        NS(target_paths=(), derivation="same_as_binding", dependency_paths=paths),
        NS(target_paths=tuple(p for v in paths for p in (v, v[:-5] + "unit")), derivation=None),
    )
    return template, old, new


def test_source_only_temperature_summary_constrains_joint_sampling_and_validation():
    template, old, new = summary_case()
    assert prose.shared_setpoint_owners(template, old) == {0: (0, 1), 1: (0, 1)}
    prose.validate(template, old, new)
    new["documentPatch"]["containers"][1]["temperatureSetpoint"]["value"] = 3
    with pytest.raises(ValueError, match="summary requires one shared"):
        prose.validate(template, old, new)


def test_summary_does_not_infer_missing_printed_units():
    template, old, _ = summary_case()
    template.bindings[1].target_paths = tuple(
        p for p in template.bindings[1].target_paths if p.endswith(".value")
    )
    with pytest.raises(ValueError, match="complete printed"):
        prose.shared_setpoint_owners(template, old)


def test_summary_rejects_contradictory_sources_and_mixed_dependencies():
    template, old, _ = summary_case()
    old["documentPatch"]["containers"][1]["temperatureSetpoint"]["unit"] = "fahrenheit"
    with pytest.raises(ValueError, match="source owners disagree"):
        prose.shared_setpoint_owners(template, old)
    template.bindings[0].dependency_paths += ("documentPatch.voyageNumber",)
    with pytest.raises(ValueError, match="incompatible dependency"):
        prose.shared_setpoint_owners(template, old)


@pytest.mark.parametrize(
    "caption", ["GOODS ARE SHIPPED AT", "Cargo is shipped at", "CARGO SHIPPED AT"]
)
def test_printed_shipping_temperature_joins_separate_complete_setpoint(caption):
    template, old, new, path = case(f"{caption} -20C")
    template.bindings = (
        NS(target_paths=(path,), derivation=None),
        NS(
            target_paths=(
                "documentPatch.containers[0].temperatureSetpoint.value",
                "documentPatch.containers[0].temperatureSetpoint.unit",
            ),
            derivation=None,
        ),
    )
    prose.require_contract(template, old)
    assert prose.generate(template, old, new) == {path: f"{caption} -1.5C"}
    old["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"] = -19
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.require_contract(template, old)


@pytest.mark.parametrize(
    "text", ["Tested at -20C", "Goods are tested at -20C", "Goods freeze at -20C"]
)
def test_matching_temperature_without_carrying_caption_is_not_a_setpoint(text):
    template, old, _, path = case(text)
    template.bindings = (
        NS(target_paths=(path,), derivation=None),
        NS(
            target_paths=(
                "documentPatch.containers[0].temperatureSetpoint.value",
                "documentPatch.containers[0].temperatureSetpoint.unit",
            ),
            derivation=None,
        ),
    )
    assert prose.cargo_setpoint_contracts(template, old) == ()
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.require_contract(template, old)


@pytest.mark.parametrize("property_name", ["FLASH POINT", "MELTING POINT", "BOILING TEMPERATURE"])
def test_equal_property_temperature_is_not_a_carrying_setpoint(property_name):
    template, old, _, path = case(f"{property_name} TEMP -20C")
    template.bindings = (
        NS(target_paths=(path,), derivation=None),
        NS(
            target_paths=(
                "documentPatch.containers[0].temperatureSetpoint.value",
                "documentPatch.containers[0].temperatureSetpoint.unit",
            ),
            derivation=None,
        ),
    )
    assert prose.cargo_setpoint_contracts(template, old) == ()
    with pytest.raises(ValueError, match="complete physical dependency"):
        prose.require_contract(template, old)
