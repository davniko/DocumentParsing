from dataclasses import replace

import pytest
from test_curated_templates import compile_sampling_blueprint, source_fixture

from document_ocr.synthesis.curated import Variable
from document_ocr.synthesis.curated_identifiers import (
    declared_identifier_values,
    generate_fictional_vin,
    vin_check_digit,
)
from document_ocr.synthesis.generators import DeterministicStream


def test_vin_checksum_uses_published_cfr_example_and_rejects_bad_characters():
    # 49 CFR 565.15 Table VI: sum 411, remainder 4.
    assert vin_check_digit("1G4AH59H45G118341") == "4"
    for malformed in ("SALKABB95SA26156", "SALKABB95SA2615630", "SALKABB95SA26156Q"):
        with pytest.raises(ValueError, match="17 uppercase"):
            vin_check_digit(malformed)


def test_fictional_vin_is_deterministic_distinct_and_valid_across_draws():
    source = "SALKABB95SA261563"
    for seed in range(100):
        stream = DeterministicStream(seed, "test", "vehicle")
        value = generate_fictional_vin(source, stream)
        assert value == generate_fictional_vin(source, stream)
        assert len(value) == 17 and value != source
        assert not set(value) & set("IOQ")
        assert value[8] == vin_check_digit(value)
        assert value[9] not in "UZ0"
        assert all(
            a.isdigit() == b.isdigit()
            for i, (a, b) in enumerate(zip(source, value, strict=True))
            if i != 8
        )


def test_identifier_policy_requires_explicit_owned_variable_and_records_limits():
    row, contract, history = source_fixture()
    blueprint = compile_sampling_blueprint(row, history, contract)
    source = "SALKABB95SA261563"
    variable = Variable(
        key="vin",
        kind="identifier",
        value=source,
        meaning="Fictional vehicle VIN",
        required_literals=[],
        occurrences=[{"text": source, "occurrence": 1, "presentation": "text"}],
    )
    blueprint = replace(
        blueprint,
        contract=blueprint.contract.model_copy(
            update={"variables": [*contract.variables, variable]}
        ),
        ownership_data={"identifier_policies": {"vin": "fictional_vin_v1"}},
    )
    values, receipt = declared_identifier_values(
        blueprint, DeterministicStream(3, "test", "sample")
    )
    assert set(values) == {"vin"} and receipt[0]["after"] == values["vin"]
    assert receipt[0]["validCheckDigit"] and not receipt[0]["manufacturerIssuedIdentity"]
    with pytest.raises(ValueError, match="unknown declared identifier"):
        declared_identifier_values(
            replace(blueprint, ownership_data={"identifier_policies": {"vin": "guess"}}),
            DeterministicStream(3, "test", "sample"),
        )
    with pytest.raises(ValueError, match="existing identifier variable"):
        declared_identifier_values(
            replace(
                blueprint, ownership_data={"identifier_policies": {"missing": "fictional_vin_v1"}}
            ),
            DeterministicStream(3, "test", "sample"),
        )
