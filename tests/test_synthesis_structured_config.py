from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.config import (
    SynthesisStructuredBaselineConfig,
    load_synthesis_structured_baseline_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/synthesis/mpci_bl_combined1157_structured_baseline50.yaml"


def _config_value() -> dict[str, object]:
    return load_synthesis_structured_baseline_config(CONFIG).model_dump(mode="json")


def test_structured_baseline_config_pins_complete_pass_one_and_two_contract() -> None:
    config = load_synthesis_structured_baseline_config(CONFIG)

    assert config.selection.requested_documents == 50
    assert config.selection.require_template_wholly_in_split is True
    assert config.selection.require_route_synthesis_support is True
    assert config.selection.maximum_per_carrier == 7
    assert config.selection.strata_exact["document_type"] == {
        "bill_of_lading": 35,
        "sea_waybill": 15,
    }
    assert config.selection.context_minimums["dangerous_goods"] == 2
    assert config.modeling.candidates == (
        "empirical",
        "gaussian_copula_default",
        "gaussian_copula_gaussian_kde",
        "gaussian_copula_domain_mixed",
        "gaussian_copula_physical_factors",
        "ctgan",
        "tvae",
    )
    assert "empirical" not in config.modeling.selectable_candidates
    assert config.modeling.grouped_folds == 5
    assert len(config.modeling.seeds) == 5
    assert config.modeling.neural_epochs == 300
    assert config.modeling.contextual_plausibility.exact_identity_role.minimum_rows == 3
    assert config.modeling.contextual_plausibility.semantic_family_role.minimum_templates == 5
    assert config.generation.transport_capacity.policy == (
        "source_type_aware_maersk_upper_bounds_v1"
    )
    assert config.generation.transport_capacity.twenty_standard_payload_kg == 28300.0
    assert config.generation.transport_capacity.published_reference_margin_fraction == 0.05
    assert config.generation.publish_training_records is False


def test_structured_config_rejects_empirical_as_production_model() -> None:
    value = _config_value()
    modeling = value["modeling"]
    assert isinstance(modeling, dict)
    modeling["selectable_candidates"] = ["empirical", "gaussian_copula_default"]

    with pytest.raises(ValidationError, match="not a production proposal model"):
        SynthesisStructuredBaselineConfig.model_validate(value, strict=True)


def test_structured_config_rejects_unreceipted_method_or_partial_context_counts() -> None:
    value = _config_value()
    generation = value["generation"]
    assert isinstance(generation, dict)
    generation["cargo_measure_method"] = "scale_heterogeneous_package_sum"

    with pytest.raises(ValidationError, match="cargo_measure_method"):
        SynthesisStructuredBaselineConfig.model_validate(value, strict=True)


def test_structured_config_rejects_route_support_below_grouped_folds() -> None:
    value = _config_value()
    modeling = value["modeling"]
    assert isinstance(modeling, dict)
    routing = modeling["routing"]
    assert isinstance(routing, dict)
    role_profile = routing["role_profile"]
    assert isinstance(role_profile, dict)
    role_profile["minimum_templates"] = 4

    with pytest.raises(ValidationError, match="at least grouped_folds"):
        SynthesisStructuredBaselineConfig.model_validate(value, strict=True)


def test_structured_config_rejects_nonpositive_transport_capacity() -> None:
    value = _config_value()
    generation = value["generation"]
    assert isinstance(generation, dict)
    transport = generation["transport_capacity"]
    assert isinstance(transport, dict)
    transport["twenty_standard_payload_kg"] = 0.0

    with pytest.raises(ValidationError, match="twenty_standard_payload_kg"):
        SynthesisStructuredBaselineConfig.model_validate(value, strict=True)
