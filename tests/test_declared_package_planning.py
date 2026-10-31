"""Uncontainerized planning budgets never masquerade as equipment capacity."""

from copy import deepcopy
from decimal import Decimal as D

import pytest

from document_ocr.synthesis.declared_package_planning import (
    component_minima,
    composition_capacities,
)


def test_independent_groups_expand_only_explicit_uncontainerized_budget():
    draw = {
        "equipment": [],
        "provenance": {"rowCapacities": [{"payloadKg": "16000", "volumeM3": "24"}]},
    }
    before = deepcopy(draw)
    capacities, rows = composition_capacities(draw, {"uncontainerizedBudget": "per_group"}, 3)
    assert capacities == ((D(48000), D(72)),)
    assert rows == [{"payloadKg": "48000", "volumeM3": "72"}]
    assert draw == before
    assert composition_capacities(draw, {}, 3)[0] == ((D(16000), D(24)),)


def test_actual_container_capacity_is_never_enlarged_for_more_groups():
    draw = {
        "equipment": [{"sizeCategory": "TWENTY_FOOT"}],
        "provenance": {"rowCapacities": [{"payloadKg": "28000", "volumeM3": "33"}]},
    }
    assert composition_capacities(draw, {}, 3)[0] == ((D(28000), D(33)),)
    with pytest.raises(ValueError, match="uncontainerized"):
        composition_capacities(draw, {"uncontainerizedBudget": "per_group"}, 3)


def test_component_minima_keep_global_and_container_scopes_separate():
    certificate = {
        "regions": [
            {"role": "component_package", "componentCount": 5},
            {
                "role": "component_measurement",
                "componentCount": 3,
                "scope": {"kind": "container", "index": 1},
            },
            {"role": "private_measurement"},
        ]
    }
    assert component_minima(certificate, 3) == (5, [1, 3, 1])
    assert component_minima({"regions": []}, 1) == (1, [1])
