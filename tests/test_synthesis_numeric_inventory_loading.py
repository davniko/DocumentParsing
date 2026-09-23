from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler import anonymous_equipment, complete_pipeline


@pytest.mark.parametrize("anonymous", [True, False])
def test_numeric_rows_use_sampler_inventory_without_enriching_labels(monkeypatch, anonymous):
    source = SimpleNamespace(target={"documentPatch": {}})
    physical = SimpleNamespace(target={"documentPatch": {"containers": [{"private": True}]}})
    inventory = SimpleNamespace(physical_source=physical) if anonymous else None
    contracts = {"mass": object()}
    monkeypatch.setattr(anonymous_equipment, "compile_inventory", lambda s, c: inventory)
    calls = []
    monkeypatch.setattr(
        complete_pipeline.equipment_row_constraints,
        "compile_rows",
        lambda s, c: calls.append((s, c)),
    )
    complete_pipeline._validate_numeric_row_ownership(source, contracts)
    assert calls == [(physical if anonymous else source, contracts)]
    assert source.target == {"documentPatch": {}}


def test_numeric_inventory_ambiguity_remains_an_explicit_failure(monkeypatch):
    def invalid(s, c):
        raise ValueError("ambiguous equipment owner")

    monkeypatch.setattr(anonymous_equipment, "compile_inventory", invalid)
    with pytest.raises(ValueError, match="ambiguous equipment owner"):
        complete_pipeline._validate_numeric_row_ownership(object(), {})
