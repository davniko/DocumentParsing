import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def select(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts" / "migrations"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "r30_wording_selection", directory / "propose_r30_new_wording.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.select_shipments


def test_selection_preserves_order_and_default(select):
    transfers = [{"descendants": [{"id": "old"}, {"id": "fresh1"}, {"id": "fresh2"}]}]
    assert [row["id"] for _, row in select(transfers, None)] == ["old", "fresh1", "fresh2"]
    assert [row["id"] for _, row in select(transfers, ["fresh2", "fresh1"])] == ["fresh1", "fresh2"]


@pytest.mark.parametrize("wanted", [[], ["x", "x"], ["missing"]])
def test_bad_selection_rejected(select, wanted):
    with pytest.raises(ValueError):
        select([{"descendants": [{"id": "x"}]}], wanted)


def test_duplicate_transfer_identity_rejected(select):
    with pytest.raises(ValueError, match="duplicate shipment"):
        select([{"descendants": [{"id": "x"}, {"id": "x"}]}], None)
