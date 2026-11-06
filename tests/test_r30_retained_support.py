import importlib
import sys
from decimal import Decimal as D
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
support_mass = importlib.import_module("apply_r30_retained_support").support_mass


def test_uneven_supports_preserve_complete_packing_budget():
    # 1865 boxes need not divide evenly among 50 supports; no such ratio is printed.
    assert support_mass(1865, 50, D(52), D(59), D(25)) == D(13055)
    assert support_mass(30, 30, D(1380), D(1660), D(25)) == D(8400)


@pytest.mark.parametrize("count", [0, -1, 31, 1.5, True])
def test_invalid_support_counts_fail(count):
    with pytest.raises(ValueError):
        support_mass(30, count, D(1380), D(1660), D(25))


@pytest.mark.parametrize("tare", ["0", "-1", "280", "281", "NaN", "Infinity"])
def test_missing_nonfinite_or_exhausted_packaging_budget_fails(tare):
    with pytest.raises(ValueError):
        support_mass(30, 30, D(1380), D(1660), D(tare))
