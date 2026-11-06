import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/migrations"))
from propose_r30_missing_numeric import required_private_fields


def test_collects_composed_and_override_recipes():
    overlay = {"extras": [{"parts": [{"physical": {"field": "grossKg"}}, {"parts": [{"allocation": {"index": 0}}]}]}], "overrides": [{"valueRecipe": {"physical": {"field": "volume"}}}]}
    assert required_private_fields(overlay) == ({"grossWeight", "volume"}, True)


def test_ignores_targets_and_other_values():
    assert required_private_fields({"targets": [{"physical": {"field": "netKg"}}], "extras": [{"literalText": "allocation"}]}) == (set(), False)
