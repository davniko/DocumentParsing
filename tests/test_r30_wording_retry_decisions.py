import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/migrations"))
spec = importlib.util.spec_from_file_location(
    "r30_wording_retry",
    Path(__file__).parents[1] / "scripts/migrations/regenerate_r30_additional_wording.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "decisions",
    [{}, {"decisions": {"sample": "fix"}}, {"unknown": "fix"}, {"sample": " "}, {"sample": None}],
)
def test_rejects_unmatched_or_unreviewed_retry(decisions):
    with pytest.raises(ValueError):
        module.validate_decisions(decisions, {"sample": {}})


def test_exact_reviewed_subset_is_valid():
    module.validate_decisions(
        {"sample": "Retain the sampled locality."}, {"sample": {}, "other": {}}
    )
