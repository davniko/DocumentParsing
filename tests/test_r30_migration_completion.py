import copy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/migrations"
sys.path.insert(0, str(SCRIPTS))
from finalize_r30_synthetic_migration import verify_completion  # noqa: E402


def case():
    manifest = {
        "syntheticMigration": {"required": 2, "certified": 2},
        "templateCatalog": "current",
        "templateControls": "controls",
    }
    audit = {
        "counts": {"real-train": 600, "validation": 60, "synthetic-train": 2},
        "syntheticUncertified": 0,
        "allPublishedViewsAndSchemaPassed": True,
        "catalog": "current",
        "controls": "controls",
        "manifest": "published",
    }
    catalog = {"templates": {"a": "a2", "b": "b1"}}
    controls = [
        (
            {"sources": ["a", "b"], "passed": 6, "failures": []},
            {"templates": {"a": "a1", "b": "b1"}},
        ),
        ({"sources": ["a"], "passed": 3, "failures": []}, catalog),
    ]
    replay = {"catalog": "current", "manifest": "published", "records": 2, "failures": []}
    return manifest, audit, catalog, controls, {"batches": [["one", "two"]]}, replay


def test_full_control_replay_plus_exact_changed_template_controls():
    assert verify_completion(*case()) == {"one", "two"}


@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "stale",
        "failed",
        "uncovered",
        "duplicate",
        "views",
        "replay_stale",
        "replay_partial",
        "replay_failed",
        "replay_manifest",
    ],
)
def test_cannot_finalize_unverified_or_incomplete_publication(failure):
    manifest, audit, catalog, controls, plan, replay = copy.deepcopy(case())
    if failure == "partial":
        manifest["syntheticMigration"]["certified"] = 1
    if failure == "stale":
        audit["catalog"] = "old"
    if failure == "failed":
        controls[1][0]["failures"] = ["held"]
    if failure == "uncovered":
        controls.pop()
    if failure == "duplicate":
        plan["batches"] = [["one", "one"]]
    if failure == "views":
        audit["allPublishedViewsAndSchemaPassed"] = False
    if failure == "replay_stale":
        replay["catalog"] = "old"
    if failure == "replay_partial":
        replay["records"] = 1
    if failure == "replay_failed":
        replay["failures"] = ["changed target"]
    if failure == "replay_manifest":
        replay["manifest"] = "old"
    with pytest.raises(ValueError):
        verify_completion(manifest, audit, catalog, controls, plan, replay)
