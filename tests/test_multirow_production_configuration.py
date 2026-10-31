"""The promoted reference config must be runnable, not only admission-valid."""

import json
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json"


def read(path):
    return json.loads(path.read_text())


def test_production_config_retains_the_explicit_certified_planning_envelope():
    policy = read(CONFIG)["declared_packages"]
    assert policy["utilization"] == [0.35, 0.75]
    assert policy["group_counts"] == [1, 2, 3]
    assert 0 < policy["utilization"][0] <= policy["utilization"][1] <= 1


@pytest.mark.parametrize("name", ["contracts", "acceptance"])
def test_production_group_authority_pins_match_current_files(name):
    config = read(CONFIG)
    pin = config["declared_packages"][name]
    assert sha256_file(ROOT / pin["path"]) == pin["sha256"]
    assert set(read(ROOT / pin["path"])["sources"]) == set(config["source_ids"])


def test_production_count_schedule_only_uses_accepted_counts_for_every_source():
    config = read(CONFIG)
    policy = config["declared_packages"]
    acceptance = read(ROOT / policy["acceptance"]["path"])
    for sid in config["source_ids"]:
        counts = policy["source_group_counts"].get(sid, policy["group_counts"])
        assert set(counts) <= set(acceptance["sources"][sid]["groupCounts"])
    assert len(policy["source_group_counts"]) == 4


def test_current_manifest_points_to_the_complete_reference_configuration():
    manifest = read(ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/manifest.json")
    assert manifest["currentProductionConfig"] == {
        "path": str(CONFIG.relative_to(ROOT)),
        "sha256": sha256_file(CONFIG),
    }
