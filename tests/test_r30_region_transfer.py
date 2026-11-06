"""Existing measured owners and unconstrained text lookup are not equivalent."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/migrations"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "r30_transfer", SCRIPTS / "r30_region_transfer_inventory.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pages(text):
    return [{"index": 0, "regions": [{"index": 0, "text": text}]}]


def test_measured_joined_subfields_remain_proposals():
    r = module.locate("14", pages("14X20'GP CONTAINERS"), [(0, 0)])
    assert r["status"] == "exact_measured_owner_candidate"
    assert r["spans"][0]["expected"] == "14"
    r = module.locate("20'GP", pages("14X20'GP CONTAINERS"), [(0, 0)])
    assert r["spans"][0]["start"] == 3


def test_global_identifier_substring_is_not_accepted():
    r = module.locate("14", pages("AB14000"), [])
    assert r["status"] == "unmatched_or_ambiguous_review"
    assert r["matches"] == []


def test_ambiguous_measured_occurrence_is_not_selected():
    r = module.locate("14", pages("14X20GP 14X40HC"), [(0, 0)])
    assert r["status"] == "unmatched_or_ambiguous_review"
