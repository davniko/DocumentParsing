from pathlib import Path

import pytest

from document_ocr.synthesis.country_presentations import (
    DEFAULT_MANIFEST_SHA256,
    DEFAULT_REGISTRY_MANIFEST,
)
from document_ocr.synthesis.paddle_render import project_owned_country_names

ROOT = Path(__file__).resolve().parents[1]
ACTION = {
    "target": "/parties/shipper/country",
    "key": "exporter_country",
    "manifestPath": str(DEFAULT_REGISTRY_MANIFEST),
    "sha256": DEFAULT_MANIFEST_SHA256,
}


@pytest.mark.parametrize("literal,expected", [
    ("VENEZUELA", "VENEZUELA"),
    ("A.E.", "UNITED ARAB EMIRATES"),
    ("A.R.E.", "EGYPT"),
    ("RUSSIAN FEDERATION", "RUSSIA"),
])
def test_country_is_read_from_its_owned_rendered_value(literal, expected):
    target = {"parties": {"shipper": {"name": "RETAINED NAME"}}}
    receipts = [{"owner": "exporter_country@20:30", "after": literal}]
    project_owned_country_names(target, receipts, [ACTION], ROOT)
    assert target["parties"]["shipper"] == {"name": "RETAINED NAME", "country": expected}
    project_owned_country_names(target, receipts, [ACTION], ROOT)


@pytest.mark.parametrize("receipts", [
    [],
    [{"owner": "exporter_country@20:30", "after": "not a country"}],
    [{"owner": "exporter_country@20:30", "after": "SG"},
     {"owner": "exporter_country@40:50", "after": "VE"}],
])
def test_missing_unknown_and_conflicting_occurrences_fail(receipts):
    with pytest.raises(ValueError):
        project_owned_country_names({}, receipts, [ACTION], ROOT)


def test_existing_conflicting_target_is_not_silently_overwritten():
    target = {"parties": {"shipper": {"country": "SINGAPORE"}}}
    with pytest.raises(ValueError, match="conflicts"):
        project_owned_country_names(
            target, [{"owner": "exporter_country@20:30", "after": "VE"}], [ACTION], ROOT
        )
    assert target["parties"]["shipper"]["country"] == "SINGAPORE"


def test_unrecognized_rule_options_fail():
    with pytest.raises(ValueError, match="unsupported fields"):
        project_owned_country_names({}, [], [{**ACTION, "fallback": "SG"}], ROOT)
