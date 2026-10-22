from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_scenarios import ScenarioLocation
from document_ocr.synthesis.curated_templates import render_sampling_blueprint
from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate,
    load_independent_geography_certificates,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_00 = "doc_00a165a2adf7156bdecd557c885de531a2a69a98a920a4e029d491bd5d50b3ed"
SOURCE_49 = "doc_049ad12d94bf5f6636510de5c91f0a70e5842542de33f93d75377a80f5cc3c8a"
AUTHORITY = (
    ROOT
    / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/independent-geography-v4"
    / "geography_certificates.json"
)
AUTHORITY_SHA256 = "5196a830f4cf35408f25a9b2aabe070f87abc3eccf15a424faf6a112ce5f6e05"
SUCCESSOR = AUTHORITY.with_name("successor-receipt.json")
LEGACY_REVIEW = (
    ROOT
    / "docs/analysis/r28-production-integration-20261021/legacy"
    / "source00-independent-geography-review.json"
)
CONFIG = ROOT / "configs/synthesis/mpci_bl_curated_v7_independent_validation.yaml"
TOOL = ROOT / "scripts/synthesis/extend_independent_geography_authority.py"

_SPEC = importlib.util.spec_from_file_location("extend_independent_geography", TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_EXTENSION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXTENSION)


@pytest.fixture(scope="module")
def blueprint():
    config = _EXTENSION.yaml.safe_load(CONFIG.read_bytes())
    return _EXTENSION._compile_blueprint(ROOT, config, SOURCE_00)


@pytest.fixture(scope="module")
def gate():
    certificates = load_independent_geography_certificates(
        AUTHORITY.read_bytes(), expected_sha256=AUTHORITY_SHA256
    )
    assert len(certificates.sources) == 29
    assert SOURCE_00 in certificates.sources
    return IndependentGeographyGate(certificates)


def _scenario():
    location = ScenarioLocation(
        name="ROTTERDAM",
        country="NETHERLANDS",
        country_code="NL",
        registry="unlocode_wpi",
        registry_id="NLRTM",
    )
    return SimpleNamespace(
        party_localities={"documentPatch.parties.consignee": location}
    )


def test_reviewed_successor_matches_current_compiled_blueprint(blueprint, gate):
    receipt = json.loads(SUCCESSOR.read_bytes())
    gate.bind_blueprint(blueprint, blueprint)
    assert receipt["predecessor"]["sha256"] == (
        "c49964c760ba4391e366505c8d169687fe8a211c158dde319b4dd6911c6b1e85"
    )
    assert receipt["changes"]["ownershipSha256"][SOURCE_00]["after"] == digest(
        dict(blueprint.ownership_data)
    )
    assert receipt["semanticProof"]["result"] == "pass"
    assert sha256_file(AUTHORITY) == AUTHORITY_SHA256


def test_legacy_extension_review_rejects_the_changed_compiled_owner():
    with pytest.raises(
        ValueError, match="geography extension differs from the current compiled blueprint"
    ):
        _EXTENSION.extend_authority(ROOT, CONFIG, LEGACY_REVIEW)


def test_current_review_fixture_extends_the_maintained_base_authority(blueprint):
    review = json.loads(LEGACY_REVIEW.read_bytes())
    review["compiledBlueprintPins"]["ownershipSha256"] = digest(
        dict(blueprint.ownership_data)
    )
    with tempfile.TemporaryDirectory(prefix=".geo-current-review-", dir=ROOT) as directory:
        current_review = Path(directory) / "source00-current-review.json"
        current_review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
        authority, receipt = _EXTENSION.extend_authority(ROOT, CONFIG, current_review)
    active = json.loads(AUTHORITY.read_bytes())
    base_pin = review["reviewBasis"]["priorGeographyAuthority"]
    base = json.loads((ROOT / base_pin["path"]).read_bytes())
    assert authority["sources"][SOURCE_00] == active["sources"][SOURCE_00]
    assert {key: value for key, value in authority["sources"].items() if key != SOURCE_00} == (
        base["sources"]
    )
    assert receipt["addedSources"] == [SOURCE_00]
    assert receipt["publicTargetChanges"] == 0


def test_source_only_agency_postal_renders_without_public_target_changes(blueprint, gate):
    gate.bind_blueprint(blueprint, blueprint)
    values = {variable.key: variable.value for variable in blueprint.contract.variables}
    values.update(
        source_only_shipping_agency_name="NORTHSTAR MARINE AGENCY B.V.",
        source_only_shipping_agency_postal="KADE 41, MERIDIAN TERMINAL\n3011 ROTTERDAM",
    )
    assert gate.validate_values(blueprint, values, _scenario()) == []
    rendered, target, proof = render_sampling_blueprint(blueprint, blueprint.target, values)
    candidate = {
        "joinedRawText": rendered,
        "renderValues": values,
        "target": target,
        "proof": proof,
    }
    assert target == blueprint.target
    assert "C C FOR MARITIME SHIPPING AGENCIES" not in rendered
    assert "16 EL GOMHOREYA STREET SHAHER TOWER" not in rendered
    assert "NORTHSTAR MARINE AGENCY B.V." in rendered
    assert "3011 ROTTERDAM" in rendered
    assert gate.validate_rendered(blueprint, candidate, _scenario()) == []


@pytest.mark.parametrize(
    "postal",
    (
        "KADE 41, MERIDIAN TERMINAL",
        "KADE 41, ROTTERDAM TERMINAL\n3011 ROTTERDAM",
        "KADE 41, MERIDIAN TERMINAL\n3011 ROTTERDAM, NETHERLANDS",
    ),
)
def test_source_only_agency_geography_rejects_missing_repeated_or_added_geography(
    blueprint, gate, postal
):
    values = {variable.key: variable.value for variable in blueprint.contract.variables}
    values.update(
        source_only_shipping_agency_name="NORTHSTAR MARINE AGENCY B.V.",
        source_only_shipping_agency_postal=postal,
    )
    assert gate.validate_values(blueprint, values, _scenario())


def test_source00_effective_contract_remains_the_reviewed_contract(blueprint):
    assert digest(blueprint.contract.model_dump(mode="json")) == (
        "1f31de160745f1602fd5308a19f40d297420a99953a2b0aa35978b9ffedb64d4"
    )


def test_source49_goods_area_phones_remain_source_only():
    case = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases" / SOURCE_49
    target = json.loads((case / "target.json").read_bytes())["documentPatch"]
    consignee = target["parties"]["consignee"]
    assert "phoneNumbers" not in consignee.get("contactDetails", {})
    source = (case / "source.txt").read_text()
    assert "**TELEPHONE NUMBER: +20102 8219120" in source
    assert "- +20 101 8503450" in source

