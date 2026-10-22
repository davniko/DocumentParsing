"""Source-preparation re-pinning must never relax certified postal ownership."""

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate,
    load_independent_geography_certificates,
)
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
RUN = BASE / "independent-flat-batch11-20261021/geography-compiled"
SID = "doc_5718f96fce89f4f7ec8616548702042eba867fd4bb61f7b90a301e5e398f6a95"
spec = importlib.util.spec_from_file_location(
    "reviewed_geography_rebase", ROOT / "scripts/synthesis/rebase_reviewed_geography.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope="module")
def pair():
    cfg = json.loads((RUN / "config.json").read_bytes())
    cfg.update(source_ids=[SID], capabilities={SID: cfg["capabilities"][SID]})
    current = _SourceOwnershipCampaign(ROOT, cfg).blueprint(SID)
    old_cfg = copy.deepcopy(cfg)
    old_cfg["ownership"] = str(RUN / (SID + "-previous-ownership.json"))
    previous = _SourceOwnershipCampaign(ROOT, old_cfg).blueprint(SID)
    receipt = json.loads((RUN / "validation.json").read_bytes())
    pin = receipt["beforeCatalog"]
    gate = IndependentGeographyGate(
        load_independent_geography_certificates(
            (ROOT / pin["path"]).read_bytes(), expected_sha256=pin["sha256"]
        )
    )
    return gate, previous, current


def test_route_only_rebase_requires_exact_old_certificate(pair):
    gate, previous, current = pair
    with pytest.raises(ValueError, match="certificate differs"):
        gate._certificate(current, allow_prepared=False)
    assert module.validate_postal_preserving_rebase(gate, previous, current) == (
        gate._blueprint_pins(current)
    )


@pytest.mark.parametrize("field", ["document_id", "source", "target"])
def test_identity_ocr_and_public_target_changes_are_rejected(pair, field):
    gate, previous, current = pair
    value = {"documentPatch": {}} if field == "target" else getattr(current, field) + "BAD"
    with pytest.raises(ValueError, match="source identity, OCR or extraction target"):
        module.validate_postal_preserving_rebase(gate, previous, replace(current, **{field: value}))


def test_changed_postal_component_is_not_recertified(pair):
    gate, previous, current = pair
    variables = [
        v.model_copy(update={"value": v.value + " CHANGED"}) if v.key == "v021" else v
        for v in current.contract.variables
    ]
    changed = replace(
        current, contract=current.contract.model_copy(update={"variables": variables})
    )
    with pytest.raises(ValueError, match="private actor/postal ownership"):
        module.validate_postal_preserving_rebase(gate, previous, changed)


def test_forged_previous_ownership_is_rejected(pair):
    gate, previous, current = pair
    ownership = copy.deepcopy(previous.ownership_data)
    ownership["unreviewed"] = True
    with pytest.raises(ValueError, match="certificate differs"):
        module.validate_postal_preserving_rebase(
            gate, replace(previous, ownership_data=ownership), current
        )


def test_catalog_changes_only_reviewed_pins():
    before = json.loads(
        (BASE / "independent-geography-v5/geography_certificates.json").read_bytes()
    )
    after = json.loads((RUN / "geography_certificates.json").read_bytes())
    assert before["sources"][SID]["effective_contract_sha256"] == (
        after["sources"][SID]["effective_contract_sha256"]
    )
    assert before["sources"][SID]["ownership_sha256"] != (
        after["sources"][SID]["ownership_sha256"]
    )
    before["sources"][SID]["ownership_sha256"] = after["sources"][SID]["ownership_sha256"]
    assert before == after


def test_vehicle_count_product_split_preserves_public_and_postal_facts():
    sid = "doc_4831fa04f123a0f7babc9ed31ae7a8d3dc295f2e27f89dde5bd47c7025dda2eb"
    run = BASE / "independent-flat-batch25-20261022/compiled"
    cfg = json.loads((run / "config.json").read_bytes())
    current = _SourceOwnershipCampaign(ROOT, cfg).blueprint(sid)
    old_contract = json.loads((ROOT / "docs/analysis/r28-donor-free-full-registry-20261021/batch25/previous-contracts" / sid / "contract.json").read_bytes())["contract"]
    new_contract = json.loads((BASE / "cases" / sid / "contract.json").read_bytes())["contract"]
    assert [v for v in old_contract["variables"] if v["kind"] in {"name", "postal"}] == [
        v for v in new_contract["variables"] if v["kind"] in {"name", "postal"}
    ]
    inspected = BASE / "independent-flat-batch22-20261021/inspection/inspection" / sid
    previous_target = json.loads((inspected / "target.json").read_bytes())
    previous_target["documentPatch"]["goodsItemDetails"][0]["description"] = "USED CHEREAU TRAILER"
    assert previous_target == current.target
    assert (inspected / "source.txt").read_bytes().decode() == current.source
    assert current.description_blocks.fields[0].spans[0].text == "USED CHEREAU TRAILER"
    product = next(r for r in current.regions if r.curated_key == "v001")
    quantity = next(r for r in current.regions if r.curated_key == "vehicle_count")
    assert product.source == "USED CHEREAU TRAILER"
    assert quantity.source == "1" and quantity.end < product.start
    assert any(r.source == "UNP." and r.kind == "package" for r in current.regions)
    chassis = "anchor:documentPatch.cargoGroups[0].marksAndNumbers[0]"
    assert current.historical_bindings[chassis]["value_kind"] == "identifier"
    assert current.ownership_data["identifier_privacy"][chassis] == "private"
    assert any(
        {"source": "SECONDHAND VEHICLE(S)", "text": "VEHICLE(S)"}
        in binding.get("auxiliary_caption_transforms", [])
        for binding in current.historical_bindings.values()
    )


def test_vehicle_reception_counts_packages_not_structured_rows():
    cfg = json.loads((BASE / "independent-flat-batch24-20261022/geography-compiled/config.json").read_bytes())
    sid = "doc_2763b891d894632d56df7eaa22ff6d81426e42a49b9f2816098120a63dc1d12a"
    bp = _SourceOwnershipCampaign(ROOT, cfg).blueprint(sid)
    assert "agent:package:count" not in bp.historical_bindings
    owner = bp.historical_bindings["vehicle_reception_quantity"]
    assert owner.get("derivation") is None
    assert owner["target_paths"] == [
        "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"
    ]
    assert owner["occurrences"][0]["source_text"] == "1"
