"""Scoped semantic regeneration cannot modify parties or accept bad proposals."""

import asyncio
import copy
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_curated_campaign import postal_campaign as postal_campaign

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest

_SPEC = importlib.util.spec_from_file_location(
    "reviewed_goods_regeneration",
    Path(__file__).resolve().parents[1] / "scripts/synthesis/regenerate_reviewed_goods.py",
)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = module
_SPEC.loader.exec_module(module)


@pytest.fixture
def correction(postal_campaign):
    fixture = postal_campaign
    campaign = fixture.campaign
    campaign.root = fixture.path.parent.parent
    campaign.variant_counts = {"source": 1}
    campaign.plan = lambda sid, variant: (
        fixture.blueprint,
        fixture.scenario,
        "sample",
        copy.deepcopy(fixture.target),
    )
    campaign._validated_wording = lambda receipt, request: receipt["values"]
    evidence = campaign.root / "review.md"
    evidence.write_text("Authored review: regenerate only the product wording.")
    state = SimpleNamespace(calls=0, fail=False, concurrent_edit=False)

    async def call(stage, identity, model, system, prompt):
        state.calls += 1
        result = model.model_validate({"s0": {"product": "ARABICA COFFEE BEANS"}})
        call_key = digest(
            [stage, identity, model.model_json_schema(), system, prompt, {"model": "offline"}]
        )
        path = campaign.output / "calls" / f"{stage}-{call_key}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"output": result.model_dump(), "costUsd": "0"}))
        if state.concurrent_edit:
            fixture.path.write_text('{"concurrent": true}')
        return result

    def validate(requests, values):
        assert values["sample"]["shipper_name"] == fixture.receipt["values"]["shipper_name"]
        if state.fail:
            raise ValueError("explicit semantic hold")

    campaign.validate_generated_wording = validate
    campaign.calls = SimpleNamespace(
        call=call,
        output=campaign.output,
        spent=Decimal(0),
        config={"provider": {"model": "offline"}},
    )
    row = {
        "sourceDocumentId": "source",
        "variant": 1,
        "sampleId": "sample",
        "wordingReceiptSha256": sha256_file(fixture.path),
        "keys": ["product"],
        "reason": "Replace the reviewed unsuitable product wording.",
        "auditEvidence": {"path": "review.md", "sha256": sha256_file(evidence)},
    }
    return SimpleNamespace(fixture=fixture, campaign=campaign, state=state, row=row)


def plan(row):
    return module.CorrectionPlan.model_validate_json(
        json.dumps({"status": "approved_for_regeneration", "corrections": [row]})
    )


def test_goods_only_regeneration_archives_exact_old_receipt(correction):
    fixture = correction.fixture
    before_bytes = fixture.path.read_bytes()
    result = asyncio.run(module.regenerate(correction.campaign, plan(correction.row)))
    assert result["corrected"] == correction.state.calls == 1
    saved = json.loads(fixture.path.read_bytes())
    assert saved["values"] == fixture.receipt["values"] | {"product": "ARABICA COFFEE BEANS"}
    archive = correction.campaign.output / saved["corrections"][-1]["beforeReceipt"]["path"]
    assert archive.read_bytes() == before_bytes
    assert saved["provenance"] == fixture.receipt["provenance"]


def test_invalid_full_wording_keeps_accepted_receipt_byte_identical(correction):
    before = correction.fixture.path.read_bytes()
    correction.state.fail = True
    with pytest.raises(ValueError, match="explicit semantic hold"):
        asyncio.run(module.regenerate(correction.campaign, plan(correction.row)))
    assert correction.fixture.path.read_bytes() == before
    assert correction.state.calls == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("keys", ["shipper_name"]),
        ("sampleId", "wrong"),
        ("variant", 2),
        ("wordingReceiptSha256", "0" * 64),
    ],
)
def test_bad_scope_or_stale_evidence_stops_before_paid_call(correction, field, value):
    correction.row[field] = value
    before = correction.fixture.path.read_bytes()
    with pytest.raises(ValueError):
        asyncio.run(module.regenerate(correction.campaign, plan(correction.row)))
    assert correction.state.calls == 0
    assert correction.fixture.path.read_bytes() == before


def test_concurrent_receipt_edit_is_not_overwritten(correction):
    correction.state.concurrent_edit = True
    with pytest.raises(ValueError, match="changed during correction"):
        asyncio.run(module.regenerate(correction.campaign, plan(correction.row)))
    assert json.loads(correction.fixture.path.read_bytes()) == {"concurrent": True}


def test_duplicate_samples_are_rejected_before_execution(correction):
    with pytest.raises(ValueError, match="duplicate reviewed sample"):
        module.CorrectionPlan.model_validate_json(
            json.dumps(
                {
                    "status": "approved_for_regeneration",
                    "corrections": [correction.row, correction.row],
                }
            )
        )


def test_superseded_candidate_is_archived_before_fresh_render(correction):
    path = correction.campaign.output / "candidates" / "sample.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "documentId": "sample",
                "sourceDocumentId": "source",
                "wordingReceiptSha256": digest(correction.fixture.receipt),
            }
        )
    )
    before = path.read_bytes()
    result = asyncio.run(module.regenerate(correction.campaign, plan(correction.row)))
    assert not path.exists()
    (relative,) = result["supersededCandidateArchives"]
    assert (correction.campaign.output / relative).read_bytes() == before


def test_published_campaign_is_not_rewritten(correction):
    (correction.campaign.output / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="published campaigns are immutable"):
        asyncio.run(module.regenerate(correction.campaign, plan(correction.row)))
    assert correction.state.calls == 0
