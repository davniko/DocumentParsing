"""Presentation failures receive bounded regeneration before contacts/publication."""

import asyncio
import json
from types import MethodType

import pytest
from test_curated_campaign import postal_campaign as postal_campaign

from document_ocr.synthesis.curated_campaign import Campaign


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_extended_semantic_gate_uses_recorded_correction_before_contacts(
    postal_campaign, repair_succeeds
):
    fixture = postal_campaign
    fixture.path.unlink()
    stages = []
    contacts = []

    def validate(self, requests, values):
        Campaign.validate_generated_wording(self, requests, values)
        if any(row["product"] == "SHELF STABLE UNSUITABLE WORDING" for row in values.values()):
            raise ValueError("explicit independent presentation hold")

    async def call(stage, sid, output_type, system, prompt):
        stages.append(stage)
        if stage == "wording-validation-repair":
            assert "rename those fictional subaddress components" in prompt
        values = dict(fixture.receipt["values"])
        if stage == "exact-wording" or not repair_succeeds:
            values["product"] = "SHELF STABLE UNSUITABLE WORDING"
        else:
            assert "explicit independent presentation hold" in prompt
        return output_type.model_validate({"s0": values})

    async def generate_contacts(sid):
        contacts.append(sid)
        return {"companies": 0}

    fixture.campaign.validate_generated_wording = MethodType(validate, fixture.campaign)
    fixture.campaign.calls.call = call
    fixture.campaign.generate_contacts = generate_contacts
    if repair_succeeds:
        result = asyncio.run(fixture.campaign.generate("source"))
        assert "independent presentation hold" in result["wordingValidationRepair"]
        assert json.loads(fixture.path.read_text())["values"] == fixture.receipt["values"]
        assert contacts == ["source"]
    else:
        with pytest.raises(ValueError, match="independent presentation hold"):
            asyncio.run(fixture.campaign.generate("source"))
        assert not fixture.path.exists()
        assert contacts == []
    assert stages == ["exact-wording", "wording-validation-repair"]


def test_scoped_receipt_keeps_native_values_when_renderer_projects(postal_campaign):
    fixture = postal_campaign
    cached = json.loads(fixture.path.read_text())
    cached["scopedExtension"] = {"test": "native chain remains immutable"}
    fixture.path.write_text(json.dumps(cached))
    original = fixture.path.read_bytes()

    def validate_receipt(receipt, request):
        result = dict(receipt["values"])
        result["product"] += " HOST-OWNED SUFFIX"
        return result

    async def forbidden(*args):
        raise AssertionError("cached accepted wording must not generate again")

    fixture.campaign._validated_wording = validate_receipt
    fixture.campaign.calls.call = forbidden
    asyncio.run(fixture.campaign.generate("source"))
    assert fixture.path.read_bytes() == original
