import asyncio
import json

import pytest

from document_ocr.synthesis import paddle_campaign as campaign


@pytest.fixture
def harness(tmp_path, monkeypatch):
    class Catalog:
        def __init__(self, root, pin):
            self.root = root
            self.manifest = {"samplingConfig": {}}

        def shipment(self, path):
            return None, None, None, {"documentId": path.stem}

        def wording_specification(self, path):
            return None

        def render(self, path, **kwargs):
            if path.stem == "bad":
                raise ValueError("unowned private field")
            return {"documentId": path.stem, "status": "FULL_RENDER_CANDIDATE_REQUIRES_AUDIT"}

    monkeypatch.setattr(campaign, "PaddleTemplateCatalog", Catalog)
    return tmp_path


def test_render_reports_held_rows_without_substituting_or_accepting(harness):
    root = harness
    with pytest.raises(ValueError, match="1 native candidates held"):
        asyncio.run(
            campaign.run(
                root, {}, [root / "good.json", root / "bad.json"], root / "output", action="render"
            )
        )
    report = json.loads((root / "output/campaign.json").read_text())
    assert report["results"][0]["status"] == "FULL_RENDER_CANDIDATE_REQUIRES_AUDIT"
    assert report["failures"] == [
        {"documentId": "bad", "error": "unowned private field", "errorType": "ValueError"}
    ]
    assert not (root / "output/bad.json").exists()
    with pytest.raises(ValueError, match="output exists"):
        asyncio.run(campaign.run(root, {}, [root / "good.json"], root / "output", action="render"))


def test_selection_rejects_empty_or_duplicate_receipts_before_render(harness):
    root = harness
    for candidates in (
        [],
        [root / "a.json", root / "a.json"],
        [root / "x/a.json", root / "y/a.json"],
    ):
        with pytest.raises(ValueError):
            asyncio.run(campaign.run(root, {}, candidates, root / "output", action="render"))
    assert not (root / "output").exists()


def test_wording_proposals_are_bounded_and_never_auto_reviewed(harness, monkeypatch):
    root = harness
    config = dict.fromkeys(campaign.CuratedSynthesisConfig.model_fields, None)
    config["concurrency"] = 2
    monkeypatch.setattr(campaign, "read_pin", lambda *args: config)
    monkeypatch.setattr(
        campaign.PaddleTemplateCatalog,
        "wording_specification",
        lambda self, path: {"documentId": path.stem},
    )

    class Runner:
        def __init__(self, *args):
            self.spent = 0

    monkeypatch.setattr(campaign, "Runner", Runner)
    active = peak = 0

    async def propose(runner, spec):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return {**spec, "status": "PROPOSED_REQUIRES_REVIEW"}

    monkeypatch.setattr(campaign, "propose", propose)
    report = asyncio.run(
        campaign.run(
            root,
            {},
            [root / f"{i}.json" for i in range(5)],
            root / "output",
            action="propose-wording",
        )
    )
    assert peak == 2 and not report["failures"]
    assert {r["status"] for r in report["results"]} == {"PROPOSED_REQUIRES_REVIEW"}
