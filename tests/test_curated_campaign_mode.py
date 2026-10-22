"""A donor-free config must never select observed shipment sampling."""

import json
from pathlib import Path

import pytest

from document_ocr.synthesis.curated_campaign import Campaign, create_campaign


@pytest.mark.parametrize("mode", [None, "", "donorfree", True, 1])
def test_unknown_sampling_mode_is_not_observed(mode):
    with pytest.raises(ValueError, match="unknown sampling_mode"):
        create_campaign(Path("."), {"sampling_mode": mode})


def test_independent_mode_requires_explicit_inputs():
    with pytest.raises(ValueError, match="explicit pinned inputs"):
        create_campaign(Path("."), {"sampling_mode": "independent"})


def test_independent_inputs_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="require sampling_mode"):
        create_campaign(Path("."), {"independent": {}})


def test_direct_observed_campaign_cannot_bypass_mode_dispatch():
    with pytest.raises(ValueError, match="implementation differs"):
        Campaign(Path("."), {"sampling_mode": "independent", "independent": {}})


def test_existing_observed_factory_is_explicit(monkeypatch):
    import document_ocr.synthesis.curated_campaign as module

    calls = []
    sentinel = object()

    def constructor(root, config):
        calls.append((root, config))
        return sentinel

    monkeypatch.setattr(module, "Campaign", constructor)
    config = {"sampling_mode": "observed"}
    assert module.create_campaign(Path("project"), config) is sentinel
    assert calls == [(Path("project"), config)]


@pytest.mark.parametrize("ready", [1, 2])
def test_inventory_cli_keeps_all_holds_without_constructing_native_campaign(
    tmp_path, monkeypatch, capsys, ready
):
    import document_ocr.synthesis.curated_campaign as module
    import document_ocr.synthesis.independent_preparation as preparation

    config = tmp_path / "campaign.yaml"
    config.write_text("output: inspection\nsampling_mode: independent\n")
    report = {
        "inventory": {"a": {"status": "ready_source"}, "b": {"status": "held_source"}},
        "selectedSources": 2,
        "plannedSlots": 6,
        "readySources": ready,
        "counts": {"source_prerequisites_pass": ready, "missing_numeric_certificate": 2 - ready},
    }
    monkeypatch.setattr(preparation, "inventory_sources", lambda root, value: report)
    monkeypatch.setattr(
        module, "create_campaign", lambda *args: pytest.fail("native campaign was constructed")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "campaign",
            "inventory-independent",
            "--config",
            str(config),
            "--project-root",
            str(tmp_path),
        ],
    )
    if ready == 1:
        with pytest.raises(SystemExit, match="1"):
            module.main()
    else:
        module.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["selectedSources"] == 2
    assert printed["plannedSlots"] == 6
    assert printed["nativeCalls"] == printed["cargoDraws"] == 0
    assert json.loads((tmp_path / printed["inventoryPath"]).read_bytes()) == report


@pytest.mark.parametrize("flag", ["--retry-invalid-output", "--retry-rate-limited"])
def test_preparation_cli_forwards_explicit_retry_permission(tmp_path, monkeypatch, capsys, flag):
    import document_ocr.synthesis.curated_campaign as module
    import document_ocr.synthesis.independent_preparation as preparation

    config = tmp_path / "campaign.yaml"
    config.write_text(
        "sampling_mode: independent\nindependent_preparation:\n  package_targets: {}\n"
    )
    calls = []

    async def prepare(root, value, **kwargs):
        calls.append((root, value, kwargs))
        return {
            "summaryPath": "summary.json",
            "preparedConfig": {"path": "prepared.json", "sha256": "a" * 64},
            "plannedSlots": 3,
            "readySources": 1,
            "readySlots": 3,
            "counts": {"ready": 3},
            "spentBeforeUsd": "0",
            "spentAfterUsd": "0",
        }

    monkeypatch.setattr(preparation, "prepare_campaign", prepare)
    monkeypatch.setattr(
        module, "create_campaign", lambda *args: pytest.fail("wrong campaign construction")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "campaign",
            "prepare-independent",
            "--config",
            str(config),
            "--project-root",
            str(tmp_path),
            flag,
        ],
    )
    module.main()
    assert calls[0][2] == {
        "package_targets": {},
        "retry_rate_limited": flag == "--retry-rate-limited",
        "retry_invalid_output": flag == "--retry-invalid-output",
        "retry_failed_receipt_sha256s": [],
    }
    assert json.loads(capsys.readouterr().out)["readySlots"] == 3
