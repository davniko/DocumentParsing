"""Compile the complete promoted configuration without drawing cargo or calling LLMs."""

import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_layout import PositionPolicy
from document_ocr.synthesis.curated_reflow import ReflowEngine
from document_ocr.synthesis.independent_preparation import inventory_sources
from document_ocr.synthesis.package_presentations import (
    PackagePresentationPolicy,
    PackagePresentations,
)


def main():
    root = Path.cwd()
    config_path = root / "configs/synthesis/mpci_bl_curated_v7_package_aliases.json"
    config = json.loads(config_path.read_text())
    work = root / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/package-alias-rollout-20261030"
    engine = ReflowEngine(root, PositionPolicy.model_validate(config["positions"]).reflow)
    presenter = PackagePresentations(
        root,
        PackagePresentationPolicy.model_validate(config["aliases"]["presentations"]),
        engine.metrics,
    )
    presenter.require_certified(config["source_ids"])
    assert len(config["source_ids"]) == len(config["capabilities"]) == 456
    assert config["independent_preparation"]["unit_profile_source_ids"] is None
    result = inventory_sources(root, config)
    assert result["selectedSources"] == result["readySources"] == 456
    result["configSha256"] = sha256_file(config_path)
    result["status"] = "PASS"
    atomic_publish_json(work / "production-config-validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
