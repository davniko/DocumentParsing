"""Exercise the real prepared campaign and publication preview after registry activation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_publication import publish_campaign
from document_ocr.synthesis.declared_package_acceptance import require_certified
from document_ocr.synthesis.independent_preparation import _runnable_config

ROOT = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")


def read(path):
    return json.loads(path.read_text())


def rejected(proxy, counts, expected):
    try:
        require_certified(proxy, counts)
    except ValueError as error:
        if expected not in str(error):
            raise
        return str(error)
    raise ValueError("invalid production configuration was admitted")


def main():
    root = Path.cwd()
    production = read(Path("configs/synthesis/mpci_bl_curated_v7_multirow.json"))
    config = read(ROOT / "batch-40/prepared-config.json")
    for name in ("ownership", "auxiliary", "declared_packages"):
        config[name] = copy.deepcopy(production[name])
    for name in ("numeric_certificates", "geography_certificates", "wording_certificates"):
        config["independent"][name] = production["independent"][name]
    sid = config["source_ids"][0]
    counts = {sid: 3}
    prepared = _runnable_config(
        root,
        config,
        counts,
        read(Path(config["independent"]["requests"]["path"])),
        read(Path(config["independent"]["private_plans"]["path"])),
        root / ROOT / "production-entrypoint-check",
    )
    if (
        prepared["independent"]["wording_certificates"]
        != production["independent"]["wording_certificates"]
    ):
        raise ValueError("preparation changed a production admission authority")
    campaign = create_campaign(root, prepared)
    spent = campaign.calls.spent
    require_certified(campaign, counts)
    campaign.package_presentations.require_certified([sid])
    preview = publish_campaign(campaign, publish=False)
    if preview["published"] or preview["valid"] != 3 or preview["failures"]:
        raise ValueError("production publication preview failed")
    bad = copy.deepcopy(prepared)
    bad["declared_packages"]["source_group_counts"][sid] = [4]
    proxy = SimpleNamespace(
        root=root,
        config=bad,
        declared_package_contracts=campaign.declared_package_contracts,
    )
    count_failure = rejected(proxy, counts, "source lacks acceptance")
    changed = copy.deepcopy(campaign.declared_package_contracts)
    changed[sid]["unreviewedChange"] = True
    proxy.config = prepared
    proxy.declared_package_contracts = changed
    contract_failure = rejected(proxy, counts, "source contract differs")
    if campaign.calls.spent != spent:
        raise ValueError("publication preview made a provider call")
    atomic_publish_json(
        ROOT / "production-entrypoint-check/result.json",
        {
            "status": "PASS",
            "preparedSourceCount": 1,
            "controls": 3,
            "positiveRuntimeAdmission": "PASS",
            "publicationPreview": preview,
            "negativeControls": {
                "unacceptedCount": count_failure,
                "changedContract": contract_failure,
            },
            "providerCalls": 0,
            "trainingDatasetMutation": False,
        },
    )
    print(json.dumps({"status": "PASS", "previewControls": 3, "negativeControls": 2}))


if __name__ == "__main__":
    main()
