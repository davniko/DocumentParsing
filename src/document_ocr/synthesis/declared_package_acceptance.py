"""Publication admission for source-specific multi-row synthesis evidence."""

from __future__ import annotations

import json
from pathlib import Path

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_preparation import selection

SOURCE_FILES = frozenset(
    {"source.txt", "target.json", "contract.json", "template.json", "alignment.json"}
)


def authority_pins(root, config):
    """Pin the actual template/planning/rendering inputs, not only source text."""
    values = {}
    for name in ("ownership", "auxiliary"):
        path = config[name]
        values[name] = {"path": path, "sha256": sha256_file(root / path)}
    for name in ("numeric_certificates", "geography_certificates", "wording_certificates"):
        pin = config["independent"][name]
        if sha256_file(root / pin["path"]) != pin["sha256"]:
            raise ValueError("multi-row authority differs from its configured pin: " + name)
        values[name] = pin
    for name, pin in config["aliases"]["presentations"].items():
        if isinstance(pin, dict) and "path" in pin:
            if sha256_file(root / pin["path"]) != pin["sha256"]:
                raise ValueError("multi-row alias authority differs from its pin: " + name)
            values["presentations." + name] = pin
    return values


def require_certified(campaign, counts):
    policy = campaign.config["declared_packages"]
    required = {
        sid: {selection(policy, sid, v) for v in range(1, n + 1)} - {1} for sid, n in counts.items()
    }
    required = {sid: values for sid, values in required.items() if values}
    if not required:
        return
    pin = policy.get("acceptance")
    if pin is None or sha256_file(campaign.root / pin["path"]) != pin["sha256"]:
        raise ValueError("multi-row publication requires pinned synthesis acceptance")
    evidence = json.loads((campaign.root / pin["path"]).read_text())
    if evidence["status"] != "certified_native_synthesis_and_geometry":
        raise ValueError("multi-row inspection runs are not production acceptance")
    if evidence["authorities"] != authority_pins(campaign.root, campaign.config):
        raise ValueError("multi-row template authorities differ from accepted synthesis")
    for pin in evidence["implementation"]:
        if sha256_file(campaign.root / pin["path"]) != pin["sha256"]:
            raise ValueError("multi-row implementation differs from certified code")
    if not evidence["implementation"]:
        raise ValueError("multi-row acceptance lacks implementation pins")
    for sid, values in required.items():
        row = evidence["sources"].get(sid)
        if row is None or row["status"] != "PASS" or not values <= set(row["groupCounts"]):
            raise ValueError("source lacks acceptance for requested public package counts: " + sid)
        if row["contractSha256"] != digest(campaign.declared_package_contracts[sid]):
            raise ValueError("multi-row source contract differs from accepted authority")
        if set(row["sourceFiles"]) != SOURCE_FILES:
            raise ValueError("multi-row source acceptance lacks complete template pins")
        for name, pin in row["sourceFiles"].items():
            expected = Path(campaign.config["source_contracts"]) / sid / name
            if (
                Path(pin["path"]) != expected
                or sha256_file(campaign.root / expected) != pin["sha256"]
            ):
                raise ValueError("multi-row source template differs from accepted synthesis")
        if not row["samples"]:
            raise ValueError("source acceptance lacks tested synthesis")
        for sample in row["samples"]:
            for name in ("candidate", "positions", "review", "adjudication"):
                pin = sample[name]
                if sha256_file(campaign.root / pin["path"]) != pin["sha256"]:
                    raise ValueError("multi-row synthesis evidence changed: " + name)
