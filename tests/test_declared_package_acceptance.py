import json
from types import SimpleNamespace

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_acceptance import (
    SOURCE_FILES,
    authority_pins,
    require_certified,
)


def test_inspection_config_cannot_publish_without_acceptance(tmp_path):
    c = SimpleNamespace(root=tmp_path, config={"declared_packages": {"group_counts": [1, 2, 3]}})
    require_certified(c, {"s": 1})
    with pytest.raises(ValueError, match="pinned synthesis acceptance"):
        require_certified(c, {"s": 3})


@pytest.mark.parametrize(
    "mutate", [None, "code", "candidate", "contract", "count", "pending", "ownership"]
)
def test_acceptance_is_bound_to_code_contract_count_and_reviewed_samples(tmp_path, mutate):
    pins = {}
    for name in ("code", "candidate", "positions", "review", "adjudication", "ownership"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({name: "evidence"}))
        pins[name] = {"path": path.name, "sha256": sha256_file(path)}
    contract = {"planning": "independent_groups", "declarations": []}
    evidence = {
        "status": "certified_native_synthesis_and_geometry",
        "implementation": [pins["code"]],
        "sources": {
            "s": {
                "status": "PASS",
                "groupCounts": [2, 3],
                "contractSha256": digest(contract),
                "samples": [
                    {key: pins[key] for key in ("candidate", "positions", "review", "adjudication")}
                ],
            }
        },
    }
    source_files = {}
    for name in SOURCE_FILES:
        file = tmp_path / "cases" / "s" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("reviewed")
        source_files[name] = {"path": str(file.relative_to(tmp_path)), "sha256": sha256_file(file)}
    evidence["sources"]["s"]["sourceFiles"] = source_files
    if mutate == "count":
        evidence["sources"]["s"]["groupCounts"] = [2]
    if mutate == "pending":
        evidence["status"] = "AWAITING_MAIN_AGENT_REVIEW"
    path = tmp_path / "acceptance.json"
    config = {
        **{
            name: pins["ownership"]["path"]
            for name in ("ownership", "auxiliary", "source_contracts")
        },
        "source_contracts": "cases",
        "independent": {
            name: pins["ownership"]
            for name in ("numeric_certificates", "geography_certificates", "wording_certificates")
        },
        "aliases": {"presentations": {"vocabulary": pins["ownership"]}},
        "declared_packages": {
            "group_counts": [3],
            "acceptance": {
                "path": path.name,
                "sha256": "pending",
            },
        },
    }
    evidence["authorities"] = authority_pins(tmp_path, config)
    path.write_text(json.dumps(evidence))
    config["declared_packages"]["acceptance"]["sha256"] = sha256_file(path)
    c = SimpleNamespace(root=tmp_path, config=config, declared_package_contracts={"s": contract})
    if mutate in ("candidate", "code", "ownership"):
        (tmp_path / pins[mutate]["path"]).write_text("changed")
    if mutate == "contract":
        contract["planning"] = "shared_units"
    if mutate:
        with pytest.raises(ValueError):
            require_certified(c, {"s": 1})
    else:
        require_certified(c, {"s": 1})
