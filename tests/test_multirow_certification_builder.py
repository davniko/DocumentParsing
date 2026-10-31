"""Acceptance cannot turn missing/stale/manual-review evidence into a certificate."""

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_acceptance import SOURCE_FILES
from document_ocr.synthesis.declared_packages import CountedPackage, DeclaredPackages

spec = importlib.util.spec_from_file_location(
    "multirow_certifier",
    Path(__file__).parents[1] / "scripts/synthesis/certify_multirow_rollout.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    def write(name, value):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return {"path": name, "sha256": sha256_file(p)}

    shared = write("authority.json", {})
    config = {
        "source_ids": ["source"],
        "source_contracts": "cases",
        "ownership": "authority.json",
        "auxiliary": "authority.json",
        "independent": {
            name: shared
            for name in ("numeric_certificates", "geography_certificates", "wording_certificates")
        },
        "aliases": {"presentations": {"vocabulary": shared}},
        "declared_packages": {"group_counts": [1, 2, 3]},
    }
    contract = {"planning": "independent_groups"}
    for n in SOURCE_FILES:
        write("cases/source/" + n, {})
    write("src/document_ocr/synthesis/example.py", {})
    controls = []
    for count in (1, 2, 3):
        groups = (CountedPackage("PACKAGE_BOX", Decimal(1), Decimal(2), Decimal(1)),) * count
        ledger = DeclaredPackages(groups, ((1,) * count,), ((Decimal(100), Decimal(100)),))
        candidate = {
            "sourceDocumentId": "source",
            "documentId": str(count),
            "target": {},
            "scenario": {
                "provenance": {
                    "declaredPackages": ledger.receipt(),
                    "declaredPackageContract": contract,
                }
            },
        }
        cp = write(f"candidate-{count}.json", candidate)
        finding = {"sample_id": str(count), "problem": "reviewed arithmetic query"}
        controls.append(
            {
                "candidate": cp,
                "positions": write(
                    f"positions-{count}.json", {"candidateSha256": cp["sha256"], "target": {}}
                ),
                "review": write(
                    f"review-{count}.json",
                    {
                        "candidateHashes": {str(count): digest(candidate)},
                        "review": {"findings": [finding]},
                    },
                ),
                "findingDecisions": {digest(finding): "Exact ledger arithmetic confirmed."},
            }
        )
    decisions = {
        "sources": {"source": {"status": "PASS", "reason": "Reviewed.", "controls": controls}},
        "reviewScope": "test",
    }
    write("decisions.json", decisions)
    return config, {"sources": {"source": contract}}, decisions, root / "decisions.json"


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "source_hold",
        "missing_source",
        "missing_count",
        "finding",
        "empty_reason",
        "candidate",
        "positions",
        "contract",
        "duplicate",
    ],
)
def test_builder_rejects_unaccepted_or_stale_evidence(tmp_path, mutation):
    config, contracts, decisions, path = fixture(tmp_path)
    source = decisions["sources"]["source"]
    if mutation == "source_hold":
        source["status"] = "HELD"
    elif mutation == "missing_source":
        config["source_ids"].append("another")
    elif mutation == "missing_count":
        source["controls"].pop()
    elif mutation == "finding":
        source["controls"][0]["findingDecisions"] = {}
    elif mutation == "empty_reason":
        source["reason"] = " "
    elif mutation in {"candidate", "positions"}:
        (tmp_path / source["controls"][0][mutation]["path"]).write_text("changed")
    elif mutation == "contract":
        contracts["sources"]["source"] = {"planning": "changed"}
    elif mutation == "duplicate":
        source["controls"][1] = source["controls"][0]
    if mutation:
        with pytest.raises(ValueError):
            module.build(tmp_path, config, contracts, decisions, path)
    else:
        report = module.build(tmp_path, config, contracts, decisions, path)
        assert report["sources"]["source"]["groupCounts"] == [1, 2, 3]
        assert report["trainingDatasetMutation"] is False
        assert len(report["sources"]["source"]["samples"]) == 3
