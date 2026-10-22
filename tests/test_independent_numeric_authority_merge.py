"""Pinned review catalogs merge without promoting unreviewed source coverage."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_ocr.atomic import AtomicConflictError
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import SourceContract

_PATH = Path(__file__).parents[1] / "scripts/synthesis/merge_independent_numeric_authority.py"
_SPEC = importlib.util.spec_from_file_location("numeric_authority_merge", _PATH)
merge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge)


@pytest.fixture
def catalog_job(tmp_path, monkeypatch):
    def pin(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False))
        return {"path": name, "sha256": sha256_file(path)}

    def source(sid, *, held=False):
        text = "Coton de qualité"
        target = {"documentPatch": {"goodsItemDetails": [{"description": text}]}}
        contract = {
            "variables": [
                {
                    "key": "product",
                    "kind": "product",
                    "value": text,
                    "meaning": "Complete goods passage",
                    "required_literals": [],
                    "occurrences": [{"text": text, "occurrence": 1, "presentation": "text"}],
                }
            ],
            "targets": [
                {"path": "documentPatch.goodsItemDetails[0].description", "expression": "{product}"}
            ],
            "fixed_context": "No cargo donor authority",
        }
        directory = tmp_path / "cases" / sid
        directory.mkdir(parents=True)
        (directory / "source.txt").write_text(text)
        pin(f"cases/{sid}/target.json", target)
        pin(f"cases/{sid}/contract.json", {"contract": contract})
        effective_contract = copy.deepcopy(contract)
        effective_contract["fixed_context"] += "; effective lexical ownership"
        review = pin(f"reviews/{sid}.json", {"sourceDocumentId": sid, "completeRead": True})
        return {
            "reviewStatus": "reviewed",
            "sourceSha256": merge.numeric_digest(text.encode()),
            "sourceContractSha256": merge.numeric_digest(effective_contract),
            "sourcePublicTargetSha256": merge.numeric_digest(target),
            "sourceContainerCount": 0,
            "sourceAllocationPresence": [],
            "regions": [],
            "wordingOwners": [],
            "constraints": [],
            "holdReasons": ["No implemented three-level accounting frame"] if held else [],
            "completeSourceReviewEvidence": review,
        }

    certificates = {sid: source(sid, held=(sid == "b")) for sid in ("a", "b")}
    catalogs = [
        pin(
            f"input/{sid}.json",
            {
                "version": 2,
                "selectedSources": [sid],
                "sources": {sid: certificate},
                "priorProvenance": {"input": sid},
            },
        )
        for sid, certificate in certificates.items()
    ]
    manifest = {
        "sourceCount": 3,
        "cases": {
            sid: {"capability": {"family": "ambient", "identity_count": 1}}
            for sid in ("a", "b", "unreviewed")
        },
    }

    class SourceCompiler:
        def __init__(self, root, config):
            self.root, self.config = root, config
            self.source_ids = config["source_ids"]

        def blueprint(self, sid):
            directory = self.root / self.config["source_contracts"] / sid
            contract = json.loads((directory / "contract.json").read_bytes())["contract"]
            contract["fixed_context"] += "; effective lexical ownership"
            return SimpleNamespace(
                document_id=sid,
                source=(directory / "source.txt").read_text(),
                target=json.loads((directory / "target.json").read_bytes()),
                contract=SourceContract.model_validate(contract),
            )

        def plan(self, *args):
            raise AssertionError("merge must never plan observed cargo")

    monkeypatch.setattr(merge, "_SourceOwnershipCampaign", SourceCompiler)
    job = {
        "catalogInputs": catalogs,
        "registryManifest": pin("manifest.json", manifest),
        "sourceCases": "cases",
        "sourceReadinessOverlay": pin("overlay.json", {"status": "approved", "sources": {}}),
        "baseConfig": pin(
            "base.json",
            {
                "source_contracts": "cases",
                "historical_catalog": "cases",
                "dataset": "data",
                "ownership": "ownership.json",
                "auxiliary": "auxiliary.json",
                "task_constraints": "task.json",
                "sampling": {},
                "independent": {},
            },
        ),
        "inventoryOutput": "inventory",
        "output": "merged",
    }
    for name in (
        "data/train.jsonl",
        "data/validation.jsonl",
        "ownership.json",
        "auxiliary.json",
        "task.json",
    ):
        pin(name, {})
    return tmp_path, job, certificates, pin


def test_merge_preserves_every_numeric_fact_hold_provenance_and_missing_source(catalog_job):
    root, job, certificates, _ = catalog_job
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    catalog, receipt = merge.merge_catalogs(root, job)
    assert catalog["sources"] == certificates
    assert catalog["selectedSources"] == ["a", "b"]
    assert catalog["mergedInputCatalogs"][0]["metadata"]["priorProvenance"] == {"input": "a"}
    assert receipt["reviewedNumericSources"] == 2
    assert receipt["numericHeldSources"] == 1 and receipt["unreviewedNumericSources"] == 1
    assert receipt["allOtherCertificateFieldsUnchanged"] is True
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    raw_contract = json.loads((root / "cases/a/contract.json").read_bytes())["contract"]
    assert merge.numeric_digest(raw_contract) != catalog["sources"]["a"]["sourceContractSha256"]


def test_identical_duplicate_source_is_fatal_not_silently_deduplicated(catalog_job):
    root, job, _, _ = catalog_job
    job["catalogInputs"].append(job["catalogInputs"][0])
    with pytest.raises(ValueError, match="duplicate numeric source"):
        merge.merge_catalogs(root, job)


def test_cached_input_drift_is_fatal_at_the_merge_barrier(catalog_job, monkeypatch):
    root, job, _, _ = catalog_job
    original = merge._SourceOwnershipCampaign

    class MutatingCompiler(original):
        def blueprint(self, sid):
            result = super().blueprint(sid)
            if sid == "a":
                (root / job["catalogInputs"][0]["path"]).write_text("mutated during compilation")
            return result

    monkeypatch.setattr(merge, "_SourceOwnershipCampaign", MutatingCompiler)
    with pytest.raises(ValueError, match=r"authority.*changed during merge|authority changed"):
        merge.merge_catalogs(root, job)


@pytest.mark.parametrize("kind", ["catalog", "evidence", "case_target", "source_bytes"])
def test_pin_or_current_source_basis_mismatch_is_fatal(catalog_job, kind):
    root, job, _, _ = catalog_job
    if kind == "catalog":
        (root / job["catalogInputs"][0]["path"]).write_text("tampered")
    elif kind == "evidence":
        (root / "reviews/a.json").write_text("tampered")
    elif kind == "case_target":
        (root / "cases/a/target.json").write_text(json.dumps({"documentPatch": {}}))
    else:
        (root / "cases/a/source.txt").write_text("Changed source")
    with pytest.raises(ValueError, match="pin mismatch"):
        merge.merge_catalogs(root, job)


def test_readiness_overlay_is_explicit_separate_and_evidence_pinned(catalog_job):
    root, job, certificates, pin = catalog_job
    hold = {
        "category": "privacy",
        "reason": "Source-only agent has no generator",
        "evidence": pin("readiness.json", {"sourceDocumentId": "a", "finding": "private agent"}),
    }
    job["sourceReadinessOverlay"] = pin(
        "overlay-approved.json",
        {
            "status": "approved",
            "sources": {"a": [hold]},
        },
    )
    catalog, receipt = merge.merge_catalogs(root, job)
    output_hold = copy.deepcopy(hold)
    output_hold["evidence"] = {
        "path": "merged/source-readiness-evidence/" + hold["evidence"]["sha256"] + ".json",
        "sha256": hold["evidence"]["sha256"],
    }
    assert catalog["sources"]["a"] == {**certificates["a"], "sourceReadinessHolds": [output_hold]}
    assert catalog["sources"]["a"]["holdReasons"] == []
    assert receipt["addedFieldsBySource"] == {"a": {"sourceReadinessHolds": [output_hold]}}
    assert receipt["readinessEvidenceRelocations"] == [
        {
            "input": hold["evidence"],
            "output": output_hold["evidence"],
        }
    ]
    assert receipt["sourceReadinessOverlay"] == job["sourceReadinessOverlay"]


def test_unapproved_or_unreviewed_overlay_never_inflates_coverage(catalog_job):
    root, job, _, pin = catalog_job
    job["sourceReadinessOverlay"] = pin("unapproved.json", {"status": "pending", "sources": {}})
    with pytest.raises(ValueError, match="root approval"):
        merge.merge_catalogs(root, job)
    job["sourceReadinessOverlay"] = pin(
        "extra.json", {"status": "approved", "sources": {"unreviewed": []}}
    )
    with pytest.raises(ValueError, match="reviewed sources"):
        merge.merge_catalogs(root, job)


def test_only_explicit_authorized_pin_amendment_changes_old_source_basis(catalog_job):
    root, job, certificates, pin = catalog_job
    target = {"documentPatch": {"goodsItemDetails": [{"description": "Révisé"}]}}
    pin("cases/a/target.json", target)
    fields = {
        "sourcePublicTargetSha256": {
            "before": certificates["a"]["sourcePublicTargetSha256"],
            "after": merge.numeric_digest(target),
        }
    }
    job["pinAmendments"] = pin(
        "amendment.json",
        {
            "status": "approved",
            "reason": "Explicit source phone label correction",
            "sources": {"a": fields},
        },
    )
    catalog, receipt = merge.merge_catalogs(root, job)
    expected = copy.deepcopy(certificates["a"])
    expected["sourcePublicTargetSha256"] = merge.numeric_digest(target)
    assert catalog["sources"]["a"] == expected
    assert receipt["changedSemanticPins"] == {"a": fields}
    fields["sourceSha256"] = {"before": certificates["a"]["sourceSha256"], "after": "a" * 64}
    job["pinAmendments"] = pin(
        "bad-amendment.json",
        {"status": "approved", "reason": "Too broad", "sources": {"a": fields}},
    )
    with pytest.raises(ValueError, match="scope"):
        merge.merge_catalogs(root, job)


def test_inventory_config_preserves_presentation_not_old_cargo_constraints():
    cap = {
        "family": "vehicle",
        "identity_count": 2,
        "allowed_hs_headings": ["8701"],
        "allowed_package_categories": ["PACKAGE_VEHICLE"],
        "fixed_package_quantity": 10,
        "quantity_multiple": 5,
        "physical_profile": "source_whole_units",
        "origin_country": "DE",
        "party_sides": {"shipper": "origin"},
        "allowed_equipment_pairs": ["FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"],
    }
    manifest = {
        "sourceCount": 2,
        "cases": {
            "a": {"capability": cap},
            "b": {"capability": {"family": "dg_vehicle"}},
        },
    }
    base = {
        "source_ids": ["old"],
        "template_sampling": {"source_counts": {"old": 99}},
        "sampling": {"load_profiles": {}, "observed_hs_overrides": {"old": {}}},
        "independent": {"registry": {}, "requests": {}, "private_plans": {}},
    }
    result = merge.inventory_config(
        base, manifest, {"path": "new.json", "sha256": "a" * 64}, output="inventory"
    )
    assert result["source_ids"] == ["a", "b"] and result["variants_per_source"] == 3
    assert result["capabilities"]["a"] == {
        "family": "ambient",
        "identity_count": 2,
        "origin_country": "DE",
        "party_sides": {"shipper": "origin"},
        "allowed_equipment_pairs": ["FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"],
    }
    assert result["capabilities"]["b"] == {"family": "dg_chemical", "identity_count": 1}
    assert result["sampling"] == {} and "template_sampling" not in result
    assert "requests" not in result["independent"] and "private_plans" not in result["independent"]


def test_explicit_cli_publication_is_immutable_and_keeps_all_source_denominators(
    catalog_job, monkeypatch, capsys
):
    root, job, _, pin = catalog_job
    job["output"] = "merged"
    hold_evidence = pin("hold-evidence.json", {"finding": "Unowned société contact"})
    source_evidence_bytes = (root / hold_evidence["path"]).read_bytes()
    job["sourceReadinessOverlay"] = pin(
        "overlay-for-publication.json",
        {
            "status": "approved",
            "sources": {
                "a": [
                    {
                        "category": "privacy",
                        "reason": "Unowned contact",
                        "evidence": hold_evidence,
                    }
                ]
            },
        },
    )
    job_pin = pin("job.json", job)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_independent_numeric_authority.py",
            "--project-root",
            str(root),
            "--job",
            job_pin["path"],
            "--publish",
        ],
    )
    merge.main()
    capsys.readouterr()
    output = root / "merged"
    config = json.loads((output / "inventory-config.json").read_bytes())
    receipt = json.loads((output / "merge-receipt.json").read_bytes())
    assert config["source_ids"] == ["a", "b", "unreviewed"]
    assert config["variants_per_source"] == 3
    assert receipt["reviewedNumericSources"] == 2 and receipt["unreviewedNumericSources"] == 1
    (relocation,) = receipt["readinessEvidenceRelocations"]
    assert relocation["input"] == hold_evidence
    assert (root / relocation["output"]["path"]).read_bytes() == source_evidence_bytes
    first = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    merge.main()
    capsys.readouterr()
    assert {
        str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()
    } == first
    (output / "numeric_certificates.json").write_text("corrupted authority")
    with pytest.raises(AtomicConflictError):
        merge.main()
    assert (output / "numeric_certificates.json").read_text() == "corrupted authority"
