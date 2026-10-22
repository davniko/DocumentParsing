"""Merge pinned reviewed numeric catalogs without inventing source coverage.

The explicit job supplies catalog pins, the admitted manifest, a base config,
and the root-approved nonnumeric readiness overlay. Publication is immutable
and never changes an existing catalog, dataset or default configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    json_artifact_bytes,
    read_regular_file_bytes,
)
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_scenarios import FilePin, SourceCapabilities
from document_ocr.synthesis.independent_numeric import (
    IndependentNumericCertificates,
    SourceReadinessHold,
)
from document_ocr.synthesis.independent_preparation import (
    _source_pins,
    _SourceOwnershipCampaign,
    _verify_pins,
)

_PRESENTATION_FIELDS = frozenset(
    {
        "identity_count",
        "allowed_equipment_pairs",
        "origin_country",
        "destination_country",
        "required_private_measures",
        "equipment_mix",
        "party_sides",
        "party_locality_groups",
        "location_sides",
        "route_topology",
        "goods_origin_identifiers",
        "require_contact_support",
    }
)
_FAMILIES = {
    "ambient": "ambient",
    "frozen": "frozen",
    "chilled": "chilled",
    "vehicle": "ambient",
    "dg_vehicle": "dg_chemical",
    "dg_chemical": "dg_chemical",
}
_PIN_FIELDS = frozenset({"sourceContractSha256", "sourcePublicTargetSha256"})


def numeric_digest(value):
    """The numeric certificate v2 semantic hash uses ASCII JSON escaping."""
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def _read_pin(root, value, *, load=json.loads):
    pin = FilePin.model_validate(value)
    relative = Path(pin.path)
    if not pin.path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("authority input pins must be project-relative")
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("authority input pin resolves outside the project")
    payload = read_regular_file_bytes(path)
    if hashlib.sha256(payload).hexdigest() != pin.sha256:
        raise ValueError("authority input pin mismatch: " + pin.path)
    return pin, path, load(payload)


def _review_for_source(value, source_id):
    rows = value.get("sources", [value])
    if not isinstance(rows, list):
        raise ValueError("complete-read evidence must contain authored source rows")
    found = [row for row in rows if row.get("sourceDocumentId") == source_id]
    if len(found) != 1 or found[0].get("completeRead") is not True:
        raise ValueError("missing unique complete-read evidence: " + source_id)


def _check_source_basis(blueprint, sid, certificate):
    # Certificates pin the live source compiler's effective contract, including
    # lexical/auxiliary owners, not the historical case contract wrapper.
    if blueprint.document_id != sid:
        raise ValueError("compiled source identity differs from certificate")
    source = blueprint.source.encode()
    target = blueprint.target
    contract = blueprint.contract.model_dump(mode="json")
    expected = {
        "sourceSha256": numeric_digest(source),
        "sourceContractSha256": numeric_digest(contract),
        "sourcePublicTargetSha256": numeric_digest(target),
    }
    for field, actual in expected.items():
        if certificate[field] != actual:
            raise ValueError(f"current source certificate pin mismatch: {sid}/{field}")
    patch = target["documentPatch"]
    mask = [
        "packageQuantity" in row
        for goods in patch.get("goodsItemDetails", [])
        for row in goods.get("splitGoodsPlacement", [])
    ]
    if (
        mask != certificate["sourceAllocationPresence"]
        or len(patch.get("containerInformation", [])) != certificate["sourceContainerCount"]
    ):
        raise ValueError("certificate changes source public visibility: " + sid)
    for region in (*certificate["regions"], *certificate["wordingOwners"]):
        if source[region["byte_start"] : region["byte_end"]] != region["source_text"].encode():
            raise ValueError("certificate ownership quote mismatch: " + sid)


def merge_catalogs(root: Path, job: dict) -> tuple[dict, dict]:
    """Construct without publishing; the source compiler may initialize its output directory."""
    root = root.resolve()
    required = {
        "catalogInputs",
        "registryManifest",
        "sourceCases",
        "sourceReadinessOverlay",
        "baseConfig",
        "inventoryOutput",
        "output",
    }
    if not required <= job.keys():
        raise ValueError("merge job lacks required pinned inputs")
    if not isinstance(job["catalogInputs"], list) or not job["catalogInputs"]:
        raise ValueError("merge requires a nonempty explicit catalog input list")
    snapshots = {}

    def read(value, *, load=json.loads):
        pin = FilePin.model_validate(value)
        if pin.path in snapshots:
            previous = snapshots[pin.path]
            if previous[0].sha256 != pin.sha256:
                raise ValueError("one input path has contradictory pins: " + pin.path)
            return previous
        loaded = _read_pin(root, value, load=load)
        snapshots[pin.path] = loaded
        return loaded

    _, _, manifest = read(job["registryManifest"])
    admitted = set(manifest["cases"])
    sources, origins, catalogs, evidence_pins = {}, {}, [], {}
    for value in job["catalogInputs"]:
        pin, path, payload = read(value)
        authority = IndependentNumericCertificates.from_path(path, expected_sha256=pin.sha256)
        current = authority.certificates()
        if set(current) - admitted:
            raise ValueError("numeric input contains sources outside admitted registry")
        for sid in payload["selectedSources"]:
            if sid in sources:
                raise ValueError("duplicate numeric source across input catalogs: " + sid)
            certificate = current[sid]
            evidence = certificate.get("completeSourceReviewEvidence")
            if evidence is None:
                raise ValueError("numeric certificate lacks complete-source evidence: " + sid)
            evidence_pin, _, review = read(evidence)
            _review_for_source(review, sid)
            evidence_pins[evidence_pin.path] = evidence_pin.sha256
            sources[sid], origins[sid] = certificate, pin.path
        catalogs.append(
            {
                "pin": value,
                "metadata": {
                    key: item
                    for key, item in payload.items()
                    if key not in {"sources", "selectedSources"}
                },
                "selectedSources": payload["selectedSources"],
            }
        )
    originals = copy.deepcopy(sources)
    amendments = {}
    if job.get("pinAmendments") is not None:
        _, _, approval = read(job["pinAmendments"])
        if approval.get("status") != "approved" or not approval.get("reason", "").strip():
            raise ValueError("numeric pin amendments require explicit approval and reason")
        amendments = approval["sources"]
        for sid, fields in amendments.items():
            if sid not in sources or not fields or not set(fields) <= _PIN_FIELDS:
                raise ValueError("numeric pin amendment exceeds approved semantic-pin scope")
            for field, change in fields.items():
                if set(change) != {"before", "after"} or sources[sid][field] != change["before"]:
                    raise ValueError("numeric pin amendment before witness differs")
                if change["before"] == change["after"]:
                    raise ValueError("numeric pin amendment is a no-op")
                sources[sid][field] = change["after"]
    _, _, overlay = read(job["sourceReadinessOverlay"])
    if overlay.get("status") != "approved" or not isinstance(overlay.get("sources"), dict):
        raise ValueError("nonnumeric readiness overlay requires explicit root approval")
    additions = {}
    relocations = {}
    output = (root / job["output"]).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("merged authority output must be a specific project directory")
    for sid, holds in overlay["sources"].items():
        if sid not in sources or not isinstance(holds, list) or not holds:
            raise ValueError("readiness overlay must add explicit holds to reviewed sources")
        if sources[sid].get("sourceReadinessHolds"):
            raise ValueError("readiness overlay would replace prior recorded source holds")
        relocated = []
        for hold in holds:
            parsed = SourceReadinessHold.model_validate(hold)
            evidence_pin, _, _ = read(parsed.evidence.model_dump(exclude_none=True))
            evidence_pins[evidence_pin.path] = evidence_pin.sha256
            relocated_pin = {
                "path": str(
                    (
                        output / "source-readiness-evidence" / (evidence_pin.sha256 + ".json")
                    ).relative_to(root)
                ),
                "sha256": evidence_pin.sha256,
            }
            relocations[evidence_pin.sha256] = {
                "input": parsed.evidence.model_dump(exclude_none=True),
                "output": relocated_pin,
            }
            relocated.append({**copy.deepcopy(hold), "evidence": relocated_pin})
        sources[sid]["sourceReadinessHolds"] = relocated
        additions[sid] = {"sourceReadinessHolds": copy.deepcopy(relocated)}
    source_cases = (root / job["sourceCases"]).resolve()
    if not source_cases.is_relative_to(root) or source_cases == root:
        raise ValueError("sourceCases must be a specific project directory")
    _, _, base = read(job["baseConfig"], load=yaml.safe_load)
    if (root / base["source_contracts"]).resolve() != source_cases:
        raise ValueError("merge source cases differ from the configured compiler authority")
    source_config = inventory_config(
        base, manifest, job["catalogInputs"][0], output=job["inventoryOutput"]
    )
    geography_pin = job.get("geographyCertificates")
    if geography_pin is not None:
        read(geography_pin)
        source_config["independent"]["geography_certificates"] = geography_pin
    compiler = _SourceOwnershipCampaign(root, source_config)
    compiler_pins = _source_pins(root, source_config, compiler.source_ids)
    for sid, certificate in sources.items():
        _check_source_basis(compiler.blueprint(sid), sid, certificate)
    _verify_pins(compiler_pins)
    # Evidence is parsed once per unique file, then all input bytes are checked
    # again at the barrier. Cached reviews cannot approve a mid-merge change.
    for pin, path, _ in snapshots.values():
        if hashlib.sha256(read_regular_file_bytes(path)).hexdigest() != pin.sha256:
            raise ValueError("authority input changed during merge: " + pin.path)
    for sid, original in originals.items():
        expected = copy.deepcopy(original)
        for field, change in amendments.get(sid, {}).items():
            expected[field] = change["after"]
        expected.update(additions.get(sid, {}))
        if sources[sid] != expected:
            raise ValueError("merge changed unapproved source certificate facts: " + sid)
    catalog = {
        "version": 2,
        "selectedSources": list(sources),
        "sources": sources,
        "mergedInputCatalogs": catalogs,
        "rootApprovedSourceReadinessOverlay": job["sourceReadinessOverlay"],
        "approvedPinAmendments": job.get("pinAmendments"),
    }
    receipt = {
        "status": "reviewed_catalog_merge_not_full_synthesis_certification",
        "inputCatalogs": job["catalogInputs"],
        "registryManifest": job["registryManifest"],
        "sourceReadinessOverlay": job["sourceReadinessOverlay"],
        "baseConfig": job["baseConfig"],
        "geographyCertificates": geography_pin,
        "approvedPinAmendments": job.get("pinAmendments"),
        "changedSemanticPins": amendments,
        "addedFieldsBySource": additions,
        "allOtherCertificateFieldsUnchanged": True,
        "sourceOrigins": origins,
        "reviewEvidencePins": evidence_pins,
        "readinessEvidenceRelocations": list(relocations.values()),
        "sourceCompilerInputPins": compiler_pins,
        "reviewedNumericSources": len(sources),
        "numericHeldSources": sum(bool(c.get("holdReasons")) for c in sources.values()),
        "separateSourceReadinessHeldSources": len(additions),
        "unreviewedNumericSources": len(admitted - sources.keys()),
        "admittedSourceCount": len(admitted),
        "nativeCalls": 0,
        "trainingAdmissions": 0,
    }
    return catalog, receipt


def inventory_config(
    base: dict,
    manifest: dict,
    numeric_pin: dict,
    *,
    output: str,
    variants: int = 3,
    geography_pin: dict | None = None,
) -> dict:
    """Retain source presentation constraints, never old commodity/load profiles."""
    if type(variants) is not int or variants < 1:
        raise ValueError("inventory variants must be a positive integer")
    result = copy.deepcopy(base)
    ids = sorted(manifest["cases"])
    if manifest["sourceCount"] != len(ids) or not ids:
        raise ValueError("admitted source manifest cardinality differs")
    result.update(
        source_ids=ids, output=output, variants_per_source=variants, sampling_mode="independent"
    )
    result.pop("template_sampling", None)
    capabilities = {}
    for sid in ids:
        original = manifest["cases"][sid]["capability"]
        cap = {
            key: copy.deepcopy(value)
            for key, value in original.items()
            if key in _PRESENTATION_FIELDS
        }
        cap["family"] = _FAMILIES[original["family"]]
        # Omission in admitted metadata has the existing capability contract's
        # explicit default, not a guessed cargo quantity or source ratio.
        cap.setdefault("identity_count", SourceCapabilities.model_fields["identity_count"].default)
        SourceCapabilities.model_validate(cap)
        capabilities[sid] = cap
    result["capabilities"] = capabilities
    for field in (
        "load_profiles",
        "profile_weights",
        "package_weights",
        "equipment_weights",
        "observed_hs_overrides",
        "ambient_hs_chapters",
        "equipment_inventory",
        "quantity_scale_min",
        "quantity_scale_max",
        "scale_steps",
        "maximum_candidates",
    ):
        result["sampling"].pop(field, None)
    for field in ("requests", "private_plans"):
        result["independent"].pop(field, None)
    result["independent"]["numeric_certificates"] = copy.deepcopy(numeric_pin)
    if geography_pin is not None:
        FilePin.model_validate(geography_pin)
        result["independent"]["geography_certificates"] = copy.deepcopy(geography_pin)
    result["independent_preparation"] = {
        "package_targets": {},
        "mixed_source_ids": [],
        "mixed_probability": 0,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    job = json.loads(read_regular_file_bytes(root / args.job))
    catalog, receipt = merge_catalogs(root, job)
    if args.publish:
        output = (root / job["output"]).resolve()
        if output == root or not output.is_relative_to(root):
            raise ValueError("merged authority output must be a specific project directory")
        path = output / "numeric_certificates.json"
        numeric_pin = {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(json_artifact_bytes(catalog)).hexdigest(),
        }
        _, _, manifest = _read_pin(root, job["registryManifest"])
        base_pin = FilePin.model_validate(job["baseConfig"])
        base_path = root / base_pin.path
        if sha256_file(base_path) != base_pin.sha256:
            raise ValueError("inventory base config pin mismatch")
        base = yaml.safe_load(read_regular_file_bytes(base_path))
        config = inventory_config(
            base,
            manifest,
            numeric_pin,
            output=job["inventoryOutput"],
            geography_pin=job.get("geographyCertificates"),
        )
        config_path = output / "inventory-config.json"
        # Finish validation before the first immutable output publication.
        _verify_pins(receipt["sourceCompilerInputPins"])
        for relocation in receipt["readinessEvidenceRelocations"]:
            pin, source_path, _ = _read_pin(root, relocation["input"])
            destination = relocation["output"]
            if pin.sha256 != destination["sha256"]:
                raise ValueError("readiness evidence relocation changed bytes")
            atomic_publish_bytes(root / destination["path"], read_regular_file_bytes(source_path))
        atomic_publish_json(path, catalog)
        atomic_publish_json(config_path, config)
        receipt.update(
            numericAuthority=numeric_pin,
            inventoryConfig={
                "path": str(config_path.relative_to(root)),
                "sha256": sha256_file(config_path),
            },
            inventoryFamilyCounts=dict(
                Counter(c["family"] for c in config["capabilities"].values())
            ),
        )
        atomic_publish_json(output / "merge-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
