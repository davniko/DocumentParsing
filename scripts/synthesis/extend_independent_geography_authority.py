"""Extend a pinned independent-geography authority from reviewed evidence.

The extension is additive and immutable.  It recompiles the current source
blueprint through the production ownership path before admitting the supplied
certificate; it never infers postal ownership from OCR text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from document_ocr.atomic import atomic_publish_json, json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.country_presentations import Registry
from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_auxiliary import (
    augment_auxiliary_blueprint,
    load_auxiliary_contract,
)
from document_ocr.synthesis.curated_campaign import related_actor_contexts
from document_ocr.synthesis.curated_countries import bind_postal_country_alias_owners
from document_ocr.synthesis.curated_ownership import load_owned_blueprint
from document_ocr.synthesis.curated_templates import SamplingBlueprint
from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate,
    load_independent_geography_certificates,
)


def _require_pin(root: Path, value: Mapping[str, Any]) -> Path:
    if set(value) != {"path", "sha256"}:
        raise ValueError("review evidence pins require exactly path and sha256")
    relative = Path(value["path"])
    if not value["path"] or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("review evidence paths must be project-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != value["sha256"]:
        raise ValueError("review evidence pin mismatch: " + value["path"])
    return path


def _compile_blueprint(
    root: Path, config: Mapping[str, Any], source_id: str
) -> SamplingBlueprint:
    dataset = root / config["dataset"] / "train.jsonl"
    matches = [
        row
        for row in map(json.loads, read_regular_file_bytes(dataset).splitlines())
        if row["documentId"] == source_id
    ]
    if len(matches) != 1:
        raise ValueError("reviewed source must occur exactly once in current training data")
    case = root / config["source_contracts"] / source_id
    contract = SourceContract.model_validate(
        json.loads(read_regular_file_bytes(case / "contract.json"))["contract"]
    )
    blueprint = load_owned_blueprint(
        matches[0],
        json.loads(read_regular_file_bytes(case / "template.json")),
        contract,
        root / config["ownership"],
    )
    countries = Registry.load(
        root,
        manifest_path=config["country_registry_manifest"],
        expected_sha256=config["country_registry_sha256"],
    )
    return augment_auxiliary_blueprint(
        bind_postal_country_alias_owners(blueprint, countries),
        load_auxiliary_contract(root / config["auxiliary"]),
    )


def extend_authority(
    root: Path, config_path: Path, review_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and construct one additive authority extension without writing it."""

    root = root.resolve()
    review_bytes = read_regular_file_bytes(review_path)
    review = json.loads(review_bytes)
    expected = {
        "schemaVersion",
        "status",
        "sourceDocumentId",
        "reviewBasis",
        "compiledBlueprintPins",
        "assembly",
        "reviewConclusions",
    }
    if set(review) != expected or review["status"] != "approved_complete_source_review_extension":
        raise ValueError("geography extension requires one exact approved review record")
    basis = review["reviewBasis"]
    if set(basis) != {"completeSourceReview", "legacyClosure", "priorGeographyAuthority"}:
        raise ValueError("geography extension review basis is incomplete")
    for value in basis.values():
        _require_pin(root, value)
    base_path = _require_pin(root, basis["priorGeographyAuthority"])
    base_bytes = read_regular_file_bytes(base_path)
    base = load_independent_geography_certificates(
        base_bytes, expected_sha256=basis["priorGeographyAuthority"]["sha256"]
    )
    config = yaml.safe_load(read_regular_file_bytes(config_path))
    source_id = review["sourceDocumentId"]
    if source_id in base.sources:
        raise ValueError("geography extension cannot replace an existing source certificate")
    blueprint = _compile_blueprint(root, config, source_id)
    pins = review["compiledBlueprintPins"]
    actual_pins = {
        "sourceSha256": digest(blueprint.source.encode()),
        "effectiveContractSha256": digest(blueprint.contract.model_dump(mode="json")),
        "ownershipSha256": digest(dict(blueprint.ownership_data)),
    }
    if pins != actual_pins:
        raise ValueError("geography extension differs from the current compiled blueprint")
    related = related_actor_contexts(blueprint)
    assembly = review["assembly"]
    if set(assembly["postal_keys"] + assembly["identity_keys"]) != set(related):
        raise ValueError("geography extension does not cover the exact related-actor keys")
    for key in assembly["postal_keys"]:
        if related[key] != related[assembly["identity_keys"][0]]:
            raise ValueError("geography assembly joins unrelated actors")
    sources = {
        key: value.model_dump(mode="json") for key, value in base.sources.items()
    }
    sources[source_id] = {
        "source_sha256": actual_pins["sourceSha256"],
        "effective_contract_sha256": actual_pins["effectiveContractSha256"],
        "ownership_sha256": actual_pins["ownershipSha256"],
        "assemblies": [assembly],
    }
    authority = {
        "schemaVersion": 1,
        "inventorySha256": hashlib.sha256(review_bytes).hexdigest(),
        "sources": sources,
    }
    payload = json_artifact_bytes(authority)
    loaded = load_independent_geography_certificates(
        payload, expected_sha256=hashlib.sha256(payload).hexdigest()
    )
    gate = IndependentGeographyGate(loaded)
    gate.bind_blueprint(blueprint, blueprint)
    receipt = {
        "status": "reviewed_additive_geography_authority",
        "baseAuthority": basis["priorGeographyAuthority"],
        "reviewEvidence": {
            "path": str(review_path.relative_to(root)),
            "sha256": hashlib.sha256(review_bytes).hexdigest(),
        },
        "addedSources": [source_id],
        "priorSourceCount": len(base.sources),
        "resultSourceCount": len(loaded.sources),
        "compiledBlueprintPins": actual_pins,
        "publicTargetChanges": 0,
        "nativeCalls": 0,
    }
    return authority, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = (root / args.config).resolve()
    review = (root / args.review).resolve()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("geography extension output must be a specific project directory")
    authority, receipt = extend_authority(root, config, review)
    if args.publish:
        path = output / "geography_certificates.json"
        atomic_publish_json(path, authority)
        receipt["authority"] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        atomic_publish_json(output / "extension-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
