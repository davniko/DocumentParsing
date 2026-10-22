"""Publish reviewed independent-source data into the synthesis template registry.

Inputs are explicit reviewed files. This does not infer missing certificates,
review sources, mutate existing templates, or certify generated descendants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_geography import load_independent_geography_certificates
from document_ocr.synthesis.independent_numeric import IndependentNumericCertificates
from document_ocr.synthesis.independent_presentation import (
    DistinctActorReview,
    SourceOccurrenceLocalityReview,
    SourceOnlyAssemblyLocalityReview,
)
from document_ocr.synthesis.independent_wording import load_certificates


def publish_json(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError(f"refusing to overwrite a different authority: {path}")
    else:
        atomic_write_json(path, value)


def prepare_authority(
    *,
    root: Path,
    numeric: Path,
    wording: Path,
    geography: Path,
    locality_reviews: Path,
    actor_reviews: Path,
    source_cases: Path,
    output: Path,
) -> dict:
    paths = {
        "numeric_certificates": numeric,
        "wording_certificates": wording,
        "geography_certificates": geography,
        "locality_owner_reviews": locality_reviews,
        "distinct_actor_reviews": actor_reviews,
    }
    source_pins = {key: sha256_file(path) for key, path in paths.items()}
    IndependentNumericCertificates.from_path(
        numeric, expected_sha256=source_pins["numeric_certificates"]
    )
    load_certificates(wording, expected_sha256=source_pins["wording_certificates"])
    geography_certificates = load_independent_geography_certificates(
        geography.read_bytes(), expected_sha256=source_pins["geography_certificates"]
    )
    payloads = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
    payloads["wording_certificates"] = {
        "version": payloads["wording_certificates"]["version"],
        "sources": payloads["wording_certificates"]["sources"],
    }

    def source_hash(sid: str) -> str:
        return sha256_file(source_cases / sid / "source.txt")

    localities = {}
    for sid, value in payloads["locality_owner_reviews"].items():
        if value["kind"] == "source_occurrence":
            expected = {"kind", "sourceSha256", "role", "expression", "keys", "sourceLocalityText"}
            if set(value) != expected or value["sourceSha256"] != source_hash(sid):
                raise ValueError(f"locality review differs from the exact source: {sid}")
            translated = SourceOccurrenceLocalityReview.model_validate_json(
                json.dumps(
                    {
                        "kind": value["kind"],
                        "source_document_id": sid,
                        "source_sha256": value["sourceSha256"],
                        "role": value["role"],
                        "expression": value["expression"],
                        "keys": value["keys"],
                        "source_locality_text": value["sourceLocalityText"],
                    }
                )
            )
        elif value["kind"] == "source_only_certificate":
            if set(value) != {"kind", "assembly"}:
                raise ValueError(f"unexpected source-only locality review fields: {sid}")
            certificate = geography_certificates.sources[sid]
            if certificate.source_sha256 != source_hash(sid):
                raise ValueError(f"source-only locality certificate source differs: {sid}")
            if value["assembly"] not in {a.key for a in certificate.assemblies}:
                raise ValueError(f"source-only locality assembly missing: {sid}")
            translated = SourceOnlyAssemblyLocalityReview(
                kind=value["kind"],
                source_document_id=sid,
                source_sha256=certificate.source_sha256,
                assembly=value["assembly"],
            )
        else:
            raise ValueError(f"unknown reviewed locality kind: {sid}")
        localities[sid] = translated.model_dump(mode="json")
    payloads["locality_owner_reviews"] = localities
    actors = {}
    for sid, value in payloads["distinct_actor_reviews"].items():
        if set(value) != {
            "sourceSha256",
            "primaryRole",
            "primaryExpression",
            "relatedKey",
            "sourceEvidence",
        } or value["sourceSha256"] != source_hash(sid):
            raise ValueError(f"distinct actor review differs from exact source: {sid}")
        actors[sid] = DistinctActorReview(
            source_document_id=sid,
            source_sha256=value["sourceSha256"],
            primary_role=value["primaryRole"],
            primary_expression=value["primaryExpression"],
            related_key=value["relatedKey"],
            source_evidence=value["sourceEvidence"],
        ).model_dump(mode="json")
    payloads["distinct_actor_reviews"] = actors
    inputs = {}
    for key, value in payloads.items():
        target = output / (key + ".json")
        publish_json(target, value)
        inputs[key] = {"path": str(target.relative_to(root)), "sha256": sha256_file(target)}
    receipt = {
        "status": "reviewed_data_relocated_not_new_source_certification",
        "inputPins": {
            str(paths[key].relative_to(root)): value for key, value in source_pins.items()
        },
        "inputs": inputs,
        "numericSourceCertificates": len(payloads["numeric_certificates"]["sources"]),
        "geographySourceCertificates": len(payloads["geography_certificates"]["sources"]),
        "wordingSourceCertificates": len(payloads["wording_certificates"]["sources"]),
        "trainingAdmissions": 0,
    }
    publish_json(output / "manifest.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    for name in (
        "numeric",
        "wording",
        "geography",
        "locality-reviews",
        "actor-reviews",
        "source-cases",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = prepare_authority(
        root=root,
        **{
            key: (root / getattr(args, key)).resolve()
            for key in (
                "numeric",
                "wording",
                "geography",
                "locality_reviews",
                "actor_reviews",
                "source_cases",
                "output",
            )
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
