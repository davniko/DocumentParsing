"""Publish the audited, non-generative B/L synthesis-readiness corpus."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, json_artifact_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading import BillOfLadingAnnotation
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingDualCargoAnnotation
from document_ocr.synthesis.anchors import build_document_anchors, format_inventory
from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.config import SynthesisPreparationConfig
from document_ocr.synthesis.domain import RelationalTables, rows_for_document
from document_ocr.synthesis.support_plots import readiness_plots
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task


def _resolve_file(project_root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else project_root / candidate
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _resolve_directory(project_root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else project_root / candidate
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _resolve_recorded_artifact_file(project_root: Path, value: str, label: str) -> Path:
    """Relocate an immutable project-artifact path across host/container project roots."""

    recorded = Path(value)
    if not recorded.is_absolute():
        return _resolve_file(project_root, value, label)
    artifact_indexes = [index for index, part in enumerate(recorded.parts) if part == "artifacts"]
    if len(artifact_indexes) != 1:
        raise ValueError(
            f"absolute {label} lineage must contain one project artifacts component: {recorded}"
        )
    relative = Path(*recorded.parts[artifact_indexes[0] :])
    return _resolve_file(project_root, str(relative), label)


def _read_pinned_jsonl(
    *, project_root: Path, path_value: str, expected_sha256: str, expected_rows: int, label: str
) -> tuple[Path, list[dict[str, Any]]]:
    path = _resolve_file(project_root, path_value, label)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    rows = []
    with path.open("rb") as stream:
        for row_number, encoded in enumerate(stream, start=1):
            try:
                value = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}:{row_number}: invalid UTF-8 JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{row_number}: row must be an object")
            rows.append(value)
    if len(rows) != expected_rows:
        raise ValueError(f"{label} row count mismatch: expected {expected_rows}, found {len(rows)}")
    return path, rows


def _unwrap_lineage(row: Mapping[str, Any], document_id: str) -> Mapping[str, Any]:
    current = row
    while "sourceCorpus" not in current:
        child = current.get("sourceLineage")
        if not isinstance(child, dict) or child.get("documentId") != document_id:
            raise ValueError(f"invalid projection lineage chain: {document_id}")
        current = child
    return current


def _annotation(
    *,
    project_root: Path,
    current_root: Path,
    lineage: Mapping[str, Any],
    document_id: str,
    joined_raw_text_sha256: str,
) -> dict[str, Any]:
    source_lineage = _unwrap_lineage(lineage, document_id)
    source_corpus = source_lineage.get("sourceCorpus")
    if source_corpus == "legacy_combined487":
        path_value = source_lineage.get("sourceValidatedAnnotationPath")
        if not isinstance(path_value, str):
            raise ValueError(f"legacy annotation path is absent: {document_id}")
        path = _resolve_recorded_artifact_file(project_root, path_value, "legacy annotation")
        model_type: type[BillOfLadingAnnotation | BillOfLadingDualCargoAnnotation] = (
            BillOfLadingAnnotation
        )
    elif source_corpus == "current_main680":
        relative = source_lineage.get("sourceRecordPath")
        if not isinstance(relative, str):
            raise ValueError(f"current annotation path is absent: {document_id}")
        path = _resolve_file(current_root, relative, "current annotation")
        if current_root not in path.parents:
            raise ValueError(f"current annotation escapes its pinned root: {document_id}")
        model_type = BillOfLadingDualCargoAnnotation
    else:
        raise ValueError(f"unsupported source corpus: {source_corpus!r}")
    digest = source_lineage.get("sourceValidatedAnnotationSha256")
    if not isinstance(digest, str) or sha256_file(path) != digest:
        raise ValueError(f"annotation SHA-256 mismatch: {document_id}")
    annotation = model_type.model_validate_json(path.read_bytes(), strict=True)
    if annotation.source.documentId != document_id:
        raise ValueError(f"annotation document ID mismatch: {document_id}")
    if annotation.source.joinedRawTextSha256 != joined_raw_text_sha256:
        raise ValueError(f"annotation raw-OCR SHA-256 mismatch: {document_id}")
    return annotation.model_dump(mode="json", exclude_none=True)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _normal_path_aliases(
    *, annotation: Mapping[str, Any], package_metadata: Mapping[str, Any]
) -> dict[str, str]:
    """Retarget immutable evidence through audited package/container projections."""

    evidence_paths = {
        cast(str, row["targetPath"])
        for row in cast(Sequence[Mapping[str, Any]], annotation.get("evidence", []))
    }
    aliases: dict[str, str] = {}
    annotation_normal = annotation.get("normalLabel") or annotation.get("label")
    annotation_patch = (
        annotation_normal.get("documentPatch", {}) if isinstance(annotation_normal, dict) else {}
    )
    for index, container in enumerate(annotation_patch.get("containers") or []):
        description_path = f"documentPatch.containers[{index}].typeDescription"
        code_path = f"documentPatch.containers[{index}].typeCode"
        if (
            description_path not in evidence_paths
            and code_path in evidence_paths
            and container.get("typeDescription") is None
            and container.get("typeCode") is not None
        ):
            aliases[description_path] = code_path

    diagnoses = {
        cast(str, row["groupId"]): row
        for row in cast(Sequence[Mapping[str, Any]], package_metadata.get("groupDiagnoses", []))
    }
    source_normal = cast(Mapping[str, Any], package_metadata.get("sourceNormalTarget", {})).get(
        "documentPatch", {}
    )
    for group_index, _goods in enumerate(source_normal.get("goodsItems") or []):
        group_id = f"g{group_index + 1}"
        diagnosis = diagnoses.get(group_id)
        if diagnosis is None:
            continue
        package_rows = cast(Sequence[Mapping[str, Any]], diagnosis.get("packages", []))
        retained = set(cast(Sequence[str], diagnosis.get("retainedPackageIds", [])))
        retained_rows = [row for row in package_rows if row.get("packageId") in retained]
        source_indexes = {
            cast(str, row["packageId"]): index for index, row in enumerate(package_rows)
        }
        for target_index, retained_row in enumerate(retained_rows):
            source_index = source_indexes[cast(str, retained_row["packageId"])]
            for field in ("quantity", "type", "typeCode"):
                current = (
                    f"documentPatch.goodsItems[{group_index}].packages[{target_index}].{field}"
                )
                original = (
                    f"documentPatch.goodsItems[{group_index}].packages[{source_index}].{field}"
                )
                if original in evidence_paths and current != original:
                    aliases[current] = original
    return aliases


def _provenance_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _validate_sdv(
    metadata_value: dict[str, Any], tables: Mapping[str, list[dict[str, Any]]]
) -> str:
    try:
        from importlib.metadata import version

        import pandas as pd  # type: ignore[import-untyped]
        from sdv.metadata import Metadata  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("SDV validation requires the isolated synthesis environment") from error
    metadata = Metadata.load_from_dict(metadata_value)
    metadata.validate()
    metadata.validate_data({name: pd.DataFrame(rows) for name, rows in tables.items()})
    return version("sdv")


_FAMILY_MATCHERS: dict[str, Callable[[str], bool]] = {
    "container_identifiers": lambda path: (
        path.endswith(".containerNumber") and ".containers[" in path
    ),
    "seal_identifiers": lambda path: ".sealNumbers[" in path,
    "document_dates": lambda path: (
        path in {"documentPatch.issueDate", "documentPatch.shippedOnBoardDate"}
    ),
    "package_quantities": lambda path: ".cargoPackages[" in path and path.endswith(".quantity"),
    "cargo_masses": lambda path: (
        ".cargoGroups[" in path
        and (path.endswith(".grossWeight.value") or path.endswith(".netWeight.value"))
    ),
    "temperature_setpoints": lambda path: path.endswith(".temperatureSetpoint.value"),
    "dangerous_goods_identifiers": lambda path: (
        ".dangerousGoods[" in path and path.endswith(".unNumber")
    ),
    "hs_codes": lambda path: ".hsCodes[" in path,
    "delivery_agent_text": lambda path: ".parties.deliveryAgent." in path,
    "forwarding_agent_text": lambda path: ".parties.forwardingAgent." in path,
}
_IMPLEMENTED_GENERATORS = {
    "container_identifiers",
    "seal_identifiers",
    "document_dates",
    "package_quantities",
    "cargo_masses",
}


def _family_support(
    *,
    document_ids: Sequence[str],
    target_leaves: Mapping[str, set[str]],
    patchable_paths: Mapping[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, bool]]]:
    readiness: dict[str, dict[str, bool]] = defaultdict(dict)
    rows = []
    for family, matcher in _FAMILY_MATCHERS.items():
        with_values = 0
        fully_patchable = 0
        for document_id in document_ids:
            expected = {path for path in target_leaves[document_id] if matcher(path)}
            ready = bool(expected) and expected <= patchable_paths[document_id]
            readiness[document_id][family] = ready
            with_values += bool(expected)
            fully_patchable += ready
        rows.append(
            {
                "mutation_family": family,
                "documents_with_values": with_values,
                "fully_patchable_documents": fully_patchable,
                "generator_implemented": family in _IMPLEMENTED_GENERATORS,
            }
        )
    return rows, readiness


def _cohort_membership(feature: Mapping[str, Any], cohort: str) -> bool:
    predicates: dict[str, Callable[[Mapping[str, Any]], bool]] = {
        "all_documents": lambda _: True,
        "multi_page": lambda row: bool(row["multi_page"]),
        "multi_container": lambda row: bool(row["multi_container"]),
        "multi_cargo": lambda row: bool(row["multi_goods"]),
        "multiple_package_levels": lambda row: bool(row["multi_package_level"]),
        "container_allocations": lambda row: bool(row["has_allocations"]),
        "refrigerated": lambda row: bool(row["temperature_present"]),
        "dangerous_goods": lambda row: bool(row["dangerous_goods_present"]),
        "delivery_agent": lambda row: bool(row["delivery_agent_present"]),
        "forwarding_agent": lambda row: bool(row["forwarding_agent_present"]),
        "hs_codes": lambda row: cast(int, row["hs_code_count"]) > 0,
    }
    return predicates[cohort](feature)


def _cohort_readiness(*, cohort: str, family: Mapping[str, bool]) -> tuple[bool, bool, str]:
    any_implemented = any(family.get(name, False) for name in _IMPLEMENTED_GENERATORS)
    if cohort in {"all_documents", "multi_page"}:
        return any_implemented, any_implemented, "deterministic_existing_fields"
    requirements = {
        "multi_container": ("container_identifiers", "deterministic_ready"),
        "multi_cargo": ("package_quantities", "deterministic_ready"),
        "multiple_package_levels": ("package_quantities", "deterministic_ready"),
        "container_allocations": ("container_identifiers", "deterministic_ready"),
        "refrigerated": ("temperature_setpoints", "generator_pending"),
        "dangerous_goods": ("dangerous_goods_identifiers", "generator_pending"),
        "delivery_agent": ("delivery_agent_text", "linguistic_generation_required"),
        "forwarding_agent": ("forwarding_agent_text", "linguistic_generation_required"),
        "hs_codes": ("hs_codes", "generator_pending"),
    }
    family_name, status = requirements[cohort]
    anchor_ready = family.get(family_name, False)
    generator_ready = anchor_ready and family_name in _IMPLEMENTED_GENERATORS
    return anchor_ready, generator_ready, status


def _support_rows(
    *,
    cohorts: Sequence[str],
    features: Mapping[str, Mapping[str, Any]],
    family_readiness: Mapping[str, Mapping[str, bool]],
    template_by_document: Mapping[str, str],
    template_sizes: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort_rows = []
    template_rows = []
    for cohort in cohorts:
        members = [
            document_id
            for document_id, feature in features.items()
            if _cohort_membership(feature, cohort)
        ]
        anchor_ready = []
        generator_ready = []
        status = ""
        for document_id in members:
            anchor, generated, status = _cohort_readiness(
                cohort=cohort,
                family=family_readiness[document_id],
            )
            if anchor:
                anchor_ready.append(document_id)
            if generated:
                generator_ready.append(document_id)
        cohort_rows.append(
            {
                "cohort": cohort,
                "source_documents": len(members),
                "source_templates": len({template_by_document[row] for row in members}),
                "anchor_ready_documents": len(anchor_ready),
                "generator_ready_documents": len(generator_ready),
                "method_status": status,
            }
        )
        by_template: dict[str, list[str]] = defaultdict(list)
        for document_id in members:
            by_template[template_by_document[document_id]].append(document_id)
        for template_id, template_members in by_template.items():
            ready_count = sum(
                _cohort_readiness(
                    cohort=cohort,
                    family=family_readiness[document_id],
                )[0]
                for document_id in template_members
            )
            template_rows.append(
                {
                    "template_id": template_id,
                    "template_documents": template_sizes[template_id],
                    "cohort": cohort,
                    "cohort_documents": len(template_members),
                    "anchor_ready_documents": ready_count,
                    "anchor_ready_fraction": ready_count / len(template_members),
                }
            )
    return cohort_rows, template_rows


def _report(
    *,
    document_count: int,
    table_counts: Mapping[str, int],
    anchor_count: int,
    anchor_status: Mapping[str, int],
    format_count: int,
    cohort_rows: Sequence[Mapping[str, Any]],
    minimum_template_documents: int,
    template_rows: Sequence[Mapping[str, Any]],
) -> str:
    anchor_supported_templates = {
        cast(str, row["template_id"])
        for row in template_rows
        if cast(int, row["template_documents"]) >= minimum_template_documents
        and cast(float, row["anchor_ready_fraction"]) == 1.0
    }
    table_lines = "\n".join(f"| `{name}` | {count:,} |" for name, count in table_counts.items())
    cohort_lines = "\n".join(
        "| `{cohort}` | {source_documents:,} | {source_templates:,} | "
        "{anchor_ready_documents:,} | {generator_ready_documents:,} | `{method_status}` |".format(
            **row
        )
        for row in cohort_rows
    )
    statuses = ", ".join(f"`{name}`={count:,}" for name, count in anchor_status.items())
    anchor_supported_count = f"{len(anchor_supported_templates):,}"
    return f"""# MPCI B/L synthesis-readiness report

This is a non-generative preparation artifact for the exact **{document_count:,}-document**
training corpus. It does not contain synthetic labels or patched OCR.

## Proven foundations

- Every task target was projected into semantic tables, reconstructed exactly, and revalidated.
- The domain surface contains **{sum(table_counts.values()):,} rows** across {len(table_counts)}
  tables. Internal `gN`/`pN` identifiers and list order remain deterministic, not synthesized.
- **{anchor_count:,}** evidence-backed OCR-anchor rows produced **{format_count:,}** role-aware
  surface profiles. Location outcomes: {statuses}.
- Generic value-graph publication remains an independent lossless audit representation; these
  domain tables are the modeling representation.

## Domain tables

| Table | Rows |
|---|---:|
{table_lines}

## Cohort support

`anchor_ready` means all values required by the named initial mutation family have an exact or
excerpt-scoped unique evidence span. `generator_ready` additionally requires an implemented
semantic generator. Neither status means that a raw-OCR renderer has been implemented: this stage
deliberately publishes no synthetic records. A zero is an explicit design gate, not silent fallback.

| Cohort | Source docs | Template proxies | Anchor-ready | Generator-ready | Current method status |
|---|---:|---:|---:|---:|---|
{cohort_lines}

Across template/cohort pairs, **{anchor_supported_count}** conservative template proxies have at
least {minimum_template_documents} source documents and 100% anchor readiness for at least one
requested cohort. Full document-, template-, family-, and cohort-level tables are published under
`support/`; plots show top proxies only while the tables are complete.

## Interpretation boundary

Anchor readiness proves that exact audited evidence is locatable. It does not prove the entire
evidence string is a scalar replacement span: composite and repeated evidence still require a
typed field or bounded-block renderer. Nor does it prove an arbitrary proposal is semantically
plausible. Registry, chronology, mass, allocation, categorical-vocabulary, collision,
inverse-projection, rendering, and unchanged-leaf gates must all pass before a future synthetic
record is publishable.
"""


def prepare_synthesis_corpus(
    *, project_root: Path, config_path: Path, config: SynthesisPreparationConfig
) -> dict[str, Any]:
    started = time.perf_counter()
    task_constraints_path = _resolve_file(
        project_root, config.task_constraints.path, "task constraints"
    )
    if sha256_file(task_constraints_path) != config.task_constraints.sha256:
        raise ValueError("task-constraints SHA-256 mismatch")
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        task_constraints_path.read_bytes(), strict=True
    )
    task = get_training_task(config.task).bind_constraints(constraints)

    source_path, source_rows = _read_pinned_jsonl(
        project_root=project_root,
        path_value=config.source.file.path,
        expected_sha256=config.source.file.sha256,
        expected_rows=config.source.file.records,
        label="synthesis source",
    )
    lineage_path, lineage_rows = _read_pinned_jsonl(
        project_root=project_root,
        path_value=config.sidecars.lineage.path,
        expected_sha256=config.sidecars.lineage.sha256,
        expected_rows=config.sidecars.lineage.records,
        label="lineage sidecar",
    )
    category_path, category_rows = _read_pinned_jsonl(
        project_root=project_root,
        path_value=config.sidecars.category_metadata.path,
        expected_sha256=config.sidecars.category_metadata.sha256,
        expected_rows=config.sidecars.category_metadata.records,
        label="category sidecar",
    )
    package_metadata_path, package_metadata_rows = _read_pinned_jsonl(
        project_root=project_root,
        path_value=config.sidecars.package_metadata.path,
        expected_sha256=config.sidecars.package_metadata.sha256,
        expected_rows=config.sidecars.package_metadata.records,
        label="package hierarchy sidecar",
    )
    feature_path, feature_rows = _read_pinned_jsonl(
        project_root=project_root,
        path_value=config.sidecars.document_features.path,
        expected_sha256=config.sidecars.document_features.sha256,
        expected_rows=config.sidecars.document_features.records,
        label="EDA document features",
    )
    template_path, template_rows = _read_pinned_jsonl(
        project_root=project_root,
        path_value=config.sidecars.template_groups.path,
        expected_sha256=config.sidecars.template_groups.sha256,
        expected_rows=config.sidecars.template_groups.records,
        label="EDA template groups",
    )
    annotation_root = _resolve_directory(
        project_root, config.sidecars.current_annotations.path, "current annotation root"
    )
    annotation_manifest = _resolve_file(annotation_root, "manifest.json", "annotation manifest")
    if sha256_file(annotation_manifest) != config.sidecars.current_annotations.manifest_sha256:
        raise ValueError("current annotation manifest SHA-256 mismatch")

    fields = config.source.fields
    assert fields.input_sha256 is not None
    source_by_id = {cast(str, row[fields.document_id]): row for row in source_rows}
    lineage_by_id = {cast(str, row["documentId"]): row for row in lineage_rows}
    category_by_id = {cast(str, row["documentId"]): row for row in category_rows}
    package_metadata_by_id = {cast(str, row["documentId"]): row for row in package_metadata_rows}
    features = {cast(str, row["document_id"]): row for row in feature_rows}
    document_ids = list(source_by_id)
    expected_ids = set(document_ids)
    for name, mapping in (
        ("lineage", lineage_by_id),
        ("category metadata", category_by_id),
        ("package hierarchy metadata", package_metadata_by_id),
        ("EDA features", features),
    ):
        if len(mapping) != len(expected_ids) or set(mapping) != expected_ids:
            raise ValueError(f"{name} document set differs from synthesis source")

    template_by_document: dict[str, str] = {}
    template_sizes: dict[str, int] = {}
    for row in template_rows:
        template_id = cast(str, row["template_id"])
        members = cast(list[str], row["member_document_ids"])
        template_sizes[template_id] = cast(int, row["document_count"])
        if len(members) != template_sizes[template_id]:
            raise ValueError(f"template member count differs: {template_id}")
        for document_id in members:
            if document_id in template_by_document:
                raise ValueError(f"document belongs to multiple templates: {document_id}")
            template_by_document[document_id] = template_id
    if set(template_by_document) != expected_ids:
        raise ValueError("template inventory does not cover the synthesis source exactly")

    domain = RelationalTables({name: [] for name in ADAPTER.table_order})
    anchors: list[dict[str, Any]] = []
    anchor_documents: list[dict[str, Any]] = []
    target_leaves: dict[str, set[str]] = {}
    patchable_paths: dict[str, set[str]] = defaultdict(set)
    from document_ocr.synthesis.anchors import leaf_items

    for row_index, document_id in enumerate(document_ids):
        source = source_by_id[document_id]
        joined = source.get(fields.input_text)
        digest = source.get(fields.input_sha256)
        if not isinstance(joined, str) or not joined:
            raise ValueError(f"empty OCR input: {document_id}")
        if digest != hashlib.sha256(joined.encode()).hexdigest():
            raise ValueError(f"OCR input SHA-256 mismatch: {document_id}")
        target_value = source.get(fields.target)
        normal_target = source.get("normalTarget")
        if not isinstance(target_value, dict) or not isinstance(normal_target, dict):
            raise ValueError(f"target views are malformed: {document_id}")
        canonical = task.canonicalize(target_value)
        projection = ADAPTER.project(
            document_id=document_id, source_row_index=row_index, target=canonical
        )
        domain.extend(projection)
        rebuilt = ADAPTER.reconstruct(
            document_id=document_id,
            tables=rows_for_document(projection.rows, document_id),
        )
        if rebuilt != canonical or task.canonicalize(rebuilt) != canonical:
            raise RuntimeError(f"domain inverse projection failed: {document_id}")
        annotation = _annotation(
            project_root=project_root,
            current_root=annotation_root,
            lineage=lineage_by_id[document_id],
            document_id=document_id,
            joined_raw_text_sha256=cast(str, digest),
        )
        document_anchors, anchor_summary = build_document_anchors(
            document_id=document_id,
            joined_raw_text=joined,
            target=canonical,
            normal_target=normal_target,
            annotation=annotation,
            normal_path_aliases=_normal_path_aliases(
                annotation=annotation,
                package_metadata=package_metadata_by_id[document_id],
            ),
            patchable_locations=frozenset(config.anchors.patchable_locations),
        )
        anchors.extend(document_anchors)
        anchor_documents.append(anchor_summary)
        target_leaves[document_id] = {path for path, _ in leaf_items(canonical)}
        for anchor in document_anchors:
            if anchor["patchable"]:
                patchable_paths[document_id].add(cast(str, anchor["relation_target_path"]))

    if config.anchors.require_all_source_fact_evidence:
        uncovered = [
            row["document_id"]
            for row in anchor_documents
            if row["evidence_covered_fact_leaves"] != row["expected_source_fact_leaves"]
        ]
        if uncovered:
            raise RuntimeError(
                f"{len(uncovered)} documents lack complete source-fact evidence; "
                f"first={uncovered[0]}"
            )

    formats = format_inventory(anchors)
    family_rows, family_readiness = _family_support(
        document_ids=document_ids,
        target_leaves=target_leaves,
        patchable_paths=patchable_paths,
    )
    cohort_rows, template_cohort_rows = _support_rows(
        cohorts=config.support.cohorts,
        features=features,
        family_readiness=family_readiness,
        template_by_document=template_by_document,
        template_sizes=template_sizes,
    )
    metadata = ADAPTER.sdv_metadata()
    sdv_version = _validate_sdv(metadata, domain.rows) if config.sdv.validate_with_sdv else None
    table_counts = {name: len(domain.rows[name]) for name in ADAPTER.table_order}
    anchor_status = Counter(cast(str, row["location_status"]) for row in anchors)
    report = _report(
        document_count=len(document_ids),
        table_counts=table_counts,
        anchor_count=len(anchors),
        anchor_status=dict(sorted(anchor_status.items())),
        format_count=len(formats),
        cohort_rows=cohort_rows,
        minimum_template_documents=config.support.minimum_template_documents,
        template_rows=template_cohort_rows,
    )
    plots = readiness_plots(
        table_counts=table_counts,
        anchor_status_counts=anchor_status,
        document_anchor_rows=anchor_documents,
        cohort_rows=cohort_rows,
        family_rows=family_rows,
        template_cohort_rows=template_cohort_rows,
    )

    output_root = Path(config.run.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    run_dir = output_root.resolve() / config.run.run_id
    payloads: dict[str, bytes] = {
        **{f"tables/{name}.jsonl": _jsonl(domain.rows[name]) for name in ADAPTER.table_order},
        "anchors/ocr-anchors.jsonl": _jsonl(anchors),
        "anchors/document-coverage.jsonl": _jsonl(anchor_documents),
        "anchors/format-profiles.jsonl": _jsonl(formats),
        "support/mutation-families.jsonl": _jsonl(family_rows),
        "support/cohorts.jsonl": _jsonl(cohort_rows),
        "support/template-cohorts.jsonl": _jsonl(template_cohort_rows),
        "sdv-metadata.json": json_artifact_bytes(metadata),
        "REPORT.md": report.encode(),
        **{f"plots/{name}": payload for name, payload in plots.items()},
    }
    for relative, payload in sorted(payloads.items()):
        atomic_publish_bytes(run_dir / relative, payload)
    files = {
        relative: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for relative, payload in sorted(payloads.items())
    }
    manifest = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "task": config.task,
        "source": {
            "path": _provenance_path(source_path, project_root),
            "sha256": config.source.file.sha256,
            "records": len(document_ids),
        },
        "sidecars": {
            "lineage": {
                "path": _provenance_path(lineage_path, project_root),
                "sha256": sha256_file(lineage_path),
            },
            "categoryMetadata": {
                "path": _provenance_path(category_path, project_root),
                "sha256": sha256_file(category_path),
            },
            "packageMetadata": {
                "path": _provenance_path(package_metadata_path, project_root),
                "sha256": sha256_file(package_metadata_path),
            },
            "documentFeatures": {
                "path": _provenance_path(feature_path, project_root),
                "sha256": sha256_file(feature_path),
            },
            "templateGroups": {
                "path": _provenance_path(template_path, project_root),
                "sha256": sha256_file(template_path),
            },
            "currentAnnotationManifestSha256": config.sidecars.current_annotations.manifest_sha256,
        },
        "domainProjection": {
            "adapter": ADAPTER.task,
            "roundTripValidatedRecords": len(document_ids),
            "tableRows": table_counts,
        },
        "anchors": {
            "rows": len(anchors),
            "formatProfiles": len(formats),
            "locationStatus": dict(sorted(anchor_status.items())),
        },
        "generatorContracts": config.generators.model_dump(mode="json"),
        "sdvValidation": {
            "enabled": config.sdv.validate_with_sdv,
            "version": sdv_version,
        },
        "config": {
            "path": _provenance_path(config_path.resolve(), project_root),
            "sha256": sha256_file(config_path),
        },
        "files": files,
    }
    atomic_publish_json(run_dir / "manifest.json", manifest)
    return {
        **manifest,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
