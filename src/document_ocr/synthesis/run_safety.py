"""Fail-closed safety primitives for reproducible synthesis runs.

This module deliberately contains no task-specific generation policy.  It
protects the boundaries around statistical/donor inputs, globally allocated
identifiers, behavior fingerprints, and immutable run publication.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import os
import re
import stat
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from document_ocr.atomic import (
    ArtifactReadError,
    AtomicConflictError,
    atomic_publish_bytes,
    atomic_publish_json,
    json_artifact_bytes,
    read_regular_file_bytes,
)
from document_ocr.hashing import canonical_json_bytes, sha256_bytes

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_ENVIRONMENT_KEY = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|PRIVATE_KEY)(?:$|_)", re.I
)
_INTERRUPTED_TEMPORARY = re.compile(r"^\..+\.[^.]+\.tmp$")
_CHUNK_SIZE = 1024 * 1024


class SynthesisSafetyError(RuntimeError):
    """A synthesis safety contract could not be proven."""


class LeakageError(SynthesisSafetyError):
    """A source, donor, or template crosses the fitted training scope."""


class IdentifierSpaceError(SynthesisSafetyError):
    """A globally collision-free identifier allocation could not be produced."""


class FingerprintMismatchError(SynthesisSafetyError):
    """Behavior or environment bytes differ from a pinned fingerprint."""


class StagedRunError(SynthesisSafetyError):
    """A staged run is incomplete, unsafe, or conflicts with committed bytes."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class DonorInputReceipt(_StrictModel):
    document_id: NonEmptyString
    source_sha256: Sha256
    template_id: NonEmptyString
    selected_value_sha256: Sha256


class DonorProvenanceReceipt(_StrictModel):
    """Immutable proof that one donor operation used only isolated fit records."""

    schema_version: Literal[1]
    operation_id: Sha256
    purpose: NonEmptyString
    field_paths: tuple[NonEmptyString, ...] = Field(min_length=1)
    fit_split: NonEmptyString
    scope_sha256: Sha256
    source_document_id: NonEmptyString
    source_sha256: Sha256
    source_template_id: NonEmptyString
    donors: tuple[DonorInputReceipt, ...] = Field(min_length=1)


class StatisticalFitInputReceipt(_StrictModel):
    document_id: NonEmptyString
    source_sha256: Sha256
    template_id: NonEmptyString
    projected_row_sha256: Sha256


class StatisticalFitProvenanceReceipt(_StrictModel):
    """Exact, train-only rows supplied to one statistical model/view fit."""

    schema_version: Literal[1]
    fit_id: Sha256
    view_name: NonEmptyString
    purpose: NonEmptyString
    field_paths: tuple[NonEmptyString, ...] = Field(min_length=1)
    fit_split: NonEmptyString
    scope_sha256: Sha256
    inputs: tuple[StatisticalFitInputReceipt, ...] = Field(min_length=1)


class LeakageAuditReceipt(_StrictModel):
    schema_version: Literal[1]
    fit_split: NonEmptyString
    scope_sha256: Sha256
    operation_count: Annotated[int, Field(ge=0)]
    donor_count: Annotated[int, Field(ge=0)]
    distinct_source_documents: Annotated[int, Field(ge=0)]
    distinct_donor_documents: Annotated[int, Field(ge=0)]
    receipts_sha256: Sha256
    leakage_detected: Literal[False]


def _unique_sorted_strings(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    rows = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in rows):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(rows))


def _validate_sha256_mapping(
    values: Mapping[str, str], expected_keys: frozenset[str], *, label: str
) -> dict[str, str]:
    if set(values) != expected_keys:
        missing = sorted(expected_keys - set(values))
        extra = sorted(set(values) - expected_keys)
        raise ValueError(f"{label} coverage mismatch; missing={missing}, extra={extra}")
    invalid = sorted(
        key for key, value in values.items() if re.fullmatch(r"[0-9a-f]{64}", value) is None
    )
    if invalid:
        raise ValueError(f"{label} contains invalid SHA-256 values for: {invalid}")
    return dict(values)


class TrainOnlySourceScope:
    """A complete corpus scope that guards sources and empirical/model donors.

    A document is usable only when it is in ``fit_document_ids`` *and* every
    document in its template proxy is also in that split.  This makes the same
    guard suitable for direct donors and for rows used to fit an SDV view.
    """

    def __init__(
        self,
        *,
        corpus_document_ids: Sequence[str],
        fit_document_ids: Sequence[str],
        template_by_document: Mapping[str, str],
        source_sha256_by_document: Mapping[str, str],
        fit_split: str = "train",
    ) -> None:
        corpus = _unique_sorted_strings(corpus_document_ids, label="corpus_document_ids")
        fit = _unique_sorted_strings(fit_document_ids, label="fit_document_ids")
        if not fit_split.strip():
            raise ValueError("fit_split must not be empty")
        corpus_set = frozenset(corpus)
        fit_set = frozenset(fit)
        if not fit_set:
            raise ValueError("fit_document_ids must not be empty")
        if not fit_set <= corpus_set:
            raise ValueError(
                "fit_document_ids contain unknown documents: "
                + ", ".join(sorted(fit_set - corpus_set))
            )
        if set(template_by_document) != corpus_set:
            missing = sorted(corpus_set - set(template_by_document))
            extra = sorted(set(template_by_document) - corpus_set)
            raise ValueError(f"template coverage mismatch; missing={missing}, extra={extra}")
        if any(not value.strip() for value in template_by_document.values()):
            raise ValueError("template IDs must not be empty")
        hashes = _validate_sha256_mapping(
            source_sha256_by_document,
            corpus_set,
            label="source_sha256_by_document",
        )
        templates: dict[str, set[str]] = defaultdict(set)
        for document_id, template_id in template_by_document.items():
            templates[template_id].add(document_id)
        isolated_templates = frozenset(
            template_id
            for template_id, document_ids in templates.items()
            if document_ids <= fit_set
        )
        eligible_fit = frozenset(
            document_id
            for document_id in fit
            if template_by_document[document_id] in isolated_templates
        )
        scope_payload = {
            "schemaVersion": 1,
            "fitSplit": fit_split,
            "corpusDocumentIds": corpus,
            "fitDocumentIds": fit,
            "documents": [
                {
                    "documentId": document_id,
                    "sourceSha256": hashes[document_id],
                    "templateId": template_by_document[document_id],
                }
                for document_id in corpus
            ],
        }
        self._corpus = corpus_set
        self._fit = fit_set
        self._eligible_fit = eligible_fit
        self._template_by_document = MappingProxyType(dict(template_by_document))
        self._source_hashes = MappingProxyType(hashes)
        self._fit_split = fit_split
        self._scope_sha256 = sha256_bytes(canonical_json_bytes(scope_payload))

    @property
    def scope_sha256(self) -> str:
        return self._scope_sha256

    @property
    def fit_split(self) -> str:
        return self._fit_split

    @property
    def corpus_document_ids(self) -> frozenset[str]:
        return self._corpus

    @property
    def fit_document_ids(self) -> frozenset[str]:
        return self._fit

    @property
    def isolated_fit_document_ids(self) -> frozenset[str]:
        return self._eligible_fit

    def _guard_document(self, document_id: str, *, role: str) -> None:
        if document_id not in self._corpus:
            raise LeakageError(f"{role} document is outside the complete corpus: {document_id}")
        if document_id not in self._fit:
            raise LeakageError(
                f"{role} document is outside fit split {self._fit_split!r}: {document_id}"
            )
        if document_id not in self._eligible_fit:
            template_id = self._template_by_document[document_id]
            raise LeakageError(
                f"{role} document template crosses fit/non-fit boundary: "
                f"{document_id} ({template_id})"
            )

    def assert_fit_rows(self, document_ids: Iterable[str], *, purpose: str) -> tuple[str, ...]:
        """Guard every row before an empirical pool or statistical fit is built."""

        if not purpose.strip():
            raise ValueError("purpose must not be empty")
        rows = _unique_sorted_strings(document_ids, label="document_ids")
        if not rows:
            raise ValueError("document_ids must not be empty")
        for document_id in rows:
            self._guard_document(document_id, role=f"{purpose} fit")
        return rows

    def create_donor_receipt(
        self,
        *,
        source_document_id: str,
        donor_payload_by_document: Mapping[str, JsonValue],
        purpose: str,
        field_paths: Iterable[str],
    ) -> DonorProvenanceReceipt:
        """Guard and receipt one source-plus-donor operation.

        Donor payloads are hashed so that a receipt proves not just which row
        was used, but the exact selected tuple or statistical input projection.
        """

        if not purpose.strip():
            raise ValueError("purpose must not be empty")
        paths = _unique_sorted_strings(field_paths, label="field_paths")
        if not paths:
            raise ValueError("field_paths must not be empty")
        self._guard_document(source_document_id, role="source")
        donor_ids = _unique_sorted_strings(
            donor_payload_by_document,
            label="donor document IDs",
        )
        if not donor_ids:
            raise ValueError("at least one donor is required")
        if source_document_id in donor_ids:
            raise LeakageError("source document must not be its own donor")
        donors: list[DonorInputReceipt] = []
        for donor_id in donor_ids:
            self._guard_document(donor_id, role="donor")
            try:
                value_hash = sha256_bytes(canonical_json_bytes(donor_payload_by_document[donor_id]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"donor payload is not strict JSON: {donor_id}") from error
            donors.append(
                DonorInputReceipt.model_validate(
                    {
                        "document_id": donor_id,
                        "source_sha256": self._source_hashes[donor_id],
                        "template_id": self._template_by_document[donor_id],
                        "selected_value_sha256": value_hash,
                    },
                    strict=True,
                )
            )
        body = {
            "schema_version": 1,
            "purpose": purpose,
            "field_paths": paths,
            "fit_split": self._fit_split,
            "scope_sha256": self._scope_sha256,
            "source_document_id": source_document_id,
            "source_sha256": self._source_hashes[source_document_id],
            "source_template_id": self._template_by_document[source_document_id],
            "donors": tuple(row.model_dump(mode="json") for row in donors),
        }
        return DonorProvenanceReceipt.model_validate(
            {
                **body,
                "operation_id": sha256_bytes(canonical_json_bytes(body)),
            },
            strict=True,
        )

    def create_statistical_fit_receipt(
        self,
        *,
        view_name: str,
        row_payload_by_document: Mapping[str, JsonValue],
        purpose: str,
        field_paths: Iterable[str],
    ) -> StatisticalFitProvenanceReceipt:
        """Guard and receipt the exact projected rows supplied to one model fit."""

        if not view_name.strip() or not purpose.strip():
            raise ValueError("view_name and purpose must not be empty")
        paths = _unique_sorted_strings(field_paths, label="field_paths")
        if not paths:
            raise ValueError("field_paths must not be empty")
        document_ids = self.assert_fit_rows(
            row_payload_by_document,
            purpose=purpose,
        )
        inputs: list[StatisticalFitInputReceipt] = []
        for document_id in document_ids:
            try:
                row_sha256 = sha256_bytes(
                    canonical_json_bytes(row_payload_by_document[document_id])
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"statistical fit payload is not strict JSON: {document_id}"
                ) from error
            inputs.append(
                StatisticalFitInputReceipt.model_validate(
                    {
                        "document_id": document_id,
                        "source_sha256": self._source_hashes[document_id],
                        "template_id": self._template_by_document[document_id],
                        "projected_row_sha256": row_sha256,
                    },
                    strict=True,
                )
            )
        body = {
            "schema_version": 1,
            "view_name": view_name,
            "purpose": purpose,
            "field_paths": paths,
            "fit_split": self._fit_split,
            "scope_sha256": self._scope_sha256,
            "inputs": tuple(row.model_dump(mode="json") for row in inputs),
        }
        return StatisticalFitProvenanceReceipt.model_validate(
            {
                **body,
                "fit_id": sha256_bytes(canonical_json_bytes(body)),
            },
            strict=True,
        )

    def validate_statistical_fit_receipt(
        self,
        receipt: StatisticalFitProvenanceReceipt,
        *,
        row_payload_by_document: Mapping[str, JsonValue],
    ) -> None:
        if receipt.fit_split != self._fit_split or receipt.scope_sha256 != self._scope_sha256:
            raise LeakageError("statistical fit receipt belongs to a different training scope")
        expected = self.create_statistical_fit_receipt(
            view_name=receipt.view_name,
            row_payload_by_document=row_payload_by_document,
            purpose=receipt.purpose,
            field_paths=receipt.field_paths,
        )
        if expected != receipt:
            raise LeakageError("statistical fit receipt provenance does not match current inputs")

    def validate_donor_receipt(
        self,
        receipt: DonorProvenanceReceipt,
        *,
        donor_payload_by_document: Mapping[str, JsonValue],
    ) -> None:
        if receipt.fit_split != self._fit_split or receipt.scope_sha256 != self._scope_sha256:
            raise LeakageError("donor receipt belongs to a different training scope")
        expected = self.create_donor_receipt(
            source_document_id=receipt.source_document_id,
            donor_payload_by_document=donor_payload_by_document,
            purpose=receipt.purpose,
            field_paths=receipt.field_paths,
        )
        if expected != receipt:
            raise LeakageError("donor receipt provenance does not match current inputs")

    def audit_donor_receipts(
        self,
        receipts_and_payloads: Sequence[tuple[DonorProvenanceReceipt, Mapping[str, JsonValue]]],
    ) -> LeakageAuditReceipt:
        operation_ids: set[str] = set()
        source_ids: set[str] = set()
        donor_ids: set[str] = set()
        for receipt, payloads in receipts_and_payloads:
            self.validate_donor_receipt(receipt, donor_payload_by_document=payloads)
            if receipt.operation_id in operation_ids:
                raise LeakageError(f"duplicate donor operation receipt: {receipt.operation_id}")
            operation_ids.add(receipt.operation_id)
            source_ids.add(receipt.source_document_id)
            donor_ids.update(row.document_id for row in receipt.donors)
        receipt_rows = [
            receipt.model_dump(mode="json")
            for receipt, _payloads in sorted(
                receipts_and_payloads,
                key=lambda pair: pair[0].operation_id,
            )
        ]
        return LeakageAuditReceipt.model_validate(
            {
                "schema_version": 1,
                "fit_split": self._fit_split,
                "scope_sha256": self._scope_sha256,
                "operation_count": len(operation_ids),
                "donor_count": sum(len(receipt.donors) for receipt, _ in receipts_and_payloads),
                "distinct_source_documents": len(source_ids),
                "distinct_donor_documents": len(donor_ids),
                "receipts_sha256": sha256_bytes(canonical_json_bytes(receipt_rows)),
                "leakage_detected": False,
            },
            strict=True,
        )


class IdentifierReservationRequest(_StrictModel):
    request_key: NonEmptyString
    identifier_kind: NonEmptyString


@dataclass(frozen=True, slots=True)
class IdentifierCandidateContext:
    request_key: str
    identifier_kind: str
    attempt: int
    entropy: bytes


IdentifierCandidateFactory = Callable[[IdentifierCandidateContext], str]
IdentifierCanonicalizer = Callable[[str], str]


class IdentifierReservation(_StrictModel):
    request_key: NonEmptyString
    identifier_kind: NonEmptyString
    identifier: NonEmptyString
    canonical_identifier: NonEmptyString
    collision_attempts: Annotated[int, Field(ge=0)]


class IdentifierReservationBundle(_StrictModel):
    schema_version: Literal[1]
    namespace: NonEmptyString
    seed_sha256: Sha256
    canonicalizer: NonEmptyString
    real_identifier_count: Annotated[int, Field(ge=0)]
    real_unique_identifier_count: Annotated[int, Field(ge=0)]
    real_corpus_sha256: Sha256
    request_count: Annotated[int, Field(gt=0)]
    request_set_sha256: Sha256
    reservations: tuple[IdentifierReservation, ...] = Field(min_length=1)
    allocation_sha256: Sha256


def identifier_corpus_sha256(
    identifiers: Iterable[str], *, canonicalize: IdentifierCanonicalizer
) -> tuple[int, int, str]:
    values = tuple(identifiers)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("real identifiers must be non-empty strings")
    canonical = tuple(
        sorted(_canonicalize_identifier(value, canonicalize=canonicalize) for value in values)
    )
    return len(values), len(set(canonical)), sha256_bytes(canonical_json_bytes(canonical))


def _canonicalize_identifier(value: str, *, canonicalize: IdentifierCanonicalizer) -> str:
    canonical = canonicalize(value)
    repeated = canonicalize(value)
    if canonical != repeated:
        raise IdentifierSpaceError("identifier canonicalizer is stateful or nondeterministic")
    if not isinstance(canonical, str) or not canonical:
        raise ValueError("identifier canonicalizer produced an empty or non-string value")
    return canonical


def reserve_global_identifiers(
    *,
    real_identifiers: Iterable[str],
    requests: Iterable[IdentifierReservationRequest],
    seed: str | bytes,
    namespace: str,
    canonicalize: IdentifierCanonicalizer,
    canonicalizer_name: str,
    candidate_factory: IdentifierCandidateFactory,
    expected_real_identifier_count: int,
    expected_real_corpus_sha256: str,
    max_attempts: int = 1_000_000,
) -> IdentifierReservationBundle:
    """Allocate a complete request set in canonical order.

    This is intentionally a one-shot allocator.  Incremental allocation would
    make outcomes depend on batch boundaries.  Candidate factories receive only
    deterministic HMAC entropy and are called twice per attempt to reject
    stateful/nondeterministic implementations.
    """

    if not namespace.strip() or not canonicalizer_name.strip():
        raise ValueError("namespace and canonicalizer_name must not be empty")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if isinstance(seed, str):
        seed_bytes = seed.encode("utf-8")
    elif isinstance(seed, bytes):
        seed_bytes = seed
    else:
        raise TypeError("seed must be str or bytes")
    if not seed_bytes:
        raise ValueError("seed must not be empty")
    real_values = tuple(real_identifiers)
    real_count, real_unique_count, real_sha = identifier_corpus_sha256(
        real_values,
        canonicalize=canonicalize,
    )
    if real_count != expected_real_identifier_count:
        raise IdentifierSpaceError(
            "complete real identifier count mismatch: "
            f"expected {expected_real_identifier_count}, found {real_count}"
        )
    if real_sha != expected_real_corpus_sha256:
        raise IdentifierSpaceError(
            "complete real identifier SHA-256 mismatch: "
            f"expected {expected_real_corpus_sha256}, found {real_sha}"
        )
    request_rows = tuple(requests)
    if not request_rows:
        raise ValueError("identifier requests must not be empty")
    request_keys = [row.request_key for row in request_rows]
    if len(request_keys) != len(set(request_keys)):
        raise ValueError("identifier request keys must be globally unique")
    ordered = tuple(sorted(request_rows, key=lambda row: row.request_key))
    request_payload = [row.model_dump(mode="json") for row in ordered]
    request_sha = sha256_bytes(canonical_json_bytes(request_payload))
    normalized_real = {
        _canonicalize_identifier(value, canonicalize=canonicalize) for value in real_values
    }
    key = hashlib.sha256(namespace.encode("utf-8") + b"\x00" + seed_bytes).digest()
    used = set(normalized_real)
    reservations: list[IdentifierReservation] = []
    for request in ordered:
        for attempt in range(max_attempts):
            message = canonical_json_bytes(
                {
                    "attempt": attempt,
                    "identifierKind": request.identifier_kind,
                    "requestKey": request.request_key,
                }
            )
            entropy = hmac.new(key, message, hashlib.sha256).digest()
            context = IdentifierCandidateContext(
                request_key=request.request_key,
                identifier_kind=request.identifier_kind,
                attempt=attempt,
                entropy=entropy,
            )
            candidate = candidate_factory(context)
            repeated_candidate = candidate_factory(context)
            if candidate != repeated_candidate:
                raise IdentifierSpaceError(
                    "identifier candidate factory is stateful or nondeterministic for "
                    f"{request.request_key!r} attempt {attempt}"
                )
            if not isinstance(candidate, str) or not candidate.strip():
                raise IdentifierSpaceError("identifier candidate factory returned an empty value")
            canonical = _canonicalize_identifier(candidate, canonicalize=canonicalize)
            if canonical in used:
                continue
            used.add(canonical)
            reservations.append(
                IdentifierReservation.model_validate(
                    {
                        "request_key": request.request_key,
                        "identifier_kind": request.identifier_kind,
                        "identifier": candidate,
                        "canonical_identifier": canonical,
                        "collision_attempts": attempt,
                    },
                    strict=True,
                )
            )
            break
        else:
            raise IdentifierSpaceError(
                f"identifier space exhausted for {request.request_key!r} "
                f"after {max_attempts} attempts"
            )
    allocation_body = {
        "namespace": namespace,
        "seedSha256": sha256_bytes(seed_bytes),
        "canonicalizer": canonicalizer_name,
        "realCorpusSha256": real_sha,
        "requestSetSha256": request_sha,
        "reservations": [row.model_dump(mode="json") for row in reservations],
    }
    return IdentifierReservationBundle.model_validate(
        {
            "schema_version": 1,
            "namespace": namespace,
            "seed_sha256": sha256_bytes(seed_bytes),
            "canonicalizer": canonicalizer_name,
            "real_identifier_count": real_count,
            "real_unique_identifier_count": real_unique_count,
            "real_corpus_sha256": real_sha,
            "request_count": len(ordered),
            "request_set_sha256": request_sha,
            "reservations": tuple(row.model_dump(mode="json") for row in reservations),
            "allocation_sha256": sha256_bytes(canonical_json_bytes(allocation_body)),
        },
        strict=True,
    )


class FingerprintedFile(_StrictModel):
    logical_name: NonEmptyString
    role: Literal["behavior", "lock"]
    relative_path: NonEmptyString
    bytes: Annotated[int, Field(ge=0)]
    sha256: Sha256


class BehaviorEnvironmentFingerprint(_StrictModel):
    schema_version: Literal[1]
    behavior_files: tuple[FingerprintedFile, ...] = Field(min_length=1)
    lock_files: tuple[FingerprintedFile, ...] = Field(min_length=1)
    image_reference: NonEmptyString
    image_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    runtime_identity: dict[str, str]
    environment_identity: dict[str, str]
    fingerprint_sha256: Sha256


def _safe_regular_file_receipt(
    *,
    project_root: Path,
    logical_name: str,
    path: Path,
    role: Literal["behavior", "lock"],
) -> FingerprintedFile:
    if not logical_name.strip():
        raise ValueError("fingerprint logical names must not be empty")
    if path.is_absolute():
        try:
            relative_input = path.relative_to(project_root)
        except ValueError as error:
            raise ValueError(f"fingerprinted file is outside project root: {path}") from error
    else:
        relative_input = path
    relative = _safe_relative_path(relative_input.as_posix())
    unresolved = project_root / relative
    current = project_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"fingerprinted path must not contain symbolic links: {current}")
    resolved = unresolved.resolve(strict=True)
    try:
        resolved_relative = resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"fingerprinted file is outside project root: {resolved}") from error
    payload = read_regular_file_bytes(resolved)
    return FingerprintedFile.model_validate(
        {
            "logical_name": logical_name,
            "role": role,
            "relative_path": resolved_relative.as_posix(),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        },
        strict=True,
    )


def build_behavior_environment_fingerprint(
    *,
    project_root: Path,
    behavior_files: Mapping[str, Path],
    lock_files: Mapping[str, Path],
    image_reference: str,
    image_digest: str,
    environment_identity: Mapping[str, str],
) -> BehaviorEnvironmentFingerprint:
    """Fingerprint all explicitly supplied behavior, lock, image, and runtime inputs."""

    if not behavior_files or not lock_files:
        raise ValueError("at least one behavior file and one lock file are required")
    if not image_reference.strip():
        raise ValueError("image_reference must not be empty")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None:
        raise ValueError("image_digest must be an immutable sha256:<digest> identity")
    if any(not key.strip() or not value.strip() for key, value in environment_identity.items()):
        raise ValueError("environment identity keys and values must not be empty")
    unsafe_keys = sorted(key for key in environment_identity if _SECRET_ENVIRONMENT_KEY.search(key))
    if unsafe_keys:
        raise ValueError(
            "secret-bearing environment keys must not enter fingerprints: " + ", ".join(unsafe_keys)
        )
    if project_root.is_symlink():
        raise ValueError("project_root must not be a symbolic link")
    root = project_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project_root must be a directory")
    overlap = set(behavior_files) & set(lock_files)
    if overlap:
        raise ValueError(f"fingerprint logical names overlap: {sorted(overlap)}")
    behavior = tuple(
        _safe_regular_file_receipt(
            project_root=root,
            logical_name=name,
            path=path,
            role="behavior",
        )
        for name, path in sorted(behavior_files.items())
    )
    locks = tuple(
        _safe_regular_file_receipt(
            project_root=root,
            logical_name=name,
            path=path,
            role="lock",
        )
        for name, path in sorted(lock_files.items())
    )
    resolved_paths = [row.relative_path for row in (*behavior, *locks)]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("the same file must not be fingerprinted under multiple logical names")
    runtime = {
        "byteorder": sys.byteorder,
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
    }
    body = {
        "schema_version": 1,
        "behavior_files": tuple(row.model_dump(mode="json") for row in behavior),
        "lock_files": tuple(row.model_dump(mode="json") for row in locks),
        "image_reference": image_reference,
        "image_digest": image_digest,
        "runtime_identity": runtime,
        "environment_identity": dict(sorted(environment_identity.items())),
    }
    return BehaviorEnvironmentFingerprint.model_validate(
        {
            **body,
            "fingerprint_sha256": sha256_bytes(canonical_json_bytes(body)),
        },
        strict=True,
    )


def verify_behavior_environment_fingerprint(
    fingerprint: BehaviorEnvironmentFingerprint,
    *,
    project_root: Path,
) -> None:
    behavior = {row.logical_name: Path(row.relative_path) for row in fingerprint.behavior_files}
    locks = {row.logical_name: Path(row.relative_path) for row in fingerprint.lock_files}
    try:
        actual = build_behavior_environment_fingerprint(
            project_root=project_root,
            behavior_files=behavior,
            lock_files=locks,
            image_reference=fingerprint.image_reference,
            image_digest=fingerprint.image_digest,
            environment_identity=fingerprint.environment_identity,
        )
    except (ArtifactReadError, OSError, ValueError) as error:
        raise FingerprintMismatchError(
            "behavior/environment inputs cannot be revalidated"
        ) from error
    if actual != fingerprint:
        raise FingerprintMismatchError(
            f"behavior/environment fingerprint mismatch: expected "
            f"{fingerprint.fingerprint_sha256}, found {actual.fingerprint_sha256}"
        )


class StagedArtifactReceipt(_StrictModel):
    relative_path: NonEmptyString
    bytes: Annotated[int, Field(ge=0)]
    sha256: Sha256


class StagedCommitReceipt(_StrictModel):
    schema_version: Literal[1]
    run_name: NonEmptyString
    transaction_sha256: Sha256
    commit_strategy: Literal["flock_guarded_atomic_rename_v1"]
    transaction_marker_sha256: Sha256
    artifacts: tuple[StagedArtifactReceipt, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue]
    content_sha256: Sha256


@dataclass(frozen=True, slots=True)
class StagedCommitResult:
    receipt: StagedCommitReceipt
    created: bool


def _safe_relative_path(value: str) -> Path:
    path = PurePath(value)
    if (
        not value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe staged artifact path: {value!r}")
    return Path(*path.parts)


def _assert_plain_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise StagedRunError(f"{label} is not a plain directory: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_receipt(root: Path, relative: str) -> StagedArtifactReceipt:
    payload = read_regular_file_bytes(root / _safe_relative_path(relative))
    return StagedArtifactReceipt.model_validate(
        {
            "relative_path": relative,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        },
        strict=True,
    )


class StagedArtifactRun:
    """A deterministic, resumable staging directory with one atomic commit.

    All artifact paths are immutable.  ``commit`` validates the exact expected
    inventory, writes the commit receipt last, and renames the complete staging
    directory while holding a process-crash-safe advisory lock.
    """

    _TRANSACTION = "_TRANSACTION.json"
    _COMMIT = "_COMMIT.json"

    def __init__(self, *, output_parent: Path, run_name: str, transaction_sha256: str) -> None:
        if _RUN_NAME.fullmatch(run_name) is None:
            raise ValueError("run_name must be one safe path component")
        if re.fullmatch(r"[0-9a-f]{64}", transaction_sha256) is None:
            raise ValueError("transaction_sha256 must be a lowercase SHA-256")
        if output_parent.is_symlink():
            raise ValueError("output_parent must not be a symbolic link")
        output_parent.mkdir(parents=True, exist_ok=True)
        parent = output_parent.resolve(strict=True)
        _assert_plain_directory(parent, label="output parent")
        self.output_parent = parent
        self.run_name = run_name
        self.transaction_sha256 = transaction_sha256
        self.final_root = parent / run_name
        self.stage_root = parent / f".{run_name}.staging-{transaction_sha256}"
        self.lock_path = parent / f".{run_name}.commit.lock"
        if self.final_root.exists() or self.final_root.is_symlink():
            self._validate_final(expected_transaction=transaction_sha256)
            self._completed = True
            return
        self._completed = False
        try:
            self.stage_root.mkdir(mode=0o700)
        except FileExistsError:
            _assert_plain_directory(self.stage_root, label="staging root")
        marker = {
            "schemaVersion": 1,
            "runName": run_name,
            "transactionSha256": transaction_sha256,
        }
        try:
            atomic_publish_json(self.stage_root / self._TRANSACTION, marker)
        except AtomicConflictError as error:
            raise StagedRunError("staging directory belongs to a different transaction") from error

    @property
    def completed(self) -> bool:
        return self._completed

    def _target_root(self) -> Path:
        return self.final_root if self._completed else self.stage_root

    def _target(self, relative_path: str) -> Path:
        relative = _safe_relative_path(relative_path)
        if relative.as_posix() in {self._TRANSACTION, self._COMMIT}:
            raise ValueError(f"staged artifact path is reserved: {relative_path}")
        root = self._target_root()
        current = root
        for component in relative.parts[:-1]:
            current = current / component
            if current.exists() and current.is_symlink():
                raise StagedRunError(f"staged artifact parent is a symbolic link: {current}")
        return root / relative

    def publish_bytes(self, relative_path: str, payload: bytes) -> bool:
        target = self._target(relative_path)
        if self._completed:
            try:
                existing = read_regular_file_bytes(target)
            except ArtifactReadError as error:
                raise StagedRunError(
                    f"committed run has no immutable artifact: {relative_path}"
                ) from error
            if existing != payload:
                raise StagedRunError(f"committed artifact conflicts: {relative_path}")
            return False
        try:
            return atomic_publish_bytes(target, payload)
        except (ArtifactReadError, AtomicConflictError) as error:
            raise StagedRunError(f"staged artifact conflicts: {relative_path}") from error

    def publish_json(self, relative_path: str, value: Any) -> bool:
        return self.publish_bytes(relative_path, json_artifact_bytes(value))

    def recover_interrupted_temporary_files(self) -> tuple[str, ...]:
        """Explicitly remove only temp files left by an interrupted atomic publish."""

        if self._completed:
            return ()
        recovered: list[str] = []
        for directory, directory_names, filenames in os.walk(self.stage_root, followlinks=False):
            directory_path = Path(directory)
            for name in directory_names:
                if (directory_path / name).is_symlink():
                    raise StagedRunError("staging tree contains a symbolic-link directory")
            for name in filenames:
                candidate = directory_path / name
                if candidate.is_symlink():
                    raise StagedRunError("staging tree contains a symbolic-link file")
                if _INTERRUPTED_TEMPORARY.fullmatch(name):
                    if not stat.S_ISREG(candidate.lstat().st_mode):
                        raise StagedRunError("interrupted temporary artifact is not regular")
                    recovered.append(candidate.relative_to(self.stage_root).as_posix())
                    candidate.unlink()
            _fsync_directory(directory_path)
        return tuple(sorted(recovered))

    def _scan_artifacts(self) -> tuple[StagedArtifactReceipt, ...]:
        root = self._target_root()
        files: list[str] = []
        for directory, directory_names, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in directory_names:
                if (directory_path / name).is_symlink():
                    raise StagedRunError("artifact tree contains a symbolic-link directory")
            for name in filenames:
                path = directory_path / name
                if path.is_symlink():
                    raise StagedRunError("artifact tree contains a symbolic-link file")
                relative = path.relative_to(root).as_posix()
                if relative not in {self._TRANSACTION, self._COMMIT}:
                    files.append(relative)
        return tuple(_artifact_receipt(root, relative) for relative in sorted(files))

    def _read_commit(self, root: Path) -> StagedCommitReceipt:
        try:
            receipt = StagedCommitReceipt.model_validate_json(
                read_regular_file_bytes(root / self._COMMIT),
                strict=True,
            )
        except (ArtifactReadError, ValueError) as error:
            raise StagedRunError("committed run has no valid commit receipt") from error
        body = receipt.model_dump(mode="json", exclude={"content_sha256"})
        if sha256_bytes(canonical_json_bytes(body)) != receipt.content_sha256:
            raise StagedRunError("commit receipt content SHA-256 is invalid")
        return receipt

    def _validate_final(self, *, expected_transaction: str) -> StagedCommitReceipt:
        _assert_plain_directory(self.final_root, label="committed run root")
        receipt = self._read_commit(self.final_root)
        if receipt.run_name != self.run_name or receipt.transaction_sha256 != expected_transaction:
            raise StagedRunError("committed run belongs to a different transaction")
        marker_payload = read_regular_file_bytes(self.final_root / self._TRANSACTION)
        if sha256_bytes(marker_payload) != receipt.transaction_marker_sha256:
            raise StagedRunError("committed transaction marker fails its receipt")
        actual = self._scan_root_artifacts(self.final_root)
        if actual != receipt.artifacts:
            raise StagedRunError("committed artifact inventory differs from its receipt")
        return receipt

    @staticmethod
    def _scan_root_artifacts(root: Path) -> tuple[StagedArtifactReceipt, ...]:
        files: list[str] = []
        for directory, directory_names, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            if any((directory_path / name).is_symlink() for name in directory_names):
                raise StagedRunError("committed tree contains a symbolic-link directory")
            for name in filenames:
                path = directory_path / name
                if path.is_symlink():
                    raise StagedRunError("committed tree contains a symbolic-link file")
                relative = path.relative_to(root).as_posix()
                if relative not in {
                    StagedArtifactRun._TRANSACTION,
                    StagedArtifactRun._COMMIT,
                }:
                    files.append(relative)
        return tuple(_artifact_receipt(root, relative) for relative in sorted(files))

    def validate_committed_run(self) -> StagedCommitReceipt:
        if not self.final_root.exists():
            raise StagedRunError("run has not been committed")
        return self._validate_final(expected_transaction=self.transaction_sha256)

    def commit(
        self,
        *,
        expected_artifacts: Iterable[str],
        metadata: Mapping[str, JsonValue],
    ) -> StagedCommitResult:
        expected = _unique_sorted_strings(expected_artifacts, label="expected_artifacts")
        if not expected:
            raise ValueError("expected_artifacts must not be empty")
        for relative in expected:
            safe = _safe_relative_path(relative).as_posix()
            if safe in {self._TRANSACTION, self._COMMIT}:
                raise ValueError(f"expected artifact path is reserved: {safe}")
        if self._completed:
            receipt = self._validate_final(expected_transaction=self.transaction_sha256)
            if tuple(row.relative_path for row in receipt.artifacts) != expected:
                raise StagedRunError("committed run differs from expected artifact inventory")
            if receipt.metadata != dict(metadata):
                raise StagedRunError("committed run differs from expected metadata")
            return StagedCommitResult(receipt=receipt, created=False)
        actual = self._scan_artifacts()
        actual_paths = tuple(row.relative_path for row in actual)
        if actual_paths != expected:
            missing = sorted(set(expected) - set(actual_paths))
            extra = sorted(set(actual_paths) - set(expected))
            raise StagedRunError(
                f"staged artifact inventory mismatch; missing={missing}, extra={extra}"
            )
        marker_payload = read_regular_file_bytes(self.stage_root / self._TRANSACTION)
        body = {
            "schema_version": 1,
            "run_name": self.run_name,
            "transaction_sha256": self.transaction_sha256,
            "commit_strategy": "flock_guarded_atomic_rename_v1",
            "transaction_marker_sha256": sha256_bytes(marker_payload),
            "artifacts": tuple(row.model_dump(mode="json") for row in actual),
            "metadata": dict(metadata),
        }
        receipt = StagedCommitReceipt.model_validate(
            {
                **body,
                "content_sha256": sha256_bytes(canonical_json_bytes(body)),
            },
            strict=True,
        )
        atomic_publish_bytes(
            self.stage_root / self._COMMIT,
            json_artifact_bytes(receipt.model_dump(mode="json")),
        )
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            lock_descriptor = os.open(self.lock_path, flags, 0o600)
        except OSError as error:
            raise StagedRunError("cannot acquire staged-run commit lock") from error
        try:
            if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
                raise StagedRunError("staged-run commit lock is not a regular file")
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            if self.final_root.exists() or self.final_root.is_symlink():
                existing = self._validate_final(expected_transaction=self.transaction_sha256)
                if existing != receipt:
                    raise StagedRunError(
                        "concurrent committed run differs from staged bytes"
                    ) from None
                self._completed = True
                return StagedCommitResult(receipt=existing, created=False)
            _fsync_directory(self.stage_root)
            # Every writer for this run name is serialized by the advisory
            # lock above.  The second existence check is therefore the
            # no-overwrite boundary for cooperating synthesis processes, while
            # rename is the same-filesystem atomic visibility boundary.  This
            # strategy is supported by both native Linux filesystems and WSL's
            # DrvFS, unlike RENAME_NOREPLACE.
            if self.final_root.exists() or self.final_root.is_symlink():
                existing = self._validate_final(expected_transaction=self.transaction_sha256)
                if existing != receipt:
                    raise StagedRunError("concurrent committed run differs from staged bytes")
                self._completed = True
                return StagedCommitResult(receipt=existing, created=False)
            os.rename(self.stage_root, self.final_root)
            _fsync_directory(self.output_parent)
            self._completed = True
            # The inventory was byte-verified before the same-filesystem atomic
            # rename.  Re-reading every artifact here would double commit I/O;
            # later resumes and the explicit validator independently re-read
            # the complete committed tree.
            return StagedCommitResult(receipt=receipt, created=True)
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
