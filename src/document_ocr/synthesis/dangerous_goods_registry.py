"""Licensed, atomic dangerous-goods regulatory tuples for synthesis.

This module contains no bundled regulatory data. Providers must present exact
source and permission receipts before their normalized rows can be compiled.
Production sampling is enabled only for a maritime-authoritative IMDG source
whose recorded terms explicitly permit automated processing, compiled
artifacts, and synthetic generation.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import ArtifactReadError, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, stable_id
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
HttpsUrl = Annotated[str, StringConstraints(pattern=r"^https://[^\s]+$", max_length=2048)]
NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
UnNumber = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")]
EntryId = Annotated[str, StringConstraints(pattern=r"^dg_[0-9a-f]{64}$")]
HazardClassDivision = Annotated[
    str,
    StringConstraints(pattern=(r"^(?:1\.[1-6]|2\.[1-3]|3|4\.[1-3]|5\.[12]|6\.[12]|7|8|9)$")),
]

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_ENTRY_NAMESPACE: Final = "dangerous-goods-regulatory-tuple-v1"


class DangerousGoodsRegistryError(RuntimeError):
    """A dangerous-goods registry contract failed."""


class DangerousGoodsGenerationDisabled(DangerousGoodsRegistryError):
    """Production generation lacks a licensed maritime IMDG registry."""


class DangerousGoodsLicenseReceipt(BaseModel):
    """Pinned legal basis and explicit machine-use permissions."""

    model_config = _STRICT

    basis: Literal["public_license", "written_permission", "commercial_contract"]
    license_name: NonEmptyText
    license_url: HttpsUrl
    terms_sha256: Sha256
    attribution: NonEmptyText
    automated_processing_permitted: bool
    compiled_artifact_permitted: bool
    synthetic_generation_permitted: bool
    permission_evidence_sha256: Sha256 | None = None
    permission_evidence_reference: NonEmptyText | None = None

    @model_validator(mode="after")
    def private_permission_has_pinned_evidence(self) -> DangerousGoodsLicenseReceipt:
        private_basis = self.basis in {"written_permission", "commercial_contract"}
        evidence_present = (
            self.permission_evidence_sha256 is not None
            and self.permission_evidence_reference is not None
        )
        if private_basis != evidence_present:
            raise ValueError(
                "written permission and commercial contracts require pinned permission evidence; "
                "public licences must not claim private permission evidence"
            )
        return self

    @property
    def permits_compilation(self) -> bool:
        return self.automated_processing_permitted and self.compiled_artifact_permitted

    @property
    def permits_production_generation(self) -> bool:
        return self.permits_compilation and self.synthetic_generation_permitted


class DangerousGoodsSourceReceipt(BaseModel):
    """Exact regulatory source identity supplied by one provider."""

    model_config = _STRICT

    provider: NonEmptyText
    authority: NonEmptyText
    dataset: NonEmptyText
    regime: Literal["IMDG", "UN_MODEL_REGULATIONS", "TDG_CANADA", "US_49_CFR"]
    edition: NonEmptyText
    source_url: HttpsUrl
    source_bytes: Annotated[int, Field(gt=0)]
    source_sha256: Sha256
    source_records: Annotated[int, Field(gt=0)]
    parser_contract: Literal["normalized_dangerous_goods_provider_v1"]
    maritime_authoritative: bool
    license: DangerousGoodsLicenseReceipt

    @model_validator(mode="after")
    def maritime_authority_matches_regime(self) -> DangerousGoodsSourceReceipt:
        if self.maritime_authoritative != (self.regime == "IMDG"):
            raise ValueError("only an IMDG source may be marked maritime-authoritative")
        return self

    @property
    def production_generation_permitted(self) -> bool:
        return (
            self.regime == "IMDG"
            and self.maritime_authoritative
            and self.license.permits_production_generation
        )


def _safe_text(value: str, *, field: str) -> None:
    if value != value.strip():
        raise ValueError(f"{field} contains outer whitespace")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError(f"{field} contains control characters")


class DangerousGoodsRegulatoryTuple(BaseModel):
    """One indivisible row from a specific regulatory source edition."""

    model_config = _STRICT

    schema_version: Literal[1]
    entry_id: EntryId
    provider_record_id: NonEmptyText
    un_number: UnNumber
    proper_shipping_name: NonEmptyText
    primary_class_division: HazardClassDivision
    compatibility_group: Annotated[str, StringConstraints(pattern=r"^[A-HJ-NP-S]$")] | None
    subsidiary_risks: tuple[HazardClassDivision, ...]
    packing_group: Literal["I", "II", "III"] | None
    technical_name_required: bool

    @model_validator(mode="after")
    def tuple_is_exact_and_coherent(self) -> DangerousGoodsRegulatoryTuple:
        _safe_text(self.provider_record_id, field="provider record ID")
        _safe_text(self.proper_shipping_name, field="proper shipping name")
        is_explosive = self.primary_class_division.startswith("1.")
        if is_explosive != (self.compatibility_group is not None):
            raise ValueError(
                "Class 1 entries require a compatibility group, other classes forbid it"
            )
        if len(self.subsidiary_risks) != len(set(self.subsidiary_risks)):
            raise ValueError("subsidiary risks must be unique and source ordered")
        if self.primary_class_division in self.subsidiary_risks:
            raise ValueError("primary class/division cannot also be a subsidiary risk")
        return self


class ShipmentFlashpoint(BaseModel):
    """Shipment/product evidence, deliberately outside the regulatory tuple."""

    model_config = _STRICT

    value: Annotated[float, Field(allow_inf_nan=False)]
    unit: Literal["celsius", "fahrenheit"]
    test_method: NonEmptyText | None = None

    @model_validator(mode="after")
    def evidence_is_safe(self) -> ShipmentFlashpoint:
        absolute_zero = -273.15 if self.unit == "celsius" else -459.67
        if self.value < absolute_zero:
            raise ValueError(f"flashpoint cannot be below absolute zero in {self.unit}")
        if self.test_method is not None:
            _safe_text(self.test_method, field="flashpoint test method")
        return self


class DangerousGoodsShipmentSelection(BaseModel):
    """One selected regulatory row plus optional shipment-specific evidence."""

    model_config = _STRICT

    regulatory: DangerousGoodsRegulatoryTuple
    flashpoint: ShipmentFlashpoint | None = None


def _entry_id(
    *,
    source: DangerousGoodsSourceReceipt,
    provider_record_id: str,
    un_number: str,
    proper_shipping_name: str,
    primary_class_division: str,
    compatibility_group: str | None,
    subsidiary_risks: Sequence[str],
    packing_group: str | None,
    technical_name_required: bool,
) -> str:
    identity = canonical_json_bytes(
        {
            "source_sha256": source.source_sha256,
            "regime": source.regime,
            "edition": source.edition,
            "provider_record_id": provider_record_id,
            "un_number": un_number,
            "proper_shipping_name": proper_shipping_name,
            "primary_class_division": primary_class_division,
            "compatibility_group": compatibility_group,
            "subsidiary_risks": tuple(subsidiary_risks),
            "packing_group": packing_group,
            "technical_name_required": technical_name_required,
        }
    )
    return stable_id("dg", _ENTRY_NAMESPACE, identity)


def build_dangerous_goods_regulatory_tuple(
    *,
    source: DangerousGoodsSourceReceipt,
    provider_record_id: str,
    un_number: str,
    proper_shipping_name: str,
    primary_class_division: str,
    compatibility_group: str | None,
    subsidiary_risks: Sequence[str],
    packing_group: str | None,
    technical_name_required: bool,
) -> DangerousGoodsRegulatoryTuple:
    """Build a source-bound row with a deterministic complete-tuple identity."""

    values = {
        "provider_record_id": provider_record_id,
        "un_number": un_number,
        "proper_shipping_name": proper_shipping_name,
        "primary_class_division": primary_class_division,
        "compatibility_group": compatibility_group,
        "subsidiary_risks": tuple(subsidiary_risks),
        "packing_group": packing_group,
        "technical_name_required": technical_name_required,
    }
    return DangerousGoodsRegulatoryTuple.model_validate(
        {
            "schema_version": 1,
            **values,
            "entry_id": _entry_id(
                source=source,
                provider_record_id=provider_record_id,
                un_number=un_number,
                proper_shipping_name=proper_shipping_name,
                primary_class_division=primary_class_division,
                compatibility_group=compatibility_group,
                subsidiary_risks=subsidiary_risks,
                packing_group=packing_group,
                technical_name_required=technical_name_required,
            ),
        },
        strict=True,
    )


def _validate_entry_identity(
    entry: DangerousGoodsRegulatoryTuple,
    *,
    source: DangerousGoodsSourceReceipt,
) -> None:
    expected = _entry_id(
        source=source,
        provider_record_id=entry.provider_record_id,
        un_number=entry.un_number,
        proper_shipping_name=entry.proper_shipping_name,
        primary_class_division=entry.primary_class_division,
        compatibility_group=entry.compatibility_group,
        subsidiary_risks=entry.subsidiary_risks,
        packing_group=entry.packing_group,
        technical_name_required=entry.technical_name_required,
    )
    if entry.entry_id != expected:
        raise DangerousGoodsRegistryError(
            f"dangerous-goods entry identity is not bound to its tuple/source: {entry.entry_id}"
        )


class DangerousGoodsProvider(Protocol):
    """Adapter boundary for a separately authorized normalized source parser."""

    def source_receipt(self) -> DangerousGoodsSourceReceipt: ...

    def iter_regulatory_tuples(self) -> Iterable[DangerousGoodsRegulatoryTuple]: ...


class RandomIndex(Protocol):
    def randrange(self, stop: int) -> int: ...


class DangerousGoodsRegistry:
    """Immutable registry whose production sampler returns complete source rows."""

    def __init__(
        self,
        *,
        source: DangerousGoodsSourceReceipt,
        entries: Sequence[DangerousGoodsRegulatoryTuple],
    ) -> None:
        ordered = tuple(entries)
        keys = tuple(_entry_sort_key(row) for row in ordered)
        if not ordered:
            raise DangerousGoodsRegistryError("dangerous-goods registry cannot be empty")
        if keys != tuple(sorted(keys)):
            raise DangerousGoodsRegistryError("dangerous-goods entries are not canonically sorted")
        entry_ids = tuple(row.entry_id for row in ordered)
        provider_ids = tuple(row.provider_record_id for row in ordered)
        if len(entry_ids) != len(set(entry_ids)) or len(provider_ids) != len(set(provider_ids)):
            raise DangerousGoodsRegistryError(
                "dangerous-goods entries contain duplicate stable/provider identities"
            )
        for row in ordered:
            _validate_entry_identity(row, source=source)
        self._source = source
        self._entries = ordered

    @property
    def source(self) -> DangerousGoodsSourceReceipt:
        return self._source

    @property
    def entries(self) -> tuple[DangerousGoodsRegulatoryTuple, ...]:
        return self._entries

    def require_production_generation(self) -> None:
        if not self._source.production_generation_permitted:
            raise DangerousGoodsGenerationDisabled(
                "dangerous-goods generation requires a licensed maritime-authoritative IMDG "
                "registry with automated-processing, compiled-artifact, and synthetic-generation "
                "permission"
            )

    def sample_for_production(self, rng: RandomIndex) -> DangerousGoodsRegulatoryTuple:
        """Return one complete immutable row; never sample fields independently."""

        self.require_production_generation()
        index = rng.randrange(len(self._entries))
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self._entries)
        ):
            raise DangerousGoodsRegistryError("random index provider returned an invalid index")
        return self._entries[index]


def _entry_sort_key(row: DangerousGoodsRegulatoryTuple) -> tuple[str, ...]:
    return (
        row.un_number,
        row.proper_shipping_name,
        row.primary_class_division,
        row.compatibility_group or "",
        ",".join(row.subsidiary_risks),
        row.packing_group or "",
        row.provider_record_id,
        row.entry_id,
    )


class DangerousGoodsArtifactReceipt(BaseModel):
    model_config = _STRICT

    role: Literal["canonical_dangerous_goods_jsonl"]
    path: Literal["dangerous-goods.jsonl"]
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    records: Annotated[int, Field(gt=0)]


class DangerousGoodsRegistryReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    purpose: Literal["production_generation", "evaluation_only"]
    source: DangerousGoodsSourceReceipt
    artifacts: tuple[DangerousGoodsArtifactReceipt, ...] = Field(min_length=1, max_length=1)
    content_sha256: Sha256

    @model_validator(mode="after")
    def receipt_is_coherent_and_self_hashed(self) -> DangerousGoodsRegistryReceipt:
        if self.artifacts[0].records != self.source.source_records:
            raise ValueError("dangerous-goods artifact count differs from source receipt")
        if self.purpose == "production_generation" and not (
            self.source.production_generation_permitted
        ):
            raise ValueError("production receipt does not contain a licensed maritime IMDG source")
        body = self.model_dump(mode="json", exclude={"content_sha256"})
        if sha256_bytes(canonical_json_bytes(body)) != self.content_sha256:
            raise ValueError("dangerous-goods registry receipt content SHA-256 is invalid")
        return self


@dataclass(frozen=True, slots=True)
class CompiledDangerousGoodsRegistry:
    root: Path
    registry: DangerousGoodsRegistry
    receipt: DangerousGoodsRegistryReceipt
    commit_receipt: StagedCommitReceipt
    created: bool


def _canonical_jsonl(entries: Sequence[DangerousGoodsRegulatoryTuple]) -> bytes:
    keys = tuple(_entry_sort_key(row) for row in entries)
    if keys != tuple(sorted(keys)):
        raise DangerousGoodsRegistryError("dangerous-goods entries are not canonically sorted")
    return b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in entries)


def compile_dangerous_goods_registry(
    *,
    provider: DangerousGoodsProvider,
    output_parent: Path,
    run_name: str,
    purpose: Literal["production_generation", "evaluation_only"] = "production_generation",
) -> CompiledDangerousGoodsRegistry:
    """Compile provider rows only after the legal/purpose gate passes."""

    source = provider.source_receipt()
    if not source.license.permits_compilation:
        raise DangerousGoodsRegistryError(
            "dangerous-goods source does not permit automated compiled-artifact creation"
        )
    if purpose == "production_generation" and not source.production_generation_permitted:
        raise DangerousGoodsGenerationDisabled(
            "production compilation requires a licensed maritime-authoritative IMDG source"
        )
    entries = tuple(sorted(provider.iter_regulatory_tuples(), key=_entry_sort_key))
    if len(entries) != source.source_records:
        raise DangerousGoodsRegistryError(
            f"dangerous-goods provider count mismatch: expected {source.source_records}, "
            f"found {len(entries)}"
        )
    registry = DangerousGoodsRegistry(source=source, entries=entries)
    payload = _canonical_jsonl(entries)
    transaction_body = {
        "schema_version": 1,
        "purpose": purpose,
        "source": source.model_dump(mode="json"),
        "canonical_records_sha256": sha256_bytes(payload),
    }
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction_body)),
    )
    artifact = DangerousGoodsArtifactReceipt(
        role="canonical_dangerous_goods_jsonl",
        path="dangerous-goods.jsonl",
        bytes=len(payload),
        sha256=sha256_bytes(payload),
        records=len(entries),
    )
    receipt_body = {
        "schema_version": 1,
        "purpose": purpose,
        "source": source.model_dump(mode="json"),
        "artifacts": (artifact.model_dump(mode="json"),),
    }
    receipt = DangerousGoodsRegistryReceipt(
        schema_version=1,
        purpose=purpose,
        source=source,
        artifacts=(artifact,),
        content_sha256=sha256_bytes(canonical_json_bytes(receipt_body)),
    )
    receipt_payload = canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n"
    stage.publish_bytes("dangerous-goods.jsonl", payload)
    stage.publish_bytes("registry-receipt.json", receipt_payload)
    committed = stage.commit(
        expected_artifacts=("dangerous-goods.jsonl", "registry-receipt.json"),
        metadata={
            "schema_version": 1,
            "purpose": purpose,
            "regime": source.regime,
            "edition": source.edition,
            "source_sha256": source.source_sha256,
            "records": len(entries),
            "registry_receipt_sha256": sha256_bytes(receipt_payload),
        },
    )
    return CompiledDangerousGoodsRegistry(
        root=stage.final_root,
        registry=registry,
        receipt=receipt,
        commit_receipt=committed.receipt,
        created=committed.created,
    )


def _read_pinned_jsonl(
    path: Path,
    *,
    source: DangerousGoodsSourceReceipt,
    expected_sha256: str,
    expected_records: int,
) -> tuple[DangerousGoodsRegulatoryTuple, ...]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected dangerous-goods SHA-256 must be lowercase hexadecimal")
    if path.is_symlink():
        raise DangerousGoodsRegistryError("dangerous-goods JSONL must not be a symbolic link")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DangerousGoodsRegistryError("dangerous-goods JSONL is not readable") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DangerousGoodsRegistryError("dangerous-goods JSONL is not a regular file")
        digest = hashlib.sha256()
        entries: list[DangerousGoodsRegulatoryTuple] = []
        previous_key: tuple[str, ...] | None = None
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for line_number, encoded in enumerate(stream, start=1):
                digest.update(encoded)
                if encoded == b"\n" or not encoded.endswith(b"\n"):
                    raise DangerousGoodsRegistryError(
                        f"dangerous-goods row {line_number} is blank or unterminated"
                    )
                try:
                    row = DangerousGoodsRegulatoryTuple.model_validate_json(encoded, strict=True)
                except ValueError as error:
                    raise DangerousGoodsRegistryError(
                        f"dangerous-goods row {line_number} is invalid: {error}"
                    ) from error
                if encoded != canonical_json_bytes(row.model_dump(mode="json")) + b"\n":
                    raise DangerousGoodsRegistryError(
                        f"dangerous-goods row {line_number} is not canonical JSON"
                    )
                key = _entry_sort_key(row)
                if previous_key is not None and key <= previous_key:
                    raise DangerousGoodsRegistryError(
                        "dangerous-goods rows are duplicated or not canonically sorted"
                    )
                _validate_entry_identity(row, source=source)
                entries.append(row)
                previous_key = key
        if digest.hexdigest() != expected_sha256:
            raise DangerousGoodsRegistryError("dangerous-goods JSONL SHA-256 mismatch")
        if len(entries) != expected_records:
            raise DangerousGoodsRegistryError(
                f"dangerous-goods JSONL count mismatch: expected {expected_records}, "
                f"found {len(entries)}"
            )
        return tuple(entries)
    finally:
        os.close(descriptor)


def load_pinned_dangerous_goods_registry(
    root: Path,
    *,
    expected_receipt_sha256: str,
) -> DangerousGoodsRegistry:
    """Load a canonical registry and its complete source/licence receipt."""

    if root.is_symlink() or not root.is_dir():
        raise DangerousGoodsRegistryError("dangerous-goods registry root is not a plain directory")
    try:
        receipt_payload = read_regular_file_bytes(root / "registry-receipt.json")
    except ArtifactReadError as error:
        raise DangerousGoodsRegistryError(
            "dangerous-goods registry receipt is unreadable"
        ) from error
    if sha256_bytes(receipt_payload) != expected_receipt_sha256:
        raise DangerousGoodsRegistryError("dangerous-goods registry receipt SHA-256 mismatch")
    try:
        receipt = DangerousGoodsRegistryReceipt.model_validate_json(receipt_payload, strict=True)
    except ValueError as error:
        raise DangerousGoodsRegistryError("dangerous-goods registry receipt is invalid") from error
    if receipt_payload != canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n":
        raise DangerousGoodsRegistryError("dangerous-goods registry receipt is not canonical JSON")
    artifact = receipt.artifacts[0]
    entries = _read_pinned_jsonl(
        root / artifact.path,
        source=receipt.source,
        expected_sha256=artifact.sha256,
        expected_records=artifact.records,
    )
    return DangerousGoodsRegistry(source=receipt.source, entries=entries)
