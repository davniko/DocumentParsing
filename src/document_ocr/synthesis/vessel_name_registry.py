"""Compile and sample a provenance-bearing public cargo-vessel name registry.

The registry is deliberately a whole-name sampler, not a lexical generator.
Names from the curated IMO-VESSEL-NAMES dataset are accepted after syntax
normalization.  NOAA AIS-only names must recur in at least two independent
annual snapshots, which removes one-year AIS entry noise without maintaining a
hand-written allow/deny list.  The compiled JSONL is small enough to load once
and sample by tuple index; no runtime joins or database queries are required.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt
from document_ocr.synthesis.transport_identity import transport_identity_key
from document_ocr.synthesis.vessel_lexical import vessel_name_fit_exclusion_reason

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=1024)]

IMO_SOURCE_FAMILY: Final = "imo_vessel_names"
NOAA_SOURCE_FAMILY: Final = "noaa_pmel_ais"
_SOURCE_FAMILIES = frozenset((IMO_SOURCE_FAMILY, NOAA_SOURCE_FAMILY))
type SourceFamily = Literal["imo_vessel_names", "noaa_pmel_ais"]
_NAME_SCHEMA_VERSION = 1
_PARSER_CONTRACT: Final[Literal["public_cargo_vessel_csv_union_v1"]] = (
    "public_cargo_vessel_csv_union_v1"
)
_NORMALIZATION_POLICY = "nfkd_ascii_upper_single_space_v1"
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_REGISTRY_BYTES = 8 * 1024 * 1024
_SPACE = re.compile(r"\s+")
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class VesselNameRegistryError(RuntimeError):
    """A vessel-name source or compiled registry violates its pinned contract."""


class VesselNameSourceFilePin(BaseModel):
    """Exact identity and role of one local public-data snapshot."""

    model_config = _STRICT

    source_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]+$")]
    family: Literal["imo_vessel_names", "noaa_pmel_ais"]
    snapshot: Annotated[str, StringConstraints(pattern=r"^(?:2020\.1\.0|20[0-9]{2})$")]
    relative_path: NonEmptyText
    source_url: NonEmptyText
    bytes: Annotated[int, Field(gt=0, le=_MAX_SOURCE_BYTES)]
    sha256: Sha256
    records: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def source_and_path_are_supported(self) -> VesselNameSourceFilePin:
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("vessel source path must be a safe relative POSIX path")
        parsed = urlsplit(self.source_url)
        if parsed.scheme != "https" or parsed.fragment:
            raise ValueError("vessel source URL must be unfragmented HTTPS")
        if self.family == IMO_SOURCE_FAMILY:
            expected_prefix = (
                "https://raw.githubusercontent.com/interreg-speed/IMO-VESSEL-NAMES/master/data/"
            )
            if not self.source_url.startswith(expected_prefix) or self.snapshot != "2020.1.0":
                raise ValueError("IMO vessel source pin does not identify the supported release")
        else:
            expected = f"https://data.pmel.noaa.gov/pmel/erddap/tabledap/AIS{self.snapshot}_AIS.csv"
            if not self.source_url.startswith(expected):
                raise ValueError("NOAA vessel source pin does not identify its annual dataset")
        return self


class VesselNameRegistryPolicy(BaseModel):
    """Evidence-based source eligibility; no name-specific exceptions."""

    model_config = _STRICT

    normalization: Literal["nfkd_ascii_upper_single_space_v1"] = "nfkd_ascii_upper_single_space_v1"
    imo_rows: Literal["all_syntax_valid_rows_v1"] = "all_syntax_valid_rows_v1"
    noaa_rows: Literal["cargo_type_and_cross_snapshot_recurrence_v1"] = (
        "cargo_type_and_cross_snapshot_recurrence_v1"
    )
    noaa_minimum_distinct_years: Annotated[int, Field(ge=2, le=5)] = 2
    noaa_minimum_vessel_type: Literal[70] = 70
    noaa_maximum_vessel_type: Literal[79] = 79
    sampling: Literal["uniform_distinct_whole_name_v1"] = "uniform_distinct_whole_name_v1"


class VesselNameSourceLicense(BaseModel):
    model_config = _STRICT

    family: Literal["imo_vessel_names", "noaa_pmel_ais"]
    authority: NonEmptyText
    dataset: NonEmptyText
    license_name: NonEmptyText
    license_url: NonEmptyText
    attribution: NonEmptyText

    @model_validator(mode="after")
    def license_matches_source_family(self) -> VesselNameSourceLicense:
        parsed = urlsplit(self.license_url)
        if parsed.scheme != "https" or parsed.fragment:
            raise ValueError("vessel source license URL must be unfragmented HTTPS")
        if self.family == IMO_SOURCE_FAMILY and "opendatacommons.org" not in parsed.netloc:
            raise ValueError("IMO-VESSEL-NAMES must retain its declared PDDL license")
        if self.family == NOAA_SOURCE_FAMILY and "creativecommons.org" not in parsed.netloc:
            raise ValueError("NOAA AIS must retain its declared CC0 dedication")
        return self


class VesselNameRegistrySourceManifest(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    snapshot_date: Annotated[str, StringConstraints(pattern=r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}$")]
    files: tuple[VesselNameSourceFilePin, ...] = Field(min_length=4)
    licenses: tuple[VesselNameSourceLicense, VesselNameSourceLicense]
    policy: VesselNameRegistryPolicy

    @model_validator(mode="after")
    def source_coverage_is_complete(self) -> VesselNameRegistrySourceManifest:
        ids = tuple(row.source_id for row in self.files)
        paths = tuple(row.relative_path for row in self.files)
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("vessel source IDs and paths must be unique")
        families = {row.family for row in self.files}
        if families != _SOURCE_FAMILIES:
            raise ValueError("vessel manifest must include both supported source families")
        license_families = tuple(row.family for row in self.licenses)
        if license_families != (IMO_SOURCE_FAMILY, NOAA_SOURCE_FAMILY):
            raise ValueError("vessel source licenses must be ordered by supported family")
        noaa_years = tuple(
            sorted(int(row.snapshot) for row in self.files if row.family == NOAA_SOURCE_FAMILY)
        )
        if len(noaa_years) < self.policy.noaa_minimum_distinct_years:
            raise ValueError("NOAA source coverage is below its recurrence threshold")
        if noaa_years != tuple(range(noaa_years[0], noaa_years[-1] + 1)):
            raise ValueError("NOAA annual snapshots must be contiguous")
        return self


class VesselNameRegistryRecord(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = 1
    identity_key: Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+$")]
    name: Annotated[str, StringConstraints(min_length=3, max_length=128)]
    source_families: tuple[Literal["imo_vessel_names", "noaa_pmel_ais"], ...] = Field(
        min_length=1, max_length=2
    )
    noaa_years: tuple[Annotated[int, Field(ge=2020, le=2099)], ...]

    @model_validator(mode="after")
    def row_is_canonical(self) -> VesselNameRegistryRecord:
        if self.name != _normalize_name(self.name):
            raise ValueError("vessel registry name is not canonical")
        if vessel_name_fit_exclusion_reason(self.name) is not None:
            raise ValueError("vessel registry name fails the lexical syntax boundary")
        if self.identity_key != transport_identity_key(self.name):
            raise ValueError("vessel registry identity differs from its name")
        if self.source_families != tuple(sorted(set(self.source_families))):
            raise ValueError("vessel source families must be unique and sorted")
        if self.noaa_years != tuple(sorted(set(self.noaa_years))):
            raise ValueError("NOAA years must be unique and sorted")
        if (NOAA_SOURCE_FAMILY in self.source_families) != bool(self.noaa_years):
            raise ValueError("NOAA provenance and annual snapshots differ")
        return self


class VesselNameNormalizationAudit(BaseModel):
    model_config = _STRICT

    source_rows: Annotated[int, Field(ge=0)]
    syntax_eligible_rows: Annotated[int, Field(ge=0)]
    source_rejection_counts: dict[str, Annotated[int, Field(gt=0)]]
    distinct_syntax_eligible_names: Annotated[int, Field(gt=0)]
    accepted_registry_names: Annotated[int, Field(gt=0)]
    accepted_imo_names: Annotated[int, Field(gt=0)]
    accepted_noaa_names: Annotated[int, Field(gt=0)]
    accepted_noaa_only_names: Annotated[int, Field(ge=0)]
    quarantined_single_year_noaa_only_names: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def accounting_is_consistent(self) -> VesselNameNormalizationAudit:
        classified = self.syntax_eligible_rows + sum(self.source_rejection_counts.values())
        if classified != self.source_rows:
            raise ValueError("vessel row disposition does not balance")
        if self.accepted_registry_names > self.distinct_syntax_eligible_names:
            raise ValueError("accepted vessel names exceed syntax-eligible names")
        return self


class VesselNameRegistryReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    parser_contract: Literal["public_cargo_vessel_csv_union_v1"]
    snapshot_date: str
    source_manifest_sha256: Sha256
    policy: VesselNameRegistryPolicy
    audit: VesselNameNormalizationAudit
    registry_path: Literal["vessel-names.jsonl"]
    registry_bytes: Annotated[int, Field(gt=0, le=_MAX_REGISTRY_BYTES)]
    registry_sha256: Sha256
    registry_records: Annotated[int, Field(gt=0)]
    content_sha256: Sha256


@dataclass(frozen=True, slots=True)
class CompiledVesselNameRegistry:
    root: Path
    receipt: VesselNameRegistryReceipt
    commit_receipt: StagedCommitReceipt
    created: bool


@dataclass(frozen=True, slots=True)
class SampledVesselName:
    name: str
    identity_key: str
    attempts: int
    source_families: tuple[str, ...]
    noaa_years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LoadedVesselNameRegistry:
    """Immutable in-memory read model with expected O(1) deterministic sampling."""

    records: tuple[VesselNameRegistryRecord, ...]
    sha256: str

    def sample(
        self,
        *,
        stream: DeterministicStream,
        excluded_identity_keys: Collection[str] = frozenset(),
        used_identity_keys: Collection[str] = frozenset(),
        maximum_attempts: int = 512,
    ) -> SampledVesselName:
        if maximum_attempts < 1:
            raise ValueError("vessel sampling attempts must be positive")
        blocked = frozenset(excluded_identity_keys) | frozenset(used_identity_keys)
        if len(blocked) >= len(self.records):
            raise VesselNameRegistryError("vessel registry has no unblocked identity")
        for attempt in range(maximum_attempts):
            row = self.records[stream.randbelow(len(self.records), counter=attempt)]
            if row.identity_key in blocked:
                continue
            return SampledVesselName(
                name=row.name,
                identity_key=row.identity_key,
                attempts=attempt + 1,
                source_families=tuple(row.source_families),
                noaa_years=tuple(row.noaa_years),
            )
        raise VesselNameRegistryError(
            f"vessel registry did not find an unblocked identity in {maximum_attempts} attempts"
        )


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _SPACE.sub(" ", normalized.strip()).upper()


def _read_source_bytes(path: Path, pin: VesselNameSourceFilePin) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise VesselNameRegistryError(f"vessel source must be a plain file: {path}")
    payload = read_regular_file_bytes(path)
    if len(payload) != pin.bytes or sha256_bytes(payload) != pin.sha256:
        raise VesselNameRegistryError(f"vessel source bytes differ from pin: {pin.source_id}")
    return payload


def _read_csv_rows(payload: bytes, *, source_id: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise VesselNameRegistryError(f"vessel source is not UTF-8: {source_id}") from error
    try:
        reader = csv.DictReader(text.splitlines())
        rows = list(reader)
    except csv.Error as error:
        raise VesselNameRegistryError(f"vessel source is not valid CSV: {source_id}") from error
    if reader.fieldnames is None or any(None in row for row in rows):
        raise VesselNameRegistryError(f"vessel source CSV shape is invalid: {source_id}")
    return rows


def _canonical_surface(surfaces: Mapping[str, Counter[str]], *, identity_key: str) -> str:
    counts = surfaces[identity_key]
    # Frequency is observed across independent source files.  Ties resolve by
    # lexical order; there is no name-specific rule.
    return min(
        counts,
        key=lambda value: (-counts[value], value),
    )


def _registry_payload(records: Sequence[VesselNameRegistryRecord]) -> bytes:
    return b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in records)


def compile_vessel_name_registry(
    *,
    source_root: Path,
    source_manifest_path: Path,
    expected_manifest_sha256: str,
    output_parent: Path,
    run_name: str,
) -> CompiledVesselNameRegistry:
    """Compile pinned public snapshots into an immutable whole-name registry."""

    if source_manifest_path.is_symlink() or not source_manifest_path.is_file():
        raise VesselNameRegistryError("vessel source manifest must be a plain file")
    manifest_bytes = read_regular_file_bytes(source_manifest_path)
    if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        raise VesselNameRegistryError("vessel source manifest SHA-256 mismatch")
    try:
        manifest = VesselNameRegistrySourceManifest.model_validate_json(manifest_bytes, strict=True)
    except Exception as error:
        raise VesselNameRegistryError("vessel source manifest is invalid") from error
    if manifest_bytes != canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n":
        raise VesselNameRegistryError("vessel source manifest is not canonical JSON")

    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    family_by_key: dict[str, set[SourceFamily]] = defaultdict(set)
    noaa_years_by_key: dict[str, set[int]] = defaultdict(set)
    rejected: Counter[str] = Counter()
    source_rows = 0
    syntax_eligible_rows = 0
    for pin in manifest.files:
        path = source_root / PurePosixPath(pin.relative_path)
        payload = _read_source_bytes(path, pin)
        rows = _read_csv_rows(payload, source_id=pin.source_id)
        if len(rows) != pin.records:
            raise VesselNameRegistryError(
                f"vessel source row count differs from pin: {pin.source_id}"
            )
        source_rows += len(rows)
        if pin.family == IMO_SOURCE_FAMILY:
            expected = {"imo", "vessel_name", "gross_tonnage", "type", "flag"}
            fieldnames = set(rows[0]) if rows else set()
            if not expected <= fieldnames or not ({"year_built", "year_build"} & fieldnames):
                raise VesselNameRegistryError(f"unexpected IMO CSV columns: {pin.source_id}")
        else:
            if not rows or set(rows[0]) != {"VesselName", "VesselType"}:
                raise VesselNameRegistryError(f"unexpected NOAA CSV columns: {pin.source_id}")

        for row in rows:
            if pin.family == NOAA_SOURCE_FAMILY:
                try:
                    vessel_type = int(row["VesselType"])
                except (KeyError, TypeError, ValueError):
                    rejected[f"{pin.family}:invalid_vessel_type"] += 1
                    continue
                if not (
                    manifest.policy.noaa_minimum_vessel_type
                    <= vessel_type
                    <= manifest.policy.noaa_maximum_vessel_type
                ):
                    rejected[f"{pin.family}:outside_cargo_type"] += 1
                    continue
                raw_name = row["VesselName"]
            else:
                raw_name = row["vessel_name"]
            name = _normalize_name(raw_name)
            reason = vessel_name_fit_exclusion_reason(name) if name else "empty"
            if reason is not None:
                rejected[f"{pin.family}:{reason}"] += 1
                continue
            syntax_eligible_rows += 1
            key = transport_identity_key(name)
            surfaces[key][name] += 1
            family_by_key[key].add(pin.family)
            if pin.family == NOAA_SOURCE_FAMILY:
                noaa_years_by_key[key].add(int(pin.snapshot))

    eligible_keys = tuple(
        sorted(
            key
            for key in surfaces
            if IMO_SOURCE_FAMILY in family_by_key[key]
            or len(noaa_years_by_key[key]) >= manifest.policy.noaa_minimum_distinct_years
        )
    )
    records = tuple(
        VesselNameRegistryRecord(
            identity_key=key,
            name=_canonical_surface(surfaces, identity_key=key),
            source_families=tuple(sorted(family_by_key[key])),
            noaa_years=tuple(sorted(noaa_years_by_key[key])),
        )
        for key in eligible_keys
    )
    if not records:
        raise VesselNameRegistryError("vessel registry policy accepted no names")
    payload = _registry_payload(records)
    if len(payload) > _MAX_REGISTRY_BYTES:
        raise VesselNameRegistryError("compiled vessel registry exceeds its size bound")
    accepted_keys = frozenset(eligible_keys)
    audit = VesselNameNormalizationAudit(
        source_rows=source_rows,
        syntax_eligible_rows=syntax_eligible_rows,
        source_rejection_counts=dict(sorted(rejected.items())),
        distinct_syntax_eligible_names=len(surfaces),
        accepted_registry_names=len(records),
        accepted_imo_names=sum(IMO_SOURCE_FAMILY in family_by_key[key] for key in accepted_keys),
        accepted_noaa_names=sum(NOAA_SOURCE_FAMILY in family_by_key[key] for key in accepted_keys),
        accepted_noaa_only_names=sum(
            family_by_key[key] == {NOAA_SOURCE_FAMILY} for key in accepted_keys
        ),
        quarantined_single_year_noaa_only_names=sum(
            family_by_key[key] == {NOAA_SOURCE_FAMILY}
            and len(noaa_years_by_key[key]) < manifest.policy.noaa_minimum_distinct_years
            for key in surfaces
        ),
    )
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "sourceManifestSha256": expected_manifest_sha256,
                "policy": manifest.policy.model_dump(mode="json"),
                "audit": audit.model_dump(mode="json"),
                "registrySha256": sha256_bytes(payload),
            }
        )
    )
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=run_name,
        transaction_sha256=transaction,
    )
    receipt_body = {
        "schema_version": 1,
        "parser_contract": _PARSER_CONTRACT,
        "snapshot_date": manifest.snapshot_date,
        "source_manifest_sha256": expected_manifest_sha256,
        "policy": manifest.policy.model_dump(mode="json"),
        "audit": audit.model_dump(mode="json"),
        "registry_path": "vessel-names.jsonl",
        "registry_bytes": len(payload),
        "registry_sha256": sha256_bytes(payload),
        "registry_records": len(records),
    }
    receipt = VesselNameRegistryReceipt(
        schema_version=1,
        parser_contract=_PARSER_CONTRACT,
        snapshot_date=manifest.snapshot_date,
        source_manifest_sha256=expected_manifest_sha256,
        policy=manifest.policy,
        audit=audit,
        registry_path="vessel-names.jsonl",
        registry_bytes=len(payload),
        registry_sha256=sha256_bytes(payload),
        registry_records=len(records),
        content_sha256=sha256_bytes(canonical_json_bytes(receipt_body)),
    )
    receipt_payload = canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n"
    stage.publish_bytes("vessel-names.jsonl", payload)
    stage.publish_bytes("registry-receipt.json", receipt_payload)
    committed = stage.commit(
        expected_artifacts=("registry-receipt.json", "vessel-names.jsonl"),
        metadata={
            "schema_version": 1,
            "registry_records": len(records),
            "registry_sha256": sha256_bytes(payload),
            "registry_receipt_sha256": sha256_bytes(receipt_payload),
        },
    )
    return CompiledVesselNameRegistry(
        root=stage.final_root,
        receipt=receipt,
        commit_receipt=committed.receipt,
        created=committed.created,
    )


def load_vessel_name_registry(
    path: Path, *, expected_sha256: str, expected_records: int
) -> LoadedVesselNameRegistry:
    """Load and fully validate a pinned compiled registry once per run."""

    if path.is_symlink() or not path.is_file():
        raise VesselNameRegistryError("vessel registry must be a plain file")
    if path.stat().st_size > _MAX_REGISTRY_BYTES or sha256_file(path) != expected_sha256:
        raise VesselNameRegistryError("vessel registry size or SHA-256 differs from its pin")
    rows: list[VesselNameRegistryRecord] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise VesselNameRegistryError(
                    f"vessel registry contains a blank row at line {line_number}"
                )
            try:
                row = VesselNameRegistryRecord.model_validate_json(raw, strict=True)
            except Exception as error:
                raise VesselNameRegistryError(
                    f"vessel registry row {line_number} is invalid"
                ) from error
            if raw != canonical_json_bytes(row.model_dump(mode="json")) + b"\n":
                raise VesselNameRegistryError(
                    f"vessel registry row {line_number} is not canonical JSON"
                )
            rows.append(row)
    if len(rows) != expected_records:
        raise VesselNameRegistryError(
            f"vessel registry expected {expected_records} rows, found {len(rows)}"
        )
    keys = tuple(row.identity_key for row in rows)
    if keys != tuple(sorted(set(keys))):
        raise VesselNameRegistryError("vessel registry identities must be unique and sorted")
    return LoadedVesselNameRegistry(records=tuple(rows), sha256=expected_sha256)
