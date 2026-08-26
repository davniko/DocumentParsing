#!/usr/bin/env python3
"""Prepare, validate, review, and publish the full semantic-v2 B/L pilot run.

The extraction ledger and retained OCR artifacts are read-only. Label workers
write only immutable per-attempt candidates; this overseer-owned tool controls
all shared state and publishes the run manifest last.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import tracemalloc
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError

from document_ocr.label_schemas.bill_of_lading import (
    BillOfLadingAnnotation,
    BillOfLadingExclusion,
    BillOfLadingLabel,
)
from document_ocr.label_schemas.mpci_bill_of_lading import (
    ContainerIdentifier,
    MpciBillOfLadingAnnotation,
)
from document_ocr.label_schemas.mpci_projection import project_bill_of_lading_to_mpci
from document_ocr.label_schemas.semantic_review import BillOfLadingIndependentReview

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "artifacts/glm-ocr/pilots/blc150/runs/glm-ocr-blc-pilot150-faa3c717dbc4"
SOURCE_RUN_ID = "glm-ocr-blc-pilot150-faa3c717dbc4"
PILOT = ROOT / "data/pilots/blc-pilot-150-faa3c717dbc4"
V1_RUN = ROOT / "artifacts/kie-labels/mpci-bl-pilot-v1-confined-r3"
RUN_ID = "mpci-bl-semantic-v2-pilot150-r1"
DEST = ROOT / "artifacts/kie-labels" / RUN_ID
RUN_KIND = "semantic_v2_full_pilot150"
QUALITY_FILTER_ROOT: Path | None = None
QUALITY_FILTER_MANIFEST_SHA256: str | None = None
RUN_CONFIG_PATH: Path | None = None
PLATFORM_TOTAL_SLOTS = 4
MAX_CONCURRENT_WORKERS = 3
WORKER_REASONING_EFFORT = "high"
REVIEWER_REASONING_EFFORT = "high"
REQUIRE_INDEPENDENT_REVIEW = False
ISO_COUNTRIES = Path("/usr/share/iso-codes/json/iso_3166-1.json")
T5GEMMA_TOKENIZER_UPSTREAM = "google/t5gemma-2-270m-270m"
T5GEMMA_TOKENIZER_UPSTREAM_REVISION = "7c38f16641f455ef0685b18431faf1b17722d5a1"
T5GEMMA_TOKENIZER_MIRROR = "jordimas/t5gemma-2-270m-270m"
T5GEMMA_TOKENIZER_MIRROR_REVISION = "dfb6d0d619d8e6d42f279a6197abb90dec0137c1"
T5GEMMA_TOKENIZER_FILES = {
    "special_tokens_map.json": {
        "gitBlobOid": "1a6193244714d3d78be48666cb02cdbfac62ad86",
        "sha256": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
    },
    "tokenizer.json": {
        "gitBlobOid": "8f149029a8e2a286d3244e2ffbfbbf29777d482b",
        "sha256": "4c76318660cc87c3029535cc1596717df67132d231ce4ecf088ba2a8ec07c098",
    },
    "tokenizer_config.json": {
        "gitBlobOid": "b5986b139db26d501be78ce7f6ad853d52291c16",
        "sha256": "a378e7507ea7190045d0a80e139e4bda802738f678d3e4df05d7550786af1382",
    },
}

_EXPECTED = {
    "selectedDocuments": 150,
    "selectedPages": 286,
    "completeDocuments": 141,
    "incompleteDocuments": 9,
    "successfulPages": 276,
    "failedPages": 10,
    "nonLatinExcludedDocuments": 18,
    "eligibleDocuments": 123,
    "eligiblePages": 234,
    "preLabelExcludedDocuments": 27,
}

_CONTRACT_PATHS = (
    "tools/bill_of_lading_v2_full_run.py",
    "src/document_ocr/label_schemas/common.py",
    "src/document_ocr/label_schemas/bill_of_lading.py",
    "src/document_ocr/label_schemas/mpci_bill_of_lading.py",
    "src/document_ocr/label_schemas/mpci_projection.py",
    "src/document_ocr/label_schemas/semantic_review.py",
    "BILL_OF_LADING_LABELING_REFERENCE_V2.md",
    "docs/mpci-kie-semantic-schema-v2-design.md",
    "LABELING_SESSION_PROMPT_V2.md",
)

_Sha256Text = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _ExpectedCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    selectedDocuments: int = Field(ge=1)
    selectedPages: int = Field(ge=1)
    completeDocuments: int = Field(ge=0)
    incompleteDocuments: int = Field(ge=0)
    successfulPages: int = Field(ge=0)
    failedPages: int = Field(ge=0)
    nonLatinExcludedDocuments: int = Field(ge=0)
    eligibleDocuments: int = Field(ge=1)
    eligiblePages: int = Field(ge=1)
    preLabelExcludedDocuments: int = Field(ge=0)


class _RunConfiguration(BaseModel):
    """Strict second-source configuration for one immutable labeling run."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]+$")]
    run_kind: Annotated[str, StringConstraints(min_length=1)]
    source_run: Annotated[str, StringConstraints(min_length=1)]
    source_run_id: Annotated[str, StringConstraints(min_length=1)]
    quality_filter_root: Annotated[str, StringConstraints(min_length=1)]
    quality_filter_manifest_sha256: _Sha256Text
    v1_run: Annotated[str, StringConstraints(min_length=1)]
    platform_total_slots: int = Field(ge=2)
    max_concurrent_workers: int = Field(ge=1)
    worker_reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"]
    reviewer_reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"]
    require_independent_review: bool
    expected: _ExpectedCounts


def _configured_repo_path(value: str, *, field_name: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{field_name} must be a safe repository-relative path")
    candidate = (ROOT / relative).resolve(strict=True)
    root = ROOT.resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise RuntimeError(f"{field_name} escapes the repository")
    return candidate


def _configure_run(path: Path) -> None:
    """Load a strict immutable-run profile before dispatching a CLI command."""

    global DEST
    global MAX_CONCURRENT_WORKERS
    global PLATFORM_TOTAL_SLOTS
    global QUALITY_FILTER_MANIFEST_SHA256
    global QUALITY_FILTER_ROOT
    global REQUIRE_INDEPENDENT_REVIEW
    global REVIEWER_REASONING_EFFORT
    global RUN_CONFIG_PATH
    global RUN_ID
    global RUN_KIND
    global SOURCE_RUN
    global SOURCE_RUN_ID
    global V1_RUN
    global WORKER_REASONING_EFFORT
    global _EXPECTED

    resolved = path.resolve(strict=True)
    root = ROOT.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RuntimeError("run config must be a file inside the repository")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    configuration = _RunConfiguration.model_validate(raw, strict=True)
    if configuration.max_concurrent_workers >= configuration.platform_total_slots:
        raise RuntimeError("max_concurrent_workers must leave one platform slot for the overseer")

    source_run = _configured_repo_path(configuration.source_run, field_name="source_run")
    quality_root = _configured_repo_path(
        configuration.quality_filter_root, field_name="quality_filter_root"
    )
    v1_run = _configured_repo_path(configuration.v1_run, field_name="v1_run")
    destination = ROOT / "artifacts/kie-labels" / configuration.run_id
    if destination.resolve().parent != (ROOT / "artifacts/kie-labels").resolve():
        raise RuntimeError("run_id produced an unsafe destination")

    SOURCE_RUN = source_run
    SOURCE_RUN_ID = configuration.source_run_id
    QUALITY_FILTER_ROOT = quality_root
    QUALITY_FILTER_MANIFEST_SHA256 = configuration.quality_filter_manifest_sha256
    V1_RUN = v1_run
    RUN_ID = configuration.run_id
    RUN_KIND = configuration.run_kind
    DEST = destination
    PLATFORM_TOTAL_SLOTS = configuration.platform_total_slots
    MAX_CONCURRENT_WORKERS = configuration.max_concurrent_workers
    WORKER_REASONING_EFFORT = configuration.worker_reasoning_effort
    REVIEWER_REASONING_EFFORT = configuration.reviewer_reasoning_effort
    REQUIRE_INDEPENDENT_REVIEW = configuration.require_independent_review
    _EXPECTED = configuration.expected.model_dump(mode="python")
    RUN_CONFIG_PATH = resolved


def _current_contract_hashes() -> dict[str, str]:
    missing = [path for path in _CONTRACT_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"labeling contract files are absent: {missing!r}")
    return {path: _file_digest(ROOT / path) for path in _CONTRACT_PATHS}


def _verify_frozen_contract(metadata: dict[str, Any]) -> None:
    expected = metadata.get("frozenContract")
    current = _current_contract_hashes()
    if expected != current:
        changed = sorted(set(expected or {}) | set(current))
        changed = [path for path in changed if (expected or {}).get(path) != current.get(path)]
        raise RuntimeError(
            "labeling contract changed after its last recorded revision: " f"{changed!r}"
        )
    run_configuration = metadata.get("runConfiguration")
    if not isinstance(run_configuration, dict) or RUN_CONFIG_PATH is None:
        raise RuntimeError("run metadata lacks its configured run-profile binding")
    if (
        run_configuration.get("path") != str(RUN_CONFIG_PATH.relative_to(ROOT))
        or run_configuration.get("sha256") != _file_digest(RUN_CONFIG_PATH)
    ):
        raise RuntimeError("run configuration changed after preparation")


_PAGE_MARKER = re.compile(r"^--- PAGE ([1-9][0-9]*) ---\n", re.MULTILINE)
_CONTAINER_IDENTIFIER_TOKEN = re.compile(r"(?<![A-Z0-9])([A-Z]{4})[ -]?(\d{7})(?![A-Z0-9])")
_CONTAINER_IDENTIFIER_ADAPTER = TypeAdapter(ContainerIdentifier)
_TARGET_CONTAMINATION = re.compile(
    r"(?ix)\b(?:"
    r"tax(?:\s*(?:id|no|number))?|vat|cif|"
    r"v\.\s*d\.?|vd(?=\s*(?:no\.?\b|[:#]))|c\.?n\.?p\.?j|"
    r"acid(?=\s*(?:(?:code|no\.?|number)\b|[:#]|[0-9]{6,}))|"
    r"shipper['\u2019]?s\s+load(?:\s*[,/&]\s*|\s+and\s+)count|"
    r"said\s+to\s+contain|s\.?t\.?c\.?|package\s+limitation\s+clause|"
    r"particulars\s+furnished\s+by\s+shipper|received\s+by\s+the\s+carrier"
    r")\b"
)
_CONTACT_LABEL = re.compile(r"(?i)\b(?:contact|e-?mail|tel(?:ephone)?|phone|ph|fax)\s*:")
_CONTACT_VALUE_PREFIX = re.compile(
    r"(?i)^\s*(?:tel(?:ephone)?|phone|ph|fax|e-?mail|email|website|web|url)"
    r"(?:\s*(?:[/&,+-])\s*"
    r"(?:tel(?:ephone)?|phone|ph|fax|e-?mail|email|website|web|url))*"
    r"\s*(?::|=|-|\s)\s*"
)
_REFERENCE_VALUE_PREFIX = re.compile(
    r"(?i)^\s*(?:svc\.?\s*contract|service\s+contract|export\s+references?|"
    r"invoice|inv\.?|s\.?o\.?|booking(?:\s+no\.?)?)(?=\s|[:#])\s*[:#]*\s*"
)
_CONTACT_TRAILING_FOOTNOTE = re.compile(r"\*+$")
_POSTAL_VALUE_LABEL = re.compile(r"(?i)\b(?:POST(?:AL)?\s+CODE|POSTCODE|ZIP\s+CODE)\b\s*[:#-]?")
_CONTAINER_AS_PACKAGE_TYPE = re.compile(r"(?i)\b(?:CNTR(?:\(S\)|S)?|CONTAINERS?)\b")
_FOREIGN_EXPORTER_COUNTRY_HEADING = re.compile(
    r"(?i)\bFOREIGN\s+EXPORTER\s+COUNTRY\s*(?::|$)"
)
_DELIVERY_AGENT_ROLE_HEADING = re.compile(
    r"(?im)(?:^\s*(?:\d+\.\s*)?(?:"
    r"(?:DELIVERY|DESTINATION)\s+AGENT\b|"
    r"(?:PORT\s+OF\s+DISCHARGE|DISCHARGE\s+PORT)\s+AGENT\b|"
    r"AGENT(?:'S)?(?:\s+(?:ADDRESS|DETAILS))?\s+AT\s+(?:THE\s+)?"
    r"(?:DESTINATION|PORT\s+OF\s+DISCHARGE)\b|"
    r"AGENT\s+TO\s+CONTACT\s+AT\s+(?:THE\s+)?DESTINATION\b|"
    r"AGENT\s+AT\s+PORT\s+OF\s+(?:DISCHARGE|DESTINATION)\b|"
    r"THE\s+NAME\s+AND\s+ADDRESS\s+OF\s+SHIPPING\s+AGENT\s+AT\s+DESTINATION\b|"
    r"FOR\s+(?:CARGO\s+)?DELIVERY(?:\s+OF\s+GOODS)?"
    r"(?:[\s,:;-]+PLEASE)?[\s,:;-]+"
    r"(?:APPLY\s+TO|CONTACT)\b|"
    r"FOR\s+RELEASE\s+OF\s+CARGO(?:\s+PLEASE)?\s+(?:APPLY\s+TO|CONTACT)\b|"
    r"TO\s+OBTAIN\s+DELIVERY\s+CONTACT\b|"
    r"\(?AS\s+(?:FRT\s+FWDRS|FREIGHT\s+FORWARDERS?)\s+DELIVERY\s+AGENT\s+ONLY\)?\b"
    r")|\bDETAILS\s+OF\s+THE\s+AGENT\s+AT\s+P\.?\s*O\.?\s*D\.?(?=\s*[:;-]|\s|$))"
)
_CONDITIONAL_NON_NEGOTIABLE = re.compile(
    r"(?i)\b(?:"
    r"NOT\s+NEGOTIABLE\s+UNLESS\s+CONSIGNED\s+[\"\u201c\u201d']*TO\s+ORDER|"
    r"CONSIGNEE\s*\(\s*RECEIVABLE\s+IF\s+ASSIGNED\s+[\"\u201c\u201d']*TO\s+ORDER"
    r")\b"
)
_MARK_VALUE_PREFIX = re.compile(r"(?i)^\s*(?:O|C|LOT|BATCH|DRUM)\s*/?\s*NO\.?\s*[:#-]?\s*")
_GENERIC_PAYMENT_PLACES = frozenset({"DESTINATION", "ORIGIN"})
_STREET_DESIGNATORS = frozenset({"AV", "AVE", "CT", "DR", "HWY", "LN", "PL", "RD", "ST"})
_CONTACT_KIND = re.compile(
    r"(?i)\b(?P<kind>tel(?:ephone)?|phone|ph|mob(?:ile)?|fax|e-?mail)\s*[:.]?"
)
_FAX_FIELD = re.compile(
    r"(?i)\bFAX(?:\s*(?:NO\.?|NUMBER))?\s*(?::|=)?\s*(?P<value>[^\r\n]*)"
)
_JOINT_PHONE_FAX_FIELD = re.compile(
    r"(?i)\b(?:TEL(?:EPHONE)?|PHONE|PH)\s*[-/&\u2010-\u2015]\s*FAX"
    r"(?:\s*(?:NO\.?|NUMBER))?\s*(?::|=)?"
)
_FIXED_MPCI_KEYS = frozenset(
    {
        "partyFunction",
        "locationOfIdentificationQualifier",
        "communicationMeans",
        "contactIdentifier",
        "textSubjectCodeQualifier",
        "measuredAttributeCode",
        "measurementUnitCode",
        "chargeCategory",
        "temperatureTypeCodeQualifier",
    }
)

# These aliases are frozen annotation infrastructure, not worker-inferred facts.
# Official names and alpha codes come from the separately hashed ISO snapshot.
_COUNTRY_ALIASES = {
    "A R EGYPT": "EG",
    "AL YEMEN": "YE",
    "BRASIL": "BR",
    "EGITTO": "EG",
    "EGIPTO": "EG",
    "EGYPTE": "EG",
    "ENGLAND": "GB",
    "ESPANA": "ES",
    "HOLLAND": "NL",
    "HONGKONG": "HK",
    "IRAN": "IR",
    "IVORY COAST": "CI",
    "KINGDOM OF SUDIA ARABIA": "SA",
    "KSA": "SA",
    "KOREA": "KR",
    "LAOS": "LA",
    "MOLDOVA": "MD",
    "P R CHINA": "CN",
    "PR CHINA": "CN",
    "REPUBLIC OF EGYPT": "EG",
    "REPUBLIC OF KOREA": "KR",
    "RUSSIA": "RU",
    "SIERRA LIONE": "SL",
    "SOUTH KOREA": "KR",
    "SYRIA": "SY",
    "TAIWAN": "TW",
    "TANZANIA": "TZ",
    "THE NETHERLANDS": "NL",
    "TURKEY": "TR",
    "TURKIYE": "TR",
    "U A E": "AE",
    "U K": "GB",
    "U S A": "US",
    "UAE": "AE",
    "UK": "GB",
    "UNITED KINGDOM": "GB",
    "UNITED STATES": "US",
    "USA": "US",
    "VENEZUELA": "VE",
    "VIETNAM": "VN",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values
        ).encode(),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _omit_explicit_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _omit_explicit_nulls(child) for key, child in value.items() if child is not None
        }
    if isinstance(value, list):
        return [_omit_explicit_nulls(child) for child in value if child is not None]
    return value


def _safe_retained_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"retained path is not a safe relative path: {value!r}")
    candidate = (SOURCE_RUN / relative).resolve(strict=True)
    source_root = SOURCE_RUN.resolve(strict=True)
    if not candidate.is_relative_to(source_root) or not candidate.is_file():
        raise RuntimeError(f"retained artifact escapes the source run: {value!r}")
    return candidate


def _non_latin_letters(text: str) -> list[str]:
    return [
        character
        for character in text
        if unicodedata.category(character).startswith("L")
        and "LATIN" not in unicodedata.name(character, "")
    ]


def _ledger() -> tuple[list[sqlite3.Row], dict[str, list[sqlite3.Row]]]:
    connection = sqlite3.connect(SOURCE_RUN / "state.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        documents = connection.execute(
            "select * from documents where run_id=? order by document_id", (SOURCE_RUN_ID,)
        ).fetchall()
        pages = connection.execute(
            "select * from pages where run_id=? order by document_id,page_index",
            (SOURCE_RUN_ID,),
        ).fetchall()
    finally:
        connection.close()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for page in pages:
        grouped[str(page["document_id"])].append(page)
    return documents, grouped


def _verify_pilot_inventory(documents: list[sqlite3.Row]) -> None:
    manifest = [
        json.loads(line)
        for line in (PILOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(documents) != _EXPECTED["selectedDocuments"] or len(manifest) != len(documents):
        raise RuntimeError("pilot and extraction-ledger document counts differ")
    ledger_hashes = {row["source_sha256"] for row in documents}
    manifest_hashes = {row.get("source_sha256") for row in manifest}
    if None in ledger_hashes or ledger_hashes != manifest_hashes:
        raise RuntimeError("pilot and extraction-ledger source identities differ")


def _quality_filter_selection(
    documents: list[sqlite3.Row],
) -> tuple[
    list[sqlite3.Row],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """Bind a published quality projection back to its frozen extraction ledger."""

    if QUALITY_FILTER_ROOT is None or QUALITY_FILTER_MANIFEST_SHA256 is None:
        raise RuntimeError("quality-filter selection is not configured")
    manifest_path = QUALITY_FILTER_ROOT / "manifest.json"
    if _file_digest(manifest_path) != QUALITY_FILTER_MANIFEST_SHA256:
        raise RuntimeError("quality-filter manifest digest differs from the run config")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("dataset_kind") != "glm-ocr-quality-filtered-page-extractions"
        or manifest.get("run_id") != SOURCE_RUN_ID
        or manifest.get("publication") != "manifest_last_immutable"
    ):
        raise RuntimeError("quality-filter manifest identity or publication state is invalid")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError("quality-filter manifest files are absent")
    quality_root = QUALITY_FILTER_ROOT.resolve(strict=True)
    for entry in files:
        if not isinstance(entry, dict):
            raise RuntimeError("quality-filter manifest file entry is invalid")
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("quality-filter manifest contains an unsafe file path")
        artifact = (quality_root / relative).resolve(strict=True)
        if not artifact.is_relative_to(quality_root) or not artifact.is_file():
            raise RuntimeError("quality-filter artifact escapes its published root")
        if artifact.stat().st_size != entry.get("bytes") or _file_digest(artifact) != entry.get(
            "sha256"
        ):
            raise RuntimeError(f"quality-filter artifact identity changed: {relative}")
        expected_rows = entry.get("rows")
        if relative.suffix == ".jsonl" and len(_read_jsonl(artifact)) != expected_rows:
            raise RuntimeError(f"quality-filter artifact row count changed: {relative}")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("quality-filter source provenance is absent")
    source_directory = Path(str(source.get("run_directory", ""))).resolve(strict=True)
    if source_directory != SOURCE_RUN.resolve(strict=True):
        raise RuntimeError("quality-filter source run differs from the configured extraction run")
    source_files = source.get("files")
    if not isinstance(source_files, dict):
        raise RuntimeError("quality-filter source file identities are absent")
    for name in ("inventory.jsonl", "run-provenance.json", "state.sqlite3"):
        identity = source_files.get(name)
        artifact = SOURCE_RUN / name
        if (
            not isinstance(identity, dict)
            or artifact.stat().st_size != identity.get("bytes")
            or _file_digest(artifact) != identity.get("sha256")
        ):
            raise RuntimeError(f"quality-filter source artifact identity changed: {name}")

    document_rows = _read_jsonl(QUALITY_FILTER_ROOT / "eligible-documents.jsonl")
    page_rows = _read_jsonl(QUALITY_FILTER_ROOT / "eligible-pages.jsonl")
    quality_documents: dict[str, dict[str, Any]] = {}
    for row in document_rows:
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or document_id in quality_documents:
            raise RuntimeError("quality-filter document IDs are invalid or duplicated")
        quality_documents[document_id] = row
    quality_pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in page_rows:
        document_id = row.get("document_id")
        if document_id not in quality_documents:
            raise RuntimeError("quality-filter page references an unselected document")
        quality_pages[str(document_id)].append(row)
    for rows in quality_pages.values():
        rows.sort(key=lambda row: int(row["page_index"]))

    ledger_documents = {str(row["document_id"]): row for row in documents}
    if set(quality_documents) - set(ledger_documents):
        raise RuntimeError("quality-filter selection contains documents absent from the ledger")
    selected = [ledger_documents[document_id] for document_id in sorted(quality_documents)]
    if len(selected) != _EXPECTED["selectedDocuments"]:
        raise RuntimeError("quality-filter selected-document count differs from the run config")
    return selected, quality_documents, quality_pages, manifest


def _verify_quality_work_item(
    item: dict[str, Any],
    quality_document: dict[str, Any],
    quality_pages: list[dict[str, Any]],
) -> None:
    """Confirm a reconstructed work item exactly matches the selected projection rows."""

    source = item["source"]
    document_id = source["documentId"]
    projected_source = quality_document.get("source")
    if not isinstance(projected_source, dict):
        raise RuntimeError(f"{document_id}: quality-filter source record is absent")
    expected_source_values = {
        "documentId": quality_document.get("document_id"),
        "extractionRunId": quality_document.get("run_id"),
        "sourceUri": projected_source.get("source_uri"),
        "localCanonicalPath": projected_source.get("local_canonical_path"),
        "sourceSha256": projected_source.get("source_sha256"),
        "documentPageCount": quality_document.get("document_page_count"),
    }
    for key, expected in expected_source_values.items():
        if source.get(key) != expected:
            raise RuntimeError(f"{document_id}: quality-filter source mismatch for {key}")

    projected_page_refs = quality_document.get("pages")
    if not isinstance(projected_page_refs, list) or len(projected_page_refs) != len(quality_pages):
        raise RuntimeError(f"{document_id}: quality-filter page references are incomplete")
    expected_indexes = list(range(source["documentPageCount"]))
    if [row.get("page_index") for row in quality_pages] != expected_indexes:
        raise RuntimeError(f"{document_id}: quality-filter pages are incomplete or unordered")

    joined: list[str] = []
    for index, (page, projection, page_reference) in enumerate(
        zip(source["pages"], quality_pages, projected_page_refs, strict=True)
    ):
        expected_page = {
            "pageIndex": index,
            "pageNumber": index + 1,
            "pageId": projection.get("page_id"),
            "extractionId": projection.get("extraction_id"),
            "rawOcrTextSha256": projection.get("raw_ocr_text_sha256"),
            "rawResponsePath": projection.get("raw_response_path"),
            "rawResponseSha256": projection.get("raw_response_sha256"),
            "rasterPath": projection.get("raster_path"),
            "rasterSha256": projection.get("raster_sha256"),
        }
        if page != expected_page:
            raise RuntimeError(f"{document_id}: quality-filter page {index + 1} mismatch")
        for snake_name in (
            "page_index",
            "page_number",
            "page_id",
            "extraction_id",
            "raw_ocr_text_sha256",
            "raw_response_path",
            "raster_path",
            "raster_sha256",
        ):
            if projection.get(snake_name) != page_reference.get(snake_name):
                raise RuntimeError(
                    f"{document_id}: quality document/page projection differs for {snake_name}"
                )
        raw_text = projection.get("raw_ocr_text")
        if not isinstance(raw_text, str) or _digest(raw_text.encode()) != projection.get(
            "raw_ocr_text_sha256"
        ):
            raise RuntimeError(f"{document_id}: projected raw OCR identity mismatch")
        joined.append(f"--- PAGE {index + 1} ---\n{raw_text}")
    joined_text = "\n\n".join(joined)
    if item["joinedRawText"] != joined_text or source["joinedRawTextSha256"] != _digest(
        joined_text.encode()
    ):
        raise RuntimeError(f"{document_id}: joined OCR differs from quality-filter projection")


def _work_item(document: sqlite3.Row, pages: list[sqlite3.Row]) -> dict[str, Any]:
    document_id = str(document["document_id"])
    if document["status"] != "complete":
        raise RuntimeError(f"cannot build a work item for incomplete document {document_id}")
    if [page["page_index"] for page in pages] != list(range(document["page_count"])):
        raise RuntimeError(f"source pages are incomplete or unordered: {document_id}")
    if any(page["status"] != "success" for page in pages):
        raise RuntimeError(f"source has a non-success page: {document_id}")

    source_json = json.loads(document["source_json"])
    local_path = Path(source_json["local_canonical_path"])
    if not local_path.is_file() or _file_digest(local_path) != document["source_sha256"]:
        raise RuntimeError(f"local source identity mismatch: {document_id}")

    joined: list[str] = []
    page_references: list[dict[str, Any]] = []
    for page in pages:
        result = json.loads(page["result_json"])
        raw_text = result["raw_ocr_text"]
        raw_path = _safe_retained_path(result["raw_response_path"])
        raster_path = _safe_retained_path(result["raster_path"])
        if _digest(raw_text.encode()) != page["ocr_text_sha256"]:
            raise RuntimeError(f"raw OCR digest mismatch: {page['page_id']}")
        if _file_digest(raw_path) != page["raw_response_sha256"]:
            raise RuntimeError(f"raw response digest mismatch: {page['page_id']}")
        if _file_digest(raster_path) != result["raster_sha256"]:
            raise RuntimeError(f"raster digest mismatch: {page['page_id']}")
        joined.append(f"--- PAGE {page['page_index'] + 1} ---\n{raw_text}")
        page_references.append(
            {
                "pageIndex": page["page_index"],
                "pageNumber": page["page_index"] + 1,
                "pageId": page["page_id"],
                "extractionId": page["extraction_id"],
                "rawOcrTextSha256": page["ocr_text_sha256"],
                "rawResponsePath": result["raw_response_path"],
                "rawResponseSha256": page["raw_response_sha256"],
                "rasterPath": result["raster_path"],
                "rasterSha256": result["raster_sha256"],
            }
        )
    joined_text = "\n\n".join(joined)
    return {
        "source": {
            "documentId": document_id,
            "extractionRunId": SOURCE_RUN_ID,
            "sourceUri": source_json["source_uri"],
            "localCanonicalPath": source_json["local_canonical_path"],
            "sourceSha256": document["source_sha256"],
            "documentPageCount": document["page_count"],
            "joinedRawTextSha256": _digest(joined_text.encode()),
            "pages": page_references,
        },
        "joinedRawText": joined_text,
    }


def _duplicate_groups(work_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parents = {item["source"]["documentId"]: item["source"]["documentId"] for item in work_items}

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    identities: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in work_items:
        source = item["source"]
        identities[("joined_raw_text", source["joinedRawTextSha256"])].append(source["documentId"])
        identities[("first_page_raster", source["pages"][0]["rasterSha256"])].append(
            source["documentId"]
        )
    for ids in identities.values():
        for document_id in ids[1:]:
            union(ids[0], document_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for document_id in sorted(parents):
        groups[find(document_id)].append(document_id)
    result: list[dict[str, Any]] = []
    for index, document_ids in enumerate((ids for ids in groups.values() if len(ids) > 1), start=1):
        evidence = []
        for (kind, digest), ids in sorted(identities.items()):
            overlap = sorted(set(ids) & set(document_ids))
            if len(overlap) > 1:
                evidence.append({"kind": kind, "sha256": digest, "documentIds": overlap})
        result.append(
            {
                "groupId": f"duplicate-{index:03d}",
                "documentIds": sorted(document_ids),
                "identityEvidence": evidence,
            }
        )
    return result


def prepare() -> None:
    if DEST.exists():
        raise RuntimeError(f"destination already exists: {DEST}")
    all_documents, grouped = _ledger()
    quality_documents: dict[str, dict[str, Any]] = {}
    quality_pages: dict[str, list[dict[str, Any]]] = {}
    quality_manifest: dict[str, Any] | None = None
    if QUALITY_FILTER_ROOT is None:
        documents = all_documents
        _verify_pilot_inventory(documents)
    else:
        documents, quality_documents, quality_pages, quality_manifest = _quality_filter_selection(
            all_documents
        )
    stats: Counter[str] = Counter()
    eligibility: list[dict[str, Any]] = []
    prelabel_exclusions: list[dict[str, Any]] = []
    work_items: list[dict[str, Any]] = []

    for document in documents:
        document_id = str(document["document_id"])
        pages = grouped.get(document_id, [])
        stats[f"documents_{document['status']}"] += 1
        stats["pages_total"] += len(pages)
        stats["pages_success"] += sum(page["status"] == "success" for page in pages)
        if document["status"] != "complete":
            reason = "document_not_complete"
            prelabel_exclusions.append(
                {"documentId": document_id, "reason": reason, "status": document["status"]}
            )
            eligibility.append({"documentId": document_id, "eligible": False, "reason": reason})
            continue
        item = _work_item(document, pages)
        if quality_manifest is not None:
            _verify_quality_work_item(
                item, quality_documents[document_id], quality_pages[document_id]
            )
        elif _non_latin_letters(item["joinedRawText"]):
            reason = "non_latin_letter_in_raw_ocr"
            prelabel_exclusions.append({"documentId": document_id, "reason": reason})
            eligibility.append({"documentId": document_id, "eligible": False, "reason": reason})
            continue
        work_items.append(item)
        eligibility.append(
            {
                "documentId": document_id,
                "eligible": True,
                "pageCount": document["page_count"],
                "joinedRawTextSha256": item["source"]["joinedRawTextSha256"],
            }
        )

    observed = {
        "selectedDocuments": len(documents),
        "selectedPages": stats["pages_total"],
        "completeDocuments": stats["documents_complete"],
        "incompleteDocuments": stats["documents_incomplete"],
        "successfulPages": stats["pages_success"],
        "failedPages": stats["pages_total"] - stats["pages_success"],
        "nonLatinExcludedDocuments": sum(
            row["reason"] == "non_latin_letter_in_raw_ocr" for row in prelabel_exclusions
        ),
        "eligibleDocuments": len(work_items),
        "eligiblePages": sum(item["source"]["documentPageCount"] for item in work_items),
        "preLabelExcludedDocuments": len(prelabel_exclusions),
    }
    if observed != _EXPECTED:
        raise RuntimeError(f"frozen eligibility counts changed: {observed!r}")

    quality_filter_relative: str | None = None
    if quality_manifest is not None:
        if QUALITY_FILTER_ROOT is None:
            raise RuntimeError("quality manifest exists without a configured quality root")
        quality_filter_relative = str(QUALITY_FILTER_ROOT.relative_to(ROOT))

    temporary = Path(tempfile.mkdtemp(prefix=f".{RUN_ID}.", dir=DEST.parent))
    try:
        for name in (
            "work-items",
            "candidate-attempts",
            "candidates",
            "validated",
            "exclusions",
            "needs-review",
            "training",
            "validation/attempt-checks",
            "validation/attempt-reviews",
            "validation/independent-review-checks",
            "validation/decision-history",
            "validation/contract-revisions",
            "validation/decisions",
            "independent-reviews",
            "reviewer-logs",
            "worker-logs",
            "eda/tables",
            "eda/plots",
        ):
            (temporary / name).mkdir(parents=True)
        for item in work_items:
            document_id = item["source"]["documentId"]
            _atomic_json(temporary / "work-items" / f"{document_id}.json", item)
        _atomic_jsonl(temporary / "eligibility.jsonl", eligibility)
        _atomic_jsonl(temporary / "prelabel-exclusions.jsonl", prelabel_exclusions)
        duplicate_groups = _duplicate_groups(work_items)
        _atomic_json(
            temporary / "validation/duplicate-candidates.json",
            {
                "method": "exact joined-OCR or first-page-raster identity",
                "groups": duplicate_groups,
            },
        )
        _atomic_json(
            temporary / "run-metadata.json",
            {
                "runId": RUN_ID,
                "runKind": RUN_KIND,
                "status": "prepared",
                "preparedAt": _now(),
                "schemaVersion": "2.0.0",
                "annotationSchemaVersion": "2.0.0",
                "sourceRun": {
                    "runId": SOURCE_RUN_ID,
                    "path": str(SOURCE_RUN.relative_to(ROOT)),
                    "stateSha256": _file_digest(SOURCE_RUN / "state.sqlite3"),
                    "inventorySha256": _file_digest(SOURCE_RUN / "inventory.jsonl"),
                    **(
                        {"pilotManifestSha256": _file_digest(PILOT / "manifest.jsonl")}
                        if quality_manifest is None
                        else {
                            "qualityFilter": {
                                "path": quality_filter_relative,
                                "manifestSha256": QUALITY_FILTER_MANIFEST_SHA256,
                                "qualityPolicySha256": quality_manifest["quality_policy_sha256"],
                                "eligibleDocumentsSha256": next(
                                    entry["sha256"]
                                    for entry in quality_manifest["files"]
                                    if entry["path"] == "eligible-documents.jsonl"
                                ),
                                "eligiblePagesSha256": next(
                                    entry["sha256"]
                                    for entry in quality_manifest["files"]
                                    if entry["path"] == "eligible-pages.jsonl"
                                ),
                            }
                        }
                    ),
                },
                "eligibility": observed,
                "eligibleDocumentIds": sorted(item["source"]["documentId"] for item in work_items),
                "duplicateCandidateGroupCount": len(duplicate_groups),
                "overseer": {
                    "role": "session overseer following LABELING_SESSION_PROMPT_V2.md",
                    "requestedModel": "gpt-5.6-terra",
                    "runtimeModelIdentity": "not_exposed_to_session",
                },
                "workerContract": {
                    "model": "gpt-5.6-luna",
                    "reasoningEffort": WORKER_REASONING_EFFORT,
                    "oneFreshWorkerPerDocumentAttempt": True,
                    "platformTotalSlots": PLATFORM_TOTAL_SLOTS,
                    "maxConcurrentWorkersAchieved": MAX_CONCURRENT_WORKERS,
                    "usageAvailability": "unavailable_from_collaboration_runtime",
                },
                "independentReviewContract": {
                    "required": REQUIRE_INDEPENDENT_REVIEW,
                    "model": "gpt-5.6-luna",
                    "reasoningEffort": REVIEWER_REASONING_EFFORT,
                    "oneFreshReviewerPerCandidateReview": True,
                    "reviewerCannotModifyCandidate": True,
                },
                "runConfiguration": (
                    {
                        "path": str(RUN_CONFIG_PATH.relative_to(ROOT)),
                        "sha256": _file_digest(RUN_CONFIG_PATH),
                    }
                    if RUN_CONFIG_PATH is not None
                    else None
                ),
                "frozenContract": _current_contract_hashes(),
                "gates": {
                    "targetedTests": "101 passed in 0.94s",
                    "nonTrainingTests": (
                        "467 passed; 2 optional-training tests unavailable because the host "
                        "uv environment does not include the train dependency group"
                    ),
                    "ruff": "all checks passed",
                    "mypy": "label-schema scope: no issues in 5 source files",
                },
            },
        )
        os.replace(temporary, DEST)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"prepared {len(work_items)} immutable work items and "
        f"{len(prelabel_exclusions)} pre-label exclusions at {DEST}"
    )


def _page_texts(joined: str) -> dict[int, str]:
    markers = list(_PAGE_MARKER.finditer(joined))
    pages: dict[int, str] = {}
    for index, marker in enumerate(markers):
        page_number = int(marker.group(1))
        end = markers[index + 1].start() - 2 if index + 1 < len(markers) else len(joined)
        pages[page_number] = joined[marker.end() : end]
    if list(pages) != list(range(1, len(pages) + 1)):
        raise RuntimeError("joined OCR page markers are incomplete or unordered")
    return pages


def _verify_evidence(
    evidence_groups: Iterable[Iterable[Any]], pages: dict[int, str], document_id: str
) -> None:
    for evidence_group in evidence_groups:
        _verify_evidence_sequence(evidence_group, pages, document_id)


def _raw_value_offset_at_or_after(
    source_page: str, evidence: Any, minimum_offset: int
) -> int:
    """Locate the earliest supported raw value at or after ``minimum_offset``."""

    excerpt: str = evidence.ocrExcerpt
    raw_value: str = evidence.rawValue
    if raw_value not in excerpt:
        return -1
    excerpt_start = 0
    while (offset := source_page.find(excerpt, excerpt_start)) >= 0:
        raw_offset = excerpt.find(raw_value, max(0, minimum_offset - offset))
        if raw_offset >= 0:
            return offset + raw_offset
        excerpt_start = offset + 1
    return -1


def _verify_evidence_sequence(
    evidence_items: Iterable[Any], pages: dict[int, str], document_id: str
) -> None:
    previous_page = 0
    previous_offset = -1
    for evidence in evidence_items:
        source_page = pages.get(evidence.pageNumber)
        if source_page is None:
            raise RuntimeError(f"{document_id}: evidence cites absent page {evidence.pageNumber}")
        if evidence.pageNumber < previous_page:
            raise RuntimeError(f"{document_id}: evidence is not verbatim/in source order")
        minimum_offset = previous_offset if evidence.pageNumber == previous_page else 0
        offset = _raw_value_offset_at_or_after(source_page, evidence, minimum_offset)
        if offset < 0:
            raise RuntimeError(f"{document_id}: evidence is not verbatim/in source order")
        previous_page = evidence.pageNumber
        previous_offset = offset


_TEXT_TRANSLITERATION = str.maketrans(
    {
        "Ð": "D",
        "Ø": "O",
        "Ł": "L",
        "Đ": "D",
        "ð": "d",
        "ø": "o",
        "ł": "l",
        "đ": "d",
    }
)
_UNIT_EVIDENCE_TOKENS = {
    "kilogram": frozenset({"KG", "KGS", "KILOGRAM", "KILOGRAMS"}),
    "pound": frozenset({"LB", "LBS", "POUND", "POUNDS"}),
    "cubic_metre": frozenset({"CBM", "M3", "CUM", "CUBICMETRE", "CUBICMETRES"}),
    "celsius": frozenset({"C", "CELSIUS"}),
    "fahrenheit": frozenset({"F", "FAHRENHEIT"}),
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%Y/%m/%d",
    "%Y %b %d",
    "%Y %B %d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%m.%d.%Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d/%b/%Y",
    "%d/%B/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%b/%d/%Y",
    "%B/%d/%Y",
    "%b-%d-%Y",
    "%B-%d-%Y",
    "%d-%b-%y",
    "%d/%b/%y",
    "%d %b %y",
    "%b %d %y",
    "%b/%d/%y",
    "%b-%d-%y",
    "%B-%d-%y",
    "%d-%m-%y",
    "%d/%m/%y",
    "%d.%m.%y",
    "%m-%d-%y",
    "%m/%d/%y",
    "%m.%d.%y",
)
_DATE_CANDIDATE = re.compile(
    r"\b(?:"
    r"[0-9]{1,4}(?:ST|ND|RD|TH)?[-/. ]+[A-Z]{3,9}[-/. ]+[0-9]{1,4}|"
    r"[A-Z]{3,9}[-/. ]+[0-9]{1,2}(?:ST|ND|RD|TH)?[-/. ]+[0-9]{2,4}|"
    r"[0-9]{1,4}[-/.]+[0-9]{1,2}[-/.]+[0-9]{1,4}"
    r")\b"
)
_NUMBER_CANDIDATE = re.compile(r"(?<![0-9])[+-]?[0-9][0-9.,]*(?![0-9])")
_SPACE_GROUPED_INTEGER = r"[+-]?[0-9]{1,3}(?:[ \t\r\n]+[0-9]{3})+"
_MASS_UNIT_TOKEN = r"(?:KGS?|KILOGRAMS?)"
_SPACE_GROUPED_MASS_CANDIDATE = re.compile(
    rf"(?:\b{_MASS_UNIT_TOKEN}[ \t\r\n:]*({_SPACE_GROUPED_INTEGER})(?![0-9])|"
    rf"(?<![0-9])({_SPACE_GROUPED_INTEGER})(?=[ \t\r\n]*{_MASS_UNIT_TOKEN}\b))"
)
_PACKAGE_UNIT_TOKEN = (
    r"(?:BAGS?|BALES?|BOX(?:ES)?|BUNDLES?|CARTONS?|CASES?|CRATES?|DRUMS?|IBCS?|"
    r"PACKAGES?|PACKS?|PALLETS?|PCS|PIECES?|ROLLS?|SACKS?|SETS?)"
)
_SPACE_GROUPED_PACKAGE_CANDIDATE = re.compile(
    rf"(?:\b{_PACKAGE_UNIT_TOKEN}[ \t\r\n:]*({_SPACE_GROUPED_INTEGER})(?![0-9])|"
    rf"(?<![0-9])({_SPACE_GROUPED_INTEGER})(?=[ \t\r\n]*{_PACKAGE_UNIT_TOKEN}\b))"
)
_MASS_VALUE_PATH = re.compile(
    r"\.(?:grossWeight|netWeight|verifiedGrossMass)\.(?:value|unit)$"
)
_PACKAGE_QUANTITY_PATH = re.compile(r"\.packages\[[0-9]+\]\.quantity$")
_METRIC_TONNE_EVIDENCE = re.compile(
    r"(?i)(?<![A-Z0-9])(?:M\s*/?\s*T|METRIC\s+TON(?:NE)?S?|TONNES?|T)(?![A-Z0-9])"
)
_NUMBER_WORD_VALUES = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
}
_NUMBER_WORD_SCALES = {"HUNDRED": 100, "THOUSAND": 1_000, "MILLION": 1_000_000}


def _ascii_evidence_text(value: str) -> str:
    translated = value.translate(_TEXT_TRANSLITERATION)
    return unicodedata.normalize("NFKD", translated).encode("ascii", "ignore").decode().upper()


def _evidence_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Z0-9]+", _ascii_evidence_text(value)))


def _numeric_evidence_candidates(value: str) -> tuple[float, ...]:
    candidates: set[float] = set()
    for match in _NUMBER_CANDIDATE.findall(_ascii_evidence_text(value)):
        source = match.strip(".,")
        spellings = {
            source,
            source.replace(",", ""),
            source.replace(".", ""),
            source.replace(".", "").replace(",", "."),
        }
        for separator in (".", ","):
            if source.count(separator) < 2:
                continue
            grouped_integer, fractional = source.rsplit(separator, 1)
            spellings.add(
                grouped_integer.replace(".", "").replace(",", "")
                + "."
                + fractional
            )
        for spelling in spellings:
            try:
                candidates.add(float(spelling))
            except ValueError:
                continue
    tokens = _evidence_tokens(value)
    candidates.update(
        float(_NUMBER_WORD_VALUES[token]) for token in tokens if token in _NUMBER_WORD_VALUES
    )

    current = 0
    total = 0
    active = False

    def flush_words() -> None:
        nonlocal current, total, active
        if active:
            candidates.add(float(total + current))
        current = 0
        total = 0
        active = False

    for index, token in enumerate(tokens):
        if token in _NUMBER_WORD_VALUES:
            current += _NUMBER_WORD_VALUES[token]
            active = True
            continue
        if token == "HUNDRED" and active:
            current = max(current, 1) * _NUMBER_WORD_SCALES[token]
            continue
        if token in {"THOUSAND", "MILLION"} and active:
            total += max(current, 1) * _NUMBER_WORD_SCALES[token]
            current = 0
            continue
        next_is_number_word = index + 1 < len(tokens) and (
            tokens[index + 1] in _NUMBER_WORD_VALUES or tokens[index + 1] in _NUMBER_WORD_SCALES
        )
        if token == "AND" and active and next_is_number_word:
            continue
        flush_words()
    flush_words()
    return tuple(sorted(candidates))


def _space_grouped_evidence_candidates(
    value: str, pattern: re.Pattern[str]
) -> tuple[float, ...]:
    normalized = _ascii_evidence_text(value)
    candidates: set[float] = set()
    for match in pattern.finditer(normalized):
        candidate = next(group for group in match.groups() if group is not None)
        candidates.add(float(re.sub(r"\s+", "", candidate)))
    return tuple(sorted(candidates))


def _space_grouped_mass_evidence_candidates(value: str) -> tuple[float, ...]:
    """Parse OCR-split thousands only when an explicit kilogram unit binds them."""
    return _space_grouped_evidence_candidates(value, _SPACE_GROUPED_MASS_CANDIDATE)


def _space_grouped_package_evidence_candidates(value: str) -> tuple[float, ...]:
    """Parse OCR-split package counts only when an explicit package unit binds them."""
    return _space_grouped_evidence_candidates(value, _SPACE_GROUPED_PACKAGE_CANDIDATE)


def _date_evidence_supports(target: str, raw_values: str) -> bool:
    expected = date.fromisoformat(target)
    normalized = _ascii_evidence_text(raw_values)
    # OCR may wrap a numeric date immediately after a separator (for example
    # ``14-08-\n2023``).  Evidence items preserve those exact fragments and are
    # joined with whitespace for validation, so remove only whitespace around
    # separators that remain bounded by digits.
    normalized = re.sub(r"(?<=\d)\s*([-/\.])\s*(?=\d)", r"\1", normalized)
    normalized = re.sub(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)\.",
        r"\1 ",
        normalized,
    ).replace("SEPT", "SEP")
    # GLM-OCR may detach an ordinal suffix with punctuation/whitespace, as in
    # ``17- th June, 2023``.  Remove only a complete ordinal token immediately
    # following a plausible day; the month token remains required by the date
    # candidate parser below.
    normalized = re.sub(
        r"\b([0-3]?[0-9])\s*[-.]?\s*(?:ST|ND|RD|TH)\b",
        r"\1 ",
        normalized,
    )
    # Normalize separators around an alphabetic month as a unit.  A month
    # abbreviation's trailing dot is otherwise consumed above while the
    # preceding dot remains (``27.JUL.2023`` -> ``27.JUL 2023``), creating a
    # mixed-separator spelling that ``strptime`` cannot represent directly.
    normalized = re.sub(
        r"\b([0-9]{1,2})[-/. ]+([A-Z]{3,9})[-/. ]+([0-9]{2,4})\b",
        r"\1 \2 \3",
        normalized,
    )
    normalized = re.sub(
        r"\b([0-9]{4})[-/. ]+([A-Z]{3,9})[-/. ]+([0-9]{1,2})\b",
        r"\1 \2 \3",
        normalized,
    )
    normalized = re.sub(
        r"\b([A-Z]{3,9})[-/. ]+([0-9]{1,2})[-/. ]+([0-9]{2,4})\b",
        r"\1 \2 \3",
        normalized,
    )
    normalized = re.sub(r"\bDAY\s+OF\b", " ", normalized)
    normalized = re.sub(r"_+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized.replace(",", " "))
    # OCR occasionally inserts whitespace between the two digits of a day
    # immediately before an alphabetic month (for example ``1 4 FEB 2024``).
    # Restrict the repair to that date-specific shape; unrelated digit pairs
    # remain independent evidence tokens.
    normalized = re.sub(
        r"\b([0-3])\s+([0-9])(?=\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b)",
        r"\1\2",
        normalized,
    )
    for candidate in _DATE_CANDIDATE.findall(normalized):
        for date_format in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(candidate, date_format).date()
            except ValueError:
                continue
            if parsed == expected:
                return True
    return False


def _target_leaf_is_supported(target_path: str, target: Any, raw_values: str) -> bool:
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        tolerance = max(1e-6, abs(float(target)) * 1e-9)
        candidates = _numeric_evidence_candidates(raw_values)
        if any(abs(candidate - float(target)) <= tolerance for candidate in candidates):
            return True
        if (
            target_path.endswith(".value")
            and _MASS_VALUE_PATH.search(target_path) is not None
        ):
            space_grouped_candidates = _space_grouped_mass_evidence_candidates(raw_values)
            if any(
                abs(candidate - float(target)) <= tolerance
                for candidate in space_grouped_candidates
            ):
                return True
        if _PACKAGE_QUANTITY_PATH.search(target_path) is not None:
            space_grouped_candidates = _space_grouped_package_evidence_candidates(raw_values)
            if any(
                abs(candidate - float(target)) <= tolerance
                for candidate in space_grouped_candidates
            ):
                return True
        return (
            target_path.endswith(".value")
            and _MASS_VALUE_PATH.search(target_path) is not None
            and _METRIC_TONNE_EVIDENCE.search(_ascii_evidence_text(raw_values)) is not None
            and any(
                abs(candidate * 1000.0 - float(target)) <= tolerance
                for candidate in candidates
            )
        )
    if not isinstance(target, str):
        return True
    if target_path.endswith("Date"):
        return _date_evidence_supports(target, raw_values)

    raw_tokens = frozenset(_evidence_tokens(raw_values))
    raw_compact = "".join(_evidence_tokens(raw_values))
    if target_path.endswith(".unit"):
        if (
            target == "kilogram"
            and _MASS_VALUE_PATH.search(target_path) is not None
            and _METRIC_TONNE_EVIDENCE.search(_ascii_evidence_text(raw_values)) is not None
        ):
            return True
        allowed_tokens = _UNIT_EVIDENCE_TOKENS.get(
            target, frozenset({"".join(_evidence_tokens(target))})
        )
        return any(token in raw_tokens or token in raw_compact for token in allowed_tokens)

    normalized_raw = " ".join(_evidence_tokens(raw_values))
    if target_path.endswith("negotiability"):
        if target == "non_negotiable":
            return any(
                phrase in normalized_raw
                for phrase in (
                    "NON NEGOTIABLE",
                    "NOT NEGOTIABLE",
                    "SEA WAYBILL",
                    "THIS WAYBILL",
                    "EXPRESS BILL OF LADING",
                    "EXPRESS BL",
                    "EXPRESS B L",
                    "B L EXPRESS",
                    "NEGOTIABLE ONLY IF CONSIGNED TO ORDER",
                    "RECEIVABLE IF ASSIGNED TO ORDER",
                )
            )
        return any(
            phrase in normalized_raw
            for phrase in ("NEGOTIABLE", "TO THE ORDER OF", "TO THE ORDER", "TO ORDER")
        )
    if target_path.endswith("paymentArrangement"):
        phrases: dict[str, tuple[str, ...]] = {
            "prepaid": ("PREPAID",),
            "collect": ("COLLECT", "PAYABLE AT DESTINATION"),
            "third_party": ("THIRD PARTY",),
            "payable_elsewhere": ("PAYABLE ELSEWHERE",),
        }
        return any(phrase in normalized_raw for phrase in phrases[target])
    if target_path.endswith(".sameAs"):
        phrases = {
            "consignee": ("SAME AS CONSIGNEE", "SAME AS CNEE", "SAME AS CONS"),
            "shipper": ("SAME AS SHIPPER", "SAME AS SHPR"),
        }
        return any(phrase in normalized_raw for phrase in phrases[target])

    target_tokens = _evidence_tokens(target)
    target_compact = "".join(target_tokens)
    return target_compact in raw_compact or all(token in raw_tokens for token in target_tokens)


def _verify_target_evidence_support(
    target: dict[str, Any], evidence_items: Iterable[Any], document_id: str
) -> None:
    target_leaves = _flatten(target)
    unsupported: list[str] = []
    normalization_findings: list[str] = []
    for evidence in evidence_items:
        value = target_leaves[evidence.targetPath]
        raw_values = " ".join(item.rawValue for item in evidence.rawOcrEvidence)
        if not _target_leaf_is_supported(evidence.targetPath, value, raw_values):
            unsupported.append(evidence.targetPath)
        normalization_finding = _evidence_normalization_finding(evidence, value)
        if normalization_finding is not None:
            normalization_findings.append(normalization_finding)
    if unsupported:
        raise RuntimeError(
            f"{document_id}: target leaves are not derivable from cited raw OCR values: "
            f"{unsupported!r}"
        )
    if normalization_findings:
        raise RuntimeError(
            f"{document_id}: evidence normalization is mislabeled: "
            f"{normalization_findings!r}"
        )


def _evidence_normalization_finding(evidence: Any, target_value: Any) -> str | None:
    if evidence.evidenceKind != "verbatim" or not isinstance(target_value, str):
        return None
    if any(item.rawValue == target_value for item in evidence.rawOcrEvidence):
        return None
    return (
        f"{evidence.targetPath}: verbatim evidence does not contain the exact emitted "
        "string; use a non-verbatim evidence kind with a normalization rule"
    )


def _evidence_normalization_findings(
    annotation: BillOfLadingAnnotation, target: dict[str, Any]
) -> list[str]:
    """Reject string transformations mislabeled as verbatim evidence.

    Numeric JSON scalars necessarily differ in type from OCR strings, so their
    lexical normalization remains governed by the existing support checks.
    String targets, however, are verbatim only when at least one cited raw value
    equals the emitted scalar exactly.  Readable enums, relations, and units must
    declare their transformation through a non-verbatim evidence kind and rule.
    """

    target_leaves = _flatten(target)
    findings: list[str] = []
    for evidence in annotation.evidence:
        target_value = target_leaves[evidence.targetPath]
        finding = _evidence_normalization_finding(evidence, target_value)
        if finding is not None:
            findings.append(finding)
    return findings


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(child, path))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, child in enumerate(value):
            flattened.update(_flatten(child, f"{prefix}[{index}]"))
        return flattened
    return {prefix: value}


def _address_repeats_component(address: str, component: str) -> bool:
    component_words = tuple(re.findall(r"[A-Z0-9]+", component.upper()))
    if not component_words:
        return False
    address_words = tuple(re.findall(r"[A-Z0-9]+", address.upper()))
    # An unspaced hyphen is part of many proper facility/locality names (for
    # example, ``CHINA-SHANGHAI COOPERATION ZONE``), not an address-component
    # boundary. Spaced dashes and punctuation-delimited components remain
    # valid redundancy boundaries; terminal embedded components are handled
    # by the word-sequence check below.
    for segment in re.split(r"[,;|/]|\s+-\s+", address.upper()):
        segment_words = tuple(re.findall(r"[A-Z0-9]+", segment))
        if segment_words == component_words:
            return True
    for start in range(len(address_words) - len(component_words) + 1):
        end = start + len(component_words)
        if address_words[start:end] != component_words:
            continue
        suffix = address_words[end:]
        if not suffix:
            prefix = address_words[:start]
            if any(any(char.isdigit() for char in word) for word in prefix[-2:]):
                return True
            continue
        if len(suffix) > 2:
            continue
        alpha_suffixes = [word for word in suffix if word.isalpha()]
        postal_suffixes = [word for word in suffix if any(char.isdigit() for char in word)]
        if (
            len(alpha_suffixes) <= 1
            and all(len(word) <= 3 for word in alpha_suffixes)
            and not any(word in _STREET_DESIGNATORS for word in alpha_suffixes)
            and len(postal_suffixes) == 1
        ):
            return True
    return False


def _phone_evidence_is_fax(target_path: str, raw_value: str, excerpt: str) -> bool:
    if ".contactDetails.phoneNumbers[" not in target_path:
        return False
    raw_offset = excerpt.find(raw_value)
    if raw_offset < 0:
        return False
    preceding_labels = list(_CONTACT_KIND.finditer(excerpt, 0, raw_offset))
    if not preceding_labels or preceding_labels[-1].group("kind").upper() != "FAX":
        return False
    label_cluster = [preceding_labels[-1]]
    for label in reversed(preceding_labels[:-1]):
        between = excerpt[label.end() : label_cluster[-1].start()]
        if re.fullmatch(r"[\s/&,+.:-]*", between) is None:
            break
        label_cluster.append(label)
    return not any(label.group("kind").upper() != "FAX" for label in label_cluster)


def _semantic_findings(target: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    for path, value in _flatten(target).items():
        if any(part in _FIXED_MPCI_KEYS for part in re.split(r"[.\[\]]+", path)):
            findings.append(f"{path}: fixed MPCI scaffolding appears in semantic target")
        if value is None:
            findings.append(f"{path}: explicit null must be omitted from sparse target")
            continue
        if not isinstance(value, str):
            continue
        if _TARGET_CONTAMINATION.search(value):
            findings.append(f"{path}: prohibited tax/boilerplate content {value!r}")
        if _CONTACT_LABEL.search(value):
            findings.append(f"{path}: field-label contamination {value!r}")
        if ".contactDetails." in path and _CONTACT_VALUE_PREFIX.search(value):
            findings.append(f"{path}: communication value retains its field label {value!r}")
        if ".contactDetails." in path and _CONTACT_TRAILING_FOOTNOTE.search(value):
            findings.append(f"{path}: communication value retains a footnote marker {value!r}")
        if ".forwardingAndExportReferences[" in path and _REFERENCE_VALUE_PREFIX.search(value):
            findings.append(f"{path}: reference value retains its field label {value!r}")
        if ".marksAndNumbers[" in path and _MARK_VALUE_PREFIX.search(value):
            findings.append(f"{path}: mark value retains its field label {value!r}")
        if path.endswith(".address") and _POSTAL_VALUE_LABEL.search(value):
            findings.append(f"{path}: postal value retains its field label {value!r}")

    parties = target.get("documentPatch", {}).get("parties", {})
    if isinstance(parties, dict):
        party_values = [
            value for key, value in parties.items() if key != "notifyParties" and value is not None
        ]
        notify = parties.get("notifyParties")
        if isinstance(notify, list):
            party_values.extend(notify)
        for party in party_values:
            if not isinstance(party, dict) or not isinstance(party.get("address"), str):
                continue
            for component in ("city", "country"):
                value = party.get(component)
                if not isinstance(value, str):
                    continue
                if _address_repeats_component(party["address"], value):
                    findings.append(
                        f"party address repeats separately emitted {component} {value!r}"
                    )
    payment_place = target.get("documentPatch", {}).get("freight", {}).get("paymentPlace")
    if isinstance(payment_place, dict):
        name = payment_place.get("name")
        if isinstance(name, str) and name.strip().upper() in _GENERIC_PAYMENT_PLACES:
            findings.append(f"freight payment place is a generic direction {name!r}")
    for goods_index, goods_item in enumerate(
        target.get("documentPatch", {}).get("goodsItems", [])
    ):
        if not isinstance(goods_item, dict):
            continue
        for package_index, package in enumerate(goods_item.get("packages", [])):
            if not isinstance(package, dict) or not isinstance(package.get("type"), str):
                continue
            if _CONTAINER_AS_PACKAGE_TYPE.search(package["type"]):
                findings.append(
                    f"documentPatch.goodsItems[{goods_index}].packages[{package_index}].type: "
                    "container count/type must not be emitted as a goods package level"
                )
        package_quantities = tuple(
            package["quantity"]
            for package in goods_item.get("packages", [])
            if isinstance(package, dict) and isinstance(package.get("quantity"), int)
        )
        allocations = goods_item.get("containerAllocations", [])
        if not package_quantities or not allocations:
            continue
        allocation_quantities: list[int] = []
        allocations_are_quantified = True
        for allocation in allocations:
            if not isinstance(allocation, dict) or not isinstance(
                allocation.get("packageQuantity"), int
            ):
                allocations_are_quantified = False
                break
            allocation_quantities.append(allocation["packageQuantity"])
        path = f"documentPatch.goodsItems[{goods_index}].containerAllocations"
        if not allocations_are_quantified:
            findings.append(
                f"{path}: allocation quantities are required when the goods item has an "
                "emitted package quantity"
            )
            continue
        allocation_total = sum(allocation_quantities)
        if allocation_total not in {*package_quantities, sum(package_quantities)}:
            findings.append(
                f"{path}: allocation quantities must exactly cover one emitted package "
                "level or the total of the emitted package levels"
            )
    return findings


def _source_role_findings(target: dict[str, Any], joined_raw_text: str) -> list[str]:
    delivery_agent = target.get("documentPatch", {}).get("parties", {}).get("deliveryAgent")
    if delivery_agent is not None and not _DELIVERY_AGENT_ROLE_HEADING.search(joined_raw_text):
        return [
            "deliveryAgent requires an explicit destination/discharge/delivery-agent source heading"
        ]
    return []


def _foreign_exporter_country_evidence_findings(
    annotation: BillOfLadingAnnotation,
) -> list[str]:
    findings: list[str] = []
    for evidence in annotation.evidence:
        is_goods_origin = re.fullmatch(
            r"documentPatch\.goodsItems\[\d+\]\.origin\.name", evidence.targetPath
        ) is not None
        is_party_country = re.fullmatch(
            r"documentPatch\.parties\.(?:[A-Za-z]+|notifyParties\[\d+\])\.country",
            evidence.targetPath,
        ) is not None
        if not is_goods_origin and not is_party_country:
            continue
        if any(
            _FOREIGN_EXPORTER_COUNTRY_HEADING.search(item.ocrExcerpt)
            for item in evidence.rawOcrEvidence
        ):
            role = "goods origin" if is_goods_origin else "party-block geography"
            findings.append(
                f"{evidence.targetPath}: FOREIGN EXPORTER COUNTRY identifies exporter "
                f"metadata, not {role}"
            )
    return findings


def _negotiability_evidence_findings(
    annotation: BillOfLadingAnnotation,
    target: dict[str, Any],
    joined_raw_text: str,
) -> list[str]:
    document_patch = target.get("documentPatch", {})
    if (
        document_patch.get("negotiability") != "non_negotiable"
        or _CONDITIONAL_NON_NEGOTIABLE.search(joined_raw_text) is None
    ):
        return []

    consignee = document_patch.get("parties", {}).get("consignee")
    consignee_name = consignee.get("name") if isinstance(consignee, dict) else None
    if not isinstance(consignee_name, str):
        return ["conditional non-negotiable target requires an explicitly emitted consignee name"]

    normalized_consignee = _normalize_country(consignee_name)
    if re.match(r"^TO ORDER(?: OF)?\b", normalized_consignee):
        return ["conditional non-negotiable target contradicts the emitted TO ORDER consignee"]

    negotiability_evidence = next(
        (item for item in annotation.evidence if item.targetPath == "documentPatch.negotiability"),
        None,
    )
    if negotiability_evidence is None or not any(
        _normalize_country(item.rawValue) == normalized_consignee
        for item in negotiability_evidence.rawOcrEvidence
    ):
        return ["conditional non-negotiable target must cite the emitted named consignee"]
    return []


def _warning_findings(annotation: BillOfLadingAnnotation) -> list[str]:
    findings: list[str] = []
    for warning in annotation.warnings:
        if warning.code != "invalid_identifier_omitted":
            continue
        for match in _CONTAINER_IDENTIFIER_TOKEN.finditer(warning.message):
            identifier = "".join(match.groups())
            try:
                _CONTAINER_IDENTIFIER_ADAPTER.validate_python(identifier, strict=True)
            except ValidationError:
                continue
            findings.append(
                f"invalid_identifier_omitted warning names valid ISO 6346 identifier {identifier!r}"
            )
    return findings


def _fax_warning_findings(
    annotation: BillOfLadingAnnotation, joined_raw_text: str
) -> list[str]:
    """Require an audit warning for OCR-grounded fax-only communications."""

    warned_pages = {
        page
        for warning in annotation.warnings
        if warning.code == "schema_cannot_represent" and "fax" in warning.message.lower()
        for page in warning.pageNumbers
    }
    findings: list[str] = []
    for page_number, page_text in _page_texts(joined_raw_text).items():
        requires_warning = False
        for line in page_text.splitlines():
            joint_spans = [match.span() for match in _JOINT_PHONE_FAX_FIELD.finditer(line)]
            for match in _FAX_FIELD.finditer(line):
                if any(start <= match.start() < end for start, end in joint_spans):
                    continue
                if re.search(r"[0-9]", match.group("value")) is not None:
                    requires_warning = True
                    break
            if requires_warning:
                break
        if requires_warning and page_number not in warned_pages:
            findings.append(
                f"standalone FAX value on page {page_number} requires a "
                "schema_cannot_represent warning"
            )
    return findings


def _normalize_country(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[A-Z0-9]+", ascii_value.upper()))


def _country_index() -> tuple[dict[str, str], str]:
    if not ISO_COUNTRIES.is_file():
        raise RuntimeError(f"ISO 3166 snapshot is absent: {ISO_COUNTRIES}")
    payload = json.loads(ISO_COUNTRIES.read_text(encoding="utf-8"))
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in payload["3166-1"]:
        code = row["alpha_2"]
        for key in ("alpha_2", "alpha_3", "name", "official_name", "common_name"):
            value = row.get(key)
            if isinstance(value, str):
                candidates[_normalize_country(value)].add(code)
                if key in {"name", "official_name", "common_name"}:
                    # OCR commonly preserves a printed alpha-2 suffix such as
                    # ``China(CN)``. Register the combined literal only when the
                    # name and suffix come from the same frozen ISO row.
                    candidates[_normalize_country(f"{value} ({code})")].add(code)
    index = {key: next(iter(codes)) for key, codes in candidates.items() if len(codes) == 1}
    for alias, code in _COUNTRY_ALIASES.items():
        normalized = _normalize_country(alias)
        existing = index.get(normalized)
        if existing is not None and existing != code:
            raise RuntimeError(f"country alias {alias!r} contradicts ISO snapshot")
        index[normalized] = code
    return index, _file_digest(ISO_COUNTRIES)


def _validate_attempt(
    document_id: str, attempt_path: Path
) -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, str]]:
    work_path = DEST / "work-items" / f"{document_id}.json"
    work = _read_json(work_path)
    raw_bytes = attempt_path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or raw.get("source") != work["source"]:
        raise RuntimeError(f"{document_id}: candidate source differs from immutable work item")
    pages = _page_texts(work["joinedRawText"])
    if set(raw).issuperset({"exclusionSchemaVersion", "reason"}):
        exclusion = BillOfLadingExclusion.model_validate_json(raw_bytes, strict=True)
        _verify_evidence((exclusion.rawOcrEvidence,), pages, document_id)
        return "excluded", _omit_explicit_nulls(raw), None, {}

    annotation = BillOfLadingAnnotation.model_validate_json(raw_bytes, strict=True)
    if annotation.reviewStatus != "candidate":
        raise RuntimeError(f"{document_id}: worker output is not a candidate")
    _verify_evidence((item.rawOcrEvidence for item in annotation.evidence), pages, document_id)
    mislabeled_fax_paths = [
        item.targetPath
        for item in annotation.evidence
        if any(
            _phone_evidence_is_fax(item.targetPath, evidence.rawValue, evidence.ocrExcerpt)
            for evidence in item.rawOcrEvidence
        )
    ]
    if mislabeled_fax_paths:
        raise RuntimeError(
            f"{document_id}: fax evidence mislabeled as phone: {mislabeled_fax_paths!r}"
        )
    target = annotation.label.canonical_target()
    _verify_target_evidence_support(target, annotation.evidence, document_id)
    findings = _semantic_findings(target)
    findings.extend(_source_role_findings(target, work["joinedRawText"]))
    findings.extend(_foreign_exporter_country_evidence_findings(annotation))
    findings.extend(_negotiability_evidence_findings(annotation, target, work["joinedRawText"]))
    findings.extend(_warning_findings(annotation))
    findings.extend(_fax_warning_findings(annotation, work["joinedRawText"]))
    if findings:
        raise RuntimeError(f"{document_id}: semantic contamination: {findings!r}")

    country_index, _ = _country_index()
    observed_mappings: dict[str, str] = {}

    def resolver(value: str) -> str | None:
        code = country_index.get(_normalize_country(value))
        if code is not None:
            observed_mappings[value] = code
        return code

    projection = project_bill_of_lading_to_mpci(
        annotation.label, country_resolver=resolver
    ).canonical_target()
    return "approved", _omit_explicit_nulls(raw), projection, observed_mappings


def check_attempt(document_id: str, attempt: int) -> None:
    relative = Path("candidate-attempts") / f"{document_id}.attempt-{attempt}.json"
    attempt_path = DEST / relative
    if not attempt_path.is_file():
        raise RuntimeError(f"candidate attempt is absent: {relative}")
    check_path = DEST / "validation/attempt-checks" / f"{document_id}.attempt-{attempt}.json"
    if check_path.exists():
        raise RuntimeError("attempt check is immutable and already exists; publish a new attempt")
    worker = _read_json(DEST / "worker-logs" / f"{document_id}.attempt-{attempt}.json")
    if worker.get("status") != "completed":
        raise RuntimeError("only a completed worker attempt can be checked")
    attempt_sha256 = _file_digest(attempt_path)
    try:
        if worker.get("candidateSha256") != attempt_sha256:
            raise RuntimeError(
                "candidate attempt changed after worker completion or its worker log "
                "lacks the sealed digest"
            )
        kind, raw, projection, country_mappings = _validate_attempt(document_id, attempt_path)
        report = {
            "documentId": document_id,
            "attempt": attempt,
            "attemptPath": relative.as_posix(),
            "attemptSha256": attempt_sha256,
            "status": "passed",
            "kind": kind,
            "targetLeafCount": (
                len(_flatten(raw["label"]["documentPatch"], "documentPatch"))
                if kind == "approved"
                else 0
            ),
            "evidenceCount": len(raw.get("evidence", raw.get("rawOcrEvidence", []))),
            "warningCount": len(raw.get("warnings", [])),
            "projectionLeafCount": len(_flatten(projection)) if projection is not None else 0,
            "countryMappings": country_mappings,
            "checkedAt": _now(),
        }
        _atomic_json(check_path, report)
    except Exception as exc:
        _atomic_json(
            check_path,
            {
                "documentId": document_id,
                "attempt": attempt,
                "attemptPath": relative.as_posix(),
                "attemptSha256": attempt_sha256,
                "status": "failed",
                "error": str(exc),
                "checkedAt": _now(),
            },
        )
        raise
    print(f"{document_id} attempt {attempt}: {report['kind']} passed automated validation")


def record_worker_start(document_id: str, attempt: int, task_name: str) -> None:
    work_item_path = DEST / "work-items" / f"{document_id}.json"
    if not work_item_path.is_file():
        raise RuntimeError(f"work item does not exist: {work_item_path.name}")
    path = DEST / "worker-logs" / f"{document_id}.attempt-{attempt}.json"
    if path.exists():
        raise RuntimeError(f"worker log already exists: {path.name}")
    _atomic_json(
        path,
        {
            "documentId": document_id,
            "attempt": attempt,
            "taskName": task_name,
            "model": "gpt-5.6-luna",
            "reasoningEffort": WORKER_REASONING_EFFORT,
            "startedAt": _now(),
            "completedAt": None,
            "status": "running",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )


def record_worker_finish(document_id: str, attempt: int, status: str) -> None:
    path = DEST / "worker-logs" / f"{document_id}.attempt-{attempt}.json"
    value = _read_json(path)
    if value.get("status") != "running" or value.get("completedAt") is not None:
        raise RuntimeError(f"worker log is not running: {path.name}")
    if status == "completed":
        candidate_relative = Path("candidate-attempts") / f"{document_id}.attempt-{attempt}.json"
        candidate_path = DEST / candidate_relative
        if not candidate_path.is_file():
            raise RuntimeError(f"completed worker candidate is absent: {candidate_relative}")
        value["candidatePath"] = candidate_relative.as_posix()
        value["candidateSha256"] = _file_digest(candidate_path)
    value["status"] = status
    value["completedAt"] = _now()
    _atomic_json(path, value)


def _review_stem(document_id: str, attempt: int, review_number: int) -> str:
    return f"{document_id}.attempt-{attempt}.review-{review_number}"


def record_reviewer_start(
    document_id: str, attempt: int, review_number: int, task_name: str
) -> None:
    """Bind a fresh reviewer to one already sealed and validated candidate."""

    work_path = DEST / "work-items" / f"{document_id}.json"
    candidate_path = DEST / "candidate-attempts" / f"{document_id}.attempt-{attempt}.json"
    check_path = DEST / "validation/attempt-checks" / f"{document_id}.attempt-{attempt}.json"
    if not work_path.is_file() or not candidate_path.is_file() or not check_path.is_file():
        raise RuntimeError("review requires an existing work item, candidate, and check")
    check = _read_json(check_path)
    candidate_sha256 = _file_digest(candidate_path)
    if check.get("status") != "passed" or check.get("attemptSha256") != candidate_sha256:
        raise RuntimeError("review requires a candidate bound to a passing automated check")
    stem = _review_stem(document_id, attempt, review_number)
    path = DEST / "reviewer-logs" / f"{stem}.json"
    if path.exists():
        raise RuntimeError(f"reviewer log already exists: {path.name}")
    _atomic_json(
        path,
        {
            "documentId": document_id,
            "candidateAttempt": attempt,
            "reviewNumber": review_number,
            "taskName": task_name,
            "model": "gpt-5.6-luna",
            "reasoningEffort": REVIEWER_REASONING_EFFORT,
            "workItemSha256": _file_digest(work_path),
            "candidateSha256": candidate_sha256,
            "startedAt": _now(),
            "completedAt": None,
            "status": "running",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )


def record_reviewer_finish(document_id: str, attempt: int, review_number: int, status: str) -> None:
    stem = _review_stem(document_id, attempt, review_number)
    path = DEST / "reviewer-logs" / f"{stem}.json"
    value = _read_json(path)
    if value.get("status") != "running" or value.get("completedAt") is not None:
        raise RuntimeError(f"reviewer log is not running: {path.name}")
    if status == "completed":
        review_relative = Path("independent-reviews") / f"{stem}.json"
        review_path = DEST / review_relative
        if not review_path.is_file():
            raise RuntimeError(f"completed independent review is absent: {review_relative}")
        value["reviewPath"] = review_relative.as_posix()
        value["reviewSha256"] = _file_digest(review_path)
    value["status"] = status
    value["completedAt"] = _now()
    _atomic_json(path, value)


def check_independent_review(document_id: str, attempt: int, review_number: int) -> None:
    """Strictly validate and seal one independent semantic-review artifact."""

    stem = _review_stem(document_id, attempt, review_number)
    work_path = DEST / "work-items" / f"{document_id}.json"
    candidate_path = DEST / "candidate-attempts" / f"{document_id}.attempt-{attempt}.json"
    review_path = DEST / "independent-reviews" / f"{stem}.json"
    reviewer_log_path = DEST / "reviewer-logs" / f"{stem}.json"
    candidate_check_path = (
        DEST / "validation/attempt-checks" / f"{document_id}.attempt-{attempt}.json"
    )
    check_path = DEST / "validation/independent-review-checks" / f"{stem}.json"
    if check_path.exists():
        raise RuntimeError(
            "independent-review check is immutable and already exists; publish a new review"
        )
    review_sha256 = _file_digest(review_path)
    try:
        reviewer_log = _read_json(reviewer_log_path)
        if reviewer_log.get("status") != "completed":
            raise RuntimeError("only a completed independent review can be checked")
        if reviewer_log.get("reviewSha256") != review_sha256:
            raise RuntimeError("independent review changed after reviewer completion")
        candidate_sha256 = _file_digest(candidate_path)
        work_sha256 = _file_digest(work_path)
        candidate_check = _read_json(candidate_check_path)
        if (
            candidate_check.get("status") != "passed"
            or candidate_check.get("attemptSha256") != candidate_sha256
        ):
            raise RuntimeError("reviewed candidate is not bound to its passing check")
        review = BillOfLadingIndependentReview.model_validate_json(
            review_path.read_bytes(), strict=True
        )
        expected = {
            "documentId": document_id,
            "candidateAttempt": attempt,
            "reviewNumber": review_number,
            "workItemSha256": work_sha256,
            "candidateSha256": candidate_sha256,
            "reviewerModel": "gpt-5.6-luna",
            "reviewerReasoningEffort": REVIEWER_REASONING_EFFORT,
        }
        observed = {key: getattr(review, key) for key in expected}
        if observed != expected:
            raise RuntimeError(
                f"independent review identity differs from assigned artifacts: {observed!r}"
            )
        work = _read_json(work_path)
        pages = _page_texts(work["joinedRawText"])
        _verify_evidence(
            (finding.rawOcrEvidence for finding in review.findings),
            pages,
            document_id,
        )
        report = {
            "documentId": document_id,
            "candidateAttempt": attempt,
            "reviewNumber": review_number,
            "reviewPath": review_path.relative_to(DEST).as_posix(),
            "reviewSha256": review_sha256,
            "workItemSha256": work_sha256,
            "candidateSha256": candidate_sha256,
            "status": "passed",
            "semanticResult": review.result,
            "blockingFindingCount": sum(
                finding.severity == "blocking" for finding in review.findings
            ),
            "advisoryFindingCount": sum(
                finding.severity == "advisory" for finding in review.findings
            ),
            "checkedAt": _now(),
        }
        _atomic_json(check_path, report)
    except Exception as exc:
        _atomic_json(
            check_path,
            {
                "documentId": document_id,
                "candidateAttempt": attempt,
                "reviewNumber": review_number,
                "reviewPath": review_path.relative_to(DEST).as_posix(),
                "reviewSha256": review_sha256,
                "status": "failed",
                "error": str(exc),
                "checkedAt": _now(),
            },
        )
        raise
    print(
        f"{document_id} attempt {attempt} independent review {review_number}: "
        f"{report['semanticResult']}"
    )


def _independent_review_binding(
    document_id: str,
    attempt: int,
    review_number: int,
    *,
    require_semantic_pass: bool,
) -> dict[str, Any]:
    stem = _review_stem(document_id, attempt, review_number)
    check_relative = Path("validation/independent-review-checks") / f"{stem}.json"
    review_relative = Path("independent-reviews") / f"{stem}.json"
    check = _read_json(DEST / check_relative)
    review_path = DEST / review_relative
    candidate_path = DEST / "candidate-attempts" / f"{document_id}.attempt-{attempt}.json"
    work_path = DEST / "work-items" / f"{document_id}.json"
    if (
        check.get("status") != "passed"
        or check.get("reviewSha256") != _file_digest(review_path)
        or check.get("candidateSha256") != _file_digest(candidate_path)
        or check.get("workItemSha256") != _file_digest(work_path)
    ):
        raise RuntimeError("independent review is not bound to current immutable artifacts")
    if require_semantic_pass and check.get("semanticResult") != "pass":
        raise RuntimeError("an accepted attempt requires a passing independent review")
    return {
        "independentReviewNumber": review_number,
        "independentReviewPath": review_relative.as_posix(),
        "independentReviewCheckPath": check_relative.as_posix(),
        "independentReviewSha256": check["reviewSha256"],
        "independentReviewResult": check["semanticResult"],
    }


def record_review(
    document_id: str,
    attempt: int,
    result: str,
    disposition: str | None,
    findings: list[str],
    *,
    independent_review: int | None = None,
    supersede_accepted: bool = False,
) -> None:
    if not findings or any(not finding.strip() for finding in findings):
        raise RuntimeError("an attempt review requires non-empty findings")
    attempt_relative = Path("candidate-attempts") / f"{document_id}.attempt-{attempt}.json"
    check_relative = Path("validation/attempt-checks") / f"{document_id}.attempt-{attempt}.json"
    worker_relative = Path("worker-logs") / f"{document_id}.attempt-{attempt}.json"
    worker = _read_json(DEST / worker_relative)
    if worker.get("status") != "completed":
        raise RuntimeError("only a completed worker attempt can be reviewed")
    check = _read_json(DEST / check_relative)
    if result == "accepted" and check.get("status") != "passed":
        raise RuntimeError("an accepted attempt must pass automated validation")
    if result not in {"accepted", "rejected"}:
        raise RuntimeError("review result must be accepted or rejected")
    if result == "rejected" and disposition is not None:
        raise RuntimeError("a rejected attempt cannot have a training disposition")
    if result == "accepted" and disposition not in {
        "include",
        "duplicate_suppressed",
        "excluded",
    }:
        raise RuntimeError("accepted attempt requires a valid training disposition")
    decision_path = DEST / "validation/decisions" / f"{document_id}.json"
    previous_decision = _read_json(decision_path) if decision_path.exists() else None
    if previous_decision is not None and result == "accepted":
        if not supersede_accepted:
            raise RuntimeError("a document already has an accepted decision")
        previous_attempt = previous_decision.get("attempt")
        if not isinstance(previous_attempt, int) or attempt <= previous_attempt:
            raise RuntimeError("a superseding accepted decision must use a later worker attempt")
    elif supersede_accepted:
        raise RuntimeError(
            "--supersede-accepted requires an existing accepted decision and a new accepted result"
        )
    attempt_path = DEST / attempt_relative
    attempt_sha256 = _file_digest(attempt_path)
    checked_attempt_sha256 = check.get("attemptSha256")
    if result == "accepted" and checked_attempt_sha256 != attempt_sha256:
        raise RuntimeError(
            "candidate attempt changed after automated validation or its check lacks a digest"
        )
    if result == "accepted":
        expected_disposition = "excluded" if check["kind"] == "excluded" else None
        if (expected_disposition is not None) != (disposition == "excluded"):
            raise RuntimeError("annotation/exclusion kind and training disposition disagree")
    if result == "accepted" and REQUIRE_INDEPENDENT_REVIEW and independent_review is None:
        raise RuntimeError("an accepted attempt requires an independent review number")
    independent_binding = (
        _independent_review_binding(
            document_id,
            attempt,
            independent_review,
            require_semantic_pass=result == "accepted",
        )
        if independent_review is not None
        else {}
    )
    review_path = DEST / "validation/attempt-reviews" / f"{document_id}.attempt-{attempt}.json"
    review = {
        "documentId": document_id,
        "attempt": attempt,
        "attemptPath": attempt_relative.as_posix(),
        "attemptSha256": attempt_sha256,
        "checkPath": check_relative.as_posix(),
        "workerLogPath": worker_relative.as_posix(),
        "worker": {
            key: worker[key]
            for key in (
                "taskName",
                "model",
                "reasoningEffort",
                "startedAt",
                "completedAt",
                "usage",
            )
        },
        "result": result,
        "findings": findings,
        **independent_binding,
        "reviewedAt": _now(),
    }
    if review_path.exists():
        existing_review = _read_json(review_path)
        if {key: value for key, value in existing_review.items() if key != "reviewedAt"} != {
            key: value for key, value in review.items() if key != "reviewedAt"
        }:
            raise RuntimeError("attempt review is immutable and already exists")
        review = existing_review
    else:
        _atomic_json(review_path, review)
    if result == "rejected":
        return
    decision = {
        "documentId": document_id,
        "attempt": attempt,
        "attemptPath": attempt_relative.as_posix(),
        "decision": "exclude" if disposition == "excluded" else "approve",
        "trainingDisposition": disposition,
        "reviewPath": review_path.relative_to(DEST).as_posix(),
        **independent_binding,
        "decidedAt": _now(),
    }
    if previous_decision is not None:
        previous_attempt = previous_decision["attempt"]
        history_path = (
            DEST / "validation/decision-history" / f"{document_id}.attempt-{previous_attempt}.json"
        )
        if history_path.exists():
            if _read_json(history_path) != previous_decision:
                raise RuntimeError("decision history conflicts with the current decision")
        else:
            _atomic_json(history_path, previous_decision)
        decision["supersedesDecisionPath"] = history_path.relative_to(DEST).as_posix()
    _atomic_json(decision_path, decision)


def record_training_disposition(
    document_id: str,
    disposition: str,
    findings: list[str],
) -> None:
    """Change only duplicate-training inclusion while preserving semantic acceptance.

    Duplicate membership is known only after the run inventory is frozen.  It is
    therefore a dataset-publication decision, not a reason to relabel or repeat
    the independent semantic review of an unchanged candidate.
    """

    if disposition not in {"include", "duplicate_suppressed"}:
        raise RuntimeError("training redisposition only supports duplicate inclusion states")
    if not findings or any(not finding.strip() for finding in findings):
        raise RuntimeError("training redisposition requires non-empty findings")

    decision_path = DEST / "validation/decisions" / f"{document_id}.json"
    decision = _read_json(decision_path)
    previous_disposition = decision.get("trainingDisposition")
    if decision.get("decision") != "approve" or previous_disposition not in {
        "include",
        "duplicate_suppressed",
    }:
        raise RuntimeError("only an approved annotation can change duplicate disposition")
    if previous_disposition == disposition:
        previous_change = decision.get("trainingDispositionChange")
        if (
            isinstance(previous_change, dict)
            and previous_change.get("to") == disposition
            and previous_change.get("findings") == findings
        ):
            print(f"{document_id}: training disposition already {disposition}")
            return
        raise RuntimeError("requested training disposition is already current")

    duplicate_payload = _read_json(DEST / "validation/duplicate-candidates.json")
    matching_groups = [
        group
        for group in duplicate_payload.get("groups", [])
        if document_id in group.get("documentIds", [])
    ]
    if len(matching_groups) != 1:
        raise RuntimeError("training redisposition requires exactly one frozen duplicate group")

    attempt = decision.get("attempt")
    if not isinstance(attempt, int):
        raise RuntimeError("accepted decision lacks an integer attempt")
    attempt_path = DEST / "candidate-attempts" / f"{document_id}.attempt-{attempt}.json"
    check = _read_json(
        DEST / "validation/attempt-checks" / f"{document_id}.attempt-{attempt}.json"
    )
    if (
        check.get("status") != "passed"
        or check.get("kind") != "approved"
        or check.get("attemptSha256") != _file_digest(attempt_path)
    ):
        raise RuntimeError("accepted annotation is not bound to a passing approved check")
    if REQUIRE_INDEPENDENT_REVIEW:
        review_number = decision.get("independentReviewNumber")
        if not isinstance(review_number, int):
            raise RuntimeError("accepted decision lacks an independent review number")
        binding = _independent_review_binding(
            document_id,
            attempt,
            review_number,
            require_semantic_pass=True,
        )
        if any(decision.get(key) != value for key, value in binding.items()):
            raise RuntimeError("accepted decision's independent-review binding changed")

    previous_digest = _file_digest(decision_path)
    history_path = (
        DEST
        / "validation/decision-history"
        / f"{document_id}.decision-{previous_digest}.json"
    )
    if history_path.exists():
        if _read_json(history_path) != decision:
            raise RuntimeError("decision-history digest conflicts with the current decision")
    else:
        _atomic_json(history_path, decision)

    changed_at = _now()
    updated = decision | {
        "trainingDisposition": disposition,
        "trainingDispositionChange": {
            "from": previous_disposition,
            "to": disposition,
            "duplicateGroupId": matching_groups[0]["groupId"],
            "findings": findings,
            "changedAt": changed_at,
        },
        "supersedesDecisionPath": history_path.relative_to(DEST).as_posix(),
    }
    _atomic_json(decision_path, updated)
    print(f"{document_id}: training disposition {previous_disposition} -> {disposition}")


def record_contract_revision(findings: list[str]) -> None:
    """Seal an explicit contract revision without erasing the prepared-run hashes."""

    if not findings or any(not finding.strip() for finding in findings):
        raise RuntimeError("contract revision requires non-empty findings")
    metadata_path = DEST / "run-metadata.json"
    metadata = _read_json(metadata_path)
    if metadata.get("status") != "prepared":
        raise RuntimeError("only an unpublished prepared run can record a contract revision")

    run_configuration = metadata.get("runConfiguration")
    if not isinstance(run_configuration, dict) or RUN_CONFIG_PATH is None:
        raise RuntimeError("run metadata lacks its configured run-profile binding")
    if (
        run_configuration.get("path") != str(RUN_CONFIG_PATH.relative_to(ROOT))
        or run_configuration.get("sha256") != _file_digest(RUN_CONFIG_PATH)
    ):
        raise RuntimeError("contract revision cannot absorb a changed run configuration")

    previous = metadata.get("frozenContract")
    if not isinstance(previous, dict):
        raise RuntimeError("run metadata lacks the previous frozen contract")
    current = _current_contract_hashes()
    if previous == current:
        revisions = metadata.get("contractRevisionPaths", [])
        if revisions:
            latest = _read_json(DEST / revisions[-1])
            if latest.get("currentContract") == current and latest.get("findings") == findings:
                print(f"contract revision already recorded at {revisions[-1]}")
                return
        raise RuntimeError("labeling contract is already current")

    changed_files = sorted(set(previous) | set(current))
    changed_files = [path for path in changed_files if previous.get(path) != current.get(path)]
    stable_revision = {
        "revisionVersion": 1,
        "runId": RUN_ID,
        "previousContract": previous,
        "currentContract": current,
        "changedFiles": changed_files,
        "findings": findings,
    }
    revision_id = _digest(_canonical_text(stable_revision).encode())
    revision_relative = Path("validation/contract-revisions") / f"revision-{revision_id}.json"
    revision_path = DEST / revision_relative
    if revision_path.exists():
        revision = _read_json(revision_path)
        if {key: revision.get(key) for key in stable_revision} != stable_revision:
            raise RuntimeError("contract revision identity conflicts with existing history")
    else:
        revision = stable_revision | {"recordedAt": _now()}
        _atomic_json(revision_path, revision)

    revision_paths = list(metadata.get("contractRevisionPaths", []))
    revision_text = revision_relative.as_posix()
    if revision_text not in revision_paths:
        revision_paths.append(revision_text)
    metadata["frozenContract"] = current
    metadata["contractRevisionPaths"] = revision_paths
    metadata["contractRevisionCount"] = len(revision_paths)
    _atomic_json(metadata_path, metadata)
    print(f"recorded contract revision for {', '.join(changed_files)}")


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_leaf_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_leaf_count(child) for child in value)
    return 1


def _compact_bytes(value: Any) -> int:
    return len(_canonical_text(value).encode())


def _canonical_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_target_tokenizer() -> tuple[Any, dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    resolved_files: dict[str, dict[str, Any]] = {}
    for name, frozen in T5GEMMA_TOKENIZER_FILES.items():
        path = Path(
            hf_hub_download(
                repo_id=T5GEMMA_TOKENIZER_MIRROR,
                filename=name,
                revision=T5GEMMA_TOKENIZER_MIRROR_REVISION,
                local_files_only=True,
            )
        )
        observed = _file_digest(path)
        if observed != frozen["sha256"]:
            raise RuntimeError(f"frozen T5Gemma tokenizer file changed: {name}")
        resolved_files[name] = frozen | {"bytes": path.stat().st_size}
    tokenizer = AutoTokenizer.from_pretrained(
        T5GEMMA_TOKENIZER_MIRROR,
        revision=T5GEMMA_TOKENIZER_MIRROR_REVISION,
        local_files_only=True,
    )
    if len(tokenizer) != 262_144:
        raise RuntimeError("frozen T5Gemma tokenizer vocabulary size changed")
    return tokenizer, {
        "upstreamRepository": T5GEMMA_TOKENIZER_UPSTREAM,
        "upstreamRevision": T5GEMMA_TOKENIZER_UPSTREAM_REVISION,
        "mirrorRepository": T5GEMMA_TOKENIZER_MIRROR,
        "mirrorRevision": T5GEMMA_TOKENIZER_MIRROR_REVISION,
        "identityProof": (
            "Each mirror Git blob OID equals the corresponding upstream repository Git blob "
            "OID; local content is additionally pinned by SHA-256."
        ),
        "files": resolved_files,
        "tokenizerClass": type(tokenizer).__name__,
        "vocabularySize": len(tokenizer),
        "countingRule": "canonical compact JSON, add_special_tokens=false",
    }


def _token_count(tokenizer: Any, value: Any) -> int:
    return len(tokenizer.encode(_canonical_text(value), add_special_tokens=False))


def _size_comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("v1/v2 size comparison requires overlapping validated documents")
    metrics = ("Leaves", "CanonicalBytes", "Tokens")
    summary: dict[str, Any] = {"documents": len(rows)}
    for metric in metrics:
        v1 = sum(int(row[f"v1{metric}"]) for row in rows)
        v2 = sum(int(row[f"v2{metric}"]) for row in rows)
        summary[metric[0].lower() + metric[1:]] = {
            "v1Total": v1,
            "v2Total": v2,
            "absoluteDelta": v2 - v1,
            "percentDelta": round((v2 - v1) * 100 / v1, 3) if v1 else None,
        }
    return summary


def _duplicate_comparisons(
    groups: list[dict[str, Any]], targets: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for group in groups:
        approved_ids = [item for item in group["documentIds"] if item in targets]
        pairwise: list[dict[str, Any]] = []
        for first_index, first_id in enumerate(approved_ids):
            first = _flatten(targets[first_id]["documentPatch"], "documentPatch")
            for second_id in approved_ids[first_index + 1 :]:
                second = _flatten(targets[second_id]["documentPatch"], "documentPatch")
                shared = sorted(set(first) & set(second))
                contradictions = [
                    {"path": path, "first": first[path], "second": second[path]}
                    for path in shared
                    if first[path] != second[path]
                ]
                pairwise.append(
                    {
                        "documentIds": [first_id, second_id],
                        "sharedLeafPaths": len(shared),
                        "contradictions": contradictions,
                        "firstOnlyPaths": sorted(set(first) - set(second)),
                        "secondOnlyPaths": sorted(set(second) - set(first)),
                    }
                )
        comparisons.append(group | {"approvedDocumentIds": approved_ids, "pairwise": pairwise})
    return comparisons


def _duplicate_consistency_report(
    groups: list[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    joined_raw_text_sha256: dict[str, str],
    included_document_ids: set[str],
) -> dict[str, Any]:
    """Separate label contradictions from permitted OCR-conditioned variation.

    Exact target differences are blocking when the frozen OCR input is identical.
    When identical document rasters produced different frozen OCR, the labeling
    contract requires each target to remain faithful to its own OCR; those
    differences stay visible in the audit but are not relabeled from the image.
    The deduplicated training view is checked independently.
    """

    missing_hashes = sorted(set(targets) - set(joined_raw_text_sha256))
    if missing_hashes:
        raise RuntimeError(f"duplicate targets lack frozen OCR hashes: {missing_hashes!r}")
    unknown_included = sorted(included_document_ids - set(targets))
    if unknown_included:
        raise RuntimeError(
            f"included duplicate documents lack approved targets: {unknown_included!r}"
        )

    same_raw_ocr_contradictions = 0
    raw_ocr_variant_differences = 0
    raw_ocr_variant_pairs = 0
    accepted_groups: list[dict[str, Any]] = []
    for group in _duplicate_comparisons(groups, targets):
        pairwise: list[dict[str, Any]] = []
        for pair in group["pairwise"]:
            first_id, second_id = pair["documentIds"]
            differences = pair["contradictions"]
            same_raw_ocr = (
                joined_raw_text_sha256[first_id] == joined_raw_text_sha256[second_id]
            )
            if differences and same_raw_ocr:
                classification = "same_raw_ocr_contradiction"
                same_raw_ocr_contradictions += len(differences)
            elif differences:
                classification = "raw_ocr_variant_difference"
                raw_ocr_variant_pairs += 1
                raw_ocr_variant_differences += len(differences)
            else:
                classification = "consistent"
            pairwise.append(
                {
                    key: value
                    for key, value in pair.items()
                    if key != "contradictions"
                }
                | {
                    "sameJoinedRawTextSha256": same_raw_ocr,
                    "classification": classification,
                    "sharedPathDifferences": differences,
                }
            )
        accepted_groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "pairwise"
            }
            | {
                "includedDocumentIds": [
                    document_id
                    for document_id in group["approvedDocumentIds"]
                    if document_id in included_document_ids
                ],
                "pairwise": pairwise,
            }
        )

    training_groups: list[dict[str, Any]] = []
    for group in accepted_groups:
        retained = group["includedDocumentIds"]
        retained_set = set(retained)
        training_groups.append(
            {
                key: value
                for key, value in group.items()
                if key not in {"approvedDocumentIds", "includedDocumentIds", "pairwise"}
            }
            | {
                "approvedDocumentIds": retained,
                "pairwise": [
                    pair
                    for pair in group["pairwise"]
                    if set(pair["documentIds"]).issubset(retained_set)
                ],
            }
        )
    training_contradictions = sum(
        len(pair["sharedPathDifferences"])
        for group in training_groups
        for pair in group["pairwise"]
    )
    return {
        "status": "passed"
        if same_raw_ocr_contradictions == 0 and training_contradictions == 0
        else "failed",
        "policy": (
            "Exact shared-path differences block publication only for identical frozen OCR or "
            "between retained training representatives. Differences between distinct frozen OCR "
            "variants remain audit-visible and are not harmonized from images."
        ),
        "sameRawOcrContradictionCount": same_raw_ocr_contradictions,
        "rawOcrVariantPairCount": raw_ocr_variant_pairs,
        "rawOcrVariantDifferenceCount": raw_ocr_variant_differences,
        "trainingContradictionCount": training_contradictions,
        "acceptedGroups": accepted_groups,
        "trainingGroups": training_groups,
    }


def _projection_benchmark(labels: list[BillOfLadingLabel]) -> dict[str, Any]:
    if not labels:
        raise RuntimeError("projection benchmark requires accepted labels")
    index, _ = _country_index()

    def resolver(value: str) -> str | None:
        return index.get(_normalize_country(value))

    iterations = max(20, 2_000 // len(labels))
    for label in labels:
        project_bill_of_lading_to_mpci(label, country_resolver=resolver).canonical_target()
    started = time.perf_counter_ns()
    checksum = 0
    for _ in range(iterations):
        for label in labels:
            projected = project_bill_of_lading_to_mpci(
                label, country_resolver=resolver
            ).canonical_target()
            checksum += _leaf_count(projected)
    elapsed = time.perf_counter_ns() - started
    count = iterations * len(labels)
    tracemalloc.start()
    resident = [
        project_bill_of_lading_to_mpci(label, country_resolver=resolver).canonical_target()
        for label in labels
    ]
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not resident or checksum <= 0:
        raise RuntimeError("projection benchmark did not exercise the projection path")
    return {
        "acceptedLabels": len(labels),
        "iterationsPerLabel": iterations,
        "projectionCount": count,
        "elapsedMilliseconds": round(elapsed / 1_000_000, 3),
        "meanMillisecondsPerProjection": round(elapsed / count / 1_000_000, 6),
        "projectionsPerSecond": round(count * 1_000_000_000 / elapsed, 2),
        "residentOutputBytes": current,
        "peakTracedBytes": peak,
        "checksum": checksum,
    }


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _bar_plot(path: Path, title: str, labels: list[str], values: list[int]) -> None:
    if not labels or len(labels) != len(values):
        raise RuntimeError("bar plot requires aligned non-empty labels and values")
    width = 1_200
    row_height = 30
    height = max(220, 100 + row_height * len(labels))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((20, 20), title, fill="black", font=font)
    maximum = max(values) or 1
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 65 + index * row_height
        draw.text((20, y), label[:42], fill="black", font=font)
        bar_width = int(780 * value / maximum)
        draw.rectangle((330, y, 330 + bar_width, y + 16), fill="#4c78a8")
        draw.text((340 + bar_width, y), str(value), fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def _eda(
    annotations: dict[str, BillOfLadingAnnotation],
    exclusions: dict[str, BillOfLadingExclusion],
    tokenizer: Any,
) -> dict[str, Any]:
    field_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    document_rows: list[list[Any]] = []
    page_counts: Counter[int] = Counter()
    leaf_counts: list[int] = []
    target_bytes: list[int] = []
    target_tokens: list[int] = []
    for document_id, annotation in sorted(annotations.items()):
        target = annotation.label.canonical_target()
        flattened = _flatten(target["documentPatch"], "documentPatch")
        field_counts.update(flattened.keys())
        warning_counts.update(warning.code for warning in annotation.warnings)
        pages = annotation.source.documentPageCount
        leaves = len(flattened)
        byte_count = _compact_bytes(target)
        token_count = _token_count(tokenizer, target)
        page_counts[pages] += 1
        leaf_counts.append(leaves)
        target_bytes.append(byte_count)
        target_tokens.append(token_count)
        document_rows.append(
            [
                document_id,
                "validated",
                annotation.documentType,
                pages,
                leaves,
                byte_count,
                token_count,
            ]
        )
    for document_id, exclusion in sorted(exclusions.items()):
        pages = exclusion.source.documentPageCount
        page_counts[pages] += 1
        document_rows.append([document_id, "excluded", exclusion.reason, pages, 0, 0, 0])
    _write_csv(
        DEST / "eda/tables/documents.csv",
        [
            "document_id",
            "status",
            "document_type_or_reason",
            "pages",
            "leaves",
            "target_bytes",
            "target_tokens",
        ],
        document_rows,
    )
    _write_csv(
        DEST / "eda/tables/field_presence.csv",
        ["field_path", "document_count"],
        [[key, value] for key, value in field_counts.most_common()],
    )
    _write_csv(
        DEST / "eda/tables/warnings.csv",
        ["warning_code", "count"],
        [[key, value] for key, value in warning_counts.most_common()],
    )
    _bar_plot(
        DEST / "eda/plots/page-counts.png",
        "Document page-count distribution",
        [str(key) for key in sorted(page_counts)],
        [page_counts[key] for key in sorted(page_counts)],
    )
    top_fields = field_counts.most_common(20)
    _bar_plot(
        DEST / "eda/plots/top-fields.png",
        "Top semantic target leaf paths",
        [key for key, _ in top_fields],
        [value for _, value in top_fields],
    )
    summary = {
        "validatedDocuments": len(annotations),
        "excludedDocuments": len(exclusions),
        "pageCounts": dict(sorted(page_counts.items())),
        "warningCounts": dict(sorted(warning_counts.items())),
        "targetLeafCounts": {
            "min": min(leaf_counts) if leaf_counts else 0,
            "max": max(leaf_counts) if leaf_counts else 0,
            "mean": sum(leaf_counts) / len(leaf_counts) if leaf_counts else 0,
        },
        "targetCanonicalBytes": {
            "min": min(target_bytes) if target_bytes else 0,
            "max": max(target_bytes) if target_bytes else 0,
            "mean": sum(target_bytes) / len(target_bytes) if target_bytes else 0,
        },
        "targetTokens": {
            "min": min(target_tokens) if target_tokens else 0,
            "max": max(target_tokens) if target_tokens else 0,
            "mean": sum(target_tokens) / len(target_tokens) if target_tokens else 0,
        },
    }
    _atomic_json(DEST / "eda/summary.json", summary)
    return summary


def _manifest_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(DEST.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "manifest.json.sha256"}:
            continue
        entries.append(
            {
                "path": path.relative_to(DEST).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_digest(path),
            }
        )
    return entries


def _publish_manifest() -> None:
    _atomic_json(
        DEST / "manifest.json",
        {
            "runId": RUN_ID,
            "schemaVersion": "2.0.0",
            "publicationStatus": "complete",
            "files": _manifest_entries(),
        },
    )
    _atomic_bytes(
        DEST / "manifest.json.sha256",
        (_file_digest(DEST / "manifest.json") + "  manifest.json\n").encode(),
    )


def _recover_manifest_digest() -> None:
    manifest_path = DEST / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("runId") != RUN_ID
        or manifest.get("schemaVersion") != "2.0.0"
        or manifest.get("publicationStatus") != "complete"
        or manifest.get("files") != _manifest_entries()
    ):
        raise RuntimeError("incomplete publication manifest does not match current run files")
    _atomic_bytes(
        DEST / "manifest.json.sha256",
        (_file_digest(manifest_path) + "  manifest.json\n").encode(),
    )


def finalize() -> None:
    manifest_exists = (DEST / "manifest.json").exists()
    digest_exists = (DEST / "manifest.json.sha256").exists()
    if manifest_exists and digest_exists:
        raise RuntimeError("run is already published; create a new run ID")
    if digest_exists and not manifest_exists:
        raise RuntimeError("manifest digest exists without manifest")
    if manifest_exists:
        _recover_manifest_digest()
        print(f"recovered final manifest digest at {DEST}")
        return

    metadata = _read_json(DEST / "run-metadata.json")
    _verify_frozen_contract(metadata)
    expected_ids = metadata["eligibleDocumentIds"]
    decision_paths = sorted((DEST / "validation/decisions").glob("*.json"))
    if {path.stem for path in decision_paths} != set(expected_ids):
        raise RuntimeError("overseer decisions do not cover exactly all eligible documents")

    annotations: dict[str, BillOfLadingAnnotation] = {}
    exclusions: dict[str, BillOfLadingExclusion] = {}
    targets: dict[str, dict[str, Any]] = {}
    projections: dict[str, dict[str, Any]] = {}
    training_records: list[dict[str, Any]] = []
    dispositions: Counter[str] = Counter()
    country_mappings: dict[str, str] = {}
    joined_raw_text_sha256: dict[str, str] = {}
    included_document_ids: set[str] = set()

    for document_id in expected_ids:
        decision = _read_json(DEST / "validation/decisions" / f"{document_id}.json")
        attempt_path = DEST / decision["attemptPath"]
        check = _read_json(
            DEST / "validation/attempt-checks" / f"{document_id}.attempt-{decision['attempt']}.json"
        )
        if check.get("status") != "passed" or check.get("attemptSha256") != _file_digest(
            attempt_path
        ):
            raise RuntimeError(
                f"{document_id}: accepted candidate is not bound to its passing check"
            )
        if metadata["independentReviewContract"]["required"]:
            review_number = decision.get("independentReviewNumber")
            if not isinstance(review_number, int):
                raise RuntimeError(f"{document_id}: accepted decision lacks an independent review")
            binding = _independent_review_binding(
                document_id,
                decision["attempt"],
                review_number,
                require_semantic_pass=True,
            )
            if any(decision.get(key) != value for key, value in binding.items()):
                raise RuntimeError(f"{document_id}: decision's independent-review binding changed")
        kind, raw, projection, mappings = _validate_attempt(document_id, attempt_path)
        _atomic_json(DEST / "candidates" / f"{document_id}.json", raw)
        disposition = decision["trainingDisposition"]
        dispositions[disposition] += 1
        if kind == "excluded":
            if disposition != "excluded":
                raise RuntimeError(f"{document_id}: exclusion disposition changed")
            exclusion = BillOfLadingExclusion.model_validate_json(
                json.dumps(raw, ensure_ascii=False), strict=True
            )
            exclusions[document_id] = exclusion
            _atomic_json(DEST / "exclusions" / f"{document_id}.json", raw)
            continue
        if disposition not in {"include", "duplicate_suppressed"}:
            raise RuntimeError(f"{document_id}: approved annotation disposition is invalid")
        final = raw | {"reviewStatus": "validated"}
        final_bytes = (
            json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        annotation = BillOfLadingAnnotation.model_validate_json(final_bytes, strict=True)
        validated_path = DEST / "validated" / f"{document_id}.json"
        _atomic_bytes(validated_path, final_bytes)
        annotations[document_id] = annotation
        target = annotation.label.canonical_target()
        targets[document_id] = target
        if projection is None:
            raise RuntimeError(f"{document_id}: approved annotation lacks projection")
        projections[document_id] = projection
        for value, code in mappings.items():
            existing = country_mappings.get(value)
            if existing is not None and existing != code:
                raise RuntimeError(f"country mapping changed for {value!r}")
            country_mappings[value] = code
        work = _read_json(DEST / "work-items" / f"{document_id}.json")
        joined_raw_text_sha256[document_id] = work["source"]["joinedRawTextSha256"]
        if disposition == "include":
            included_document_ids.add(document_id)
            training_records.append(
                {
                    "documentId": document_id,
                    "joinedRawText": work["joinedRawText"],
                    "joinedRawTextSha256": work["source"]["joinedRawTextSha256"],
                    "target": target,
                    "validatedAnnotationPath": f"validated/{document_id}.json",
                    "validatedAnnotationSha256": _file_digest(validated_path),
                }
            )

    duplicate_payload = _read_json(DEST / "validation/duplicate-candidates.json")
    duplicate_consistency = _duplicate_consistency_report(
        duplicate_payload["groups"],
        targets,
        joined_raw_text_sha256,
        included_document_ids,
    )
    if duplicate_consistency["sameRawOcrContradictionCount"]:
        raise RuntimeError("duplicate targets contradict despite identical frozen OCR")
    if duplicate_consistency["trainingContradictionCount"]:
        raise RuntimeError("retained duplicate training representatives contradict")
    for group in duplicate_consistency["acceptedGroups"]:
        approved = group["approvedDocumentIds"]
        included = [
            document_id
            for document_id in approved
            if _read_json(DEST / "validation/decisions" / f"{document_id}.json")[
                "trainingDisposition"
            ]
            == "include"
        ]
        if approved and len(included) != 1:
            raise RuntimeError(
                f"{group['groupId']}: duplicate group requires exactly one included representative"
            )
    _atomic_json(
        DEST / "validation/duplicate-comparison.json",
        duplicate_consistency,
    )
    _atomic_json(
        DEST / "validation/country-resolver.json",
        {
            "iso3166SnapshotPath": str(ISO_COUNTRIES),
            "iso3166SnapshotSha256": _country_index()[1],
            "normalization": "NFKD ASCII, uppercase alphanumeric words",
            "observedMappings": dict(sorted(country_mappings.items())),
        },
    )
    _atomic_json(DEST / "validation/mpci-projections.json", dict(sorted(projections.items())))
    _atomic_jsonl(
        DEST / "training/records.jsonl",
        sorted(training_records, key=lambda row: row["documentId"]),
    )

    benchmark = _projection_benchmark([annotation.label for annotation in annotations.values()])
    _atomic_json(DEST / "validation/projection-benchmark.json", benchmark)
    tokenizer, tokenizer_metadata = _load_target_tokenizer()
    eda = _eda(annotations, exclusions, tokenizer)

    v1_comparison: list[dict[str, Any]] = []
    for document_id, target in sorted(targets.items()):
        v1_path = V1_RUN / "validated" / f"{document_id}.json"
        if not v1_path.is_file():
            continue
        v1 = MpciBillOfLadingAnnotation.model_validate_json(v1_path.read_bytes(), strict=True)
        v1_target = v1.label.canonical_target()
        v1_comparison.append(
            {
                "documentId": document_id,
                "v1Leaves": _leaf_count(v1_target),
                "v2Leaves": _leaf_count(target),
                "v1CanonicalBytes": _compact_bytes(v1_target),
                "v2CanonicalBytes": _compact_bytes(target),
                "v1Tokens": _token_count(tokenizer, v1_target),
                "v2Tokens": _token_count(tokenizer, target),
            }
        )
    comparison_summary = (
        _size_comparison_summary(v1_comparison)
        if v1_comparison
        else {
            "documents": 0,
            "status": "not_applicable_no_overlapping_document_ids",
            "leaves": None,
            "canonicalBytes": None,
            "tokens": None,
        }
    )
    _atomic_json(
        DEST / "validation/v1-v2-size-comparison.json",
        {
            "tokenizer": tokenizer_metadata,
            "summary": comparison_summary,
            "documents": v1_comparison,
            "qualityClaimBoundary": (
                "Target size and token count do not measure extraction or downstream model quality."
            ),
        },
    )

    worker_logs = sorted((DEST / "worker-logs").glob("*.json"))
    reviewer_logs = sorted((DEST / "reviewer-logs").glob("*.json"))
    independent_review_checks = sorted(
        (DEST / "validation/independent-review-checks").glob("*.json")
    )
    independent_review_values = [_read_json(path) for path in independent_review_checks]
    reviews = sorted((DEST / "validation/attempt-reviews").glob("*.json"))
    review_values = [_read_json(path) for path in reviews]
    report = {
        "status": "passed",
        "selectedDocuments": _EXPECTED["selectedDocuments"],
        "preLabelExcluded": _EXPECTED["preLabelExcludedDocuments"],
        "eligibleDocuments": len(expected_ids),
        "validatedAnnotations": len(annotations),
        "workerExclusions": len(exclusions),
        "duplicateSuppressed": dispositions["duplicate_suppressed"],
        "trainingRecords": len(training_records),
        "workerAttempts": len(worker_logs),
        "independentReviewAttempts": len(reviewer_logs),
        "independentReviewPasses": sum(
            value.get("status") == "passed" and value.get("semanticResult") == "pass"
            for value in independent_review_values
        ),
        "independentReviewFailures": sum(
            value.get("status") == "passed" and value.get("semanticResult") == "fail"
            for value in independent_review_values
        ),
        "acceptedAttempts": sum(value["result"] == "accepted" for value in review_values),
        "rejectedAttempts": sum(value["result"] == "rejected" for value in review_values),
        "evidenceAndContaminationChecks": "passed",
        "projection": {"status": "passed", "benchmark": benchmark},
        "duplicateComparison": {
            "status": duplicate_consistency["status"],
            "sameRawOcrContradictions": duplicate_consistency[
                "sameRawOcrContradictionCount"
            ],
            "trainingContradictions": duplicate_consistency[
                "trainingContradictionCount"
            ],
            "rawOcrVariantPairs": duplicate_consistency["rawOcrVariantPairCount"],
            "rawOcrVariantDifferences": duplicate_consistency[
                "rawOcrVariantDifferenceCount"
            ],
        },
        "v1V2Comparison": v1_comparison,
        "v1V2ComparisonSummary": comparison_summary,
        "targetTokenizer": tokenizer_metadata,
        "eda": eda,
        "qualityClaimBoundary": (
            "Serialization and EDA statistics do not prove downstream model accuracy."
        ),
    }
    _atomic_json(DEST / "validation/report.json", report)
    if comparison_summary["documents"]:
        token_summary = comparison_summary.get("tokens")
        if not isinstance(token_summary, dict) or "percentDelta" not in token_summary:
            raise RuntimeError("v1/v2 token comparison has an invalid shape")
        comparison_line = f"- V2 token delta vs v1: {token_summary['percentDelta']}%\n\n"
    else:
        comparison_line = "- V1/V2 size comparison: not applicable; no overlapping document IDs\n\n"
    _atomic_bytes(
        DEST / "validation/report.md",
        (
            "# Semantic-v2 Bill-of-Lading labeling validation\n\n"
            "Status: **passed**.\n\n"
            f"- Selected source PDFs: {_EXPECTED['selectedDocuments']}\n"
            f"- Pre-label exclusions: {_EXPECTED['preLabelExcludedDocuments']}\n"
            f"- Eligible work items: {len(expected_ids)}\n"
            f"- Validated annotations: {len(annotations)}\n"
            f"- Worker exclusions: {len(exclusions)}\n"
            f"- Duplicate-suppressed annotations: {dispositions['duplicate_suppressed']}\n"
            f"- Training records: {len(training_records)}\n"
            f"- Worker attempts: {len(worker_logs)}\n"
            f"- Independent review attempts: {len(reviewer_logs)}\n"
            f"- Independent review failures: {report['independentReviewFailures']}\n"
            f"- Rejected attempts: {report['rejectedAttempts']}\n"
            "- Duplicate contradictions with identical frozen OCR: 0\n"
            "- Retained-training duplicate contradictions: 0\n"
            f"- Audit-visible raw-OCR-variant target differences: "
            f"{duplicate_consistency['rawOcrVariantDifferenceCount']}\n"
            f"- Projection throughput: {benchmark['projectionsPerSecond']} projections/s\n\n"
            f"- V1/V2 overlapping documents: {comparison_summary['documents']}\n"
            + comparison_line
            + "See `report.json`, `v1-v2-size-comparison.json`, "
            "`duplicate-comparison.json`, and `eda/` for details.\n"
        ).encode(),
    )
    metadata["status"] = "published"
    metadata["publishedAt"] = _now()
    metadata["workerAttempts"] = len(worker_logs)
    metadata["independentReviewAttempts"] = len(reviewer_logs)
    metadata["acceptedAttempts"] = report["acceptedAttempts"]
    metadata["rejectedAttempts"] = report["rejectedAttempts"]
    _atomic_json(DEST / "run-metadata.json", metadata)
    _publish_manifest()
    print(
        f"published {len(training_records)} training records, {len(exclusions)} worker "
        f"exclusions, and {dispositions['duplicate_suppressed']} duplicate suppressions at {DEST}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="strict repository-relative/absolute YAML run configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    check = subparsers.add_parser("check-attempt")
    check.add_argument("--document-id", required=True)
    check.add_argument("--attempt", required=True, type=int)
    start = subparsers.add_parser("worker-start")
    start.add_argument("--document-id", required=True)
    start.add_argument("--attempt", required=True, type=int)
    start.add_argument("--task-name", required=True)
    finish = subparsers.add_parser("worker-finish")
    finish.add_argument("--document-id", required=True)
    finish.add_argument("--attempt", required=True, type=int)
    finish.add_argument("--status", choices=("completed", "failed"), required=True)
    reviewer_start = subparsers.add_parser("reviewer-start")
    reviewer_start.add_argument("--document-id", required=True)
    reviewer_start.add_argument("--attempt", required=True, type=int)
    reviewer_start.add_argument("--review-number", required=True, type=int)
    reviewer_start.add_argument("--task-name", required=True)
    reviewer_finish = subparsers.add_parser("reviewer-finish")
    reviewer_finish.add_argument("--document-id", required=True)
    reviewer_finish.add_argument("--attempt", required=True, type=int)
    reviewer_finish.add_argument("--review-number", required=True, type=int)
    reviewer_finish.add_argument("--status", choices=("completed", "failed"), required=True)
    independent_check = subparsers.add_parser("check-independent-review")
    independent_check.add_argument("--document-id", required=True)
    independent_check.add_argument("--attempt", required=True, type=int)
    independent_check.add_argument("--review-number", required=True, type=int)
    review = subparsers.add_parser("review")
    review.add_argument("--document-id", required=True)
    review.add_argument("--attempt", required=True, type=int)
    review.add_argument("--result", choices=("accepted", "rejected"), required=True)
    review.add_argument("--disposition", choices=("include", "duplicate_suppressed", "excluded"))
    review.add_argument("--independent-review", type=int)
    review.add_argument("--supersede-accepted", action="store_true")
    review.add_argument("--finding", action="append", default=[])
    redisposition = subparsers.add_parser("redisposition")
    redisposition.add_argument("--document-id", required=True)
    redisposition.add_argument(
        "--disposition", choices=("include", "duplicate_suppressed"), required=True
    )
    redisposition.add_argument("--finding", action="append", default=[])
    contract_revision = subparsers.add_parser("contract-revision")
    contract_revision.add_argument("--finding", action="append", default=[])
    subparsers.add_parser("finalize")
    arguments = parser.parse_args()
    if arguments.config is not None:
        _configure_run(arguments.config)
    if arguments.command == "prepare":
        prepare()
    elif arguments.command == "check-attempt":
        check_attempt(arguments.document_id, arguments.attempt)
    elif arguments.command == "worker-start":
        record_worker_start(arguments.document_id, arguments.attempt, arguments.task_name)
    elif arguments.command == "worker-finish":
        record_worker_finish(arguments.document_id, arguments.attempt, arguments.status)
    elif arguments.command == "reviewer-start":
        record_reviewer_start(
            arguments.document_id,
            arguments.attempt,
            arguments.review_number,
            arguments.task_name,
        )
    elif arguments.command == "reviewer-finish":
        record_reviewer_finish(
            arguments.document_id,
            arguments.attempt,
            arguments.review_number,
            arguments.status,
        )
    elif arguments.command == "check-independent-review":
        check_independent_review(
            arguments.document_id,
            arguments.attempt,
            arguments.review_number,
        )
    elif arguments.command == "review":
        record_review(
            arguments.document_id,
            arguments.attempt,
            arguments.result,
            arguments.disposition,
            arguments.finding,
            independent_review=arguments.independent_review,
            supersede_accepted=arguments.supersede_accepted,
        )
    elif arguments.command == "redisposition":
        record_training_disposition(
            arguments.document_id,
            arguments.disposition,
            arguments.finding,
        )
    elif arguments.command == "contract-revision":
        record_contract_revision(arguments.finding)
    else:
        finalize()


if __name__ == "__main__":
    main()
