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
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from pydantic import TypeAdapter, ValidationError

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

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    ROOT / "artifacts/glm-ocr/pilots/blc150/runs/glm-ocr-blc-pilot150-faa3c717dbc4"
)
SOURCE_RUN_ID = "glm-ocr-blc-pilot150-faa3c717dbc4"
PILOT = ROOT / "data/pilots/blc-pilot-150-faa3c717dbc4"
V1_RUN = ROOT / "artifacts/kie-labels/mpci-bl-pilot-v1-confined-r3"
RUN_ID = "mpci-bl-semantic-v2-pilot150-r1"
DEST = ROOT / "artifacts/kie-labels" / RUN_ID
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
_PAGE_MARKER = re.compile(r"^--- PAGE ([1-9][0-9]*) ---\n", re.MULTILINE)
_CONTAINER_IDENTIFIER_TOKEN = re.compile(
    r"(?<![A-Z0-9])([A-Z]{4})[ -]?(\d{7})(?![A-Z0-9])"
)
_CONTAINER_IDENTIFIER_ADAPTER = TypeAdapter(ContainerIdentifier)
_TARGET_CONTAMINATION = re.compile(
    r"(?ix)\b(?:"
    r"tax(?:\s*(?:id|no|number))?|vat|cif|v\.?d\.?|c\.?n\.?p\.?j|"
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
_POSTAL_VALUE_LABEL = re.compile(
    r"(?i)\b(?:POST(?:AL)?\s+CODE|POSTCODE|ZIP\s+CODE)\b\s*[:#-]?"
)
_DELIVERY_AGENT_ROLE_HEADING = re.compile(
    r"(?im)^\s*(?:\d+\.\s*)?(?:"
    r"(?:DELIVERY|DESTINATION)\s+AGENT\b|"
    r"(?:PORT\s+OF\s+DISCHARGE|DISCHARGE\s+PORT)\s+AGENT\b|"
    r"AGENT(?:'S)?(?:\s+(?:ADDRESS|DETAILS))?\s+AT\s+(?:THE\s+)?"
    r"(?:DESTINATION|PORT\s+OF\s+DISCHARGE)\b|"
    r"AGENT\s+AT\s+PORT\s+OF\s+DISCHARGE\b|"
    r"THE\s+NAME\s+AND\s+ADDRESS\s+OF\s+SHIPPING\s+AGENT\s+AT\s+DESTINATION\b|"
    r"FOR\s+DELIVERY(?:\s+OF\s+GOODS)?(?:\s+PLEASE)?\s+"
    r"(?:APPLY\s+TO|CONTACT)\b|"
    r"TO\s+OBTAIN\s+DELIVERY\s+CONTACT\b|"
    r"\(?AS\s+FRT\s+FWDRS\s+DELIVERY\s+AGENT\s+ONLY\)?\b"
    r")"
)
_CONDITIONAL_NON_NEGOTIABLE = re.compile(
    r"(?i)\bNOT\s+NEGOTIABLE\s+UNLESS\s+CONSIGNED\s+[\"\u201c\u201d']*"
    r"TO\s+ORDER\b"
)
_MARK_VALUE_PREFIX = re.compile(
    r"(?i)^\s*(?:O|C|LOT|BATCH|DRUM)\s*/?\s*NO\.?\s*[:#-]?\s*"
)
_GENERIC_PAYMENT_PLACES = frozenset({"DESTINATION", "ORIGIN"})
_CONTACT_KIND = re.compile(
    r"(?i)\b(?P<kind>tel(?:ephone)?|phone|ph|mob(?:ile)?|fax|e-?mail)\s*[:.]?"
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
    "BRASIL": "BR",
    "ENGLAND": "GB",
    "HOLLAND": "NL",
    "IRAN": "IR",
    "IVORY COAST": "CI",
    "KOREA": "KR",
    "LAOS": "LA",
    "MOLDOVA": "MD",
    "REPUBLIC OF KOREA": "KR",
    "RUSSIA": "RU",
    "SOUTH KOREA": "KR",
    "SYRIA": "SY",
    "TAIWAN": "TW",
    "TANZANIA": "TZ",
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


def _omit_explicit_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _omit_explicit_nulls(child)
            for key, child in value.items()
            if child is not None
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
        identities[("joined_raw_text", source["joinedRawTextSha256"])].append(
            source["documentId"]
        )
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
    for index, document_ids in enumerate(
        (ids for ids in groups.values() if len(ids) > 1), start=1
    ):
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
    documents, grouped = _ledger()
    _verify_pilot_inventory(documents)
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
            eligibility.append(
                {"documentId": document_id, "eligible": False, "reason": reason}
            )
            continue
        item = _work_item(document, pages)
        if _non_latin_letters(item["joinedRawText"]):
            reason = "non_latin_letter_in_raw_ocr"
            prelabel_exclusions.append({"documentId": document_id, "reason": reason})
            eligibility.append(
                {"documentId": document_id, "eligible": False, "reason": reason}
            )
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
            "validation/decision-history",
            "validation/decisions",
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
                "runKind": "semantic_v2_full_pilot150",
                "status": "prepared",
                "preparedAt": _now(),
                "schemaVersion": "2.0.0",
                "annotationSchemaVersion": "2.0.0",
                "sourceRun": {
                    "runId": SOURCE_RUN_ID,
                    "path": str(SOURCE_RUN.relative_to(ROOT)),
                    "stateSha256": _file_digest(SOURCE_RUN / "state.sqlite3"),
                    "inventorySha256": _file_digest(SOURCE_RUN / "inventory.jsonl"),
                    "pilotManifestSha256": _file_digest(PILOT / "manifest.jsonl"),
                },
                "eligibility": observed,
                "eligibleDocumentIds": sorted(
                    item["source"]["documentId"] for item in work_items
                ),
                "duplicateCandidateGroupCount": len(duplicate_groups),
                "overseer": {
                    "role": "session overseer following LABELING_SESSION_PROMPT_V2.md",
                    "requestedModel": "gpt-5.6-terra",
                    "runtimeModelIdentity": "not_exposed_to_session",
                },
                "workerContract": {
                    "model": "gpt-5.6-luna",
                    "reasoningEffort": "high",
                    "oneFreshWorkerPerDocumentAttempt": True,
                    "platformTotalSlots": 4,
                    "maxConcurrentWorkersAchieved": 3,
                    "usageAvailability": "unavailable_from_collaboration_runtime",
                },
                "frozenContract": {
                    path: _file_digest(ROOT / path)
                    for path in (
                        "src/document_ocr/label_schemas/common.py",
                        "src/document_ocr/label_schemas/bill_of_lading.py",
                        "src/document_ocr/label_schemas/mpci_bill_of_lading.py",
                        "src/document_ocr/label_schemas/mpci_projection.py",
                        "BILL_OF_LADING_LABELING_REFERENCE_V2.md",
                        "docs/mpci-kie-semantic-schema-v2-design.md",
                        "LABELING_SESSION_PROMPT_V2.md",
                    )
                },
                "gates": {
                    "targetedTests": "23 passed",
                    "ruff": "passed",
                    "mypy": "passed in 28 source files",
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
        previous_page = 0
        previous_offset = -1
        for evidence in evidence_group:
            source_page = pages.get(evidence.pageNumber)
            if source_page is None:
                raise RuntimeError(
                    f"{document_id}: evidence cites absent page {evidence.pageNumber}"
                )
            start = previous_offset if evidence.pageNumber == previous_page else 0
            offset = source_page.find(evidence.ocrExcerpt, start)
            if offset < 0 or evidence.rawValue not in evidence.ocrExcerpt:
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
    "%d-%b-%y",
    "%d/%b/%y",
    "%d %b %y",
    "%b %d %y",
    "%b/%d/%y",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%m.%d.%Y",
)
_DATE_CANDIDATE = re.compile(
    r"\b(?:"
    r"[0-9]{1,4}(?:ST|ND|RD|TH)?[-/. ]+[A-Z]{3,9}[-/. ]+[0-9]{1,4}|"
    r"[A-Z]{3,9}[-/. ]+[0-9]{1,2}(?:ST|ND|RD|TH)?[-/. ]+[0-9]{2,4}|"
    r"[0-9]{1,4}[-/.]+[0-9]{1,2}[-/.]+[0-9]{1,4}"
    r")\b"
)
_NUMBER_CANDIDATE = re.compile(r"(?<![0-9])[+-]?[0-9][0-9.,]*(?![0-9])")
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
    return (
        unicodedata.normalize("NFKD", translated)
        .encode("ascii", "ignore")
        .decode()
        .upper()
    )


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
        for spelling in spellings:
            try:
                candidates.add(float(spelling))
            except ValueError:
                continue
    tokens = _evidence_tokens(value)
    candidates.update(
        float(_NUMBER_WORD_VALUES[token])
        for token in tokens
        if token in _NUMBER_WORD_VALUES
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
            tokens[index + 1] in _NUMBER_WORD_VALUES
            or tokens[index + 1] in _NUMBER_WORD_SCALES
        )
        if token == "AND" and active and next_is_number_word:
            continue
        flush_words()
    flush_words()
    return tuple(sorted(candidates))


def _date_evidence_supports(target: str, raw_values: str) -> bool:
    expected = date.fromisoformat(target)
    normalized = _ascii_evidence_text(raw_values)
    normalized = re.sub(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)\.",
        r"\1 ",
        normalized,
    ).replace("SEPT", "SEP")
    normalized = re.sub(r"(?<=[0-9])(?:ST|ND|RD|TH)\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized.replace(",", " "))
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
        return any(
            abs(candidate - float(target)) <= tolerance
            for candidate in _numeric_evidence_candidates(raw_values)
        )
    if not isinstance(target, str):
        return True
    if target_path.endswith("Date"):
        return _date_evidence_supports(target, raw_values)

    raw_tokens = frozenset(_evidence_tokens(raw_values))
    raw_compact = "".join(_evidence_tokens(raw_values))
    if target_path.endswith(".unit"):
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
                    "NEGOTIABLE ONLY IF CONSIGNED TO ORDER",
                )
            )
        return any(
            phrase in normalized_raw
            for phrase in ("NEGOTIABLE", "TO THE ORDER OF", "TO ORDER")
        )
    if target_path.endswith("paymentArrangement"):
        phrases = {
            "prepaid": ("PREPAID",),
            "collect": ("COLLECT",),
            "third_party": ("THIRD PARTY",),
            "payable_elsewhere": ("PAYABLE ELSEWHERE",),
        }
        return any(phrase in normalized_raw for phrase in phrases[target])
    if target_path.endswith(".sameAs"):
        return f"SAME AS {' '.join(_evidence_tokens(target))}" in normalized_raw

    target_tokens = _evidence_tokens(target)
    target_compact = "".join(target_tokens)
    return target_compact in raw_compact or all(token in raw_tokens for token in target_tokens)


def _verify_target_evidence_support(
    target: dict[str, Any], evidence_items: Iterable[Any], document_id: str
) -> None:
    target_leaves = _flatten(target)
    unsupported: list[str] = []
    for evidence in evidence_items:
        value = target_leaves[evidence.targetPath]
        raw_values = " ".join(item.rawValue for item in evidence.rawOcrEvidence)
        if not _target_leaf_is_supported(evidence.targetPath, value, raw_values):
            unsupported.append(evidence.targetPath)
    if unsupported:
        raise RuntimeError(
            f"{document_id}: target leaves are not derivable from cited raw OCR values: "
            f"{unsupported!r}"
        )


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
    for segment in re.split(r"[,;|/]|\s+-\s+|(?<=[A-Z])-(?=[A-Z])", address.upper()):
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
        if ".forwardingAndExportReferences[" in path and _REFERENCE_VALUE_PREFIX.search(
            value
        ):
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
    payment_place = (
        target.get("documentPatch", {}).get("freight", {}).get("paymentPlace")
    )
    if isinstance(payment_place, dict):
        name = payment_place.get("name")
        if isinstance(name, str) and name.strip().upper() in _GENERIC_PAYMENT_PLACES:
            findings.append(f"freight payment place is a generic direction {name!r}")
    return findings


def _source_role_findings(
    target: dict[str, Any], joined_raw_text: str
) -> list[str]:
    delivery_agent = (
        target.get("documentPatch", {}).get("parties", {}).get("deliveryAgent")
    )
    if delivery_agent is not None and not _DELIVERY_AGENT_ROLE_HEADING.search(
        joined_raw_text
    ):
        return [
            "deliveryAgent requires an explicit destination/discharge/delivery-agent "
            "source heading"
        ]
    return []


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
        return [
            "conditional non-negotiable target requires an explicitly emitted consignee name"
        ]

    normalized_consignee = _normalize_country(consignee_name)
    if re.match(r"^TO ORDER(?: OF)?\b", normalized_consignee):
        return [
            "conditional non-negotiable target contradicts the emitted TO ORDER consignee"
        ]

    negotiability_evidence = next(
        (
            item
            for item in annotation.evidence
            if item.targetPath == "documentPatch.negotiability"
        ),
        None,
    )
    if negotiability_evidence is None or not any(
        _normalize_country(item.rawValue) == normalized_consignee
        for item in negotiability_evidence.rawOcrEvidence
    ):
        return [
            "conditional non-negotiable target must cite the emitted named consignee"
        ]
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
                "invalid_identifier_omitted warning names valid ISO 6346 identifier "
                f"{identifier!r}"
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
        _verify_evidence(
            ([item] for item in exclusion.rawOcrEvidence), pages, document_id
        )
        return "excluded", _omit_explicit_nulls(raw), None, {}

    annotation = BillOfLadingAnnotation.model_validate_json(raw_bytes, strict=True)
    if annotation.reviewStatus != "candidate":
        raise RuntimeError(f"{document_id}: worker output is not a candidate")
    _verify_evidence(
        (item.rawOcrEvidence for item in annotation.evidence), pages, document_id
    )
    mislabeled_fax_paths = [
        item.targetPath
        for item in annotation.evidence
        if any(
            _phone_evidence_is_fax(
                item.targetPath, evidence.rawValue, evidence.ocrExcerpt
            )
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
    findings.extend(
        _negotiability_evidence_findings(annotation, target, work["joinedRawText"])
    )
    findings.extend(_warning_findings(annotation))
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
    check_path = (
        DEST / "validation/attempt-checks" / f"{document_id}.attempt-{attempt}.json"
    )
    if check_path.exists():
        raise RuntimeError(
            "attempt check is immutable and already exists; publish a new attempt"
        )
    attempt_sha256 = _file_digest(attempt_path)
    try:
        worker = _read_json(
            DEST / "worker-logs" / f"{document_id}.attempt-{attempt}.json"
        )
        if worker.get("status") != "completed":
            raise RuntimeError("only a completed worker attempt can be checked")
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
            "reasoningEffort": "high",
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
        candidate_relative = (
            Path("candidate-attempts") / f"{document_id}.attempt-{attempt}.json"
        )
        candidate_path = DEST / candidate_relative
        if not candidate_path.is_file():
            raise RuntimeError(
                f"completed worker candidate is absent: {candidate_relative}"
            )
        value["candidatePath"] = candidate_relative.as_posix()
        value["candidateSha256"] = _file_digest(candidate_path)
    value["status"] = status
    value["completedAt"] = _now()
    _atomic_json(path, value)


def record_review(
    document_id: str,
    attempt: int,
    result: str,
    disposition: str | None,
    findings: list[str],
    *,
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
            raise RuntimeError(
                "a superseding accepted decision must use a later worker attempt"
            )
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
        "reviewedAt": _now(),
    }
    if review_path.exists():
        existing_review = _read_json(review_path)
        if {
            key: value for key, value in existing_review.items() if key != "reviewedAt"
        } != {key: value for key, value in review.items() if key != "reviewedAt"}:
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
        "decidedAt": _now(),
    }
    if previous_decision is not None:
        previous_attempt = previous_decision["attempt"]
        history_path = (
            DEST
            / "validation/decision-history"
            / f"{document_id}.attempt-{previous_attempt}.json"
        )
        if history_path.exists():
            if _read_json(history_path) != previous_decision:
                raise RuntimeError("decision history conflicts with the current decision")
        else:
            _atomic_json(history_path, previous_decision)
        decision["supersedesDecisionPath"] = history_path.relative_to(DEST).as_posix()
    _atomic_json(decision_path, decision)


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

    for document_id in expected_ids:
        decision = _read_json(DEST / "validation/decisions" / f"{document_id}.json")
        attempt_path = DEST / decision["attemptPath"]
        check = _read_json(
            DEST
            / "validation/attempt-checks"
            / f"{document_id}.attempt-{decision['attempt']}.json"
        )
        if check.get("status") != "passed" or check.get("attemptSha256") != _file_digest(
            attempt_path
        ):
            raise RuntimeError(
                f"{document_id}: accepted candidate is not bound to its passing check"
            )
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
        if disposition == "include":
            work = _read_json(DEST / "work-items" / f"{document_id}.json")
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
    duplicate_comparisons = _duplicate_comparisons(duplicate_payload["groups"], targets)
    contradictions = [
        contradiction
        for group in duplicate_comparisons
        for pair in group["pairwise"]
        for contradiction in pair["contradictions"]
    ]
    if contradictions:
        raise RuntimeError("duplicate semantic targets contradict on shared paths")
    for group in duplicate_comparisons:
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
        {"groups": duplicate_comparisons, "contradictionCount": 0},
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
    _atomic_json(
        DEST / "validation/mpci-projections.json", dict(sorted(projections.items()))
    )
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
    comparison_summary = _size_comparison_summary(v1_comparison)
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
        "acceptedAttempts": sum(value["result"] == "accepted" for value in review_values),
        "rejectedAttempts": sum(value["result"] == "rejected" for value in review_values),
        "evidenceAndContaminationChecks": "passed",
        "projection": {"status": "passed", "benchmark": benchmark},
        "duplicateComparison": {"status": "passed", "contradictions": 0},
        "v1V2Comparison": v1_comparison,
        "v1V2ComparisonSummary": comparison_summary,
        "targetTokenizer": tokenizer_metadata,
        "eda": eda,
        "qualityClaimBoundary": (
            "Serialization and EDA statistics do not prove downstream model accuracy."
        ),
    }
    _atomic_json(DEST / "validation/report.json", report)
    _atomic_bytes(
        DEST / "validation/report.md",
        (
            "# Semantic-v2 full pilot validation\n\n"
            "Status: **passed**.\n\n"
            f"- Selected source PDFs: {_EXPECTED['selectedDocuments']}\n"
            f"- Pre-label exclusions: {_EXPECTED['preLabelExcludedDocuments']}\n"
            f"- Eligible work items: {len(expected_ids)}\n"
            f"- Validated annotations: {len(annotations)}\n"
            f"- Worker exclusions: {len(exclusions)}\n"
            f"- Duplicate-suppressed annotations: {dispositions['duplicate_suppressed']}\n"
            f"- Training records: {len(training_records)}\n"
            f"- Worker attempts: {len(worker_logs)}\n"
            f"- Rejected attempts: {report['rejectedAttempts']}\n"
            f"- Duplicate contradictions: 0\n"
            f"- Projection throughput: {benchmark['projectionsPerSecond']} projections/s\n\n"
            f"- V1/V2 overlapping documents: {comparison_summary['documents']}\n"
            f"- V2 token delta vs v1: {comparison_summary['tokens']['percentDelta']}%\n\n"
            "See `report.json`, `v1-v2-size-comparison.json`, "
            "`duplicate-comparison.json`, and `eda/` for details.\n"
        ).encode(),
    )
    metadata["status"] = "published"
    metadata["publishedAt"] = _now()
    metadata["workerAttempts"] = len(worker_logs)
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
    review = subparsers.add_parser("review")
    review.add_argument("--document-id", required=True)
    review.add_argument("--attempt", required=True, type=int)
    review.add_argument("--result", choices=("accepted", "rejected"), required=True)
    review.add_argument(
        "--disposition", choices=("include", "duplicate_suppressed", "excluded")
    )
    review.add_argument("--supersede-accepted", action="store_true")
    review.add_argument("--finding", action="append", default=[])
    subparsers.add_parser("finalize")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare()
    elif arguments.command == "check-attempt":
        check_attempt(arguments.document_id, arguments.attempt)
    elif arguments.command == "worker-start":
        record_worker_start(arguments.document_id, arguments.attempt, arguments.task_name)
    elif arguments.command == "worker-finish":
        record_worker_finish(arguments.document_id, arguments.attempt, arguments.status)
    elif arguments.command == "review":
        record_review(
            arguments.document_id,
            arguments.attempt,
            arguments.result,
            arguments.disposition,
            arguments.finding,
            supersede_accepted=arguments.supersede_accepted,
        )
    else:
        finalize()


if __name__ == "__main__":
    main()
