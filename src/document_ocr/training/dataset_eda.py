"""Deep, provenance-aware EDA for the combined MPCI Bill-of-Lading corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from document_ocr.atomic import atomic_publish_bytes, read_regular_file_bytes
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingAnnotation, BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingDualCargoAnnotation,
    BillOfLadingRelationExplicitLabel,
    validate_dual_cargo_consistency,
)
from document_ocr.label_schemas.common import ExtractionSourceReference, LabelSchemaModel
from document_ocr.labeling_agents.work_items import (
    AgentWorkItem,
    WorkItemError,
    validate_annotation_evidence,
)
from document_ocr.training.eda_plots import (
    ScatterPoint,
    grouped_bar,
    heatmap,
    histogram,
    horizontal_bar,
    line_chart,
    quantile_ranges,
    scatter,
    vertical_bar,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PAGE_HEADER = re.compile(r"(?m)^--- PAGE ([1-9][0-9]*) ---$")


class DatasetEdaError(RuntimeError):
    """A source, feature, or publication violated the EDA contract."""


class _ConfigModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _absolute_path(value: str, field_name: str, *, directory: bool) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    normalized = Path(os.path.abspath(value))
    if directory and normalized == Path(normalized.anchor):
        raise ValueError(f"{field_name} must not be the filesystem root")
    return value


class PinnedRoot(_ConfigModel):
    root: NonEmptyString
    manifest_sha256: Sha256

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value, "pinned root", directory=True)


class DatasetSource(PinnedRoot):
    expected_records: PositiveInteger


class TemplateSensitivity(_ConfigModel):
    label: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
    visual_minimum: Probability
    ocr_minimum: Probability
    combined_minimum: Probability


class EdaAnalysisSettings(_ConfigModel):
    top_n: Annotated[int, Field(ge=10, le=50)]
    histogram_bins: Annotated[int, Field(ge=8, le=50)]
    raster_workers: Annotated[int, Field(ge=1, le=16)]
    raster_grid_size: Annotated[int, Field(ge=8, le=64)]
    template_visual_minimum: Probability
    template_ocr_minimum: Probability
    template_combined_minimum: Probability
    template_visual_weight: Probability
    template_sensitivity: tuple[TemplateSensitivity, ...] = Field(min_length=2)
    augmentation_minimum_documents: PositiveInteger
    augmentation_minimum_joint_documents: PositiveInteger

    @field_validator("template_sensitivity", mode="before")
    @classmethod
    def sensitivity_is_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("template_sensitivity must be a YAML sequence")
        return tuple(value)

    @model_validator(mode="after")
    def template_settings_are_valid(self) -> EdaAnalysisSettings:
        if len({row.label for row in self.template_sensitivity}) != len(
            self.template_sensitivity
        ):
            raise ValueError("template sensitivity labels must be unique")
        if not any(row.label == "conservative" for row in self.template_sensitivity):
            raise ValueError("template sensitivity must include conservative")
        return self


class EdaOutput(_ConfigModel):
    root: NonEmptyString

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value, "EDA output root", directory=True)


class DatasetEdaConfig(_ConfigModel):
    schema_version: Literal[1]
    analysis_id: NonEmptyString
    dataset: DatasetSource
    current_annotation_dataset: PinnedRoot
    raster_run_roots: dict[NonEmptyString, NonEmptyString]
    analysis: EdaAnalysisSettings
    output: EdaOutput

    @field_validator("analysis_id")
    @classmethod
    def analysis_id_is_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("analysis_id contains unsupported characters")
        return value

    @field_validator("raster_run_roots")
    @classmethod
    def raster_roots_are_absolute(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("raster_run_roots must not be empty")
        for root in value.values():
            _absolute_path(root, "raster run root", directory=True)
        return value

    @model_validator(mode="after")
    def output_does_not_alias_source(self) -> DatasetEdaConfig:
        sources = {
            self.dataset.root,
            self.current_annotation_dataset.root,
            *self.raster_run_roots.values(),
        }
        if self.output.root in sources:
            raise ValueError("EDA output root aliases a source root")
        return self


def load_dataset_eda_config(path: Path) -> DatasetEdaConfig:
    try:
        return DatasetEdaConfig.model_validate(load_strict_yaml_mapping(path), strict=True)
    except ValueError as error:
        raise DatasetEdaError(f"invalid dataset EDA config: {path}: {error}") from error


def _json(payload: bytes, *, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DatasetEdaError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise DatasetEdaError(f"{context} must be a JSON object")
    return value


def _jsonl(payload: bytes, *, context: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise DatasetEdaError(f"{context} contains blank row {number}")
        rows.append(_json(line, context=f"{context} row {number}"))
    return tuple(rows)


def _root(path: str, *, context: str) -> Path:
    try:
        root = Path(path).resolve(strict=True)
    except OSError as error:
        raise DatasetEdaError(f"{context} is absent: {path}") from error
    if not root.is_dir():
        raise DatasetEdaError(f"{context} is not a directory: {root}")
    return root


def _regular_file(path: Path, *, context: str) -> Path:
    if path.is_symlink():
        raise DatasetEdaError(f"{context} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DatasetEdaError(f"{context} is absent: {path}") from error
    if not resolved.is_file():
        raise DatasetEdaError(f"{context} is not a regular file: {resolved}")
    return resolved


def _contained(root: Path, relative: str, *, context: str) -> Path:
    path = _regular_file(root / relative, context=context)
    if root not in path.parents:
        raise DatasetEdaError(f"{context} escapes its root: {relative}")
    return path


def _load_manifest(root: Path, expected_sha256: str, *, context: str) -> dict[str, Any]:
    path = _contained(root, "manifest.json", context=f"{context} manifest")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != expected_sha256:
        raise DatasetEdaError(f"{context} manifest SHA-256 differs: {path}")
    return _json(payload, context=f"{context} manifest")


def _manifest_file(
    root: Path, manifest: dict[str, Any], *, kind: str
) -> tuple[Path, bytes, tuple[dict[str, Any], ...]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise DatasetEdaError("dataset manifest files are absent")
    matches = [row for row in files if isinstance(row, dict) and row.get("kind") == kind]
    if len(matches) != 1:
        raise DatasetEdaError(f"dataset manifest must contain exactly one {kind!r} artifact")
    entry = matches[0]
    relative, digest, declared_rows = entry.get("path"), entry.get("sha256"), entry.get("rows")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise DatasetEdaError(f"dataset {kind} manifest entry is malformed")
    path = _contained(root, relative, context=f"dataset {kind}")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != digest:
        raise DatasetEdaError(f"dataset {kind} SHA-256 differs: {path}")
    rows = _jsonl(payload, context=f"dataset {kind}")
    if declared_rows != len(rows):
        raise DatasetEdaError(f"dataset {kind} row count differs from manifest")
    return path, payload, rows


def _document_id(row: dict[str, Any], *, context: str) -> str:
    value = row.get("documentId")
    if not isinstance(value, str) or re.fullmatch(r"doc_[0-9a-f]{64}", value) is None:
        raise DatasetEdaError(f"{context} has invalid documentId")
    return value


def _load_inputs(
    config: DatasetEdaConfig,
) -> tuple[
    Path,
    dict[str, Any],
    tuple[dict[str, Any], ...],
    dict[str, dict[str, Any]],
    Path,
]:
    dataset_root = _root(config.dataset.root, context="EDA dataset")
    manifest = _load_manifest(
        dataset_root, config.dataset.manifest_sha256, context="EDA dataset"
    )
    _, _, records = _manifest_file(dataset_root, manifest, kind="training_records")
    _, _, lineage_rows = _manifest_file(dataset_root, manifest, kind="lineage")
    if len(records) != config.dataset.expected_records:
        raise DatasetEdaError("dataset record count differs from EDA config")
    if len(lineage_rows) != len(records):
        raise DatasetEdaError("dataset lineage count differs from training records")
    record_ids = [_document_id(row, context="training record") for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise DatasetEdaError("training records contain duplicate document IDs")
    raw_hashes = [row.get("joinedRawTextSha256") for row in records]
    if len(raw_hashes) != len(set(raw_hashes)):
        raise DatasetEdaError("training records contain duplicate raw-OCR hashes")
    lineage: dict[str, dict[str, Any]] = {}
    for row in lineage_rows:
        document_id = _document_id(row, context="lineage row")
        if document_id in lineage:
            raise DatasetEdaError(f"duplicate lineage document: {document_id}")
        lineage[document_id] = row
    if set(record_ids) != set(lineage):
        raise DatasetEdaError("record and lineage document sets differ")
    current_root = _root(
        config.current_annotation_dataset.root, context="current annotation dataset"
    )
    _load_manifest(
        current_root,
        config.current_annotation_dataset.manifest_sha256,
        context="current annotation dataset",
    )
    return dataset_root, manifest, records, lineage, current_root


_CARRIER_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("MAERSK", r"\bMAERSK\b|\bMAERK\b"),
        ("CMA CGM", r"\bCMA\s+CGM\b"),
        (
            "MSC",
            r"\bMSC\b.*\bMEDITERRANEAN\b|\bMEDITERRANEAN\s+SHIPPING\s+(?:COMPANY|CO)\b",
        ),
        ("VASCO MARITIME", r"\bVASCO\s+MARITIME\b"),
        ("HAPAG-LLOYD", r"\bHAPAG\s+LLOYD\b"),
        ("HMM", r"^HMM\b|\bHYUNDAI\s+MERCHANT\s+MARINE\b"),
        ("ONE", r"\bOCEAN\s+NETWORK\s+EXPRESS\b|^ONE\s+LINE$"),
        ("YANG MING", r"\bYANG\s+MING\b"),
        ("TURKON", r"\bTURKON\b"),
        ("EMIRATES SHIPPING LINE", r"\bEMIRATES\s+SHIPPING\s+LINE\b"),
        ("EVERGREEN", r"\bEVERGREEN\b"),
        ("COSCO", r"\bCOSCO\b"),
        ("ARKAS", r"\bARKAS\b"),
        ("PIL", r"\bPACIFIC\s+INTERNATIONAL\s+LINES\b"),
        ("WAN HAI", r"\bWAN\s+HAI\b"),
        ("ZIM", r"^ZIM\b"),
        ("OOCL", r"\bOOCL\b|\bORIENT\s+OVERSEAS\s+CONTAINER\s+LINE\b"),
        ("TARROS", r"\bTARROS\b"),
    )
)


def _normalized_words(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_like = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(re.findall(r"[A-Z0-9]+", ascii_like.upper()))


def carrier_family(value: str | None) -> tuple[str, bool]:
    """Return a conservative observed-name family and whether a curated rule matched."""

    if value is None:
        return "<MISSING>", False
    normalized = _normalized_words(value)
    for name, pattern in _CARRIER_RULES:
        if pattern.search(normalized):
            return name, True
    return f"OTHER::{normalized}", False


_COUNTRY_ALIASES: dict[str, str] = {
    "U S A": "UNITED STATES",
    "USA": "UNITED STATES",
    "UNITED STATES OF AMERICA": "UNITED STATES",
    "UAE": "UNITED ARAB EMIRATES",
    "U A E": "UNITED ARAB EMIRATES",
    "TURKEY": "TURKIYE/TURKEY",
    "TURKIYE": "TURKIYE/TURKEY",
    "BRASIL": "BRAZIL",
    "BRAZILIAN": "BRAZIL",
    "REPUBLIC OF KOREA": "SOUTH KOREA",
    "TAIWAN R O C": "TAIWAN",
    "TAIWAN ROC": "TAIWAN",
    "TAIWAN PROVINCE OF CHINA": "TAIWAN",
    "PEOPLE S REPUBLIC OF CHINA": "CHINA",
    "P R CHINA": "CHINA",
    "PRC": "CHINA",
    "ARAB REPUBLIC OF EGYPT": "EGYPT",
    "REPUBLIC OF EGYPT": "EGYPT",
    "EG": "EGYPT",
    "UK": "UNITED KINGDOM",
    "U K": "UNITED KINGDOM",
    "GREAT BRITAIN": "UNITED KINGDOM",
    "RUSSIAN FEDERATION": "RUSSIA",
    "KSA": "SAUDI ARABIA",
    "VIET NAM": "VIETNAM",
    "NETHERLAND": "NETHERLANDS",
    "CZECH REPUBLIC": "CZECHIA",
    "REPUBLIC OF SOUTH AFRICA": "SOUTH AFRICA",
    "AUSTRALIAN": "AUSTRALIA",
    "SPANISH ORIGIN": "SPAIN",
    "ITALIAN": "ITALY",
    "U S": "UNITED STATES",
    "CN": "CHINA",
    "IN": "INDIA",
}


def country_group(value: str | None) -> str:
    """Conservatively group printed variants for EDA without changing labels."""

    if value is None:
        return "<MISSING>"
    normalized = _normalized_words(value)
    return _COUNTRY_ALIASES.get(normalized, normalized)


_ANCHORS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("bill_of_lading", r"\bBILL\s+OF\s+LADING\b"),
        ("sea_waybill", r"\bSEA\s+WAYBILL\b|\bWAYBILL\b"),
        ("shipper", r"\bSHIPPER\b"),
        ("consignee", r"\bCONSIGNEE\b"),
        ("notify_party", r"\bNOTIFY\s+PARTY\b"),
        ("vessel", r"\bVESSEL\b"),
        ("voyage", r"\bVOYAGE\b"),
        ("place_of_receipt", r"\bPLACE\s+OF\s+RECEIPT\b"),
        ("port_of_loading", r"\bPORT\s+OF\s+LOADING\b"),
        ("port_of_discharge", r"\bPORT\s+OF\s+DISCHARGE\b"),
        ("place_of_delivery", r"\bPLACE\s+OF\s+DELIVERY\b"),
        ("marks", r"\bMARKS\s+(?:AND|&)\s+(?:NOS|NUMBERS)\b"),
        (
            "packages",
            r"\b(?:NO\.?|NUMBER)\s+(?:AND|&)\s+(?:KIND\s+OF\s+)?PACKAGES\b",
        ),
        ("description_of_goods", r"\bDESCRIPTION\s+OF\s+(?:PACKAGES\s+AND\s+)?GOODS\b"),
        ("gross_weight", r"\bGROSS\s+WEIGHT\b"),
        ("measurement", r"\bMEASUREMENT\b"),
        ("freight", r"\bFREIGHT\b"),
        ("place_of_issue", r"\bPLACE\s+(?:AND\s+DATE\s+)?OF\s+ISSUE\b"),
        ("shipped_on_board", r"\bSHIPPED\s+ON\s+BOARD\b"),
        ("agent_at_destination", r"\bAGENT\s+AT\s+DESTINATION\b"),
        ("number_of_originals", r"\bNUMBER\s+OF\s+ORIGINAL\b"),
    )
)


def _page_texts(joined_raw_text: str) -> tuple[str, ...]:
    matches = list(_PAGE_HEADER.finditer(joined_raw_text))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        raise DatasetEdaError("joined raw OCR page headers are absent or non-contiguous")
    pages: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined_raw_text)
        pages.append(joined_raw_text[start:end].strip("\n"))
    return tuple(pages)


def _anchor_sequence(first_page: str) -> tuple[str, ...]:
    hits: list[tuple[int, str]] = []
    for name, pattern in _ANCHORS:
        match = pattern.search(first_page)
        if match is not None:
            hits.append((match.start(), name))
    return tuple(name for _, name in sorted(hits))


def _leaf_items(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else key
            yield from _leaf_items(item, next_path)
    elif isinstance(value, list):
        for item in value:
            yield from _leaf_items(item, f"{path}[]")
    else:
        yield path, value


def _temperature_celsius(value: float, unit: str) -> float:
    if unit == "celsius":
        return value
    if unit == "fahrenheit":
        return (value - 32.0) * 5.0 / 9.0
    raise DatasetEdaError(f"unsupported temperature unit: {unit}")


@dataclass(frozen=True, slots=True)
class _RasterTask:
    document_id: str
    path: Path
    sha256: str
    grid_size: int


@dataclass(frozen=True, slots=True)
class _RasterFeature:
    document_id: str
    width: int
    height: int
    aspect_ratio: float
    orientation: str
    ink_density: float
    layout_hash: str
    vector: tuple[float, ...]


def _raster_feature(task: _RasterTask) -> _RasterFeature:
    payload = read_regular_file_bytes(task.path)
    if sha256_bytes(payload) != task.sha256:
        raise DatasetEdaError(f"first-page raster SHA-256 differs: {task.document_id}")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            width, height = source.size
            if width <= 0 or height <= 0:
                raise DatasetEdaError(f"first-page raster has invalid dimensions: {task.document_id}")
            grayscale = ImageOps.grayscale(source).resize(
                (task.grid_size, task.grid_size), Image.Resampling.BOX
            )
            pixels = tuple(grayscale.tobytes())
    except (OSError, ValueError) as error:
        raise DatasetEdaError(f"cannot decode first-page raster: {task.document_id}") from error
    ink = tuple(max(0.0, (255 - value) / 255.0 - 0.01) for value in pixels)
    norm = math.sqrt(sum(value * value for value in ink)) or 1.0
    vector = tuple(value / norm for value in ink)
    quantized = bytes(min(255, round(value * 255)) for value in ink)
    aspect = width / height
    orientation = "landscape" if aspect > 1.05 else "portrait" if aspect < 0.95 else "square"
    return _RasterFeature(
        document_id=task.document_id,
        width=width,
        height=height,
        aspect_ratio=round(aspect, 6),
        orientation=orientation,
        ink_density=round(sum(ink) / len(ink), 6),
        layout_hash=sha256_bytes(quantized),
        vector=vector,
    )


def _party_contact_count(parties: dict[str, Any]) -> int:
    count = 0
    for name in ("shipper", "consignee", "carrier", "deliveryAgent", "forwardingAgent"):
        party = parties.get(name)
        if isinstance(party, dict) and party.get("contactDetails"):
            count += 1
    for party in parties.get("notifyParties", []):
        if isinstance(party, dict) and party.get("contactDetails"):
            count += 1
    return count


def _first_nonempty(values: Iterable[str | None]) -> str | None:
    return next((value for value in values if value), None)


def _source_class(path: Path) -> str:
    for part in reversed(path.parts):
        lowered = part.casefold()
        if lowered in {"blc", "swb"}:
            return lowered
    return "unclassified"


def _record_feature(
    *,
    record: dict[str, Any],
    lineage: dict[str, Any],
    source: ExtractionSourceReference,
    document_type: str,
    annotation_path: Path,
) -> dict[str, Any]:
    document_id = cast(str, record["documentId"])
    raw_text = cast(str, record["joinedRawText"])
    pages = _page_texts(raw_text)
    patch = cast(dict[str, Any], record["target"]["documentPatch"])
    normal_patch = cast(dict[str, Any], record["normalTarget"]["documentPatch"])
    parties = cast(dict[str, Any], patch.get("parties", {}))
    route = cast(dict[str, Any], patch.get("route", {}))
    transport = cast(dict[str, Any], patch.get("transport", {}))
    containers = cast(list[dict[str, Any]], patch.get("containers", []))
    goods = cast(list[dict[str, Any]], patch.get("cargoGroups", []))
    packages = cast(list[dict[str, Any]], patch.get("cargoPackages", []))
    allocation_groups = cast(
        list[dict[str, Any]], patch.get("cargoAllocationGroups", [])
    )

    carrier_name = cast(dict[str, Any], parties.get("carrier", {})).get("name")
    carrier_name = carrier_name if isinstance(carrier_name, str) else None
    carrier_key, carrier_rule_matched = carrier_family(carrier_name)
    shipper_country = cast(dict[str, Any], parties.get("shipper", {})).get("country")
    shipper_country = shipper_country if isinstance(shipper_country, str) else None
    load_country = cast(dict[str, Any], route.get("portOfLoading", {})).get("country")
    load_country = load_country if isinstance(load_country, str) else None
    discharge_country = cast(dict[str, Any], route.get("portOfDischarge", {})).get(
        "country"
    )
    discharge_country = discharge_country if isinstance(discharge_country, str) else None
    delivery_country = cast(dict[str, Any], route.get("placeOfDelivery", {})).get(
        "country"
    )
    delivery_country = delivery_country if isinstance(delivery_country, str) else None
    origins = [
        _first_nonempty(
            (
                cast(dict[str, Any], group.get("origin", {})).get("name"),
                cast(dict[str, Any], group.get("origin", {})).get("identifier"),
            )
        )
        for group in goods
        if group.get("origin")
    ]
    origin_values = [value for value in origins if value is not None]

    container_types = [
        cast(str, row["typeDescription"])
        for row in containers
        if isinstance(row.get("typeDescription"), str)
    ]
    seal_count = sum(len(row.get("sealNumbers", [])) for row in containers)
    temperatures: list[tuple[float, str, float]] = []
    for row in containers:
        setting = row.get("temperatureSetpoint")
        if isinstance(setting, dict):
            value, unit = setting.get("value"), setting.get("unit")
            if not isinstance(value, int | float) or not isinstance(unit, str):
                raise DatasetEdaError(f"malformed temperature setting: {document_id}")
            temperatures.append((float(value), unit, _temperature_celsius(float(value), unit)))

    package_types = [
        cast(str, row["typeDescription"])
        for row in packages
        if isinstance(row.get("typeDescription"), str)
    ]
    package_quantity_values = [
        cast(int, row["quantity"])
        for row in packages
        if isinstance(row.get("quantity"), int)
    ]
    packages_by_group = Counter(cast(str, row["groupId"]) for row in packages)
    allocation_rows = [
        allocation
        for group in allocation_groups
        for allocation in cast(list[dict[str, Any]], group.get("allocations", []))
    ]
    allocation_coverages = [
        cast(str, row["coverage"])
        for row in allocation_groups
        if isinstance(row.get("coverage"), str)
    ]
    dangerous = [
        row
        for group in goods
        for row in cast(list[dict[str, Any]], group.get("dangerousGoods", []))
    ]
    hazard_categories = [
        cast(str, row["hazardCategory"])
        for row in dangerous
        if isinstance(row.get("hazardCategory"), str)
    ]
    subsidiary_hazards = [
        cast(str, row["subsidiaryHazardCategory"])
        for row in dangerous
        if isinstance(row.get("subsidiaryHazardCategory"), str)
    ]
    un_numbers = [
        cast(str, row["unNumber"])
        for row in dangerous
        if isinstance(row.get("unNumber"), str)
    ]
    hs_codes = [
        cast(str, value)
        for group in goods
        for value in cast(list[str], group.get("hsCodes", []))
    ]

    issue_date = patch.get("issueDate")
    board_date = patch.get("shippedOnBoardDate")
    year_value = issue_date or board_date
    label_year = (
        int(cast(str, year_value)[:4])
        if isinstance(year_value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", year_value)
        else None
    )
    source_path = _regular_file(Path(source.localCanonicalPath), context="source PDF")
    filename_year_match = re.match(r"([0-9]{4})-", source_path.name)
    filename_year = int(filename_year_match.group(1)) if filename_year_match else None
    page_characters = [len(page) for page in pages]
    target_leaf_count = sum(1 for _ in _leaf_items(patch))
    normal_leaf_count = sum(1 for _ in _leaf_items(normal_patch))
    target_leaf_paths = sorted({path for path, _ in _leaf_items(patch)})
    freight = cast(dict[str, Any], patch.get("freight", {}))
    forwarding = parties.get("forwardingAgent")
    delivery_agent = parties.get("deliveryAgent")
    notify_parties = cast(list[dict[str, Any]], parties.get("notifyParties", []))
    first_page_anchors = _anchor_sequence(pages[0])
    correction_ids = lineage.get("policyCorrectionIds", [])
    overlap_paths = lineage.get("legacyCurrentStrictEvidenceFailurePaths", [])
    if not isinstance(correction_ids, list) or not isinstance(overlap_paths, list):
        raise DatasetEdaError(f"lineage audit fields are malformed: {document_id}")

    return {
        "document_id": document_id,
        "source_corpus": lineage["sourceCorpus"],
        "document_type": document_type,
        "source_class": _source_class(source_path),
        "extraction_run_id": source.extractionRunId,
        "source_path": str(source_path),
        "source_filename": source_path.name,
        "source_sha256": source.sourceSha256,
        "source_file_bytes": source_path.stat().st_size,
        "source_filename_year": filename_year,
        "annotation_path": str(annotation_path),
        "annotation_style": (
            "semantic_v2_with_transform_lineage"
            if lineage["sourceCorpus"] == "legacy_combined487"
            else "dual_relation_single_source_v4"
        ),
        "policy_correction_count": len(correction_ids),
        "legacy_overlap_evidence_field_count": len(overlap_paths),
        "page_count": len(pages),
        "ocr_characters": len(raw_text),
        "ocr_words": len(re.findall(r"\S+", raw_text)),
        "ocr_lines": len(raw_text.splitlines()),
        "ocr_characters_per_page_mean": round(statistics.fmean(page_characters), 3),
        "ocr_characters_per_page_min": min(page_characters),
        "ocr_characters_per_page_max": max(page_characters),
        "target_canonical_bytes": len(canonical_json_bytes(record["target"])),
        "target_leaf_count": target_leaf_count,
        "target_leaf_paths": target_leaf_paths,
        "normal_target_leaf_count": normal_leaf_count,
        "label_year": label_year,
        "issue_date_present": issue_date is not None,
        "shipped_on_board_date_present": board_date is not None,
        "negotiability": patch.get("negotiability", "<MISSING>"),
        "bill_of_lading_number_present": bool(patch.get("billOfLadingNumber")),
        "place_of_issue_present": bool(patch.get("placeOfIssue")),
        "vessel_present": bool(transport.get("vessel")),
        "voyage_present": bool(transport.get("voyageNumber")),
        "port_of_loading_present": bool(route.get("portOfLoading")),
        "port_of_discharge_present": bool(route.get("portOfDischarge")),
        "place_of_delivery_present": bool(route.get("placeOfDelivery")),
        "carrier_name": carrier_name or "<MISSING>",
        "carrier_family": carrier_key,
        "carrier_family_curated": carrier_rule_matched,
        "shipper_present": bool(parties.get("shipper")),
        "consignee_present": bool(parties.get("consignee")),
        "carrier_present": bool(parties.get("carrier")),
        "delivery_agent_present": bool(delivery_agent),
        "forwarding_agent_present": bool(forwarding),
        "notify_party_count": len(notify_parties),
        "forwarding_export_reference_count": len(
            cast(list[str], patch.get("forwardingAndExportReferences", []))
        ),
        "party_with_contact_details_count": _party_contact_count(parties),
        "shipper_country_raw": shipper_country or "<MISSING>",
        "export_country_proxy": country_group(shipper_country),
        "port_of_loading_country_raw": load_country or "<MISSING>",
        "port_of_loading_country": country_group(load_country),
        "port_of_discharge_country_raw": discharge_country or "<MISSING>",
        "port_of_discharge_country": country_group(discharge_country),
        "place_of_delivery_country": country_group(delivery_country),
        "goods_origins_raw": origin_values,
        "goods_origins": [country_group(value) for value in origin_values],
        "trade_lane_proxy": (
            f"{country_group(load_country)} → {country_group(discharge_country)}"
        ),
        "freight_payment_arrangement": freight.get("paymentArrangement", "<MISSING>"),
        "freight_payment_place_present": bool(freight.get("paymentPlace")),
        "container_count": len(containers),
        "container_types": container_types,
        "container_type_fact_count": len(container_types),
        "seal_count": seal_count,
        "container_vgm_count": sum(bool(row.get("verifiedGrossMass")) for row in containers),
        "temperature_setting_count": len(temperatures),
        "temperature_present": bool(temperatures),
        "temperature_values_printed": [row[0] for row in temperatures],
        "temperature_units": [row[1] for row in temperatures],
        "temperature_values_celsius": [round(row[2], 4) for row in temperatures],
        "goods_group_count": len(goods),
        "goods_description_count": sum(bool(row.get("description")) for row in goods),
        "package_fact_count": len(packages),
        "package_types": package_types,
        "package_type_fact_count": len(package_types),
        "package_quantity_fact_count": len(package_quantity_values),
        "package_quantity_sum": sum(package_quantity_values),
        "max_package_levels_per_goods": max(packages_by_group.values(), default=0),
        "allocation_group_count": len(allocation_groups),
        "allocation_row_count": len(allocation_rows),
        "allocation_coverages": allocation_coverages,
        "hs_codes": hs_codes,
        "hs_code_count": len(hs_codes),
        "dangerous_goods_count": len(dangerous),
        "dangerous_goods_present": bool(dangerous),
        "hazard_categories": hazard_categories,
        "subsidiary_hazard_categories": subsidiary_hazards,
        "un_numbers": un_numbers,
        "gross_weight_group_count": sum(bool(row.get("grossWeight")) for row in goods),
        "net_weight_group_count": sum(bool(row.get("netWeight")) for row in goods),
        "volume_group_count": sum(bool(row.get("volume")) for row in goods),
        "marks_group_count": sum(bool(row.get("marksAndNumbers")) for row in goods),
        "handling_instruction_group_count": sum(
            bool(row.get("handlingInstructions")) for row in goods
        ),
        "additional_information_group_count": sum(
            bool(row.get("additionalInformation")) for row in goods
        ),
        "origin_group_count": len(origin_values),
        "multi_page": len(pages) > 1,
        "multi_container": len(containers) > 1,
        "multi_goods": len(goods) > 1,
        "multi_package_level": max(packages_by_group.values(), default=0) > 1,
        "has_allocations": bool(allocation_groups),
        "high_container_complexity": len(containers) >= 5,
        "high_hs_complexity": len(hs_codes) >= 5,
        "first_page_anchor_sequence": list(first_page_anchors),
        "ocr_structure_hash": sha256_bytes(canonical_json_bytes(first_page_anchors)),
    }


def _annotation_and_source(
    *,
    record: dict[str, Any],
    lineage: dict[str, Any],
    current_root: Path,
) -> tuple[ExtractionSourceReference, str, Path, int]:
    document_id = cast(str, record["documentId"])
    source_corpus = lineage.get("sourceCorpus")
    if source_corpus == "legacy_combined487":
        raw_path = lineage.get("sourceValidatedAnnotationPath")
        digest = lineage.get("sourceValidatedAnnotationSha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise DatasetEdaError(f"legacy annotation lineage is malformed: {document_id}")
        path = _regular_file(Path(raw_path), context="legacy annotation")
    elif source_corpus == "current_main680":
        relative = lineage.get("sourceRecordPath")
        digest = lineage.get("sourceValidatedAnnotationSha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DatasetEdaError(f"current annotation lineage is malformed: {document_id}")
        path = _contained(current_root, relative, context="current annotation")
    else:
        raise DatasetEdaError(f"unsupported source corpus in lineage: {document_id}")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != digest:
        raise DatasetEdaError(f"annotation SHA-256 differs: {document_id}")
    raw_text = cast(str, record["joinedRawText"])
    try:
        if source_corpus == "legacy_combined487":
            annotation = BillOfLadingAnnotation.model_validate_json(payload, strict=True)
            source = annotation.source
            document_type = annotation.documentType
            evidence_fields = len(annotation.evidence)
        else:
            annotation = BillOfLadingDualCargoAnnotation.model_validate_json(payload, strict=True)
            source = annotation.source
            document_type = annotation.documentType
            evidence_fields = len(annotation.evidence) + len(annotation.relationEvidence)
            validate_annotation_evidence(
                AgentWorkItem(source=source, joinedRawText=raw_text), annotation
            )
    except (ValueError, WorkItemError) as error:
        raise DatasetEdaError(f"annotation validation failed: {document_id}: {error}") from error
    if source.documentId != document_id:
        raise DatasetEdaError(f"annotation document ID differs: {document_id}")
    if source.joinedRawTextSha256 != record.get("joinedRawTextSha256"):
        raise DatasetEdaError(f"annotation raw-OCR hash differs: {document_id}")
    return source, document_type, path, evidence_fields


def _build_features(
    config: DatasetEdaConfig,
    records: Sequence[dict[str, Any]],
    lineage_by_id: dict[str, dict[str, Any]],
    current_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_roots = {
        run_id: _root(path, context=f"raster run {run_id}")
        for run_id, path in config.raster_run_roots.items()
    }
    features: list[dict[str, Any]] = []
    raster_tasks: list[_RasterTask] = []
    source_hashes: set[str] = set()
    evidence_field_count = 0
    all_page_rasters = 0
    for record in records:
        document_id = _document_id(record, context="training record")
        raw_text = record.get("joinedRawText")
        raw_hash = record.get("joinedRawTextSha256")
        if not isinstance(raw_text, str) or sha256_bytes(raw_text.encode("utf-8")) != raw_hash:
            raise DatasetEdaError(f"training raw-OCR hash differs: {document_id}")
        try:
            normal = BillOfLadingLabel.model_validate_json(
                canonical_json_bytes(record["normalTarget"]), strict=True
            )
            relation = BillOfLadingRelationExplicitLabel.model_validate_json(
                canonical_json_bytes(record["target"]), strict=True
            )
            validate_dual_cargo_consistency(normal, relation)
            if normal.canonical_target() != record["normalTarget"]:
                raise ValueError("normal target is not canonical")
            if relation.canonical_target() != record["target"]:
                raise ValueError("relation target is not canonical")
        except ValueError as error:
            raise DatasetEdaError(f"training target validation failed: {document_id}") from error
        source, document_type, annotation_path, annotation_evidence_fields = (
            _annotation_and_source(
                record=record,
                lineage=lineage_by_id[document_id],
                current_root=current_root,
            )
        )
        pages = _page_texts(raw_text)
        if len(pages) != source.documentPageCount:
            raise DatasetEdaError(f"raw/source page count differs: {document_id}")
        run_root = run_roots.get(source.extractionRunId)
        if run_root is None:
            raise DatasetEdaError(f"raster run is not configured: {source.extractionRunId}")
        for page in source.pages:
            _contained(run_root, page.rasterPath, context="page raster")
            all_page_rasters += 1
        first_page = source.pages[0]
        raster_tasks.append(
            _RasterTask(
                document_id=document_id,
                path=_contained(run_root, first_page.rasterPath, context="first-page raster"),
                sha256=first_page.rasterSha256,
                grid_size=config.analysis.raster_grid_size,
            )
        )
        source_hashes.add(source.sourceSha256)
        evidence_field_count += annotation_evidence_fields
        feature = _record_feature(
            record=record,
            lineage=lineage_by_id[document_id],
            source=source,
            document_type=document_type,
            annotation_path=annotation_path,
        )
        feature["annotation_evidence_field_count"] = annotation_evidence_fields
        features.append(feature)

    with ThreadPoolExecutor(max_workers=config.analysis.raster_workers) as executor:
        raster_features = tuple(executor.map(_raster_feature, raster_tasks))
    raster_by_id = {row.document_id: row for row in raster_features}
    if len(raster_by_id) != len(features):
        raise DatasetEdaError("raster feature count differs from documents")
    for feature in features:
        raster = raster_by_id[cast(str, feature["document_id"])]
        feature.update(
            {
                "first_page_raster_width": raster.width,
                "first_page_raster_height": raster.height,
                "first_page_aspect_ratio": raster.aspect_ratio,
                "first_page_orientation": raster.orientation,
                "first_page_ink_density": raster.ink_density,
                "visual_layout_hash": raster.layout_hash,
                "_visual_vector": raster.vector,
            }
        )
    validation = {
        "recordsSchemaAndRelationValid": len(features),
        "annotationsHashAndSchemaValid": len(features),
        "currentAnnotationsEvidenceValid": sum(
            row["source_corpus"] == "current_main680" for row in features
        ),
        "pageSequencesValid": len(features),
        "sourcePdfsPresent": len(features),
        "uniqueSourcePdfHashes": len(source_hashes),
        "firstPageRastersHashValid": len(features),
        "allPageRastersPresent": all_page_rasters,
        "annotationEvidenceFields": evidence_field_count,
    }
    return features, validation


def _visual_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise DatasetEdaError("visual template vectors have different dimensions")
    return min(1.0, max(0.0, sum(a * b for a, b in zip(left, right, strict=True))))


def _ocr_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    return SequenceMatcher(a=tuple(left), b=tuple(right), autojunk=False).ratio()


@dataclass(slots=True)
class _TemplateCluster:
    representative: dict[str, Any]
    members: list[tuple[dict[str, Any], float, float, float]]


def _template_clusters(
    features: Sequence[dict[str, Any]],
    *,
    visual_minimum: float,
    ocr_minimum: float,
    combined_minimum: float,
    visual_weight: float,
    annotate: bool,
) -> list[dict[str, Any]]:
    """Greedily cluster within carrier/type using measured layout and heading order.

    This intentionally produces a conservative proxy rather than asserting that two
    documents came from the same source template.  Frequency-first deterministic
    ordering keeps common forms from being fragmented by rare representatives.
    """

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for feature in features:
        groups[
            (cast(str, feature["carrier_family"]), cast(str, feature["document_type"]))
        ].append(feature)

    all_clusters: list[_TemplateCluster] = []
    for group_key in sorted(groups):
        group = groups[group_key]
        sequence_counts = Counter(
            tuple(cast(list[str], feature["first_page_anchor_sequence"]))
            for feature in group
        )
        ordered = sorted(
            group,
            key=lambda feature: (
                -sequence_counts[
                    tuple(cast(list[str], feature["first_page_anchor_sequence"]))
                ],
                cast(str, feature["document_id"]),
            ),
        )
        clusters: list[_TemplateCluster] = []
        for feature in ordered:
            vector = cast(tuple[float, ...], feature["_visual_vector"])
            anchors = cast(list[str], feature["first_page_anchor_sequence"])
            candidates: list[tuple[float, float, float, _TemplateCluster]] = []
            for cluster in clusters:
                representative = cluster.representative
                visual = _visual_similarity(
                    vector, cast(tuple[float, ...], representative["_visual_vector"])
                )
                if visual < visual_minimum:
                    continue
                ocr = _ocr_similarity(
                    anchors,
                    cast(list[str], representative["first_page_anchor_sequence"]),
                )
                combined = visual_weight * visual + (1.0 - visual_weight) * ocr
                if ocr >= ocr_minimum and combined >= combined_minimum:
                    candidates.append((combined, visual, ocr, cluster))
            if candidates:
                combined, visual, ocr, selected = max(
                    candidates,
                    key=lambda row: (
                        row[0],
                        row[1],
                        row[2],
                        cast(str, row[3].representative["document_id"]),
                    ),
                )
                selected.members.append((feature, visual, ocr, combined))
            else:
                clusters.append(
                    _TemplateCluster(
                        representative=feature,
                        members=[(feature, 1.0, 1.0, 1.0)],
                    )
                )
        all_clusters.extend(clusters)

    summaries: list[dict[str, Any]] = []
    for cluster in all_clusters:
        representative = cluster.representative
        identity = {
            "carrierFamily": representative["carrier_family"],
            "documentType": representative["document_type"],
            "representativeDocumentId": representative["document_id"],
            "representativeLayoutHash": representative["visual_layout_hash"],
            "representativeAnchors": representative["first_page_anchor_sequence"],
        }
        template_id = "template_" + sha256_bytes(canonical_json_bytes(identity))[:16]
        member_ids = sorted(cast(str, row[0]["document_id"]) for row in cluster.members)
        visual_values = [row[1] for row in cluster.members]
        ocr_values = [row[2] for row in cluster.members]
        combined_values = [row[3] for row in cluster.members]
        summary = {
            "template_id": template_id,
            "carrier_family": representative["carrier_family"],
            "document_type": representative["document_type"],
            "representative_document_id": representative["document_id"],
            "representative_source_filename": representative["source_filename"],
            "representative_anchor_sequence": representative[
                "first_page_anchor_sequence"
            ],
            "document_count": len(cluster.members),
            "member_document_ids": member_ids,
            "source_corpora": dict(
                sorted(Counter(row[0]["source_corpus"] for row in cluster.members).items())
            ),
            "page_count_min": min(cast(int, row[0]["page_count"]) for row in cluster.members),
            "page_count_max": max(cast(int, row[0]["page_count"]) for row in cluster.members),
            "visual_similarity_mean": round(statistics.fmean(visual_values), 6),
            "visual_similarity_min": round(min(visual_values), 6),
            "ocr_similarity_mean": round(statistics.fmean(ocr_values), 6),
            "ocr_similarity_min": round(min(ocr_values), 6),
            "combined_similarity_mean": round(statistics.fmean(combined_values), 6),
            "combined_similarity_min": round(min(combined_values), 6),
        }
        summaries.append(summary)
        if annotate:
            cluster_size = len(cluster.members)
            for feature, visual, ocr, combined in cluster.members:
                feature.update(
                    {
                        "template_proxy_id": template_id,
                        "template_proxy_cluster_size": cluster_size,
                        "template_proxy_singleton": cluster_size == 1,
                        "template_proxy_visual_similarity": round(visual, 6),
                        "template_proxy_ocr_similarity": round(ocr, 6),
                        "template_proxy_combined_similarity": round(combined, 6),
                    }
                )
    summaries.sort(key=lambda row: (-cast(int, row["document_count"]), row["template_id"]))
    return summaries


def _apply_template_analysis(
    features: Sequence[dict[str, Any]], settings: EdaAnalysisSettings
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    templates = _template_clusters(
        features,
        visual_minimum=settings.template_visual_minimum,
        ocr_minimum=settings.template_ocr_minimum,
        combined_minimum=settings.template_combined_minimum,
        visual_weight=settings.template_visual_weight,
        annotate=True,
    )
    sensitivity: list[dict[str, Any]] = []
    for row in settings.template_sensitivity:
        clusters = _template_clusters(
            features,
            visual_minimum=row.visual_minimum,
            ocr_minimum=row.ocr_minimum,
            combined_minimum=row.combined_minimum,
            visual_weight=settings.template_visual_weight,
            annotate=False,
        )
        cluster_sizes = [cast(int, cluster["document_count"]) for cluster in clusters]
        sensitivity.append(
            {
                "label": row.label,
                "visual_minimum": row.visual_minimum,
                "ocr_minimum": row.ocr_minimum,
                "combined_minimum": row.combined_minimum,
                "template_proxy_count": len(clusters),
                "singleton_template_proxy_count": sum(size == 1 for size in cluster_sizes),
                "largest_template_proxy_documents": max(cluster_sizes),
                "documents_in_non_singleton_templates": sum(
                    size for size in cluster_sizes if size > 1
                ),
            }
        )
    return templates, sensitivity


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DatasetEdaError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise DatasetEdaError("cannot describe an empty sequence")
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "p25": round(_percentile(values, 0.25), 6),
        "median": round(_percentile(values, 0.50), 6),
        "mean": round(statistics.fmean(values), 6),
        "p75": round(_percentile(values, 0.75), 6),
        "p90": round(_percentile(values, 0.90), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "max": round(max(values), 6),
    }


def _gini(counts: Sequence[int]) -> float:
    positive = sorted(value for value in counts if value > 0)
    if not positive:
        return 0.0
    total = sum(positive)
    weighted = sum(index * value for index, value in enumerate(positive, start=1))
    return (2.0 * weighted) / (len(positive) * total) - (len(positive) + 1) / len(
        positive
    )


def _concentration(counter: Counter[str]) -> dict[str, float | int]:
    total = counter.total()
    if total == 0:
        return {
            "categories": 0,
            "hhi": 0.0,
            "normalized_entropy": 0.0,
            "gini": 0.0,
            "top1_share": 0.0,
            "top5_share": 0.0,
        }
    probabilities = [count / total for count in counter.values()]
    entropy = -sum(value * math.log(value, 2) for value in probabilities)
    normalized_entropy = entropy / math.log(len(probabilities), 2) if len(probabilities) > 1 else 0.0
    ordered = sorted(counter.values(), reverse=True)
    return {
        "categories": len(counter),
        "hhi": round(sum(value * value for value in probabilities), 6),
        "normalized_entropy": round(normalized_entropy, 6),
        "gini": round(_gini(tuple(counter.values())), 6),
        "top1_share": round(ordered[0] / total, 6),
        "top5_share": round(sum(ordered[:5]) / total, 6),
    }


def _counter(
    features: Sequence[dict[str, Any]], key: str, *, explode: bool = False
) -> Counter[str]:
    values: list[str] = []
    for feature in features:
        value = feature[key]
        if explode:
            if not isinstance(value, list):
                raise DatasetEdaError(f"exploded feature is not a list: {key}")
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return Counter(values)


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    total = counter.total()
    cumulative = 0
    rows: list[dict[str, Any]] = []
    for rank, (value, count) in enumerate(counter.most_common(), start=1):
        cumulative += count
        rows.append(
            {
                "rank": rank,
                "value": value,
                "count": count,
                "share": round(count / total, 8) if total else 0.0,
                "cumulative_share": round(cumulative / total, 8) if total else 0.0,
            }
        )
    return rows


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise DatasetEdaError("correlation inputs are empty or have unequal lengths")
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)
    )
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum * right_sum)
    return 0.0 if denominator == 0.0 else numerator / denominator


def _ks_statistic(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        raise DatasetEdaError("KS statistic requires two non-empty samples")
    left_sorted, right_sorted = sorted(left), sorted(right)
    left_index = right_index = 0
    maximum = 0.0
    for value in sorted(set(left_sorted) | set(right_sorted)):
        while left_index < len(left_sorted) and left_sorted[left_index] <= value:
            left_index += 1
        while right_index < len(right_sorted) and right_sorted[right_index] <= value:
            right_index += 1
        maximum = max(
            maximum,
            abs(left_index / len(left_sorted) - right_index / len(right_sorted)),
        )
    return maximum


def _js_divergence(left: Counter[str], right: Counter[str]) -> float:
    if left.total() == 0 or right.total() == 0:
        raise DatasetEdaError("JS divergence requires two non-empty distributions")
    keys = set(left) | set(right)
    left_total, right_total = left.total(), right.total()
    divergence = 0.0
    for key in keys:
        p = left[key] / left_total
        q = right[key] / right_total
        midpoint = (p + q) / 2.0
        if p:
            divergence += 0.5 * p * math.log(p / midpoint, 2)
        if q:
            divergence += 0.5 * q * math.log(q / midpoint, 2)
    return divergence
