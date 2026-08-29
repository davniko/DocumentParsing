"""Deep, provenance-aware EDA for the combined MPCI Bill-of-Lading corpus."""

from __future__ import annotations

import argparse
import csv
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
        if len({row.label for row in self.template_sensitivity}) != len(self.template_sensitivity):
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
    manifest = _load_manifest(dataset_root, config.dataset.manifest_sha256, context="EDA dataset")
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
    ascii_like = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
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
                raise DatasetEdaError(
                    f"first-page raster has invalid dimensions: {task.document_id}"
                )
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
    for name in (
        "shipper",
        "consignee",
        "carrier",
        "deliveryAgent",
        "forwardingAgent",
        "consolidator",
    ):
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


def _vessel_present(transport: dict[str, Any]) -> bool:
    return any(
        transport.get(field) for field in ("vesselName", "vesselImoNumber", "vesselFlagCountry")
    )


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
    allocation_groups = cast(list[dict[str, Any]], patch.get("cargoAllocationGroups", []))

    carrier_name = cast(dict[str, Any], parties.get("carrier", {})).get("name")
    carrier_name = carrier_name if isinstance(carrier_name, str) else None
    carrier_key, carrier_rule_matched = carrier_family(carrier_name)
    shipper_country = cast(dict[str, Any], parties.get("shipper", {})).get("country")
    shipper_country = shipper_country if isinstance(shipper_country, str) else None
    load_country = cast(dict[str, Any], route.get("portOfLoading", {})).get("country")
    load_country = load_country if isinstance(load_country, str) else None
    discharge_country = cast(dict[str, Any], route.get("portOfDischarge", {})).get("country")
    discharge_country = discharge_country if isinstance(discharge_country, str) else None
    delivery_country = cast(dict[str, Any], route.get("placeOfDelivery", {})).get("country")
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
        cast(int, row["quantity"]) for row in packages if isinstance(row.get("quantity"), int)
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
        cast(str, row["unNumber"]) for row in dangerous if isinstance(row.get("unNumber"), str)
    ]
    hs_codes = [value for group in goods for value in cast(list[str], group.get("hsCodes", []))]

    issue_date = patch.get("issueDate")
    board_date = patch.get("shippedOnBoardDate")
    year_value = issue_date or board_date
    label_year = (
        int(year_value[:4])
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
    consolidator = parties.get("consolidator")
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
        "original_bill_of_lading_number_present": bool(patch.get("originalBillOfLadingNumber")),
        "master_bill_of_lading_number_present": bool(patch.get("masterBillOfLadingNumber")),
        "place_of_issue_present": bool(patch.get("placeOfIssue")),
        "transport_present": bool(transport),
        "vessel_present": _vessel_present(transport),
        "voyage_present": bool(transport.get("voyageNumber")),
        "place_of_receipt_present": bool(route.get("placeOfReceipt")),
        "pre_carriage_place_present": bool(route.get("preCarriagePlace")),
        "port_of_loading_present": bool(route.get("portOfLoading")),
        "transshipment_port_present": bool(route.get("transshipmentPort")),
        "port_of_discharge_present": bool(route.get("portOfDischarge")),
        "place_of_delivery_present": bool(route.get("placeOfDelivery")),
        "final_destination_present": bool(route.get("finalDestination")),
        "carrier_name": carrier_name or "<MISSING>",
        "carrier_family": carrier_key,
        "carrier_family_curated": carrier_rule_matched,
        "shipper_present": bool(parties.get("shipper")),
        "consignee_present": bool(parties.get("consignee")),
        "carrier_present": bool(parties.get("carrier")),
        "delivery_agent_present": bool(delivery_agent),
        "forwarding_agent_present": bool(forwarding),
        "consolidator_present": bool(consolidator),
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
        "trade_lane_proxy": (f"{country_group(load_country)} → {country_group(discharge_country)}"),
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
            legacy_annotation = BillOfLadingAnnotation.model_validate_json(payload, strict=True)
            source = legacy_annotation.source
            document_type = legacy_annotation.documentType
            evidence_fields = len(legacy_annotation.evidence)
        else:
            current_annotation = BillOfLadingDualCargoAnnotation.model_validate_json(
                payload, strict=True
            )
            source = current_annotation.source
            document_type = current_annotation.documentType
            evidence_fields = len(current_annotation.evidence) + len(
                current_annotation.relationEvidence
            )
            validate_annotation_evidence(
                AgentWorkItem(source=source, joinedRawText=raw_text), current_annotation
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
        source, document_type, annotation_path, annotation_evidence_fields = _annotation_and_source(
            record=record,
            lineage=lineage_by_id[document_id],
            current_root=current_root,
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
        groups[(cast(str, feature["carrier_family"]), cast(str, feature["document_type"]))].append(
            feature
        )

    all_clusters: list[_TemplateCluster] = []
    for group_key in sorted(groups):
        group = groups[group_key]
        sequence_counts = Counter(
            tuple(cast(list[str], feature["first_page_anchor_sequence"])) for feature in group
        )
        ordered = sorted(
            group,
            key=lambda feature: (
                -sequence_counts[tuple(cast(list[str], feature["first_page_anchor_sequence"]))],
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
            "representative_anchor_sequence": representative["first_page_anchor_sequence"],
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
    return (2.0 * weighted) / (len(positive) * total) - (len(positive) + 1) / len(positive)


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
    normalized_entropy = (
        entropy / math.log(len(probabilities), 2) if len(probabilities) > 1 else 0.0
    )
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
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
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


_COVERAGE_FIELDS: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
    ("bill_of_lading_number", lambda row: cast(bool, row["bill_of_lading_number_present"])),
    (
        "original_bill_of_lading_number",
        lambda row: cast(bool, row["original_bill_of_lading_number_present"]),
    ),
    (
        "master_bill_of_lading_number",
        lambda row: cast(bool, row["master_bill_of_lading_number_present"]),
    ),
    ("issue_date", lambda row: cast(bool, row["issue_date_present"])),
    ("shipped_on_board_date", lambda row: cast(bool, row["shipped_on_board_date_present"])),
    ("place_of_issue", lambda row: cast(bool, row["place_of_issue_present"])),
    ("vessel", lambda row: cast(bool, row["vessel_present"])),
    ("voyage", lambda row: cast(bool, row["voyage_present"])),
    ("place_of_receipt", lambda row: cast(bool, row["place_of_receipt_present"])),
    (
        "pre_carriage_place",
        lambda row: cast(bool, row["pre_carriage_place_present"]),
    ),
    ("port_of_loading", lambda row: cast(bool, row["port_of_loading_present"])),
    ("port_of_discharge", lambda row: cast(bool, row["port_of_discharge_present"])),
    (
        "transshipment_port",
        lambda row: cast(bool, row["transshipment_port_present"]),
    ),
    ("place_of_delivery", lambda row: cast(bool, row["place_of_delivery_present"])),
    (
        "final_destination",
        lambda row: cast(bool, row["final_destination_present"]),
    ),
    ("shipper", lambda row: cast(bool, row["shipper_present"])),
    ("consignee", lambda row: cast(bool, row["consignee_present"])),
    ("carrier", lambda row: cast(bool, row["carrier_present"])),
    ("delivery_agent", lambda row: cast(bool, row["delivery_agent_present"])),
    ("forwarding_agent", lambda row: cast(bool, row["forwarding_agent_present"])),
    ("consolidator", lambda row: cast(bool, row["consolidator_present"])),
    ("notify_party", lambda row: cast(int, row["notify_party_count"]) > 0),
    (
        "shipper_country_export_proxy",
        lambda row: row["export_country_proxy"] != "<MISSING>",
    ),
    (
        "port_of_loading_country",
        lambda row: row["port_of_loading_country"] != "<MISSING>",
    ),
    (
        "port_of_discharge_country",
        lambda row: row["port_of_discharge_country"] != "<MISSING>",
    ),
    ("freight_payment", lambda row: row["freight_payment_arrangement"] != "<MISSING>"),
    ("containers", lambda row: cast(int, row["container_count"]) > 0),
    ("container_seals", lambda row: cast(int, row["seal_count"]) > 0),
    ("container_vgm", lambda row: cast(int, row["container_vgm_count"]) > 0),
    ("temperature_setting", lambda row: cast(bool, row["temperature_present"])),
    ("goods", lambda row: cast(int, row["goods_group_count"]) > 0),
    ("packages", lambda row: cast(int, row["package_fact_count"]) > 0),
    (
        "package_quantities",
        lambda row: cast(int, row["package_quantity_fact_count"]) > 0,
    ),
    ("container_allocations", lambda row: cast(bool, row["has_allocations"])),
    ("hs_codes", lambda row: cast(int, row["hs_code_count"]) > 0),
    ("dangerous_goods", lambda row: cast(bool, row["dangerous_goods_present"])),
    ("gross_weight", lambda row: cast(int, row["gross_weight_group_count"]) > 0),
    ("net_weight", lambda row: cast(int, row["net_weight_group_count"]) > 0),
    ("volume", lambda row: cast(int, row["volume_group_count"]) > 0),
    ("marks", lambda row: cast(int, row["marks_group_count"]) > 0),
    (
        "handling_instructions",
        lambda row: cast(int, row["handling_instruction_group_count"]) > 0,
    ),
    (
        "additional_cargo_information",
        lambda row: cast(int, row["additional_information_group_count"]) > 0,
    ),
    (
        "forwarding_export_references",
        lambda row: cast(int, row["forwarding_export_reference_count"]) > 0,
    ),
    ("goods_origin", lambda row: cast(int, row["origin_group_count"]) > 0),
)


def _coverage_rows(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    cohorts = ("all", "legacy_combined487", "current_main680")
    rows: list[dict[str, Any]] = []
    for field, predicate in _COVERAGE_FIELDS:
        for cohort in cohorts:
            selected = (
                list(features)
                if cohort == "all"
                else [row for row in features if row["source_corpus"] == cohort]
            )
            present = sum(predicate(row) for row in selected)
            rows.append(
                {
                    "field": field,
                    "cohort": cohort,
                    "present_documents": present,
                    "total_documents": len(selected),
                    "coverage": round(present / len(selected), 8),
                }
            )
    return rows


_NUMERIC_FEATURES: tuple[str, ...] = (
    "page_count",
    "ocr_characters",
    "ocr_characters_per_page_mean",
    "source_file_bytes",
    "target_canonical_bytes",
    "target_leaf_count",
    "container_count",
    "seal_count",
    "goods_group_count",
    "package_fact_count",
    "package_quantity_sum",
    "max_package_levels_per_goods",
    "allocation_group_count",
    "allocation_row_count",
    "hs_code_count",
    "dangerous_goods_count",
    "notify_party_count",
    "party_with_contact_details_count",
    "annotation_evidence_field_count",
)


def _numeric_summaries(features: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for field in _NUMERIC_FEATURES:
        values = [float(cast(int | float, row[field])) for row in features]
        summaries[field] = _describe(values)
    return summaries


def _correlation_rows(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(_NUMERIC_FEATURES):
        left = [float(cast(int | float, row[left_name])) for row in features]
        for right_name in _NUMERIC_FEATURES[left_index:]:
            right = [float(cast(int | float, row[right_name])) for row in features]
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "pearson": round(_pearson(left, right), 8),
                }
            )
    return rows


def _cohort_comparison(features: Sequence[dict[str, Any]]) -> dict[str, Any]:
    legacy = [row for row in features if row["source_corpus"] == "legacy_combined487"]
    current = [row for row in features if row["source_corpus"] == "current_main680"]
    if not legacy or not current:
        raise DatasetEdaError("both source cohorts are required for drift analysis")
    numeric: list[dict[str, Any]] = []
    for field in _NUMERIC_FEATURES:
        left = [float(cast(int | float, row[field])) for row in legacy]
        right = [float(cast(int | float, row[field])) for row in current]
        pooled_variance = (
            ((len(left) - 1) * statistics.variance(left) if len(left) > 1 else 0.0)
            + ((len(right) - 1) * statistics.variance(right) if len(right) > 1 else 0.0)
        ) / max(1, len(left) + len(right) - 2)
        pooled_sd = math.sqrt(pooled_variance)
        mean_difference = statistics.fmean(right) - statistics.fmean(left)
        numeric.append(
            {
                "field": field,
                "legacy_mean": round(statistics.fmean(left), 6),
                "legacy_median": round(_percentile(left, 0.5), 6),
                "current_mean": round(statistics.fmean(right), 6),
                "current_median": round(_percentile(right, 0.5), 6),
                "current_minus_legacy_mean": round(mean_difference, 6),
                "standardized_mean_difference": round(
                    mean_difference / pooled_sd if pooled_sd else 0.0, 6
                ),
                "ks_statistic": round(_ks_statistic(left, right), 6),
            }
        )
    categorical: list[dict[str, Any]] = []
    for field in (
        "document_type",
        "carrier_family",
        "export_country_proxy",
        "port_of_loading_country",
        "port_of_discharge_country",
        "negotiability",
        "freight_payment_arrangement",
        "template_proxy_id",
    ):
        left_counter = _counter(legacy, field)
        right_counter = _counter(current, field)
        categorical.append(
            {
                "field": field,
                "legacy_categories": len(left_counter),
                "current_categories": len(right_counter),
                "jensen_shannon_divergence_bits": round(
                    _js_divergence(left_counter, right_counter), 6
                ),
            }
        )
    return {
        "cohort_documents": {"legacy_combined487": len(legacy), "current_main680": len(current)},
        "numeric": numeric,
        "categorical": categorical,
    }


def _outlier_rows(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = {
        field: _percentile([float(cast(int | float, row[field])) for row in features], 0.95)
        for field in (
            "page_count",
            "ocr_characters",
            "target_leaf_count",
            "container_count",
            "goods_group_count",
            "package_fact_count",
            "hs_code_count",
        )
    }
    rows: list[dict[str, Any]] = []
    for feature in features:
        reasons = [
            f"{field}>={threshold:g}"
            for field, threshold in thresholds.items()
            if float(cast(int | float, feature[field])) >= threshold
            and float(cast(int | float, feature[field])) > 0
        ]
        if feature["temperature_present"]:
            reasons.append("temperature_controlled")
        if feature["dangerous_goods_present"]:
            reasons.append("dangerous_goods")
        if reasons:
            rows.append(
                {
                    "document_id": feature["document_id"],
                    "source_filename": feature["source_filename"],
                    "source_corpus": feature["source_corpus"],
                    "carrier_family": feature["carrier_family"],
                    "template_proxy_id": feature["template_proxy_id"],
                    "reasons": reasons,
                    **{field: feature[field] for field in thresholds},
                }
            )
    rows.sort(key=lambda row: (-len(cast(list[str], row["reasons"])), row["document_id"]))
    return rows


def _joint_count(
    features: Sequence[dict[str, Any]], predicates: Sequence[Callable[[dict[str, Any]], bool]]
) -> int:
    return sum(all(predicate(row) for predicate in predicates) for row in features)


def _augmentation_priorities(
    features: Sequence[dict[str, Any]], settings: EdaAnalysisSettings
) -> list[dict[str, Any]]:
    single = settings.augmentation_minimum_documents
    joint = settings.augmentation_minimum_joint_documents
    scenarios: tuple[tuple[str, str, int, tuple[Callable[[dict[str, Any]], bool], ...]], ...] = (
        (
            "container_verified_gross_mass",
            "unsupported_in_real_corpus",
            single,
            (lambda row: cast(int, row["container_vgm_count"]) > 0,),
        ),
        (
            "pre_carriage_place",
            "unsupported_in_real_corpus",
            single,
            (lambda row: cast(bool, row["pre_carriage_place_present"]),),
        ),
        (
            "consolidator",
            "unsupported_in_real_corpus",
            single,
            (lambda row: cast(bool, row["consolidator_present"]),),
        ),
        (
            "original_bill_of_lading_number",
            "near_zero_schema_field",
            single,
            (lambda row: cast(bool, row["original_bill_of_lading_number_present"]),),
        ),
        (
            "master_bill_of_lading_number",
            "near_zero_schema_field",
            single,
            (lambda row: cast(bool, row["master_bill_of_lading_number_present"]),),
        ),
        (
            "transshipment_port",
            "rare_route_field",
            single,
            (lambda row: cast(bool, row["transshipment_port_present"]),),
        ),
        (
            "final_destination",
            "rare_route_field",
            single,
            (lambda row: cast(bool, row["final_destination_present"]),),
        ),
        (
            "handling_instructions",
            "rare_cargo_field",
            single,
            (lambda row: cast(int, row["handling_instruction_group_count"]) > 0,),
        ),
        (
            "dangerous_goods",
            "rare_field",
            single,
            (lambda row: cast(bool, row["dangerous_goods_present"]),),
        ),
        (
            "temperature_controlled",
            "rare_field",
            single,
            (lambda row: cast(bool, row["temperature_present"]),),
        ),
        (
            "forwarding_agent",
            "rare_party_role",
            single,
            (lambda row: cast(bool, row["forwarding_agent_present"]),),
        ),
        (
            "multi_goods",
            "cargo_complexity",
            single,
            (lambda row: cast(bool, row["multi_goods"]),),
        ),
        (
            "multi_package_level",
            "cargo_complexity",
            single,
            (lambda row: cast(bool, row["multi_package_level"]),),
        ),
        (
            "five_or_more_containers",
            "container_complexity",
            single,
            (lambda row: cast(bool, row["high_container_complexity"]),),
        ),
        (
            "five_or_more_hs_codes",
            "classification_complexity",
            single,
            (lambda row: cast(bool, row["high_hs_complexity"]),),
        ),
        (
            "dangerous_goods_and_temperature",
            "rare_joint",
            joint,
            (
                lambda row: cast(bool, row["dangerous_goods_present"]),
                lambda row: cast(bool, row["temperature_present"]),
            ),
        ),
        (
            "multi_goods_and_multi_container",
            "relation_joint",
            joint,
            (
                lambda row: cast(bool, row["multi_goods"]),
                lambda row: cast(bool, row["multi_container"]),
            ),
        ),
        (
            "multi_goods_and_multi_package_level",
            "relation_joint",
            joint,
            (
                lambda row: cast(bool, row["multi_goods"]),
                lambda row: cast(bool, row["multi_package_level"]),
            ),
        ),
        (
            "forwarding_and_delivery_agents",
            "party_joint",
            joint,
            (
                lambda row: cast(bool, row["forwarding_agent_present"]),
                lambda row: cast(bool, row["delivery_agent_present"]),
            ),
        ),
        (
            "allocations_with_multiple_goods",
            "relation_joint",
            joint,
            (
                lambda row: cast(bool, row["has_allocations"]),
                lambda row: cast(bool, row["multi_goods"]),
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    for scenario, category, target, predicates in scenarios:
        count = _joint_count(features, predicates)
        gap = max(0, target - count)
        rows.append(
            {
                "scenario": scenario,
                "category": category,
                "observed_documents": count,
                "observed_share": round(count / len(features), 8),
                "minimum_coverage_target": target,
                "minimum_additional_examples": gap,
                "priority_score": round(gap / target, 6),
            }
        )
    rows.sort(
        key=lambda row: (
            -cast(float, row["priority_score"]),
            cast(int, row["observed_documents"]),
            cast(str, row["scenario"]),
        )
    )
    return rows


def _rare_combinations(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in features:
        flags = [
            name
            for name, present in (
                ("dangerous", row["dangerous_goods_present"]),
                ("temperature", row["temperature_present"]),
                ("multi_goods", row["multi_goods"]),
                ("multi_container", row["multi_container"]),
                ("multi_package", row["multi_package_level"]),
                ("delivery_agent", row["delivery_agent_present"]),
                ("forwarding_agent", row["forwarding_agent_present"]),
                ("allocations", row["has_allocations"]),
            )
            if present
        ]
        counter[" + ".join(flags) if flags else "none_of_selected_features"] += 1
    return _counter_rows(counter)


def _normalized_exploded_counter(features: Sequence[dict[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for feature in features:
        values = feature[key]
        if not isinstance(values, list):
            raise DatasetEdaError(f"normalized exploded feature is not a list: {key}")
        counter.update(_normalized_words(str(value)) for value in values)
    return counter


def _categorical_analysis(
    features: Sequence[dict[str, Any]], templates: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {
        "document_type": _counter(features, "document_type"),
        "source_corpus": _counter(features, "source_corpus"),
        "source_class": _counter(features, "source_class"),
        "carrier_name_exact": _counter(features, "carrier_name"),
        "carrier_family": _counter(features, "carrier_family"),
        "export_country_proxy": _counter(features, "export_country_proxy"),
        "port_of_loading_country": _counter(features, "port_of_loading_country"),
        "port_of_discharge_country": _counter(features, "port_of_discharge_country"),
        "place_of_delivery_country": _counter(features, "place_of_delivery_country"),
        "goods_origin": _counter(features, "goods_origins", explode=True),
        "trade_lane_proxy": _counter(features, "trade_lane_proxy"),
        "container_type_printed_normalized": _normalized_exploded_counter(
            features, "container_types"
        ),
        "package_type_printed_normalized": _normalized_exploded_counter(features, "package_types"),
        "hazard_category": _counter(features, "hazard_categories", explode=True),
        "subsidiary_hazard_category": _counter(
            features, "subsidiary_hazard_categories", explode=True
        ),
        "un_number": _counter(features, "un_numbers", explode=True),
        "freight_payment_arrangement": _counter(features, "freight_payment_arrangement"),
        "allocation_coverage": _counter(features, "allocation_coverages", explode=True),
        "temperature_unit": _counter(features, "temperature_units", explode=True),
        "negotiability": _counter(features, "negotiability"),
        "first_page_orientation": _counter(features, "first_page_orientation"),
        "template_proxy": _counter(features, "template_proxy_id"),
    }
    result: dict[str, Any] = {}
    for name, counter in counters.items():
        result[name] = {
            "distribution": _counter_rows(counter),
            "concentration": _concentration(counter),
        }
    template_carriers = Counter(cast(str, row["carrier_family"]) for row in templates)
    result["template_proxy_carrier_families"] = {
        "distribution": _counter_rows(template_carriers),
        "concentration": _concentration(template_carriers),
    }
    return result


def _bucket_count(value: int, boundaries: Sequence[int]) -> str:
    if not boundaries:
        raise DatasetEdaError("bucket boundaries must not be empty")
    for boundary in boundaries:
        if value <= boundary:
            return str(value) if boundary <= 4 else f"≤{boundary}"
    return f">{boundaries[-1]}"


def _count_buckets(
    features: Sequence[dict[str, Any]], field: str, boundaries: Sequence[int]
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in features:
        counter[_bucket_count(cast(int, row[field]), boundaries)] += 1
    return counter


def _top_items(
    counter: Counter[str], limit: int, *, omit_missing: bool = False
) -> list[tuple[str, float]]:
    items = counter.most_common()
    if omit_missing:
        items = [(key, value) for key, value in items if key != "<MISSING>"]
    return [(key, float(value)) for key, value in items[:limit]]


def _pareto_series(counter: Counter[str]) -> list[tuple[float, float]]:
    counts = sorted(counter.values(), reverse=True)
    total = sum(counts) or 1
    cumulative = 0
    points: list[tuple[float, float]] = []
    for index, count in enumerate(counts, start=1):
        cumulative += count
        points.append((float(index), cumulative / total))
    return points


def _coverage_matrix(
    coverage: Sequence[dict[str, Any]], fields: Sequence[str], cohorts: Sequence[str]
) -> list[list[float]]:
    lookup = {
        (cast(str, row["field"]), cast(str, row["cohort"])): cast(float, row["coverage"])
        for row in coverage
    }
    return [[lookup[(field, cohort)] for cohort in cohorts] for field in fields]


def _correlation_matrix(
    correlations: Sequence[dict[str, Any]], fields: Sequence[str]
) -> list[list[float]]:
    lookup: dict[tuple[str, str], float] = {}
    for row in correlations:
        left, right = cast(str, row["left"]), cast(str, row["right"])
        value = cast(float, row["pearson"])
        lookup[(left, right)] = value
        lookup[(right, left)] = value
    return [[lookup[(left, right)] for right in fields] for left in fields]


def _joint_matrix(
    features: Sequence[dict[str, Any]], left: str, right: str, maximum: int
) -> tuple[list[str], list[str], list[list[float]]]:
    x_labels = [str(value) for value in range(maximum)] + [f"{maximum}+"]
    y_labels = list(x_labels)
    matrix = [[0.0 for _ in x_labels] for _ in y_labels]
    for row in features:
        x_value = min(cast(int, row[left]), maximum)
        y_value = min(cast(int, row[right]), maximum)
        matrix[y_value][x_value] += 1.0
    return x_labels, y_labels, matrix


def _plot_artifacts(
    *,
    features: Sequence[dict[str, Any]],
    templates: Sequence[dict[str, Any]],
    sensitivity: Sequence[dict[str, Any]],
    coverage: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    cohort: dict[str, Any],
    augmentation: Sequence[dict[str, Any]],
    categorical: dict[str, Any],
    settings: EdaAnalysisSettings,
) -> list[tuple[str, bytes]]:
    total = len(features)
    artifacts: list[tuple[str, bytes]] = []

    def add(name: str, payload: bytes) -> None:
        artifacts.append((f"plots/{name}.png", payload))

    def distribution(name: str) -> Counter[str]:
        rows = categorical[name]["distribution"]
        return Counter({cast(str, row["value"]): cast(int, row["count"]) for row in rows})

    add(
        "01_document_types",
        horizontal_bar(
            title="Document type composition",
            subtitle="Validated OCR-conditioned labels",
            items=_top_items(distribution("document_type"), 10),
            total=total,
            footnote=f"n={total:,} documents",
        ),
    )
    add(
        "02_source_cohorts",
        horizontal_bar(
            title="Training source cohorts",
            subtitle="Aligned legacy versus current single-source policy labels",
            items=_top_items(distribution("source_corpus"), 10),
            total=total,
            footnote="Cohorts are compared separately for coverage and drift.",
        ),
    )
    pages = Counter(str(row["page_count"]) for row in features)
    add(
        "03_page_count",
        vertical_bar(
            title="Pages per document",
            subtitle="All source pages remain ordered in joined raw OCR",
            items=[(key, float(pages[key])) for key in sorted(pages, key=int)],
            footnote=f"median={_percentile([float(row['page_count']) for row in features], 0.5):g}",
        ),
    )
    add(
        "04_ocr_characters",
        histogram(
            title="Raw OCR length",
            subtitle="Joined page-ordered text characters per document",
            values=[float(row["ocr_characters"]) for row in features],
            bins=settings.histogram_bins,
            footnote="Long inputs should be represented in training and evaluation splits.",
        ),
    )
    add(
        "05_ocr_density",
        histogram(
            title="OCR characters per page",
            subtitle="Mean page density within each document",
            values=[float(row["ocr_characters_per_page_mean"]) for row in features],
            bins=settings.histogram_bins,
            footnote="Density is descriptive; it is not an OCR-quality score.",
        ),
    )
    add(
        "06_source_pdf_size",
        histogram(
            title="Source PDF size",
            subtitle="File size in MiB",
            values=[float(row["source_file_bytes"]) / 1_048_576 for row in features],
            bins=settings.histogram_bins,
            footnote="Large files may reflect scans, page count, or compression.",
        ),
    )
    for number, (field, title, subtitle, boundaries) in enumerate(
        (
            (
                "container_count",
                "Containers per document",
                "Relation target container rows",
                (0, 1, 2, 3, 4, 9),
            ),
            (
                "goods_group_count",
                "Goods groups per document",
                "Distinct source-grounded cargo groups",
                (0, 1, 2, 3, 4, 9),
            ),
            (
                "package_fact_count",
                "Package facts per document",
                "Outer and inner package facts retained by the relation schema",
                (0, 1, 2, 3, 4, 9),
            ),
            (
                "hs_code_count",
                "HS codes per document",
                "Printed HS-code values; no application-side maximum imposed",
                (0, 1, 2, 3, 4, 9),
            ),
        ),
        start=7,
    ):
        bucket = _count_buckets(features, field, boundaries)
        order = ["0", "1", "2", "3", "4", "≤9", ">9"]
        add(
            f"{number:02d}_{field}",
            vertical_bar(
                title=title,
                subtitle=subtitle,
                items=[(key, float(bucket[key])) for key in order if bucket[key]],
                footnote=f"n={total:,} documents",
            ),
        )
    add(
        "11_target_leaf_count",
        histogram(
            title="Target fact density",
            subtitle="Canonical relation-target leaf values per document",
            values=[float(row["target_leaf_count"]) for row in features],
            bins=settings.histogram_bins,
            footnote="Higher counts usually reflect containers, cargo rows, and relation detail.",
        ),
    )
    carrier_family = distribution("carrier_family")
    add(
        "12_carrier_families",
        horizontal_bar(
            title="Top carrier families",
            subtitle="Conservative curated name grouping; missing remains a real category",
            items=_top_items(carrier_family, settings.top_n),
            total=total,
            footnote=(
                f"{len(carrier_family):,} grouped values including missing and ungrouped names"
            ),
        ),
    )
    add(
        "13_carrier_pareto",
        line_chart(
            title="Carrier concentration",
            subtitle="Cumulative share of documents across carrier families",
            series=(("carrier families", _pareto_series(carrier_family)),),
            x_label="carrier-family rank",
            y_label="cumulative document share",
            footnote="The missing-carrier category is retained because the model cannot infer it.",
        ),
    )
    add(
        "14_exact_carrier_names",
        horizontal_bar(
            title="Top printed carrier names",
            subtitle="Exact label strings before conservative grouping",
            items=_top_items(distribution("carrier_name_exact"), settings.top_n),
            total=total,
            footnote=(
                "Spelling/legal-name variation remains visible in the machine-readable tables."
            ),
        ),
    )
    template_sizes = Counter(str(row["template_proxy_cluster_size"]) for row in features)
    add(
        "15_template_cluster_sizes",
        histogram(
            title="Conservative template-proxy cluster sizes",
            subtitle="First-page raster layout plus OCR heading order, within carrier/type",
            values=[float(row["document_count"]) for row in templates],
            bins=settings.histogram_bins,
            footnote=(
                f"{len(templates):,} proxy clusters; {template_sizes['1']:,} singleton documents"
            ),
        ),
    )
    add(
        "16_largest_template_clusters",
        horizontal_bar(
            title="Largest conservative template proxies",
            subtitle="Labels show carrier family and short template ID",
            items=[
                (
                    f"{row['carrier_family']} · {cast(str, row['template_id'])[-8:]}",
                    float(row["document_count"]),
                )
                for row in templates[: settings.top_n]
            ],
            total=total,
            footnote="Proxy clusters are not asserted ground-truth production templates.",
        ),
    )
    add(
        "17_template_sensitivity",
        grouped_bar(
            title="Template-proxy sensitivity",
            subtitle="Cluster count and singleton count across threshold choices",
            categories=[cast(str, row["label"]) for row in sensitivity],
            series=(
                (
                    "all clusters",
                    [float(row["template_proxy_count"]) for row in sensitivity],
                ),
                (
                    "singletons",
                    [float(row["singleton_template_proxy_count"]) for row in sensitivity],
                ),
            ),
            footnote="Conclusions should be stable across this range, not tied to one cutoff.",
        ),
    )
    for number, (key, title, subtitle) in enumerate(
        (
            (
                "export_country_proxy",
                "Shipper-country export proxy",
                "Country printed for the shipper; not a guaranteed country of export",
            ),
            (
                "port_of_loading_country",
                "Port-of-loading countries",
                "Country printed with port of loading",
            ),
            (
                "port_of_discharge_country",
                "Port-of-discharge countries",
                "Country printed with port of discharge",
            ),
            (
                "goods_origin",
                "Printed goods origins",
                "Explicit cargo origin only; much sparser than shipper country",
            ),
            (
                "trade_lane_proxy",
                "Top printed trade-lane proxies",
                "Loading-country to discharge-country; missing endpoints retained",
            ),
        ),
        start=18,
    ):
        counter = distribution(key)
        add(
            f"{number:02d}_{key}",
            horizontal_bar(
                title=title,
                subtitle=subtitle,
                items=_top_items(counter, settings.top_n),
                total=counter.total(),
                footnote=f"{len(counter):,} observed grouped values",
            ),
        )
    for number, (key, title, subtitle) in enumerate(
        (
            (
                "container_type_printed_normalized",
                "Printed container-type vocabulary",
                "Normalized spelling for EDA only; labels remain as printed",
            ),
            (
                "package_type_printed_normalized",
                "Printed package-type vocabulary",
                "Normalized spelling for EDA only; outer/inner facts remain separate",
            ),
        ),
        start=23,
    ):
        counter = distribution(key)
        add(
            f"{number:02d}_{key}",
            horizontal_bar(
                title=title,
                subtitle=subtitle,
                items=_top_items(counter, settings.top_n),
                total=counter.total(),
                footnote=(
                    f"{counter.total():,} facts across {len(counter):,} normalized printed values"
                ),
            ),
        )
    party_values = [
        (
            field,
            float(
                next(
                    row["present_documents"]
                    for row in coverage
                    if row["field"] == field and row["cohort"] == "all"
                )
            ),
        )
        for field in (
            "shipper",
            "consignee",
            "carrier",
            "notify_party",
            "delivery_agent",
            "forwarding_agent",
            "consolidator",
        )
    ]
    add(
        "25_party_role_coverage",
        horizontal_bar(
            title="Party-role coverage",
            subtitle="Documents with an OCR-grounded role in the target",
            items=party_values,
            total=total,
            footnote=(
                "Absent values remain null; the pipeline does not infer parties from the PDF image."
            ),
        ),
    )
    special = [
        ("container VGM", sum(cast(int, row["container_vgm_count"]) > 0 for row in features)),
        ("temperature", sum(bool(row["temperature_present"]) for row in features)),
        ("dangerous goods", sum(bool(row["dangerous_goods_present"]) for row in features)),
        ("multi-goods", sum(bool(row["multi_goods"]) for row in features)),
        ("multi-package level", sum(bool(row["multi_package_level"]) for row in features)),
        ("5+ containers", sum(bool(row["high_container_complexity"]) for row in features)),
        ("5+ HS codes", sum(bool(row["high_hs_complexity"]) for row in features)),
    ]
    add(
        "26_rare_and_complex_features",
        horizontal_bar(
            title="Rare and complex document features",
            subtitle="Coverage signals most relevant to targeted augmentation",
            items=[(name, float(value)) for name, value in special],
            total=total,
            footnote=(
                "Small counts imply high variance in evaluation and weak coverage of combinations."
            ),
        ),
    )
    add(
        "27_dangerous_goods_hazards",
        horizontal_bar(
            title="Dangerous-goods hazard categories",
            subtitle="Printed hazard categories across dangerous-goods rows",
            items=_top_items(distribution("hazard_category"), settings.top_n),
            footnote="Fact counts, not document counts.",
        ),
    )
    add(
        "28_dangerous_goods_un_numbers",
        horizontal_bar(
            title="Dangerous-goods UN numbers",
            subtitle="Most common printed UN numbers",
            items=_top_items(distribution("un_number"), settings.top_n),
            footnote="Long-tail values require constrained synthetic diversification.",
        ),
    )
    temperature_values = [
        float(value)
        for row in features
        for value in cast(list[float], row["temperature_values_celsius"])
    ]
    if temperature_values:
        add(
            "29_temperature_setpoints_celsius",
            histogram(
                title="Temperature setpoints",
                subtitle="All printed setpoints converted to Celsius for analysis only",
                values=temperature_values,
                bins=min(settings.histogram_bins, 16),
                footnote="Training labels retain the printed value and unit.",
            ),
        )
    add(
        "30_allocation_coverage_modes",
        horizontal_bar(
            title="Cargo-container relation modes",
            subtitle="Explicit allocation coverage ontology",
            items=_top_items(distribution("allocation_coverage"), settings.top_n),
            footnote="Fact counts; missing relations are represented by no allocation group.",
        ),
    )
    coverage_fields = [field for field, _ in _COVERAGE_FIELDS]
    add(
        "31_field_coverage_by_cohort",
        heatmap(
            title="Target field coverage by source cohort",
            subtitle="Document-level presence rates",
            x_labels=("all", "legacy", "current"),
            y_labels=coverage_fields,
            matrix=_coverage_matrix(
                coverage,
                coverage_fields,
                ("all", "legacy_combined487", "current_main680"),
            ),
            value_format="percent",
            footnote="Differences can reflect source mix as well as annotation-policy evolution.",
        ),
    )
    correlation_fields = (
        "page_count",
        "ocr_characters",
        "target_leaf_count",
        "container_count",
        "goods_group_count",
        "package_fact_count",
        "allocation_row_count",
        "hs_code_count",
        "seal_count",
    )
    correlation_labels = (
        "pages",
        "OCR chars",
        "target leaves",
        "containers",
        "goods",
        "packages",
        "allocations",
        "HS codes",
        "seals",
    )
    correlation_values = _correlation_matrix(correlations, correlation_fields)
    add(
        "32_numeric_correlations",
        heatmap(
            title="Numeric feature correlations",
            subtitle="Pearson r shown in cells; color scale spans -1 to +1",
            x_labels=correlation_labels,
            y_labels=correlation_labels,
            matrix=correlation_values,
            value_format="float",
            center_zero=True,
            footnote=(
                "Blue is positive, red is negative; exact values are also in correlations.csv."
            ),
        ),
    )
    x_labels, y_labels, matrix = _joint_matrix(features, "container_count", "goods_group_count", 6)
    add(
        "33_containers_by_goods",
        heatmap(
            title="Containers versus goods groups",
            subtitle="Document counts; 6+ values are tail-bucketed",
            x_labels=x_labels,
            y_labels=y_labels,
            matrix=matrix,
            footnote="Sparse off-diagonal cells identify relation-complexity augmentation needs.",
        ),
    )
    x_labels, y_labels, matrix = _joint_matrix(
        features, "goods_group_count", "package_fact_count", 6
    )
    add(
        "34_goods_by_package_facts",
        heatmap(
            title="Goods groups versus package facts",
            subtitle="Document counts; package-level nesting creates off-diagonal cases",
            x_labels=x_labels,
            y_labels=y_labels,
            matrix=matrix,
            footnote=(
                "The relation target retains outer/inner facts without duplicating "
                "goods descriptions."
            ),
        ),
    )
    numeric_lookup = {
        cast(str, row["field"]): row for row in cast(list[dict[str, Any]], cohort["numeric"])
    }
    add(
        "35_cohort_numeric_shift",
        grouped_bar(
            title="Selected cohort means",
            subtitle="Legacy versus current-policy records",
            categories=("pages", "containers", "goods", "packages", "HS codes"),
            series=(
                (
                    "legacy",
                    [
                        float(numeric_lookup[field]["legacy_mean"])
                        for field in (
                            "page_count",
                            "container_count",
                            "goods_group_count",
                            "package_fact_count",
                            "hs_code_count",
                        )
                    ],
                ),
                (
                    "current",
                    [
                        float(numeric_lookup[field]["current_mean"])
                        for field in (
                            "page_count",
                            "container_count",
                            "goods_group_count",
                            "package_fact_count",
                            "hs_code_count",
                        )
                    ],
                ),
            ),
            footnote="See cohort-comparison.json for KS and standardized mean differences.",
        ),
    )
    year_counter = Counter(
        str(row["source_filename_year"])
        for row in features
        if row["source_filename_year"] is not None
    )
    add(
        "36_document_years",
        vertical_bar(
            title="Document years from source filenames",
            subtitle="Acquisition-year proxy, independently retained from label dates",
            items=[(key, float(year_counter[key])) for key in sorted(year_counter)],
            footnote="Filename year is provenance metadata, not a model target.",
        ),
    )
    top_carriers = [name for name, _ in carrier_family.most_common(10)]
    add(
        "37_document_type_by_carrier",
        grouped_bar(
            title="Document type within top carrier families",
            subtitle="B/L versus sea-waybill composition",
            categories=top_carriers,
            series=tuple(
                (
                    document_type,
                    [
                        float(
                            sum(
                                row["carrier_family"] == carrier
                                and row["document_type"] == document_type
                                for row in features
                            )
                        )
                        for carrier in top_carriers
                    ],
                )
                for document_type in ("bill_of_lading", "sea_waybill")
            ),
            footnote=(
                "Missing carrier is intentionally visible when it ranks in the top categories."
            ),
        ),
    )
    carrier_documents = Counter(str(row["carrier_family"]) for row in features)
    carrier_templates: dict[str, set[str]] = defaultdict(set)
    for row in features:
        carrier_templates[cast(str, row["carrier_family"])].add(cast(str, row["template_proxy_id"]))
    add(
        "38_carrier_template_diversity",
        scatter(
            title="Carrier size versus template diversity",
            subtitle="Conservative proxy clusters within each carrier family",
            points=[
                ScatterPoint(
                    x=float(carrier_documents[carrier]),
                    y=float(len(template_ids)),
                    label=carrier,
                    size=float(carrier_documents[carrier]),
                )
                for carrier, template_ids in carrier_templates.items()
            ],
            x_label="documents",
            y_label="template-proxy clusters",
            footnote="A high ratio can indicate genuine form diversity or OCR/layout variability.",
        ),
    )
    visible_carrier_templates = {
        carrier: template_ids
        for carrier, template_ids in carrier_templates.items()
        if carrier != "<MISSING>"
    }
    add(
        "38b_carrier_template_diversity_known_carriers",
        scatter(
            title="Known-carrier template diversity",
            subtitle="Same view with the missing-carrier category excluded",
            points=[
                ScatterPoint(
                    x=float(carrier_documents[carrier]),
                    y=float(len(template_ids)),
                    label=carrier,
                    size=float(carrier_documents[carrier]),
                )
                for carrier, template_ids in visible_carrier_templates.items()
            ],
            x_label="documents",
            y_label="template-proxy clusters",
            footnote="Use with the all-carrier plot; missingness remains part of the dataset.",
        ),
    )
    add(
        "39_augmentation_minimum_gaps",
        horizontal_bar(
            title="Minimum targeted augmentation gaps",
            subtitle="Additional examples to reach configured coverage floors",
            items=[
                (cast(str, row["scenario"]), float(row["minimum_additional_examples"]))
                for row in augmentation
            ],
            footnote="Floors are planning heuristics, not guarantees of model accuracy.",
            color="#F59E0B",
        ),
    )
    source_quantile_fields = (
        ("pages", "page_count"),
        ("OCR chars / 1k", "ocr_characters"),
        ("target leaves", "target_leaf_count"),
        ("containers", "container_count"),
        ("packages", "package_fact_count"),
    )
    quantile_rows: list[tuple[str, float, float, float, float, float]] = []
    scale = {"ocr_characters": 1000.0}
    for label, field in source_quantile_fields:
        values = [float(cast(int | float, row[field])) / scale.get(field, 1.0) for row in features]
        quantile_rows.append(
            (
                label,
                min(values),
                _percentile(values, 0.25),
                _percentile(values, 0.50),
                _percentile(values, 0.75),
                max(values),
            )
        )
    add(
        "40_complexity_quantile_ranges",
        quantile_ranges(
            title="Dataset complexity ranges",
            subtitle="Minimum, interquartile range, median, and maximum",
            rows=quantile_rows,
            footnote="Extreme tails are real documents and should not be lost in random splitting.",
        ),
    )
    document_types = ("bill_of_lading", "sea_waybill")
    special_predicates: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
        ("temperature", lambda row: cast(bool, row["temperature_present"])),
        ("dangerous", lambda row: cast(bool, row["dangerous_goods_present"])),
        ("delivery agent", lambda row: cast(bool, row["delivery_agent_present"])),
        ("forwarder", lambda row: cast(bool, row["forwarding_agent_present"])),
        ("multi-goods", lambda row: cast(bool, row["multi_goods"])),
        ("multi-container", lambda row: cast(bool, row["multi_container"])),
    )
    add(
        "41_special_features_by_document_type",
        grouped_bar(
            title="Special-feature coverage by document type",
            subtitle="Document-level rates within bills of lading and sea waybills",
            categories=[name for name, _ in special_predicates],
            series=tuple(
                (
                    document_type,
                    [
                        sum(
                            predicate(row)
                            for row in features
                            if row["document_type"] == document_type
                        )
                        / sum(row["document_type"] == document_type for row in features)
                        for _, predicate in special_predicates
                    ],
                )
                for document_type in document_types
            ),
            percent=True,
            footnote="Rates expose whether rare features are confined to one document type.",
        ),
    )
    known_carriers = [
        carrier for carrier, _ in carrier_family.most_common() if carrier != "<MISSING>"
    ][:10]
    party_roles: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
        ("notify", lambda row: cast(int, row["notify_party_count"]) > 0),
        ("delivery", lambda row: cast(bool, row["delivery_agent_present"])),
        ("forwarder", lambda row: cast(bool, row["forwarding_agent_present"])),
        ("contacts", lambda row: cast(int, row["party_with_contact_details_count"]) > 0),
    )
    carrier_party_matrix: list[list[float]] = []
    for carrier in known_carriers:
        rows = [row for row in features if row["carrier_family"] == carrier]
        carrier_party_matrix.append(
            [sum(predicate(row) for row in rows) / len(rows) for _, predicate in party_roles]
        )
    add(
        "42_party_roles_by_carrier",
        heatmap(
            title="Party-role coverage in top known carriers",
            subtitle="Within-carrier document-level rates",
            x_labels=[name for name, _ in party_roles],
            y_labels=known_carriers,
            matrix=carrier_party_matrix,
            value_format="percent",
            footnote="Carrier associations describe this sample; they are not business rules.",
        ),
    )
    carrier_complexity_fields = (
        ("containers", "container_count"),
        ("goods", "goods_group_count"),
        ("packages", "package_fact_count"),
    )
    add(
        "43_cargo_complexity_by_carrier",
        grouped_bar(
            title="Cargo complexity in top known carriers",
            subtitle="Mean target rows per document",
            categories=known_carriers,
            series=tuple(
                (
                    label,
                    [
                        statistics.fmean(
                            float(cast(int, row[field]))
                            for row in features
                            if row["carrier_family"] == carrier
                        )
                        for carrier in known_carriers
                    ],
                )
                for label, field in carrier_complexity_fields
            ),
            footnote="Means are sensitive to a few high-container tail documents.",
        ),
    )
    add(
        "44_template_repetition_by_carrier",
        grouped_bar(
            title="Template repetition in top known carriers",
            subtitle="Share inside non-singleton versus singleton proxy clusters",
            categories=known_carriers,
            series=(
                (
                    "repeated proxy",
                    [
                        sum(
                            not cast(bool, row["template_proxy_singleton"])
                            for row in features
                            if row["carrier_family"] == carrier
                        )
                        / carrier_documents[carrier]
                        for carrier in known_carriers
                    ],
                ),
                (
                    "singleton proxy",
                    [
                        sum(
                            cast(bool, row["template_proxy_singleton"])
                            for row in features
                            if row["carrier_family"] == carrier
                        )
                        / carrier_documents[carrier]
                        for carrier in known_carriers
                    ],
                ),
            ),
            percent=True,
            footnote="Proxy sensitivity remains documented separately.",
        ),
    )
    cohorts = ("legacy_combined487", "current_main680")
    add(
        "45_special_features_by_cohort",
        grouped_bar(
            title="Special-feature coverage by source cohort",
            subtitle="Aligned legacy versus current-policy document rates",
            categories=[name for name, _ in special_predicates],
            series=tuple(
                (
                    cohort_name,
                    [
                        sum(
                            predicate(row)
                            for row in features
                            if row["source_corpus"] == cohort_name
                        )
                        / sum(row["source_corpus"] == cohort_name for row in features)
                        for _, predicate in special_predicates
                    ],
                )
                for cohort_name in cohorts
            ),
            percent=True,
            footnote="Coverage differences motivate cohort-specific evaluation metrics.",
        ),
    )
    freight_counter = distribution("freight_payment_arrangement")
    add(
        "46_freight_payment_arrangements",
        horizontal_bar(
            title="Freight payment arrangements",
            subtitle="Sparse categorical target values",
            items=_top_items(freight_counter, 10),
            total=total,
            footnote="Missing means no grounded arrangement, not a default payment term.",
        ),
    )
    source_classes = tuple(sorted({cast(str, row["source_class"]) for row in features}))
    source_type_matrix = [
        [
            float(
                sum(
                    row["source_class"] == source_class and row["document_type"] == document_type
                    for row in features
                )
            )
            for document_type in document_types
        ]
        for source_class in source_classes
    ]
    add(
        "47_source_folder_by_document_type",
        heatmap(
            title="Source folder versus validated document type",
            subtitle="Corpus path class is provenance; annotation type is the label",
            x_labels=document_types,
            y_labels=source_classes,
            matrix=source_type_matrix,
            footnote="This exposes BLC/SWB folder-label mixtures without changing targets.",
        ),
    )
    page_quantiles: list[tuple[str, float, float, float, float, float]] = []
    for document_type in document_types:
        values = [
            float(row["page_count"]) for row in features if row["document_type"] == document_type
        ]
        page_quantiles.append(
            (
                document_type,
                min(values),
                _percentile(values, 0.25),
                _percentile(values, 0.50),
                _percentile(values, 0.75),
                max(values),
            )
        )
    add(
        "48_page_ranges_by_document_type",
        quantile_ranges(
            title="Page-count ranges by document type",
            subtitle="Minimum, interquartile range, median, and maximum",
            rows=page_quantiles,
            footnote="Tail pages should remain represented in group-aware evaluation.",
        ),
    )
    add(
        "49_template_repetition_by_document_type",
        grouped_bar(
            title="Template repetition by document type",
            subtitle="Share inside repeated versus singleton proxy clusters",
            categories=document_types,
            series=(
                (
                    "repeated proxy",
                    [
                        sum(
                            row["document_type"] == document_type
                            and not cast(bool, row["template_proxy_singleton"])
                            for row in features
                        )
                        / sum(row["document_type"] == document_type for row in features)
                        for document_type in document_types
                    ],
                ),
                (
                    "singleton proxy",
                    [
                        sum(
                            row["document_type"] == document_type
                            and cast(bool, row["template_proxy_singleton"])
                            for row in features
                        )
                        / sum(row["document_type"] == document_type for row in features)
                        for document_type in document_types
                    ],
                ),
            ),
            percent=True,
            footnote="Document-type imbalance and template repetition are separate effects.",
        ),
    )
    return artifacts


def _carrier_group_rows(features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        grouped[cast(str, row["carrier_family"])].append(row)
    output: list[dict[str, Any]] = []
    for carrier, rows in grouped.items():
        output.append(
            {
                "carrier_family": carrier,
                "document_count": len(rows),
                "share": round(len(rows) / len(features), 8),
                "exact_name_count": len({row["carrier_name"] for row in rows}),
                "template_proxy_count": len({row["template_proxy_id"] for row in rows}),
                "bill_of_lading_count": sum(
                    row["document_type"] == "bill_of_lading" for row in rows
                ),
                "sea_waybill_count": sum(row["document_type"] == "sea_waybill" for row in rows),
                "source_corpora": dict(Counter(row["source_corpus"] for row in rows)),
                "top_export_country_proxies": [
                    {"value": value, "count": count}
                    for value, count in Counter(
                        row["export_country_proxy"] for row in rows
                    ).most_common(10)
                ],
                "page_count_mean": round(
                    statistics.fmean(cast(int, row["page_count"]) for row in rows), 6
                ),
                "container_count_mean": round(
                    statistics.fmean(cast(int, row["container_count"]) for row in rows),
                    6,
                ),
            }
        )
    output.sort(key=lambda row: (-cast(int, row["document_count"]), row["carrier_family"]))
    return output


def _joint_feature_counts(features: Sequence[dict[str, Any]]) -> dict[str, int]:
    predicates: dict[str, Callable[[dict[str, Any]], bool]] = {
        "temperature_and_dangerous_goods": lambda row: bool(
            row["temperature_present"] and row["dangerous_goods_present"]
        ),
        "temperature_and_multi_container": lambda row: bool(
            row["temperature_present"] and row["multi_container"]
        ),
        "dangerous_goods_and_multi_container": lambda row: bool(
            row["dangerous_goods_present"] and row["multi_container"]
        ),
        "multi_goods_and_multi_container": lambda row: bool(
            row["multi_goods"] and row["multi_container"]
        ),
        "multi_goods_and_multi_package_level": lambda row: bool(
            row["multi_goods"] and row["multi_package_level"]
        ),
        "delivery_and_forwarding_agents": lambda row: bool(
            row["delivery_agent_present"] and row["forwarding_agent_present"]
        ),
        "sea_waybill_and_dangerous_goods": lambda row: bool(
            row["document_type"] == "sea_waybill" and row["dangerous_goods_present"]
        ),
        "sea_waybill_and_temperature": lambda row: bool(
            row["document_type"] == "sea_waybill" and row["temperature_present"]
        ),
    }
    return {name: sum(predicate(row) for row in features) for name, predicate in predicates.items()}


def _summary(
    *,
    config: DatasetEdaConfig,
    manifest: dict[str, Any],
    features: Sequence[dict[str, Any]],
    templates: Sequence[dict[str, Any]],
    sensitivity: Sequence[dict[str, Any]],
    validation: dict[str, Any],
    numeric: dict[str, Any],
    categorical: dict[str, Any],
    coverage: Sequence[dict[str, Any]],
    cohort: dict[str, Any],
    augmentation: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    total = len(features)
    missing_carrier = sum(row["carrier_name"] == "<MISSING>" for row in features)
    singletons = sum(cast(bool, row["template_proxy_singleton"]) for row in features)
    correction_documents = sum(cast(int, row["policy_correction_count"]) > 0 for row in features)
    counts = {
        "billOfLading": sum(row["document_type"] == "bill_of_lading" for row in features),
        "seaWaybill": sum(row["document_type"] == "sea_waybill" for row in features),
        "multiPage": sum(cast(bool, row["multi_page"]) for row in features),
        "multiContainer": sum(cast(bool, row["multi_container"]) for row in features),
        "multiGoods": sum(cast(bool, row["multi_goods"]) for row in features),
        "multiPackageLevel": sum(cast(bool, row["multi_package_level"]) for row in features),
        "temperatureControlled": sum(cast(bool, row["temperature_present"]) for row in features),
        "dangerousGoods": sum(cast(bool, row["dangerous_goods_present"]) for row in features),
        "deliveryAgent": sum(cast(bool, row["delivery_agent_present"]) for row in features),
        "forwardingAgent": sum(cast(bool, row["forwarding_agent_present"]) for row in features),
        "consolidator": sum(cast(bool, row["consolidator_present"]) for row in features),
        "hasContainers": sum(cast(int, row["container_count"]) > 0 for row in features),
        "hasAllocations": sum(cast(bool, row["has_allocations"]) for row in features),
        "hasHsCodes": sum(cast(int, row["hs_code_count"]) > 0 for row in features),
        "containerVerifiedGrossMass": sum(
            cast(int, row["container_vgm_count"]) > 0 for row in features
        ),
        "missingCarrier": missing_carrier,
        "policyCorrectedDocuments": correction_documents,
    }
    template_sizes = [cast(int, row["document_count"]) for row in templates]
    return {
        "analysisId": config.analysis_id,
        "analysisContractVersion": 1,
        "dataset": {
            "root": config.dataset.root,
            "manifestSha256": config.dataset.manifest_sha256,
            "datasetId": manifest.get("datasetId"),
            "documents": total,
        },
        "truthBoundary": (
            "Training labels are conditioned on page-ordered raw OCR. PDFs and page "
            "rasters are provenance and layout-analysis inputs, not sources for adding "
            "image-only target values."
        ),
        "validation": validation,
        "documentCounts": counts,
        "documentShares": {key: round(value / total, 8) for key, value in counts.items()},
        "numericDistributions": numeric,
        "templateProxy": {
            "definition": (
                "Greedy conservative clustering by first-page raster ink layout and OCR "
                "heading order, constrained within carrier family and document type."
            ),
            "clusters": len(templates),
            "singletonDocuments": singletons,
            "singletonShare": round(singletons / total, 8),
            "largestClusterDocuments": max(template_sizes),
            "documentsInNonSingletonClusters": total - singletons,
            "sensitivity": list(sensitivity),
        },
        "categoricalConcentration": {
            key: value["concentration"] for key, value in categorical.items()
        },
        "fieldCoverage": list(coverage),
        "jointFeatureCounts": _joint_feature_counts(features),
        "sourceCohortComparison": cohort,
        "augmentationPriorities": list(augmentation),
    }


def _coverage_lookup(
    coverage: Sequence[dict[str, Any]], field: str, cohort: str = "all"
) -> tuple[int, float]:
    matches = [row for row in coverage if row["field"] == field and row["cohort"] == cohort]
    if len(matches) != 1:
        raise DatasetEdaError(f"coverage row is absent or duplicated: {field}/{cohort}")
    return cast(int, matches[0]["present_documents"]), cast(float, matches[0]["coverage"])


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _report_markdown(
    *,
    config: DatasetEdaConfig,
    features: Sequence[dict[str, Any]],
    templates: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    coverage: Sequence[dict[str, Any]],
    cohort: dict[str, Any],
    augmentation: Sequence[dict[str, Any]],
    categorical: dict[str, Any],
) -> str:
    total = len(features)
    counts = cast(dict[str, int], summary["documentCounts"])
    pages = cast(dict[str, Any], summary["numericDistributions"])["page_count"]
    containers = cast(dict[str, Any], summary["numericDistributions"])["container_count"]
    goods = cast(dict[str, Any], summary["numericDistributions"])["goods_group_count"]
    packages = cast(dict[str, Any], summary["numericDistributions"])["package_fact_count"]
    hs_codes = cast(dict[str, Any], summary["numericDistributions"])["hs_code_count"]
    template = cast(dict[str, Any], summary["templateProxy"])
    carrier_rows = categorical["carrier_family"]["distribution"]
    export_rows = categorical["export_country_proxy"]["distribution"]
    top_carriers = [row for row in carrier_rows if row["value"] != "<MISSING>"][:10]
    top_exports = [row for row in export_rows if row["value"] != "<MISSING>"][:10]
    discharge_rows = categorical["port_of_discharge_country"]["distribution"]
    egypt_discharge = next(
        (cast(int, row["count"]) for row in discharge_rows if row["value"] == "EGYPT"),
        0,
    )
    known_discharge = sum(
        cast(int, row["count"]) for row in discharge_rows if row["value"] != "<MISSING>"
    )
    egypt_known_discharge_share = egypt_discharge / known_discharge if known_discharge else 0.0
    allocation_rows = categorical["allocation_coverage"]["distribution"]
    container_vocabulary = cast(
        int,
        categorical["container_type_printed_normalized"]["concentration"]["categories"],
    )
    package_vocabulary = cast(
        int,
        categorical["package_type_printed_normalized"]["concentration"]["categories"],
    )
    coverage_order = sorted(
        (row for row in coverage if row["cohort"] == "all"),
        key=lambda row: cast(float, row["coverage"]),
    )
    numeric_drift = sorted(
        cast(list[dict[str, Any]], cohort["numeric"]),
        key=lambda row: abs(cast(float, row["standardized_mean_difference"])),
        reverse=True,
    )[:8]
    categorical_drift = sorted(
        cast(list[dict[str, Any]], cohort["categorical"]),
        key=lambda row: cast(float, row["jensen_shannon_divergence_bits"]),
        reverse=True,
    )[:8]
    page_tail = sum(cast(int, row["page_count"]) >= 5 for row in features)
    one_goods = sum(cast(int, row["goods_group_count"]) == 1 for row in features)
    no_container = sum(cast(int, row["container_count"]) == 0 for row in features)
    one_container = sum(cast(int, row["container_count"]) == 1 for row in features)
    correction_documents = counts["policyCorrectedDocuments"]
    source_folder_mismatches = sum(
        (row["source_class"] == "blc" and row["document_type"] == "sea_waybill")
        or (row["source_class"] == "swb" and row["document_type"] == "bill_of_lading")
        for row in features
    )
    validation = cast(dict[str, Any], summary["validation"])
    template_sensitivity = cast(list[dict[str, Any]], template["sensitivity"])
    lowest_templates = min(cast(int, row["template_proxy_count"]) for row in template_sensitivity)
    highest_templates = max(cast(int, row["template_proxy_count"]) for row in template_sensitivity)
    lowest_singletons = min(
        cast(int, row["singleton_template_proxy_count"]) for row in template_sensitivity
    )
    highest_singletons = max(
        cast(int, row["singleton_template_proxy_count"]) for row in template_sensitivity
    )
    dangerous = counts["dangerousGoods"]
    temperature = counts["temperatureControlled"]
    container_vgm = counts["containerVerifiedGrossMass"]
    forwarder = counts["forwardingAgent"]
    multi_goods = counts["multiGoods"]
    joint = cast(dict[str, int], summary["jointFeatureCounts"])
    port_discharge = _coverage_lookup(coverage, "port_of_discharge_country")
    goods_origin = _coverage_lookup(coverage, "goods_origin")
    exporter = _coverage_lookup(coverage, "shipper_country_export_proxy")
    unsupported_rows = [row for row in coverage_order if cast(int, row["present_documents"]) <= 18]

    augmentation_rows = [
        (
            row["scenario"],
            row["observed_documents"],
            f"{cast(float, row['observed_share']):.1%}",
            row["minimum_additional_examples"],
        )
        for row in augmentation
    ]
    carrier_table = [
        (row["value"], row["count"], f"{cast(float, row['share']):.1%}") for row in top_carriers
    ]
    export_table = [
        (row["value"], row["count"], f"{cast(float, row['share']):.1%}") for row in top_exports
    ]
    coverage_table = [
        (
            row["field"],
            row["present_documents"],
            f"{cast(float, row['coverage']):.1%}",
        )
        for row in coverage_order
    ]
    drift_table = [
        (
            row["field"],
            row["legacy_mean"],
            row["current_mean"],
            row["standardized_mean_difference"],
            row["ks_statistic"],
        )
        for row in numeric_drift
    ]
    categorical_drift_table = [
        (
            row["field"],
            row["legacy_categories"],
            row["current_categories"],
            row["jensen_shannon_divergence_bits"],
        )
        for row in categorical_drift
    ]
    unsupported_table = [
        (
            row["field"],
            row["present_documents"],
            f"{cast(float, row['coverage']):.2%}",
        )
        for row in unsupported_rows
    ]
    allocation_table = [
        (row["value"], row["count"], f"{cast(float, row['share']):.1%}") for row in allocation_rows
    ]
    return f"""# MPCI Bill-of-Lading dataset EDA

Analysis ID: `{config.analysis_id}`

Pinned source: `{config.dataset.root}`

Documents: **{total:,}**

## Executive diagnosis

This corpus is provenance-complete and structurally valid, but it is not uniformly balanced.
It is strongest on ordinary one-goods, one-container maritime documents and common carrier
forms. It is weakest on rare operational regimes and relation-heavy cargo: only
**{dangerous:,} ({dangerous / total:.1%})** documents carry dangerous-goods facts,
**{temperature:,} ({temperature / total:.1%})** contain temperature settings,
**{forwarder:,} ({forwarder / total:.1%})** contain a forwarding agent, and
**{multi_goods:,} ({multi_goods / total:.1%})** contain multiple goods groups.
Those are not interchangeable gaps: augmentation must preserve their cargo-container,
package-level, party-role, and OCR-layout relationships.

The dataset is also geographically and operationally skewed. Shipper country is available
as an export proxy in **{exporter[0]:,} ({exporter[1]:.1%})** documents, but this is not a
guaranteed country of export. Explicit goods origin appears in only **{goods_origin[0]:,}
({goods_origin[1]:.1%})**, while discharge-country labels occur in **{port_discharge[0]:,}
({port_discharge[1]:.1%})**. Synthetic generation should therefore keep these semantics
separate rather than filling sparse origin from shipper or route.

## Data and quality contract

- Every one of the **{validation["recordsSchemaAndRelationValid"]:,}** records validates
  against both the normal and relation-explicit Pydantic targets, and their two cargo views
  agree after deterministic projection.
- All **{validation["pageSequencesValid"]:,}** joined OCR samples have contiguous page order;
  **{validation["allPageRastersPresent"]:,}** page rasters and all source PDFs are present.
- All **{validation["currentAnnotationsEvidenceValid"]:,}** current-policy annotations pass
  exact raw-OCR evidence validation. Legacy labels retain their validated source annotation
  and immutable alignment lineage; **{correction_documents:,}** documents carry explicit
  policy-correction lineage.
- There are **{validation["uniqueSourcePdfHashes"]:,}** distinct source-PDF hashes and no
  duplicate document IDs or joined-raw-OCR hashes.
- Truth boundary: labels are conditioned on page-ordered raw OCR. The PDF/raster can resolve
  layout relationships during annotation, but image-only values are not added to targets.

## Composition and complexity

- **{counts["billOfLading"]:,} ({counts["billOfLading"] / total:.1%})** bills of lading and
  **{counts["seaWaybill"]:,} ({counts["seaWaybill"] / total:.1%})** sea waybills.
- Pages: median **{pages["median"]:g}**, mean **{pages["mean"]:g}**, maximum **{pages["max"]:g}**;
  **{page_tail:,} ({page_tail / total:.1%})** have five or more pages.
- Containers: median **{containers["median"]:g}**, mean **{containers["mean"]:g}**, maximum
  **{containers["max"]:g}**. **{no_container:,} ({no_container / total:.1%})** have no
  OCR-grounded container row and **{one_container:,} ({one_container / total:.1%})** have one.
- Goods groups: median **{goods["median"]:g}**, maximum **{goods["max"]:g}**;
  **{one_goods:,} ({one_goods / total:.1%})** contain exactly one goods group.
- Package facts: median **{packages["median"]:g}**, maximum **{packages["max"]:g}**. Multiple
  package levels are retained as relation metadata, while goods descriptions remain goods
  descriptions rather than absorbing package wording.
- HS codes: median **{hs_codes["median"]:g}**, maximum **{hs_codes["max"]:g}**. No MPCI UI
  maximum is imposed on the training target; pruning is downstream business logic.
- **{source_folder_mismatches:,}** documents have a validated B/L/SWB type different from their
  BLC/SWB source-folder name. Use the annotation type, not the acquisition folder, for splitting
  or task conditioning.

Key plots: [pages](plots/03_page_count.png),
[containers](plots/07_container_count.png), [goods](plots/08_goods_group_count.png),
[packages](plots/09_package_fact_count.png), and
[cargo/package relations](plots/34_goods_by_package_facts.png).

## Carrier and template surface

Carrier is absent from the OCR-conditioned target in **{counts["missingCarrier"]:,}
({counts["missingCarrier"] / total:.1%})** documents. This is retained as real missingness,
not repaired from logos or visual knowledge.

{_markdown_table(("carrier family", "documents", "share"), carrier_table)}

The selected conservative template proxy yields **{template["clusters"]:,}** clusters,
including **{template["singletonDocuments"]:,} ({template["singletonShare"]:.1%})** singleton
documents; the largest contains **{template["largestClusterDocuments"]:,}** documents. Across
configured strict-to-broad sensitivity settings, the result ranges from
**{lowest_templates:,} to {highest_templates:,}** clusters and **{lowest_singletons:,} to
{highest_singletons:,}** singleton documents. The absolute count is therefore not a ground-
truth template inventory. The stable conclusion is that the corpus combines several repeated
forms with a very long tail of visually/OCR-structurally distinct documents.

For generalization tests, split by template proxy first, then stratify by carrier and rare
features. A random document split will leak repeated layouts and overstate deployment quality.

Key plots: [carrier concentration](plots/13_carrier_pareto.png),
[template sizes](plots/15_template_cluster_sizes.png),
[threshold sensitivity](plots/17_template_sensitivity.png), and
[carrier/template diversity](plots/38_carrier_template_diversity.png). Cross-sections show
[party roles](plots/42_party_roles_by_carrier.png),
[cargo complexity](plots/43_cargo_complexity_by_carrier.png), and
[template repetition](plots/44_template_repetition_by_carrier.png) within the largest known
carrier families.

## Geography

The table below is explicitly a **shipper-country export proxy**, not an asserted customs
country of export.

{_markdown_table(("shipper-country proxy", "documents", "share"), export_table)}

Compare [shipper-country proxy](plots/18_export_country_proxy.png),
[loading countries](plots/19_port_of_loading_country.png),
[discharge countries](plots/20_port_of_discharge_country.png), and
[explicit goods origins](plots/21_goods_origin.png). Missingness and the concentration of
particular discharge markets should be deliberately counterbalanced, but synthetic values
must remain internally consistent with ports, addresses, routes, and OCR text.

The discharge surface is especially narrow: **{egypt_discharge:,} of {known_discharge:,}
({egypt_known_discharge_share:.1%})** documents with a known discharge country say Egypt.
This is a deployment-domain strength if Egypt is the primary target, but a major generalization
weakness for other discharge markets. The
[source-folder/type matrix](plots/47_source_folder_by_document_type.png) separately confirms
that folder labels cannot substitute for validated document types.

## Field coverage

{_markdown_table(("field", "documents", "coverage"), coverage_table)}

The machine-readable [field coverage table](data/field-coverage.csv) includes separate legacy
and current-policy columns. The heatmap makes policy/source-mix differences visible:
[field coverage by cohort](plots/31_field_coverage_by_cohort.png).

## Rare features and relations

- Dangerous goods and temperature co-occur in **{joint["temperature_and_dangerous_goods"]:,}**
  documents; temperature and multiple containers co-occur in
  **{joint["temperature_and_multi_container"]:,}**.
- Multiple goods and multiple containers co-occur in
  **{joint["multi_goods_and_multi_container"]:,}** documents; multiple goods and multiple
  package levels co-occur in **{joint["multi_goods_and_multi_package_level"]:,}**.
- Delivery and forwarding agents co-occur in only
  **{joint["delivery_and_forwarding_agents"]:,}** documents.

These joint counts matter more than marginal totals for relation learning. Naively generating
more common one-goods documents would enlarge the corpus without improving the current weak
surface. Cross-sections: [document type](plots/41_special_features_by_document_type.png) and
[source cohort](plots/45_special_features_by_cohort.png).

Observed allocation modes show useful relation diversity but a sharp tail:

{_markdown_table(("allocation mode", "facts", "share"), allocation_table)}

Only the explicit relation target uses this ontology; it remains deterministic metadata for
training transforms rather than text copied into goods descriptions.

{
        _markdown_table(
            ("augmentation scenario", "observed", "share", "minimum additional"),
            augmentation_rows,
        )
    }

Configured floors are planning heuristics: 100 documents for selected marginal features and
50 for selected joint features. They are not sufficient-sample guarantees. Final synthesis
counts should be driven by held-out learning curves and per-field confidence intervals.

## Legacy/current cohort alignment

The two cohorts are schema-aligned, but distribution equality is neither assumed nor observed.
Largest standardized numeric differences:

{_markdown_table(("field", "legacy mean", "current mean", "SMD", "KS"), drift_table)}

Largest categorical Jensen-Shannon divergences:

{
        _markdown_table(
            ("field", "legacy categories", "current categories", "JS bits"),
            categorical_drift_table,
        )
    }

This is a source-mix/policy warning, not proof of label-quality difference. Training and eval
should monitor cohort-specific metrics so improvement on the larger cohort cannot hide
regression on the other.

## Strengths

1. Strong provenance: source PDF, page rasters, raw-OCR hash, annotation hash, and dataset
   lineage are all auditable.
2. Large ordinary maritime core: common one-goods and one-container patterns are well
   represented across B/L and sea-waybill documents.
3. Real carrier/form variety: repeated common forms coexist with a substantial long tail,
   useful for testing template generalization.
4. Relation-explicit cargo labels: container membership and outer/inner package facts are
   available without imposing MPCI UI truncation rules on model learning.

## Weaknesses and augmentation design

1. **Rare operational regimes:** temperature, dangerous goods, forwarding agents, multi-goods,
   and their intersections are too sparse for reliable fine-grained evaluation. Generate them
   conditionally and measure per-field learning curves.
2. **Unsupported or near-zero schema surface:** container VGM has **{container_vgm:,}**
   OCR-grounded examples, and other schema fields are similarly absent or nearly absent:

{_markdown_table(("field", "documents", "coverage"), unsupported_table)}

   Decide whether each field belongs in the deployed task. For retained fields, acquire and
   label real seed documents before synthesis; unconstrained zero-shot synthetic labels would
   be unauditable.
3. **Template leakage risk:** reserve entire template proxies for evaluation. Add synthetic
   layout/OCR variants to training only; never clone a held-out template into training.
4. **Geographic skew:** diversify shipper/load/discharge/origin combinations while maintaining
   plausible route and address consistency. Do not treat shipper country as explicit origin.
5. **Cargo hierarchy skew:** over {one_goods / total:.1%} of documents have one goods group.
   Synthesis should emphasize several goods rows, several package levels, split allocations,
   container-only membership, and one-to-one package allocation modes.
6. **Vocabulary tails:** printed package/container wording is diverse. Preserve surface forms
   and use downstream canonicalization; do not replace training targets with opaque MPCI codes.
   Even after conservative spelling normalization, this corpus contains
   **{container_vocabulary:,}** container-type values and **{package_vocabulary:,}** package-type
   values, so augmentation must distinguish true semantic variety from OCR/format aliases.
7. **OCR realism:** augment page order, line wrapping, table flattening, repeated fields, and
   bounded OCR corruption together with labels grounded in the resulting raw text. Never add
   a target fact that the synthetic OCR input does not contain.
8. **Missingness realism:** preserve true null patterns (especially missing carrier, country,
   allocation, and origin). Fully populated synthetic documents would teach hallucination.

## Recommended experimental sequence

1. Freeze a group-aware evaluation set by conservative template proxy, with explicit rare-
   feature coverage and separate legacy/current reporting.
2. Train the 270M model on real data and record per-field and relation-specific baselines.
3. Add targeted synthetic blocks one at a time: cargo hierarchy, dangerous goods, reefer,
   party roles, geography, then cross-feature combinations.
4. Run ablations against an equal-sized random oversampling baseline. Accept augmentation only
   when unseen-template and rare-field metrics improve without degrading common fields.
5. Audit generated raw-text/label pairs with the same Pydantic, evidence, page-order, and
   relation-consistency gates used by the real dataset.

## Artifact guide

- `data/document-features.parquet` / `.jsonl` / `.csv`: one row per document.
- `data/template-groups.jsonl`: proxy membership and similarity diagnostics.
- `data/carrier-groups.jsonl`: carrier size, form diversity, and country mix.
- `data/field-coverage.csv`: overall and cohort-specific target presence.
- `data/cohort-comparison.json`: KS, SMD, and categorical divergence diagnostics.
- `data/correlations.csv`, `data/outliers.jsonl`, `data/rare-combinations.csv`.
- `data/augmentation-priorities.csv`: observed counts and configured minimum gaps.
- `summary.json`: compact machine-readable findings.
- `plots/`: deterministic PNG figures covering composition, coverage, drift, relations, and
  augmentation gaps.

All percentages and tables in this report are generated from the pinned dataset; no source
label or training record was modified by the EDA.
"""


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    return canonical_json_bytes(value) + b"\n"


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _csv_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    if not rows:
        raise DatasetEdaError("cannot publish an empty CSV")
    fieldnames = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        encoded: dict[str, Any] = {}
        for key in fieldnames:
            value = row.get(key)
            encoded[key] = (
                canonical_json_bytes(value).decode("utf-8")
                if isinstance(value, list | dict)
                else value
            )
        writer.writerow(encoded)
    return stream.getvalue().encode("utf-8")


def _parquet_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    if not rows:
        raise DatasetEdaError("cannot publish an empty Parquet table")
    table = pa.Table.from_pylist(list(rows))
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    return cast(bytes, sink.getvalue().to_pybytes())


def _publish_artifact(root: Path, relative: str, payload: bytes) -> dict[str, Any]:
    path = root / relative
    atomic_publish_bytes(path, payload)
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def analyze_dataset(config_path: Path) -> Path:
    config_file = _regular_file(config_path, context="dataset EDA config")
    config_payload = read_regular_file_bytes(config_file)
    config = load_dataset_eda_config(config_file)
    _, source_manifest, records, lineage, current_root = _load_inputs(config)
    features, validation = _build_features(config, records, lineage, current_root)
    templates, sensitivity = _apply_template_analysis(features, config.analysis)
    features.sort(key=lambda row: cast(str, row["document_id"]))
    public_features = [
        {key: value for key, value in row.items() if not key.startswith("_")} for row in features
    ]
    numeric = _numeric_summaries(features)
    coverage = _coverage_rows(features)
    correlations = _correlation_rows(features)
    cohort = _cohort_comparison(features)
    augmentation = _augmentation_priorities(features, config.analysis)
    rare = _rare_combinations(features)
    outliers = _outlier_rows(features)
    categorical = _categorical_analysis(features, templates)
    carriers = _carrier_group_rows(features)
    summary = _summary(
        config=config,
        manifest=source_manifest,
        features=features,
        templates=templates,
        sensitivity=sensitivity,
        validation=validation,
        numeric=numeric,
        categorical=categorical,
        coverage=coverage,
        cohort=cohort,
        augmentation=augmentation,
    )
    report = _report_markdown(
        config=config,
        features=features,
        templates=templates,
        summary=summary,
        coverage=coverage,
        cohort=cohort,
        augmentation=augmentation,
        categorical=categorical,
    )
    plots = _plot_artifacts(
        features=features,
        templates=templates,
        sensitivity=sensitivity,
        coverage=coverage,
        correlations=correlations,
        cohort=cohort,
        augmentation=augmentation,
        categorical=categorical,
        settings=config.analysis,
    )

    cohort_rows = [
        {"metric_type": "numeric", **row} for row in cast(list[dict[str, Any]], cohort["numeric"])
    ] + [
        {"metric_type": "categorical", **row}
        for row in cast(list[dict[str, Any]], cohort["categorical"])
    ]
    categorical_rows = [
        {"distribution": name, **row}
        for name, section in categorical.items()
        for row in cast(list[dict[str, Any]], section["distribution"])
    ]
    files: list[tuple[str, bytes]] = [
        ("config.yaml", config_payload),
        ("summary.json", _json_bytes(summary)),
        ("EDA_REPORT.md", report.encode("utf-8")),
        ("data/document-features.jsonl", _jsonl_bytes(public_features)),
        ("data/document-features.csv", _csv_bytes(public_features)),
        ("data/document-features.parquet", _parquet_bytes(public_features)),
        ("data/template-groups.jsonl", _jsonl_bytes(templates)),
        ("data/template-groups.csv", _csv_bytes(templates)),
        ("data/template-sensitivity.csv", _csv_bytes(sensitivity)),
        ("data/carrier-groups.jsonl", _jsonl_bytes(carriers)),
        ("data/carrier-groups.csv", _csv_bytes(carriers)),
        ("data/field-coverage.csv", _csv_bytes(coverage)),
        ("data/numeric-summaries.json", _json_bytes(numeric)),
        ("data/categorical-distributions.json", _json_bytes(categorical)),
        ("data/categorical-distributions.csv", _csv_bytes(categorical_rows)),
        ("data/cohort-comparison.json", _json_bytes(cohort)),
        ("data/cohort-comparison.csv", _csv_bytes(cohort_rows)),
        ("data/correlations.csv", _csv_bytes(correlations)),
        ("data/outliers.jsonl", _jsonl_bytes(outliers)),
        ("data/outliers.csv", _csv_bytes(outliers)),
        ("data/rare-combinations.csv", _csv_bytes(rare)),
        ("data/augmentation-priorities.csv", _csv_bytes(augmentation)),
        *plots,
    ]
    output_root = Path(config.output.root).resolve(strict=False)
    if output_root == Path(output_root.anchor):
        raise DatasetEdaError("refusing to publish EDA at the filesystem root")
    output_root.mkdir(parents=True, exist_ok=True)
    entries = [
        _publish_artifact(output_root, relative, payload) for relative, payload in sorted(files)
    ]
    manifest = {
        "schemaVersion": 1,
        "analysisId": config.analysis_id,
        "analysisContract": "mpci_bl_provenance_aware_dataset_eda_v1",
        "sourceDatasetManifestSha256": config.dataset.manifest_sha256,
        "configSha256": sha256_bytes(config_payload),
        "documents": len(features),
        "plots": len(plots),
        "files": entries,
    }
    atomic_publish_bytes(output_root / "manifest.json", _json_bytes(manifest))
    return output_root / "manifest.json"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def _fatal(error: BaseException) -> NoReturn:
    print(
        json.dumps(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1) from error


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    try:
        manifest = analyze_dataset(args.config)
    except (DatasetEdaError, OSError, ValueError) as error:
        _fatal(error)
    print(
        json.dumps(
            {
                "status": "complete",
                "command": "dataset-eda",
                "manifest": str(manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
