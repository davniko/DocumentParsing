"""Compile and sample coherent dangerous-goods tuples from pinned public data.

PHMSA's Hazardous Materials Table is the regulatory tuple source. ECICS is a
strict, exact-CUS bridge for the subset of chemical records whose official
chemical name exactly matches the PHMSA proper shipping name after mechanical
normalization. The two sources are never fuzzy joined, and HS is never inferred
from a UN number alone.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, cast
from urllib.parse import urlsplit
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import atomic_publish_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v3 import HazardCategory
from document_ocr.label_schemas.bill_of_lading_v4 import (
    PackingGroupCategoryV4,
    RelationExplicitDangerousGoodsV4,
)
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
UnNumber = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")]
Hs6 = Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_PHMSA_URL: Final = "https://phmsaservices.phmsa.dot.gov/phmsas3documentswspublicaccess/download"
_ECICS_URL: Final = "https://ec.europa.eu/taxation_customs/dds2/ecics/chemicalsubstance_list.jsp"
_ECICS_EXPORT_URL: Final = (
    "https://ec.europa.eu/taxation_customs/dds2/ecics/"
    "ecics_export_management.jsp?message=extractFull"
)
_HMT_HEADERS: Final = (
    "Symbols",
    "Proper Shipping Names (PSN)",
    "Hazard Class",
    "UN ID Number",
    "Packaging Group",
    "Label Codes",
    "Special Provisions",
    "Packaging Exceptions",
    "Non-bulk Packaging",
    "Bulk Packaging",
    "Passenger Limitations",
    "Aircraft Cargo Limitations",
    "Vessel Stowage - Location",
    "Vessel Stowage - Other",
)
_ECICS_XLSX_HEADERS: Final = (
    "CUS NUMBER",
    "CAS REGISTRY NUMBERS",
    "CN CODE",
    "IUPAC DESCRIPTION",
    "FIRST NOMENCLATURE",
    "FIRST NOMENCLATURE DESCRIPTION",
    "SYNONYMS",
)
_HAZARD_TOKEN = re.compile(r"1\.[1-6][A-Z]|(?:2|4|5|6)\.[1-3]|[1-9]")
_UN = re.compile(r"^UN([0-9]{4})$")
_SPACE = re.compile(r"\s+")
_MAX_HMT_BYTES = 16 * 1024 * 1024
_MAX_ECICS_ARCHIVE_BYTES = 16 * 1024 * 1024
_MAX_ECICS_PAGE_BYTES = 512 * 1024
_MAX_COMPILED_REGISTRY_BYTES = 64 * 1024 * 1024

_BROAD_HAZARD: dict[str, HazardCategory] = {
    "1": "EXPLOSIVES",
    "2": "GASES",
    "3": "FLAMMABLE_LIQUIDS",
    "4": "FLAMMABLE_SOLIDS",
    "5": "OXIDIZING_SUBSTANCES_AND_ORGANIC_PEROXIDES",
    "6": "TOXIC_AND_INFECTIOUS_SUBSTANCES",
    "7": "RADIOACTIVE_MATERIAL",
    "8": "CORROSIVE_SUBSTANCES",
    "9": "MISCELLANEOUS_DANGEROUS_SUBSTANCES_AND_ARTICLES",
}
_PACKING_GROUP: dict[str, PackingGroupCategoryV4] = {
    "I": "HIGH_DANGER",
    "II": "MEDIUM_DANGER",
    "III": "LOW_DANGER",
}


class DangerousGoodsRegistryError(RuntimeError):
    """A DG source, compiled registry, or sampling request violates its contract."""


class RegistryFilePin(BaseModel):
    model_config = _STRICT

    relative_path: NonEmptyText
    source_url: NonEmptyText
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256

    @model_validator(mode="after")
    def path_and_url_are_safe(self) -> RegistryFilePin:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("DG source path must be a safe relative POSIX path")
        parsed = urlsplit(self.source_url)
        if parsed.scheme != "https" or parsed.fragment:
            raise ValueError("DG source URL must be unfragmented HTTPS")
        return self


class EcicsPagePin(RegistryFilePin):
    offset: Annotated[int, Field(ge=0, multiple_of=25)]
    rows: Annotated[int, Field(gt=0, le=25)]


class DangerousGoodsSourceManifest(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    snapshot_date: Annotated[str, StringConstraints(pattern=r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}$")]
    phmsa_records_through: Annotated[
        str, StringConstraints(pattern=r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}$")
    ]
    phmsa_hmt: RegistryFilePin
    ecics_archive: RegistryFilePin
    ecics_xlsx_member: Literal["ECICS.xlsx"]
    ecics_xlsx_bytes: Annotated[int, Field(gt=0)]
    ecics_xlsx_sha256: Sha256
    ecics_total_records: Annotated[int, Field(gt=0)]
    ecics_pages: tuple[EcicsPagePin, ...] = Field(min_length=1)
    phmsa_attribution: NonEmptyText
    ecics_attribution: NonEmptyText

    @model_validator(mode="after")
    def page_coverage_is_complete(self) -> DangerousGoodsSourceManifest:
        offsets = tuple(row.offset for row in self.ecics_pages)
        expected = tuple(range(0, self.ecics_total_records, 25))
        if offsets != expected:
            raise ValueError("ECICS page offsets do not exactly cover the declared record count")
        if sum(row.rows for row in self.ecics_pages) != self.ecics_total_records:
            raise ValueError("ECICS page row counts do not balance")
        paths = tuple(row.relative_path for row in self.ecics_pages)
        if len(paths) != len(set(paths)):
            raise ValueError("ECICS page paths must be unique")
        return self


class DangerousGoodsHmtRecord(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = 1
    record_id: Annotated[str, StringConstraints(pattern=r"^hmt_[0-9a-f]{64}$")]
    source_row: Annotated[int, Field(ge=2)]
    un_number: UnNumber
    proper_shipping_name: NonEmptyText
    proper_shipping_name_markup: NonEmptyText
    optional_qualifiers: tuple[str, ...]
    exact_hazard_class: NonEmptyText
    hazard_category: HazardCategory
    exact_label_codes: tuple[str, ...]
    exact_subsidiary_hazards: tuple[str, ...]
    subsidiary_hazard_categories: tuple[HazardCategory, ...]
    packing_group_code: Literal["I", "II", "III"] | None = None
    packing_group_category: PackingGroupCategoryV4 | None = None
    symbols: tuple[str, ...]
    technical_name_required: bool
    nos_entry: bool
    vessel_stowage_location: str | None = None
    vessel_stowage_other: tuple[str, ...]
    maritime_eligible: bool
    maritime_disposition: Literal[
        "eligible",
        "missing_vessel_stowage",
        "domestic_only",
        "air_only",
    ]
    source_fields: dict[str, str]

    @model_validator(mode="after")
    def tuple_is_coherent(self) -> DangerousGoodsHmtRecord:
        if self.hazard_category != _broad_hazard(self.exact_hazard_class):
            raise ValueError("HMT semantic primary hazard differs from exact class")
        expected_subsidiaries = tuple(
            _broad_hazard(value) for value in self.exact_subsidiary_hazards
        )
        if self.subsidiary_hazard_categories != expected_subsidiaries:
            raise ValueError("HMT semantic subsidiary hazards differ from exact classes")
        if (self.packing_group_code is None) != (self.packing_group_category is None):
            raise ValueError("HMT packing group code and semantic category must be paired")
        if self.maritime_eligible != (self.maritime_disposition == "eligible"):
            raise ValueError("HMT maritime eligibility and disposition differ")
        return self


EcicsLinkDisposition = Literal[
    "eligible_unique_maritime_hmt",
    "chemical_identity_name_mismatch",
    "cn_shorter_than_hs6",
    "no_hmt_record",
    "no_maritime_hmt_record",
    "ambiguous_maritime_hmt_tuple",
]


class DangerousGoodsEcicsLink(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = 1
    cus_number: NonEmptyText
    un_number: UnNumber
    cn_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{2,8}$")]
    hs6: Hs6 | None = None
    cas_numbers: tuple[str, ...]
    ec_number: str | None = None
    nomenclature: NonEmptyText
    name: NonEmptyText
    iupac_description: str | None = None
    first_nomenclature: str | None = None
    first_nomenclature_description: str | None = None
    synonyms: tuple[str, ...]
    hmt_record_id: str | None = None
    chemical_identity_match: Literal["normalized_exact_official_name_v1"] | None = None
    disposition: EcicsLinkDisposition

    @model_validator(mode="after")
    def eligibility_is_exact(self) -> DangerousGoodsEcicsLink:
        eligible = self.disposition == "eligible_unique_maritime_hmt"
        complete_link = (
            self.hs6 is not None
            and self.hmt_record_id is not None
            and self.chemical_identity_match is not None
        )
        if eligible != complete_link:
            raise ValueError(
                "ECICS eligibility, HS6, HMT link, and identity match must be present together"
            )
        if self.hs6 is not None and self.hs6 != self.cn_code[:6]:
            raise ValueError("ECICS HS6 must be the first six CN digits")
        return self


class DangerousGoodsRegistryAudit(BaseModel):
    model_config = _STRICT

    hmt_source_rows: Annotated[int, Field(gt=0)]
    hmt_valid_un_rows: Annotated[int, Field(gt=0)]
    hmt_records: Annotated[int, Field(gt=0)]
    hmt_distinct_un_numbers: Annotated[int, Field(gt=0)]
    hmt_maritime_records: Annotated[int, Field(gt=0)]
    hmt_maritime_distinct_un_numbers: Annotated[int, Field(gt=0)]
    hmt_multiple_subsidiary_records: Annotated[int, Field(ge=0)]
    ecics_rows: Annotated[int, Field(gt=0)]
    ecics_distinct_un_numbers: Annotated[int, Field(gt=0)]
    ecics_exact_hs_links: Annotated[int, Field(gt=0)]
    ecics_exact_hs_distinct_un_numbers: Annotated[int, Field(gt=0)]
    ecics_exact_hs_distinct_hs6: Annotated[int, Field(gt=0)]
    ecics_dispositions: dict[EcicsLinkDisposition, Annotated[int, Field(gt=0)]]


class DangerousGoodsRegistryReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    parser_contract: Literal["phmsa_hmt_plus_exact_ecics_cus_and_chemical_identity_join_v2"]
    implementation_sha256: Sha256
    source_manifest_sha256: Sha256
    hmt_path: Literal["hmt-records.jsonl"]
    hmt_sha256: Sha256
    ecics_path: Literal["ecics-links.jsonl"]
    ecics_sha256: Sha256
    audit: DangerousGoodsRegistryAudit
    content_sha256: Sha256


@dataclass(frozen=True, slots=True)
class CompiledDangerousGoodsRegistry:
    root: Path
    receipt: DangerousGoodsRegistryReceipt
    commit_receipt: StagedCommitReceipt
    created: bool


@dataclass(frozen=True, slots=True)
class SampledDangerousGoods:
    method: Literal["general_regulatory_tuple", "hs_linked_exact_chemical"]
    target: RelationExplicitDangerousGoodsV4
    hs_codes: tuple[str, ...]
    hmt_record: DangerousGoodsHmtRecord
    ecics_link: DangerousGoodsEcicsLink | None


@dataclass(frozen=True, slots=True)
class LoadedDangerousGoodsRegistry:
    hmt_records: tuple[DangerousGoodsHmtRecord, ...]
    ecics_links: tuple[DangerousGoodsEcicsLink, ...]
    hmt_sha256: str
    ecics_sha256: str
    _candidate_indexes: Mapping[
        tuple[str, int],
        Mapping[
            HazardCategory,
            Mapping[
                str,
                tuple[tuple[DangerousGoodsHmtRecord, DangerousGoodsEcicsLink | None], ...],
            ],
        ],
    ] = field(init=False, repr=False)
    _maximum_subsidiary_hazards: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        hmt_by_id = {row.record_id: row for row in self.hmt_records}
        maximum = max(
            (len(row.exact_subsidiary_hazards) for row in self.hmt_records),
            default=0,
        )
        indexes: dict[
            tuple[str, int],
            Mapping[
                HazardCategory,
                Mapping[
                    str,
                    tuple[tuple[DangerousGoodsHmtRecord, DangerousGoodsEcicsLink | None], ...],
                ],
            ],
        ] = {}
        for threshold in range(maximum + 1):
            general = [
                (row, None)
                for row in self.hmt_records
                if row.maritime_eligible and len(row.exact_subsidiary_hazards) <= threshold
            ]
            exact = [
                (hmt_by_id[cast(str, link.hmt_record_id)], link)
                for link in self.ecics_links
                if link.disposition == "eligible_unique_maritime_hmt"
                and len(hmt_by_id[cast(str, link.hmt_record_id)].exact_subsidiary_hazards)
                <= threshold
            ]
            indexes[("general_regulatory_tuple", threshold)] = _candidate_index(general)
            indexes[("hs_linked_exact_chemical", threshold)] = _candidate_index(exact)
        object.__setattr__(self, "_candidate_indexes", MappingProxyType(indexes))
        object.__setattr__(self, "_maximum_subsidiary_hazards", maximum)

    def sample(
        self,
        *,
        stream: DeterministicStream,
        method: Literal["general_regulatory_tuple", "hs_linked_exact_chemical"],
        category_weights: Mapping[HazardCategory, int] | None = None,
        maximum_subsidiary_hazards: int | None = None,
    ) -> SampledDangerousGoods:
        """Sample category, then UN, then one complete tuple/chemical uniformly."""

        if maximum_subsidiary_hazards is not None and maximum_subsidiary_hazards < 0:
            raise DangerousGoodsRegistryError("maximum subsidiary hazards must be non-negative")
        threshold = (
            self._maximum_subsidiary_hazards
            if maximum_subsidiary_hazards is None
            else min(maximum_subsidiary_hazards, self._maximum_subsidiary_hazards)
        )
        index = self._candidate_indexes[(method, threshold)]
        if not index:
            raise DangerousGoodsRegistryError("DG sampling policy has no eligible records")
        selected_record, selected_link = _hierarchical_sample(
            index,
            stream=stream,
            category_weights=category_weights,
        )
        target = RelationExplicitDangerousGoodsV4.model_validate(
            {
                "unNumber": selected_record.un_number,
                "hazardCategory": selected_record.hazard_category,
                "subsidiaryHazardCategories": (
                    selected_record.subsidiary_hazard_categories or None
                ),
                "packingGroupCategory": selected_record.packing_group_category,
            },
            strict=True,
        )
        hs_codes = () if selected_link is None else (cast(str, selected_link.hs6),)
        return SampledDangerousGoods(
            method=method,
            target=target,
            hs_codes=hs_codes,
            hmt_record=selected_record,
            ecics_link=selected_link,
        )


class _EcicsHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.total_records: int | None = None
        self.in_table = False
        self.current_row_id: str | None = None
        self.current_cell = False
        self.cell_parts: list[str] = []
        self.cells: list[str] = []
        self.rows: list[tuple[str, ...]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "input" and attributes.get("id") == "totalRecords":
            value = attributes.get("value")
            if value is not None and value.isdigit():
                self.total_records = int(value)
        if tag == "table" and attributes.get("id") == "tblData":
            self.in_table = True
        elif self.in_table and tag == "tr":
            classes = (attributes.get("class") or "").split()
            if "tariff_ecics-inner-table" in classes:
                self.current_row_id = attributes.get("id")
                self.cells = []
        elif self.current_row_id is not None and tag == "td":
            self.current_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.current_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.current_cell:
            self.cells.append(_clean_text("".join(self.cell_parts)))
            self.current_cell = False
            self.cell_parts = []
        elif tag == "tr" and self.current_row_id is not None:
            if len(self.cells) != 7:
                raise DangerousGoodsRegistryError("ECICS result row does not contain seven cells")
            self.rows.append(tuple(self.cells))
            self.current_row_id = None
            self.cells = []
        elif tag == "table" and self.in_table:
            self.in_table = False


class _PsnParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.italic_depth = 0
        self.italic_parts: list[str] = []
        self.qualifiers: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "i":
            self.italic_depth += 1
            if self.italic_depth == 1:
                self.italic_parts = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
        if self.italic_depth:
            self.italic_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "i" and self.italic_depth:
            self.italic_depth -= 1
            if self.italic_depth == 0:
                value = _clean_text("".join(self.italic_parts))
                if value:
                    self.qualifiers.append(value)


def _clean_text(value: object) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _broad_hazard(exact_code: str) -> HazardCategory:
    try:
        return _BROAD_HAZARD[exact_code[0]]
    except (IndexError, KeyError) as error:
        raise DangerousGoodsRegistryError(
            f"unsupported dangerous-goods hazard class: {exact_code!r}"
        ) from error


def _parse_ecics_page(payload: bytes) -> tuple[int, tuple[tuple[str, ...], ...]]:
    if len(payload) > _MAX_ECICS_PAGE_BYTES:
        raise DangerousGoodsRegistryError("ECICS HTML page exceeds the size bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DangerousGoodsRegistryError("ECICS HTML page is not UTF-8") from error
    parser = _EcicsHtmlParser()
    parser.feed(text)
    parser.close()
    if parser.total_records is None or not parser.rows:
        raise DangerousGoodsRegistryError("ECICS page lacks total count or result rows")
    return parser.total_records, tuple(parser.rows)


def _safe_source_path(source_root: Path, relative_path: str) -> Path:
    path = source_root / PurePosixPath(relative_path)
    if path.is_symlink() or not path.is_file():
        raise DangerousGoodsRegistryError(f"DG source must be a plain file: {path}")
    resolved_root = source_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise DangerousGoodsRegistryError("DG source escapes its configured root")
    return resolved


def _validate_pin(source_root: Path, pin: RegistryFilePin, *, maximum_bytes: int) -> bytes:
    path = _safe_source_path(source_root, pin.relative_path)
    payload = read_regular_file_bytes(path)
    if len(payload) > maximum_bytes or len(payload) != pin.bytes:
        raise DangerousGoodsRegistryError(f"DG source size differs from pin: {path}")
    if sha256_bytes(payload) != pin.sha256:
        raise DangerousGoodsRegistryError(f"DG source SHA-256 differs from pin: {path}")
    return payload


def build_dangerous_goods_source_manifest(
    *,
    source_root: Path,
    phmsa_relative_path: str,
    ecics_archive_relative_path: str,
    ecics_pages_relative_dir: str,
    snapshot_date: str,
    phmsa_records_through: str,
    output_path: Path,
) -> DangerousGoodsSourceManifest:
    """Inspect acquired official bytes and publish their exact immutable manifest."""

    phmsa_path = _safe_source_path(source_root, phmsa_relative_path)
    archive_path = _safe_source_path(source_root, ecics_archive_relative_path)
    phmsa_payload = read_regular_file_bytes(phmsa_path)
    archive_payload = read_regular_file_bytes(archive_path)
    if len(phmsa_payload) > _MAX_HMT_BYTES or len(archive_payload) > _MAX_ECICS_ARCHIVE_BYTES:
        raise DangerousGoodsRegistryError("DG source exceeds its acquisition size bound")
    with ZipFile(BytesIO(archive_payload)) as archive:
        if archive.namelist() != ["ECICS.xlsx"]:
            raise DangerousGoodsRegistryError("ECICS archive must contain only ECICS.xlsx")
        xlsx = archive.read("ECICS.xlsx")

    pages_root = source_root / PurePosixPath(ecics_pages_relative_dir)
    if pages_root.is_symlink() or not pages_root.is_dir():
        raise DangerousGoodsRegistryError("ECICS pages path must be a plain directory")
    page_pins: list[EcicsPagePin] = []
    declared_total: int | None = None
    for path in sorted(pages_root.glob("*.html")):
        if path.is_symlink() or re.fullmatch(r"[0-9]{6}\.html", path.name) is None:
            raise DangerousGoodsRegistryError(f"unexpected ECICS page file: {path}")
        offset = int(path.stem)
        payload = read_regular_file_bytes(path)
        total, rows = _parse_ecics_page(payload)
        if declared_total is None:
            declared_total = total
        elif total != declared_total:
            raise DangerousGoodsRegistryError("ECICS pages disagree on total record count")
        page_pins.append(
            EcicsPagePin(
                relative_path=(PurePosixPath(ecics_pages_relative_dir) / path.name).as_posix(),
                source_url=f"{_ECICS_URL}?Lang=en&offset={offset}&UnCode=%25",
                bytes=len(payload),
                sha256=sha256_bytes(payload),
                offset=offset,
                rows=len(rows),
            )
        )
    if declared_total is None:
        raise DangerousGoodsRegistryError("no ECICS HTML pages were found")
    manifest = DangerousGoodsSourceManifest(
        schema_version=1,
        snapshot_date=snapshot_date,
        phmsa_records_through=phmsa_records_through,
        phmsa_hmt=RegistryFilePin(
            relative_path=phmsa_relative_path,
            source_url=_PHMSA_URL,
            bytes=len(phmsa_payload),
            sha256=sha256_bytes(phmsa_payload),
        ),
        ecics_archive=RegistryFilePin(
            relative_path=ecics_archive_relative_path,
            source_url=_ECICS_EXPORT_URL,
            bytes=len(archive_payload),
            sha256=sha256_bytes(archive_payload),
        ),
        ecics_xlsx_member="ECICS.xlsx",
        ecics_xlsx_bytes=len(xlsx),
        ecics_xlsx_sha256=sha256_bytes(xlsx),
        ecics_total_records=declared_total,
        ecics_pages=tuple(page_pins),
        phmsa_attribution=(
            "U.S. Department of Transportation PHMSA, 49 CFR 172.101 HMT oCFR export"
        ),
        ecics_attribution=(
            "European Commission DG TAXUD, European Customs Inventory of Chemical Substances"
        ),
    )
    payload = canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    atomic_publish_bytes(output_path, payload)
    return manifest


def _parse_hmt(payload: bytes) -> tuple[tuple[DangerousGoodsHmtRecord, ...], int, int]:
    if len(payload) > _MAX_HMT_BYTES:
        raise DangerousGoodsRegistryError("PHMSA HMT exceeds the size bound")
    try:
        import xlrd  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - exercised by synthesis image
        raise DangerousGoodsRegistryError("xlrd is required to compile the PHMSA HMT") from error
    try:
        workbook = xlrd.open_workbook(file_contents=payload, on_demand=True)
    except Exception as error:
        raise DangerousGoodsRegistryError("PHMSA HMT is not a valid BIFF workbook") from error
    if workbook.sheet_names() != ["Export Worksheet", "SQL"]:
        raise DangerousGoodsRegistryError("PHMSA HMT workbook sheet contract changed")
    sheet = workbook.sheet_by_name("Export Worksheet")
    headers = tuple(_clean_text(sheet.cell_value(0, column)) for column in range(sheet.ncols))
    if headers[:14] != _HMT_HEADERS or any(headers[14:]):
        raise DangerousGoodsRegistryError("PHMSA HMT columns changed")
    records: list[DangerousGoodsHmtRecord] = []
    valid_un_rows = 0
    for source_index in range(1, sheet.nrows):
        values = tuple(_clean_text(sheet.cell_value(source_index, column)) for column in range(14))
        source = dict(zip(_HMT_HEADERS, values, strict=True))
        match = _UN.fullmatch(source["UN ID Number"])
        if match is None:
            continue
        valid_un_rows += 1
        exact_hazard = source["Hazard Class"]
        if not exact_hazard:
            continue
        exact_labels = tuple(_HAZARD_TOKEN.findall(source["Label Codes"]))
        if exact_labels and exact_labels[0] != exact_hazard:
            raise DangerousGoodsRegistryError(
                f"PHMSA primary label differs from hazard class at row {source_index + 1}"
            )
        subsidiaries = exact_labels[1:] if exact_labels else ()
        packing_code = source["Packaging Group"] or None
        if packing_code is not None and packing_code not in _PACKING_GROUP:
            raise DangerousGoodsRegistryError(
                f"unsupported PHMSA packing group at row {source_index + 1}"
            )
        symbols = tuple(re.findall(r"[A-Z]|\+", source["Symbols"]))
        vessel_location = source["Vessel Stowage - Location"] or None
        if "D" in symbols:
            disposition = "domestic_only"
        elif "A" in symbols and "W" not in symbols:
            disposition = "air_only"
        elif vessel_location is None:
            disposition = "missing_vessel_stowage"
        else:
            disposition = "eligible"
        psn_parser = _PsnParser()
        psn_parser.feed(source["Proper Shipping Names (PSN)"])
        psn_parser.close()
        psn = _clean_text("".join(psn_parser.parts))
        if not psn:
            raise DangerousGoodsRegistryError(
                f"PHMSA proper shipping name is empty at row {source_index + 1}"
            )
        record_body = {
            "sourceRow": source_index + 1,
            "unNumber": match.group(1),
            "properShippingNameMarkup": source["Proper Shipping Names (PSN)"],
            "exactHazardClass": exact_hazard,
            "exactLabelCodes": exact_labels,
            "packingGroupCode": packing_code,
        }
        records.append(
            DangerousGoodsHmtRecord(
                record_id=f"hmt_{sha256_bytes(canonical_json_bytes(record_body))}",
                source_row=source_index + 1,
                un_number=match.group(1),
                proper_shipping_name=psn,
                proper_shipping_name_markup=source["Proper Shipping Names (PSN)"],
                optional_qualifiers=tuple(psn_parser.qualifiers),
                exact_hazard_class=exact_hazard,
                hazard_category=_broad_hazard(exact_hazard),
                exact_label_codes=exact_labels,
                exact_subsidiary_hazards=subsidiaries,
                subsidiary_hazard_categories=tuple(_broad_hazard(value) for value in subsidiaries),
                packing_group_code=cast(Any, packing_code),
                packing_group_category=(
                    _PACKING_GROUP[packing_code] if packing_code is not None else None
                ),
                symbols=symbols,
                technical_name_required="G" in symbols,
                nos_entry=bool(re.search(r"\bn\.?o\.?s\.?\b", psn, flags=re.IGNORECASE)),
                vessel_stowage_location=vessel_location,
                vessel_stowage_other=tuple(
                    part.strip()
                    for part in source["Vessel Stowage - Other"].split(",")
                    if part.strip()
                ),
                maritime_eligible=disposition == "eligible",
                maritime_disposition=cast(Any, disposition),
                source_fields=source,
            )
        )
    workbook.release_resources()
    if not records:
        raise DangerousGoodsRegistryError("PHMSA HMT yielded no supported records")
    return tuple(records), sheet.nrows - 1, valid_un_rows


def _ecics_html_rows(
    source_root: Path,
    manifest: DangerousGoodsSourceManifest,
) -> tuple[dict[str, str], ...]:
    output: list[dict[str, str]] = []
    for pin in manifest.ecics_pages:
        payload = _validate_pin(source_root, pin, maximum_bytes=_MAX_ECICS_PAGE_BYTES)
        total, rows = _parse_ecics_page(payload)
        if total != manifest.ecics_total_records or len(rows) != pin.rows:
            raise DangerousGoodsRegistryError("ECICS page contents differ from manifest")
        for cells in rows:
            cus, cn, cas, ec, un, nomenclature, name = cells
            if not cus or not cn.isdigit() or not un.isdigit() or len(un) > 4:
                raise DangerousGoodsRegistryError("ECICS row contains malformed identifiers")
            output.append(
                {
                    "cus": cus,
                    "cn": cn,
                    "cas": cas,
                    "ec": ec,
                    "un": un.zfill(4),
                    "nomenclature": nomenclature,
                    "name": name,
                }
            )
    if len(output) != manifest.ecics_total_records:
        raise DangerousGoodsRegistryError("ECICS compiled rows do not balance")
    cus_values = tuple(row["cus"] for row in output)
    if len(cus_values) != len(set(cus_values)):
        raise DangerousGoodsRegistryError("ECICS HTML CUS values are not unique")
    return tuple(output)


def _xlsx_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _clean_text(value)


def _ecics_enrichment(
    archive_payload: bytes,
    *,
    manifest: DangerousGoodsSourceManifest,
    required_cus: frozenset[str],
) -> dict[str, dict[str, str]]:
    with ZipFile(BytesIO(archive_payload)) as archive:
        if archive.namelist() != [manifest.ecics_xlsx_member]:
            raise DangerousGoodsRegistryError("ECICS ZIP member contract changed")
        xlsx = archive.read(manifest.ecics_xlsx_member)
    if len(xlsx) != manifest.ecics_xlsx_bytes or sha256_bytes(xlsx) != manifest.ecics_xlsx_sha256:
        raise DangerousGoodsRegistryError("ECICS XLSX member differs from its pin")
    try:
        from openpyxl import load_workbook  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - exercised by synthesis image
        raise DangerousGoodsRegistryError("openpyxl is required to compile ECICS") from error
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Workbook contains no default style",
            category=UserWarning,
        )
        workbook = load_workbook(BytesIO(xlsx), read_only=True, data_only=True)
    if workbook.sheetnames != ["DDS2-ECICS Extract"]:
        raise DangerousGoodsRegistryError("ECICS workbook sheet contract changed")
    sheet = workbook["DDS2-ECICS Extract"]
    iterator = sheet.iter_rows(values_only=True)
    headers = tuple(_xlsx_string(value) for value in next(iterator))
    if headers != _ECICS_XLSX_HEADERS:
        raise DangerousGoodsRegistryError("ECICS XLSX columns changed")
    output: dict[str, dict[str, str]] = {}
    seen = 0
    for row in iterator:
        seen += 1
        values = tuple(_xlsx_string(value) for value in row)
        cus = values[0]
        if cus in required_cus:
            if cus in output:
                raise DangerousGoodsRegistryError("ECICS XLSX contains duplicate required CUS")
            output[cus] = dict(zip(_ECICS_XLSX_HEADERS, values, strict=True))
    workbook.close()
    if seen != 106_909:
        raise DangerousGoodsRegistryError(
            f"ECICS XLSX row count changed: expected 106909, found {seen}"
        )
    if set(output) != required_cus:
        missing = sorted(required_cus - set(output))
        raise DangerousGoodsRegistryError(f"ECICS XLSX lacks required CUS rows: {missing[:5]}")
    return output


def _parse_multivalue(value: str) -> tuple[str, ...]:
    return tuple(part for part in (_clean_text(item) for item in re.split(r"[,;|]", value)) if part)


def _normalized_chemical_identity(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.lower()))


def _chemical_identity_matches(
    hmt_record: DangerousGoodsHmtRecord,
    *,
    html_name: str,
    enrichment: Mapping[str, str],
) -> bool:
    expected = _normalized_chemical_identity(hmt_record.proper_shipping_name)
    if not expected:
        raise DangerousGoodsRegistryError("normalized HMT proper shipping name is empty")
    candidates = (
        html_name,
        enrichment["IUPAC DESCRIPTION"],
        enrichment["FIRST NOMENCLATURE DESCRIPTION"],
        *_parse_multivalue(enrichment["SYNONYMS"]),
    )
    return expected in {
        normalized
        for value in candidates
        if value and (normalized := _normalized_chemical_identity(value))
    }


def _compile_ecics_links(
    html_rows: Sequence[Mapping[str, str]],
    enrichment: Mapping[str, Mapping[str, str]],
    hmt_records: Sequence[DangerousGoodsHmtRecord],
) -> tuple[DangerousGoodsEcicsLink, ...]:
    all_hmt_by_un: dict[str, list[DangerousGoodsHmtRecord]] = defaultdict(list)
    maritime_hmt_by_un: dict[str, list[DangerousGoodsHmtRecord]] = defaultdict(list)
    for record in hmt_records:
        all_hmt_by_un[record.un_number].append(record)
        if record.maritime_eligible:
            maritime_hmt_by_un[record.un_number].append(record)
    output: list[DangerousGoodsEcicsLink] = []
    for html in sorted(html_rows, key=lambda row: row["cus"]):
        full = enrichment[html["cus"]]
        full_cas_numbers = _parse_multivalue(full["CAS REGISTRY NUMBERS"])
        if full["CN CODE"] != html["cn"] or (html["cas"] and html["cas"] not in full_cas_numbers):
            raise DangerousGoodsRegistryError("ECICS exact-CUS join has a CN or CAS mismatch")
        maritime = maritime_hmt_by_un.get(html["un"], [])
        if len(html["cn"]) < 6:
            disposition = "cn_shorter_than_hs6"
        elif html["un"] not in all_hmt_by_un:
            disposition = "no_hmt_record"
        elif not maritime:
            disposition = "no_maritime_hmt_record"
        elif len(maritime) != 1:
            disposition = "ambiguous_maritime_hmt_tuple"
        elif not _chemical_identity_matches(
            maritime[0],
            html_name=html["name"],
            enrichment=full,
        ):
            disposition = "chemical_identity_name_mismatch"
        else:
            disposition = "eligible_unique_maritime_hmt"
        eligible = disposition == "eligible_unique_maritime_hmt"
        output.append(
            DangerousGoodsEcicsLink(
                cus_number=html["cus"],
                un_number=html["un"],
                cn_code=html["cn"],
                hs6=html["cn"][:6] if eligible else None,
                cas_numbers=full_cas_numbers,
                ec_number=html["ec"] or None,
                nomenclature=html["nomenclature"],
                name=html["name"],
                iupac_description=full["IUPAC DESCRIPTION"] or None,
                first_nomenclature=full["FIRST NOMENCLATURE"] or None,
                first_nomenclature_description=(full["FIRST NOMENCLATURE DESCRIPTION"] or None),
                synonyms=_parse_multivalue(full["SYNONYMS"]),
                hmt_record_id=maritime[0].record_id if eligible else None,
                chemical_identity_match=("normalized_exact_official_name_v1" if eligible else None),
                disposition=cast(Any, disposition),
            )
        )
    return tuple(output)


def _jsonl_payload(rows: Sequence[BaseModel]) -> bytes:
    return b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows)


def compile_dangerous_goods_registry(
    *,
    source_root: Path,
    source_manifest_path: Path,
    expected_manifest_sha256: str,
    output_parent: Path,
    run_name: str,
) -> CompiledDangerousGoodsRegistry:
    """Compile pinned PHMSA and ECICS bytes into an immutable runtime registry."""

    manifest_bytes = read_regular_file_bytes(source_manifest_path)
    if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        raise DangerousGoodsRegistryError("DG source manifest SHA-256 mismatch")
    try:
        manifest = DangerousGoodsSourceManifest.model_validate_json(manifest_bytes, strict=True)
    except Exception as error:
        raise DangerousGoodsRegistryError("DG source manifest is invalid") from error
    if manifest_bytes != canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n":
        raise DangerousGoodsRegistryError("DG source manifest is not canonical JSON")
    hmt_payload = _validate_pin(source_root, manifest.phmsa_hmt, maximum_bytes=_MAX_HMT_BYTES)
    archive_payload = _validate_pin(
        source_root,
        manifest.ecics_archive,
        maximum_bytes=_MAX_ECICS_ARCHIVE_BYTES,
    )
    hmt_records, hmt_source_rows, hmt_valid_un_rows = _parse_hmt(hmt_payload)
    html_rows = _ecics_html_rows(source_root, manifest)
    enrichment = _ecics_enrichment(
        archive_payload,
        manifest=manifest,
        required_cus=frozenset(row["cus"] for row in html_rows),
    )
    ecics_links = _compile_ecics_links(html_rows, enrichment, hmt_records)
    hmt_bytes = _jsonl_payload(hmt_records)
    ecics_bytes = _jsonl_payload(ecics_links)
    if len(hmt_bytes) + len(ecics_bytes) > _MAX_COMPILED_REGISTRY_BYTES:
        raise DangerousGoodsRegistryError("compiled DG registry exceeds its size bound")
    disposition_counts = Counter(row.disposition for row in ecics_links)
    eligible_links = [
        row for row in ecics_links if row.disposition == "eligible_unique_maritime_hmt"
    ]
    maritime_records = [row for row in hmt_records if row.maritime_eligible]
    audit = DangerousGoodsRegistryAudit(
        hmt_source_rows=hmt_source_rows,
        hmt_valid_un_rows=hmt_valid_un_rows,
        hmt_records=len(hmt_records),
        hmt_distinct_un_numbers=len({row.un_number for row in hmt_records}),
        hmt_maritime_records=len(maritime_records),
        hmt_maritime_distinct_un_numbers=len({row.un_number for row in maritime_records}),
        hmt_multiple_subsidiary_records=sum(
            len(row.exact_subsidiary_hazards) > 1 for row in hmt_records
        ),
        ecics_rows=len(ecics_links),
        ecics_distinct_un_numbers=len({row.un_number for row in ecics_links}),
        ecics_exact_hs_links=len(eligible_links),
        ecics_exact_hs_distinct_un_numbers=len({row.un_number for row in eligible_links}),
        ecics_exact_hs_distinct_hs6=len({row.hs6 for row in eligible_links}),
        ecics_dispositions=cast(Any, dict(sorted(disposition_counts.items()))),
    )
    implementation_sha256 = sha256_file(Path(__file__))
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "sourceManifestSha256": expected_manifest_sha256,
                "parserContract": ("phmsa_hmt_plus_exact_ecics_cus_and_chemical_identity_join_v2"),
                "implementationSha256": implementation_sha256,
                "hmtSha256": sha256_bytes(hmt_bytes),
                "ecicsSha256": sha256_bytes(ecics_bytes),
                "audit": audit.model_dump(mode="json"),
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
        "parser_contract": ("phmsa_hmt_plus_exact_ecics_cus_and_chemical_identity_join_v2"),
        "implementation_sha256": implementation_sha256,
        "source_manifest_sha256": expected_manifest_sha256,
        "hmt_path": "hmt-records.jsonl",
        "hmt_sha256": sha256_bytes(hmt_bytes),
        "ecics_path": "ecics-links.jsonl",
        "ecics_sha256": sha256_bytes(ecics_bytes),
        "audit": audit.model_dump(mode="json"),
    }
    receipt = DangerousGoodsRegistryReceipt.model_validate(
        {
            **receipt_body,
            "content_sha256": sha256_bytes(canonical_json_bytes(receipt_body)),
        },
        strict=True,
    )
    receipt_payload = canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n"
    stage.publish_bytes("hmt-records.jsonl", hmt_bytes)
    stage.publish_bytes("ecics-links.jsonl", ecics_bytes)
    stage.publish_bytes("registry-receipt.json", receipt_payload)
    committed = stage.commit(
        expected_artifacts=(
            "ecics-links.jsonl",
            "hmt-records.jsonl",
            "registry-receipt.json",
        ),
        metadata={
            "schema_version": 1,
            "hmt_records": len(hmt_records),
            "ecics_records": len(ecics_links),
            "exact_hs_links": len(eligible_links),
        },
    )
    return CompiledDangerousGoodsRegistry(
        root=stage.final_root,
        receipt=receipt,
        commit_receipt=committed.receipt,
        created=committed.created,
    )


def _load_jsonl(
    path: Path,
    *,
    expected_sha256: str,
    model: type[BaseModel],
) -> tuple[BaseModel, ...]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise DangerousGoodsRegistryError(f"compiled DG registry pin mismatch: {path}")
    output: list[BaseModel] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise DangerousGoodsRegistryError(
                    f"compiled DG registry has blank row {line_number}: {path}"
                )
            try:
                row = model.model_validate_json(raw, strict=True)
            except Exception as error:
                raise DangerousGoodsRegistryError(
                    f"compiled DG registry row {line_number} is invalid: {path}"
                ) from error
            if raw != canonical_json_bytes(row.model_dump(mode="json")) + b"\n":
                raise DangerousGoodsRegistryError(
                    f"compiled DG registry row {line_number} is not canonical: {path}"
                )
            output.append(row)
    return tuple(output)


def load_dangerous_goods_registry(
    *,
    hmt_path: Path,
    hmt_sha256: str,
    ecics_path: Path,
    ecics_sha256: str,
) -> LoadedDangerousGoodsRegistry:
    hmt = cast(
        tuple[DangerousGoodsHmtRecord, ...],
        _load_jsonl(
            hmt_path,
            expected_sha256=hmt_sha256,
            model=DangerousGoodsHmtRecord,
        ),
    )
    ecics = cast(
        tuple[DangerousGoodsEcicsLink, ...],
        _load_jsonl(
            ecics_path,
            expected_sha256=ecics_sha256,
            model=DangerousGoodsEcicsLink,
        ),
    )
    hmt_ids = {row.record_id for row in hmt}
    linked_ids = {row.hmt_record_id for row in ecics if row.hmt_record_id is not None}
    if not linked_ids <= hmt_ids:
        raise DangerousGoodsRegistryError("ECICS registry references absent HMT records")
    return LoadedDangerousGoodsRegistry(
        hmt_records=hmt,
        ecics_links=ecics,
        hmt_sha256=hmt_sha256,
        ecics_sha256=ecics_sha256,
    )


def _weighted_category(
    categories: Sequence[HazardCategory],
    *,
    weights: Mapping[HazardCategory, int] | None,
    stream: DeterministicStream,
) -> HazardCategory:
    available = tuple(sorted(set(categories)))
    if weights is None:
        return available[stream.randbelow(len(available))]
    if any(value <= 0 for value in weights.values()):
        raise DangerousGoodsRegistryError("DG category weights must be positive integers")
    unknown = set(weights) - set(available)
    if unknown:
        raise DangerousGoodsRegistryError(
            f"DG category weights name unavailable categories: {sorted(unknown)}"
        )
    weighted = tuple((category, weights.get(category, 0)) for category in available)
    total = sum(weight for _, weight in weighted)
    if total <= 0:
        raise DangerousGoodsRegistryError("DG category weights select no available category")
    position = stream.randbelow(total)
    cumulative = 0
    for category, weight in weighted:
        cumulative += weight
        if position < cumulative:
            return category
    raise AssertionError("weighted category selection did not terminate")


def _candidate_index(
    candidates: Sequence[tuple[DangerousGoodsHmtRecord, DangerousGoodsEcicsLink | None]],
) -> Mapping[
    HazardCategory,
    Mapping[
        str,
        tuple[tuple[DangerousGoodsHmtRecord, DangerousGoodsEcicsLink | None], ...],
    ],
]:
    mutable: dict[
        HazardCategory,
        dict[str, list[tuple[DangerousGoodsHmtRecord, DangerousGoodsEcicsLink | None]]],
    ] = defaultdict(lambda: defaultdict(list))
    for record, link in candidates:
        mutable[record.hazard_category][record.un_number].append((record, link))
    return MappingProxyType(
        {
            category: MappingProxyType(
                {
                    un_number: tuple(
                        sorted(
                            rows,
                            key=lambda pair: (
                                pair[0].record_id,
                                pair[1].cus_number if pair[1] is not None else "",
                            ),
                        )
                    )
                    for un_number, rows in sorted(by_un.items())
                }
            )
            for category, by_un in sorted(mutable.items())
        }
    )


def _hierarchical_sample(
    index: Mapping[
        HazardCategory,
        Mapping[
            str,
            tuple[tuple[DangerousGoodsHmtRecord, DangerousGoodsEcicsLink | None], ...],
        ],
    ],
    *,
    stream: DeterministicStream,
    category_weights: Mapping[HazardCategory, int] | None,
) -> tuple[DangerousGoodsHmtRecord, DangerousGoodsEcicsLink | None]:
    category = _weighted_category(
        tuple(index),
        weights=category_weights,
        stream=stream.derive("hazard-category"),
    )
    un_numbers = tuple(index[category])
    un_number = un_numbers[stream.derive("un-number").randbelow(len(un_numbers))]
    rows = index[category][un_number]
    return rows[stream.derive("tuple").randbelow(len(rows))]
