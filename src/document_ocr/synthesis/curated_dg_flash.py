"""Pinned, exact-identity NIOSH flash points for DG extraction synthesis."""

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.dangerous_goods_registry import LoadedDangerousGoodsRegistry


class DgFlashPoint(BaseModel):
    """One scalar closed-cup property joined by CAS and the same DOT/UN identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    hmt_record_id: str = Field(pattern=r"^hmt_[0-9a-f]{64}$")
    cas_number: str = Field(pattern=r"^[0-9]{2,7}-[0-9]{2}-[0-9]$")
    un_number: str = Field(pattern=r"^[0-9]{4}$")
    chemical_name: str = Field(min_length=1)
    value_fahrenheit: Decimal = Field(ge=Decimal("-459.67"))
    source_url: str = Field(pattern=r"^https://www\.cdc\.gov/niosh/npg/npgd[0-9]{4}\.html$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_literal: str = Field(min_length=1)
    method: str = Field(pattern=r"^closed_cup$")

    def value(self, unit: str) -> Decimal:
        if unit == "fahrenheit":
            return self.value_fahrenheit
        if unit == "celsius":
            return (self.value_fahrenheit - 32) * Decimal(5) / 9
        raise ValueError(f"unsupported flash-point scale: {unit}")


class DgFlashPointRegistry(BaseModel):
    """Finite official-property support; unknown or formulation-specific values are absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = Field(ge=1, le=1)
    source_index_url: str
    source_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hmt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ecics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[DgFlashPoint, ...] = Field(min_length=1)

    def identity_index(self, dg: LoadedDangerousGoodsRegistry) -> Mapping[str, DgFlashPoint]:
        if (self.hmt_sha256, self.ecics_sha256) != (dg.hmt_sha256, dg.ecics_sha256):
            raise ValueError("flash-point registry references a different HMT/ECICS authority")
        hmt = {r.record_id: r for r in dg.hmt_records}
        links = {
            (r.hmt_record_id, r.un_number, cas)
            for r in dg.ecics_links
            if r.disposition == "eligible_unique_maritime_hmt" and len(r.cas_numbers) == 1
            for cas in r.cas_numbers
        }
        result = {}
        for row in self.records:
            if row.hmt_record_id in result:
                raise ValueError("duplicate flash-point HMT identity")
            record = hmt.get(row.hmt_record_id)
            if (
                record is None
                or record.un_number != row.un_number
                or record.nos_entry
                or record.technical_name_required
                or (row.hmt_record_id, row.un_number, row.cas_number) not in links
            ):
                raise ValueError(
                    "flash-point CAS/UN identity is not an exact eligible registry link"
                )
            result[row.hmt_record_id] = row
        return result


def load_flash_points(path: Path, sha256: str, records: int | None) -> DgFlashPointRegistry:
    if sha256_file(path) != sha256:
        raise ValueError("flash-point registry content differs from its configured pin")
    value = DgFlashPointRegistry.model_validate(json.loads(path.read_text()))
    if records is not None and len(value.records) != records:
        raise ValueError("flash-point registry count differs from its configured pin")
    return value
