"""Explicit, reproducible calibration and font inputs for synthetic page reflow."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import ImageFont
from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.hashing import sha256_file


class ReflowPolicy(BaseModel):
    """Source-conditioned layout settings; every external input is hash-pinned."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    calibration_path: Path
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    font_path: Path
    font_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    row_lock_height_fraction: float = Field(gt=0, le=1)
    column_guard_height_fraction: float = Field(ge=0, le=1)
    bottom_guard_height_fraction: float = Field(ge=0, le=2)


class ReflowCalibration(BaseModel):
    """Measured priors, fitted once and shared across campaigns and variants."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    method: Literal["owned_source_spacing_v1"]
    source_dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_alignment_sha256: dict[str, str]
    fitting_sources: tuple[str, ...] = Field(min_length=1)
    diagnostic_sources: tuple[str, ...]
    observation_count: int = Field(gt=0)
    pitch_ratio_range: tuple[float, float]
    clearance_height_ratio: float = Field(ge=0)
    minimum_page_median_height: float = Field(gt=0, le=1000)
    density_quantile: float = Field(gt=0, lt=1)
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self):
        low, high = self.pitch_ratio_range
        if not 1 <= low <= high:
            raise ValueError("pitch ratio range must be ordered and at least one glyph height")
        fitted, diagnostics = set(self.fitting_sources), set(self.diagnostic_sources)
        if len(fitted) != len(self.fitting_sources) or len(diagnostics) != len(
            self.diagnostic_sources
        ):
            raise ValueError("duplicate calibration sources")
        if fitted & diagnostics or set(self.source_alignment_sha256) != fitted:
            raise ValueError("calibration source partitions/hashes disagree")
        if any(
            len(h) != 64 or set(h) - set("0123456789abcdef")
            for h in self.source_alignment_sha256.values()
        ):
            raise ValueError("invalid calibration alignment hash")
        return self


class FontMetrics:
    """Bounded, per-engine advance cache; no global font state or font substitution."""

    def __init__(self, path: Path, expected_sha256: str):
        if sha256_file(path) != expected_sha256:
            raise ValueError("reflow font hash mismatch")
        # Normalized reference size, cancelled by the measured source width ratio.
        self.font = ImageFont.truetype(str(path), 100)
        self.length = lru_cache(maxsize=4096)(self.font.getlength)


def load_calibration(root: Path, policy: ReflowPolicy) -> ReflowCalibration:
    path = root / policy.calibration_path
    if sha256_file(path) != policy.calibration_sha256:
        raise ValueError("reflow calibration hash mismatch")
    return ReflowCalibration.model_validate_json(path.read_bytes())
