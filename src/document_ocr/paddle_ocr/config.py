"""Explicit runtime, geometry and source contracts for structured OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.config import (
    LocalSourceConfig,
    OutputConfig,
    RasterConfig,
    RunConfig,
    load_strict_yaml_mapping,
)

Positive = Annotated[int, Field(gt=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ModelSpec(StrictModel):
    """An immutable Hugging Face model snapshot, downloaded before inference."""

    name: Annotated[str, StringConstraints(min_length=1)]
    repository: Annotated[str, StringConstraints(pattern=r"^[\w.-]+/[\w.-]+$")]
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class PaddleSettings(StrictModel):
    """Pinned local Paddle engine; preprocessing outputs retain their own frame."""

    device: Annotated[str, StringConstraints(pattern=r"^(cpu|gpu:[0-9]+)$")]
    cpu_threads: Positive
    enable_mkldnn: bool
    mkldnn_cache_capacity: Positive
    models_directory: str
    detection: ModelSpec
    recognition: ModelSpec
    document_orientation: ModelSpec | None
    document_unwarping: ModelSpec | None
    textline_orientation: ModelSpec | None
    recognition_batch_size: Positive
    orientation_batch_size: Positive
    detection_limit_side_len: Positive
    detection_limit_type: Literal["min", "max"]
    detection_max_side_limit: Positive
    detection_threshold: Probability
    detection_box_threshold: Probability
    detection_unclip_ratio: Annotated[float, Field(gt=0)]
    recognition_score_threshold: Probability
    return_word_boxes: bool

    def models(self) -> dict[str, ModelSpec]:
        return {
            key: value
            for key in (
                "detection",
                "recognition",
                "document_orientation",
                "document_unwarping",
                "textline_orientation",
            )
            if (value := getattr(self, key)) is not None
        }

    @model_validator(mode="after")
    def explicit_runtime(self) -> PaddleSettings:
        if not Path(self.models_directory).is_absolute():
            raise ValueError("models_directory must be absolute")
        if self.device != "cpu" and self.enable_mkldnn:
            raise ValueError("MKLDNN is CPU-only; disable it for GPU processing")
        return self


class Exclusion(StrictModel):
    source_sha256: Sha256
    reason: Annotated[str, StringConstraints(min_length=1)]


class Concurrency(StrictModel):
    """One persistent recognizer overlaps a bounded process-isolated render queue."""

    renderer_processes: Positive
    prefetch_pages: Positive


class PaddleConfig(StrictModel):
    schema_version: Literal[1]
    backend: Literal["paddleocr"]
    sources: list[LocalSourceConfig] = Field(min_length=1)
    exclusions: list[Exclusion]
    expected_source_documents: Positive | None
    expected_unique_documents: Positive | None
    output: OutputConfig
    run: RunConfig
    raster: RasterConfig
    paddle: PaddleSettings
    concurrency: Concurrency

    @model_validator(mode="after")
    def non_overlapping_paths(self) -> PaddleConfig:
        roots = [Path(source.root).resolve() for source in self.sources]
        output = Path(self.output.root).resolve()
        for i, root in enumerate(roots):
            if root == Path(root.anchor):
                raise ValueError("a source cannot be the filesystem root")
            if root == output or root in output.parents or output in root.parents:
                raise ValueError("source and output directories must not overlap")
            for other in roots[i + 1 :]:
                if root == other or root in other.parents or other in root.parents:
                    raise ValueError("source directories must not overlap")
        exclusions = [item.source_sha256 for item in self.exclusions]
        if len(exclusions) != len(set(exclusions)):
            raise ValueError("duplicate excluded source hashes")
        return self


def load_config(path: Path) -> PaddleConfig:
    return PaddleConfig.model_validate(load_strict_yaml_mapping(path))
