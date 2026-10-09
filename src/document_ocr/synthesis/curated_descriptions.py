"""Project descriptions from reviewed source blocks through exact render edits.

Block membership is an explicit source-level decision, independent of mutable
product wording and shipment accounting. This module never infers membership
from old labels, nearby captions, or substring matches in generated documents.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class DescriptionSpan(BaseModel):
    """An exact, reviewed occurrence inside a goods-description block."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str = Field(min_length=1)
    occurrence: int = Field(ge=1, strict=True)


class DescriptionField(BaseModel):
    """Source-ordered fragments forming one complete description target.

    Select one representative occurrence of repeated copies explicitly. Separate
    spans permit genuine product continuation around HS, references, and other
    excluded fields, not detached auxiliary enrichment. Boundary uncertainty is
    adjudicated before admitting a source contract, not guessed by this compiler.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(pattern=r"^documentPatch\.goodsItemDetails\[\d+\]\.description$")
    spans: tuple[DescriptionSpan, ...] = Field(min_length=1)


class DescriptionBlocks(BaseModel):
    """Complete reviewed description membership pinned to exact source OCR."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fields: tuple[DescriptionField, ...] = Field(min_length=1)


@dataclass(frozen=True)
class CompiledDescriptionSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class CompiledDescriptionField:
    path: str
    spans: tuple[CompiledDescriptionSpan, ...]


@dataclass(frozen=True)
class CompiledDescriptionBlocks:
    source_sha256: str
    contract_sha256: str
    fields: tuple[CompiledDescriptionField, ...]

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(field.path for field in self.fields)


class MutableRegion(Protocol):
    start: int
    end: int
    key: str
    kind: str
    target_paths: tuple[str, ...]


def normalized_description(
    text: str, casing: Literal["uppercase", "preserve"] = "uppercase"
) -> str:
    if casing not in {"uppercase", "preserve"}:
        raise ValueError(f"unknown description target casing: {casing}")
    value = " ".join(text.split())
    if not value:
        raise ValueError("approved description block is empty")
    return value.upper() if casing == "uppercase" else value


def compile_description_blocks(
    source: str,
    target: dict[str, Any],
    declaration: DescriptionBlocks | dict | None,
) -> CompiledDescriptionBlocks | None:
    """Resolve exact selectors and prove the source target equals their projection."""
    descriptions = {
        f"documentPatch.goodsItemDetails[{i}].description": item["description"]
        for i, item in enumerate(target.get("documentPatch", {}).get("goodsItemDetails", []))
        if "description" in item
    }
    if declaration is None:
        if descriptions:
            raise ValueError("description targets require reviewed description_blocks")
        return None
    contract = DescriptionBlocks.model_validate(declaration)
    raw = source.encode()
    if hashlib.sha256(raw).hexdigest() != contract.source_sha256:
        raise ValueError("description_blocks source SHA-256 differs from OCR")
    paths = [field.path for field in contract.fields]
    if len(paths) != len(set(paths)) or set(paths) != set(descriptions):
        raise ValueError("description_blocks must cover every description target exactly once")
    fields = []
    owned_spans = []
    for field in contract.fields:
        spans = []
        previous_end = -1
        for selector in field.spans:
            matches = list(re.finditer(re.escape(selector.text), source))
            if selector.occurrence > len(matches):
                raise ValueError(f"description quote occurrence absent: {field.path}: {selector}")
            match = matches[selector.occurrence - 1]
            start = len(source[: match.start()].encode())
            end = start + len(selector.text.encode())
            if start < previous_end:
                raise ValueError("description spans must be disjoint and in source order")
            previous_end = end
            spans.append(CompiledDescriptionSpan(start, end, selector.text))
            owned_spans.append((start, end))
        expected = normalized_description(" ".join(span.text for span in spans))
        if descriptions[field.path] != expected:
            raise ValueError(f"source description differs from approved block: {field.path}")
        fields.append(CompiledDescriptionField(field.path, tuple(spans)))
    ordered = sorted(owned_spans)
    if any(a[1] > b[0] for a, b in pairwise(ordered)):
        raise ValueError("description blocks for different goods overlap")
    serialized = json.dumps(contract.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return CompiledDescriptionBlocks(
        contract.source_sha256, hashlib.sha256(serialized.encode()).hexdigest(), tuple(fields)
    )


def validate_description_regions(
    blocks: CompiledDescriptionBlocks | None, regions: tuple[MutableRegion, ...]
) -> None:
    """A mutable edit is wholly included or excluded, never sliced at a boundary."""
    if blocks is None:
        if any(any(path.endswith(".description") for path in r.target_paths) for r in regions):
            raise ValueError("description owners require reviewed description_blocks")
        return
    selected_products: set[str] = set()
    product_keys = {region.key for region in regions if region.kind == "product"}
    for field in blocks.fields:
        for span in field.spans:
            for region in regions:
                if region.start < span.end and span.start < region.end:
                    if not (span.start <= region.start and region.end <= span.end):
                        raise ValueError(
                            f"description boundary crosses mutable region {region.key}: "
                            f"block {span.start}:{span.end}, owner {region.start}:{region.end}"
                        )
                    if region.kind == "product":
                        if field.path not in region.target_paths:
                            raise ValueError(
                                f"product owner {region.key} belongs to another description: "
                                f"{field.path}"
                            )
                        selected_products.add(region.key)
    if missing := product_keys - selected_products:
        raise ValueError(
            f"product owners lack an approved description occurrence: {sorted(missing)}"
        )


def project_rendered_descriptions(
    blocks: CompiledDescriptionBlocks | None,
    source: bytes,
    rendered: bytes,
    edits: list[dict[str, Any]],
    *,
    casing: Literal["uppercase", "preserve"] = "uppercase",
) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Rebuild approved spans independently and verify their exact rendered bytes."""
    if blocks is None:
        return {}, None
    if hashlib.sha256(source).hexdigest() != blocks.source_sha256:
        raise ValueError("description projection source SHA-256 differs")
    previous_end = 0
    for edit in edits:
        start, end = edit["byteStart"], edit["byteEnd"]
        if not previous_end <= start < end <= len(source):
            raise ValueError("description projection edits overlap or are out of order")
        if source[start:end].decode() != edit["before"]:
            raise ValueError("description projection edit differs from source")
        previous_end = end
    values, receipts = {}, []
    for field in blocks.fields:
        fragments, spans = [], []
        for span in field.spans:
            shift = 0
            chunks = []
            cursor = span.start
            for edit in edits:
                start, end, replacement = edit["byteStart"], edit["byteEnd"], edit["after"].encode()
                if end <= span.start:
                    shift += len(replacement) - (end - start)
                    continue
                if start >= span.end:
                    break
                if not span.start <= start < end <= span.end:
                    raise ValueError("description projection edit crosses approved block boundary")
                chunks.extend((source[cursor:start], replacement))
                cursor = end
            chunks.append(source[cursor : span.end])
            fragment = b"".join(chunks)
            start = span.start + shift
            end = start + len(fragment)
            if rendered[start:end] != fragment:
                raise ValueError("description projection differs from rendered bytes")
            fragments.append(fragment.decode())
            spans.append(
                {
                    "sourceByteStart": span.start,
                    "sourceByteEnd": span.end,
                    "renderedByteStart": start,
                    "renderedByteEnd": end,
                    "text": fragment.decode(),
                }
            )
        value = normalized_description(" ".join(fragments), casing)
        values[field.path] = value
        receipts.append({"path": field.path, "spans": spans, "value": value})
    return values, {
        "sourceSha256": blocks.source_sha256,
        "contractSha256": blocks.contract_sha256,
        "casing": casing,
        "fields": receipts,
    }
