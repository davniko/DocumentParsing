"""Committed deep EDA for compiled-template descendant synthesis runs.

The analysis distinguishes byte/target correctness proven by the rendering host from
whole-document plausibility.  The latter cannot be inferred from slot-local validators, so
this module combines deterministic cross-field screens with a pinned purposive manual review.
"""

from __future__ import annotations

import io
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
import yaml
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.country_registry import CountryRegistry, load_iso_country_registry
from document_ocr.synthesis.run_safety import StagedArtifactRun

from .descendant import _validate_committed_run
from .descendant_models import DescendantCaseResult, PinnedCommittedRun
from .models import NonEmptyText, PinnedFile

plt.switch_backend("Agg")

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_ACCEPTANCE_COLUMNS = (
    "carrier_unchanged",
    "exact_topology",
    "every_slot_bound_once",
    "exact_literal_regions",
    "page_markers_unchanged",
    "line_endings_preserved",
    "format_envelopes_valid",
    "target_binding_semantics_valid",
    "source_relationships_valid",
    "semantic_coherence_valid",
)
_ROUTE_COLUMNS = (
    "document_id",
    "binding_id",
    "logical_key",
    "group_kind",
    "group_key",
    "value_kind",
    "render_mode",
    "compiled_mode",
    "runtime_route",
    "route_reason",
    "slot_count",
)
_ADAPTATION_COLUMNS = (
    "document_id",
    "target_origin",
    "target_path",
    "target_family",
    "category",
    "reason",
    "source_value",
    "proposed_value",
    "adapted_value",
)
_TARGET_CHANGE_COLUMNS = (
    "document_id",
    "target_path",
    "family",
    "source_value",
    "target_value",
)
_COUNTRY_SCREEN_COLUMNS = (
    "document_id",
    "relationship",
    "target_party_path",
    "expected_country",
    "expected_country_code",
    "exporter_country_surfaces",
    "exporter_country_codes",
    "comparable",
    "consistent",
    "target_origin",
    "compilation_lineage",
)
_NUMBER_CONTRADICTION_COLUMNS = (
    "document_id",
    "line_number",
    "kind",
    "surface",
    "left_value",
    "right_value",
    "target_origin",
    "compilation_lineage",
)
_ORIGIN_LABELS = {
    "existing_linguistic_target_carrier_restored": "Existing full target",
    "controlled_source_variant": "Controlled source variant",
}
_PALETTE = {
    "deterministic": "#1976D2",
    "agent": "#EF6C00",
    "pass": "#2E7D32",
    "concern": "#C62828",
    "pass_with_source_limitation": "#F9A825",
    "Existing full target": "#00897B",
    "Controlled source variant": "#8E24AA",
}
_EXPORTER_COUNTRY_INLINE = re.compile(
    r"(?:FOREIGN\s+EXPORTER(?:\s+REGISTRATION)?\s+COUNTRY|"
    r"EXPORTER\s+REGISTRATION\s+COUNTRY)\s*:\s*(.+)",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
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
_NUMBER_TOKEN = "|".join((*_NUMBER_WORDS, "HUNDRED", "THOUSAND", "AND"))
_PARENTHETICAL_NUMBER = re.compile(
    rf"\b((?:{_NUMBER_TOKEN})(?:[\s-]+(?:{_NUMBER_TOKEN})){{0,7}})\s*\(\s*(\d+)\s*\)",
    re.IGNORECASE,
)
_SEQUENCE_OF_TOTAL = re.compile(
    r"\b(\d+)\s+OF\s+(ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b",
    re.IGNORECASE,
)


class DescendantEdaConfig(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["bill_of_lading_compiled_descendant_eda_v1"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    descendant_run: PinnedCommittedRun
    template_catalog: PinnedCommittedRun
    iso3166_snapshot: PinnedFile
    manual_review: PinnedFile
    expected_documents: Annotated[int, Field(gt=0, le=100_000)]
    expected_manual_reviews: Annotated[int, Field(gt=0, le=1_000)]


ReviewGrade = Literal["pass", "concern", "pass_with_source_limitation"]


class ManualReviewEntry(BaseModel):
    model_config = _STRICT

    document_id: NonEmptyText
    target_origin: Literal[
        "existing_linguistic_target_carrier_restored",
        "controlled_source_variant",
    ]
    compilation_lineage: NonEmptyText
    selection_reason: NonEmptyText
    mechanical_fidelity: Literal["pass"]
    target_binding_fidelity: Literal["pass"]
    carrier_preservation: Literal["pass"]
    layout_preservation: Literal["pass"]
    whole_document_coherence: ReviewGrade
    quality_decision: ReviewGrade | Literal["review_required"]
    primary_surface: NonEmptyText
    evidence_lines: Annotated[tuple[int, ...], Field(min_length=1)]
    observation: NonEmptyText

    @model_validator(mode="after")
    def evidence_lines_are_positive_and_ordered(self) -> ManualReviewEntry:
        if any(line < 1 for line in self.evidence_lines):
            raise ValueError("manual-review evidence lines must be positive")
        if tuple(sorted(set(self.evidence_lines))) != self.evidence_lines:
            raise ValueError("manual-review evidence lines must be unique and sorted")
        if self.whole_document_coherence == "concern" and self.quality_decision != (
            "review_required"
        ):
            raise ValueError("a coherence concern must be review_required")
        return self


class ManualReview(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    run: PinnedCommittedRun
    template_catalog: PinnedCommittedRun
    selection_method: NonEmptyText
    reviews: Annotated[tuple[ManualReviewEntry, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def reviewed_documents_are_unique(self) -> ManualReview:
        document_ids = tuple(row.document_id for row in self.reviews)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("manual review repeats a document")
        return self


def load_eda_config(path: Path) -> DescendantEdaConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return DescendantEdaConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, default=str)
    )


def _load_manual_review(path: Path) -> ManualReview:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return ManualReview.model_validate_json(json.dumps(payload, ensure_ascii=False, default=str))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_regular_file_bytes(path).splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: JSONL is empty")
    return tuple(rows)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return str(frame.to_csv(index=False, lineterminator="\n")).encode("utf-8")


def _figure_bytes(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def _describe(series: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(series.min()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "maximum": float(series.max()),
        "mean": float(series.mean()),
    }


def _flatten_leaves(value: Any, *, path: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            output.update(_flatten_leaves(child, path=child_path))
        return output
    if isinstance(value, list):
        output = {}
        for index, child in enumerate(value):
            output.update(_flatten_leaves(child, path=f"{path}[{index}]"))
        return output
    return {path: value}


def _change_family(path: str) -> str:
    normalized = path.removeprefix("documentPatch.")
    root = normalized.split(".", maxsplit=1)[0].split("[", maxsplit=1)[0]
    if root == "parties":
        return "parties"
    if root in {"route", "placeOfIssue"}:
        return "route"
    if root == "transport":
        return "transport"
    if root == "containers":
        return "equipment"
    if root == "cargoGroups":
        return "cargo"
    if root in {"cargoPackages", "cargoAllocationGroups"}:
        return "packages"
    if root.lower().startswith("dangerous"):
        return "dangerous_goods"
    if root == "freight":
        return "freight"
    if root in {"issueDate", "shippedOnBoardDate"}:
        return "dates"
    if root in {
        "billOfLadingNumber",
        "bookingReference",
        "forwarderReference",
        "negotiability",
    }:
        return "document"
    return "other"


def _target_change_rows(
    *, document_id: str, source_target: Mapping[str, Any], target: Mapping[str, Any]
) -> list[dict[str, Any]]:
    source = _flatten_leaves(source_target)
    rendered = _flatten_leaves(target)
    rows: list[dict[str, Any]] = []
    for path in sorted(set(source) | set(rendered)):
        source_value = source.get(path)
        target_value = rendered.get(path)
        if source_value != target_value:
            rows.append(
                {
                    "document_id": document_id,
                    "target_path": path,
                    "family": _change_family(path),
                    "source_value": json.dumps(source_value, ensure_ascii=False, default=str),
                    "target_value": json.dumps(target_value, ensure_ascii=False, default=str),
                }
            )
    return rows


def _number_words_to_int(text: str) -> int | None:
    tokens = text.upper().replace("-", " ").split()
    allowed = {*_NUMBER_WORDS, "HUNDRED", "THOUSAND", "AND"}
    if not tokens or any(token not in allowed for token in tokens):
        return None
    total = 0
    current = 0
    for token in tokens:
        if token == "AND":
            continue
        if token == "HUNDRED":
            current = max(current, 1) * 100
        elif token == "THOUSAND":
            total += max(current, 1) * 1_000
            current = 0
        else:
            current += _NUMBER_WORDS[token]
    return total + current


def _number_contradictions(document_id: str, text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in _PARENTHETICAL_NUMBER.finditer(line):
            words_value = _number_words_to_int(match.group(1))
            numeric_value = int(match.group(2))
            if words_value is not None and words_value != numeric_value:
                rows.append(
                    {
                        "document_id": document_id,
                        "line_number": line_number,
                        "kind": "words_parenthetical_digits_disagree",
                        "surface": match.group(0),
                        "left_value": words_value,
                        "right_value": numeric_value,
                    }
                )
        for match in _SEQUENCE_OF_TOTAL.finditer(line):
            sequence = int(match.group(1))
            total = _NUMBER_WORDS[match.group(2).upper()]
            if sequence > total or (total == 0 and sequence != 0):
                rows.append(
                    {
                        "document_id": document_id,
                        "line_number": line_number,
                        "kind": "sequence_exceeds_total",
                        "surface": match.group(0),
                        "left_value": sequence,
                        "right_value": total,
                    }
                )
    return rows


def _country_prefix(registry: CountryRegistry, value: str) -> tuple[str | None, str]:
    cleaned = re.split(
        r"\bFOREIGN\s+EXPORTER\s+COUNTRY\s+CODE\b|\bFREIGHT\b|\*\*",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,.;*/")
    tokens = cleaned.split()
    for count in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:count]).strip(" ,.;*/")
        resolved = registry.resolve(candidate)
        if resolved is not None:
            return resolved, candidate
    return None, cleaned


def _exporter_country_surfaces(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    surfaces: list[str] = []
    for index, line in enumerate(lines):
        inline = _EXPORTER_COUNTRY_INLINE.search(line)
        if inline is not None:
            surfaces.append(inline.group(1).strip())
            continue
        normalized = " ".join(line.upper().replace(":", " ").split())
        if normalized == "FOREIGN EXPORTER COUNTRY" and index + 1 < len(lines):
            next_value = lines[index + 1].strip()
            if next_value and not next_value.upper().startswith("CODE"):
                surfaces.append(next_value)
        if normalized == "FOREIGN EXPORTER" and index + 1 < len(lines):
            match = re.match(r"\s*COUNTRY\s*:\s*(.+)", lines[index + 1], re.IGNORECASE)
            if match is not None:
                surfaces.append(match.group(1).strip())
    return tuple(dict.fromkeys(surfaces))


def _country_screen_row(
    *,
    document_id: str,
    text: str,
    target: Mapping[str, Any],
    template: Mapping[str, Any],
    registry: CountryRegistry,
) -> dict[str, Any] | None:
    surfaces = _exporter_country_surfaces(text)
    if not surfaces:
        return None
    plan = template.get("auxiliary_semantic_plan")
    entities = plan.get("entities") if isinstance(plan, Mapping) else None
    exporter_entities = tuple(
        row
        for row in entities or ()
        if isinstance(row, Mapping)
        and str(row.get("role", "")).split(":", maxsplit=1)[0] in {"exporter", "foreign_exporter"}
    )
    if len(exporter_entities) != 1:
        raise ValueError(
            f"exporter country surface lacks one canonical exporter entity: {document_id}"
        )
    entity = exporter_entities[0]
    relationship = entity.get("relationship")
    target_party_path = entity.get("target_party_path")
    expected_country: str | None = None
    if relationship == "same_as_target_party":
        if not isinstance(target_party_path, str):
            raise ValueError(f"target-linked exporter has no party path: {document_id}")
        current: Any = target
        for name, index_text in re.findall(r"([A-Za-z0-9_]+)(?:\[([0-9]+)\])?", target_party_path):
            if not isinstance(current, Mapping) or name not in current:
                raise ValueError(f"exporter target party path is absent: {document_id}")
            current = current[name]
            if index_text:
                if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                    raise ValueError(f"exporter target party index is invalid: {document_id}")
                current = current[int(index_text)]
        if isinstance(current, Mapping) and isinstance(current.get("country"), str):
            expected_country = cast(str, current["country"])
    elif relationship != "independent":
        raise ValueError(f"unsupported exporter relationship: {document_id}")
    expected_code = registry.resolve(expected_country) if expected_country is not None else None
    resolved = tuple(_country_prefix(registry, value) for value in surfaces)
    exporter_codes = tuple(sorted({code for code, _ in resolved if code is not None}))
    comparable = bool(exporter_codes) and all(code is not None for code, _ in resolved)
    consistent = (
        exporter_codes == (expected_code,)
        if comparable and expected_code is not None
        else len(exporter_codes) == 1
        if comparable
        else None
    )
    return {
        "document_id": document_id,
        "relationship": relationship,
        "target_party_path": target_party_path,
        "expected_country": expected_country,
        "expected_country_code": expected_code,
        "exporter_country_surfaces": " | ".join(surfaces),
        "exporter_country_codes": " | ".join(exporter_codes),
        "comparable": comparable,
        "consistent": consistent if comparable else None,
    }


def _adaptation_category(reason: str) -> str:
    if reason.startswith(
        ("atomic party representability constraint;", "atomic party auxiliary-facet constraint;")
    ):
        return "atomic party retention"
    if reason.startswith("compiled descendant compatibility constraint:"):
        return "certified format constraint"
    if reason.startswith("compiled inclusive-range coherence contract"):
        return "inclusive-range reconciliation"
    if reason.startswith("compiled legacy equipment surface"):
        return "legacy equipment retention"
    if reason == "carrier-bound templates retain the complete source carrier object":
        return "carrier retention"
    return "other"


def _list_length(value: Any) -> int:
    return len(value) if isinstance(value, list) else int(value is not None)


def _rows_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    sort_by: Sequence[str] = (),
) -> pd.DataFrame:
    """Materialize an analysis table without losing its schema when it has no rows."""
    frame = pd.DataFrame.from_records(rows, columns=columns)
    if not frame.empty and sort_by:
        frame = frame.sort_values(list(sort_by)).reset_index(drop=True)
    return frame


def _build_frames(
    *,
    run_root: Path,
    catalog_root: Path,
    results: Sequence[DescendantCaseResult],
    registry: CountryRegistry,
) -> dict[str, pd.DataFrame]:
    catalog_rows = {
        cast(str, row["documentId"]): row for row in _jsonl(catalog_root / "catalog.jsonl")
    }
    documents: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    adaptations: list[dict[str, Any]] = []
    target_changes: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []
    contradiction_rows: list[dict[str, Any]] = []

    for result in results:
        document_id = result.document_id
        if document_id not in catalog_rows:
            raise ValueError(f"descendant document is absent from template catalog: {document_id}")
        case_root = run_root / "cases" / document_id
        catalog_case = catalog_root / "cases" / document_id
        catalog = catalog_rows[document_id]
        case_catalog = _json(catalog_case / "catalog-row.json")
        if canonical_json_bytes(case_catalog) != canonical_json_bytes(catalog):
            raise ValueError(f"catalog row differs from joined case artifact: {document_id}")
        lineage = _json(catalog_case / "lineage.json")
        template = _json(catalog_case / "template.json")
        template_bindings = {
            cast(str, row["binding_id"]): row
            for row in cast(list[dict[str, Any]], template["bindings"])
        }
        route_plan = json.loads(read_regular_file_bytes(case_root / "route-plan.json"))
        if not isinstance(route_plan, list):
            raise ValueError(f"route plan is not a list: {document_id}")
        if {cast(str, row["binding_id"]) for row in route_plan} != set(template_bindings):
            raise ValueError(f"runtime route coverage differs from template: {document_id}")
        for route in route_plan:
            binding_id = cast(str, route["binding_id"])
            compiled = template_bindings[binding_id]
            routes.append(
                {
                    "document_id": document_id,
                    "binding_id": binding_id,
                    "logical_key": route["logical_key"],
                    "group_kind": compiled["group_kind"],
                    "group_key": compiled["group_key"],
                    "value_kind": compiled["value_kind"],
                    "render_mode": compiled["render_mode"],
                    "compiled_mode": route["compiled_mode"],
                    "runtime_route": route["runtime_route"],
                    "route_reason": route["route_reason"],
                    "slot_count": len(route["slot_ids"]),
                }
            )

        rendered_bytes = read_regular_file_bytes(case_root / "rendered.txt")
        rendered_text = rendered_bytes.decode("utf-8")
        source_target = _json(case_root / "source-target.json")
        target = _json(case_root / "target.json")
        target_receipt = _json(case_root / "target-receipt.json")
        replay = _json(case_root / "replay-receipt.json")
        agent_stage = _json(case_root / "source-agent-stage.json")
        usage = agent_stage.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError(f"agent-stage usage is absent: {document_id}")
        stage_cost = Decimal(cast(str, usage["estimatedCostUsd"]))
        source_requests = int(usage["requests"])
        current_provider_required = result.agent_binding_count > 0

        changes = _target_change_rows(
            document_id=document_id,
            source_target=source_target,
            target=target,
        )
        target_changes.extend(changes)
        final_change_counts = Counter(cast(str, row["family"]) for row in changes)
        for adaptation in cast(list[dict[str, Any]], target_receipt["compatibility_adaptations"]):
            target_path = cast(str, adaptation["target_path"])
            reason = cast(str, adaptation["reason"])
            adaptations.append(
                {
                    "document_id": document_id,
                    "target_origin": result.target_origin,
                    "target_path": target_path,
                    "target_family": _change_family(target_path),
                    "category": _adaptation_category(reason),
                    "reason": reason,
                    "source_value": json.dumps(
                        adaptation["source_value"], ensure_ascii=False, default=str
                    ),
                    "proposed_value": json.dumps(
                        adaptation["proposed_value"], ensure_ascii=False, default=str
                    ),
                    "adapted_value": json.dumps(
                        adaptation["adapted_value"], ensure_ascii=False, default=str
                    ),
                }
            )

        country_row = _country_screen_row(
            document_id=document_id,
            text=rendered_text,
            target=target,
            template=template,
            registry=registry,
        )
        if country_row is not None:
            country_row.update(
                {
                    "target_origin": result.target_origin,
                    "compilation_lineage": lineage["sourceRun"],
                }
            )
            country_rows.append(country_row)
        contradictions = _number_contradictions(document_id, rendered_text)
        for contradiction in contradictions:
            contradiction.update(
                {
                    "target_origin": result.target_origin,
                    "compilation_lineage": lineage["sourceRun"],
                }
            )
        contradiction_rows.extend(contradictions)

        patch = target.get("documentPatch")
        patch = patch if isinstance(patch, Mapping) else {}
        parties = patch.get("parties")
        documents.append(
            {
                **result.model_dump(mode="json"),
                "target_origin_label": _ORIGIN_LABELS[result.target_origin],
                "compilation_lineage": lineage["sourceRun"],
                "carrier": catalog["carrier"],
                "carrier_family": catalog["carrierFamily"],
                "document_type": catalog["documentType"],
                "pages": int(catalog["pages"]),
                "source_lines": int(catalog["lines"]),
                "source_characters": int(catalog["characters"]),
                "output_lines": len(rendered_bytes.splitlines()),
                "byte_delta": result.output_bytes - result.source_bytes,
                "deterministic_slot_fraction": (
                    result.deterministic_slot_count / result.template_slot_count
                ),
                "changed_slot_fraction": result.changed_slot_count / result.template_slot_count,
                "final_changed_target_leaves": len(changes),
                "target_adaptations": len(target_receipt["compatibility_adaptations"]),
                "current_provider_required": current_provider_required,
                "source_provider_requests": source_requests,
                "source_provider_cost_usd": float(stage_cost),
                "current_route_cost_proxy_usd": (
                    float(stage_cost) if current_provider_required else 0.0
                ),
                "source_provider_duration_seconds": float(agent_stage["duration_seconds"]),
                "source_residual_slots": int(replay["source_output_slot_count"]),
                "replayed_residual_slots": int(replay["replayed_output_slot_count"]),
                "dropped_residual_slots": int(replay["dropped_output_slot_count"]),
                "visible_output_tokens": result.output_tokens - result.reasoning_tokens,
                "containers": _list_length(patch.get("containers")),
                "cargo_groups": _list_length(patch.get("cargoGroups")),
                "cargo_packages": _list_length(patch.get("cargoPackages")),
                "dangerous_goods": _list_length(patch.get("dangerousGoods")),
                "party_roles": len(parties) if isinstance(parties, Mapping) else 0,
                **{
                    f"final_changes_{family}": int(final_change_counts[family])
                    for family in (
                        "document",
                        "dates",
                        "parties",
                        "route",
                        "transport",
                        "equipment",
                        "cargo",
                        "packages",
                        "dangerous_goods",
                        "freight",
                        "other",
                    )
                },
            }
        )

    document_frame = pd.DataFrame(documents).sort_values("document_id").reset_index(drop=True)
    route_frame = _rows_frame(
        routes,
        columns=_ROUTE_COLUMNS,
        sort_by=("document_id", "binding_id"),
    )
    adaptation_frame = _rows_frame(
        adaptations,
        columns=_ADAPTATION_COLUMNS,
        sort_by=("document_id", "target_path"),
    )
    target_change_frame = _rows_frame(
        target_changes,
        columns=_TARGET_CHANGE_COLUMNS,
        sort_by=("document_id", "target_path"),
    )
    country_frame = _rows_frame(
        country_rows,
        columns=_COUNTRY_SCREEN_COLUMNS,
        sort_by=("document_id",),
    )
    contradiction_frame = _rows_frame(
        contradiction_rows,
        columns=_NUMBER_CONTRADICTION_COLUMNS,
        sort_by=("document_id", "line_number", "kind"),
    )
    return {
        "documents": document_frame,
        "routes": route_frame,
        "adaptations": adaptation_frame,
        "target_changes": target_change_frame,
        "country_screen": country_frame,
        "number_contradictions": contradiction_frame,
    }


def _manual_frame(review: ManualReview) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump(mode="json") for row in review.reviews]).sort_values(
        "document_id"
    )


def _outlier_frame(documents: pd.DataFrame) -> pd.DataFrame:
    dimensions = {
        "template_slot_count": "largest template",
        "source_lines": "most source lines",
        "changed_slot_count": "most changed slots",
        "changed_target_leaf_count": "most proposed target changes",
        "agent_slot_count": "most residual-agent slots",
        "source_provider_cost_usd": "highest historical residual cost",
        "source_provider_duration_seconds": "longest historical provider latency",
        "target_adaptations": "most compatibility adaptations",
        "containers": "most containers",
        "cargo_groups": "most cargo groups",
        "cargo_packages": "most cargo packages",
    }
    rows: list[dict[str, Any]] = []
    for column, label in dimensions.items():
        ordered = documents.sort_values([column, "document_id"], ascending=[False, True]).head(5)
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            rows.append(
                {
                    "dimension": column,
                    "label": label,
                    "rank": rank,
                    "document_id": row["document_id"],
                    "value": row[column],
                    "target_origin": row["target_origin"],
                    "compilation_lineage": row["compilation_lineage"],
                }
            )
    return pd.DataFrame(rows)


def _plots(
    *, frames: Mapping[str, pd.DataFrame], manual: pd.DataFrame, summary: Mapping[str, Any]
) -> dict[str, bytes]:
    sns.set_theme(style="whitegrid", context="notebook")
    documents = frames["documents"]
    routes = frames["routes"]
    adaptations = frames["adaptations"]
    target_changes = frames["target_changes"]
    country_screen = frames["country_screen"]
    contradictions = frames["number_contradictions"]
    origin_palette = {label: _PALETTE[label] for label in documents["target_origin_label"].unique()}
    output: dict[str, bytes] = {}

    invariants = {
        "Accepted": int((documents["status"] == "passed").sum()),
        "Carrier unchanged": int(documents["carrier_unchanged"].sum()),
        "Exact topology": int(documents["exact_topology"].sum()),
        "All slots bound": int(documents["every_slot_bound_once"].sum()),
        "Literal regions exact": int(documents["exact_literal_regions"].sum()),
        "Page markers": int(documents["page_markers_unchanged"].sum()),
        "Line endings": int(documents["line_endings_preserved"].sum()),
        "Formats valid": int(documents["format_envelopes_valid"].sum()),
        "Target semantics": int(documents["target_binding_semantics_valid"].sum()),
        "Source relations": int(documents["source_relationships_valid"].sum()),
        "Compiled coherence": int(documents["semantic_coherence_valid"].sum()),
    }
    figure, axis = plt.subplots(figsize=(10, 6.5))
    labels = list(invariants)
    values = list(invariants.values())
    bars = axis.barh(labels[::-1], values[::-1], color=_PALETTE["pass"])
    axis.bar_label(bars)
    axis.set(
        xlim=(0, int(summary["documents"]) * 1.08),
        xlabel="Documents",
        title="Every machine acceptance invariant passed",
    )
    output["plots/01_machine_acceptance.png"] = _figure_bytes(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    origin = documents["target_origin_label"].value_counts()
    axes[0].bar(origin.index, origin.values, color=[origin_palette[x] for x in origin.index])
    axes[0].bar_label(axes[0].containers[0])
    axes[0].set(title="Target origin", ylabel="Documents")
    axes[0].tick_params(axis="x", rotation=18)
    lineage = documents["compilation_lineage"].str.extract(r"compilation(30|200)")[0]
    lineage_counts = lineage.map({"30": "30-doc lineage", "200": "200-doc lineage"}).value_counts()
    axes[1].bar(lineage_counts.index, lineage_counts.values, color=["#5C6BC0", "#26A69A"])
    axes[1].bar_label(axes[1].containers[0])
    axes[1].set(title="Compilation lineage", ylabel="Documents")
    carrier = documents["carrier_family"].map(
        lambda value: "Other / NVOCC" if str(value).startswith("OTHER::") else value
    )
    carrier_counts = carrier.value_counts().head(10).sort_values()
    axes[2].barh(carrier_counts.index, carrier_counts.values, color="#546E7A")
    axes[2].set(title="Carrier-family coverage", xlabel="Documents")
    figure.suptitle("Cohort composition across both joined template lineages")
    output["plots/02_cohort_composition.png"] = _figure_bytes(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(
        documents["deterministic_slot_fraction"] * 100,
        bins=15,
        color=_PALETTE["deterministic"],
        ax=axes[0],
    )
    axes[0].axvline(
        float(summary["routing"]["deterministicSlotFraction"]) * 100,
        color="#263238",
        linestyle="--",
        label="slot-weighted mean",
    )
    axes[0].legend()
    axes[0].set(title="Per-document deterministic coverage", xlabel="Deterministic slots (%)")
    totals = [
        int(summary["routing"]["deterministicSlots"]),
        int(summary["routing"]["agentSlots"]),
    ]
    axes[1].pie(
        totals,
        labels=["Deterministic", "Residual agent"],
        autopct="%1.2f%%",
        colors=[_PALETTE["deterministic"], _PALETTE["agent"]],
        startangle=90,
    )
    axes[1].set_title(f"All {sum(totals):,} compiled slots")
    figure.suptitle("Compiled templates make insertion overwhelmingly deterministic")
    output["plots/03_slot_routing.png"] = _figure_bytes(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(
        data=documents,
        x="target_origin_label",
        y="changed_slot_fraction",
        hue="target_origin_label",
        palette=origin_palette,
        legend=False,
        ax=axes[0],
    )
    axes[0].set(title="Rendered slot-change scope", xlabel="", ylabel="Changed slots / all slots")
    axes[0].tick_params(axis="x", rotation=15)
    sns.boxplot(
        data=documents,
        x="target_origin_label",
        y="final_changed_target_leaves",
        hue="target_origin_label",
        palette=origin_palette,
        legend=False,
        ax=axes[1],
    )
    axes[1].set(title="Final structured-label changes", xlabel="", ylabel="Changed leaves")
    axes[1].tick_params(axis="x", rotation=15)
    figure.suptitle("Full linguistic targets exercise substantially broader edits")
    output["plots/04_change_scope.png"] = _figure_bytes(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, column, label in (
        (axes[0, 0], "template_slot_count", "Template slots"),
        (axes[0, 1], "template_binding_count", "Template bindings"),
        (axes[1, 0], "source_lines", "Source lines"),
        (axes[1, 1], "pages", "Pages"),
    ):
        sns.histplot(
            data=documents,
            x=column,
            hue="target_origin_label",
            palette=origin_palette,
            multiple="layer",
            bins=18,
            ax=axis,
            legend=axis is axes[0, 0],
        )
        axis.set(xlabel=label, ylabel="Documents")
    legend = axes[0, 0].get_legend()
    if legend is not None:
        legend.set_title("Target origin")
    figure.suptitle("Template complexity spans compact one-page forms to 12-page documents")
    output["plots/05_template_complexity.png"] = _figure_bytes(figure)

    correlation_columns = {
        "template_slot_count": "slots",
        "template_binding_count": "bindings",
        "source_lines": "lines",
        "pages": "pages",
        "changed_slot_count": "changed slots",
        "agent_slot_count": "agent slots",
        "source_provider_cost_usd": "cost",
        "source_provider_duration_seconds": "provider latency",
    }
    correlations = documents[list(correlation_columns)].rename(columns=correlation_columns).corr()
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axis)
    axis.set_title("Complexity, residual work, cost, and latency correlations")
    output["plots/06_complexity_correlations.png"] = _figure_bytes(figure)

    residual = documents[documents["source_provider_requests"] > 0].sort_values(
        "source_provider_cost_usd", ascending=False
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(residual["agent_slot_count"], bins=14, color=_PALETTE["agent"], ax=axes[0])
    axes[0].set(title="Current residual scope in historically paid documents", xlabel="Agent slots")
    cumulative = (
        residual["source_provider_cost_usd"].cumsum() / residual["source_provider_cost_usd"].sum()
    )
    axes[1].plot(range(1, len(residual) + 1), cumulative * 100, color=_PALETTE["agent"])
    axes[1].axhline(80, linestyle="--", color="#455A64")
    axes[1].set(
        title="Historical residual-cost concentration",
        xlabel="Documents ranked by cost",
        ylabel="Cumulative cost (%)",
        ylim=(0, 105),
    )
    figure.suptitle("Residual-agent work is sparse and cost is concentrated")
    output["plots/07_residual_cost_distribution.png"] = _figure_bytes(figure)

    agent_routes = routes[routes["runtime_route"] == "agent"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    group = agent_routes.groupby("group_kind")["slot_count"].sum().sort_values()
    axes[0].barh(group.index, group.values, color=_PALETTE["agent"])
    axes[0].set(title="Agent-routed slots by semantic group", xlabel="Slots")
    value = agent_routes.groupby("value_kind")["slot_count"].sum().sort_values()
    axes[1].barh(value.index, value.values, color="#FFB74D")
    axes[1].set(title="Agent-routed slots by value kind", xlabel="Slots")
    figure.suptitle("What remains outside deterministic rendering")
    output["plots/08_residual_semantics.png"] = _figure_bytes(figure)

    transition = pd.crosstab(routes["compiled_mode"], routes["runtime_route"])
    transition = transition.reindex(columns=["deterministic", "agent"], fill_value=0)
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.heatmap(transition, annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set(title="Compiler mode to runtime route", xlabel="Runtime route", ylabel="Compiled mode")
    output["plots/09_route_transition_matrix.png"] = _figure_bytes(figure)

    topology = documents.melt(
        id_vars=["target_origin_label"],
        value_vars=["containers", "cargo_groups", "cargo_packages", "party_roles"],
        var_name="surface",
        value_name="count",
    )
    figure, axis = plt.subplots(figsize=(12, 5.5))
    sns.boxplot(
        data=topology,
        x="surface",
        y="count",
        hue="target_origin_label",
        palette=origin_palette,
        showfliers=True,
        ax=axis,
    )
    axis.set(title="Structured target topology", xlabel="", ylabel="Entities per document")
    axis.legend(title="Target origin", fontsize=8)
    output["plots/10_target_topology.png"] = _figure_bytes(figure)

    change_plot = target_changes.merge(
        documents[["document_id", "target_origin_label"]], on="document_id", how="left"
    )
    change_counts = (
        change_plot.groupby(["family", "target_origin_label"]).size().unstack(fill_value=0)
    )
    change_counts = change_counts.loc[change_counts.sum(axis=1).sort_values().index]
    figure, axis = plt.subplots(figsize=(11, 6))
    change_counts.plot(kind="barh", stacked=True, color=origin_palette, ax=axis)
    axis.set(
        title="Final structured-label changes by semantic family",
        xlabel="Changed leaves",
        ylabel="",
    )
    axis.legend(title="Target origin", fontsize=8)
    output["plots/11_target_change_families.png"] = _figure_bytes(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    category = adaptations["category"].value_counts().sort_values()
    axes[0].barh(category.index, category.values, color="#7E57C2")
    axes[0].set(title="Adaptation reason", xlabel="Adaptations")
    families = adaptations["target_family"].value_counts().sort_values()
    axes[1].barh(families.index, families.values, color="#AB47BC")
    axes[1].set(title="Adapted target family", xlabel="Adaptations")
    figure.suptitle("Compatibility adaptation keeps targets representable by immutable layouts")
    output["plots/12_compatibility_adaptations.png"] = _figure_bytes(figure)

    efficiency = cast(Mapping[str, Any], summary["efficiency"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = ["Replay-attributed route", "Current planned route"]
    calls = [int(efficiency["realizedProviderRequests"]), int(efficiency["plannedFreshRequests"])]
    bars = axes[0].bar(labels, calls, color=["#90A4AE", _PALETTE["agent"]])
    axes[0].bar_label(bars)
    axes[0].set(title="Provider calls", ylabel="Calls")
    costs = [
        float(efficiency["realizedResidualCostUsd"]),
        float(efficiency["conservativeFreshCostProxyUsd"]),
    ]
    bars = axes[1].bar(labels, costs, color=["#90A4AE", _PALETTE["agent"]])
    axes[1].bar_label(bars, fmt="$%.5f")
    axes[1].set(title="Residual-rendering cost", ylabel="USD")
    figure.suptitle("Current residual route and conservative historical-cost proxy")
    output["plots/13_cost_efficiency.png"] = _figure_bytes(figure)

    comparable = country_screen[country_screen["comparable"] == True]  # noqa: E712
    country_counts = pd.Series(
        {
            "Consistent": int((comparable["consistent"] == True).sum()),  # noqa: E712
            "Mismatch": int((comparable["consistent"] == False).sum()),  # noqa: E712
            "Not comparable": len(country_screen) - len(comparable),
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(
        country_counts.index,
        country_counts.values,
        color=[_PALETTE["pass"], _PALETTE["concern"], "#90A4AE"],
    )
    axes[0].bar_label(bars)
    axes[0].set(title="Exporter-country semantic relationship screen", ylabel="Documents")
    mismatch_origin = (
        comparable[comparable["consistent"] == False][  # noqa: E712
            "target_origin"
        ]
        .map(_ORIGIN_LABELS)
        .value_counts()
    )
    if mismatch_origin.empty:
        axes[1].text(0.5, 0.5, "No mismatches", ha="center", va="center", fontsize=14)
        axes[1].set_xticks([])
    else:
        axes[1].bar(
            mismatch_origin.index,
            mismatch_origin.values,
            color=[origin_palette[x] for x in mismatch_origin.index],
        )
        axes[1].bar_label(axes[1].containers[0])
    axes[1].set(title="Country mismatches by target origin", ylabel="Documents")
    axes[1].tick_params(axis="x", rotation=15)
    figure.suptitle("Exporter countries are checked against their compiled entity relationship")
    output["plots/14_country_consistency_screen.png"] = _figure_bytes(figure)

    contradiction_docs = contradictions.drop_duplicates("document_id")
    flag_counts = pd.Series(
        {
            "No clear count contradiction": len(documents) - len(contradiction_docs),
            "Flagged": len(contradiction_docs),
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(
        flag_counts.index,
        flag_counts.values,
        color=[_PALETTE["pass"], _PALETTE["concern"]],
    )
    axes[0].bar_label(bars)
    axes[0].set(title="Clear number-word contradictions", ylabel="Documents")
    kind = (
        contradictions["kind"].value_counts().sort_values()
        if not contradictions.empty
        else pd.Series(dtype=int)
    )
    kind = kind.rename(
        index={
            "words_parenthetical_digits_disagree": "Words vs digits disagree",
            "sequence_exceeds_total": "Sequence exceeds total",
        }
    )
    if kind.empty:
        axes[1].text(0.5, 0.5, "No flagged occurrences", ha="center", va="center", fontsize=14)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    else:
        axes[1].barh(kind.index, kind.values, color=_PALETTE["concern"])
    axes[1].set(title="Flagged occurrences by rule", xlabel="Occurrences")
    figure.suptitle("Host-local format validity does not imply phrase-level semantic agreement")
    output["plots/15_number_consistency_screen.png"] = _figure_bytes(figure)

    manual_dimensions = pd.Series(
        {
            "Mechanical fidelity": int((manual["mechanical_fidelity"] == "pass").sum()),
            "Target-binding fidelity": int((manual["target_binding_fidelity"] == "pass").sum()),
            "Carrier preservation": int((manual["carrier_preservation"] == "pass").sum()),
            "Layout preservation": int((manual["layout_preservation"] == "pass").sum()),
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].barh(manual_dimensions.index, manual_dimensions.values, color=_PALETTE["pass"])
    axes[0].bar_label(bars)
    axes[0].set(
        title="Reviewed mechanical dimensions",
        xlabel="Passing documents",
        xlim=(0, len(manual) * 1.08),
    )
    coherence = manual["whole_document_coherence"].value_counts()
    coherence_labels = {
        "concern": "Concern",
        "pass_with_source_limitation": "Pass with source limitation",
        "pass": "Pass",
    }
    bars = axes[1].bar(
        [coherence_labels[value] for value in coherence.index],
        coherence.values,
        color=[_PALETTE.get(value, "#546E7A") for value in coherence.index],
    )
    axes[1].bar_label(bars)
    axes[1].set(title="Whole-document review", ylabel="Documents")
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle(f"Purposive manual stress review (n={len(manual)}; not a population estimate)")
    output["plots/16_manual_stress_review.png"] = _figure_bytes(figure)

    figure, axis = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=documents,
        x="template_slot_count",
        y="changed_slot_count",
        size="agent_slot_count",
        sizes=(30, 450),
        hue="target_origin_label",
        palette=origin_palette,
        alpha=0.8,
        ax=axis,
    )
    axis.set(
        title="Complexity and edit-scope outliers (bubble size = residual-agent slots)",
        xlabel="Template slots",
        ylabel="Changed slots",
    )
    axis.legend(title="Target origin / agent slots", fontsize=8)
    output["plots/17_complexity_outliers.png"] = _figure_bytes(figure)
    return output


def _summary(
    *,
    documents: pd.DataFrame,
    routes: pd.DataFrame,
    adaptations: pd.DataFrame,
    target_changes: pd.DataFrame,
    country_screen: pd.DataFrame,
    contradictions: pd.DataFrame,
    manual: pd.DataFrame,
    run_summary: Mapping[str, Any],
    selection_method: str,
) -> dict[str, Any]:
    document_count = len(documents)
    deterministic_slots = int(documents["deterministic_slot_count"].sum())
    agent_slots = int(documents["agent_slot_count"].sum())
    template_slots = int(documents["template_slot_count"].sum())
    realized_cost = Decimal(str(documents["source_provider_cost_usd"].sum()))
    fresh_proxy_cost = Decimal(str(documents["current_route_cost_proxy_usd"].sum()))
    realized_requests = int(documents["source_provider_requests"].sum())
    planned_requests = int(documents["current_provider_required"].sum())
    comparable = country_screen[country_screen["comparable"] == True]  # noqa: E712
    country_matches = int((comparable["consistent"] == True).sum())  # noqa: E712
    country_mismatches = int((comparable["consistent"] == False).sum())  # noqa: E712
    contradiction_documents = int(contradictions["document_id"].nunique())
    concern_count = int((manual["whole_document_coherence"] == "concern").sum())
    source_limit_count = int(
        (manual["whole_document_coherence"] == "pass_with_source_limitation").sum()
    )
    manual_pass_count = int((manual["whole_document_coherence"] == "pass").sum())
    all_invariants = bool(
        (documents["status"] == "passed").all()
        and all(bool(documents[column].all()) for column in _ACCEPTANCE_COLUMNS)
    )
    scale_ready = bool(
        all_invariants
        and country_mismatches == 0
        and contradiction_documents == 0
        and concern_count == 0
    )
    per_document_proxy = fresh_proxy_cost / document_count
    return {
        "schemaVersion": 1,
        "status": "complete",
        "decision": (
            "compiled_semantic_insertion_validated"
            if scale_ready
            else "semantic_acceptance_gate_not_met"
        ),
        "scaleReady": scale_ready,
        "documents": document_count,
        "passedDocuments": int((documents["status"] == "passed").sum()),
        "allMachineAcceptanceInvariantsPassed": all_invariants,
        "targetOrigins": {
            key: int(value)
            for key, value in documents["target_origin"].value_counts().sort_index().items()
        },
        "compilationLineages": {
            key: int(value)
            for key, value in documents["compilation_lineage"].value_counts().sort_index().items()
        },
        "routing": {
            "templateBindings": int(documents["template_binding_count"].sum()),
            "templateSlots": template_slots,
            "deterministicBindings": int(documents["deterministic_binding_count"].sum()),
            "agentBindings": int(documents["agent_binding_count"].sum()),
            "deterministicSlots": deterministic_slots,
            "agentSlots": agent_slots,
            "deterministicSlotFraction": deterministic_slots / template_slots,
            "providerFreeDocuments": int((documents["agent_binding_count"] == 0).sum()),
            "residualDocuments": planned_requests,
            "sourceResidualSlots": int(documents["source_residual_slots"].sum()),
            "replayedResidualSlots": int(documents["replayed_residual_slots"].sum()),
            "droppedResidualSlots": int(documents["dropped_residual_slots"].sum()),
            "agentRouteGroups": {
                key: int(value)
                for key, value in routes[routes["runtime_route"] == "agent"]
                .groupby("group_kind")["slot_count"]
                .sum()
                .sort_index()
                .items()
            },
        },
        "changeScope": {
            "changedSlots": int(documents["changed_slot_count"].sum()),
            "changedSlotFraction": float(documents["changed_slot_count"].sum() / template_slots),
            "proposedChangedTargetLeaves": int(documents["changed_target_leaf_count"].sum()),
            "finalChangedTargetLeaves": len(target_changes),
            "finalChangeFamilies": {
                key: int(value)
                for key, value in target_changes["family"].value_counts().sort_index().items()
            },
            "compatibilityAdaptations": len(adaptations),
            "adaptationCategories": {
                key: int(value)
                for key, value in adaptations["category"].value_counts().sort_index().items()
            },
        },
        "efficiency": {
            "realizedProviderRequests": realized_requests,
            "plannedFreshRequests": planned_requests,
            "retiredCompleteProviderRequests": realized_requests - planned_requests,
            "realizedResidualCostUsd": f"{realized_cost:.12f}",
            "conservativeFreshCostProxyUsd": f"{fresh_proxy_cost:.12f}",
            "conservativeFreshCostProxyPerDocumentUsd": f"{per_document_proxy:.12f}",
            "realizedCostPerDocumentUsd": f"{realized_cost / document_count:.12f}",
            "freshProxyReductionPercent": float(
                (Decimal(1) - fresh_proxy_cost / realized_cost) * 100
            ),
            "projectedResidualCost10kUsd": f"{per_document_proxy * 10_000:.6f}",
            "projectedResidualCost50kUsd": f"{per_document_proxy * 50_000:.6f}",
            "scope": (
                "Post-compilation residual rendering only. Template compilation and any "
                "separate model-based full-target synthesis are excluded. The fresh cost is "
                "conservative because partially reduced residual prompts retain their paid "
                "historical whole-document cost."
            ),
        },
        "deterministicQualityScreens": {
            "documentsWithExporterCountrySurface": len(country_screen),
            "relationshipComparableExporterCountryDocuments": len(comparable),
            "relationshipConsistentExporterCountryDocuments": country_matches,
            "relationshipMismatchedExporterCountryDocuments": country_mismatches,
            "mismatchFractionAmongComparable": (
                country_mismatches / len(comparable) if len(comparable) else 0.0
            ),
            "clearNumberContradictionDocuments": contradiction_documents,
            "clearNumberContradictionOccurrences": len(contradictions),
            "interpretation": (
                "Exporter countries are compared with the target party only when compilation "
                "declares the same entity; independent exporters are checked for internal "
                "country consistency instead. Count screens remain conservative lexical guards."
            ),
        },
        "manualReview": {
            "documents": len(manual),
            "selectionMethod": selection_method,
            "mechanicalPasses": int((manual["mechanical_fidelity"] == "pass").sum()),
            "targetBindingPasses": int((manual["target_binding_fidelity"] == "pass").sum()),
            "layoutPasses": int((manual["layout_preservation"] == "pass").sum()),
            "carrierPasses": int((manual["carrier_preservation"] == "pass").sum()),
            "wholeDocumentPasses": manual_pass_count,
            "wholeDocumentPassesWithSourceLimitation": source_limit_count,
            "wholeDocumentConcerns": concern_count,
            "populationEstimate": False,
        },
        "distributions": {
            "templateSlots": _describe(documents["template_slot_count"]),
            "templateBindings": _describe(documents["template_binding_count"]),
            "sourceLines": _describe(documents["source_lines"]),
            "pages": _describe(documents["pages"]),
            "deterministicSlotFraction": _describe(documents["deterministic_slot_fraction"]),
            "changedSlotFraction": _describe(documents["changed_slot_fraction"]),
            "agentSlots": _describe(documents["agent_slot_count"]),
            "historicalResidualCostUsd": _describe(documents["source_provider_cost_usd"]),
        },
        "trainingRecordsPresent": int(run_summary["trainingRecords"]),
        "recommendedUse": (
            "validated_for_configured_compiled_template_synthesis"
            if scale_ready
            else "test_only_until_semantic_acceptance_gate_passes"
        ),
    }


def _report(summary: Mapping[str, Any]) -> str:
    routing = cast(Mapping[str, Any], summary["routing"])
    efficiency = cast(Mapping[str, Any], summary["efficiency"])
    screens = cast(Mapping[str, Any], summary["deterministicQualityScreens"])
    manual = cast(Mapping[str, Any], summary["manualReview"])
    changes = cast(Mapping[str, Any], summary["changeScope"])
    target_origins = cast(Mapping[str, Any], summary["targetOrigins"])
    existing_target_count = int(
        target_origins.get("existing_linguistic_target_carrier_restored", 0)
    )
    controlled_variant_count = int(target_origins.get("controlled_source_variant", 0))
    return "\n".join(
        (
            f"# Compiled-template {summary['documents']}-document synthesis EDA",
            "",
            "## Outcome",
            "",
            (
                f"- Machine acceptance: **{summary['passedDocuments']}/{summary['documents']}**; "
                "carrier, topology, literal bytes, page markers, line endings, format "
                "envelopes, target semantics, source relationships, and compiled formal "
                "coherence all passed for every document."
            ),
            (
                f"- Routing: **{routing['deterministicSlots']:,}/{routing['templateSlots']:,} "
                f"slots ({routing['deterministicSlotFraction']:.3%}) deterministic**; "
                f"**{routing['agentSlots']} slots** across **{routing['residualDocuments']} "
                "documents** still use the residual renderer."
            ),
            (
                f"- Edit scope: **{changes['changedSlots']:,} slots** and "
                f"**{changes['finalChangedTargetLeaves']:,} final structured leaves** changed."
            ),
            "",
            "## Cost and throughput interpretation",
            "",
            (
                "- The replay-attributed route contains "
                f"**{efficiency['realizedProviderRequests']} calls** and "
                f"**${efficiency['realizedResidualCostUsd']}**. The current route requires "
                f"**{efficiency['plannedFreshRequests']} calls**."
            ),
            (
                "- Reusing the historical prices of only the documents that still need a call "
                "gives a conservative fresh-run proxy of "
                f"**${efficiency['conservativeFreshCostProxyUsd']} "
                f"total** (**${efficiency['conservativeFreshCostProxyPerDocumentUsd']}/document**)."
            ),
            (
                f"- At that proxy rate: **${efficiency['projectedResidualCost10kUsd']} for "
                f"10,000** and **${efficiency['projectedResidualCost50kUsd']} for 50,000** "
                "post-compilation insertions."
            ),
            (
                "- Scope caveat: those figures exclude one-time template compilation and any "
                "separate model-based full-target synthesis. This cohort used "
                f"{existing_target_count} pre-existing full linguistic targets and "
                f"{controlled_variant_count} deterministic controlled source variants."
            ),
            "",
            "## Deep quality finding",
            "",
            (
                "The schema-v6 compiler contract and renderer passed the configured global "
                "semantic acceptance gates."
                if summary["scaleReady"]
                else "One or more configured global semantic acceptance gates did not pass."
            ),
            (
                "- Exporter-country text was found in "
                f"**{screens['documentsWithExporterCountrySurface']} "
                f"documents**. Of **{screens['relationshipComparableExporterCountryDocuments']}** "
                "with resolvable compiled relationships and rendered exporter countries, "
                f"**{screens['relationshipMismatchedExporterCountryDocuments']} disagreed** "
                f"({screens['mismatchFractionAmongComparable']:.1%})."
            ),
            (
                "- A conservative number-word screen found "
                f"**{screens['clearNumberContradictionDocuments']} "
                "documents** with clear contradictions such as `ZERO (2)` or a sequence larger "
                "than its stated total."
            ),
            (
                f"- Purposive stress review: **{manual['mechanicalPasses']}/{manual['documents']} "
                "mechanical passes**, but **"
                f"{manual['wholeDocumentConcerns']}/{manual['documents']} "
                "whole-document concerns**; "
                "this deliberately difficult review is not a population estimate."
            ),
            "",
            "## Decision",
            "",
            (
                "The scale-readiness decision above is computed from machine invariants, "
                "relationship-aware exporter checks, count/sequence contradictions, and the "
                "pinned purposive manual review. Failed or review-required cases remain excluded "
                "from training publication."
            ),
            "",
            "## Artifact map",
            "",
            "- `data/documents.csv`: one row per synthesized document.",
            "- `data/routes.csv`: every compiled binding and runtime route.",
            "- `data/target-changes.csv`: every final structured-label change.",
            "- `data/adaptations.csv`: every target compatibility adaptation.",
            "- `data/country-consistency-screen.csv`: relationship-aware exporter screen.",
            "- `data/number-contradictions.csv`: conservative phrase-level count screen.",
            "- `data/manual-review.csv`: pinned purposive manual reviews with evidence lines.",
            "- `data/outliers.csv`: top-five documents across eleven stress dimensions.",
            "- `plots/`: seventeen deterministic PNG figures.",
            "",
        )
    )


def analyze_descendant_run(*, project_root: Path, config_path: Path) -> Path:
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("EDA configuration is outside the project root") from error
    config = load_eda_config(config_path)
    manual_path = (project_root / config.manual_review.path).resolve(strict=True)
    iso_path = (project_root / config.iso3166_snapshot.path).resolve(strict=True)
    for path, expected, label in (
        (manual_path, config.manual_review.sha256, "manual review"),
        (iso_path, config.iso3166_snapshot.sha256, "ISO-3166 snapshot"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular file")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 differs from configuration")

    run_root = _validate_committed_run(project_root, config.descendant_run)
    catalog_root = _validate_committed_run(project_root, config.template_catalog)
    manual_review = _load_manual_review(manual_path)
    if manual_review.run != config.descendant_run:
        raise ValueError("manual review pins a different descendant run")
    if manual_review.template_catalog != config.template_catalog:
        raise ValueError("manual review pins a different template catalog")
    if len(manual_review.reviews) != config.expected_manual_reviews:
        raise ValueError("manual review count differs from configuration")

    run_summary = _json(run_root / "summary.json")
    if run_summary.get("status") != "passed":
        raise ValueError("EDA input descendant run did not pass")
    if run_summary.get("documents") != config.expected_documents:
        raise ValueError("descendant summary count differs from EDA configuration")
    raw_results = _jsonl(run_root / "results.jsonl")
    results = tuple(
        DescendantCaseResult.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in raw_results
    )
    document_ids = tuple(row.document_id for row in results)
    if len(results) != config.expected_documents or len(set(document_ids)) != len(document_ids):
        raise ValueError("descendant results are incomplete or contain duplicate documents")
    if any(row.status != "passed" for row in results):
        raise ValueError("descendant results contain a non-passing document")
    dataset_records = _jsonl(run_root / "dataset" / "records.jsonl")
    dataset_lineage = _jsonl(run_root / "dataset" / "lineage.jsonl")
    if len(dataset_records) != len(results) or len(dataset_lineage) != len(results):
        raise ValueError("published dataset does not cover every descendant result")

    registry = load_iso_country_registry(
        iso_path=iso_path,
        iso_sha256=config.iso3166_snapshot.sha256,
    )
    frames = _build_frames(
        run_root=run_root,
        catalog_root=catalog_root,
        results=results,
        registry=registry,
    )
    documents = frames["documents"]
    manual = _manual_frame(manual_review)
    if set(manual["document_id"]) - set(document_ids):
        raise ValueError("manual review contains a document outside the descendant run")
    document_index = documents.set_index("document_id")
    for review in manual_review.reviews:
        document = document_index.loc[review.document_id]
        if review.target_origin != document["target_origin"]:
            raise ValueError(f"manual target origin differs: {review.document_id}")
        if review.compilation_lineage != document["compilation_lineage"]:
            raise ValueError(f"manual compilation lineage differs: {review.document_id}")
        rendered_lines = read_regular_file_bytes(
            run_root / "cases" / review.document_id / "rendered.txt"
        ).splitlines()
        if any(line > len(rendered_lines) for line in review.evidence_lines):
            raise ValueError(f"manual evidence line is outside rendered text: {review.document_id}")

    summary = _summary(
        documents=documents,
        routes=frames["routes"],
        adaptations=frames["adaptations"],
        target_changes=frames["target_changes"],
        country_screen=frames["country_screen"],
        contradictions=frames["number_contradictions"],
        manual=manual,
        run_summary=run_summary,
        selection_method=manual_review.selection_method,
    )
    outliers = _outlier_frame(documents)
    transaction = {
        "schemaVersion": 1,
        "kind": "compiled_template_descendant_eda",
        "config": config.model_dump(mode="json"),
        "configSha256": sha256_file(config_path),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "manualReviewSha256": sha256_file(manual_path),
        "descendantSummarySha256": sha256_file(run_root / "summary.json"),
        "descendantResultsSha256": sha256_file(run_root / "results.jsonl"),
        "templateCatalogSha256": sha256_file(catalog_root / "catalog.jsonl"),
    }
    output_parent = (project_root / config.output_dir).resolve()
    if output_parent != project_root and project_root not in output_parent.parents:
        raise ValueError("EDA output directory escapes the project root")
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if staged.completed:
        staged.validate_committed_run()
        return staged.final_root
    staged.recover_interrupted_temporary_files()
    staged.publish_json("config.json", config.model_dump(mode="json"))
    staged.publish_json("transaction.json", transaction)
    staged.publish_json("summary.json", summary)
    staged.publish_bytes("REPORT.md", _report(summary).encode("utf-8"))
    staged.publish_bytes("inputs/manual-review.yaml", read_regular_file_bytes(manual_path))
    for name, frame in (
        ("documents", documents),
        ("routes", frames["routes"]),
        ("target-changes", frames["target_changes"]),
        ("adaptations", frames["adaptations"]),
        ("country-consistency-screen", frames["country_screen"]),
        ("number-contradictions", frames["number_contradictions"]),
        ("manual-review", manual),
        ("outliers", outliers),
    ):
        staged.publish_bytes(f"data/{name}.csv", _csv_bytes(frame))
    for relative_path, payload in _plots(frames=frames, manual=manual, summary=summary).items():
        staged.publish_bytes(relative_path, payload)
    expected_artifacts = tuple(
        path.relative_to(staged.stage_root).as_posix()
        for path in sorted(staged.stage_root.rglob("*"))
        if path.is_file() and path.name not in {"_TRANSACTION.json", "_COMMIT.json"}
    )
    staged.commit(
        expected_artifacts=expected_artifacts,
        metadata={
            "schemaVersion": 1,
            "kind": "compiled_template_descendant_eda",
            "documents": config.expected_documents,
            "passedDocuments": summary["passedDocuments"],
            "decision": summary["decision"],
            "scaleReady": summary["scaleReady"],
        },
    )
    return staged.final_root
