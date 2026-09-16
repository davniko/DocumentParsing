from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .models import ExtractionConfig, SelectionManifest, SelectionRow


def _source_label_carrier_name(source: Mapping[str, Any]) -> str | None:
    try:
        value = source["target"]["documentPatch"]["parties"]["carrier"]["name"]
    except (KeyError, TypeError):
        return None
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source target carrier name is empty")
    return value.strip()


_NON_REUSABLE_CARRIER_IDENTITY = re.compile(r"^(?:THE )?MASTER(?: OF\b|$)")
_NULL_CARRIER_IDENTITIES = frozenset(
    {
        "N A",
        "NA",
        "NIL",
        "NONE",
        "NOT APPLICABLE",
        "NOT AVAILABLE",
        "THE CARRIER",
        "UNKNOWN",
    }
)
_PLACEHOLDER_CARRIER_TOKENS = frozenset(
    {
        "A",
        "CARRIER",
        "CO",
        "COMPANY",
        "INC",
        "L L C",
        "LIMITED",
        "LTD",
        "NAME",
        "OF",
        "S",
        "THE",
    }
)


def _identity_key(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").upper()
    )
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", ascii_value).split())


def _target_country_identity_keys(source: Mapping[str, Any]) -> frozenset[str]:
    countries: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "country" and isinstance(child, str) and child.strip():
                    countries.add(_identity_key(child))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(source.get("target"))
    return frozenset(countries)


def _non_reusable_carrier_reason(*, source: Mapping[str, Any], name: str) -> str | None:
    identity = _identity_key(name)
    if _NON_REUSABLE_CARRIER_IDENTITY.match(identity):
        return "source_label_vessel_master_not_reusable_carrier"
    if identity in _target_country_identity_keys(source):
        return "source_label_country_role_not_carrier_principal"
    tokens = frozenset(identity.split())
    if identity in _NULL_CARRIER_IDENTITIES or (
        "NAME" in tokens and tokens <= _PLACEHOLDER_CARRIER_TOKENS
    ):
        return "source_label_placeholder_not_carrier_principal"
    return None


def carrier_resolution_classification(*, source: Mapping[str, Any]) -> str:
    carrier_name = _source_label_carrier_name(source)
    if carrier_name is not None:
        invalid_reason = _non_reusable_carrier_reason(source=source, name=carrier_name)
        if invalid_reason is not None:
            return invalid_reason
        return "source_label_present"
    raw = source.get("joinedRawText")
    if not isinstance(raw, str):
        raise ValueError("source record lacks joinedRawText")
    return "missing_unresolvable_without_external_enrichment"


def _tie(seed: int, document_id: str) -> int:
    payload = f"{seed}:{document_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _feature_tags(row: Mapping[str, Any]) -> set[str]:
    pages = int(row["page_count"])
    containers = int(row["container_count"])
    goods = int(row["goods_group_count"])
    packages = int(row["package_fact_count"])
    return {
        f"document_type:{row['document_type']}",
        f"page_bin:{'1' if pages == 1 else '2' if pages == 2 else '3+'}",
        f"container_bin:{'0' if containers == 0 else '1' if containers == 1 else '2+'}",
        f"goods_bin:{'0' if goods == 0 else '1' if goods == 1 else '2+'}",
        f"package_bin:{'0' if packages == 0 else '1' if packages == 1 else '2+'}",
        f"carrier_family:{row['carrier_family']}",
        f"template_proxy:{row['template_proxy_id']}",
        "dangerous_goods" if row["dangerous_goods_present"] else "not_dangerous_goods",
        "temperature" if row["temperature_present"] else "not_temperature",
        "multi_container" if row["multi_container"] else "not_multi_container",
        "multi_goods" if row["multi_goods"] else "not_multi_goods",
        "multi_package" if row["multi_package_level"] else "not_multi_package",
    }


def _greedy_cover(
    *,
    candidates: Sequence[str],
    tags_by_id: Mapping[str, set[str]],
    count: int,
    seed: int,
    initial_tags: set[str] | None = None,
) -> list[str]:
    remaining = set(candidates)
    selected: list[str] = []
    covered = set(initial_tags or ())
    frequencies: dict[str, int] = defaultdict(int)
    for document_id in candidates:
        for tag in tags_by_id[document_id]:
            frequencies[tag] += 1
    while remaining and len(selected) < count:

        def score(document_id: str) -> tuple[float, int]:
            novelty = sum(
                1.0 / max(1, frequencies[tag])
                for tag in tags_by_id[document_id]
                if tag not in covered
            )
            return (novelty, -_tie(seed, document_id))

        chosen = max(remaining, key=score)
        selected.append(chosen)
        covered.update(tags_by_id[chosen])
        remaining.remove(chosen)
    if len(selected) != count:
        raise ValueError(f"selection pool has only {len(selected)} of {count} required documents")
    return selected


def _carrier_anchor_ids(anchors: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row["document_id"])
        for row in anchors
        if row.get("patchable")
        and str(row.get("relation_target_path", "")).startswith("documentPatch.parties.carrier")
    }


def _selection_row(
    *,
    ordinal: int,
    document_id: str,
    basis: Sequence[str],
    source: Mapping[str, Any],
    feature: Mapping[str, Any],
) -> SelectionRow:
    carrier_name = _source_label_carrier_name(source)
    carrier_family = feature["carrier_family"]
    return SelectionRow.model_validate(
        {
            "ordinal": ordinal,
            "document_id": document_id,
            "selection_basis": tuple(basis),
            "source_sha256": source["joinedRawTextSha256"],
            "carrier_name": carrier_name,
            "carrier_family": None if carrier_family == "<MISSING>" else carrier_family,
            "template_proxy_id": feature["template_proxy_id"],
            "document_type": feature["document_type"],
            "page_count": feature["page_count"],
            "ocr_lines": feature["ocr_lines"],
            "ocr_characters": feature["ocr_characters"],
            "container_count": feature["container_count"],
            "cargo_group_count": feature["goods_group_count"],
            "package_count": feature["package_fact_count"],
            "dangerous_goods_count": feature["dangerous_goods_count"],
            "temperature_count": feature["temperature_setting_count"],
        }
    )


def build_selection_manifest(
    *,
    config: ExtractionConfig,
    source_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
) -> SelectionManifest:
    sources = {str(row["documentId"]): row for row in source_rows}
    features = {str(row["document_id"]): row for row in feature_rows}
    carrier_anchors = _carrier_anchor_ids(anchor_rows)
    carrier_classes = {
        document_id: carrier_resolution_classification(source=sources[document_id])
        for document_id in features
        if document_id in sources
    }
    eligible = {
        document_id
        for document_id in carrier_classes
        if carrier_classes[document_id] == "source_label_present"
        and document_id not in config.excluded_document_ids
    }

    def carrier_basis(document_id: str) -> str:
        return (
            "ocr_anchored_source_carrier"
            if document_id in carrier_anchors
            else "carrier_requires_ocr_resolution"
        )

    if config.pinned_document_ids:
        missing = sorted(set(config.pinned_document_ids) - eligible)
        if missing:
            raise ValueError(
                "pinned documents are absent or ineligible for carrier-bound extraction: "
                + ", ".join(missing)
            )
        ordered = list(config.pinned_document_ids)
        basis_by_id = {
            document_id: ("explicit_pinned_probe", carrier_basis(document_id))
            for document_id in ordered
        }
    else:
        pool = sorted(eligible)
        ordered = _greedy_cover(
            candidates=pool,
            tags_by_id={document_id: _feature_tags(features[document_id]) for document_id in pool},
            count=config.workflow.documents,
            seed=config.selection_seed,
        )
        basis_by_id = {
            document_id: (
                "production_stratified_template_compilation",
                carrier_basis(document_id),
            )
            for document_id in ordered
        }

    if len(ordered) != config.workflow.documents or len(set(ordered)) != len(ordered):
        raise ValueError("selection count or uniqueness differs from configured workflow")
    rows = tuple(
        _selection_row(
            ordinal=index,
            document_id=document_id,
            basis=basis_by_id[document_id],
            source=sources[document_id],
            feature=features[document_id],
        )
        for index, document_id in enumerate(ordered, start=1)
    )
    return SelectionManifest.model_validate(
        {
            "schema_version": 1,
            "phase": config.phase,
            "selection_seed": config.selection_seed,
            "source_corpus_sha256": config.inputs.source_corpus.sha256,
            "document_features_sha256": config.inputs.document_features.sha256,
            "rows": rows,
        }
    )
