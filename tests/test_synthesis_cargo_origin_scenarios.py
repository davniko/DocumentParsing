from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.cargo_origin_scenarios import (
    CargoOriginScenarioError,
    build_cargo_origin_support,
    project_cargo_origin_target,
    sample_cargo_origin_projection,
)
from document_ocr.synthesis.country_registry import (
    CountryEntry,
    CountryRegistry,
    load_iso_country_registry,
)
from document_ocr.synthesis.generators import DeterministicStream


def _registry() -> CountryRegistry:
    entries = tuple(
        CountryEntry.model_validate(
            {"alpha2": code, "alpha3": alpha3, "numeric": numeric, "name": name},
            strict=True,
        )
        for code, alpha3, numeric, name in (
            ("BE", "BEL", "056", "Belgium"),
            ("CN", "CHN", "156", "China"),
            ("DE", "DEU", "276", "Germany"),
            ("FR", "FRA", "250", "France"),
        )
    )
    return CountryRegistry(
        entries=entries,
        observed_aliases={},
        iso_sha256="1" * 64,
        observed_aliases_sha256="2" * 64,
    )


def _target(shipper_country: str | None, origins: list[dict[str, str] | None]) -> dict[str, Any]:
    return {
        "documentPatch": {
            "parties": (
                {"shipper": {"country": shipper_country}} if shipper_country is not None else {}
            ),
            "cargoGroups": [
                {"description": f"GROUP {index}", **({"origin": origin} if origin else {})}
                for index, origin in enumerate(origins)
            ],
        }
    }


def _support_targets() -> dict[str, dict[str, Any]]:
    return {
        "same": _target("Belgium", [{"name": "Belgium"}]),
        "different-fr": _target("Belgium", [{"name": "France"}]),
        "different-de": _target("France", [{"name": "Germany"}]),
        "unresolved-name": _target("Belgium", [{"name": "Brussels"}]),
        "identifier": _target("Belgium", [{"identifier": "BEANR"}]),
        "unpaired": _target(None, [{"name": "China"}]),
        "absent": _target("Belgium", [None]),
    }


def test_restricted_fit_balances_explicit_exclusions_and_isolates_scope() -> None:
    targets = _support_targets()
    support = build_cargo_origin_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        country_registry=_registry(),
    )

    assert support.audit.cargo_groups == 7
    assert support.audit.origin_present_groups == 6
    assert support.audit.origin_absent_groups == 1
    assert support.audit.generatable_country_name_groups == 5
    assert support.audit.fit_eligible_groups == 3
    assert support.audit.same_country_groups == 1
    assert support.audit.different_country_groups == 2
    assert support.audit.excluded_identifier_present_groups == 1
    assert support.audit.excluded_unresolved_origin_name_groups == 1
    assert support.audit.excluded_unresolved_shipper_country_groups == 1
    assert [(row.value, row.count) for row in support.relation_weights] == [
        ("different_country", 2),
        ("same_country", 1),
    ]
    assert [(row.value, row.count) for row in support.different_country_weights] == [
        ("DE", 1),
        ("FR", 1),
    ]
    assert {(row.document_id, row.reason) for row in support.exclusions} == {
        ("identifier", "identifier_present"),
        ("unpaired", "unresolved_shipper_country"),
        ("unresolved-name", "unresolved_origin_name"),
    }

    isolated = build_cargo_origin_support(
        source_targets={**targets, "outside": _target("China", [{"name": "China"}])},
        fit_document_ids=tuple(targets),
        country_registry=_registry(),
    )
    assert isolated.audit == support.audit
    assert isolated.relation_weights == support.relation_weights
    assert isolated.different_country_weights == support.different_country_weights


def test_projection_is_deterministic_name_only_and_preserves_source_group_presence() -> None:
    targets = _support_targets()
    registry = _registry()
    support = build_cargo_origin_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        country_registry=registry,
    )
    source = _target(None, [None, {"name": "Brussels"}, None, {"name": "Belgium"}])
    first = sample_cargo_origin_projection(
        source_target=source,
        support=support,
        country_registry=registry,
        commercial_origin_country_code="FR",
        stream=DeterministicStream(42, "cargo-origin-test", "doc"),
    )
    second = sample_cargo_origin_projection(
        source_target=source,
        support=support,
        country_registry=registry,
        commercial_origin_country_code="FR",
        stream=DeterministicStream(42, "cargo-origin-test", "doc"),
    )
    assert first == second
    assert tuple(row.group_index for row in first.origins) == (1, 3)
    for row in first.origins:
        assert registry.entry(row.country_code).name == row.country_name
        if row.relation_to_commercial_origin == "different_country":
            assert row.country_code != "FR"

    different = sample_cargo_origin_projection(
        source_target=_target(None, [{"name": "China"}]),
        support=support,
        country_registry=registry,
        commercial_origin_country_code="FR",
        stream=DeterministicStream(0, "cargo-origin-test", "diff"),
    ).origins[0]
    assert different.relation_to_commercial_origin == "different_country"
    assert different.country_code == "DE"
    assert different.country_code != "FR"

    projected = project_cargo_origin_target(source_target=source, projection=first)
    groups = projected["documentPatch"]["cargoGroups"]
    assert ["origin" in group for group in groups] == [False, True, False, True]
    assert all(set(groups[index]["origin"]) == {"name"} for index in (1, 3))
    assert source["documentPatch"]["cargoGroups"][1]["origin"] == {"name": "Brussels"}


@pytest.mark.parametrize(
    ("origin", "reason"),
    (
        ({"identifier": "BEANR"}, "identifier_present"),
        ({"other": "value"}, "missing_origin_name"),
        ({"name": "China", "identifier": "CNTAO"}, "identifier_present"),
    ),
)
def test_identifier_or_missing_name_origin_forms_fail_loudly_without_preservation(
    origin: dict[str, str], reason: str
) -> None:
    targets = _support_targets()
    registry = _registry()
    support = build_cargo_origin_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        country_registry=registry,
    )
    with pytest.raises(CargoOriginScenarioError, match=reason):
        sample_cargo_origin_projection(
            source_target=_target("Belgium", [origin]),
            support=support,
            country_registry=registry,
            commercial_origin_country_code="BE",
            stream=DeterministicStream(0, "cargo-origin-test", "rejected"),
        )


def test_alias_bearing_registry_is_rejected() -> None:
    base = _registry()
    aliases = CountryRegistry(
        entries=(base.entry(code) for code in base.country_codes),
        observed_aliases={"CHINA P.R.": "CN"},
        iso_sha256="1" * 64,
        observed_aliases_sha256="3" * 64,
    )
    with pytest.raises(ValueError, match="alias-free ISO country registry"):
        build_cargo_origin_support(
            source_targets=_support_targets(),
            fit_document_ids=tuple(_support_targets()),
            country_registry=aliases,
        )


def test_pinned_1157_corpus_probe_has_expected_restricted_contract_stats() -> None:
    root = Path(__file__).resolve().parents[1]
    records_path = (
        root
        / "artifacts/kie-training/datasets"
        / "mpci-bl-combined1157-task-facing-package-categories-v2"
        / "records.jsonl"
    )
    assert sha256_file(records_path) == (
        "2a3e2ea3231cfff7674e85b54a52f66d0fee98b59b207bc7b04fe0f9916dfc42"
    )
    source_targets: dict[str, Mapping[str, Any]] = {}
    for row in map(json.loads, records_path.read_text().splitlines()):
        source_targets[row["documentId"]] = row["target"]
    assert len(source_targets) == 1157
    registry = load_iso_country_registry(
        iso_path=root / "data/registries/countries/iso-codes-4.9.0-1/iso_3166-1.json",
        iso_sha256="f7dc5542a692ad8e23b9b85a6a1800a63f7a05e5246065fac1ede04ed209ce00",
    )
    support = build_cargo_origin_support(
        source_targets=source_targets,
        fit_document_ids=tuple(sorted(source_targets)),
        country_registry=registry,
    )
    assert support.audit.cargo_groups == 1360
    assert support.audit.origin_present_groups == 185
    assert support.audit.origin_absent_groups == 1175
    assert support.audit.generatable_country_name_groups == 175
    assert support.audit.fit_eligible_groups == 105
    assert support.audit.same_country_groups == 89
    assert support.audit.different_country_groups == 16
    assert support.audit.excluded_identifier_present_groups == 10
    assert support.audit.excluded_unresolved_origin_name_groups == 45
    assert support.audit.excluded_unresolved_shipper_country_groups == 25
    assert len(support.exclusions) == 80

    # The probe deliberately never projects an exclusion: the only generated
    # value form is canonical ISO country name, not a copied source surface.
    excluded = next(row for row in support.exclusions if row.reason == "identifier_present")
    with pytest.raises(CargoOriginScenarioError, match="identifier_present"):
        sample_cargo_origin_projection(
            source_target=source_targets[excluded.document_id],
            support=support,
            country_registry=registry,
            commercial_origin_country_code="BE",
            stream=DeterministicStream(20260831, "cargo-origin-probe", excluded.document_id),
        )
