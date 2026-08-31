from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import (
    UkGlobalTariffRegistry,
    compile_uk_global_tariff_registry,
    load_ukgt_source_pin,
)
from document_ocr.synthesis.hs_scenarios import (
    HsFitObservation,
    HsScenarioPolicy,
    build_hs_scenario_support,
    hs_fit_observations,
    sample_hs_scenario,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data/registries/hs/ukgt-v4.0.1590"
ON_DATE = date(2026, 8, 31)


@pytest.fixture(scope="module")
def registry() -> UkGlobalTariffRegistry:
    return compile_uk_global_tariff_registry(
        metadata_path=SOURCE_ROOT / "raw/metadata.csvw.json",
        report_path=SOURCE_ROOT / "raw/commodities-report.csv",
        source=load_ukgt_source_pin(SOURCE_ROOT / "source-manifest.json"),
    )


def _policy(
    *,
    observed: int,
    registry_wide: int,
    gb_extension: int,
) -> HsScenarioPolicy:
    return HsScenarioPolicy(
        schema_version=1,
        observed_chapter_mixture_permyriad=observed,
        registry_wide_mixture_permyriad=registry_wide,
        gb_tariff_extension_permyriad=gb_extension,
        observed_chapter_weighting="fit_document_count_v1",
        observed_chapter_hs6_weighting=("uniform_registry_hs6_within_selected_chapter_v1"),
        registry_hs6_weighting="uniform_registry_hs6_v1",
        gb_tariff_leaf_weighting="uniform_registry_leaves_v1",
    )


def _support(registry: UkGlobalTariffRegistry):
    observations = (
        HsFitObservation("doc-1", "row-1", "0101210000"),
        HsFitObservation("doc-1", "row-2", "010129"),
        HsFitObservation("doc-2", "row-3", "05010000"),
        HsFitObservation("doc-3", "row-4", "940540"),
    )
    return build_hs_scenario_support(
        fit_document_ids=("doc-1", "doc-2", "doc-3"),
        observations=observations,
        registry=registry,
        on_date=ON_DATE,
    )


def test_fit_projection_excludes_non_fit_rows_and_requires_unique_source_rows() -> None:
    rows = (
        {"cargo_group_value_id": "row-2", "document_id": "doc-2", "value": "05010000"},
        {"cargo_group_value_id": "row-1", "document_id": "doc-1", "value": "010121"},
        {"cargo_group_value_id": "held", "document_id": "doc-held", "value": "010129"},
    )

    observations = hs_fit_observations(
        fit_document_ids=("doc-2", "doc-1"),
        cargo_hs_code_rows=rows,
    )
    assert tuple(row.source_document_id for row in observations) == ("doc-1", "doc-2")
    assert tuple(row.semantic_code for row in observations) == ("010121", "05010000")

    with pytest.raises(ValueError, match="duplicate cargo HS row identity"):
        hs_fit_observations(
            fit_document_ids=("doc-1",),
            cargo_hs_code_rows=(rows[1], rows[1]),
        )


def test_support_uses_document_weighted_chapters_and_audits_unsupported_codes(
    registry: UkGlobalTariffRegistry,
) -> None:
    support = _support(registry)

    assert tuple(row.chapter_code for row in support.chapters) == ("01", "05")
    assert tuple(row.fit_document_count for row in support.chapters) == (1, 1)
    assert tuple(row.code for row in support.chapters[0].observed_hs6) == (
        "010121",
        "010129",
    )
    assert support.audit.input_source_rows == 4
    assert support.audit.supported_source_rows == 3
    assert support.audit.excluded_source_rows == 1
    assert support.audit.source_length_counts == ((6, 2), (8, 1), (10, 1))
    assert support.exclusions[0].semantic_code == "940540"
    assert support.exclusions[0].reason == "global_hs6_absent_from_pinned_edition"


def test_non_gb_routes_always_emit_exact_global_hs6(registry: UkGlobalTariffRegistry) -> None:
    support = _support(registry)
    policy = _policy(observed=10_000, registry_wide=0, gb_extension=10_000)

    scenarios = tuple(
        sample_hs_scenario(
            support=support,
            registry=registry,
            policy=policy,
            customs_jurisdiction="EG",
            stream=DeterministicStream(seed=5, namespace="hs", identity=f"sample-{index}"),
        )
        for index in range(100)
    )
    assert {row.component for row in scenarios} == {"fit_observed_chapter_registry_hs6"}
    assert {row.global_identity.chapter_code for row in scenarios} == {"01", "05"}
    assert {len(row.output_code) for row in scenarios} == {6}
    assert all(row.gb_tariff_identity is None for row in scenarios)
    assert all(
        row.dangerous_goods_status == "independent_not_inferred_from_hs" for row in scenarios
    )


def test_gb_extension_is_explicit_exact_and_never_odd_length(
    registry: UkGlobalTariffRegistry,
) -> None:
    support = _support(registry)
    always_gb = _policy(observed=0, registry_wide=10_000, gb_extension=10_000)
    never_gb = _policy(observed=0, registry_wide=10_000, gb_extension=0)

    extended = tuple(
        sample_hs_scenario(
            support=support,
            registry=registry,
            policy=always_gb,
            customs_jurisdiction="GB",
            stream=DeterministicStream(seed=8, namespace="hs", identity=f"gb-{index}"),
        )
        for index in range(50)
    )
    assert {len(row.output_code) for row in extended} == {10}
    assert all(row.gb_tariff_identity is not None for row in extended)
    assert all(row.output_code.startswith(row.global_identity.code) for row in extended)
    assert all(
        registry.require_uk(row.output_code, on_date=ON_DATE) == row.gb_tariff_identity
        for row in extended
    )

    global_only = sample_hs_scenario(
        support=support,
        registry=registry,
        policy=never_gb,
        customs_jurisdiction="GB",
        stream=DeterministicStream(seed=8, namespace="hs", identity="global-only"),
    )
    assert len(global_only.output_code) == 6
    assert global_only.gb_tariff_identity is None


def test_sampling_exclusions_produce_distinct_semantic_codes(
    registry: UkGlobalTariffRegistry,
) -> None:
    support = _support(registry)
    policy = _policy(observed=0, registry_wide=10_000, gb_extension=0)
    output_codes: set[str] = set()
    global_codes: set[str] = set()

    for index in range(25):
        scenario = sample_hs_scenario(
            support=support,
            registry=registry,
            policy=policy,
            customs_jurisdiction="EG",
            stream=DeterministicStream(seed=17, namespace="hs", identity=f"unique-{index}"),
            excluded_output_codes=output_codes,
            excluded_global_hs6=global_codes,
        )
        output_codes.add(scenario.output_code)
        global_codes.add(scenario.global_identity.code)

    assert len(output_codes) == 25
    assert len(global_codes) == 25


def test_mixture_and_sampling_are_deterministic_and_configured(
    registry: UkGlobalTariffRegistry,
) -> None:
    support = _support(registry)
    policy = _policy(observed=7_500, registry_wide=2_500, gb_extension=0)

    def run() -> Counter[str]:
        return Counter(
            sample_hs_scenario(
                support=support,
                registry=registry,
                policy=policy,
                customs_jurisdiction="EG",
                stream=DeterministicStream(
                    seed=99,
                    namespace="hs",
                    identity=f"mixture-{index}",
                ),
            ).component
            for index in range(2_000)
        )

    first = run()
    assert first == run()
    assert 1_400 <= first["fit_observed_chapter_registry_hs6"] <= 1_600
    assert 400 <= first["registry_wide_hs6_exploration"] <= 600


def test_hs_policy_rejects_incomplete_probability_contract() -> None:
    with pytest.raises(ValueError, match="sum exactly to 10000"):
        _policy(observed=7_000, registry_wide=2_000, gb_extension=0)
