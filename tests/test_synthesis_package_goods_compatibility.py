from __future__ import annotations

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.package_goods_compatibility import (
    apply_package_signature,
    build_package_goods_fit_support,
    sample_compatible_cargo,
    sample_dangerous_goods_package,
    sample_supported_thermal_profile,
    supported_thermal_profiles,
)
from document_ocr.synthesis.thermal_goods import (
    AmbientGoodsIdentity,
    ThermalGoodsIdentity,
    ThermalGoodsSupport,
)


def _target(
    *,
    hs_code: str,
    package: str,
    dangerous: str | None = None,
    temperature: float | None = None,
) -> dict[str, object]:
    group: dict[str, object] = {"groupId": "g1", "hsCodes": [hs_code]}
    if dangerous is not None:
        group["dangerousGoods"] = [{"hazardCategory": dangerous, "unNumber": "9999"}]
    container: dict[str, object] = {"containerNumber": "C1"}
    allocations: list[dict[str, object]] = []
    if temperature is not None:
        container["temperatureSetpoint"] = {"value": temperature, "unit": "celsius"}
        allocations = [
            {
                "groupId": "g1",
                "coverage": "single_package_level",
                "packageIds": ["p1"],
                "allocations": [{"containerNumber": "C1", "packageId": "p1", "packageQuantity": 1}],
            }
        ]
    return {
        "documentPatch": {
            "cargoGroups": [group],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 1,
                    "typeCategory": package,
                }
            ],
            "containers": [container],
            "cargoAllocationGroups": allocations,
        }
    }


def _goods() -> ThermalGoodsSupport:
    return ThermalGoodsSupport(
        frozen=(ThermalGoodsIdentity("030363", "Fish", "Frozen fish", "Frozen cod", "FROZEN"),),
        chilled=(
            ThermalGoodsIdentity("070320", "Vegetables", "Garlic", "Fresh garlic", "CHILLED"),
        ),
        ambient=(
            AmbientGoodsIdentity("392113", "Plastics", "Other plates", "Polyurethane sheet"),
            AmbientGoodsIdentity("392190", "Plastics", "Other plates", "Other plastic sheet"),
            AmbientGoodsIdentity("481910", "Paper", "Cartons", "Corrugated cartons"),
        ),
        ambient_chapters=("39", "48"),
    )


def _stream(value: str) -> DeterministicStream:
    return DeterministicStream(7, "package-goods-test", value)


def test_ambient_goods_and_package_are_sampled_from_one_observed_heading_joint() -> None:
    support = build_package_goods_fit_support(
        source_targets={
            "doc_fit": _target(hs_code="392113", package="PACKAGE_ROLL"),
        },
        fit_document_ids=("doc_fit",),
        allowed_category_tokens=("PACKAGE_ROLL", "PACKAGE_CARTON"),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    selected = sample_compatible_cargo(
        support=support,
        goods_support=_goods(),
        profile=None,
        package_count=1,
        identity_count=1,
        stream=_stream("ambient"),
        excluded_hs6=set(),
    )
    assert selected.identities[0].hs6[:4] == "3921"
    assert selected.package_signature == ("PACKAGE_ROLL",)
    assert selected.basis == "fit_hs_signature_conditioned_heading_pool"


@pytest.mark.parametrize(
    "profile,hs,category",
    [
        (None, "392190", "PACKAGE_ROLL"),
        ("FROZEN", "030363", "PACKAGE_CARTON"),
    ],
)
def test_shared_identity_is_a_sampling_constraint_not_a_post_sampling_replacement(
    profile, hs, category
):
    source = _target(hs_code=hs, package=category, temperature=-20 if profile else None)
    support = build_package_goods_fit_support(
        source_targets={"fit": source},
        fit_document_ids=("fit",),
        allowed_category_tokens=(category,),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    for seed in range(10):
        chosen = sample_compatible_cargo(
            support=support,
            goods_support=_goods(),
            profile=profile,
            package_count=1,
            identity_count=1,
            stream=_stream(str(seed)),
            excluded_hs6=set(),
            required_hs6_by_index={0: hs},
        )
        assert chosen.identities[0].hs6 == hs
        assert chosen.package_signature == (category,)
    with pytest.raises(ValueError, match="no fit-supported candidate"):
        sample_compatible_cargo(
            support=support,
            goods_support=_goods(),
            profile=profile,
            package_count=1,
            identity_count=1,
            stream=_stream("unsupported"),
            excluded_hs6=set(),
            required_hs6_by_index={0: "481910"},
        )


def test_required_goods_positions_stay_distinct_and_validate_exclusion():
    support = build_package_goods_fit_support(
        source_targets={"fit": _target(hs_code="392113", package="PACKAGE_ROLL")},
        fit_document_ids=("fit",),
        allowed_category_tokens=("PACKAGE_ROLL",),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    for position in (0, 1):
        result = sample_compatible_cargo(
            support=support,
            goods_support=_goods(),
            profile=None,
            package_count=1,
            identity_count=2,
            stream=_stream(str(position)),
            excluded_hs6=set(),
            required_hs6_by_index={position: "392190"},
        )
        assert result.identities[position].hs6 == "392190"
        assert len({i.hs6 for i in result.identities}) == 2
    with pytest.raises(ValueError, match="conflict with excluded"):
        sample_compatible_cargo(
            support=support,
            goods_support=_goods(),
            profile=None,
            package_count=1,
            identity_count=2,
            stream=_stream("excluded"),
            excluded_hs6={"392190"},
            required_hs6_by_index={0: "392190"},
        )


def test_multi_identity_cargo_uses_a_fit_conditioned_signature_heading_pool() -> None:
    source = _target(hs_code="392113", package="PACKAGE_ROLL")
    patch = source["documentPatch"]
    assert isinstance(patch, dict)
    group = patch["cargoGroups"][0]
    assert isinstance(group, dict)
    group["hsCodes"] = ["392113", "481910", "392190"]
    support = build_package_goods_fit_support(
        source_targets={"doc_fit": source},
        fit_document_ids=("doc_fit",),
        allowed_category_tokens=("PACKAGE_ROLL",),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )

    selected = sample_compatible_cargo(
        support=support,
        goods_support=_goods(),
        profile=None,
        package_count=1,
        identity_count=3,
        stream=_stream("multi-heading-profile"),
        excluded_hs6=set(),
    )

    assert {value.hs6[:4] for value in selected.identities} <= {"3921", "4819"}
    assert len({value.hs6 for value in selected.identities}) == 3
    assert selected.package_signature == ("PACKAGE_ROLL",)
    assert selected.basis == "fit_hs_signature_conditioned_heading_pool"


def test_ambient_heading_index_excludes_exhausted_identity_pools() -> None:
    support = build_package_goods_fit_support(
        source_targets={
            "doc_plastic": _target(hs_code="392113", package="PACKAGE_ROLL"),
            "doc_paper": _target(hs_code="481910", package="PACKAGE_CARTON"),
        },
        fit_document_ids=("doc_plastic", "doc_paper"),
        allowed_category_tokens=("PACKAGE_ROLL", "PACKAGE_CARTON"),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    selected = sample_compatible_cargo(
        support=support,
        goods_support=_goods(),
        profile=None,
        package_count=1,
        identity_count=1,
        stream=_stream("exhausted-heading"),
        excluded_hs6={"392113", "392190"},
    )
    assert selected.identities[0].hs6 == "481910"
    assert selected.package_signature == ("PACKAGE_CARTON",)


def test_frozen_goods_use_temperature_linked_package_support_not_ambient_role_marginal() -> None:
    support = build_package_goods_fit_support(
        source_targets={
            "doc_frozen": _target(hs_code="030363", package="PACKAGE_CARTON", temperature=-20),
            "doc_ambient": _target(hs_code="392113", package="PACKAGE_ROLL"),
        },
        fit_document_ids=("doc_frozen", "doc_ambient"),
        allowed_category_tokens=("PACKAGE_CARTON", "PACKAGE_ROLL"),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    selected = sample_compatible_cargo(
        support=support,
        goods_support=_goods(),
        profile="FROZEN",
        package_count=1,
        identity_count=1,
        stream=_stream("frozen"),
        excluded_hs6=set(),
    )
    assert selected.identities[0].description == "Frozen cod"
    assert selected.package_signature == ("PACKAGE_CARTON",)
    assert selected.basis == "fit_thermal_profile_joint"


def test_dangerous_goods_abstains_without_hazard_support_instead_of_guessing() -> None:
    support = build_package_goods_fit_support(
        source_targets={
            "doc_dg": _target(
                hs_code="292130",
                package="PACKAGE_DRUM_FIBRE",
                dangerous="FLAMMABLE_SOLIDS",
            )
        },
        fit_document_ids=("doc_dg",),
        allowed_category_tokens=("PACKAGE_DRUM_FIBRE", "PACKAGE_BAG"),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    resolved = sample_dangerous_goods_package(
        support=support,
        hazard_categories=("FLAMMABLE_SOLIDS",),
        package_count=1,
        stream=_stream("supported-dg"),
    )
    assert resolved is not None
    assert resolved.package_signature == ("PACKAGE_DRUM_FIBRE",)
    assert (
        sample_dangerous_goods_package(
            support=support,
            hazard_categories=("OXIDIZING_SUBSTANCES_AND_ORGANIC_PEROXIDES",),
            package_count=1,
            stream=_stream("unsupported-dg"),
        )
        is None
    )


def test_package_application_preserves_ids_quantities_and_group_topology() -> None:
    target = _target(hs_code="392113", package="PACKAGE_ROLL")
    patch = target["documentPatch"]
    assert isinstance(patch, dict)
    before = dict(patch["cargoPackages"][0])
    apply_package_signature(target=target, group_id="g1", signature=("PACKAGE_CARTON",))
    after = patch["cargoPackages"][0]
    assert after["typeCategory"] == "PACKAGE_CARTON"
    assert {key: after[key] for key in ("groupId", "packageId", "quantity")} == {
        key: before[key] for key in ("groupId", "packageId", "quantity")
    }


def test_thermal_profile_is_selected_only_from_fit_supported_package_cardinality() -> None:
    chilled = _target(hs_code="070320", package="PACKAGE_CARTON", temperature=2)
    patch = chilled["documentPatch"]
    assert isinstance(patch, dict)
    patch["cargoPackages"] = [
        {
            "groupId": "g1",
            "packageId": f"p{index}",
            "quantity": 1,
            "typeCategory": "PACKAGE_CARTON",
        }
        for index in range(1, 4)
    ]
    support = build_package_goods_fit_support(
        source_targets={"doc_chilled": chilled},
        fit_document_ids=("doc_chilled",),
        allowed_category_tokens=("PACKAGE_CARTON",),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    profiles = supported_thermal_profiles(
        support=support,
        goods_support=_goods(),
        package_count=3,
        identity_count=1,
    )
    assert profiles == ("CHILLED",)
    assert (
        sample_supported_thermal_profile(
            profiles=profiles,
            weights_permyriad={"FROZEN": 9_999, "CHILLED": 1},
            stream=_stream("supported-profile"),
        )
        == "CHILLED"
    )


@pytest.mark.parametrize("profile,code", [(None, "481910"), ("FROZEN", "030363")])
def test_numeric_compatibility_filters_before_drawing_without_widening(profile, code):
    targets = {
        "paper": _target(hs_code="481910", package="PACKAGE_CARTON"),
        "plastic": _target(hs_code="392113", package="PACKAGE_ROLL"),
        "frozen": _target(hs_code="030363", package="PACKAGE_CARTON", temperature=-20),
    }
    support = build_package_goods_fit_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        allowed_category_tokens=("PACKAGE_CARTON", "PACKAGE_ROLL"),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    kwargs = dict(
        support=support,
        goods_support=_goods(),
        profile=profile,
        package_count=1,
        identity_count=1,
        excluded_hs6=set(),
    )
    selected = sample_compatible_cargo(
        **kwargs,
        stream=_stream("allowed"),
        allowed_heading_signatures=frozenset({(code[:4], ("PACKAGE_CARTON",))}),
    )
    assert selected.identities[0].hs6 == code
    kwargs["excluded_hs6"] = set()
    with pytest.raises(ValueError, match="no fit-supported candidate"):
        sample_compatible_cargo(
            **kwargs,
            stream=_stream("empty"),
            allowed_heading_signatures=frozenset(),
        )
