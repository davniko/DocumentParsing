"""Train-only numeric plausibility support, conditioned on goods and packaging.

Container capacity alone does not prevent a hundred vehicles weighing one tonne.
Keep mass and volume together from ONE observed cargo group; do not combine the
lightest mass and largest volume from unrelated observations. The configured
expansion is a synthesis policy, not a physical-law or compliance assertion.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.semantic_completion_pipeline import _group_allocations

MEASUREMENT_FIELDS = frozenset({"grossWeight", "netWeight", "volume"})
_FACTORS = {
    "kilogram": Decimal(1),
    "metric_tonne": Decimal(1000),
    "pound": Decimal("0.45359237"),
    "cubic_metre": Decimal(1),
}


def allocated_container_counts(patch: Mapping[str, Any]) -> dict[str, int]:
    allocations = _group_allocations(patch)
    groups = patch.get("cargoGroups", ())
    if len(groups) == 1 and patch.get("containers"):
        return {groups[0]["groupId"]: len(patch["containers"])}
    return {key: len(values) for key, values in allocations.items()}


def measures_per_package(
    group: Mapping[str, Any],
    packages: Sequence[Mapping[str, Any]],
    *,
    container_count: int | None = None,
    include_total_measures: bool = False,
) -> dict[str, float]:
    if not packages:
        return {}
    count = (
        sum(float(p["quantity"]) for p in packages)
        if all("quantity" in p for p in packages)
        else None
    )
    if count is not None and count <= 0:
        raise ValueError("cargo package quantity must be positive")
    result = {}
    if container_count is not None:
        if container_count < 1:
            raise ValueError("package-count support requires positive container ownership")
        if count is not None:
            result["packagesPerContainer"] = count / container_count
    for field in ("grossWeight", "netWeight", "volume"):
        if field not in group:
            continue
        value = group[field]
        factor = {
            "kilogram": 1,
            "metric_tonne": 1000,
            "pound": 0.45359237,
            "cubic_metre": 1,
        }[value["unit"]]
        absolute = float(value["value"]) * factor
        if count is not None:
            result[field] = absolute / count
        if count is None or include_total_measures:
            result[field + "PerGroup"] = absolute
            if container_count is not None:
                result[field + "PerContainer"] = absolute / container_count
                if count is None:
                    # Compare the supported physical allocation scale, not the
                    # number of unprinted packages. No count is invented.
                    del result[field + "PerGroup"]
    return result


@dataclass(frozen=True)
class MeasurementSupport:
    points: Mapping[tuple[str, tuple[str, ...]], tuple[dict[str, float], ...]]
    lower_multiplier: float
    upper_multiplier: float

    def allowed_heading_signatures(
        self,
        group: Mapping[str, Any],
        packages: Sequence[Mapping[str, Any]],
        *,
        container_count: int | None = None,
        mutable_fields: frozenset[str] = frozenset(),
    ) -> frozenset[tuple[str, tuple[str, ...]]] | None:
        actual = measures_per_package(group, packages, container_count=container_count)
        fixed = {
            k: v
            for k, v in actual.items()
            if not any(k == field or k.startswith(field + "Per") for field in mutable_fields)
        }
        if not actual:
            return None
        return frozenset(
            key
            for key, points in self.points.items()
            if len(key[1]) == len(packages)
            and any(
                actual.keys() <= point.keys()
                and all(
                    point[k] * self.lower_multiplier <= v <= point[k] * self.upper_multiplier
                    for k, v in fixed.items()
                )
                for point in points
            )
        )

    def draw_measures(
        self,
        hs6: str,
        group: dict[str, Any],
        packages: Sequence[Mapping[str, Any]],
        *,
        quanta: Mapping[str, Decimal],
        container_count: int | None,
        stream: DeterministicStream,
        vary_fit_scale: bool = False,
    ) -> dict[str, Any]:
        """Generate visible measurements jointly from one fit-only observation.

        Package quantities and label visibility are unchanged. Source equations
        are represented by omitted (locked) fields in ``quanta``; they are never
        repaired by copying old values after generation. When all printed
        measurements are free, rejection sampling can explore the configured
        fit envelope with ONE common multiplier, retaining the observation's
        density and gross/net relationship before source-precision rounding.
        """
        if not quanta:
            return {"method": "compiled_physical_constraints_preserved", "fields": []}
        actual = measures_per_package(group, packages, container_count=container_count)
        if not actual:
            raise ValueError("joint measurement generation requires observed package structure")
        count = (
            sum(Decimal(str(p["quantity"])) for p in packages)
            if all("quantity" in p for p in packages)
            else None
        )
        metrics = {
            field: field
            if count is not None
            else field + "PerContainer"
            if container_count is not None
            else field + "PerGroup"
            for field in quanta
        }
        fixed = {k: v for k, v in actual.items() if k not in metrics.values()}
        signature = tuple(p["typeCategory"] for p in packages)
        candidates = [
            point
            for point in self.points.get((hs6[:4], signature), ())
            if actual.keys() <= point.keys()
            and all(
                point[k] * self.lower_multiplier <= v <= point[k] * self.upper_multiplier
                for k, v in fixed.items()
            )
        ]
        if not candidates:
            raise ValueError("joint measurement generation has no conditioned fit point")
        point = candidates[stream.derive("physical-fit-point").randbelow(len(candidates))]
        factor = Decimal(1)
        policy = "exact_joint_fit_point"
        if vary_fit_scale:
            if set(quanta) == MEASUREMENT_FIELDS.intersection(group):
                # A 53-bit uniform fraction is the usual double-precision RNG
                # resolution; it is not an extra business-distribution bound.
                fraction = Decimal(
                    stream.derive("common-fit-multiplier").randbelow(2**53)
                ) / Decimal(2**53)
                factor = Decimal(str(self.lower_multiplier)) + fraction * (
                    Decimal(str(self.upper_multiplier)) - Decimal(str(self.lower_multiplier))
                )
                policy = "common_multiplier_within_configured_fit_bounds"
            else:
                policy = "locked_measurements_preserve_exact_fit_point"
        denominator = count if count is not None else Decimal(container_count or 1)
        generated = {}
        for field, quantum in quanta.items():
            if quantum <= 0 or not quantum.is_finite():
                raise ValueError("joint measurement quantum must be finite and positive")
            unit = group[field]["unit"]
            value = Decimal(str(point[metrics[field]])) * denominator * factor / _FACTORS[unit]
            rounded = (value / quantum).quantize(Decimal(1), rounding=ROUND_HALF_UP) * quantum
            if rounded <= 0:
                raise ValueError("fit measurement cannot be represented at source precision")
            generated[field] = {"value": float(rounded), "unit": unit}
        trial = {**group, **generated}
        if {"grossWeight", "netWeight"} <= trial.keys():
            gross = (
                Decimal(str(trial["grossWeight"]["value"])) * _FACTORS[trial["grossWeight"]["unit"]]
            )
            net = Decimal(str(trial["netWeight"]["value"])) * _FACTORS[trial["netWeight"]["unit"]]
            if net > gross:
                raise ValueError("joint fit measurement has net weight above gross weight")
        if not self.compatible(hs6, trial, packages, container_count=container_count):
            raise ValueError("rounded joint measurements violate fit support")
        group.update(generated)
        return dict(
            method="one_joint_fit_observation_at_source_precision",
            heading=hs6[:4],
            packageSignature=list(signature),
            fields=sorted(quanta),
            fitPoint=point,
            commonFitScale=str(factor),
            fitScalePolicy=policy,
            generated=generated,
        )

    def compatible(
        self,
        hs6: str,
        group: Mapping[str, Any],
        packages: Sequence[Mapping[str, Any]],
        *,
        container_count: int | None = None,
    ) -> bool:
        actual = measures_per_package(group, packages, container_count=container_count)
        if not actual:
            return True  # No measured assertion exists in this target, not a filled default.
        signature = tuple(p["typeCategory"] for p in packages)
        return any(
            actual.keys() <= point.keys()
            and all(
                point[k] * self.lower_multiplier <= v <= point[k] * self.upper_multiplier
                for k, v in actual.items()
            )
            for point in self.points.get((hs6[:4], signature), ())
        )


def build_support(
    targets: Mapping[str, Mapping[str, Any]], *, lower_multiplier: float, upper_multiplier: float
) -> MeasurementSupport:
    points: dict[tuple[str, tuple[str, ...]], list[dict[str, float]]] = defaultdict(list)
    for target in targets.values():
        patch = target["documentPatch"]
        container_counts = allocated_container_counts(patch)
        for group in patch.get("cargoGroups", []):
            if group.get("dangerousGoods"):
                continue
            packages = [
                p for p in patch.get("cargoPackages", []) if p["groupId"] == group["groupId"]
            ]
            if not packages or any("typeCategory" not in p for p in packages):
                continue  # Untyped source rows cannot establish categorical support.
            measures = measures_per_package(
                group,
                packages,
                container_count=container_counts.get(group["groupId"]),
                include_total_measures=True,
            )
            if not measures:
                continue
            signature = tuple(p["typeCategory"] for p in packages)
            for heading in {re.sub(r"\D", "", c)[:4] for c in group.get("hsCodes", [])}:
                points[(heading, signature)].append(measures)
    return MeasurementSupport(
        {k: tuple(v) for k, v in points.items()}, lower_multiplier, upper_multiplier
    )
