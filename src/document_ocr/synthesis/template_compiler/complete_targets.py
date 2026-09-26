"""Template-conditioned, complete target synthesis before the immutable render boundary.

Identifiers and arithmetic are local. Linguistic requests describe whole related
parties and cargo, with explicit representability constraints. No renderer-side
target repair and no source-copy substitute is permitted.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from itertools import pairwise
from math import gcd, lcm
from typing import Any, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_from_surface_pattern,
    reconcile_allocation_group,
    surface_pattern,
)
from document_ocr.synthesis.hs_registry import UkGlobalTariffRegistry
from document_ocr.synthesis.raw_text_template import printed_topology_mismatches
from document_ocr.synthesis.rendering import render_number_surface
from document_ocr.synthesis.structured_semantics import (
    apply_identifier_plan,
    build_identifier_request_inventory,
    reserve_structured_identifiers,
)
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_V5_TASK_ADAPTER
from document_ocr.synthesis.template_integrity import source_template_integrity_issues
from document_ocr.synthesis.thermal_goods import classify_thermal_hs

from . import (
    cargo_identifiers,
    count_aliases,
    lexical_partitions,
    measurement_prose,
    nested_package_prose,
    package_equations,
    package_prose,
)
from . import descendant as render
from .coherence import coherence_dependency_paths, validate_render_coherence
from .generation_contract import (
    fixed_carrier_role_paths,
    order_party_reference,
    party_owned_surfaces,
    require_complete_variation,
    validate_compiled_party_contract,
    validate_party_evidence,
)
from .host import (
    validate_compiled_extraction_dates,
    validate_compiled_global_shared_temperature_scope,
    validate_compiled_location_payment_grounding,
    validate_compiled_repeated_cargo_temperature_scope,
    validate_compiled_signed_temperature_word_scope,
    validate_compiled_single_printed_hs_scope,
)
from .latest_target import latest_target_from_source
from .models import CertifiedSemanticTemplate, SemanticBinding
from .numeric_auxiliary import surface_quantum
from .range_generation import condition_lexical_ranges as condition_lexical_ranges
from .range_generation import formal_range_text, plan_ranges
from .realization_contract import (
    character_partition_frames,
    complete_token_intervals,
    effective_realization_template,
    fixed_projection_ranges,
    projected_auxiliary_values,
    token_projection_intervals,
    token_projection_pattern,
)


@dataclass(frozen=True)
class SourceTemplate:
    document_id: str
    source: bytes
    source_target: dict[str, Any]
    target: dict[str, Any]
    template: CertifiedSemanticTemplate
    original_template_sha256: str


def load_source(root: Any, document_id: str) -> SourceTemplate:
    source, label, template_bytes = render._case_files(root, document_id)
    template = CertifiedSemanticTemplate.model_validate_json(template_bytes, strict=True)
    if template.document_id != document_id or template.source_sha256 != sha256_bytes(source):
        raise ValueError("source/template identity differs")
    issues = source_template_integrity_issues(source.decode(), label)
    if issues:
        raise ValueError("source template integrity requires review: " + "; ".join(issues))
    from .temperature_prose import require_singleton_printed_setpoint

    require_singleton_printed_setpoint(label)
    validate_compiled_extraction_dates(
        raw=source.decode("utf-8"), source_target=label, template=template
    )
    validate_compiled_location_payment_grounding(source_target=label, template=template)
    validate_compiled_party_contract(
        raw=source,
        source_target=label,
        bindings=template.bindings,
        entities=template.auxiliary_semantic_plan.entities,
    )
    validate_compiled_single_printed_hs_scope(
        raw=source.decode("utf-8"), source_target=label, template=template
    )
    validate_compiled_global_shared_temperature_scope(
        raw=source.decode("utf-8"), source_target=label, template=template
    )
    validate_compiled_repeated_cargo_temperature_scope(
        raw=source.decode("utf-8"), source_target=label, template=template
    )
    validate_compiled_signed_temperature_word_scope(
        raw=source.decode("utf-8"), source_target=label, template=template
    )
    validate_party_evidence(
        target=label,
        binding_paths={path for binding in template.bindings for path in binding.target_paths},
        party_surfaces=party_owned_surfaces(
            template.bindings,
            {slot.slot_id: slot.source_text for slot in template.byte_template.slots},
            template.auxiliary_semantic_plan.entities,
        ),
    )
    return SourceTemplate(
        document_id,
        source,
        label,
        latest_target_from_source(label),
        effective_realization_template(template),
        sha256_bytes(template_bytes),
    )


def _set(target: dict[str, Any], path: str, value: Any) -> None:
    parts = render._path_parts(path)
    current: Any = target
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def identifier_source(target: Mapping[str, Any]) -> dict[str, Any]:
    """Narrative export references are linguistic, not arbitrary character strings."""
    return {
        **target,
        "documentPatch": {
            key: value
            for key, value in target["documentPatch"].items()
            if key != "forwardingAndExportReferences"
        },
    }


def reserve_identifiers(
    samples: Sequence[Mapping[str, Any]], sources: Mapping[str, SourceTemplate], *, seed: int
) -> dict[str, str]:
    unique = {key: identifier_source(source.target) for key, source in sources.items()}
    selected = {row["sampleId"]: unique[row["sourceDocumentId"]] for row in samples}
    inventory = build_identifier_request_inventory(
        source_targets=selected, selected_document_ids=tuple(selected)
    )
    globally_unique = tuple(
        r
        for r in inventory.generic_requests
        if "BillOfLadingNumber" in r.request_key or "billOfLadingNumber" in r.request_key
    )
    globally_unique_keys = {request.request_key for request in globally_unique}
    result = reserve_structured_identifiers(
        all_source_targets=unique,
        inventory=replace(inventory, generic_requests=globally_unique),
        seed=seed,
    ).values()
    # A voyage or seal is not a globally unique document identity. Reserve these
    # within each sample, avoiding source values and intra-document collisions.
    grouped: dict[str, list[Any]] = {}
    for request in inventory.generic_requests:
        if request.request_key not in globally_unique_keys:
            grouped.setdefault(request.request_key.split("/", 1)[0], []).append(request)
    for sample_id, requests in grouped.items():
        local = replace(inventory, container_requests=(), generic_requests=tuple(requests))
        result.update(
            reserve_structured_identifiers(
                all_source_targets={sample_id: selected[sample_id]}, inventory=local, seed=seed
            ).values()
        )
    return result


def _numeric_quantum(source: SourceTemplate, path: str, old: int | float) -> Decimal:
    precisions = []
    for binding in source.template.bindings:
        if path not in binding.target_paths or binding.realization.adapter != "numeric":
            continue
        for slot in binding.occurrences:
            # Use the actual certified renderer, not a guessed decimal convention.
            for precision in range(0, 10):
                quantum = Decimal(1).scaleb(-precision)
                try:
                    render_number_surface(slot.source_text, old, Decimal(str(old)) + quantum)
                except ValueError as error:
                    if precision == 0:
                        raise ValueError(
                            f"numeric source cannot represent a whole increment: {path}"
                        ) from error
                    break
            else:
                precision = 10
            precisions.append(max(0, precision - 1))
    if precisions:
        return Decimal(1).scaleb(-min(precisions))
    # A compiled measurement may be marked agent-assisted at extraction time
    # yet be printed by the deterministic typed-measurement renderer. Its
    # canonical source value is a float (e.g. 7740.0), whose serialization
    # does not imply that a grouped `7.740 KG` slot prints tenths of a kg.
    # Use the same source unit and numeric-surface proof as that renderer.
    if re.search(r"\.(?:grossWeight|netWeight|volume)\.value$", path):
        proven_steps = []
        for binding in source.template.bindings:
            if path not in binding.target_paths:
                continue
            for slot in binding.occurrences:
                try:
                    factor, _ = render._measurement_factor_for_source(
                        binding,
                        source=source.source,
                        source_target=source.target,
                        target=source.target,
                        template=source.template,
                        value_paths=(path,),
                        occurrences=(slot,),
                    )
                    printed = Decimal(str(old)) * factor
                    proven_steps.append(surface_quantum(slot.source_text, printed) / factor)
                except ValueError:
                    # A segmented/ambiguous slot is not evidence of precision.
                    # Another occurrence may be independently source-proven.
                    continue
            # OCR segmentation can split a single numeric token into adjacent
            # slots such as `11200` + `.000`. The joined source bytes, not either
            # component alone, prove the decimal precision.
            ordered = sorted(binding.occurrences, key=lambda slot: slot.byte_start)
            for left, right in pairwise(ordered):
                if (
                    source.source[left.byte_end : right.byte_start] not in {b".", b","}
                    or re.fullmatch(r"\d+", left.source_text) is None
                    or re.fullmatch(r"\d+", right.source_text) is None
                ):
                    continue
                try:
                    factor, _ = render._measurement_factor_for_source(
                        binding,
                        source=source.source,
                        source_target=source.target,
                        target=source.target,
                        template=source.template,
                        value_paths=(path,),
                        occurrences=(right,),
                    )
                    combined = source.source[left.byte_start : right.byte_end].decode("utf-8")
                    printed = Decimal(str(old)) * factor
                    proven_steps.append(surface_quantum(combined, printed) / factor)
                except ValueError:
                    continue
        if proven_steps:
            # A finer repeated occurrence can support the target even when a
            # summary occurrence rounds it. Do not force the model to infer
            # precision absent from *every* owned surface.
            return min(proven_steps)
    # A converted/composite measurement is not a direct numeric adapter. Its
    # absence here does not prove integer precision in the TARGET unit (e.g.
    # 48.751 tonnes printed as 48,751 kg). Preserve the observed target precision;
    # numeric dependency preparation and the typed renderer still prove every
    # printed occurrence before acceptance. Rounding those values to whole tonnes
    # could make a generated gross weight smaller than the corresponding net.
    return Decimal(1).scaleb(min(0, int(Decimal(str(old)).as_tuple().exponent)))


def _equal_allocation_quantities(
    weights: list[int],
    lower_bounds: list[int],
    total: int,
    equal_rows: tuple[int, ...],
    *,
    independent_total: bool,
) -> list[int]:
    """Draw one shared count before apportioning the other observed rows.

    An unlinked subtotal is its own sampling dimension, not the shipment total.
    A linked total is already a label fact and must never be silently changed.
    """
    equal = set(equal_rows)
    if (
        len(equal) != len(equal_rows)
        or len(equal) < 2
        or any(i < 0 or i >= len(weights) for i in equal)
        or len({weights[i] for i in equal}) != 1
        or weights[equal_rows[0]] <= 0
    ):
        raise ValueError("invalid shared allocation equality contract")
    if len(lower_bounds) != len(weights) or any(
        bound < 0 or (weight == 0 and bound != 0)
        for weight, bound in zip(weights, lower_bounds, strict=True)
    ):
        raise ValueError("allocation lower bounds conflict with observed row support")
    minimum = max(1, *(lower_bounds[i] for i in equal))
    other = [i for i, weight in enumerate(weights) if weight > 0 and i not in equal]
    minimum_total = len(equal) * minimum + sum(lower_bounds[i] for i in other)
    if total < minimum_total:
        raise ValueError("new package total cannot preserve shared allocation lower bounds")
    if not other:
        if independent_total:
            total = max(minimum_total, total // len(equal) * len(equal))
        elif total % len(equal):
            raise ValueError("linked package total is indivisible by its shared allocation count")
        shared = total // len(equal)
    else:
        maximum = (total - sum(lower_bounds[i] for i in other)) // len(equal)
        numerator = total * weights[equal_rows[0]]
        denominator = sum(weights)
        shared = min(maximum, max(minimum, (2 * numerator + denominator) // (2 * denominator)))
    if independent_total and total > sum(weights):
        raise ValueError("shared allocation constraints exceed observed source capacity")
    result = [0] * len(weights)
    for index in equal:
        result[index] = shared
    remaining = total - len(equal) * shared - sum(lower_bounds[i] for i in other)
    if other:
        weight_sum = sum(weights[i] for i in other)
        raw = [divmod(remaining * weights[i], weight_sum) for i in other]
        for index, (quotient, _) in zip(other, raw, strict=True):
            result[index] = lower_bounds[index] + quotient
        for index in sorted(range(len(other)), key=lambda i: (-raw[i][1], i))[
            : total - sum(result)
        ]:
            result[other[index]] += 1
    if sum(result) != total:
        raise ValueError("shared allocation apportionment did not close its subtotal")
    return result


def _scale_numbers(
    source: SourceTemplate,
    target: dict[str, Any],
    stream: DeterministicStream,
    minimum_quantities: Mapping[str, int],
    quantity_multiples: Mapping[str, int],
    measurement_steps: Mapping[str, Decimal] | None = None,
) -> None:
    # Downscaling keeps an observed shipment within its existing equipment capacity.
    # Integer indivisibility and the minimum nonzero allocation are explicit constraints.
    ratio = Decimal(600 + stream.derive("cargo-scale").randbelow(350)) / 1000
    fixed_quantities = count_aliases.fixed_quantities(source.source, source.template, source.target)
    allocation_equalities = nested_package_prose.allocation_equalities(
        source.template, source.target
    )
    patch = target["documentPatch"]
    allocation_by_group = {row["groupId"]: row for row in patch.get("cargoAllocationGroups", [])}
    allocation_indices = {
        row["groupId"]: i for i, row in enumerate(patch.get("cargoAllocationGroups", []))
    }
    package_by_id = {row["packageId"]: row for row in patch.get("cargoPackages", [])}
    for index, package in enumerate(patch.get("cargoPackages", [])):
        old = package.get("quantity")
        if old is None:
            continue
        allocation = allocation_by_group.get(package["groupId"])
        minimum = max(
            1, minimum_quantities.get(f"documentPatch.cargoPackages[{index}].quantity", 1)
        )
        if allocation is not None:
            allocation_index = allocation_indices[package["groupId"]]
            bounds = [
                minimum_quantities.get(
                    f"documentPatch.cargoAllocationGroups[{allocation_index}].allocations[{i}].packageQuantity",
                    int(row.get("packageQuantity", 0) > 0),
                )
                for i, row in enumerate(allocation["allocations"])
                if allocation["coverage"] != "one_to_one_package_allocations"
                or row.get("packageId") == package["packageId"]
            ]
            equal_rows = allocation_equalities.get(allocation_index, ())
            if equal_rows and allocation["coverage"] != "one_to_one_package_allocations":
                shared_minimum = max(1, *(bounds[i] for i in equal_rows))
                for row_index in equal_rows:
                    bounds[row_index] = shared_minimum
            minimum = max(minimum, sum(bounds))
        if (
            allocation is not None
            and allocation["coverage"] != "one_to_one_package_allocations"
            and package["packageId"] in allocation["packageIds"]
        ):
            minimum = max(
                minimum, sum(row.get("packageQuantity", 0) > 0 for row in allocation["allocations"])
            )
        new = max(
            minimum, 1, int((Decimal(old) * ratio).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        )
        if old > minimum and new == old:
            new -= 1
        multiple = quantity_multiples.get(f"documentPatch.cargoPackages[{index}].quantity", 1)
        new = max(((minimum + multiple - 1) // multiple) * multiple, (new // multiple) * multiple)
        if f"documentPatch.cargoPackages[{index}].quantity" in fixed_quantities:
            new = old
            if new < minimum or new % multiple:
                raise ValueError("immutable printed count conflicts with allocation constraints")
        if new > old:
            raise ValueError("quantity constraints exceed the observed source capacity")
        package["quantity"] = new
    for allocation in allocation_by_group.values():
        allocation_index = allocation_indices[allocation["groupId"]]
        equal_rows = allocation_equalities.get(allocation_index, ())
        if allocation["coverage"] in {
            "one_to_one_package_allocations",
            "container_membership_only",
        }:
            reconciled = reconcile_allocation_group(
                packages=list(package_by_id.values()), allocation_group=allocation
            )
            if (
                equal_rows
                and len({reconciled["allocations"][i].get("packageQuantity") for i in equal_rows})
                != 1
            ):
                raise ValueError("linked package quantities contradict shared allocation equality")
            allocation.update(reconciled)
            continue
        quantified = [row for row in allocation["allocations"] if "packageQuantity" in row]
        if not quantified:
            continue
        if len(quantified) != len(allocation["allocations"]):
            raise ValueError(
                "partially quantified allocation requires an explicit generation contract"
            )
        weights = [row["packageQuantity"] for row in quantified]
        lower_bounds = [
            minimum_quantities.get(
                f"documentPatch.cargoAllocationGroups[{allocation_index}].allocations[{i}].packageQuantity",
                int(weight > 0),
            )
            for i, weight in enumerate(weights)
        ]
        if equal_rows:
            shared_minimum = max(1, *(lower_bounds[i] for i in equal_rows))
            for index in equal_rows:
                lower_bounds[index] = shared_minimum
        total = (
            max(
                sum(lower_bounds),
                int((sum(weights) * ratio).quantize(Decimal(1), rounding=ROUND_HALF_UP)),
            )
            if allocation["coverage"] == "unlinked_package_quantities"
            else sum(package_by_id[pid]["quantity"] for pid in allocation["packageIds"])
        )
        if equal_rows:
            assigned_counts = _equal_allocation_quantities(
                weights,
                lower_bounds,
                total,
                equal_rows,
                independent_total=allocation["coverage"] == "unlinked_package_quantities",
            )
            for row, count in zip(quantified, assigned_counts, strict=True):
                row["packageQuantity"] = count
            continue
        positive = [i for i, w in enumerate(weights) if w > 0]
        if total < sum(lower_bounds) or not positive:
            raise ValueError("new package total cannot preserve allocation support")
        remaining = total - sum(lower_bounds)
        raw = [divmod(remaining * weights[i], sum(weights)) for i in positive]
        values = [lower_bounds[index] + q for index, (q, _) in zip(positive, raw, strict=True)]
        for i in sorted(range(len(positive)), key=lambda i: (-raw[i][1], i))[: total - sum(values)]:
            values[i] += 1
        assigned = dict(zip(positive, values, strict=True))
        for i, row in enumerate(quantified):
            row["packageQuantity"] = assigned.get(i, 0)
    for index, group in enumerate(patch.get("cargoGroups", [])):
        quanta = {
            field: _numeric_quantum(
                source, f"documentPatch.cargoGroups[{index}].{field}.value", group[field]["value"]
            )
            for field in ("grossWeight", "netWeight", "volume")
            if field in group
        }
        for field, quantum in quanta.items():
            path = f"documentPatch.cargoGroups[{index}].{field}.value"
            if measurement_steps is not None and path in measurement_steps:
                partition_steps = (Fraction(quantum), Fraction(measurement_steps[path]))
                if any(step <= 0 for step in partition_steps):
                    raise ValueError("measurement precision steps must be positive")
                quanta[field] = Decimal(lcm(*(s.numerator for s in partition_steps))) / Decimal(
                    gcd(*(s.denominator for s in partition_steps))
                )
        # Equal printed gross/net values in different units still describe one
        # physical magnitude. Round on their common representable lattice, not
        # independently in kg and tonnes (which can otherwise put net above gross).
        equal_mass = None
        mass_factors = {
            "kilogram": Decimal(1),
            "metric_tonne": Decimal(1000),
            "pound": Decimal("0.45359237"),
        }
        if {"grossWeight", "netWeight"} <= group.keys():
            masses = {
                field: Decimal(str(group[field]["value"])) * mass_factors[group[field]["unit"]]
                for field in ("grossWeight", "netWeight")
            }
            if masses["grossWeight"] == masses["netWeight"]:
                steps = [Fraction(quanta[f] * mass_factors[group[f]["unit"]]) for f in masses]
                step = Decimal(lcm(*(s.numerator for s in steps))) / Decimal(
                    gcd(*(s.denominator for s in steps))
                )
                equal_mass = max(
                    step,
                    (masses["grossWeight"] * ratio / step).quantize(
                        Decimal(1), rounding=ROUND_HALF_UP
                    )
                    * step,
                )
        for field in ("grossWeight", "netWeight", "volume"):
            if field not in group:
                continue
            old = group[field]["value"]
            quantum = quanta[field]
            new_measurement = max(
                quantum,
                (Decimal(str(old)) * ratio / quantum).quantize(Decimal(1), rounding=ROUND_HALF_UP)
                * quantum,
            )
            if equal_mass is not None and field != "volume":
                new_measurement = equal_mass / mass_factors[group[field]["unit"]]
            group[field]["value"] = (
                int(new_measurement)
                if isinstance(old, int) and new_measurement == new_measurement.to_integral_value()
                else float(new_measurement)
            )


def scenario_scale(sample_id: str, seed: int) -> Decimal:
    stream = DeterministicStream(seed, "complete-template-scenario-v1", sample_id)
    return Decimal(600 + stream.derive("cargo-scale").randbelow(350)) / 1000


def _is_lexical(path: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if path.startswith("documentPatch.parties.") and not path.startswith(
        "documentPatch.parties.carrier."
    ):
        return bool(
            re.search(
                r"\.(name|address|contactName)$|\.(phoneNumbers|emailAddresses|websiteUrls)\[", path
            )
        )
    return bool(
        re.search(
            r"\.cargoGroups\[\d+\]\.(description|additionalInformation\[|marksAndNumbers\[|handlingInstructions\[)|\.forwardingAndExportReferences\[",
            path,
        )
    )


def lexical_contract(source: SourceTemplate) -> tuple[dict[str, Any], ...]:
    from . import labelled_context

    leaves = render._flatten_leaves(source.target)
    paths = {path for path, value in leaves.items() if _is_lexical(path, value)}
    context_paths = labelled_context.owned_paths(source)
    paths.difference_update(context_paths)
    paths.difference_update(fixed_dimension_paths(source))
    paths.difference_update(p for b in _opaque_reference_bindings(source) for p in b.target_paths)
    formal_paths = plan_ranges(source.template, source.target, source.target).target_values
    paths.difference_update(p for p, value in formal_paths.items() if formal_range_text(value))
    _, package_summaries = package_prose.generate(source.template, source.target, source.target)
    paths.difference_update(package_summaries)
    paths.difference_update(measurement_prose.formal_paths(source.target))
    paths.difference_update(cargo_identifiers.labelled_references(source.target))
    for binding in source.template.bindings:
        if (
            binding.derivation is None
            and len(binding.target_paths) > 1
            and any(p not in paths for p in binding.target_paths)
        ):
            values = [render._binding_target_value(source.target, p) for p in binding.target_paths]
            if all(isinstance(v, (str, int, float)) and v == values[0] for v in values):
                paths.difference_update(binding.target_paths)
    paths = {p for p in paths if not order_party_reference(p, leaves[p])}
    parent = {path: path for path in paths}

    def root(path: str) -> str:
        while parent[path] != path:
            path = parent[path]
        return path

    for binding in source.template.bindings:
        selected = [path for path in binding.target_paths if path in paths]
        # A composite surface can print several DISTINCT facts (for example two
        # invoice references sharing a date). Ownership alone is not equality.
        owners: dict[bytes, str] = {}
        for path in selected:
            owner = owners.setdefault(canonical_json_bytes(leaves[path]), path)
            parent[root(path)] = root(owner)
    # Explicitly identical party objects retain their identity relationship.
    parties = source.target["documentPatch"].get("parties", {})
    signatures: dict[bytes, str] = {}
    for role, raw in parties.items():
        if role == "carrier":
            continue
        for index, party in enumerate(raw if isinstance(raw, list) else [raw]):
            prefix = f"documentPatch.parties.{role}" + (
                f"[{index}]" if isinstance(raw, list) else ""
            )
            signature = canonical_json_bytes(party)
            owner = signatures.setdefault(signature, prefix)
            if owner != prefix:
                for path in sorted(paths):
                    other = owner + path[len(prefix) :]
                    if path.startswith(prefix + ".") and other in paths:
                        parent[root(path)] = root(other)
    grouped: dict[str, list[str]] = {}
    for path in sorted(paths):
        grouped.setdefault(root(path), []).append(path)
    requests: list[dict[str, Any]] = []
    context_auxiliaries = projected_auxiliary_values(source.template, source.target)
    for group in grouped.values():
        partitions = [
            plan
            for binding in source.template.bindings
            if set(group).intersection(binding.target_paths)
            and (plan := lexical_partitions.partition(binding)) is not None
        ]
        if partitions:
            if len(partitions) != 1 or set(partitions[0].binding.target_paths) != set(group):
                raise ValueError("cargo fragment ownership differs from the lexical target group")
            requests.extend(
                lexical_partitions.requests(
                    partitions[0], source.source, key=f"field_{len(requests):04d}"
                )
            )
            continue
        constraints: list[dict[str, Any]] = []
        for binding in source.template.bindings:
            dependent_name_mark = (
                bool(context_paths.intersection(binding.target_paths))
                and len(binding.dependency_paths) == 1
                and binding.dependency_paths[0] in group
            )
            if not set(group).intersection(binding.target_paths) and not dependent_name_mark:
                continue
            intervals = complete_token_intervals(binding)
            character_frames = character_partition_frames(binding)
            constraints.append(
                {
                    "mode": binding.realization.mode,
                    "adapter": binding.realization.adapter,
                    "surfaces": [slot.source_text for slot in binding.occurrences],
                    "minimumWords": 1
                    if character_frames is not None
                    else sum(
                        len(slot.format_envelope.newline_sequence) + 1
                        for slot in binding.occurrences
                    )
                    if binding.realization.mode == "segmented_surface"
                    else max(
                        len(slot.format_envelope.newline_sequence) + 1
                        for slot in binding.occurrences
                    ),
                    "completeTokenPartition": intervals,
                    "mutableTokenIntervals": token_projection_intervals(binding),
                    "fixedLiteralTokens": [
                        list(row[2]) for row in fixed_projection_ranges(binding)
                    ],
                    "fixedProjectionFrames": []
                    if token_projection_intervals(binding) is not None
                    else [
                        {
                            "prefix": slot.required_target_prefix_tokens,
                            "suffix": slot.required_target_suffix_tokens,
                            "normalizedPrefix": slot.required_target_prefix_normalized,
                            "normalizedSuffix": slot.required_target_suffix_normalized,
                        }
                        for slot in binding.realization.slots
                    ],
                }
            )
            if character_frames is not None:
                constraints[-1]["characterPartition"] = True
                constraints[-1]["minimumCharacters"] = len(character_frames)
            projected = token_projection_intervals(binding)
            if projected is not None:
                boundaries = {i for pair in projected for i in pair}
                fixed_tokens = sum(len(row[2]) for row in fixed_projection_ranges(binding))
                constraints[-1]["minimumWords"] = max(
                    constraints[-1]["minimumWords"], len(boundaries) - 1 + fixed_tokens
                )
            if any(s.render_policy == "opaque_identifier" for s in binding.occurrences):
                constraints[-1]["surfacePatterns"] = [
                    s.format_envelope.exact_surface_pattern for s in binding.occurrences
                ]
        requests.append(
            {
                "key": f"field_{len(requests):04d}",
                "paths": group,
                "source": leaves[group[0]],
                "constraints": constraints,
            }
        )
        frame = immutable_lexical_frame(cast(str, leaves[group[0]]), constraints)
        if frame is not None:
            requests[-1]["hostAssembly"] = frame
    for entity in source.template.auxiliary_semantic_plan.entities:
        if entity.target_party_path is None:
            continue
        party = render._resolve_path(source.target, entity.target_party_path)
        for member in entity.members:
            if member.logical_key in context_auxiliaries:
                continue
            if member.field in {"registration_type"} or render._party_exposes_auxiliary_field(
                party, member.field
            ):
                continue
            binding = next(
                b for b in source.template.bindings if b.logical_key == member.logical_key
            )
            if deterministic_auxiliary_identifier(binding, member.field):
                continue
            if member.field == "email" and render._direct_auxiliary_route(binding)[0]:
                continue
            requests.append(
                {
                    "key": f"field_{len(requests):04d}",
                    "paths": [],
                    "auxiliaryKey": member.logical_key,
                    "partyPath": entity.target_party_path,
                    "entityField": member.field,
                    "source": binding.occurrences[0].source_text,
                    "constraints": [
                        {
                            "surfaces": [s.source_text for s in binding.occurrences],
                            "policies": [s.render_policy for s in binding.occurrences],
                        }
                    ],
                }
            )
    from . import cargo_identity_derivations

    cargo_identity_derivations.validate_source(source.template.bindings, source.target)
    for binding in source.template.bindings:
        if binding.derivation in cargo_identity_derivations.DERIVATIONS:
            requests.append(
                {
                    "key": f"field_{len(requests):04d}",
                    "paths": [],
                    "auxiliaryKey": binding.logical_key,
                    "cargoIdentityPath": binding.dependency_paths[0],
                    "source": binding.occurrences[0].source_text,
                    "constraints": [],
                }
            )
    return tuple(requests)


def fixed_dimension_paths(source: SourceTemplate) -> frozenset[str]:
    """Keep explicitly stated cargo geometry as a condition of the new product.

    This is a scenario choice like fixed equipment and refrigeration settings,
    not recovery from a failed generation. Quantities/weights are not dimensions.
    Any dependency-bearing or ambiguous declaration remains a linguistic request.
    """
    constrained = {
        key for c in source.template.coherence_constraints for key in c.member_logical_keys
    }
    result = set()
    for binding in source.template.bindings:
        if (
            binding.logical_key in constrained
            or binding.dependency_paths
            or binding.dependency_bindings
            or binding.source_relationships
            or binding.derivation
        ):
            continue
        for path in binding.target_paths:
            if not re.fullmatch(
                r"documentPatch\.cargoGroups\[\d+\]\.additionalInformation\[\d+\]", path
            ):
                continue
            value = render._binding_target_value(source.target, path)
            if isinstance(value, str) and re.fullmatch(
                r"(?:O/[HWL]|OVER(?:HEIGHT|WIDTH|LENGTH))\s+\d+(?:\.\d+)?\s+"
                r"(?:MM|CM|CMS|M|METRES?|METERS?|INCH(?:ES)?|IN|FT)",
                value,
                re.IGNORECASE,
            ):
                result.add(path)
    return frozenset(result)


def deterministic_auxiliary_identifier(binding: SemanticBinding, field: str) -> bool:
    """Use the existing shape/relationship solver for opaque registration facts."""
    return bool(
        (
            field in {"tax_identifier", "registration_identifier"}
            or (
                field == "other_identifier"
                and "acid" in re.findall(r"[a-z]+", binding.logical_key.lower())
            )
        )
        and not binding.target_paths
        and binding.value_kind == "identifier"
        and all(slot.render_policy == "opaque_identifier" for slot in binding.occurrences)
        and render._direct_auxiliary_route(binding)[0]
    )


def immutable_lexical_frame(
    source: str, constraints: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Assemble one mutable interval locally when every projection agrees.

    Multiple independent intervals remain an explicit linguistic request. This
    capability is based on compiler coordinates, never guessed source substrings.
    """
    projected = [c for c in constraints if c.get("fixedLiteralTokens")]
    if not projected:
        return None
    intervals = [
        tuple(dict.fromkeys(tuple(pair) for pair in c["mutableTokenIntervals"]))
        if c.get("mutableTokenIntervals") is not None
        else None
        for c in projected
    ]
    if any(value is None or len(value) != 1 for value in intervals):
        return None
    if any(value != intervals[0] for value in intervals[1:]):
        return None
    interval = intervals[0]
    assert interval is not None  # Proven by the capability check above.
    start, end = interval[0]
    tokens = render._token_spans(source)
    if not 0 <= start < end <= len(tokens):
        raise ValueError("lexical mutable interval exceeds source value")
    left, right = tokens[start][1], tokens[end - 1][2]
    # Token coordinates exclude punctuation. A mutable parenthesized qualifier
    # owns its closing punctuation too; otherwise a new name leaves a stray ')'.
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        while source[right : right + 1] == closing and source[left:right].count(opening) > source[
            left:right
        ].count(closing):
            right += 1
    # Compiler token coordinates are ASCII-normalized. If a boundary bisects
    # an accented word, the full-value linguistic contract must retain ownership;
    # never invent character ownership beyond the proven token interval.
    if any(
        c.isalnum() or unicodedata.combining(c)
        for c in source[max(0, left - 1) : left] + source[right : right + 1]
    ):
        return None
    return {
        "prefix": source[:left],
        "suffix": source[right:],
        "mutableSource": source[left:right],
        "minimumWords": max(
            1, max(c["minimumWords"] for c in constraints) - (len(tokens) - end + start)
        ),
    }


def assemble_lexical_value(request: Mapping[str, Any], value: str) -> str:
    if "hostAssembly" not in request:
        return value
    frame = request["hostAssembly"]
    prefix, suffix = cast(str, frame["prefix"]), cast(str, frame["suffix"])
    # Idempotent framing accepts a complete scalar as well as a mutable fragment.
    # Only an exact edge frame is already supplied; an interior locality is not.
    # This never substitutes source identity or drops a generated semantic token.
    if prefix and value.casefold().startswith(prefix.casefold()):
        value = value[len(prefix) :]
    if suffix and value.casefold().endswith(suffix.casefold()):
        value = value[: -len(suffix)]
    if not value.strip():
        raise ValueError("host-assembled lexical value has no mutable identity")
    return prefix + value + suffix


def lexical_repair_requirements(
    source: SourceTemplate, proposed: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Expose source-proven quantity ownership and field-local repair obligations.

    Error strings identify paths, but a model should not have to infer which of
    several nested package counts a prose field must retain. This is validation
    context, not permission to rewrite the structured scenario.
    """
    result = []
    for request in requests:
        obligations = []
        for path in request["paths"]:
            old = render._resolve_path(source.target, path)
            if not isinstance(old, str):
                continue
            group_match = re.match(r"documentPatch\.cargoGroups\[(\d+)\]\.", path)
            if group_match:
                group_id = proposed["documentPatch"]["cargoGroups"][int(group_match[1])]["groupId"]
                obligations.append(
                    {
                        "targetPath": path,
                        "declaredPackages": [
                            package
                            for package in proposed["documentPatch"].get("cargoPackages", ())
                            if package["groupId"] == group_id
                        ],
                        "requirement": (
                            "Do not invent packaging levels or counts. If you added an optional "
                            "count that validation rejected, remove that assertion or express the "
                            "declared package facts exactly. Required source mentions and complete "
                            "document totals below must remain and use their computed quantities."
                        ),
                    }
                )
            for mention in package_prose.mentions(source.target, path, old):
                obligations.append(
                    {
                        "sourceMention": old[mention.start : mention.noun_end],
                        "targetPath": path,
                        "quantityFacts": [
                            {"path": p, "value": render._resolve_path(proposed, p)}
                            for p in mention.quantity_paths
                        ],
                        "aggregation": "sum" if mention.aggregate else "same_value",
                        "requirement": (
                            "Express the new quantity with its same package kind here; "
                            "another package level is not a substitute."
                        ),
                    }
                )
        if "hostAssembly" in request:
            obligations.append(
                {
                    "immutableFrame": request["hostAssembly"],
                    "requirement": (
                        "Return only the mutable interior, without either fixed edge fragment."
                    ),
                }
            )
        for constraint in source.template.coherence_constraints:
            members = {
                p
                for b in source.template.bindings
                if b.logical_key in constraint.member_logical_keys
                for p in b.target_paths
            }
            if members.intersection(request["paths"]):
                obligations.append(
                    {
                        "coherenceContract": constraint.model_dump(mode="json"),
                        "dependencyValues": {
                            p: render._resolve_path(proposed, p)
                            for p in coherence_dependency_paths(constraint)
                        },
                        "requirement": (
                            "Preserve the supplied interval endpoints and their required "
                            "cardinalities. Do not append package-count annotations absent "
                            "from the source; the interval itself expresses the cardinality."
                            if constraint.kind
                            in {
                                "inclusive_range_cardinality",
                                "aggregate_inclusive_range_cardinality",
                            }
                            else "Express each dependent count as a separated number and "
                            "its package noun, not as part of a model number or identifier."
                        ),
                    }
                )
        for constraint in request["constraints"]:
            if "hostAssembly" in request:
                continue  # These constraints apply to the assembled target, not model output.
            if constraint.get("mutableTokenIntervals") and constraint.get("fixedLiteralTokens"):
                obligations.append(
                    {
                        "partitionContract": constraint,
                        "requirement": (
                            "Keep each fixed literal between nonempty mutable segments in source "
                            "order. Do not place an interior fixed literal at the end of the value."
                        ),
                    }
                )
        for binding in source.template.bindings:
            if (
                "hostAssembly" not in request
                and "sampledContext" not in request
                and set(binding.target_paths).intersection(request["paths"])
            ):
                pattern = token_projection_pattern(binding)
                if pattern is not None:
                    obligations.append({"partitionPattern": pattern})
        dependent_phones = {
            key
            for b in source.template.bindings
            if b.derivation == "same_as_binding"
            for key in b.dependency_bindings
        }
        for binding in source.template.bindings:
            if (
                binding.value_kind == "phone"
                and binding.logical_key in dependent_phones
                and set(binding.target_paths).intersection(request["paths"])
            ):
                obligations.append(
                    {
                        "opaqueShapes": [
                            {
                                "source": binding.occurrences[0].source_text,
                                "pattern": surface_pattern(binding.occurrences[0].source_text),
                            }
                        ],
                        "requirement": (
                            "Retain this local phone's exact digit width and punctuation; the host "
                            "supplies its existing international/area prefix in dependent "
                            "occurrences."
                        ),
                    }
                )
        if request.get("auxiliaryKey") and "cargoFragment" not in request:
            binding = next(
                b for b in source.template.bindings if b.logical_key == request["auxiliaryKey"]
            )
            shapes = [
                {
                    "source": s.source_text,
                    "pattern": s.format_envelope.exact_surface_pattern,
                    "alphanumericCount": len(render._alphanumeric(s.source_text)),
                }
                for s in binding.occurrences
                if s.render_policy == "opaque_identifier"
            ]
            if shapes:
                obligations.append(
                    {
                        "opaqueShapes": shapes,
                        "requirement": (
                            "Preserve the identifier's exact alphanumeric width and structure, "
                            "including its country/designator prefix; change only the private "
                            "identifier value."
                        ),
                    }
                )
        if any(
            ".additionalInformation[" in p or ".marksAndNumbers[" in p for p in request["paths"]
        ):
            obligations.append(
                {
                    "sourceRole": request["source"],
                    "requirement": (
                        "Keep this entry's distinct source role; do not duplicate any sibling "
                        "mark or additionalInformation entry, change a reference label, "
                        "or replace a declared origin with a different "
                        "origin."
                    ),
                }
            )
        if obligations:
            result.append({"key": request["key"], "obligations": obligations})
    return result


def _opaque_reference_bindings(source: SourceTemplate) -> tuple[SemanticBinding, ...]:
    result = []
    related = {dependency for b in source.template.bindings for dependency in b.dependency_bindings}
    for binding in source.template.bindings:
        if (
            binding.source_relationships
            or binding.logical_key in related
            or binding.realization.mode not in {"single_surface", "repeated_surface"}
            or not binding.target_paths
            or not all(
                p.startswith("documentPatch.forwardingAndExportReferences[")
                for p in binding.target_paths
            )
            or not all(s.render_policy == "opaque_identifier" for s in binding.occurrences)
        ):
            continue
        values = [render._binding_target_value(source.target, p) for p in binding.target_paths]
        if all(
            isinstance(v, str) and v == values[0] and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/.-]*", v)
            for v in values
        ):
            result.append(binding)
    return tuple(result)


def structured_proposal(
    source: SourceTemplate,
    *,
    sample_id: str,
    seed: int,
    allocations: Mapping[str, str],
    vessel_names: Sequence[str],
    minimum_quantities: Mapping[str, int] | None = None,
    quantity_multiples: Mapping[str, int] | None = None,
    measurement_steps: Mapping[str, Decimal] | None = None,
) -> dict[str, Any]:
    target = deepcopy(source.target)
    references = target["documentPatch"].pop("forwardingAndExportReferences", None)
    apply_identifier_plan(document_id=sample_id, target=target, allocations=allocations)
    if references is not None:
        target["documentPatch"]["forwardingAndExportReferences"] = references
    stream = DeterministicStream(seed, "complete-template-scenario-v1", sample_id)
    for path, reference in cargo_identifiers.generate_references(
        source.target, stream, source.template
    ).items():
        _set(target, path, reference)
    for binding in _opaque_reference_bindings(source):
        value = render._binding_target_value(source.target, binding.target_paths[0])
        if not isinstance(value, str):
            raise ValueError("opaque reference source is not a string")
        generated = generate_from_surface_pattern(
            pattern=surface_pattern(value),
            stream=stream.derive(binding.logical_key),
            additional_excluded=value,
        )
        for path in binding.target_paths:
            _set(target, path, generated)
    _scale_numbers(
        source,
        target,
        stream,
        minimum_quantities or {},
        quantity_multiples or {},
        measurement_steps,
    )
    for path, package_mass in package_prose.package_mass_values(
        source.template, source.target, target
    ).items():
        _set(target, path, float(package_mass))
    for path, package_mass in measurement_prose.per_package_totals(source.target, target).items():
        _set(target, path, float(package_mass))
    for path, value in plan_ranges(source.template, source.target, target).target_values.items():
        _set(target, path, value)
    prose_source = deepcopy(source.target)
    for path, value in measurement_prose.generate(source.target, target).items():
        _set(target, path, value)
        _set(prose_source, path, value)
    for path, value in package_prose.generate(source.template, prose_source, target)[0].items():
        _set(target, path, value)
    for path, value in package_equations.one_to_one_target_surfaces(source.target, target).items():
        _set(target, path, value)
    patch = target["documentPatch"]
    # Chronology is a fixed scenario context, including source-only invoice,
    # regulatory and audit dates. Independent date randomization breaks that context.
    transport = patch.get("transport", {})
    if "vesselName" in transport and "vesselImoNumber" not in transport:
        candidates = [v for v in vessel_names if v != transport["vesselName"]]
        if not candidates:
            raise ValueError("vessel-name registry has no source-distinct candidate")
        transport["vesselName"] = candidates[stream.derive("vessel").randbelow(len(candidates))]
    # A shared printed scalar represents one fact, even if multiple target paths
    # name it (for example a seal repeated as a cargo mark).
    for binding in source.template.bindings:
        if len(binding.target_paths) < 2 or binding.derivation is not None:
            continue
        old = [render._binding_target_value(source.target, path) for path in binding.target_paths]
        if not all(isinstance(value, (str, int, float)) and value == old[0] for value in old):
            continue
        changed = [
            render._binding_target_value(target, path)
            for path, value in zip(binding.target_paths, old, strict=True)
            if render._binding_target_value(target, path) != value
        ]
        if changed:
            if any(value != changed[0] for value in changed):
                raise ValueError(
                    f"structured generators disagree on shared printed fact: {binding.logical_key}"
                )
            for path in binding.target_paths:
                _set(target, path, changed[0])
    return target


def propose_goods(
    target: dict[str, Any], *, registry: UkGlobalTariffRegistry, sample_id: str, seed: int
) -> dict[str, Any]:
    """Change tariff-backed identities within their observed product heading.

    National extensions remain fixed without an explicit tariff jurisdiction.
    DG identities remain coupled to their original UN/class declaration. Commodity
    variants in that branch are lexical, not unverified chemical reclassification.
    """
    stream = DeterministicStream(seed, "template-goods-identities-v1", sample_id)
    scenario_date_text = target["documentPatch"].get("shippedOnBoardDate") or target[
        "documentPatch"
    ].get("issueDate")
    scenario_date = date.fromisoformat(scenario_date_text) if scenario_date_text else None
    global_codes = set(registry.global_codes)
    occupied_codes = {
        "".join(c for c in code if c.isdigit())
        for group in target["documentPatch"].get("cargoGroups", [])
        for code in group.get("hsCodes", [])
    }
    selected_codes: dict[str, str] = {}
    fixed_national_roots = {code[:6] for code in occupied_codes if len(code) > 6}
    hazard_codes = {
        "".join(c for c in code if c.isdigit())
        for g in target["documentPatch"].get("cargoGroups", [])
        if g.get("dangerousGoods")
        for code in g.get("hsCodes", [])
    }
    descriptions = {}
    for group in target["documentPatch"].get("cargoGroups", []):
        rows = []
        for code_index, source_code in enumerate(group.get("hsCodes", [])):
            digits = "".join(c for c in source_code if c.isdigit())
            if digits in hazard_codes or cargo_identifiers.vehicle_ids(
                group.get("description", "")
            ):
                rows.append(
                    {
                        "code": source_code,
                        "reason": (
                            "fixed source vehicle or UN/class/tariff identity; vary commercial "
                            "variant, serials and quantities within this classification"
                        ),
                    }
                )
                continue
            if len(digits) != 6:
                rows.append(
                    {
                        "code": source_code,
                        "reason": (
                            "national tariff jurisdiction is not declared; "
                            "source classification retained"
                        ),
                    }
                )
                continue
            if digits in fixed_national_roots:
                row = {
                    "code": source_code,
                    "reason": "global identity is shared with a fixed national tariff extension",
                }
                if digits in global_codes:
                    identity = registry.require_global(
                        digits, on_date=registry.receipt.snapshot_date
                    )
                    row.update(
                        heading=identity.heading_description, description=identity.description
                    )
                rows.append(row)
                continue
            old_global = (
                registry.require_global(digits[:6], on_date=registry.receipt.snapshot_date)
                if digits[:6] in global_codes
                else None
            )
            if (
                old_global is None
                or scenario_date is None
                or not old_global.valid_from <= scenario_date <= registry.receipt.snapshot_date
            ):
                rows.append(
                    {
                        "code": source_code,
                        "reason": (
                            "source chronology or classification is outside "
                            "the pinned registry's verified support"
                        ),
                    }
                )
                continue
            thermal = classify_thermal_hs(old_global) if old_global is not None else None
            candidates = [
                code
                for code in registry.global_codes
                if code[:4] == digits[:4]
                and classify_thermal_hs(
                    registry.require_global(code, on_date=registry.receipt.snapshot_date)
                )
                == thermal
            ]
            exact_codes = [
                code
                for code in candidates
                if registry.require_global(code, on_date=registry.receipt.snapshot_date).valid_from
                <= scenario_date
            ]
            exact_codes = sorted(set(exact_codes) - occupied_codes)
            if not exact_codes and digits not in selected_codes:
                rows.append(
                    {
                        "code": source_code,
                        "reason": (
                            "no distinct compatible identity in the pinned same-heading support"
                        ),
                    }
                )
                continue
            selected = selected_codes.get(digits)
            if selected is None:
                selected = exact_codes[stream.derive(digits).randbelow(len(exact_codes))]
                selected_codes[digits] = selected
                occupied_codes.add(selected)
            group["hsCodes"][code_index] = render._shape_alphanumeric_like_source(
                source_code, selected
            )
            semantic = registry.require_global(selected[:6], on_date=registry.receipt.snapshot_date)
            rows.append(
                {
                    "code": group["hsCodes"][code_index],
                    "heading": semantic.heading_description,
                    "description": semantic.description,
                    "thermalProfile": thermal,
                    "basis": "pinned same-heading tariff identity",
                }
            )
        descriptions[group["groupId"]] = rows
    return descriptions


def complete_proposal(
    source: SourceTemplate,
    proposal: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    values: Mapping[str, str],
    *,
    sample_id: str,
    prepared_auxiliary: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    if set(values) != {row["key"] for row in requests}:
        raise ValueError("linguistic response does not cover the requested fields exactly")
    target = deepcopy(dict(proposal))
    auxiliary = dict(prepared_auxiliary or {})
    for request in requests:
        value = values[request["key"]]
        if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
            raise ValueError(
                f"linguistic value must be nonempty single-line text: {request['key']}"
            )
        if re.search(r"\bPACKAGE_[A-Z_]+\b", value):
            raise ValueError(
                "linguistic field contains an internal package enum; use printed names"
            )
        value = assemble_lexical_value(request, value)
        cargo_identifiers.validate(request, value)
        for path in request["paths"]:
            _set(target, path, value)
        if "auxiliaryKey" in request:
            if request["auxiliaryKey"] in auxiliary:
                raise ValueError(
                    "host-owned auxiliary geography was incorrectly requested from the model"
                )
            auxiliary[request["auxiliaryKey"]] = value
            if "cargoFragment" not in request:
                binding = next(
                    b for b in source.template.bindings if b.logical_key == request["auxiliaryKey"]
                )
                try:
                    output = render._render_text_candidate(binding, value)
                    render._validate_binding_format(
                        source=source.source, template=source.template.byte_template, output=output
                    )
                except ValueError as error:
                    raise ValueError(
                        f"invalid auxiliary field {request['key']}: {error}"
                    ) from error
    for path, text in lexical_partitions.assembled_targets(source.template, auxiliary).items():
        _set(target, path, text)
    from . import labelled_context

    context_labels = labelled_context.generated_values(source, target)
    if any(path in context_labels for request in requests for path in request["paths"]):
        raise ValueError("derived cargo labels were incorrectly requested from the model")
    for path, text in context_labels.items():
        _set(target, path, text)
    for key, value in projected_auxiliary_values(source.template, target).items():
        if key in auxiliary:
            raise ValueError("derived projection context was incorrectly requested from the model")
        auxiliary[key] = value
    canonical = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(document_id=sample_id, target=target)
    if canonical != target:
        raise ValueError("complete target is not already canonical")
    render._require_source_carrier(source_target=source.source_target, target=target)
    if printed_topology_mismatches(source.target, target):
        raise ValueError("complete generation changed printed topology")
    package_prose.validate(source.target, target, template=source.template)
    package_prose.validate_package_masses(source.template, source.target, target)
    from . import temperature_prose

    temperature_prose.validate(source.template, source.target, target)
    measurement_prose.validate(source.target, target)
    fixed_quantities = count_aliases.fixed_quantities(source.source, source.template, source.target)
    if any(render._resolve_path(target, p) != v for p, v in fixed_quantities.items()):
        raise ValueError("generated quantity conflicts with an immutable printed count alias")
    validate_render_coherence(
        bindings=source.template.bindings,
        constraints=source.template.coherence_constraints,
        source_target=source.source_target,
        target=target,
        outputs=None,
    )
    render._validate_target_compatibility(
        source=source.source,
        source_target=source.source_target,
        target=target,
        template=source.template,
        auxiliary_values=auxiliary,
    )
    original, generated = render._flatten_leaves(source.target), render._flatten_leaves(target)
    changes = {
        p: {"source": original[p], "accepted": v} for p, v in generated.items() if original[p] != v
    }
    required = require_complete_variation(source.target, target, bindings=source.template.bindings)
    fixed_roles = fixed_carrier_role_paths(source.target, source.template.bindings)
    fixed_dimensions = fixed_dimension_paths(source)
    fixed_references = cargo_identifiers.fixed_references(source.target, source.template)
    retained = {
        p: {
            "value": v,
            "reason": (
                "explicitly printed shared role of the template-bound carrier"
                if p in fixed_roles
                else "explicit cargo geometry constrains the new commercial product"
                if p in fixed_dimensions
                else "immutable printed count alias fixes this quantity before generation"
                if p in fixed_quantities
                else "reference code is outside mutable compiled spans; only its label is owned"
                if p in fixed_references
                else _retention_reason(p, v)
            ),
        }
        for p, v in generated.items()
        if p not in changes
    }
    return (
        target,
        auxiliary,
        {
            "requestedPaths": required,
            "changes": changes,
            "retained": retained,
            "targetSha256": sha256_bytes(canonical_json_bytes(target)),
            "sourceTemplateSha256": source.original_template_sha256,
            "targetDerivedAuxiliaryContext": projected_auxiliary_values(source.template, target),
            "targetDerivedLabelContext": context_labels,
            "effectiveTemplateSha256": sha256_bytes(
                canonical_json_bytes(source.template.model_dump(mode="json"))
            ),
        },
    )


def _retention_reason(path: str, value: Any) -> str:
    if path in {"documentPatch.transport.vesselName", "documentPatch.transport.vesselImoNumber"}:
        return (
            "vessel identity is retained where an IMO number requires "
            "a verified name-identity pairing"
        )
    if path in {"documentPatch.issueDate", "documentPatch.shippedOnBoardDate"}:
        return "complete template chronology is deliberately retained as scenario context"
    if path.startswith("documentPatch.parties.carrier."):
        return "carrier identity is fixed by template"
    if order_party_reference(path, value):
        return "legal order or party-reference instruction, not an organization identity"
    if (
        re.search(r"\.(groupId|packageId|coverage|relation|schemaVersion)$|\.packageIds\[", path)
        or path == "schemaVersion"
    ):
        return "printed relational topology is fixed"
    if re.search(r"\.(countryCode|city|postalCode|region)$|\.locations\.|\.ports\.", path):
        return "scenario uses template geography; dependent addresses are newly generated"
    if ".hsCodes[" in path:
        return (
            "fixed DG identity or no verified jurisdiction/date-compatible distinct tariff code; "
            "see goods identity ledger"
        )
    if ".dangerousGoods" in path:
        return (
            "hazard identity and regulatory declarations stay coupled; "
            "commercial variant and quantities change"
        )
    if "temperature" in path.lower():
        return (
            "preserved goods-compatible thermal setting; cargo variant conditioned on this setting"
        )
    if re.search(r"\.(quantity|packageQuantity)$", path):
        return "integer indivisibility at preserved allocation support"
    if _is_lexical(path, value):
        return (
            "non-identity lexical instruction retained by generation; "
            "identity and cargo novelty checked separately"
        )
    return "fixed source category, equipment capacity, unit or structural metadata"
