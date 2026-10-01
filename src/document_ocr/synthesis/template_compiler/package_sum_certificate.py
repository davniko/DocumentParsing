"""Prove printed package sums before admitting a compiled template."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .generation_contract import leaves
from .models import SemanticBinding


def _set_numeric_path(document: dict[str, Any], path: str, value: int) -> None:
    parts = re.findall(r"[^.\[\]]+|\[[0-9]+\]", path)
    current: Any = document
    for part in parts[:-1]:
        current = current[int(part[1:-1])] if part.startswith("[") else current[part]
    last = parts[-1]
    if last.startswith("["):
        current[int(last[1:-1])] = value
    else:
        current[last] = value


def _change_one_component(
    binding: SemanticBinding,
    source_target: Mapping[str, Any],
    by_key: Mapping[str, SemanticBinding],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    from . import descendant

    changed = copy.deepcopy(dict(source_target))
    paths = list(binding.dependency_paths)
    for key in binding.dependency_bindings:
        paths.extend(by_key[key].target_paths)
    for path in paths:
        for leaf_path, value in leaves(descendant._resolve_path(source_target, path), path).items():
            if not leaf_path.endswith((".quantity", ".packageQuantity")):
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                continue
            aliases = {leaf_path}
            for key in binding.dependency_bindings:
                dependency = by_key[key]
                if leaf_path in dependency.target_paths:
                    aliases.update(
                        alias
                        for alias in dependency.target_paths
                        if descendant._resolve_path(source_target, alias) == value
                    )
                    outputs[key] = descendant.BindingOutput(
                        replacements={}, canonical_value=value + 1
                    )
            for alias in aliases:
                _set_numeric_path(changed, alias, value + 1)
            return changed
    for key in binding.dependency_bindings:
        if key not in outputs:
            continue
        value = descendant._numeric_value(outputs[key].canonical_value)
        outputs[key] = descendant.BindingOutput(
            replacements={},
            canonical_value=int(value + 1)
            if value == value.to_integral_value()
            else float(value + 1),
        )
        return changed
    raise ValueError(f"package sum has no mutable numeric component: {binding.logical_key}")


def certify_source(
    *, bindings: Sequence[SemanticBinding], source_target: Mapping[str, Any]
) -> None:
    """Use the production arithmetic renderer to reject untrue source sums.

    Some source-only per-container counts are independent auxiliary bindings.
    Their exact repeated printed values supply the source-side numeric inputs;
    the same production renderer then verifies every aggregate occurrence.
    """
    from . import descendant

    by_key = {binding.logical_key: binding for binding in bindings}
    if len(by_key) != len(bindings):
        raise ValueError("package sum certificate requires unique logical keys")
    for binding in bindings:
        if binding.derivation != "sum_package_quantity":
            continue
        outputs: dict[str, descendant.BindingOutput] = {}
        for key in binding.dependency_bindings:
            dependency = by_key[key]
            try:
                values = {
                    descendant._numeric_value(slot.source_text)
                    for slot in dependency.occurrences
                }
            except ValueError:
                # A dependency also can be a structured path whose values are
                # read directly by the arithmetic renderer. If it is needed as
                # a scalar, the missing output fails closed below.
                continue
            if len(values) != 1:
                raise ValueError(
                    f"package sum dependency has inconsistent printed numbers: {key}"
                )
            value = next(iter(values))
            if not value.is_finite():
                raise ValueError(f"package sum dependency is not finite: {key}")
            outputs[key] = descendant.BindingOutput(
                replacements={},
                canonical_value=int(value) if value == value.to_integral_value() else float(value),
            )
        try:
            old, same = descendant._derivation_numeric_values(
                binding=binding,
                source_target=source_target,
                target=source_target,
                bindings=by_key,
                outputs=outputs,
            )
            if old != same:
                raise ValueError("unchanged package sum produced different numeric values")
            descendant._render_proven_numeric_derivation(
                binding, source_value=old, target_value=same
            )
            changed_target = _change_one_component(
                binding, source_target, by_key, outputs
            )
            source_again, changed = descendant._derivation_numeric_values(
                binding=binding,
                source_target=source_target,
                target=changed_target,
                bindings=by_key,
                outputs=outputs,
            )
            if source_again != old or changed == old:
                raise ValueError("package sum did not react to a changed component")
            output = descendant._render_proven_numeric_derivation(
                binding, source_value=old, target_value=changed
            )
            if any(
                output.replacements[slot.slot_id] == slot.source_text
                for slot in binding.occurrences
            ):
                raise ValueError("changed package sum retained its printed source value")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"printed package sum fails source arithmetic: {binding.logical_key}: {error}"
            ) from error
