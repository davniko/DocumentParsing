"""Reviewed source declarations for full-scenario current-label synthesis.

Only this data layer knows source-specific ownership decisions. Execution uses
the same exact-quote, target-path and disjoint-byte rules for every family.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from document_ocr.synthesis.curated import (
    Occurrence,
    SourceContract,
    TargetBinding,
    Variable,
    assign,
    flat,
    locate,
)
from document_ocr.synthesis.curated_measurements import converted_measure_surfaces
from document_ocr.synthesis.curated_templates import (
    LexicalOwnership,
    SamplingBlueprint,
    compile_sampling_blueprint,
    current_path,
    substitute_expression_literals,
)
from document_ocr.synthesis.template_compiler.descendant import _package_candidate


def _occurrences(raw: str, selectors: list[dict]) -> list[Occurrence]:
    result = []
    for selector in selectors:
        text = selector["text"]
        matches = list(re.finditer(re.escape(text), raw))
        if not matches:
            raise ValueError(f"reviewed ownership quote is absent: {text!r}")
        indices = (
            range(1, len(matches) + 1) if selector.get("all") else [selector.get("occurrence", 1)]
        )
        for index in indices:
            occurrence = Occurrence(text=text, occurrence=index, presentation="text")
            locate(raw, occurrence)
            result.append(occurrence)
    return result


def _historical_occurrences(raw: str, selectors: list[dict]) -> list[dict]:
    result = []
    for occurrence in _occurrences(raw, selectors):
        start, end = locate(raw, occurrence)
        result.append(dict(byte_start=start, byte_end=end, source_text=occurrence.text))
    return result


def build_owned_blueprint(
    row: dict, historical: dict, contract: SourceContract, declarations: dict
) -> SamplingBlueprint:
    """Apply explicit source ownership corrections and verify current bytes."""
    raw, leaves = row["joinedRawText"], flat(row["target"])
    if declarations.get("variables") or declarations.get("expressions"):
        variables = list(contract.variables)
        keys = {v.key for v in variables}
        for item in declarations.get("variables", []):
            if item["key"] in keys:
                raise ValueError(f"new source variable duplicates existing key: {item['key']}")
            variable = Variable.model_validate(item)
            variables.append(variable)
            keys.add(variable.key)
        expressions = declarations.get("expressions", {})
        unknown_paths = set(expressions) - {t.path for t in contract.targets}
        if unknown_paths:
            raise ValueError(
                f"expression correction has unknown current paths: {sorted(unknown_paths)}"
            )
        targets = [
            TargetBinding(path=t.path, expression=expressions.get(t.path, t.expression))
            for t in contract.targets
        ]
        contract = contract.model_copy(update={"variables": variables, "targets": targets})
    historical = deepcopy(historical)
    by_key = {b["logical_key"]: b for b in historical["bindings"]}
    for key, reason in declarations.get("retire", {}).items():
        if key not in by_key or not reason.strip():
            raise ValueError(f"retired binding requires existing key and reason: {key}")
        del by_key[key]
    for key, update in declarations.get("bindings", {}).items():
        if key not in by_key:
            raise ValueError(f"binding correction has unknown key: {key}")
        if "paths" in update:
            if not set(update["paths"]) <= leaves.keys():
                raise ValueError(f"binding correction has unknown current paths: {key}")
            by_key[key]["target_paths"] = update["paths"]
        if "occurrences" in update:
            by_key[key]["occurrences"] = _historical_occurrences(raw, update["occurrences"])
        if "kind" in update:
            by_key[key]["value_kind"] = update["kind"]
    for declaration in declarations.get("add", []):
        key = declaration["key"]
        if key in by_key:
            raise ValueError(f"duplicate added ownership key: {key}")
        paths = declaration.get("paths", [])
        if not set(paths) <= leaves.keys():
            raise ValueError(f"added ownership has unknown current paths: {key}")
        by_key[key] = dict(
            logical_key=key,
            target_paths=paths,
            value_kind=declaration.get("kind", "other_text"),
            occurrences=_historical_occurrences(raw, declaration["occurrences"]),
        )
    # Equipment receipts and thermal structs are established typed relationships,
    # not an excuse to treat every parent-object span as every child scalar.
    for binding in by_key.values():
        expanded = set(binding["target_paths"])
        for path in binding["target_paths"]:
            path = current_path(path)
            if binding.get("derivation") == "equipment_receipt":
                expanded.update(
                    p
                    for p in leaves
                    if p.startswith(path) and p.endswith((".sizeCategory", ".typeCategory"))
                )
            elif path.endswith(".temperatureSetpoint"):
                expanded.update(p for p in leaves if p.startswith(path + "."))
        binding["target_paths"] = sorted(expanded)
    historical["bindings"] = list(by_key.values())
    overrides = tuple(
        LexicalOwnership(
            key=item["key"],
            kind=item["kind"],
            target_paths=tuple(item["paths"]),
            occurrences=tuple(_occurrences(raw, item["occurrences"])),
        )
        for item in declarations.get("lexical", [])
    )
    blueprint = compile_sampling_blueprint(row, historical, contract, ownership_overrides=overrides)
    target_expressions = {
        binding.path: binding.expression for binding in blueprint.contract.targets
    }
    for path, expression in declarations.get("render_expressions", {}).items():
        if path not in target_expressions or not path.endswith(".addressLine"):
            raise ValueError(f"postal order correction requires an owned address target: {path}")
        original = target_expressions[path]
        if sorted(re.findall(r"\{[^{}]+\}", expression)) != sorted(
            re.findall(r"\{[^{}]+\}", original)
        ) or " ".join(re.sub(r"\{[^{}]+\}", "", expression).split()) != " ".join(
            re.sub(r"\{[^{}]+\}", "", original).split()
        ):
            raise ValueError(f"postal order correction changes facts rather than order: {path}")
    historical_owned_paths = {
        path
        for region in blueprint.regions
        if region.curated_key is None
        for path in region.target_paths
    }
    targets = [
        binding
        for binding in blueprint.contract.targets
        if not (
            "{" not in binding.expression
            and binding.path in historical_owned_paths
            and isinstance(leaves[binding.path], (int, float))
            and not isinstance(leaves[binding.path], bool)
        )
    ]
    contract = blueprint.contract.model_copy(
        update={
            "targets": targets,
            "variables": [
                variable.model_copy(update={"required_literals": []})
                for variable in blueprint.contract.variables
            ],
            "fixed_context": (
                "Current-schema scenario supplies shipment facts. Reuse source ownership, "
                "line span "
                "and printed topology; do not retain original countries or commodity identities."
            ),
        }
    )
    blueprint = replace(blueprint, contract=contract)
    # Country labels already present in current gold can be grounded inside a
    # complete postal owner, including when an old template omitted that label.
    regions = []
    for region in blueprint.regions:
        paths = set(region.target_paths)
        for path in tuple(paths):
            if path.endswith(".addressLine"):
                country_path = path.rsplit(".", 1)[0] + ".country"
                country = leaves.get(country_path)
                if country and re.search(
                    r"(?<!\w)" + re.escape(country) + r"(?!\w)", region.source, re.I
                ):
                    paths.add(country_path)
        allow_empty = region.key in declarations.get("delete", {})
        regions.append(replace(region, target_paths=tuple(sorted(paths)), allow_empty=allow_empty))
    blueprint = replace(blueprint, regions=tuple(regions), ownership_data=declarations)
    if blueprint.blocked_bindings:
        raise ValueError(f"unresolved source ownership overlaps: {blueprint.blocked_bindings}")
    return blueprint


def load_owned_blueprint(
    row: dict, historical: dict, contract: SourceContract, path: Path
) -> SamplingBlueprint:
    payload = yaml.safe_load(path.read_text())
    if payload.get("version") != 1 or row["documentId"] not in payload["sources"]:
        raise ValueError("full-sampling ownership inventory does not cover this source")
    return build_owned_blueprint(row, historical, contract, payload["sources"][row["documentId"]])


def apply_dependent_text(blueprint: SamplingBlueprint, target: dict) -> dict:
    """Rebind reviewed handling clauses to sampled route names without changing facts."""
    declarations = blueprint.ownership_data.get("dependent_text", [])
    if not isinstance(declarations, list):
        raise ValueError("dependent_text must be a list of reviewed handling clauses")
    if not declarations:
        return target
    source, sampled = flat(blueprint.target), flat(target)
    owned = {path for region in blueprint.regions for path in region.target_paths}
    updates: dict[str, str] = {}
    for item in declarations:
        if not isinstance(item, dict) or set(item) != {"path", "source", "expression"}:
            raise ValueError("dependent text requires exactly path, source and expression")
        path, original, expression = item["path"], item["source"], item["expression"]
        if not all(isinstance(v, str) and v.strip() for v in (path, original, expression)):
            raise ValueError("dependent text fields must be nonempty strings")
        if (
            not re.fullmatch(
                r"documentPatch\.goodsItemDetails\[\d+\]\.handlingInstructions\[\d+\]", path
            )
            or path not in owned
        ):
            raise ValueError(f"dependent text requires an owned handling instruction: {path}")
        if path in updates:
            raise ValueError(f"duplicate dependent text path: {path}")
        if source.get(path) != original:
            raise ValueError(f"dependent text source differs from pinned target: {path}")
        references = re.findall(r"\{([^{}]+)\}", expression)
        literal = re.sub(r"\{[^{}]+\}", "", expression)
        if not references or "{" in literal or "}" in literal:
            raise ValueError(f"dependent text expression has invalid placeholders: {path}")
        for reference in references:
            if not re.fullmatch(r"documentPatch\.route\.[A-Za-z][A-Za-z0-9]*\.name", reference):
                raise ValueError(f"dependent text only accepts route name references: {reference}")
            if any(
                not isinstance(leaves.get(reference), str) or not leaves[reference].strip()
                for leaves in (source, sampled)
            ):
                raise ValueError(f"dependent text route name is absent or invalid: {reference}")
        # The reviewed expression may replace a printed alias with the canonical
        # route name; the exact source declaration above pins that decision.
        value = re.sub(r"\{([^{}]+)\}", lambda m: sampled[m[1]], expression)
        if sampled.get(path) not in (original, value):
            raise ValueError(f"dependent text handling instruction is absent or conflicts: {path}")
        updates[path] = value
    # Validate the whole declaration set first; a rejected dependency cannot leave
    # a partially changed scenario target behind.
    result = deepcopy(target)
    for path, value in updates.items():
        assign(result, path, value)
    return result


def ownership_surfaces(
    blueprint: SamplingBlueprint, target: dict, scenario: Any
) -> dict[str, str | list[str]]:
    """Render explicitly declared source-bound dependencies and unit aliases."""
    leaves = flat(target)
    result = dict(blueprint.ownership_data.get("constants", {}))
    result.update({key: "" for key in blueprint.ownership_data.get("delete", {})})
    for key, recipe in blueprint.ownership_data.get("surfaces", {}).items():
        if "measure_path" in recipe:
            result[key] = converted_measure_surfaces(blueprint, target, key, recipe)
        elif "expression" in recipe:
            result[key] = re.sub(r"\{([^{}]+)\}", lambda m: str(leaves[m[1]]), recipe["expression"])
        elif "split_path" in recipe:
            parts = leaves[recipe["split_path"]].split(recipe["separator"])
            if len(parts) != recipe["count"]:
                raise ValueError(f"generated combined contact has wrong name count: {key}")
            result[key] = parts[recipe["index"]]
        elif "date_path" in recipe:
            path = recipe["date_path"]
            source_date = date.fromisoformat(flat(blueprint.target)[path])
            occurrences = blueprint.historical_bindings[key]["occurrences"]
            formats = recipe.get("formats", [recipe.get("format")] * len(occurrences))
            if len(formats) != len(occurrences) or any(not f for f in formats):
                raise ValueError(f"date format count differs from source occurrences: {key}")
            rendered = []
            for occurrence, date_format in zip(occurrences, formats, strict=True):
                original = _declared_date_surface(source_date, date_format)
                if original.casefold() != occurrence["source_text"].casefold():
                    raise ValueError(f"declared date format disagrees with source: {key}")
                value = _declared_date_surface(date.fromisoformat(leaves[path]), date_format)
                rendered.append(value.upper() if recipe.get("case") == "upper" else value)
            result[key] = rendered
        elif "package_path" in recipe:
            path = recipe["package_path"]
            quantity = leaves[path.rsplit(".", 1)[0] + ".packageQuantity"]
            # This recipe deliberately replaces a malformed/generic source noun
            # with a complete noun agreeing with the same package row's count.
            value = _package_candidate("PACKAGES", leaves[path], quantity=quantity)
            result[key] = [value for _ in blueprint.historical_bindings[key]["occurrences"]]
        else:
            raise ValueError(f"unknown source surface recipe: {key}")
    for key, binding in blueprint.historical_bindings.items():
        if key in result:
            continue
        paths = [current_path(p) for p in binding.get("target_paths", [])]
        paths = [p for p in paths if p in leaves]
        if len(set(paths)) != 1:
            continue
        path = paths[0]
        if ".parties." in path and path.endswith(".country"):
            role = path.rsplit(".", 1)[0]
            geo = scenario.party_localities[role]
            result[key] = [
                geo.country_code
                if re.fullmatch(r"[A-Z]{2}", o["source_text"].strip())
                else str(leaves[path])
                for o in binding["occurrences"]
            ]
        elif path.endswith((".name", ".country")) and any(
            x in path for x in (".route.", ".placeOfIssue.", ".freight.paymentPlace.")
        ):
            result[key] = str(leaves[path])
    return result


def _declared_date_surface(value: date, date_format: str) -> str:
    """Render reviewed date typography, including a fully spelled day ordinal."""
    if "{day_ordinal}" in date_format:
        words = [
            "FIRST",
            "SECOND",
            "THIRD",
            "FOURTH",
            "FIFTH",
            "SIXTH",
            "SEVENTH",
            "EIGHTH",
            "NINTH",
            "TENTH",
            "ELEVENTH",
            "TWELFTH",
            "THIRTEENTH",
            "FOURTEENTH",
            "FIFTEENTH",
            "SIXTEENTH",
            "SEVENTEENTH",
            "EIGHTEENTH",
            "NINETEENTH",
            "TWENTIETH",
        ]
        if value.day <= 20:
            ordinal = words[value.day - 1]
        elif value.day < 30:
            ordinal = "TWENTY-" + words[value.day - 21]
        else:
            ordinal = "THIRTIETH" if value.day == 30 else "THIRTY-FIRST"
        date_format = date_format.replace("{day_ordinal}", ordinal)
    return value.strftime(date_format)


def lexical_expression(blueprint: SamplingBlueprint, binding: Any, scenario: Any) -> str:
    """Rebind separately printed country literals, including parenthesized ISO codes."""
    if not binding.path.endswith(".addressLine"):
        return binding.expression
    role = binding.path.rsplit(".", 1)[0]
    geo = scenario.party_localities[role]
    old_country = flat(blueprint.target).get(role + ".country")
    expression = blueprint.ownership_data.get("render_expressions", {}).get(
        binding.path, binding.expression
    )
    country_bindings = [
        b
        for b in blueprint.historical_bindings.values()
        if role + ".country" in [current_path(p) for p in b.get("target_paths", [])]
    ]
    substitutions = {}
    for historical in country_bindings:
        for occurrence in historical["occurrences"]:
            old = occurrence["source_text"]
            substitutions[old] = (
                geo.country_code if re.fullmatch(r"[A-Z]{2}", old.strip()) else geo.country
            )
    if old_country:
        substitutions.setdefault(old_country, geo.country)
    return substitute_expression_literals(expression, substitutions)
