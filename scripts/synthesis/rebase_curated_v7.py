"""Rebind current V7 sources using exact text and historical span hints.

Current OCR and labels must replay exactly. Ambiguous regions stay in inventory.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import yaml

from document_ocr.synthesis.curated import (
    SourceContract,
    _number_style,
    compile_contract,
    digest,
    flat,
    save,
)
from document_ocr.synthesis.template_compiler.descendant import _number_word_phrase

ROOT = Path(__file__).resolve().parents[2]


def occurrences(raw, matches):
    result = []
    for start, end in matches:
        text = raw[start:end]
        number = next(
            i + 1 for i, m in enumerate(re.finditer(re.escape(text), raw)) if m.start() == start
        )
        result.append(dict(text=text, occurrence=number, presentation="text"))
    return result


def translated(path):
    path = path.replace(".cargoGroups[", ".goodsItemDetails[")
    path = re.sub(
        r"\.cargoPackages\[(\d+)\]\.quantity$",
        r".goodsItemDetails[0].numberAndTypeOfPackages[\1].packageQuantity",
        path,
    )
    path = re.sub(
        r"\.cargoAllocationGroups\[0\]\.allocations\[(\d+)\]\.packageQuantity$",
        r".goodsItemDetails[0].splitGoodsPlacement[\1].packageQuantity",
        path,
    )
    return path


def placement_paths(row, template):
    """Reconcile historical allocation indices by their printed equipment ID."""
    current = {
        placement["equipmentIdentifier"]: index
        for index, placement in enumerate(
            row["target"]["documentPatch"]
            .get("goodsItemDetails", [{}])[0]
            .get("splitGoodsPlacement", [])
        )
        if "equipmentIdentifier" in placement
    }
    result = {}
    for binding in template["bindings"]:
        for value in binding.get("realization", {}).get("target_values", []):
            match = re.fullmatch(
                r"documentPatch\.cargoAllocationGroups\[0\]\.allocations\[(\d+)\]\.containerNumber",
                value["target_path"],
            )
            if not match:
                continue
            identity = re.sub(r"[\s/-]", "", str(value["source_value"]))
            if identity not in current:
                raise ValueError(
                    f"historical allocation equipment is absent from current labels: {identity}"
                )
            if any(
                re.sub(r"[\s/-]", "", occurrence["source_text"]) != identity
                for occurrence in binding["occurrences"]
            ):
                raise ValueError(
                    "allocation identity is not proven by its historical source surfaces"
                )
            index = int(match[1])
            if index in result and result[index] != current[identity]:
                raise ValueError("historical allocation index has conflicting equipment identities")
            result[index] = current[identity]
    return result


def numeric_paths(binding, leaves, placements):
    paths = []
    for original in binding["target_paths"]:
        path = translated(original)
        match = re.fullmatch(
            r"documentPatch\.cargoAllocationGroups\[0\]\.allocations\[(\d+)\]\.packageQuantity",
            original,
        )
        if match and int(match[1]) in placements:
            path = (
                "documentPatch.goodsItemDetails[0].splitGoodsPlacement["
                + str(placements[int(match[1])])
                + "].packageQuantity"
            )
        if (
            path in leaves
            and isinstance(leaves[path], (int, float))
            and not isinstance(leaves[path], bool)
        ):
            paths.append(path)
    return list(dict.fromkeys(paths))


def numeric_span(token):
    """Locate a single numeric surface without interpreting its separators."""
    matches = list(re.finditer(r"(?<![A-Za-z0-9])[0-9][0-9, .]*[0-9]|(?<![A-Za-z0-9])[0-9]", token))
    if len(matches) == 1:
        start, end = matches[0].span()
        return start, start + len(token[start:end].rstrip()), "number"
    if not matches:
        _, start, end = _number_word_phrase(token)
        return start, end, "words"
    raise ValueError("multiple/absent numeric tokens")


def prints_value(binding, value):
    for occurrence in binding["occurrences"]:
        try:
            start, end, presentation = numeric_span(occurrence["source_text"])
            token = occurrence["source_text"][start:end]
            if presentation == "number":
                _number_style(token, str(value), str(value))
            elif Decimal(_number_word_phrase(token)[0]) != Decimal(str(value)):
                return False
        except ValueError:
            return False
    return bool(binding["occurrences"])


def package_dependencies(binding, by_key, seen=()):
    """A number-to-words derivation is not necessarily a cargo-package count."""
    key = binding["logical_key"]
    if key in seen:
        raise ValueError("cyclic historical numeric dependencies")
    paths = set(binding.get("dependency_paths", [])) | set(binding["target_paths"])
    for dependency in binding.get("dependency_bindings", []):
        if dependency not in by_key:
            raise ValueError(f"missing historical numeric dependency: {dependency}")
        paths.update(package_dependencies(by_key[dependency], by_key, (*seen, key)))
    return {
        p if p.endswith(".quantity") else p + ".quantity"
        for p in paths
        if re.fullmatch(r"documentPatch\.cargoPackages\[\d+\](?:\.quantity)?", p)
    }


def measure_role(binding, side, allocation_rows=None):
    """Use declared numeric metadata, never matching numbers from arbitrary OCR."""
    metadata = (binding["logical_key"] + " " + binding.get("group_key", "")).casefold()
    if any(word in metadata for word in ("tare", "temperature", "ventilat")):
        return None
    row = re.search(
        r"container[:_ ]?(\d+)|row_(?:gross_weight|net_weight|volume):(\d+)"
        r"|container_measurement:(?:gross|net|volume)_(\d+)",
        metadata,
    )
    row_index = int(next(v for v in row.groups() if v is not None)) if row else None
    allocation = re.search(r"allocation:(\d+):(\d+)", metadata)
    if allocation:
        coordinate = (int(allocation[1]), int(allocation[2]))
        if allocation_rows is None or coordinate not in allocation_rows:
            raise ValueError("allocation measurement lacks a source-proven current equipment row")
        row_index = allocation_rows[coordinate]
    explicit = {
        name
        for path in binding["target_paths"] + side.get("target_paths", [])
        for name in ("grossWeight", "netWeight", "volume")
        if path.endswith("." + name + ".value")
    }
    if len(explicit) == 1:
        return explicit.pop(), row_index
    if "net_weight" in metadata or "netweight" in metadata:
        return "netWeight", row_index
    if "gross_weight" in metadata or "grossweight" in metadata or "gross" in metadata:
        return "grossWeight", row_index
    if (
        side.get("role") == "cargo_mass"
        and "weight" in metadata
        and (row or "total" in metadata or binding.get("group_key", "").startswith("cargo:"))
    ):
        return "grossWeight", row_index
    if side.get("role") == "cargo_volume" and "volume" in metadata:
        return "volume", row_index
    return None


def prints_unit(raw, variable, unit, *, allow_absent=False):
    aliases = {
        "kilogram": ("KILOGRAMS", "KILOGRAM", "KGS", "KGM", "KG"),
        "metric_tonne": ("TONNES", "TONNE", "MTS", "MT"),
        "pound": ("POUNDS", "POUND", "LBS", "LB"),
        "cubic_metre": ("CBM", "M3", "M³", "CU. M."),
    }
    if unit not in aliases:
        return False
    pattern = "(?:" + "|".join(map(re.escape, aliases[unit])) + ")"
    any_pattern = (
        "(?:" + "|".join(re.escape(alias) for group in aliases.values() for alias in group) + ")"
    )
    for occurrence in variable["occurrences"]:
        matches = list(re.finditer(re.escape(occurrence["text"]), raw))
        match = matches[occurrence["occurrence"] - 1]
        before = raw[raw.rfind("\n", 0, match.start()) + 1 : match.start()]
        end = raw.find("\n", match.end())
        after = raw[match.end() : end if end >= 0 else len(raw)]
        if not (
            re.match(r"\s*" + pattern + r"(?!\w)", after, re.I)
            or re.search(r"(?<!\w)" + pattern + r"\s*$", before, re.I)
        ) and (
            not allow_absent
            or re.match(r"\s*" + any_pattern + r"(?!\w)", after, re.I)
            or re.search(r"(?<!\w)" + any_pattern + r"\s*$", before, re.I)
        ):
            return False
    return True


def draft(row, template, numeric, hints):
    raw, leaves = row["joinedRawText"], flat(row["target"])
    variables, targets, missing = [], {}, []
    occupied = []
    identities = {}
    placements = placement_paths(row, template)
    patch = row["target"]["documentPatch"]
    equipment_indices = {
        c["equipmentIdentifier"]: i for i, c in enumerate(patch.get("containerInformation", []))
    }
    current_placements = patch.get("goodsItemDetails", [{}])[0].get("splitGoodsPlacement", [])
    allocation_rows = {
        (0, old): equipment_indices[current_placements[new]["equipmentIdentifier"]]
        for old, new in placements.items()
    }
    by_key = {binding["logical_key"]: binding for binding in template["bindings"]}
    numeric_sources = {}
    hint_fields = hints.get("fields", {})
    for path, value in leaves.items():
        if path in hint_fields:
            continue
        if not isinstance(value, str):
            continue
        if path.endswith(".addressLine"):
            kind = "postal"
        elif ".parties." in path and path.endswith(".name"):
            kind = "name"
        elif path.endswith(".description"):
            kind = "product"
        elif path.endswith(".equipmentIdentifier"):
            kind = "container"
        elif path.endswith(".billOfLadingNumber") or ".sealNumbers[" in path:
            kind = "identifier"
        else:
            continue
        identity = (kind, value)
        if identity in identities:
            targets[path] = "{" + identities[identity] + "}"
            continue
        pattern = r"\s+".join(map(re.escape, value.split()))
        if kind == "container":
            pattern = r"[\s/\-]*".join(map(re.escape, value))
        matches = [(m.start(), m.end()) for m in re.finditer(pattern, raw, re.I)]
        if not matches:
            missing.append(
                dict(path=path, value=value, reason="no exact whitespace/case-normalized surface")
            )
            continue
        if any(a < d and c < b for a, b in matches for c, d in occupied):
            missing.append(dict(path=path, value=value, reason="overlapping owned surface"))
            continue
        key = f"v{len(variables):03d}"
        country = leaves.get(path.rsplit(".", 1)[0] + ".country") if kind == "postal" else None
        literals = [country] if country and country.upper() in value.upper() else []
        variables.append(
            dict(
                key=key,
                kind=kind,
                value=value,
                meaning=f"{path}; preserve source country {
                    country or ('as printed')
                } and commodity classification; complete owned value",
                required_literals=literals,
                occurrences=occurrences(raw, matches),
            )
        )
        occupied.extend(matches)
        identities[identity] = key
        targets[path] = "{" + key + "}"
    for binding in template["bindings"]:
        if binding["logical_key"] in hints.get("fixed_historical_bindings", {}):
            continue
        paths = numeric_paths(binding, leaves, placements)
        if binding["derivation"] == "container_package_count":
            packages = patch.get("goodsItemDetails", [{}])[0].get("numberAndTypeOfPackages", [])
            # A carrier's receipt can select either side of its printed
            # "containers or packages" caption. Certify the package reading
            # only when the source count rules out the equipment count.
            if (
                len(packages) == 1
                and prints_value(binding, packages[0]["packageQuantity"])
                and not prints_value(binding, len(patch.get("containerInformation", [])))
            ):
                paths = [
                    "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"
                ]
        if binding["derivation"] in {"sum_package_quantity", "number_to_words"}:
            if binding["derivation"] == "number_to_words" and not package_dependencies(
                binding, by_key
            ):
                # Original B/L counts, waybill sequence numbers and equipment
                # counts remain historical dependencies, not package targets.
                continue
            package_paths = [
                p
                for p in leaves
                if re.fullmatch(
                    r"documentPatch.goodsItemDetails\[0\].numberAndTypeOfPackages\[\d+\].packageQuantity",
                    p,
                )
            ]
            if len(package_paths) != 1:
                raise ValueError("package-total rebase needs an explicit multi-package equation")
            paths = package_paths
        side = numeric.get(binding["logical_key"])
        role = side.get("role") if side else None
        if paths:
            if all(p.endswith(".packageQuantity") for p in paths):
                kind = "count"
            elif all(".Weight." in p for p in paths) or all("Weight.value" in p for p in paths):
                kind = "mass"
            elif all(".volume.value" in p for p in paths):
                kind = "volume"
            else:
                continue
            matching_paths = [p for p in paths if prints_value(binding, leaves[p])]
            if not matching_paths:
                missing.append(
                    dict(
                        binding=binding["logical_key"],
                        reason="historical numeric surface does not print current target value",
                        paths=paths,
                        texts=[o["source_text"] for o in binding["occurrences"]],
                    )
                )
                continue
            # Historical package rows may have been collapsed into one current
            # shipment total. Retain only proven equal leaves, then derive the
            # new total from all owned row counts below.
            paths = matching_paths
            value = str(leaves[paths[0]])
        elif side and role in {"cargo_mass", "cargo_volume", "cargo_quantity"}:
            if side["mode"] in {"source_fixed", "surface_fixed", "review_required"}:
                continue
            kind = {"cargo_mass": "mass", "cargo_volume": "volume", "cargo_quantity": "count"}[role]
            value = side["source_value"]
        else:
            continue
        matches = []
        specs = []
        for occ in binding["occurrences"]:
            start = len(raw.encode()[: occ["byte_start"]].decode())
            token = occ["source_text"]
            if (
                raw.encode()[occ["byte_start"] : occ["byte_start"] + len(token.encode())].decode()
                != token
            ):
                raise ValueError(
                    f"historical numeric source bytes differ: {binding['logical_key']}"
                )
            nums = list(
                re.finditer(r"(?<![A-Za-z0-9])[0-9][0-9, .]*[0-9]|(?<![A-Za-z0-9])[0-9]", token)
            )
            if len(nums) == 1:
                a, b = nums[0].span()
                fragment = token[a:b].rstrip()
                b = a + len(fragment)
                presentation = "number"
            elif not nums and kind == "count":
                try:
                    n, a, b = _number_word_phrase(token)
                except ValueError:
                    missing.append(
                        dict(
                            binding=binding["logical_key"],
                            reason="unparsed spelled count",
                            text=token,
                        )
                    )
                    continue
                if Decimal(n) != Decimal(value):
                    missing.append(
                        dict(
                            binding=binding["logical_key"],
                            reason="spelled count differs",
                            text=token,
                        )
                    )
                    continue
                presentation = "words"
            else:
                missing.append(
                    dict(
                        binding=binding["logical_key"],
                        reason="multiple/absent numeric tokens",
                        text=token,
                    )
                )
                continue
            match = (start + a, start + b)
            if any(match[0] < d and c < match[1] for c, d in occupied):
                missing.append(
                    dict(
                        binding=binding["logical_key"],
                        reason="numeric overlap",
                        text=raw[match[0] : match[1]],
                    )
                )
                continue
            matches.append(match)
            spec = occurrences(raw, [match])[0]
            spec["presentation"] = presentation
            specs.append(spec)
        if not specs:
            continue
        key = f"v{len(variables):03d}"
        current_meaning = (
            "; ".join(paths)
            if paths
            else "Printed cargo "
            + {"mass": "mass", "volume": "volume", "count": "package count"}[kind]
            + ": "
            + binding["logical_key"].split("agent:")[-1]
        )
        owned_measure = (
            measure_role(binding, side or {}, allocation_rows)
            if kind in {"mass", "volume"}
            else None
        )
        if not paths and owned_measure:
            measure, index = owned_measure
            name = {"grossWeight": "gross_weight", "netWeight": "net_weight", "volume": "volume"}[
                measure
            ]
            current_meaning = "Printed cargo " + name
            if index is not None:
                current_meaning += f" container:{index}"
        variables.append(
            dict(
                key=key,
                kind=kind,
                value=value,
                meaning=current_meaning,
                required_literals=[],
                occurrences=specs,
            )
        )
        occupied.extend(matches)
        # Resolve sharing before assigning any new path. Otherwise a new path
        # preceding an existing one can retain the key we remove during merging.
        prior_expressions = {targets[path] for path in paths if path in targets}
        if len(prior_expressions) > 1:
            missing.append(
                dict(binding=binding["logical_key"], reason="conflicting numeric owners")
            )
        elif prior_expressions:
            prior_key = next(iter(prior_expressions))[1:-1]
            prior = next(v for v in variables if v["key"] == prior_key)
            if prior["kind"] == kind and Decimal(prior["value"]) == Decimal(value):
                prior["occurrences"].extend(specs)
                variables.pop()
                key = prior_key
            else:
                missing.append(
                    dict(binding=binding["logical_key"], reason="conflicting numeric binding")
                )
        for path in paths:
            targets[path] = "{" + key + "}"
        numeric_sources.setdefault(key, []).append(binding)
        # Recover fields absent from old labels only from a dimension-specific
        # source-owned total; per-container rows do not imply shipment totals.
        if not paths and kind in {"mass", "volume"}:
            candidates = [
                p
                for p, v in leaves.items()
                if isinstance(v, (int, float))
                and Decimal(str(v)) == Decimal(value)
                and ("Weight.value" in p if kind == "mass" else ".volume.value" in p)
                and p in {translated(p) for p in side.get("target_paths", [])}
                and side.get("mode") in {"target_sum", "target_converted"}
            ]
            for p in candidates:
                if p not in targets:
                    targets[p] = "{" + key + "}"
    for path, field in hint_fields.items():
        targets[path] = field["expression"]
        for key, (kind, quote, meaning) in field["parts"].items():
            matches = [m.span() for m in re.finditer(re.escape(quote), raw)]
            if not matches:
                raise ValueError(f"hint quote absent: {quote}")
            if any(a < d and c < b for a, b in matches for c, d in occupied):
                raise ValueError(f"hint overlaps prior region: {key}")
            specs = occurrences(raw, matches)
            value = " ".join(quote.split()).upper()
            literals = []
            if kind == "count":
                n, _, _ = _number_word_phrase(quote)
                value = str(n)
                for spec in specs:
                    spec["presentation"] = "words"
            if kind == "postal":
                country = leaves.get(path.rsplit(".", 1)[0] + ".country")
                if country and country.upper() in value:
                    literals.append(country)
            variables.append(
                dict(
                    key=key,
                    kind=kind,
                    value=value,
                    meaning=meaning,
                    required_literals=literals,
                    occurrences=specs,
                )
            )
            occupied.extend(matches)
    # Recover totals only from owned dimensional facts: one printed total, or
    # exactly one row per current container, in the same explicitly printed unit.
    # Equal values alone do not establish mass/volume or row/total ownership.
    container_count = len(row["target"]["documentPatch"].get("containerInformation", []))
    for path, value in leaves.items():
        if (
            path in targets
            or path in hints.get("dependent_targets", {})
            or not isinstance(value, (int, float))
        ):
            continue
        measure = next(
            (
                m
                for m in ("grossWeight", "netWeight", "volume")
                if path.endswith("." + m + ".value")
            ),
            None,
        )
        if measure:
            unit = leaves[path.rsplit(".", 1)[0] + ".unit"]
            candidates = []
            for variable in variables:
                bindings = numeric_sources.get(variable["key"], [])
                roles = {
                    measure_role(b, numeric.get(b["logical_key"], {}), allocation_rows)
                    for b in bindings
                }
                if len(roles) != 1:
                    continue
                role = next(iter(roles))
                if (
                    role is not None
                    and role[0] == measure
                    and prints_unit(raw, variable, unit, allow_absent=role[1] is None)
                ):
                    candidates.append((variable, role[1]))
            totals = [
                v
                for v, index in candidates
                if index is None and Decimal(v["value"]) == Decimal(str(value))
            ]
            rows = [(v, index) for v, index in candidates if index is not None]
            if len(totals) == 1:
                components = totals
            elif (
                not totals
                and container_count
                and len(rows) == container_count
                and {index for _, index in rows} == set(range(container_count))
            ):
                components = [v for v, _ in sorted(rows, key=lambda pair: pair[1])]
            else:
                components = []
            if components and sum(Decimal(v["value"]) for v in components) == Decimal(str(value)):
                targets[path] = " + ".join("{" + v["key"] + "}" for v in components)
                for variable, index in candidates:
                    if variable in components and index is not None:
                        name = {
                            "grossWeight": "gross_weight",
                            "netWeight": "net_weight",
                            "volume": "volume",
                        }[measure]
                        variable["meaning"] = (
                            f"Printed cargo {name} container:{index}; exact component of {path}"
                        )
            else:
                missing.append(
                    dict(
                        path=path,
                        value=value,
                        reason="no current scalar or complete same-unit row sum binding",
                    )
                )
        elif re.fullmatch(
            r"documentPatch\.goodsItemDetails\[0\]\.numberAndTypeOfPackages\[0\]\.packageQuantity",
            path,
        ):
            placements_current = row["target"]["documentPatch"]["goodsItemDetails"][0].get(
                "splitGoodsPlacement", []
            )
            components = [
                f"documentPatch.goodsItemDetails[0].splitGoodsPlacement[{index}].packageQuantity"
                for index in range(len(placements_current))
            ]
            if (
                components
                and all(p in targets for p in components)
                and sum(Decimal(str(leaves[p])) for p in components) == Decimal(str(value))
            ):
                targets[path] = " + ".join(targets[p] for p in components)
            else:
                missing.append(
                    dict(
                        path=path,
                        value=value,
                        reason="no current scalar or exact complete allocation sum binding",
                    )
                )
    # A shipment with one printed container and one complete placement has a
    # single package-count fact. Current labels can add this placement where
    # historical labels owned only the shipment total. Link it to the proven
    # count expression, never to an incidental equal-valued OCR number.
    groups = row["target"]["documentPatch"].get("goodsItemDetails", [])
    containers = row["target"]["documentPatch"].get("containerInformation", [])
    if len(groups) == 1 and len(containers) == 1:
        packages = groups[0].get("numberAndTypeOfPackages", [])
        allocation = groups[0].get("splitGoodsPlacement", [])
        total_path = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"
        placement_path = "documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].packageQuantity"
        identity_path = "documentPatch.containerInformation[0].equipmentIdentifier"
        if (
            len(packages) == len(allocation) == 1
            and placement_path in leaves
            and placement_path not in targets
            and total_path in targets
            and identity_path in targets
            and allocation[0].get("equipmentIdentifier") == containers[0]["equipmentIdentifier"]
            and allocation[0]["packageQuantity"] == packages[0]["packageQuantity"]
        ):
            targets[placement_path] = targets[total_path]
    for key, additional in hints.get("extra_occurrences", {}).items():
        variable = next(v for v in variables if v["key"] == key)
        for quote in additional:
            matches = [m.span() for m in re.finditer(re.escape(quote), raw)]
            if not matches:
                raise ValueError(f"additional numeric occurrence is absent: {key}")
            start, end, presentation = numeric_span(quote)
            if start != 0 or end != len(quote):
                raise ValueError(f"additional numeric occurrence must own only its scalar: {key}")
            if presentation == "words":
                if Decimal(_number_word_phrase(quote)[0]) != Decimal(variable["value"]):
                    raise ValueError(f"additional number words disagree with source value: {key}")
            else:
                _number_style(quote, variable["value"], variable["value"])
            for match in matches:
                if any(match[0] < d and c < match[1] for c, d in occupied):
                    continue
                spec = occurrences(raw, [match])[0]
                spec["presentation"] = presentation
                variable["occurrences"].append(spec)
                occupied.append(match)
    for variable in hints.get("extra_variables", []):
        variables.append(variable)
    for key, literals in hints.get("required_literals", {}).items():
        variable = next(v for v in variables if v["key"] == key)
        variable["required_literals"] = sorted(set(variable["required_literals"] + literals))
    for path, expression in hints.get("dependent_targets", {}).items():
        if path not in leaves:
            raise ValueError("dependent target must already exist in current labels")
        targets[path] = expression
    for key, reason in hints.get("frozen_variables", {}).items():
        variable = next(v for v in variables if v["key"] == key)
        for path in targets:
            targets[path] = targets[path].replace("{" + key + "}", variable["value"])
        variables.remove(variable)
        if not reason.strip():
            raise ValueError("a source-fixed variable needs a reason")
    result = dict(
        variables=variables,
        targets=[dict(path=p, expression=e) for p, e in targets.items()],
        fixed_context=(
            "Countries, routes, dates, carrier, contacts, tax/customs data, HS/UN identity, "
            "equipment categories/counts, thermal settings and packaging categories remain "
            "source-fixed. Only contracted postal/company/product surfaces, identifiers, "
            "counts and measurements vary. Product numeric specifications remain fixed. "
            + " ".join(hints.get("frozen_variables", {}).values())
        ),
    )
    return result, missing


def main():
    config = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_pilot24.yaml").read_text()
    )
    rows = {
        r["documentId"]: r
        for r in map(
            json.loads, (ROOT / config["dataset"] / "train.jsonl").read_text().splitlines()
        )
    }
    catalog = ROOT / (
        "artifacts/kie-synthesis-production/template-base/catalogs/mpci-bl"
        "-production-template-catalog1212-v38b-docb7-frozen-provisional/ca"
        "ses"
    )
    sidecars = {
        r["sourceDocumentId"]: r
        for r in map(
            json.loads,
            (
                ROOT
                / (
                    "artifacts/kie-synthesis-production/numeric-contract-catalogs/mpci"
                    "-bl-numeric-contracts-discharge-country-v6/contracts.jsonl"
                )
            )
            .read_text()
            .splitlines(),
        )
    }
    hints = yaml.safe_load(
        (ROOT / "configs/synthesis/contracts/curated_v7_pilot24_rebinding.yaml").read_text()
    )
    summaries = []
    for sid in config["source_ids"]:
        row = rows[sid]
        if (catalog / sid / "source.txt").read_text() != row["joinedRawText"]:
            raise ValueError("source OCR differs from historical span authority")
        contract, missing = draft(
            row,
            json.loads((catalog / sid / "template.json").read_text()),
            sidecars[sid]["contracts"],
            hints.get(sid, {}),
        )
        error = None
        try:
            compile_contract(row, SourceContract.model_validate(contract))
        except Exception as exc:
            error = str(exc)
        envelope = dict(
            sourceSha256=digest(row["joinedRawText"].encode()),
            targetSha256=digest(row["target"]),
            contract=contract,
            missing=missing,
            replayError=error,
        )
        directory = ROOT / config["output"] / "sources" / sid
        save(directory / "rebase-draft.json", envelope)
        if not missing and error is None:
            save(
                directory / "contract.json",
                {k: envelope[k] for k in ("sourceSha256", "targetSha256", "contract")},
            )
        summaries.append(
            dict(documentId=sid, missing=missing, error=error, variables=len(contract["variables"]))
        )
        print(sid[4:12], len(contract["variables"]), "missing", len(missing), "error", error)
    save(ROOT / config["output"] / "rebase-summary.json", summaries)


if __name__ == "__main__":
    main()
