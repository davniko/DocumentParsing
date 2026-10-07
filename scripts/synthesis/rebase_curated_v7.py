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


def draft(row, template, numeric, hints):
    raw, leaves = row["joinedRawText"], flat(row["target"])
    variables, targets, missing = [], {}, []
    occupied = []
    identities = {}
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
        paths = [translated(p) for p in binding["target_paths"]]
        paths = [
            p
            for p in paths
            if p in leaves
            and isinstance(leaves[p], (int, float))
            and not isinstance(leaves[p], bool)
        ]
        if binding["derivation"] in {"sum_package_quantity", "number_to_words"}:
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
            if len({str(leaves[p]) for p in paths}) != 1:
                missing.append(
                    dict(
                        binding=binding["logical_key"],
                        reason="shared old paths disagree in current labels",
                    )
                )
                continue
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
        for path in paths:
            if path in targets:
                # Multiple printed realizations of one fact belong to one variable.
                prior_key = targets[path][1:-1]
                prior = next(v for v in variables if v["key"] == prior_key)
                if prior["kind"] == kind and Decimal(prior["value"]) == Decimal(value):
                    prior["occurrences"].extend(specs)
                    variables.pop()
                    key = prior_key
                else:
                    missing.append(dict(path=path, reason="conflicting numeric binding"))
            targets[path] = "{" + key + "}"
        # Recover fields absent from old labels only from a dimension-specific
        # source-owned total; per-container rows do not imply shipment totals.
        if not paths and kind in {"mass", "volume"}:
            candidates = [
                p
                for p, v in leaves.items()
                if isinstance(v, (int, float))
                and Decimal(str(v)) == Decimal(value)
                and ("Weight.value" in p if kind == "mass" else ".volume.value" in p)
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
    for path, value in leaves.items():
        if path in targets or not isinstance(value, (int, float)):
            continue
        if path.endswith(".grossWeight.value"):
            components = [
                v
                for v in variables
                if v["kind"] == "mass" and re.search(r"gross[_ ]?weight", v["meaning"], re.I)
            ]
        elif path.endswith(".volume.value"):
            components = [v for v in variables if v["kind"] == "volume"]
        else:
            continue
        if components and sum(Decimal(v["value"]) for v in components) == Decimal(str(value)):
            targets[path] = " + ".join("{" + v["key"] + "}" for v in components)
        else:
            missing.append(
                dict(
                    path=path,
                    value=value,
                    reason="no current scalar or exact component sum binding",
                )
            )
    for key, additional in hints.get("extra_occurrences", {}).items():
        variable = next(v for v in variables if v["key"] == key)
        for quote in additional:
            matches = [m.span() for m in re.finditer(re.escape(quote), raw)]
            for match in matches:
                if any(match[0] < d and c < match[1] for c, d in occupied):
                    continue
                spec = occurrences(raw, [match])[0]
                spec["presentation"] = "number"
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
