"""Prepare the 100-source expansion from current train-only source evidence.

Selection is explicit and persisted before generation. Existing publications
remain untouched. Drafts and compilation failures are inventory, not admission.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml
from rebase_curated_v7 import draft

from document_ocr.synthesis.curated import SourceContract, compile_contract, digest, save
from document_ocr.synthesis.curated_ownership import build_owned_blueprint

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/kie-synthesis-production/curated-v7-expansion100-v1"
CONFIG = ROOT / "configs/synthesis/mpci_bl_curated_v7_expansion100.yaml"
CONTRACTS = ROOT / "configs/synthesis/contracts/curated_v7_expansion100"


def read(path):
    return json.loads(path.read_text())


def merge_reviewed(left, right, trail=""):
    """Merge independent declarations; conflicting decisions must be resolved explicitly."""
    result = deepcopy(left)
    for key, value in right.items():
        if key not in result:
            result[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = merge_reviewed(result[key], value, trail + "." + key)
        elif isinstance(value, list) and isinstance(result[key], list):
            result[key].extend(item for item in value if item not in result[key])
        elif result[key] != value:
            raise ValueError(f"conflicting reviewed declarations: {trail}.{key}")
    return result


def apply_owner_closure(declarations, entry):
    """Apply an explicitly reviewed replacement, rather than unioning retired owner paths."""
    result = deepcopy(declarations)
    for field, value in entry["ownership"].items():
        if field == "add":
            replacements = {item["key"]: item for item in value}
            result[field] = [
                item for item in result.get(field, []) if item["key"] not in replacements
            ] + list(replacements.values())
        elif field == "bindings":
            for key, patch in value.items():
                result.setdefault(field, {}).setdefault(key, {}).update(deepcopy(patch))
        elif isinstance(value, dict):
            result.setdefault(field, {}).update(deepcopy(value))
        else:
            raise ValueError(f"unsupported reviewed owner replacement: {field}")
    return result


def admit_reviewed():
    """Rebase new sources with reviewed declarations and exact baseline replay."""
    config = yaml.safe_load(CONFIG.read_text())
    existing = set(read(OUT / "selection.json")["existing"])
    rows = {
        row["documentId"]: row
        for row in map(
            json.loads, (ROOT / config["dataset"] / "train.jsonl").read_text().splitlines()
        )
    }
    numeric = {
        row["sourceDocumentId"]: row["contracts"]
        for row in map(
            json.loads,
            (
                ROOT
                / "artifacts/kie-synthesis-production/numeric-contract-catalogs"
                / "mpci-bl-numeric-contracts-discharge-country-v6/contracts.jsonl"
            )
            .read_text()
            .splitlines(),
        )
    }
    numeric_review = read(OUT / "audit/numeric-admission.json")["proposals"]
    numeric_render = read(OUT / "audit/numeric-render-admission.json")["sources"]
    replacement_hints = {
        item["included"]: item.get("rebase_hints", {})
        for item in read(OUT / "audit/selection-revision.json")["replacements"]
    }
    lexical_review = {
        item["documentId"]: item
        for item in read(OUT / "audit/lexical-admission/declarations.json")["sources"]
    }
    supplements = [
        read(OUT / "audit/numeric-wrapper-ownership.json")["sources"],
        read(OUT / "audit/lexical-admission/supplemental-ownership.json")["sources"],
        read(OUT / "audit/lexical-admission/closure-ownership.json")["sources"],
        read(OUT / "audit/equipment-ownership.json")["sources"],
        read(OUT / "audit/transport-render-ownership.json")["sources"],
        read(OUT / "audit/country-render-ownership.json")["sources"],
        {sid: item.get("ownership", {}) for sid, item in numeric_render.items()},
        {
            sid: item["ownership"]
            for sid, item in read(OUT / "audit/ce1f-tare-ownership.json")["sources"].items()
        },
        {
            sid: item["ownership"]
            for sid, item in read(OUT / "audit/tare-source-ownership.json")["sources"].items()
        },
        {
            item["documentId"]: item["ownership"]
            for item in read(OUT / "audit/proposed-auxiliary-ownership.json")["sources"]
        },
        {
            item["documentId"]: item["ownership"]
            for item in read(OUT / "audit/proposed-route-handling-ownership.json")["sources"]
        },
    ]
    ownership_path = ROOT / config["ownership"]
    ownership = yaml.safe_load(ownership_path.read_text())
    closure_entries = [
        item
        for name in (
            "proposed-render-owner-closure.json",
            "proposed-render-owner-closure-supplement.json",
            "proposed-origin-explicit-ownership.json",
        )
        for item in read(OUT / "audit" / name)["sources"]
    ]
    closures = {item["documentId"]: item for item in closure_entries}
    if len(closures) != len(closure_entries):
        raise ValueError("duplicate source in reviewed ownership closure")
    for sid, item in closures.items():
        if digest(rows[sid]["joinedRawText"].encode()) != item["source_sha256"]:
            raise ValueError(f"ownership closure source identity mismatch: {sid}")
    closure_receipts = []
    report = []
    for sid in config["source_ids"]:
        if sid in existing:
            continue
        row = rows[sid]
        historical = read(ROOT / config["historical_catalog"] / sid / "template.json")
        nr, lr = numeric_review.get(sid, {}), lexical_review.get(sid, {})
        hashes = dict(
            sourceSha256=digest(row["joinedRawText"].encode()), targetSha256=digest(row["target"])
        )
        if lr and any(lr[key] != value for key, value in hashes.items()):
            raise ValueError(f"lexical review source identity mismatch: {sid}")
        # Sidecar metadata supplements the historical typed bindings. Its absence
        # is inventoried; exact baseline/numeric ownership checks still apply.
        hints = merge_reviewed(
            nr.get("rebaseHints", {}), numeric_render.get(sid, {}).get("hints", {})
        )
        hints = merge_reviewed(hints, replacement_hints.get(sid, {}))
        contract, missing = draft(row, historical, numeric.get(sid, {}), hints)
        additions = lr.get("contract_additions", {})
        contract["variables"].extend(additions.get("variables", []))
        contract["targets"].extend(additions.get("targets", []))
        declarations = merge_reviewed(ownership["sources"][sid], nr.get("ownership", {}))
        declarations = merge_reviewed(declarations, lr.get("ownership", {}))
        for supplement in supplements:
            declarations = merge_reviewed(declarations, supplement.get(sid, {}))
        if sid in closures:
            before = deepcopy(declarations)
            declarations = apply_owner_closure(declarations, closures[sid])
            closure_receipts.append(
                {
                    "documentId": sid,
                    "before": before,
                    "after": declarations,
                    "rationale": closures[sid]["rationale"],
                }
            )
        covered = {item["path"] for item in contract["targets"]}
        covered.update(p for item in declarations.get("lexical", []) for p in item["paths"])
        unresolved = [item for item in missing if item.get("path") not in covered]
        try:
            if lr.get("held") or unresolved:
                raise ValueError(f"unresolved admission fields: {lr.get('held', []) + unresolved}")
            blueprint = build_owned_blueprint(
                row, historical, SourceContract.model_validate(contract), declarations
            )
        except (ValueError, KeyError) as error:
            report.append(
                {"documentId": sid, "status": "held", "error": str(error), "missing": unresolved}
            )
            continue
        save(OUT / "sources" / sid / "contract.json", {**hashes, "contract": contract})
        ownership["sources"][sid] = declarations
        report.append(
            {
                "documentId": sid,
                "status": "baseline_replay_passed",
                **hashes,
                "numericSidecarPresent": sid in numeric,
                "inventory": blueprint.inventory(),
            }
        )
    ownership_path.write_text(yaml.safe_dump(ownership, sort_keys=False, allow_unicode=True))
    auxiliary_path = ROOT / config["auxiliary"]
    auxiliary = yaml.safe_load(auxiliary_path.read_text())
    final_auxiliary = read(OUT / "audit/proposed-final-auxiliary-closure.json")
    for item in final_auxiliary["sources"]:
        sid = item["documentId"]
        if digest(rows[sid]["joinedRawText"].encode()) != item["sourceSha256"]:
            raise ValueError(f"final auxiliary source identity differs: {sid}")
        bindings = auxiliary["sources"][sid].setdefault("bindings", {})
        for key, revision in item["bindings"].items():
            if bindings.get(key) not in (revision["before"], revision["after"]):
                raise ValueError(f"final auxiliary binding precondition differs: {sid}/{key}")
            bindings.pop(key, None)
    auxiliary_proposals = yaml.safe_load((OUT / "audit/proposed-auxiliary.yaml").read_text())
    auxiliary_proposals["sources"] = merge_reviewed(
        auxiliary_proposals["sources"],
        yaml.safe_load((OUT / "audit/proposed-auxiliary-final-supplement.yaml").read_text())[
            "sources"
        ],
    )
    if set(auxiliary_proposals["sources"]) != set(config["source_ids"]) - existing:
        raise ValueError("auxiliary proposals do not cover the active new sources")
    auxiliary["sources"] = merge_reviewed(auxiliary["sources"], auxiliary_proposals["sources"])
    residuals = read(OUT / "audit/proposed-render-auxiliary-residuals.json")
    for item in residuals["ledger"]:
        if digest(rows[item["documentId"]]["joinedRawText"].encode()) != item["sourceSha256"]:
            raise ValueError(f"auxiliary residual source identity mismatch: {item['documentId']}")
    auxiliary["sources"] = merge_reviewed(auxiliary["sources"], residuals["sources"])
    for revision in residuals["revisions"]:
        values = auxiliary["sources"][revision["documentId"]][revision["section"]]
        if revision["before"] not in values and revision["after"] not in values:
            raise ValueError(
                f"auxiliary replacement precondition differs: {revision['documentId']}"
            )
        values[:] = [item for item in values if item != revision["before"]]
        if revision["after"] not in values:
            values.append(revision["after"])
    for sid, item in closures.items():
        for key in item.get("auxiliary_remove_bindings", []):
            auxiliary["sources"][sid].get("bindings", {}).pop(key, None)
    for item in final_auxiliary["sources"]:
        source = auxiliary["sources"][item["documentId"]]
        for key, revision in item["bindings"].items():
            source.setdefault("bindings", {})[key] = deepcopy(revision["after"])
        source.setdefault("spans", []).extend(
            span for span in item["spans"] if span not in source.get("spans", [])
        )
    auxiliary_path.write_text(yaml.safe_dump(auxiliary, sort_keys=False, allow_unicode=True))
    save(OUT / "audit/integrated-owner-closure-receipts.json", closure_receipts)
    save(OUT / "audit/integrated-auxiliary-residual-receipts.json", residuals)
    save(OUT / "audit/integrated-final-auxiliary-receipts.json", final_auxiliary)
    capabilities = read(OUT / "audit/capability-admission.json")["sources"]
    if {item["documentId"] for item in capabilities} != set(config["source_ids"]):
        raise ValueError("reviewed capability inventory does not cover the active source selection")
    for item in capabilities:
        sid = item["documentId"]
        if item["sourceSha256"] != digest(rows[sid]["joinedRawText"].encode()):
            raise ValueError(f"capability source identity differs: {sid}")
        if item["status"] != "scenario_admitted_pending_render_and_semantic_review":
            raise ValueError(f"capability remains unresolved: {sid}")
        config["capabilities"][sid] = item["effectiveCapability"]
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    save(OUT / "audit/integrated-admission.json", report)
    print(
        json.dumps(
            {
                "sources": len(report),
                "passed": sum(r["status"] == "baseline_replay_passed" for r in report),
                "held": [
                    {"documentId": r["documentId"], "error": r["error"]}
                    for r in report
                    if r["status"] == "held"
                ],
            },
            indent=2,
        )
    )


def stage_replacements(path: Path):
    """Record eligibility exclusions and stage substitutes without deleting evidence."""
    config = yaml.safe_load(CONFIG.read_text())
    instructions = yaml.safe_load(path.read_text())
    rows = {
        r["documentId"]: r
        for r in map(
            json.loads, (ROOT / config["dataset"] / "train.jsonl").read_text().splitlines()
        )
    }
    catalog = (
        ROOT
        / "artifacts/kie-synthesis-production/template-base/catalogs"
        / "mpci-bl-production-template-catalog1212-v38b-docb7-frozen-provisional/cases"
    )
    numeric = {
        r["sourceDocumentId"]: r["contracts"]
        for r in map(
            json.loads,
            (
                ROOT
                / "artifacts/kie-synthesis-production/numeric-contract-catalogs"
                / "mpci-bl-numeric-contracts-discharge-country-v6/contracts.jsonl"
            )
            .read_text()
            .splitlines(),
        )
    }
    ownership = yaml.safe_load((ROOT / config["ownership"]).read_text())
    auxiliary = yaml.safe_load((ROOT / config["auxiliary"]).read_text())
    receipt_path = OUT / "audit/selection-revision.json"
    prior_receipts = read(receipt_path)["replacements"] if receipt_path.exists() else []
    receipts = []
    for item in instructions["replacements"]:
        old_matches = [s for s in rows if s[4:].startswith(item["exclude"])]
        new_matches = [s for s in rows if s[4:].startswith(item["include"])]
        if len(old_matches) != 1 or len(new_matches) != 1 or not item["reason"].strip():
            raise ValueError("replacement requires unique source identities and explicit reason")
        old, new = old_matches[0], new_matches[0]
        if old not in config["source_ids"]:
            if new not in config["source_ids"]:
                raise ValueError("replacement does not match configured scope")
            continue
        if new in config["source_ids"]:
            raise ValueError("replacement would duplicate a selected source")
        row = rows[new]
        historical = read(catalog / new / "template.json")
        hints = item.get("rebase_hints", {})
        declarations = item.get("ownership", {})
        contract, missing = draft(row, historical, numeric[new], hints)
        blueprint = build_owned_blueprint(
            row, historical, SourceContract.model_validate(contract), declarations
        )
        covered = {path for region in blueprint.regions for path in region.target_paths}
        unresolved = [entry for entry in missing if entry.get("path") not in covered]
        if unresolved:
            raise ValueError(f"replacement has unresolved source bindings: {new}: {unresolved}")
        envelope = dict(
            sourceSha256=digest(row["joinedRawText"].encode()),
            targetSha256=digest(row["target"]),
            contract=contract,
        )
        save(OUT / "sources" / new / "contract.json", envelope)
        save(OUT / "sources" / new / "draft.json", {**envelope, "missing": [], "error": None})
        save(OUT / "catalog" / new / "template.json", historical)
        config["source_ids"][config["source_ids"].index(old)] = new
        del config["capabilities"][old]
        config["capabilities"][new] = item.get("capability", {"family": "ambient"})
        ownership["sources"][new] = declarations
        auxiliary["sources"][new] = {}
        receipts.append(
            {
                "excluded": old,
                "included": new,
                "reason": item["reason"],
                "sourceSha256": envelope["sourceSha256"],
                "targetSha256": envelope["targetSha256"],
                "rebase_hints": hints,
                "ownership": declarations,
                "rawDraftMissingResolvedByReviewedOwnership": missing,
            }
        )
    if not receipts:
        print("Replacement selection already staged")
        return
    if len(config["source_ids"]) != 100 or len(set(config["source_ids"])) != 100:
        raise ValueError("replacement changed the requested 100-source scope")
    save(
        receipt_path,
        {
            "instructionSha256": digest(instructions),
            "replacements": prior_receipts + receipts,
            "activeSourceIds": config["source_ids"],
        },
    )
    for key, content in (("ownership", ownership), ("auxiliary", auxiliary)):
        (ROOT / config[key]).write_text(
            yaml.safe_dump(content, sort_keys=False, allow_unicode=True)
        )
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    print(json.dumps({"replaced": len(receipts), "activeSources": len(config["source_ids"])}))


def prepare():
    base = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_registry_pilot72.yaml").read_text()
    )
    trans = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_transshipment_pilot6.yaml").read_text()
    )
    rows = {
        r["documentId"]: r
        for r in map(json.loads, (ROOT / base["dataset"] / "train.jsonl").read_text().splitlines())
    }
    with (ROOT / "docs/analysis/synthesis-restart-20261006/inventory.csv").open() as handle:
        inventory = list(csv.DictReader(handle))
    shortlist = read(ROOT / "docs/analysis/synthesis-restart-20261006/seed-candidates.json")
    known = set(base["source_ids"]) | set(trans["source_ids"])
    selected = [r for r in shortlist if r["documentId"] not in known]
    used = known | {r["documentId"] for r in selected}
    counts = Counter(r["carrierFamily"] for r in shortlist)
    extras = []
    for row in inventory:
        sid = row["documentId"]
        if sid in used or sid not in rows or row["catalog"] != "v38b":
            continue
        patch = rows[sid]["target"]["documentPatch"]
        goods = patch.get("goodsItemDetails", [])
        if (
            row["sourceMatchesCurrentOCR"].lower() != "true"
            or row["validationProxyOverlap"].lower() == "true"
            or len(goods) != 1
            or len(goods[0].get("numberAndTypeOfPackages", [])) != 1
            or not goods[0].get("hsCodes")
            or not goods[0].get("grossWeight")
            or row["currentDG"].lower() == "true"
            or row["currentTemperature"].lower() == "true"
            or "transshipmentPort" in patch.get("route", {})
        ):
            continue
        extras.append(row)
    # Favor additional carriers, then short/simple existing source evidence.
    while len(used) < 100:
        if not extras:
            raise ValueError("insufficient eligible expansion candidates")
        chosen = min(
            extras,
            key=lambda r: (
                counts[r["carrierFamily"]],
                int(r["residualBindings"]),
                int(r["pages"]),
                int(r["currentContainers"]),
                r["documentId"],
            ),
        )
        extras.remove(chosen)
        selected.append(chosen)
        used.add(chosen["documentId"])
        counts[chosen["carrierFamily"]] += 1
    selection = {
        "existing": sorted(known),
        "shortlist": [
            r["documentId"]
            for r in selected
            if r["documentId"] in {s["documentId"] for s in shortlist}
        ],
        "additional": [
            r["documentId"]
            for r in selected
            if r["documentId"] not in {s["documentId"] for s in shortlist}
        ],
        "sources": selected,
    }
    path = OUT / "selection.json"
    if path.exists() and read(path) != selection:
        raise ValueError("expansion selection differs from its saved authority")
    save(path, selection)
    numeric = {
        r["sourceDocumentId"]: r["contracts"]
        for r in map(
            json.loads,
            (
                ROOT
                / "artifacts/kie-synthesis-production/numeric-contract-catalogs"
                / "mpci-bl-numeric-contracts-discharge-country-v6/contracts.jsonl"
            )
            .read_text()
            .splitlines(),
        )
    }
    ownership = {"version": 1, "sources": {}}
    auxiliary = {
        "version": 1,
        "policy": "generic_fictional_trade_references_not_national_registration_validation",
        "sources": {},
    }
    capabilities = {}
    for old in (base, trans):
        old_ownership = yaml.safe_load((ROOT / old["ownership"]).read_text())
        old_auxiliary = yaml.safe_load((ROOT / old["auxiliary"]).read_text())
        for sid in old["source_ids"]:
            save(
                OUT / "sources" / sid / "contract.json",
                read(ROOT / old["source_contracts"] / sid / "contract.json"),
            )
            save(
                OUT / "catalog" / sid / "template.json",
                read(ROOT / old["historical_catalog"] / sid / "template.json"),
            )
            ownership["sources"][sid] = old_ownership["sources"][sid]
            matches = [
                v for k, v in old_auxiliary["sources"].items() if sid == k or sid[4:].startswith(k)
            ]
            if len(matches) != 1:
                raise ValueError("ambiguous original auxiliary declaration")
            auxiliary["sources"][sid] = matches[0]
            capabilities[sid] = old["capabilities"][sid]
    reports = []
    for entry in selected:
        sid = entry["documentId"]
        row = rows[sid]
        historical = read(ROOT / base["historical_catalog"] / sid / "template.json")
        save(OUT / "catalog" / sid / "template.json", historical)
        error = None
        contract, missing = None, []
        try:
            contract, missing = draft(row, historical, numeric.get(sid, {}), {})
            compiled = SourceContract.model_validate(contract)
            compile_contract(row, compiled)
            build_owned_blueprint(row, historical, compiled, {})
        except (ValueError, KeyError) as exc:
            error = str(exc)
        envelope = dict(
            sourceSha256=digest(row["joinedRawText"].encode()),
            targetSha256=digest(row["target"]),
            contract=contract,
        )
        save(OUT / "sources" / sid / "draft.json", {**envelope, "missing": missing, "error": error})
        if not missing and not error:
            save(OUT / "sources" / sid / "contract.json", envelope)
        reports.append(dict(documentId=sid, missing=missing, error=error))
        capabilities[sid] = {"family": "ambient"}
        ownership["sources"][sid] = {}
        auxiliary["sources"][sid] = {}
    CONTRACTS.mkdir(parents=True, exist_ok=True)
    for name, value in (("ownership", ownership), ("auxiliary", auxiliary)):
        path = CONTRACTS / f"{name}.yaml"
        if not path.exists():
            path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    config = deepcopy(base)
    config.update(
        output=str(OUT.relative_to(ROOT)),
        seed=202610091,
        variants_per_source=2,
        source_ids=list(base["source_ids"])
        + list(trans["source_ids"])
        + [r["documentId"] for r in selected],
        capabilities=capabilities,
        source_contracts=str((OUT / "sources").relative_to(ROOT)),
        historical_catalog=str((OUT / "catalog").relative_to(ROOT)),
        ownership=str((CONTRACTS / "ownership.yaml").relative_to(ROOT)),
        auxiliary=str((CONTRACTS / "auxiliary.yaml").relative_to(ROOT)),
    )
    if not CONFIG.exists():
        CONFIG.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    save(OUT / "draft-inventory.json", reports)
    print(
        json.dumps(
            {
                "existing": len(known),
                "shortlist": len(selection["shortlist"]),
                "additional": len(selection["additional"]),
                "draft_pass": sum(not r["missing"] and not r["error"] for r in reports),
                "needs_work": sum(bool(r["missing"] or r["error"]) for r in reports),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-replacements", type=Path)
    parser.add_argument("--admit-reviewed", action="store_true")
    arguments = parser.parse_args()
    if arguments.admit_reviewed:
        admit_reviewed()
    elif arguments.stage_replacements:
        stage_replacements(arguments.stage_replacements)
    else:
        prepare()
