"""Prepare a controlled production multi-row batch from reviewed exact owners."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_contracts import compile_declarations
from document_ocr.synthesis.declared_package_preparation import prepare
from document_ocr.synthesis.independent_campaign import _complete_optional_package_suffix
from document_ocr.synthesis.independent_preparation import _load_source_context, prepare_campaign
from document_ocr.synthesis.independent_wording import prepare_blueprint


def compile_batch(
    config_path,
    out,
    prefixes,
    count,
    *,
    group_counts=None,
    context=None,
    reviewed=None,
    seed=None,
    source_group_counts=None,
):
    root = Path.cwd()
    config = json.loads(config_path.read_text())
    selected = [
        sid for sid in config["source_ids"] if any(sid.startswith("doc_" + p) for p in prefixes)
    ]
    if len(selected) != len(prefixes):
        raise ValueError("requested source prefixes are not unique and complete")
    config["source_ids"] = selected
    config["capabilities"] = {sid: config["capabilities"][sid] for sid in selected}
    counts = [count] if group_counts is None else group_counts
    if not counts or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError("positive explicit package group counts required")
    config["variants_per_source"] = len(counts)
    config.pop("template_sampling", None)
    config["output"] = str(out / "campaign")
    config["seed"] = 2026103100 + count if seed is None else seed
    config["concurrency"] = 4
    config["budget_usd"] = 10.0
    equipment = config["sampling"].get("equipment_inventory")
    if equipment:
        equipment["source_ids"] = [sid for sid in equipment["source_ids"] if sid in selected]
        if not equipment["source_ids"]:
            config["sampling"].pop("equipment_inventory")
    if context is None:
        context = _load_source_context(root, config)
    contracts = {}
    source_counts = dict(source_group_counts or {})
    if not set(source_counts) <= set(selected) or any(
        len(values) != len(counts) or any(type(n) is not int or n < 1 for n in values)
        for values in source_counts.values()
    ):
        raise ValueError("explicit per-source group counts must align with selected variants")
    for sid in selected:
        bp = context.numeric.apply_blueprint(context.source.blueprint(sid), sid)
        if sid in context.wording:
            bp = prepare_blueprint(bp, context.wording[sid])
        bp = _complete_optional_package_suffix(bp)
        override = (reviewed or {}).get(sid)
        if override is None:
            projection, owners = compile_declarations(bp)
        else:
            projection, owners = override["projection"], override["declarations"]
        certificate = context.numeric.certificate(sid)
        article_groups = (certificate.get("cargoPresentation") or {}).get("articleGroups")
        constraints = {item["kind"] for item in certificate.get("constraints", [])}
        if constraints - {"equal_container_loads", "mixed_package_conservation"}:
            raise ValueError("source requires an explicit constrained-group contract: " + sid)
        if len(constraints) > 1:
            raise ValueError("combined equality/mixed constraints need a joint contract: " + sid)
        contracts[sid] = {
            "sourceSha256": digest(bp.source.encode()),
            "sourceTargetSha256": digest(bp.target),
            "projection": projection,
            "declarations": owners,
            "planning": "shared_units"
            if "equal_container_loads" in constraints
            else "independent_groups",
            "sourceContractScope": (
                "adjacent count/type owners; direct source quantity bindings; "
                "complete private clauses remain group-specific and bare total cells are additive"
            ),
        }
        if override is not None:
            contracts[sid].update(
                {k: v for k, v in override.items() if k not in {"projection", "declarations"}}
            )
        if article_groups is not None:
            contracts[sid]["planning"] = "shared_units"
            contracts[sid]["sourceContractScope"] += (
                "; fixed article manifest is preserved, not cloned for additional public rows"
            )
            if len(article_groups) == 1:
                source_counts[sid] = [1] * len(counts)
            elif max(counts) > len(article_groups):
                raise ValueError("public row count exceeds fixed article accounting groups")
        if (
            not bp.target["documentPatch"].get("containerInformation")
            and contracts[sid]["planning"] == "independent_groups"
        ):
            contracts[sid]["uncontainerizedBudget"] = "per_group"
    path = out / "source-contracts.json"
    atomic_publish_json(
        path, {"version": 1, "sources": contracts, "status": "AWAITING_SYNTHESIS_ACCEPTANCE"}
    )
    atomic_publish_json(out / "base-config.json", config)
    policy = {
        "contracts": {"path": str(path), "sha256": sha256_file(path)},
        "group_counts": counts,
        "utilization": [0.35, 0.75],
        **({"source_group_counts": source_counts} if source_counts else {}),
    }
    atomic_publish_json(out / "policy.json", policy)
    return config, policy


async def run(out):
    config = json.loads((out / "base-config.json").read_text())
    base = out / "base-preparation.json"
    if base.exists():
        result = json.loads(base.read_text())
    else:
        result = await prepare_campaign(Path.cwd(), config, **config["independent_preparation"])
        atomic_publish_json(base, result)
    if result["config"] is None:
        print(
            json.dumps({"status": "BASE_PREPARATION_HELD", "counts": result["counts"]}), flush=True
        )
        return
    config = copy.deepcopy(result["config"])
    config["declared_packages"] = json.loads((out / "policy.json").read_text())
    result = await prepare(Path.cwd(), config)
    atomic_publish_json(out / "group-preparation.json", result)
    if result["config"] is not None:
        atomic_publish_json(out / "prepared-config.json", result["config"])
    print(json.dumps({k: v for k, v in result.items() if k != "config"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthesis/mpci_bl_curated_v7_package_aliases.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefixes", nargs="+")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--group-counts", type=int, nargs="+")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prefixes:
        compile_batch(
            args.config, args.output, args.prefixes, args.groups, group_counts=args.group_counts
        )
    if args.run:
        asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
