"""Reproducible source inventory and current-label rebinding for the route pilot.

No source OCR or real labels are changed. Drafts with missing ownership do not
become executable contracts. Reviewed corrections live in the pilot YAMLs.
"""

import json
import re
from pathlib import Path

import yaml
from rebase_curated_v7 import draft

from document_ocr.synthesis.curated import SourceContract, compile_contract, digest, save

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/kie-synthesis-production/curated-v7-transshipment-pilot"
PREFIXES = ("551657f3", "8579256f")


def main():
    base = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_registry_pilot72.yaml").read_text()
    )
    dataset = ROOT / base["dataset"]
    rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
    inventory = []
    for split in ("train", "validation"):
        for row in map(json.loads, (dataset / f"{split}.jsonl").read_text().splitlines()):
            lines = [
                (i, line)
                for i, line in enumerate(row["joinedRawText"].splitlines(), 1)
                if re.search(r"\btrans?[\s-]?ship|\bT/S\b", line, re.I)
            ]
            route = row["target"]["documentPatch"].get("route", {})
            if lines or "transshipmentPort" in route:
                inventory.append(
                    dict(documentId=row["documentId"], split=split, route=route, mentions=lines)
                )
    save(OUTPUT / "source-inventory.json", inventory)
    numeric_path = (
        ROOT
        / "artifacts/kie-synthesis-production/numeric-contract-catalogs"
        / "mpci-bl-numeric-contracts-discharge-country-v6/contracts.jsonl"
    )
    sidecars = {
        r["sourceDocumentId"]: r["contracts"]
        for r in map(json.loads, numeric_path.read_text().splitlines())
    }
    hints_path = ROOT / "configs/synthesis/contracts/curated_v7_transshipment_rebinding.yaml"
    hints = yaml.safe_load(hints_path.read_text())
    selected = [r for r in rows if r["documentId"][4:12] in PREFIXES]
    if len(selected) != len(PREFIXES):
        raise ValueError("pilot source selection is missing or ambiguous")
    for row in selected:
        sid = row["documentId"]
        if sid[4:12] == "551657f3":
            # This source has no historical compiled template. The reviewed
            # ownership YAML supplies its complete non-lexical binding map.
            template = {"document_id": sid, "bindings": []}
            numeric = []
        else:
            template = json.loads(
                (ROOT / base["historical_catalog"] / sid / "template.json").read_text()
            )
            numeric = sidecars[sid]
        save(OUTPUT / "catalog" / sid / "template.json", template)
        contract, missing = draft(row, template, numeric, hints.get(sid, {}))
        error = None
        try:
            compile_contract(row, SourceContract.model_validate(contract))
        except ValueError as exc:
            error = str(exc)
        envelope = dict(
            sourceSha256=digest(row["joinedRawText"].encode()),
            targetSha256=digest(row["target"]),
            contract=contract,
        )
        save(
            OUTPUT / "sources" / sid / "draft.json",
            {**envelope, "missing": missing, "error": error},
        )
        if not missing and error is None:
            save(OUTPUT / "sources" / sid / "contract.json", envelope)
        print(sid, "missing:", missing, "error:", error)
        if missing or error:
            raise ValueError(f"source contract did not compile: {sid}; inspect its draft receipt")


if __name__ == "__main__":
    main()
