"""Inventory unbound equal-valued numbers for source-level role adjudication.

This is a screen, not an equality-based replacement rule. Equipment dimensions,
page numbers and boilerplate can share numbers with shipment facts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from document_ocr.synthesis.curated import (
    SourceContract,
    _number_style,
    compile_contract,
    digest,
    save,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = yaml.safe_load((ROOT / "configs/synthesis/mpci_bl_curated_v7_pilot24.yaml").read_text())
    rows = {
        r["documentId"]: r
        for r in map(json.loads, (ROOT / cfg["dataset"] / "train.jsonl").read_text().splitlines())
    }
    reports = []
    for sid in cfg["source_ids"]:
        row = rows[sid]
        raw = row["joinedRawText"]
        contract = SourceContract.model_validate(
            json.loads((ROOT / cfg["output"] / "sources" / sid / "contract.json").read_text())[
                "contract"
            ]
        )
        template, _ = compile_contract(row, contract)
        occupied = [(s.byte_start, s.byte_end) for s in template.slots]
        flags = []
        for match in re.finditer(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w.])", raw):
            start = len(raw[: match.start()].encode())
            end = len(raw[: match.end()].encode())
            if any(a < end and start < b for a, b in occupied):
                continue
            candidates = []
            for variable in contract.variables:
                if variable.kind not in {"count", "mass", "volume"}:
                    continue
                try:
                    _number_style(match[0], variable.value, variable.value)
                except ValueError:
                    continue
                candidates.append(dict(key=variable.key, kind=variable.kind, value=variable.value))
            if candidates:
                line = raw.count("\n", 0, match.start()) + 1
                flags.append(
                    dict(
                        line=line,
                        text=raw.splitlines()[line - 1],
                        token=match[0],
                        candidates=candidates,
                    )
                )
        reports.append(
            dict(source=sid, contractSha256=digest(contract.model_dump(mode="json")), flags=flags)
        )
    save(ROOT / cfg["output"] / "numeric-repetition-screen.json", reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
