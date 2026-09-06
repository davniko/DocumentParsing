#!/usr/bin/env python3
"""Audit a raw-text certification run against its immutable inventory run.

This tool deliberately does not assign semantic quality automatically.  It materializes every
changed OCR line, identifies which stage changed it, and records whether the original compiler
granted write authority for that line.  The resulting workbook is the evidence surface for a
human quality adjudication and for tightening the compiler contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON object array: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_number(line_id: object) -> int:
    if not isinstance(line_id, str) or len(line_id) != 6 or not line_id.startswith("L"):
        raise ValueError(f"invalid line ID: {line_id!r}")
    return int(line_id[1:])


def _case_dirs(root: Path) -> dict[str, Path]:
    cases = root / "cases"
    if not cases.is_dir():
        raise ValueError(f"run has no cases directory: {root}")
    return {path.name: path for path in cases.iterdir() if path.is_dir()}


def _assert_committed(root: Path) -> None:
    if not (root / "_COMMIT.json").is_file():
        raise ValueError(f"run is not committed: {root}")


def _source_line_ids(rows: Iterable[Mapping[str, Any]]) -> set[int]:
    output: set[int] = set()
    for row in rows:
        line_ids = row.get("sourceLineIds", row.get("lineIds", ()))
        if not isinstance(line_ids, Sequence) or isinstance(line_ids, (str, bytes)):
            continue
        output.update(_line_number(value) for value in line_ids)
    return output


def _compiler_authority(case: Path) -> tuple[set[int], dict[int, set[str]]]:
    authority: set[int] = set()
    reasons: dict[int, set[str]] = {}

    def add(number: int, reason: str) -> None:
        authority.add(number)
        reasons.setdefault(number, set()).add(reason)

    for slot in _read_json_array(case / "model-slots.json"):
        number = _line_number(slot.get("lineId"))
        add(number, "model_slot")
    for row in _read_json_array(case / "deterministic-edits.json"):
        for number in _source_line_ids((row,)):
            add(number, "deterministic_auxiliary")
    for row in _read_json_array(case / "inventory.json"):
        for number in _source_line_ids((row,)):
            add(number, f"inventory:{row.get('category', 'unknown')}")

    contract = _read_json(case / "contract.json")
    contract_collections = {
        "anchoredScalarReplacementRequirements": "anchored_scalar",
        "cargoFlavorRewriteRequirements": "cargo_flavor",
        "compoundPartyFlavorRequirements": "compound_party",
        "inlineSlotTopologyRequirements": "inline_topology",
        "jurisdictionalSurfaceRequirements": "jurisdiction",
        "operationalFlavorRequirements": "operational_flavor",
        "rawAuxiliaryIdentityRequirements": "raw_auxiliary_identity",
        "sourceStatusPreservationRequirements": "status_preservation",
        "surfaceRenderingRequirements": "surface_rendering",
    }
    for key, reason in contract_collections.items():
        rows = contract.get(key, ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for number in _source_line_ids(row for row in rows if isinstance(row, Mapping)):
            add(number, reason)
    return authority, reasons


def _line_changes(before: str, after: str) -> list[tuple[int, str, str]]:
    left = before.splitlines()
    right = after.splitlines()
    if len(left) != len(right):
        raise ValueError(f"line topology changed: {len(left)} -> {len(right)}")
    return [
        (number, source, target)
        for number, (source, target) in enumerate(zip(left, right, strict=True), start=1)
        if source != target
    ]


def _stage_usage(case: Path) -> tuple[int, int, int, int, float, int]:
    stages = _read_json_array(case / "stages.json")
    requests = input_tokens = reasoning_tokens = visible_tokens = failures = 0
    cost = 0.0
    for stage in stages:
        usage = stage.get("usage")
        if not isinstance(usage, Mapping):
            continue
        requests += int(usage.get("requests", 0))
        input_tokens += int(usage.get("inputTokens", 0))
        reasoning_tokens += int(usage.get("reasoningTokens", 0))
        visible_tokens += int(usage.get("visibleOutputTokens", 0))
        value = usage.get("providerReportedCostUsd", usage.get("estimatedCostUsd", 0))
        cost += float(value or 0)
        failures += int(stage.get("errorType") is not None)
    return requests, input_tokens, reasoning_tokens, visible_tokens, cost, failures


def _target_summary(label: Mapping[str, Any]) -> dict[str, Any]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return {}
    parties = patch.get("parties")
    party_names: dict[str, Any] = {}
    if isinstance(parties, Mapping):
        for role, party in parties.items():
            if isinstance(party, Mapping):
                party_names[str(role)] = party.get("name")
            elif isinstance(party, Sequence) and not isinstance(party, (str, bytes)):
                party_names[str(role)] = [
                    row.get("name") for row in party if isinstance(row, Mapping)
                ]
    cargo = patch.get("cargoGroups")
    groups = []
    if isinstance(cargo, Sequence) and not isinstance(cargo, (str, bytes)):
        for row in cargo:
            if isinstance(row, Mapping):
                groups.append(
                    {
                        key: row.get(key)
                        for key in (
                            "description",
                            "grossWeight",
                            "netWeight",
                            "volume",
                            "hsCodes",
                            "dangerousGoods",
                        )
                        if row.get(key) is not None
                    }
                )
    return {
        "billOfLadingNumber": patch.get("billOfLadingNumber"),
        "parties": party_names,
        "route": patch.get("route"),
        "transport": patch.get("transport"),
        "containers": patch.get("containers"),
        "cargoGroups": groups,
        "cargoPackages": patch.get("cargoPackages"),
        "cargoAllocationGroups": patch.get("cargoAllocationGroups"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _context(lines: Sequence[str], number: int, radius: int = 1) -> str:
    start = max(1, number - radius)
    end = min(len(lines), number + radius)
    return "\n".join(f"{index:05d}: {lines[index - 1]}" for index in range(start, end + 1))


def _case_markdown(
    *,
    document_id: str,
    source: str,
    inventory_output: str,
    final: str,
    target: Mapping[str, Any],
    authority_reasons: Mapping[int, set[str]],
) -> str:
    source_lines = source.splitlines()
    inventory_lines = inventory_output.splitlines()
    final_lines = final.splitlines()
    compiler = _line_changes(source, inventory_output)
    certification = _line_changes(inventory_output, final)
    all_numbers = sorted({row[0] for row in compiler} | {row[0] for row in certification})
    rows = [
        f"# {document_id}",
        "",
        "## Target summary",
        "",
        "```json",
        json.dumps(_target_summary(target), indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Changed-line audit",
        "",
        "| Line | Stage | Compiler authority | Source | Inventory output | Final |",
        "|---:|---|---|---|---|---|",
    ]
    compiler_numbers = {row[0] for row in compiler}
    certification_numbers = {row[0] for row in certification}
    for number in all_numbers:
        stage = (
            "compiler+certifier"
            if number in compiler_numbers and number in certification_numbers
            else "compiler"
            if number in compiler_numbers
            else "certifier-only"
        )
        reasons = ", ".join(sorted(authority_reasons.get(number, ()))) or "NONE"
        values = [source_lines[number - 1], inventory_lines[number - 1], final_lines[number - 1]]
        escaped = [value.replace("|", "\\|").replace("`", "\\`") for value in values]
        rows.append(
            f"| {number} | {stage} | {reasons} | `{escaped[0]}` | `{escaped[1]}` | "
            f"`{escaped[2]}` |"
        )
    rows.extend(
        [
            "",
            "## Context for certification-only edits",
            "",
        ]
    )
    if not certification:
        rows.append("No certification-only edits.")
    for number, before, after in certification:
        rows.extend(
            [
                f"### L{number:05d}",
                "",
                "Compiler authority: `"
                + (", ".join(sorted(authority_reasons.get(number, ()))) or "NONE")
                + "`",
                "",
                "Source context:",
                "",
                "```text",
                _context(source_lines, number),
                "```",
                "",
                "Before certification:",
                "",
                "```text",
                before,
                "```",
                "",
                "After certification:",
                "",
                "```text",
                after,
                "```",
                "",
            ]
        )
    rows.extend(
        [
            "## Human adjudication",
            "",
            "- Training usable: `TODO`",
            "- Issue categories: `TODO`",
            "- Evidence / correction: `TODO`",
            "",
        ]
    )
    return "\n".join(rows)


def _plots(frame: pd.DataFrame, output: Path) -> None:
    plot_root = output / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    long = frame.melt(
        id_vars="document_id",
        value_vars=["compiler_changed_lines", "certifier_changed_lines", "unauthorized_lines"],
        var_name="change_class",
        value_name="lines",
    )
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=long, x="change_class", y="lines")
    sns.stripplot(data=long, x="change_class", y="lines", color="black", alpha=0.45)
    plt.title("Changed OCR lines by authority stage")
    plt.xlabel("")
    plt.ylabel("lines per document")
    plt.xticks(rotation=12)
    plt.savefig(plot_root / "01_changed_lines_by_stage.png", dpi=180, bbox_inches="tight")
    plt.close()

    token_long = frame.melt(
        id_vars="document_id",
        value_vars=["input_tokens", "reasoning_tokens", "visible_tokens"],
        var_name="token_class",
        value_name="tokens",
    )
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=token_long, x="token_class", y="tokens")
    plt.title("Certification token composition")
    plt.xlabel("")
    plt.ylabel("tokens per document")
    plt.savefig(plot_root / "02_token_composition.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=frame,
        x="certifier_changed_lines",
        y="cost_usd",
        size="requests",
        hue="unauthorized_lines",
        palette="viridis",
        sizes=(40, 240),
    )
    plt.title("Certification edits, authority violations, and cost")
    plt.xlabel("certifier-changed lines")
    plt.ylabel("USD")
    plt.savefig(plot_root / "03_edits_vs_cost.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--certification-root", type=Path, required=True)
    parser.add_argument("--override-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inventory_root = args.inventory_root.resolve(strict=True)
    certification_root = args.certification_root.resolve(strict=True)
    override_roots = [path.resolve(strict=True) for path in args.override_root]
    for root in (inventory_root, certification_root, *override_roots):
        _assert_committed(root)

    inventory_cases = _case_dirs(inventory_root)
    certification_cases = _case_dirs(certification_root)
    override_cases: dict[str, Path] = {}
    for root in override_roots:
        for document_id, case in _case_dirs(root).items():
            if document_id in override_cases:
                raise ValueError(f"duplicate certification override: {document_id}")
            override_cases[document_id] = case
    if set(inventory_cases) != set(certification_cases):
        raise ValueError("inventory and certification document sets differ")
    if not set(override_cases) <= set(certification_cases):
        raise ValueError("certification override contains an unknown document")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_output = output / "cases"
    case_output.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    authority_counter: Counter[str] = Counter()
    for document_id in sorted(inventory_cases):
        inventory_case = inventory_cases[document_id]
        certification_case = override_cases.get(document_id, certification_cases[document_id])
        source = (inventory_case / "source.txt").read_text(encoding="utf-8")
        inventory_output = (inventory_case / "final.txt").read_text(encoding="utf-8")
        final = (certification_case / "final.txt").read_text(encoding="utf-8")
        if source != (certification_cases[document_id] / "source.txt").read_text(encoding="utf-8"):
            raise ValueError(f"certification source differs for {document_id}")
        authority, authority_reasons = _compiler_authority(inventory_case)
        compiler_changes = _line_changes(source, inventory_output)
        certification_changes = _line_changes(inventory_output, final)
        cert_numbers = {row[0] for row in certification_changes}
        unauthorized = sorted(cert_numbers - authority)
        for number in cert_numbers:
            reason_values = authority_reasons.get(number)
            if reason_values:
                authority_counter.update(reason_values)
            else:
                authority_counter["NONE"] += 1

        requests = input_tokens = reasoning_tokens = visible_tokens = failures = 0
        cost = 0.0
        stage_cases = [certification_cases[document_id]]
        if document_id in override_cases:
            stage_cases.append(override_cases[document_id])
        for case in stage_cases:
            values = _stage_usage(case)
            requests += values[0]
            input_tokens += values[1]
            reasoning_tokens += values[2]
            visible_tokens += values[3]
            cost += values[4]
            failures += values[5]

        result = _read_json(certification_case / "result.json")
        rows.append(
            {
                "document_id": document_id,
                "machine_status": result.get("status"),
                "source_lines": len(source.splitlines()),
                "compiler_changed_lines": len(compiler_changes),
                "certifier_changed_lines": len(certification_changes),
                "unauthorized_lines": len(unauthorized),
                "unauthorized_line_ids": "|".join(f"L{number:05d}" for number in unauthorized),
                "requests": requests,
                "provider_failures": failures,
                "input_tokens": input_tokens,
                "reasoning_tokens": reasoning_tokens,
                "visible_tokens": visible_tokens,
                "cost_usd": cost,
                "final_sha256": hashlib.sha256(final.encode()).hexdigest(),
                "override_used": document_id in override_cases,
                "human_training_usable": "TODO",
                "human_issue_categories": "TODO",
                "human_notes": "TODO",
            }
        )
        target = _read_json(inventory_case / "target-label.json")
        (case_output / f"{document_id}.md").write_text(
            _case_markdown(
                document_id=document_id,
                source=source,
                inventory_output=inventory_output,
                final=final,
                target=target,
                authority_reasons=authority_reasons,
            ),
            encoding="utf-8",
        )

    _write_csv(output / "cases.csv", rows)
    frame = pd.DataFrame(rows)
    _plots(frame, output)
    totals = {
        "schemaVersion": 1,
        "documents": len(rows),
        "inventoryRoot": str(inventory_root),
        "inventoryCommitSha256": _sha256(inventory_root / "_COMMIT.json"),
        "certificationRoot": str(certification_root),
        "certificationCommitSha256": _sha256(certification_root / "_COMMIT.json"),
        "overrideRoots": [str(root) for root in override_roots],
        "compilerChangedLines": int(frame["compiler_changed_lines"].sum()),
        "certifierChangedLines": int(frame["certifier_changed_lines"].sum()),
        "certifierEditsOutsideCompilerAuthority": int(frame["unauthorized_lines"].sum()),
        "documentsWithEditsOutsideCompilerAuthority": int((frame["unauthorized_lines"] > 0).sum()),
        "requests": int(frame["requests"].sum()),
        "providerFailures": int(frame["provider_failures"].sum()),
        "inputTokens": int(frame["input_tokens"].sum()),
        "reasoningTokens": int(frame["reasoning_tokens"].sum()),
        "visibleTokens": int(frame["visible_tokens"].sum()),
        "costUsd": float(frame["cost_usd"].sum()),
        "certificationEditAuthority": dict(authority_counter),
        "humanAuditComplete": False,
        "trainingRecordsPublished": False,
    }
    (output / "summary.json").write_text(
        json.dumps(totals, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    index = [
        "# Raw-text certification quality audit",
        "",
        "## Machine evidence",
        "",
        f"- Documents: **{totals['documents']}**.",
        f"- Compiler-changed lines: **{totals['compilerChangedLines']}**.",
        f"- Certification-changed lines: **{totals['certifierChangedLines']}**.",
        "- Certification edits outside compiler authority: "
        f"**{totals['certifierEditsOutsideCompilerAuthority']} lines across "
        f"{totals['documentsWithEditsOutsideCompilerAuthority']} documents**.",
        f"- Certification requests / failed provider attempts: **{totals['requests']} / "
        f"{totals['providerFailures']}**.",
        f"- Input / reasoning / visible tokens: **{totals['inputTokens']:,} / "
        f"{totals['reasoningTokens']:,} / {totals['visibleTokens']:,}**.",
        f"- Recorded provider-or-estimated cost: **${totals['costUsd']:.8f}**.",
        "- Human audit complete: **no**; training records published: **0**.",
        "",
        "## Cases",
        "",
        "| Document | Machine status | Compiler lines | Certifier lines | "
        "Outside authority | Audit |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        index.append(
            f"| `{row['document_id']}` | {row['machine_status']} | "
            f"{row['compiler_changed_lines']} | {row['certifier_changed_lines']} | "
            f"{row['unauthorized_lines']} | [workbook](cases/{row['document_id']}.md) |"
        )
    (output / "REPORT.md").write_text("\n".join(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
