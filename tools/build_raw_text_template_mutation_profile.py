#!/usr/bin/env python3
"""Build an exact-line, source-template mutation profile from reviewed audit evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.synthesis.raw_text_inventory import parse_template_mutation_profile


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certification-run", action="append", type=Path, required=True)
    parser.add_argument("--manual-findings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular JSON file: {path}")
    return json.loads(path.read_bytes())


def _source_lines(case_root: Path) -> tuple[str, ...]:
    path = case_root / "source.txt"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"certification case lacks source text: {case_root}")
    return tuple(path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    arguments = _arguments()
    evidence: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    source_text_by_id: dict[str, str] = {}
    sources: list[dict[str, str]] = []

    for raw_root in arguments.certification_run:
        root = raw_root.resolve(strict=True)
        commit = root / "_COMMIT.json"
        if root.is_symlink() or not root.is_dir() or commit.is_symlink() or not commit.is_file():
            raise ValueError(f"certification run is not committed: {root}")
        sources.append(
            {
                "path": raw_root.as_posix(),
                "sha256": sha256_file(commit),
                "kind": "certification_commit",
            }
        )
        for case_root in sorted((root / "cases").iterdir()):
            if not case_root.is_dir() or case_root.is_symlink():
                continue
            document_id = case_root.name
            source_path = case_root / "source.txt"
            source_text = source_path.read_text(encoding="utf-8")
            prior = source_text_by_id.setdefault(document_id, source_text)
            if prior != source_text:
                raise ValueError(f"certification sources disagree for {document_id}")
            stages = _read_json(case_root / "stages.json")
            if not isinstance(stages, list):
                raise ValueError(f"certification stages are not a list: {case_root}")
            outputs = [
                row["modelOutput"]
                for row in stages
                if isinstance(row, dict)
                and isinstance(row.get("modelOutput"), dict)
                and row.get("hostError") is None
            ]
            if not outputs:
                continue
            findings = outputs[-1].get("findings")
            if not isinstance(findings, list):
                raise ValueError(f"certification output lacks findings: {case_root}")
            for finding in findings:
                if not isinstance(finding, dict) or not isinstance(
                    finding.get("findingKind"), str
                ):
                    raise ValueError(f"invalid certification finding: {case_root}")
                rows = finding.get("evidence")
                if not isinstance(rows, list):
                    raise ValueError(f"finding evidence is not a list: {case_root}")
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(row.get("lineId"), str):
                        raise ValueError(f"invalid finding evidence: {case_root}")
                    evidence[document_id][row["lineId"]].add(finding["findingKind"])

    manual_path = arguments.manual_findings.resolve(strict=True)
    manual = _read_json(manual_path)
    if not isinstance(manual, dict) or manual.get("schemaVersion") != 1:
        raise ValueError("manual findings have an invalid root")
    manual_cases = manual.get("cases")
    if not isinstance(manual_cases, list):
        raise ValueError("manual findings cases are not a list")
    sources.append(
        {
            "path": arguments.manual_findings.as_posix(),
            "sha256": sha256_file(manual_path),
            "kind": "manual_audit",
        }
    )
    for case in manual_cases:
        if not isinstance(case, dict) or not isinstance(case.get("documentId"), str):
            raise ValueError("manual finding case is invalid")
        document_id = case["documentId"]
        if document_id not in source_text_by_id:
            raise ValueError(f"manual finding lacks a certified source: {document_id}")
        rows = case.get("lines")
        if not isinstance(rows, list):
            raise ValueError(f"manual finding lines are invalid: {document_id}")
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("lineId"), str)
                or not isinstance(row.get("findingKind"), str)
            ):
                raise ValueError(f"manual finding line is invalid: {document_id}")
            evidence[document_id][row["lineId"]].add(row["findingKind"])

    cases: list[dict[str, Any]] = []
    for document_id in sorted(evidence):
        source_text = source_text_by_id[document_id]
        lines = source_text.splitlines()
        profiled_lines: list[dict[str, Any]] = []
        for line_id in sorted(evidence[document_id], key=lambda value: int(value[1:])):
            number = int(line_id[1:])
            if not 1 <= number <= len(lines):
                raise ValueError(f"profile line is outside source: {document_id} {line_id}")
            source_line = lines[number - 1]
            if not source_line.strip() or source_line.startswith("--- PAGE "):
                raise ValueError(f"profile owns blank/page-marker: {document_id} {line_id}")
            profiled_lines.append(
                {
                    "lineId": line_id,
                    "sourceLineSha256": sha256_bytes(source_line.encode("utf-8")),
                    "findingKinds": sorted(evidence[document_id][line_id]),
                }
            )
        cases.append(
            {
                "documentId": document_id,
                "sourceTextSha256": sha256_bytes(source_text.encode("utf-8")),
                "lines": profiled_lines,
            }
        )

    payload = {
        "schemaVersion": 1,
        "name": arguments.name,
        "sources": sources,
        "cases": cases,
    }
    parse_template_mutation_profile(payload)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "documents": len(cases),
                "lines": sum(len(row["lines"]) for row in cases),
                "output": arguments.output.as_posix(),
                "sha256": sha256_file(arguments.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
