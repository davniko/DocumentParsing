"""Narrow literal-fidelity gates for direct review, not ownership/completeness proofs."""

from __future__ import annotations

import re
from typing import Any

from document_ocr.labeling_agents.direct_cargo import identifier_present
from document_ocr.labeling_agents.direct_models import ReviewFinding, Section


def source_fidelity_findings(
    candidate: dict[str, Any], ocr: str, section: Section
) -> list[ReviewFinding]:
    """Flag absent HS digit sequences and main-carriage names/identifiers.

    Formatting separators may differ. A successful match establishes only literal
    support; the reviewer must still decide field meaning, ownership and completeness.
    This function never repairs a value or obtains evidence from PDF pixels.
    """
    patch = candidate.get("documentPatch", {})
    values: list[tuple[str, str, str]] = []
    if section == "cargo":
        for index, goods in enumerate(patch.get("goodsItemDetails") or []):
            for code in goods.get("hsCodes") or []:
                if not isinstance(code, str) or not code.isascii() or not code.isdigit():
                    continue  # The application schema reports malformed value types.
                # Do not match a prefix/suffix of a longer continuous or dotted code.
                pattern = r"(?<![\w.])" + r"[.\s]*".join(code) + r"(?!\w|\.\d)"
                values.append((f"goodsItemDetails[{index}].hsCodes", code, pattern))
    elif section == "route_transport":
        transport = patch.get("transport") or {}
        for field in ("vesselName", "vesselImoNumber", "voyageNumber"):
            value = transport.get(field)
            if not isinstance(value, str):
                continue
            characters = [c for c in value if c.isalnum()]
            if not characters:
                continue
            pattern = r"(?<!\w)" + r"[\W_]*".join(map(re.escape, characters)) + r"(?!\w)"
            values.append((f"transport.{field}", value, pattern))
    findings = [
        ReviewFinding(
            field=field,
            issue="unsupported",
            explanation=(
                f"Literal-fidelity gate: {value!r} has no complete OCR match under "
                "the field's permitted separator normalization. PDF text is not OCR evidence."
            ),
            suggestedCorrection=(
                "Check the exact OCR spelling/digit sequence and field ownership; correct "
                "to the supported value, or omit the unsupported value. Do not pad, "
                "truncate or invent characters."
            ),
        )
        for field, value, pattern in values
        if re.search(pattern, ocr, flags=re.IGNORECASE) is None
    ]
    identifiers = []
    if section == "equipment":
        for i, row in enumerate(patch.get("containerInformation") or []):
            identifiers.append(
                (f"containerInformation[{i}].equipmentIdentifier", row.get("equipmentIdentifier"))
            )
            identifiers.extend(
                (f"containerInformation[{i}].sealNumbers", value)
                for value in row.get("sealNumbers") or []
            )
    elif section == "cargo":
        for i, goods in enumerate(patch.get("goodsItemDetails") or []):
            identifiers.extend(
                (f"goodsItemDetails[{i}].splitGoodsPlacement", p.get("equipmentIdentifier"))
                for p in goods.get("splitGoodsPlacement") or []
            )
    for field, value in identifiers:
        if isinstance(value, str) and not identifier_present(value, ocr):
            findings.append(
                ReviewFinding(
                    field=field,
                    issue="unsupported",
                    explanation=(
                        f"Identifier {value!r} is absent from OCR under separator "
                        "normalization. PDF-only identifiers cannot be target values."
                    ),
                    suggestedCorrection=(
                        "Omit this unsupported identifier/placement; preserve "
                        "independently supported cargo facts. Do not repair "
                        "characters from the PDF."
                    ),
                )
            )
    return findings
