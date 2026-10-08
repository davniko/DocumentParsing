"""Narrow literal-fidelity gates for direct review, not ownership/completeness proofs."""

from __future__ import annotations

import re
from typing import Any

from document_ocr.labeling_agents.direct_cargo import identifier_present
from document_ocr.labeling_agents.direct_models import ReviewFinding, Section

_ORDER_INSTRUCTION = re.compile(
    r"^[ \t]*(?:[\"'\u201c\u201d\u2018\u2019])?(?:CONSIGNEE[ \t]*:[ \t]*)?"
    r"(?:TO\s+(?:THE\s+)?ORDER\b|THE\s+ORDER\s+OF\b|CONSIGNED\s+TO\s+ORDER\s+OF\b)"
    r"[^\r\n]*",
    re.IGNORECASE | re.MULTILINE,
)


def order_consignment_candidates(ocr: str) -> list[str]:
    """Find affirmative order wording for ownership review, not automatic labeling.

    Anchoring excludes conditional consignee captions and narrative boilerplate.
    Newlines within a short instruction are allowed. The reviewer still establishes
    that the instruction belongs to the consignee and that an OF field is populated.
    No fixed character window, party-name lookup or PDF-only evidence is used.
    """
    return list(dict.fromkeys(match.group().strip() for match in _ORDER_INSTRUCTION.finditer(ocr)))


def source_fidelity_findings(
    candidate: dict[str, Any], ocr: str, section: Section
) -> list[ReviewFinding]:
    """Flag absent identifiers and missed affirmative order-consignment wording.

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
                # Match presentation separators, never a prefix/suffix of a longer code.
                pattern = r"(?<![\w.-])" + r"[.\s-]*".join(code) + r"(?!\w|[.-]\d)"
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
    if section == "metadata_freight":
        instructions = order_consignment_candidates(ocr)
        if instructions and patch.get("negotiability") != "negotiable":
            findings.append(
                ReviewFinding(
                    field="negotiability",
                    issue="wrong_value",
                    explanation=(
                        "OCR contains affirmative order-consignment wording: "
                        + repr(instructions)
                        + ". Check its consignee ownership and any named continuation; "
                        "the normalized consignee name alone loses this instruction."
                    ),
                    suggestedCorrection=(
                        "Set negotiability to negotiable when this is the actual consignee "
                        "instruction, including a populated Consigned to order of field. "
                        "Copy/document titles do not override it under the target policy. "
                        "If the wording belongs elsewhere or the OF field is empty, explain "
                        "that ownership instead. This diagnostic does not edit labels."
                    ),
                )
            )
        elif not instructions and (
            "negotiability" not in patch
            or patch["negotiability"]
            != ("non_negotiable" if (patch.get("parties") or {}).get("consignee") else None)
        ):
            findings.append(
                ReviewFinding(
                    field="negotiability",
                    issue="wrong_value",
                    explanation=(
                        "Negotiability is mandatory. The literal scan found no affirmative "
                        "consignee order instruction, and the decision conflicts with the "
                        "candidate's consignee availability. Check the actual OCR block for "
                        "missing party details or equivalent order wording missed by this scan."
                    ),
                    suggestedCorrection=(
                        "Use non_negotiable for a readable named consignee without order "
                        "wording, negotiable for an actual order instruction, or null when "
                        "OCR does not establish the consignee instruction. Check ownership; "
                        "copy stamps and another role's identity cannot fill a missing block. "
                        "This diagnostic does not edit labels."
                    ),
                )
            )
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
