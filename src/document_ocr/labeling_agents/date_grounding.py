"""Review diagnostics for explicitly captioned numeric dates, never auto-labeling.

Only date tokens directly attached to an unambiguous field caption are checked.
Detached labels, place-name interveners and table ordering stay with the reviewer;
this gate neither guesses those boundaries nor edits the extraction.
"""

from __future__ import annotations

import re
from typing import Any

from document_ocr.date_policy import adjacent_printed_date_order, resolve_numeric_date
from document_ocr.labeling_agents.direct_models import ReviewFinding

_HEADS = {
    "issueDate": (
        r"(?:DATE\s+OF\s+ISSUE\s+OF\s+(?:B/L|WAYBILL)|"
        r"DATE(?:\s+AND\s+PLACE|\s*&\s*PLACE)\s+(?:OF\s+)?ISSU(?:E|ED)|"
        r"PLACE\s+AND\s+DATE\s+OF\s+ISSUE)"
    ),
    "shippedOnBoardDate": r"(?:(?:SHIPPED|SHIP|LADEN|CLEAN)\s+)?ON\s+BOARD(?:\s+DATE)?",
}
_OWNED = {
    field: re.compile(
        rf"(?<!\w){heading}[ \t:]*\n?[ \t]*"
        r"(?P<date>\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4})(?!\d)",
        re.I,
    )
    for field, heading in _HEADS.items()
}


def date_fidelity_findings(patch: dict[str, Any], ocr: str) -> list[ReviewFinding]:
    """Ask the metadata reviewer to resolve missing/misnormalized headed dates."""
    findings = []
    for field, pattern in _OWNED.items():
        matches = list(pattern.finditer(ocr))
        if not matches:
            continue
        expected = set()
        for match in matches:
            try:
                order = adjacent_printed_date_order(ocr, match.start("date"), match.end("date"))
                expected.add(resolve_numeric_date(match["date"], explicit_order=order).isoformat())
            except ValueError:
                # Calendar-impossible source is evidence to review, not a fallback.
                expected.add("invalid/conflicting source date evidence")
        if len(expected) == 1 and patch.get(field) in expected:
            continue
        findings.append(
            ReviewFinding(
                field=field,
                issue="ambiguous" if len(expected) != 1 else "wrong_value",
                explanation=(
                    "Direct date-caption diagnostic: OCR prints "
                    + repr([m.group() for m in matches])
                    + f"; candidate {field} is {patch.get(field)!r}. The default numeric "
                    "interpretation is " + repr(sorted(expected)) + "."
                ),
                suggestedCorrection=(
                    "Verify this caption's ownership and any explicit format or unambiguous "
                    "same-convention evidence. Use that evidence first, otherwise DD-MM; "
                    "order ambiguity alone does not justify omission. Explain a printed "
                    "format override or competing source dates rather than guessing. "
                    "This diagnostic has not applied a correction."
                ),
            )
        )
    return findings
