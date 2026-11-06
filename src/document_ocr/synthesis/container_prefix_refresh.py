"""Change container identities only, preserving each printed formatting mask."""

import re
from copy import deepcopy
from itertools import pairwise

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_container_ids import format_container_identifier
from document_ocr.synthesis.generators import DeterministicStream, generate_unique_container_numbers


def refresh_prefix(row, *, prefix, seed, excluded):
    before = row["joinedRawText"]
    if digest(before.encode()) != row["joinedRawTextSha256"]:
        raise ValueError("input text hash differs")
    ids = [c["equipmentIdentifier"] for c in row["target"]["containerInformation"]]
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError("distinct multiple containers required")
    replacements = generate_unique_container_numbers(
        owner_by_identity={value: prefix for value in ids},
        stream=DeterministicStream(seed, "shared-container-prefix", row["documentId"]),
        excluded=set(excluded) | set(ids),
    )
    spans = []
    for old, new in replacements.items():
        pattern = re.compile(r"(?<![A-Z0-9])" + r"[ \t./-]*".join(old) + r"(?![A-Z0-9])")
        matches = list(pattern.finditer(before))
        if not matches:
            raise ValueError("container has no unambiguous printed occurrence: " + old)
        for match in matches:
            value = format_container_identifier(match.group(), old, new)
            spans.append(
                dict(start=match.start(), end=match.end(), before=match.group(), after=value)
            )
    spans.sort(key=lambda s: s["start"])
    if any(a["end"] > b["start"] for a, b in pairwise(spans)):
        raise ValueError("overlapping container occurrences")
    raw = before
    for span in reversed(spans):
        raw = raw[: span["start"]] + span["after"] + raw[span["end"] :]
    old_lines, new_lines = before.splitlines(keepends=True), raw.splitlines(keepends=True)
    positioned = row["positionedText"].splitlines(keepends=True)
    if len(old_lines) != len(new_lines) or len(positioned) != len(old_lines):
        raise ValueError("container replacement changed line topology")
    output_lines = []
    for old, new, located in zip(old_lines, new_lines, positioned, strict=True):
        old_text = old.rstrip("\r\n")
        new_text = new.rstrip("\r\n")
        if not located.startswith(old_text):
            raise ValueError("positioned line text differs")
        suffix = located[len(old_text) :]
        if suffix.strip() and not re.fullmatch(r" \|\|(?: \d+,\d+)?\s*", suffix):
            raise ValueError("unexpected positioned suffix")
        output_lines.append(new_text + suffix)
    after = deepcopy(row)
    for c in after["target"]["containerInformation"]:
        c["equipmentIdentifier"] = replacements[c["equipmentIdentifier"]]
    for g in after["target"].get("goodsItemDetails", []):
        for allocation in g.get("splitGoodsPlacement", []):
            allocation["equipmentIdentifier"] = replacements[allocation["equipmentIdentifier"]]
    after.update(
        joinedRawText=raw,
        joinedRawTextSha256=digest(raw.encode()),
        positionedText="".join(output_lines),
        positionedTextSha256=digest("".join(output_lines).encode()),
    )
    return after, dict(
        beforeSha256=digest(row),
        afterSha256=digest(after),
        identifiers=replacements,
        spans=spans,
        coordinatePolicy="same line anchors and unchanged eleven-character ID masks",
    )
