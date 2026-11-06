"""Preservation, privacy ownership and relation gates for native Paddle candidates."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from document_ocr.spatial_inputs.paddle import RegionSpan, fingerprint, project, serialize
from document_ocr.spatial_inputs.paddle import pages_from_record as load_pages
from document_ocr.synthesis.paddle_bindings import PaddleReplacement, compile_blocks
from document_ocr.synthesis.paddle_render import spans_for


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_views(record, pages):
    require(record["joinedRawText"] == serialize(pages, geometry="none"), "raw view differs")
    require(record["positionedText"] == serialize(pages), "centroid view differs")
    require(record["bboxText"] == serialize(pages, geometry="bbox"), "bbox view differs")


def check_conservation(source, record, overlay):
    """Every unchanged native region and every final physical fragment survives."""
    pages = load_pages(source)
    require(record["sourceFingerprint"] == fingerprint(pages), "source fingerprint differs")
    changes = []
    for receipt in record["proof"]:
        spans = tuple(RegionSpan(**s) for s in receipt["spans"])
        project(pages, spans, separator="")  # Exact source evidence, not fuzzy matching.
        value = receipt["after"] if "after" in receipt else receipt["text"]
        changes.append(PaddleReplacement(receipt["owner"], spans, value))
    blocks = compile_blocks(
        pages,
        tuple(changes),
        physical_groups=tuple((p, tuple(ix)) for p, ix in overlay.get("physicalGroups", [])),
    )
    expected_blocks = {(b.page, b.owner): b for b in blocks if b.action == "replace"}
    found_blocks = {}
    for page in record["geometry"]["pages"]:
        require(not page.get("geometryFailures"), "geometry validator reported failures")
        for block in page.get("blocks", []):
            key = (page["page"], block["owner"])
            require(key not in found_blocks, "duplicate geometry block")
            found_blocks[key] = block
    require(expected_blocks.keys() == found_blocks.keys(), "geometry owner inventory differs")

    def compact(s):
        return "".join(s.split())

    for key, block in expected_blocks.items():
        physical = found_blocks[key]
        require(list(block.regions) == physical["sourceRegions"], "source regions differ")
        require(block.text == physical["text"], "rendered owner value differs")
        require(
            compact(block.text)
            == compact("".join(f["text"] for f in physical["physicalFragments"])),
            "physical wrapping lost or inserted content",
        )
    for page, rendered in zip(pages, load_pages(record), strict=True):
        owned = {i for b in blocks if b.page == page.index for i in b.regions}
        expected = Counter(r.text for r in page.regions if r.index not in owned)
        for (p, _), block in found_blocks.items():
            if p == page.index:
                expected.update(f["text"] for f in block["physicalFragments"])
        require(expected == Counter(r.text for r in rendered.regions), "unowned text changed")


def check_private_residue(privacy, rendered):
    def compact(s):
        return "".join(s.upper().split())

    text = compact(rendered["joinedRawText"])
    for edit in privacy["proof"]:
        before = compact(" ".join(s["expected"] for s in edit["spans"]))
        # This is an additional screen of reviewed owners, not general PII detection.
        if len(before) >= 5:
            survives = before in text
            if survives and before.isdecimal():
                # A complete old numeric identifier is sensitive, not every
                # occurrence of its digits inside a different, longer identifier.
                # Keep the fast substring path for absent values and retain the
                # stricter existing check for names and mixed-format identifiers.
                survives = re.search(r"(?<!\d)" + re.escape(before) + r"(?!\d)", text) is not None
            require(not survives, f"original private owner survives: {edit['owner']}")


def check_synthetic_relations(target):
    equipment = [c["equipmentIdentifier"] for c in target.get("containerInformation", [])]
    require(len(equipment) == len(set(equipment)), "duplicate generated equipment identity")
    for goods in target.get("goodsItemDetails", []):
        allocations = goods.get("splitGoodsPlacement", [])
        ids = [a["equipmentIdentifier"] for a in allocations]
        require(set(ids) <= set(equipment), "allocation references absent equipment")
        require(len(ids) == len(set(ids)), "duplicate generated allocation")
        packages = goods.get("numberAndTypeOfPackages", [])
        if (
            packages
            and ids
            and set(ids) == set(equipment)
            and all("packageQuantity" in a for a in allocations)
        ):
            require(
                sum(a["packageQuantity"] for a in allocations)
                == sum(p["packageQuantity"] for p in packages),
                "complete allocation ledger differs from declared packages",
            )


def source_privacy_edits(record):
    """Wider-only reference sources are not anonymized training records."""
    if record["privacyPlan"] is not None:
        return record["privacyPlan"]["rawEdits"]
    if record["split"] != "template-only":
        raise ValueError("training source requires its reviewed anonymization plan")
    return []


def check_new_private_ownership(source, record, decision):
    """Every newly reviewed private token needs a synthetic replacement owner.

    This checks native character ownership, not string absence: a generated
    value may legitimately repeat a short number. Separators between separately
    owned names need not be regenerated, but their alphanumeric content does.
    """
    covered = defaultdict(set)
    for owner in record["proof"]:
        for span in owner["spans"]:
            native = source["pages"][span["page"]]["regions"][span["region"]]["text"]
            if native[span["start"] : span["end"]] != span["expected"]:
                raise ValueError("private ownership proof differs from source")
            covered[(span["page"], span["region"])].update(range(span["start"], span["end"]))
    pages = load_pages(source)
    template_private = decision.get("templatePrivateOwners", [])
    if template_private and source["split"] != "template-only":
        raise ValueError("new real-training private owners also require anonymization")
    for index, extra in enumerate(decision["privacy"]["extras"] + template_private):
        for span in spans_for(extra, pages):
            required = {
                span.start + offset for offset, char in enumerate(span.expected) if char.isalnum()
            }
            if not required <= covered[(span.page, span.region)]:
                raise ValueError(f"new private source owner lacks a synthetic binding: {index}")


def check_private_lexemes(source, record, allowed):
    """Reject known private residues in real stages and synthetic replays alike."""
    pages = load_pages(record)
    used = set()
    for ix, edit in enumerate(source_privacy_edits(source)):
        before = edit["before"]
        expression = r"\s+".join(re.escape(s) for s in before.split())
        # Existing anonymization plans use newline guards for standalone marks.
        # Dropping those guards turns MP into a match inside PUMP or COMPANY.
        if before.startswith("\n"):
            expression = r"^\s*" + expression
        if before.endswith("\n"):
            expression += r"\s*$"
        pattern = re.compile(expression, re.I)
        for page in pages:
            for region in page.regions:
                if not pattern.search(region.text):
                    continue
                # A newly sampled phone can coincidentally contain an old short
                # address number. Only the exact generated span is exempt, and
                # only when that owner's original text did not contain the value.
                generated_intervals = []
                for owner in record.get("proof", []):
                    surface = owner.get("after", "")
                    spans = owner.get("spans", [])
                    if not surface or not spans or {s["page"] for s in spans} != {page.index}:
                        continue
                    if pattern.search(" ".join(s["expected"] for s in spans)):
                        continue
                    if region.text.count(surface) != 1:
                        continue
                    for span in spans:
                        native = source["pages"][span["page"]]["regions"][span["region"]]["text"]
                        if native[span["start"] : span["end"]] != span["expected"]:
                            raise ValueError(
                                "generated private-collision proof differs from source"
                            )
                    start = region.text.index(surface)
                    generated_intervals.append((start, start + len(surface)))
                if all(
                    any(lo <= match.start() and match.end() <= hi for lo, hi in generated_intervals)
                    for match in pattern.finditer(region.text)
                ):
                    continue
                witnesses = [
                    n
                    for n, item in enumerate(allowed)
                    if item["planIndex"] == ix
                    and item["page"] == page.index
                    and item["text"] == region.text
                    and item["reason"]
                ]
                if len(witnesses) != 1:
                    raise ValueError(f"private-plan lexeme survives without role review: {ix}")
                used.add(witnesses[0])
    if used != set(range(len(allowed))):
        raise ValueError("retained lexical witness is stale or unused")
