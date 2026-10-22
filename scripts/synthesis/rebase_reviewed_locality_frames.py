"""Propose whole-frame ownership for an explicitly complete-read source batch.

This read-only preparation command emits an apply_patch payload. It never
infers a missing location, changes source input or silently accepts an unpaired
owner. Runtime already owns homonym projection; this only supplies its frames.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

from document_ocr.synthesis.independent_geography import _is_locality_country_path
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign


def frame_proposals(blueprint, approved_gaps=None):
    approved_gaps = approved_gaps or {}
    grouped = defaultdict(list)
    raw = blueprint.source.encode()
    regions = blueprint.regions
    for country in regions:
        if len(country.target_paths) != 1 or country.curated_key is not None:
            continue
        path = country.target_paths[0]
        if not _is_locality_country_path(path):
            continue
        name_path = path.removesuffix("country") + "name"
        matches = []
        for name in regions:
            if name.target_paths != (name_path,) or name.curated_key is not None:
                continue
            if not name.end < country.start:
                continue
            gap = raw[name.end : country.start].decode()
            if re.fullmatch(r"[ \t,;()/:-]+", gap):
                matches.append((name, country, gap, gap))
            elif gap in approved_gaps.get(path, {}):
                matches.append((name, country, gap, approved_gaps[path][gap]))
        if len(matches) > 1:
            raise ValueError("ambiguous complete locality frame: " + path)
        if matches:
            grouped[path].append(matches[0])
    proposals = []
    for path, pairs in grouped.items():
        keys = {r.key for pair in pairs for r in pair[:2]}
        owned = {(r.key, r.occurrence) for r in regions if r.key in keys}
        paired = {(r.key, r.occurrence) for pair in pairs for r in pair[:2]}
        if owned != paired:
            raise ValueError(
                "frame rebase requires complete consistent repeated ownership: " + path
            )
        styles = defaultdict(list)
        for pair in pairs:
            # A homonym projection may remove the entire country presentation.
            # Its closing delimiter must not remain as an unowned dangling byte.
            country = pair[1]
            gap = pair[3]
            suffix = (
                ")"
                if gap.count("(") > gap.count(")") and raw[country.end : country.end + 1] == b")"
                else ""
            )
            styles[(gap, suffix)].append(pair)
        for index, ((render_gap, suffix), group) in enumerate(styles.items()):
            occurrences = []
            for name, country, _, _ in group:
                quote = raw[name.start : country.end + len(suffix)].decode()
                starts = [m.start() for m in re.finditer(re.escape(quote.encode()), raw)]
                occurrences.append({"text": quote, "occurrence": starts.index(name.start) + 1})
            proposals.append(
                {
                    "path": path,
                    "oldKeys": sorted(keys),
                    "key": "reviewed:locality_frame:"
                    + path.removeprefix("documentPatch.")
                    + f":{index}",
                    "paths": [path.removesuffix("country") + "name", path],
                    "occurrences": occurrences,
                    "expression": "{"
                    + path.removesuffix("country")
                    + "name}"
                    + render_gap
                    + "{"
                    + path
                    + "}"
                    + suffix,
                }
            )
    return proposals


def rebase_owner_frames(declarations, proposals):
    """Replace source-added owners, but retire only historical bindings.

    The compiler retires historical keys before applying ``add`` declarations.
    An added owner therefore must be removed, not listed as a historical retire.
    """
    value = copy.deepcopy(declarations)
    old_keys = {key for proposal in proposals for key in proposal["oldKeys"]}
    added_keys = {row["key"] for row in value.get("add", [])}
    if "add" in value:
        value["add"] = [row for row in value["add"] if row["key"] not in old_keys]
    for key in old_keys:
        if key not in added_keys:
            value.setdefault("retire", {})[key] = (
                "Complete reviewed inline locality/country frame owns both scalars and punctuation."
            )
        for mapping in ("bindings", "surfaces"):
            value.get(mapping, {}).pop(key, None)
    for proposal in proposals:
        value.setdefault("add", []).append(
            {
                "key": proposal["key"],
                "kind": "location",
                "paths": proposal["paths"],
                "occurrences": proposal["occurrences"],
            }
        )
        value.setdefault("surfaces", {})[proposal["key"]] = {"expression": proposal["expression"]}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--approved-gaps", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    review = json.loads(args.review.read_bytes())
    reviewed = {
        r["prefix"] for r in review["sources"] if r.get("completeRead") and r.get("decision")
    }
    if reviewed != {sid[4:12] for sid in config["source_ids"]}:
        raise ValueError("every selected source needs complete-read review")
    approved = {} if args.approved_gaps is None else json.loads(args.approved_gaps.read_bytes())
    campaign = _SourceOwnershipCampaign(Path.cwd(), config)
    path = Path(config["ownership"])
    content = path.read_text()
    authority = yaml.safe_load(content)
    patches, summary = [], []
    for sid in config["source_ids"]:
        proposals = frame_proposals(campaign.blueprint(sid), approved.get(sid[4:12], {}))
        if not proposals:
            continue
        value = rebase_owner_frames(authority["sources"][sid], proposals)
        pattern = r"^  " + re.escape(sid) + r":\n.*?(?=^  doc_|\Z)"
        match = re.search(pattern, content, re.M | re.S)
        if match is None:
            raise ValueError("source ownership block missing")
        before = match[0]
        after = (
            "\n".join(
                "  " + line if line else ""
                for line in yaml.safe_dump(
                    {sid: value}, allow_unicode=True, sort_keys=False, width=100
                ).splitlines()
            )
            + "\n"
        )
        patches.append(
            (
                match.start(),
                "@@\n"
                + "".join("-" + line + "\n" for line in before.splitlines())
                + "".join("+" + line + "\n" for line in after.splitlines()),
            )
        )
        summary.append({"source": sid, "frames": proposals})
    print(
        json.dumps(
            {
                "summary": summary,
                "patch": "*** Begin Patch\n*** Update File: "
                + str(path.resolve())
                + "\n"
                + "".join(patch for _, patch in sorted(patches))
                + "*** End Patch",
            }
        )
    )


if __name__ == "__main__":
    main()
