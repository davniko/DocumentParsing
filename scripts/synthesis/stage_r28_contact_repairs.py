"""Stage the explicitly authored contact-owner corrections; never publish them."""

import json
import re
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_reflow_policy import FontMetrics
from document_ocr.synthesis.package_alias_refresh import apply_reviewed_cleanup
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task


def run():
    root = Path.cwd()
    data = root / "data/curated/mpci-bl-r28-positions-v1"
    out = root / "docs/analysis/r28-multirow-refresh1500-20261031"
    config = json.loads((root / "configs/synthesis/mpci_bl_curated_v7_multirow.json").read_text())
    font = config["positions"]["reflow"]
    metrics = FontMetrics(Path(font["font_path"]), font["font_sha256"])
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        (data / "task-constraints.json").read_bytes()
    )
    task = get_training_task(constraints.task).bind_constraints(constraints)
    manual = json.loads((out / "reviewed-contact-mailboxes.json").read_text())
    rows = {}
    for filename in ("synthetic-train.jsonl", "previous-uae-synthetic.jsonl"):
        for row in map(json.loads, (data / filename).open()):
            if row["documentId"].removeprefix("syn_full_v7_") in manual:
                existing = rows.setdefault(row["documentId"], row)
                if existing != row:
                    raise ValueError("existing sidecar differs from main record")
    repairs, failures = [], []
    for short, roles in manual.items():
        row = rows["syn_full_v7_" + short]
        leaves, raw = flat(row["target"]), row["joinedRawText"]
        replacements, target_edits, details = {}, [], []
        for role, local_parts in roles.items():
            base = f"parties.{role}.contactDetails"
            names = leaves[base + ".contactName"].split(";")
            if len(names) != len(local_parts):
                raise ValueError("reviewed contact cardinality differs")
            before = [leaves[f"{base}.emailAddresses[{i}]"] for i in range(len(names))]
            printed = []
            for address in before:
                matches = list(re.finditer(re.escape(address), raw, re.I))
                if not matches or len({m[0] for m in matches}) != 1:
                    raise ValueError("contact missing or inconsistent repeated casing")
                printed.append((matches[0].start(), matches[0][0]))
            printed.sort()
            after = []
            for name, local, (_, old) in zip(names, local_parts, printed, strict=True):
                old_local, domain = old.split("@")
                new = (local.upper() if old_local.isupper() else local) + "@" + domain
                after.append(new)
                if metrics.length(new) > metrics.length(old) + 1e-7:
                    failures.append([short, name, old, new])
                if new != old:
                    if old in replacements:
                        raise ValueError("one mailbox assigned to two distinct owners")
                    replacements[old] = new
                details.append(
                    dict(
                        name=name.strip(),
                        beforePrinted=old,
                        after=new,
                        beforeAdvance=metrics.length(old),
                        afterAdvance=metrics.length(new),
                    )
                )
            for index, (old, new) in enumerate(zip(before, after, strict=True)):
                if new != old:
                    target_edits.append(
                        dict(path=f"{base}.emailAddresses[{index}]", before=old, after=new)
                    )
        # Every repeated occurrence is explicitly identified in the immutable
        # current text. Replacements are simultaneous, never cascaded.
        raw_edits = []
        for old, new in replacements.items():
            for occurrence in range(1, raw.count(old) + 1):
                raw_edits.append(dict(before=old, after=new, occurrence=occurrence))
        repair = dict(
            documentId=row["documentId"],
            beforeRecordSha256=digest(row),
            rawEdits=raw_edits,
            targetEdits=target_edits,
            reviewedContacts=details,
            reason=(
                "Explicitly reviewed personal name/email ownership; preserve domains, casing, "
                "line boundaries, unrelated facts and existing coordinate regions. Every new "
                "email fits its previous measured footprint."
            ),
        )
        updated, _ = apply_reviewed_cleanup(row, repair)
        if task.canonicalize(updated["target"]) != updated["target"]:
            raise ValueError("contact repair is not canonical")
        if len(updated["joinedRawText"].splitlines()) != len(raw.splitlines()):
            raise ValueError("contact repair changed line boundaries")
        repairs.append(repair)
    print(json.dumps(dict(repairs=len(repairs), footprintFailures=failures)), flush=True)
    if failures:
        raise ValueError("explicit mailbox choices require narrower reviewed alternatives")
    atomic_publish_json(out / "contact-repairs.json", repairs)


if __name__ == "__main__":
    run()
