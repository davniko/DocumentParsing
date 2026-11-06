"""Stage exact, reviewed template edits; publication still requires fresh controls."""

import argparse
import copy

from migrate_r30_synthetic import ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.paddle_templates import validate_override_selectors


def apply_edit(template, edit):
    parts = edit["path"].split("/")
    if parts[0] or len(parts) < 2:
        raise ValueError("template edits require an absolute JSON pointer")
    owner = template
    for part in parts[1:-1]:
        key = part.replace("~1", "/").replace("~0", "~")
        owner = owner[int(key)] if isinstance(owner, list) else owner[key]
    key = parts[-1].replace("~1", "/").replace("~0", "~")
    if isinstance(owner, list):
        key = int(key)
    action = edit["action"]
    if action == "add":
        if not isinstance(owner, dict) or key in owner:
            raise ValueError("add requires a genuinely absent object member")
        owner[key] = copy.deepcopy(edit["after"])
    elif action == "replace":
        if owner[key] != edit["before"]:
            raise ValueError(f"reviewed template member changed: {edit['path']}")
        owner[key] = copy.deepcopy(edit["after"])
    elif action == "append":
        if not isinstance(owner[key], list) or not isinstance(edit["after"], list):
            raise ValueError("append requires explicit list values")
        if len(owner[key]) != edit["beforeLength"]:
            raise ValueError("reviewed append position changed")
        owner[key].extend(copy.deepcopy(edit["after"]))
    else:
        raise ValueError("unsupported template edit action")


def main(review_path, destination):
    review = read(review_path)
    if review["status"] != "MAIN_AGENT_REVIEWED_TEMPLATE_CORRECTIONS":
        raise ValueError("explicit reviewed correction authority required")
    catalog = read(verify_pin(review["baseCatalog"]))
    changed = set()
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in review["sources"]:
            continue
        if prefix in changed:
            raise ValueError("source prefix is ambiguous")
        template = read(verify_pin(entry["template"]))
        before = copy.deepcopy(template)
        for edit in review["sources"][prefix]["edits"]:
            apply_edit(template, edit)
        if template == before:
            raise ValueError("reviewed correction made no change")
        validate_override_selectors(template["overlay"])
        path = destination / (sid + ".json")
        atomic_publish_json(path, template)
        entry["template"] = pin(path)
        changed.add(prefix)
    if changed != set(review["sources"]):
        raise ValueError("reviewed sources do not match catalog membership")
    atomic_publish_json(destination / "catalog.json", catalog)
    atomic_publish_json(
        destination / "review.json",
        {
            **review,
            "catalog": pin(destination / "catalog.json"),
            "reviewAuthority": pin(review_path),
        },
    )
    print(destination / "catalog.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review")
    parser.add_argument("destination")
    args = parser.parse_args()
    main(ROOT / args.review, ROOT / args.destination)
