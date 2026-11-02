"""Execute explicitly authored per-component review decisions, never infer them."""

import argparse
import json
import subprocess
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--from-component")
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    if args.from_component and args.from_component not in bundle:
        raise ValueError("explicit resume component absent from authored bundle")
    started = args.from_component is None
    for component, decisions in bundle.items():
        started = started or component == args.from_component
        if not started:
            continue
        config = sorted((RUN / component / "campaign/declared-package-preparation").glob("*/prepared-config.json"))[-1]
        pieces = component.split("/")
        label = f"r{pieces[-1].split('-')[1]}-b{pieces[0].split('-')[1]}"
        if len(pieces) == 3:
            label += "-" + pieces[1]
        notes = dict(decisions)
        notes["rejectedScopedFindings"] = {}
        for key, decision in notes.pop("scopedDecisions", {}).items():
            sample, index = key.rsplit(":", 1)
            candidate = RUN / component / "campaign/candidates" / (sample + ".json")
            scope = RUN / "scoped-cargo-review/reviews" / (sample + "-" + sha256_file(candidate) + ".json")
            findings = json.loads(scope.read_text())["review"]["findings"]
            notes["rejectedScopedFindings"][key] = {**decision, "finding": findings[int(index)]}
        path = OUT / "redraw-reviews" / f"{label}-{args.name}-notes.json"
        atomic_publish_json(path, notes)
        pinned = path.with_name(path.stem.removesuffix("-notes") + ".json")
        subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/synthesis/pin_expansion_review_notes.py",
                        "--config", str(config), "--notes", str(path), "--output", str(pinned)], check=True)
        log = path.with_name(path.stem + "-certification.log")
        with log.open("xb") as stream:
            subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/synthesis/certify_r28_expansion_component.py",
                            "--config", str(config), "--review", str(pinned), "--name", label],
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        print(label, "CERTIFIED", flush=True)


if __name__ == "__main__":
    main()
