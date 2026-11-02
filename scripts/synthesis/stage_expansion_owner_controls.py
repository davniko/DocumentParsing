"""Stage fresh synthesis for the reviewed source-only caption/identifier owners."""

import argparse
import json
from pathlib import Path

from expand_r28_2500 import selected_config
from repair_expansion_source_owners import (
    BRAND, COUNTRY, IMPORTER, IMPORTER_CAPTIONS, IMPORTER_SECOND, LOCALITY_FRAME,
    LOCALITY_REPETITIONS, MARKS, OUT, PHONE,
)

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, choices=[1, 2, 3], nargs="+", default=[1, 2, 3])
    parser.add_argument("--sources", nargs="+", choices=["country", "phone", "importer", "marks", "brand", "importer_second", "locality_frame", *IMPORTER_CAPTIONS, *LOCALITY_REPETITIONS, "9b35d0b6", "600a1e89", "48973902", "74860ac9"], required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("control name must be a directory basename")
    ids = {"country": COUNTRY, "phone": PHONE, "importer": IMPORTER, "marks": MARKS, "brand": BRAND,
           "importer_second": IMPORTER_SECOND, "locality_frame": LOCALITY_FRAME}
    base = json.loads((OUT / "complete-source-authority-config.json").read_text())
    for prefix in [*IMPORTER_CAPTIONS, *LOCALITY_REPETITIONS, "9b35d0b6", "600a1e89", "48973902", "74860ac9"]:
        matches = [s for s in base["source_ids"] if s.startswith("doc_" + prefix)]
        if len(matches) != 1:
            raise ValueError("reviewed importer source not unique in base registry")
        ids[prefix] = matches[0]
    path = (
        ROOT
        / "artifacts/kie-synthesis-production/r28-expansion2500-20261101" / args.name / "config.json"
    )
    value = selected_config(
        base,
        dict.fromkeys([ids[name] for name in args.sources], 3),
        path.parent / "campaign",
        args.seed,
        group_counts=tuple(args.groups),
    )
    value["ownership"] = str((OUT / "ownership.json").relative_to(ROOT))
    value["auxiliary"] = str((OUT / "auxiliary.json").relative_to(ROOT))
    geography = OUT / "geography_certificates.json"
    value["independent"]["geography_certificates"] = {
        "path": str(geography.relative_to(ROOT)), "sha256": sha256_file(geography),
    }
    atomic_publish_json(path, value)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
