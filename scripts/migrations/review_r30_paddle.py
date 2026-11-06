"""Print bounded native-region review packets; never decide or publish labels."""

from __future__ import annotations

import argparse
import json

from prepare_r30_paddle_pilot import ART, read


def differences(old, new, path=""):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            child = path + "/" + key
            if key not in old:
                yield child, "ABSENT", new[key]
            elif key not in new:
                yield child, old[key], "ABSENT"
            else:
                yield from differences(old[key], new[key], child)
    elif old != new:
        yield path, old, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixes", nargs="+")
    parser.add_argument("--no-regions", action="store_true")
    args = parser.parse_args()
    for prefix in args.prefixes:
        matches = list((ART / "sources").glob("doc_" + prefix + "*.json"))
        if len(matches) != 1:
            raise ValueError(f"source not unique: {prefix}")
        record = read(matches[0])
        proposed_path = ART / "target-reviews" / matches[0].name
        print("\nDOCUMENT", prefix, record["split"], "PDF", record["source"]["pdf"])
        print("CURRENT", json.dumps(record["current"]["target"], ensure_ascii=False))
        if proposed_path.exists():
            proposal = read(proposed_path)
            for path, old, new in differences(
                proposal["lookupTarget"], proposal["candidateTarget"]
            ):
                print("PROPOSED", path, json.dumps([old, new], ensure_ascii=False))
            print("UNRESOLVED", proposal.get("proposal", {}).get("unresolved"))
        if not args.no_regions:
            for page in record["pages"]:
                for region in page["regions"]:
                    print(f"{page['index']}:{region['index']} {region['text']}")


if __name__ == "__main__":
    main()
