"""Inventory country-bearing product passages; proposals are not label authority."""

import json
import re
import resource
import time

from migrate_r30_synthetic import DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_write_json


def main():
    started = time.perf_counter()
    manifest = read(DATA / "manifest.json")
    names = read(
        ROOT
        / "artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1"
        / "preferred-names.json"
    )["names"]
    terms = set(names.values()) | {
        "UKRAINIAN",
        "INDIAN",
        "ARGENTINIAN",
        "ARGENTINE",
        "FRENCH",
        "SWEDISH",
        "AUSTRALIAN",
        "BRAZILIAN",
        "CHINESE",
        "EGYPTIAN",
        "TURKISH",
        "RUSSIAN",
        "INDONESIAN",
        "KENYAN",
        "GERMAN",
        "ITALIAN",
        "SPANISH",
        "AMERICAN",
        "JAPANESE",
        "KOREAN",
        "TAIWANESE",
        "CANADIAN",
        "MALAYSIAN",
        "USA",
        "U.S.A.",
    }
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(t) for t in sorted(terms, key=lambda t: (-len(t), t)))
        + r")(?!\w)",
        re.I,
    )
    paths = {
        **manifest["aggregateRecords"],
        "synthetic-r29-baseline": read(OUT / "receipt-inventory.json")["baseline"],
    }
    rows, counts = [], {}
    for split, source in paths.items():
        count = with_origin = 0
        for line in verify_pin(source).open():
            row = json.loads(line)
            count += 1
            for index, goods in enumerate(row["target"].get("goodsItemDetails", [])):
                origin = goods.get("origin")
                with_origin += bool(origin)
                description = goods.get("description", "")
                matches = list(dict.fromkeys(m.group(0) for m in pattern.finditer(description)))
                if not matches:
                    continue
                rows.append(
                    {
                        "split": split,
                        "id": row["documentId"],
                        "source": row.get("sourceDocumentId"),
                        "goodsIndex": index,
                        "description": description,
                        "origin": origin,
                        "countryTerms": matches,
                    }
                )
        counts[split] = {"documents": count, "goodsRowsWithOrigin": with_origin}
    result = {
        "status": "SCREEN_REQUIRES_SEMANTIC_REVIEW_NOT_AUTOMATIC_ORIGIN_LABELS",
        "inputs": paths,
        "manifest": pin(DATA / "manifest.json"),
        "counts": counts,
        "rows": rows,
        "scope": "Whole-name lexical screen plus explicit demonym controls. Country mention alone "
        "does not establish its role; this is not an exhaustive alias or semantic-origin detector.",
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "goods-origin-country-screen.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print("screened passages", len(rows))


if __name__ == "__main__":
    main()
