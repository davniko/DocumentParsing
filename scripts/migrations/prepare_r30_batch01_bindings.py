"""Source-reviewed split/composite occurrence coverage for retained shipments."""

import copy
import json

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    catalog_pin = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(catalog_pin))
    # Exact byte owners and native spans reviewed against both source contracts.
    additions = {
        "fbfcab7a": [("v007", 2500, [0, 58, 28, 29]), ("v007", 2882, [0, 70, 23, 24])],
        "531130c4": [("v008", 1359, [0, 43, 28, 29]), ("v008", 1707, [0, 51, 36, 37])],
        "9184af53": [
            (f"allocation_{i // 3}", byte, span)
            for i, (byte, span) in enumerate(
                [
                    (1181, [0, 42, 28, 32]),
                    (1206, [0, 43, 0, 4]),
                    (1329, [0, 45, 37, 41]),
                    (1489, [0, 52, 28, 32]),
                    (1514, [0, 53, 0, 4]),
                    (1637, [0, 55, 37, 41]),
                    (1816, [0, 63, 28, 32]),
                    (4759, [1, 3, 0, 4]),
                    (5433, [1, 17, 37, 41]),
                ]
            )
        ],
        "d684b168": [
            ("v008", 1015, [0, 44, 10, 12]),
            ("v008", 1053, [0, 49, 10, 12]),
            ("v008", 1067, [0, 50, 0, 2]),
        ],
    }
    changed = []
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in additions and prefix != "fe1eea92":
            continue
        template = copy.deepcopy(read(verify_pin(entry["template"])))
        source = read(verify_pin(template["source"]))
        raw = verify_pin(entry["samplingSource"]).read_bytes()
        for key, byte, span in additions.get(prefix, []):
            p, r, lo, hi = span
            text = source["pages"][p]["regions"][r]["text"][lo:hi]
            if raw[byte : byte + len(text.encode())].decode() != text:
                raise ValueError("reviewed split occurrence differs from source")
            template["overlay"]["overrides"].append(
                {"key": key, "byteStart": byte, "spans": [span]}
            )
        if prefix == "fe1eea92":
            extra = next(
                x for x in template["overlay"]["extras"] if x["owner"] == "first-description-count"
            )
            if extra["allocation"] != {"index": 0}:
                raise ValueError("allocation authority already changed")
            extra["allocation"]["authority"] = "public_target"
        path = ART / "binding-revision-01" / (sid + ".json")
        atomic_publish_json(path, template)
        entry["template"] = pin(path)
        changed.append(sid)
    destination = ART / "binding-revision-01/catalog.json"
    atomic_publish_json(destination, catalog)
    atomic_publish_json(
        OUT / "binding-revision-01.json",
        {
            "status": "SOURCE_REVIEWED_REQUIRES_CONTROLS_AND_BATCH_REPLAY",
            "baseCatalog": catalog_pin,
            "candidateCatalog": pin(destination),
            "sources": changed,
            "reason": "Exact numeric subowners of already reviewed compound package occurrences; existing public allocation reused explicitly, without inventing an independent plan.",
        },
    )
    print(json.dumps({"sources": len(changed), "catalog": pin(destination)}, indent=2))


if __name__ == "__main__":
    main()
