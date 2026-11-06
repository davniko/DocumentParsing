"""Use semantic shipment authorities for reviewed Paddle repetitions.

Older retained receipts and new synthesis may own a source declaration as one
compound span or as separate count/type spans. Both exact topologies are reviewed;
neither permits an unowned native occurrence or a guessed substring match.
"""

import copy
import json

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    templates = {
        sid: read(verify_pin(entry["template"])) for sid, entry in catalog["templates"].items()
    }
    tare_support = {}
    for template in templates.values():
        for extra in template["overlay"].get("extras", []):
            if "equipmentTare" in extra:
                for pair, rows in extra["equipmentTare"]["support"]["pairs"].items():
                    bucket = tare_support.setdefault(pair, [])
                    for row in rows:
                        if row not in bucket:
                            bucket.append(row)
    if not tare_support:
        raise ValueError("no reviewed empirical equipment tare support")
    spans = {
        "53cf4d99": ("v008", 1576, [0, 56, 36, 37], "9"),
        "54aadfb2": ("v009", 603, [0, 47, 0, 1], "4"),
        "1149521e": ("dg_0_class", 4273, [1, 70, 25, 26], "3"),
    }
    declarations = {
        "94709757": "paddle:packages-repeat",
        "73fdbd15": "new:main_packages",
    }
    changed = {}
    for sid, original in templates.items():
        short = sid[4:12]
        if short not in {*spans, *declarations, "9d4a6b90", "89227fc3", "ee74bdff"}:
            continue
        template = copy.deepcopy(original)
        overlay = template["overlay"]
        source = read(verify_pin(template["source"]))
        if short in spans:
            key, byte, span, expected = spans[short]
            p, r, lo, hi = span
            raw = verify_pin(catalog["templates"][sid]["samplingSource"]).read_bytes()
            if source["pages"][p]["regions"][r]["text"][lo:hi] != expected:
                raise ValueError("native occurrence changed")
            if raw[byte : byte + len(expected)].decode() != expected:
                raise ValueError("sampling occurrence changed")
            overlay["overrides"].append({"key": key, "byteStart": byte, "spans": [span]})
        if short in declarations:
            extra = next(x for x in overlay["extras"] if x["owner"] == declarations[short])
            if extra.pop("fromKey") != "declared-packages:0":
                raise ValueError("package repetition owner changed")
            extra["packageRows"] = {"goods": 0, "style": "full"}
        if short == "73fdbd15":
            extra = next(x for x in overlay["extras"] if x["owner"] == "new:main_equipment")
            if source["pages"][0]["regions"][53]["text"] != "1 x 20ST":
                raise ValueError("equipment summary occurrence changed")
            extra.pop("fromKey")
            extra["equipmentSummary"] = {"authority": "public_target", "source": "20ST"}
        if short == "9d4a6b90":
            extra = next(x for x in overlay["extras"] if x["owner"] == "paddle:quantity_column")
            if extra["spans"] != [[0, 73, 0, 2], [0, 78, 0, 7]]:
                raise ValueError("quantity/type column changed")
            overlay["extras"].remove(extra)
            for part, span in zip(("quantity", "type"), extra["spans"], strict=True):
                overlay["extras"].append(
                    {
                        "owner": "paddle:column-" + part,
                        "spans": [span],
                        "packageSummary": {"goods": 0, "part": part, "style": "full"},
                    }
                )
        if short == "89227fc3":
            if source["pages"][0]["regions"][51]["text"] != "TARE":
                raise ValueError("tare column caption changed")
            extra = next(x for x in overlay["extras"] if x["owner"] == "new:row-61")
            extra.pop("fromKey")
            extra["equipmentTare"] = {
                "index": 0,
                "authority": "public_target",
                "minPlaces": 0,
                "support": {"pairs": tare_support},
            }
        if short == "ee74bdff":
            (old,) = overlay["absentEdits"]
            if old["key"] != "declared-packages:0" or old["before"] != "10 PALLETS":
                raise ValueError("package disposition changed")
            overlay["absentEdits"] = [
                {
                    "reason": old["reason"],
                    "alternatives": [
                        [{k: old[k] for k in ("key", "byteStart", "before")}],
                        [
                            {"key": "v008", "byteStart": 1003, "before": "10"},
                            {
                                "key": "anchor:documentPatch.cargoPackages[0].typeCategory",
                                "byteStart": 1006,
                                "before": "PALLETS",
                            },
                        ],
                    ],
                }
            ]
        path = ART / "binding-revision-10" / (sid + ".json")
        atomic_publish_json(path, template)
        catalog["templates"][sid]["template"] = pin(path)
        changed[short] = {
            "targetPaths": [],
            "inputChanges": True,
            "reason": "Reviewed native repetition now uses complete semantic shipment authority; "
            "exact alternate historical source-owner topology is explicit. No target expansion.",
        }
    destination = ART / "binding-revision-10/catalog.json"
    atomic_publish_json(destination, catalog)
    atomic_publish_json(
        OUT / "binding-review-10.json",
        {
            "baseCatalog": base,
            "catalog": pin(destination),
            "sources": changed,
        },
    )
    print(json.dumps({"catalog": pin(destination), "sources": list(changed)}))


if __name__ == "__main__":
    main()
