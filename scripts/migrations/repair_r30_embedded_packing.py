"""Retain the reviewed per-unit packing phrase within a product description."""

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    sid, entry = next((s, e) for s, e in catalog["templates"].items() if "819af5e4" in s)
    template = read(verify_pin(entry["template"]))
    source = read(verify_pin(template["source"]))
    assert source["pages"][1]["regions"][17]["text"] == "1800 CARTON(S) X 13.61 KGS OF IQF WILD"
    assert source["pages"][1]["regions"][19]["text"] == "BLUEBERRIES FROZEN"
    template["overlay"].setdefault("targetFromSourceSpans", []).append(
        {
            "regions": [[1, 17], [1, 19]],
            "targetCase": "upper",
            "target": "/goodsItemDetails/0/description",
        }
    )
    folder = ART / "readback-revision-14"
    atomic_publish_json(folder / (sid + ".json"), template)
    entry["template"] = pin(folder / (sid + ".json"))
    atomic_publish_json(folder / "catalog.json", catalog)
    atomic_publish_json(
        OUT / "readback-review-14.json",
        {
            "baseCatalog": base,
            "catalog": pin(folder / "catalog.json"),
            "sources": {
                "819af5e4": {
                    "targetPaths": ["/goodsItemDetails/0/description"],
                    "inputChanges": False,
                    "reason": "Per-unit X ... KGS OF is an embedded packing/weight phrase, "
                    "analogous "
                    "to the policy's 80 DRUMS OF 210 KGS NET example, not a bare count heading. "
                    "Retain both product regions; exclude interleaved gross-weight/seal cells.",
                }
            },
        },
    )


if __name__ == "__main__":
    main()
