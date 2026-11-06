"""Disambiguate the quantity in the reviewed 551657f3 container summary row."""

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = ART / "binding-revision-04/catalog.json"
    catalog = read(base)
    sid = next(s for s in catalog["templates"] if s[4:12] == "551657f3")
    entry = catalog["templates"][sid]
    template = read(verify_pin(entry["template"]))
    source = read(verify_pin(template["source"]))
    text = source["pages"][0]["regions"][53]["text"]
    if text[69:70] != "4" or text[71:78] != "PALLETS":
        raise ValueError("reviewed quantity/type occurrence changed")
    template["overlay"]["overrides"].append(
        {"key": "shipment_packages", "byteStart": 903, "spans": [[0, 53, 69, 70]]}
    )
    path = ART / "binding-revision-05" / (sid + ".json")
    atomic_publish_json(path, template)
    entry["template"] = pin(path)
    output = ART / "binding-revision-05/catalog.json"
    atomic_publish_json(output, catalog)
    atomic_publish_json(
        OUT / "binding-revision-05.json",
        {
            "baseCatalog": pin(base),
            "candidateCatalog": pin(output),
            "reason": "The 4 immediately before PALLETS in the equipment summary, not the unrelated digits elsewhere. All other quantity occurrences retain their already reviewed bindings.",
        },
    )


if __name__ == "__main__":
    main()
