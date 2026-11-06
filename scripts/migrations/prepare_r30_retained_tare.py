"""Reuse existing sampled tare authority, including an unchanged source value."""

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    catalog = read(ART / "binding-revision-10/catalog.json")
    sid = next(s for s in catalog["templates"] if s[4:12] == "89227fc3")
    template = read(verify_pin(catalog["templates"][sid]["template"]))
    extra = next(x for x in template["overlay"]["extras"] if x["owner"] == "new:row-61")
    extra.pop("equipmentTare")
    extra["receiptEquipmentTare"] = {"index": 0, "minPlaces": 0}
    path = ART / "binding-revision-11" / (sid + ".json")
    atomic_publish_json(path, template)
    catalog["templates"][sid]["template"] = pin(path)
    path = ART / "binding-revision-11/catalog.json"
    atomic_publish_json(path, catalog)
    review = read(OUT / "binding-review-10.json")
    review["catalog"] = pin(path)
    review["sources"]["89227fc3"]["reason"] = (
        "Native TARE column belongs to equipment row 0. Both retained and fresh receipts "
        "already contain physical.tares with equipment pair and observation provenance. "
        "Use that authority directly even when the old renderer omitted an unchanged edit; "
        "do not resample an existing tare or treat it as cargo packing mass."
    )
    atomic_publish_json(OUT / "binding-review-11.json", review)


if __name__ == "__main__":
    main()
