"""Make the reviewed literal instruction projections use canonical target casing."""

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = ART / "semantic-revision-07/catalog.json"
    catalog = read(base)
    for sid, entry in catalog["templates"].items():
        if sid[4:12] not in {"13280227", "d684b168"}:
            continue
        template = read(verify_pin(entry["template"]))
        actions = [
            a
            for a in template["overlay"]["targetFromOwners"]
            if a["target"] == "/goodsItemDetails/0/handlingInstructions/1"
        ]
        if len(actions) != 1:
            raise ValueError("reviewed handling owner changed")
        actions[0]["targetCase"] = "upper"
        path = ART / "semantic-revision-08" / (sid + ".json")
        atomic_publish_json(path, template)
        entry["template"] = pin(path)
    output = ART / "semantic-revision-08/catalog.json"
    atomic_publish_json(output, catalog)
    atomic_publish_json(
        OUT / "semantic-revision-08.json",
        {
            "baseCatalog": pin(base),
            "candidateCatalog": pin(output),
            "review": "Canonical uppercase applies only to the two explicitly reviewed handling projections; rendered source text is unchanged.",
        },
    )


if __name__ == "__main__":
    main()
