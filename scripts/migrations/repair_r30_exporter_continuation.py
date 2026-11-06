"""Own the Paddle-only exporter-country continuation across the page boundary."""

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = ART / "semantic-revision-08/catalog.json"
    catalog = read(base)
    sid = next(s for s in catalog["templates"] if s[4:12] == "f766e12f")
    entry = catalog["templates"][sid]
    template = read(verify_pin(entry["template"]))
    source = read(verify_pin(template["source"]))
    if (
        source["pages"][0]["regions"][64]["text"] != "Foreign Exporter Country"
        or source["pages"][1]["regions"][3]["text"] != "Code: US"
    ):
        raise ValueError("reviewed exporter country continuation changed")
    template["overlay"]["extras"].append(
        {
            "owner": "audit:exporter-country-code",
            "spans": [[1, 3, 6, 8]],
            "scenarioText": "/origin/country_code",
        }
    )
    path = ART / "semantic-revision-09" / (sid + ".json")
    atomic_publish_json(path, template)
    entry["template"] = pin(path)
    output = ART / "semantic-revision-09/catalog.json"
    atomic_publish_json(output, catalog)
    review = read(OUT / "semantic-review-08.json")
    review["catalog"] = pin(output)
    review["sources"]["f766e12f"] = {
        "targetPaths": [],
        "inputChanges": True,
        "reason": "Source page 0 ends the goods block with Foreign Exporter Country; page 1 continues Code: US. Sample that source-only country from export context; it is not a notify-party country or an additional target.",
    }
    atomic_publish_json(OUT / "semantic-review-09.json", review)


if __name__ == "__main__":
    main()
