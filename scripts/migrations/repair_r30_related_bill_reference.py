"""Preserve the PDF-confirmed shared numeric body of the tank bill's footer ID."""

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    sid, entry = next((s, e) for s, e in catalog["templates"].items() if "0dbc75a0" in s)
    template = read(verify_pin(entry["template"]))
    source = read(verify_pin(template["source"]))
    assert source["pages"][0]["regions"][7]["text"] == "HCLE02300719341"
    assert source["pages"][0]["regions"][95]["text"] == "S02300719341"
    key = "agent:source:reference_s02300719341"
    template["overlay"]["overrides"].append(
        {
            "key": key,
            "byteStart": 1586,
            "regions": [[0, 95]],
            "valueRecipe": {
                "parts": [
                    {"fromKey": key, "segments": [[0, 1]], "join": ""},
                    {"fromKey": "v000", "segments": [[4, None]], "join": ""},
                ]
            },
        }
    )
    folder = ART / "readback-revision-18"
    atomic_publish_json(folder / (sid + ".json"), template)
    entry["template"] = pin(folder / (sid + ".json"))
    atomic_publish_json(folder / "catalog.json", catalog)
    atomic_publish_json(
        OUT / "readback-review-18.json",
        {
            "baseCatalog": base,
            "catalog": pin(folder / "catalog.json"),
            "sources": {
                "0dbc75a0": {
                    "targetPaths": [],
                    "inputChanges": True,
                    "reason": "PDF header HCLE02300719341 and footer S02300719341 share their "
                    "numeric body. Keep the already sampled one-letter footer prefix and "
                    "project the sampled header body rather than drawing unrelated digits. "
                    "Anonymized real input already preserves this relationship.",
                }
            },
        },
    )


if __name__ == "__main__":
    main()
