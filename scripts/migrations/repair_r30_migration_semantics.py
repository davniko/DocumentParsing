"""Apply explicitly reviewed role-boundary corrections to migration authorities."""

from migrate_r30_synthetic import ART, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.country_presentations import DEFAULT_REGISTRY_MANIFEST


def remove_projection(overlay, collection, target, expected=1):
    entries = overlay.get(collection, [])
    selected = [e for e in entries if e["target"].startswith(target)]
    if len(selected) != expected:
        raise ValueError(f"reviewed projection changed: {collection} {target}")
    overlay[collection] = [e for e in entries if e not in selected]


def main():
    base = ART / "binding-revision-05/catalog.json"
    catalog = read(base)
    changed = []
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in {"751f068c", "ecdbc2db", "13280227"}:
            continue
        template = read(verify_pin(entry["template"]))
        overlay = template["overlay"]
        before = entry["template"]
        if prefix == "751f068c":
            remove_projection(
                overlay, "literalProjections", "/parties/consignee/contactDetails/emailAddresses"
            )
        elif prefix == "ecdbc2db":
            remove_projection(
                overlay,
                "targetFromSourceSpans",
                "/parties/consignee/contactDetails/phoneNumbers",
                3,
            )
        else:
            remove_projection(
                overlay, "targetFromSourceSpans", "/goodsItemDetails/0/handlingInstructions/1"
            )
            remove_projection(overlay, "literalProjections", "/parties/shipper/country")
            registry = pin(ROOT / DEFAULT_REGISTRY_MANIFEST)
            overlay.setdefault("targetPrintedPostalCountries", []).append(
                {
                    "target": "/parties/shipper/country",
                    "addressTarget": "/parties/shipper/addressLine",
                    "codePath": "/party_localities/documentPatch.parties.shipper/country_code",
                    "manifestPath": registry["path"],
                    "sha256": registry["sha256"],
                    "omitIfUnprinted": True,
                }
            )
        path = ART / "semantic-revision-06" / (sid + ".json")
        atomic_publish_json(path, template)
        entry["template"] = pin(path)
        changed.append({"source": sid, "before": before, "after": pin(path)})
    if len(changed) != 3:
        raise ValueError("semantic repair source membership changed")
    output = ART / "semantic-revision-06/catalog.json"
    atomic_publish_json(output, catalog)
    atomic_publish_json(
        OUT / "semantic-revision-06.json",
        {
            "baseCatalog": pin(base),
            "candidateCatalog": pin(output),
            "changes": changed,
            "decisions": {
                "751f068c": "Keep the goods-area email in input, not direct-consignee contacts.",
                "ecdbc2db": "Keep goods-area consignee phones in input; notify phones remain eligible.",
                "13280227": "Use the complete owned payment-before-release sentence, not the preceding cost fragment. Shipper country requires its own printed postal witness, not exporter country.",
            },
        },
    )


if __name__ == "__main__":
    main()
