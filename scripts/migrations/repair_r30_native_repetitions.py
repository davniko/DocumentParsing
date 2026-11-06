"""Repair reviewed attachment repetitions and native product/handling boundaries."""

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = ART / "semantic-revision-06/catalog.json"
    catalog = read(base)
    changes = []
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in {"ecdbc2db", "d684b168", "0ad3c6b5"}:
            continue
        template = read(verify_pin(entry["template"]))
        overlay = template["overlay"]
        source = read(verify_pin(template["source"]))
        before = entry["template"]
        if prefix == "d684b168":
            expected = {
                (1, 1): "VESSEL/VOYAGE:ACHELOOS-0US76E1TK",
                (1, 3): ":NEWYORK APM TERMINAL ELIZABETH",
                (1, 5): ":PORT SAID WEST EGYPT",
            }
            for (p, r), text in expected.items():
                if source["pages"][p]["regions"][r]["text"] != text:
                    raise ValueError("reviewed attachment text changed")
            overlay["extras"].extend(
                [
                    {
                        "owner": "audit:attached-vessel",
                        "spans": [[1, 1, 14, 22]],
                        "fromKey": "anchor:documentPatch.transport.vesselName",
                    },
                    {
                        "owner": "audit:attached-voyage",
                        "spans": [[1, 1, 23, 32]],
                        "fromKey": "anchor:documentPatch.transport.voyageNumber",
                    },
                    {
                        "owner": "audit:attached-loading",
                        "spans": [[1, 3, 1, 31]],
                        "fromKey": "anchor:documentPatch.route.portOfLoading.name",
                    },
                    {
                        "owner": "audit:attached-discharge",
                        "spans": [[1, 5, 1, 15]],
                        "scenarioText": "/destination/name",
                    },
                    {
                        "owner": "audit:attached-discharge-country",
                        "spans": [[1, 5, 16, 21]],
                        "countryName": next(
                            e["countryName"]
                            for e in overlay["extras"]
                            if e["owner"] == "paddle:podcountry"
                        ),
                    },
                ]
            )
            overlay.setdefault("targetFromSourceSpans", []).append(
                {
                    "regions": [[0, 65], [0, 67]],
                    "target": "/goodsItemDetails/0/handlingInstructions/0",
                    "targetCase": "upper",
                }
            )
            overlay.setdefault("targetFromOwners", []).append(
                {
                    "keys": ["aux:declared:0:0"],
                    "separator": " ",
                    "target": "/goodsItemDetails/0/handlingInstructions/1",
                }
            )
        if prefix == "ecdbc2db":
            if source["pages"][0]["regions"][86]["text"] != "50,400 KGS FORMIC ACID 85% MIN":
                raise ValueError("reviewed goods-column passage changed")
            overlay.setdefault("targetFromSourceSpans", []).append(
                {
                    "regions": [[0, 86]],
                    "target": "/goodsItemDetails/0/description",
                    "targetCase": "upper",
                }
            )
        if prefix in {"ecdbc2db", "0ad3c6b5"}:
            destinations = ["/placeOfIssue"]
            if prefix == "ecdbc2db":
                destinations.append("/freight/paymentPlace")
            overlay.setdefault("targetCountryCoreferences", []).extend(
                {"from": "/route/portOfLoading", "to": destination} for destination in destinations
            )
        path = ART / "semantic-revision-07" / (sid + ".json")
        atomic_publish_json(path, template)
        entry["template"] = pin(path)
        changes.append({"source": sid, "before": before, "after": pin(path)})
    if len(changes) != 3:
        raise ValueError("reviewed source membership changed")
    output = ART / "semantic-revision-07/catalog.json"
    atomic_publish_json(output, catalog)
    atomic_publish_json(
        OUT / "semantic-revision-07.json",
        {
            "baseCatalog": pin(base),
            "candidateCatalog": pin(output),
            "changes": changes,
            "review": "Attachment route/vessel repetitions share the main shipment. Product-column mass qualifier stays with product text; detached per-package measurements do not. Country co-reference is conditional on equal retained place names and an explicit qualified loading occurrence.",
        },
    )


if __name__ == "__main__":
    main()
