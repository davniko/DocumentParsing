"""Stage PDF-verified field boundaries and repeated locality owners."""

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    folder = ART / "readback-revision-35"
    scopes = {}
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in {
            "c06401a6",
            "873a72fe",
            "eed2a09c",
            "a915a411",
            "69209200",
            "8fc5897f",
            "a65cb58e",
        }:
            continue
        template = read(verify_pin(entry["template"]))
        source = read(verify_pin(template["source"]))
        overlay = template["overlay"]
        paths, input_changes = [], False
        if prefix == "c06401a6":
            assert source["pages"][0]["regions"][43]["text"] == "PORT SAID WEST FREE ZONE"
            action = next(
                a
                for a in overlay["targetFromSourceSpans"]
                if a["target"] == "/route/finalDestination/name"
            )
            overlay["targetFromSourceSpans"].remove(action)
            overlay["targetFromSourceSpans"].append(
                {
                    "regions": [[0, 42], [0, 43]],
                    "target": "/goodsItemDetails/0/description",
                    "targetCase": "upper",
                }
            )
            paths = ["/route/finalDestination", "/goodsItemDetails/0/description"]
            reason = "Bare free-zone qualifier is attached to the product, not captioned as a "
            reason += (
                "destination. Preserve it in description; remove the unsupported role projection."
            )
        elif prefix == "873a72fe":
            assert source["pages"][0]["regions"][60]["text"] == "** EMAIL: REDFWD.LOGS@GMAIL.COM"
            action = next(
                a
                for a in overlay["literalProjections"]
                if a["target"] == "/parties/consignee/contactDetails/emailAddresses/0"
            )
            overlay["literalProjections"].remove(action)
            paths = ["/parties/consignee/contactDetails/emailAddresses"]
            reason = "Do not project a goods-area-only consignee email. Keep its anonymized "
            reason += "rendered owner and the independently printed consignee phone."
        elif prefix == "eed2a09c":
            assert source["pages"][1]["regions"][63]["text"] == "ISTANBUL,TR"
            overlay["extras"].append(
                {
                    "owner": "audit:issue-place-middle",
                    "fromKey": "reviewed:locality_frame:placeOfIssue.country:0",
                    "regions": [[1, 63]],
                }
            )
            input_changes = True
            reason = "The middle sheet repeats the same source issue place/country; consume "
            reason += (
                "the existing sampled complete locality owner, as the front and last sheets do."
            )
        elif prefix == "a65cb58e":
            for region, offset in ((53, 997), (66, 1192)):
                literal = source["pages"][0]["regions"][region]["text"]
                assert literal in {"1 x 40C", "1 X 40C"}
                overlay["extras"].append(
                    {
                        "owner": f"audit:equipment-occurrence-{region}",
                        "existingKey": "agent:equipment_receipt:container:0",
                        "existingByteStart": offset,
                        "regions": [[0, region]],
                        "parts": [
                            {"owner": "equipment-count", "literalText": literal[:4]},
                            {
                                "owner": "equipment-type",
                                "equipment": {
                                    "index": 0,
                                    "source": "40C",
                                },
                            },
                        ],
                    }
                )
            input_changes = True
            reason = "The source Paddle 40C lacks the GLM HC marker. Even when sampled equipment "
            reason += "equals the old GLM equipment and creates no edit, both native occurrences "
            reason += "must render the retained sampled category. Explicit existing-edit ownership "
            reason += "prevents duplicate edits; no source high-cube inference or target change."
        elif prefix in {"a915a411", "69209200"}:
            destinations = ["/placeOfIssue"]
            if prefix == "69209200":
                destinations.append("/route/placeOfReceipt")
            overlay.setdefault("targetCountryCoreferences", []).extend(
                {"from": "/route/portOfLoading", "to": target} for target in destinations
            )
            paths = [target + "/country" for target in destinations]
            reason = "Use the existing reviewed same-place country mechanism only when the "
            reason += "sampled retained role names agree and loading explicitly supplies country. "
            reason += "Original source places differ; no source country is inferred."
        else:
            assert source["pages"][0]["regions"][61]["text"] == "FRESH APPLES AS PER"
            action = next(
                a
                for a in overlay["targetFromSourceSpans"]
                if a["target"] == "/goodsItemDetails/0/description"
            )
            action["spans"] = [[0, 61, 0, 12]]
            paths = [action["target"]]
            reason = "End the product passage before the explicitly separate AS PER INV. "
            reason += "commercial reference. The later HS, support-packing and net/ventilation "
            reason += (
                "statements are independently owned auxiliary fields. Keep their input unchanged."
            )
        destination = folder / (sid + ".json")
        atomic_publish_json(destination, template)
        entry["template"] = pin(destination)
        scopes[prefix] = {"targetPaths": paths, "inputChanges": input_changes, "reason": reason}
    if len(scopes) != 7:
        raise ValueError("reviewed template membership differs")
    atomic_publish_json(folder / "catalog.json", catalog)
    atomic_publish_json(
        OUT / "readback-review-35.json",
        {
            "baseCatalog": base,
            "catalog": pin(folder / "catalog.json"),
            "sources": scopes,
        },
    )


if __name__ == "__main__":
    main()
