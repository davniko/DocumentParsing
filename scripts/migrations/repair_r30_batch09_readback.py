"""Stage reviewed repeated-description, contact-boundary and inventory-row repairs."""

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    folder = ART / "readback-revision-38"
    scopes = {}
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in {"ec7035fc", "d2dd2e5f", "68748b4d", "a3171859"}:
            continue
        template = read(verify_pin(entry["template"]))
        source = read(verify_pin(template["source"]))
        overlay = template["overlay"]
        paths, input_changes = [], False
        if prefix == "ec7035fc":
            assert "VAT No:3721333" in source["pages"][0]["regions"][14]["text"]
            assert source["pages"][0]["regions"][54]["text"].endswith("3721333")
            overlay.setdefault("literalProjections", []).append(
                {
                    "target": "/parties/shipper/country",
                    "pieces": [{"fromKey": "exporter_country"}],
                    "separator": "",
                    "targetCase": "upper",
                }
            )
            paths = ["/parties/shipper/country"]
            reason = (
                "Matching printed VAT and exporter-registration identifiers explicitly link "
                "the exporter country to this shipper. This is not inference from route, phone "
                "or a generic exporter caption."
            )
        elif prefix == "d2dd2e5f":
            action = next(
                x for x in overlay["overrides"] if x["key"] == "reviewed:consignee:obsolete_marker"
            )
            assert action.pop("valueRecipe") == {"fromKey": "reviewed:notify:in_block_contacts"}
            input_changes = True
            reason = (
                "The old receipt intentionally removes the obsolete consignee marker. Do not "
                "replace it with notify-owned contacts: that moved remote contacts into the "
                "consignee box while its policy-correct target remained absent. Consume the "
                "actual retained empty edit; notify contacts remain untouched."
            )
        elif prefix == "68748b4d":
            assert len(overlay["literalProjections"]) == 1
            assert overlay["literalProjections"][0]["target"] == "/goodsItemDetails/0/description"
            del overlay["literalProjections"]
            overlay.setdefault("targetFromSourceSpans", []).append(
                {
                    "target": "/goodsItemDetails/0/description",
                    "targetCase": "upper",
                    "groups": [
                        {"regions": [[0, 54]]},
                        {"regions": [[0, 61]]},
                        {"regions": [[0, 65], [0, 67]]},
                    ],
                    "deduplicatePassages": True,
                }
            )
            paths = ["/goodsItemDetails/0/description"]
            reason = (
                "The first complete product passage repeats verbatim. Project the ordered "
                "literal product groups and deduplicate only identical complete passages; "
                "preserve the distinct second product wording. No OCR text changes."
            )
        else:
            assert overlay["physicalGroups"] == [[0, [66, 67, 70, 73]]]
            overlay["physicalGroups"] = []
            key = "anchor:documentPatch.containers[2].typeDescription"
            action = next(
                x for x in overlay["overrides"] if x["key"] == key and x["byteStart"] == 1426
            )
            spans = [[0, 66, 25, 31], [0, 67, 0, 4]]
            assert action["spans"] == spans
            recipe = {"index": 2, "source": "40 DRY 9'6"}
            action["valueRecipe"] = {"equipment": recipe}
            action.update(separateRegionFrames=True, allowEmptyContinuations=True)
            overlay["extras"].append(
                {
                    "owner": "audit:split-third-equipment",
                    "existingKey": key,
                    "existingByteStart": 1426,
                    "spans": spans,
                    "equipment": recipe,
                    "separateRegionFrames": True,
                    "allowEmptyContinuations": True,
                }
            )
            input_changes = True
            reason = (
                "Three separate inventory rows were incorrectly one physical flow block, "
                "losing alignment with their quantities. Retain each row independently. Bind "
                "the two side-by-side pieces of the third equipment declaration as one "
                "sampled alias, including the unchanged-category branch; discard no equipment fact."
            )
        destination = folder / (sid + ".json")
        atomic_publish_json(destination, template)
        entry["template"] = pin(destination)
        scopes[prefix] = {"targetPaths": paths, "inputChanges": input_changes, "reason": reason}
    if len(scopes) != 4:
        raise ValueError("reviewed source membership differs")
    atomic_publish_json(folder / "catalog.json", catalog)
    atomic_publish_json(
        OUT / "readback-review-38.json",
        {"baseCatalog": base, "catalog": pin(folder / "catalog.json"), "sources": scopes},
    )


if __name__ == "__main__":
    main()
