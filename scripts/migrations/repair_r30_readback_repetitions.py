"""Close reviewed repeated mass, punctuation, casing and country co-reference gaps."""

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    changes = {}
    selected = {"1464f450", "8579256f", "341a94a0", "88deddad", "ede7dfe3"}
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in selected:
            continue
        template = read(verify_pin(entry["template"]))
        overlay = template["overlay"]
        if prefix == "1464f450":
            source = read(verify_pin(template["source"]))
            assert source["pages"][1]["regions"][16]["text"] == "22,500.00"
            overlay.setdefault("extras", []).append(
                {
                    "owner": "audit:attachment-gross",
                    "regions": [[1, 16]],
                    "fromKey": "v007",
                }
            )
            paths, input_changes = [], True
            reason = "Attachment mass repeats the same gross-weight owner as the main declaration."
        elif prefix == "8579256f":
            actions = [
                x
                for x in overlay["literalProjections"]
                if x["target"].startswith("/goodsItemDetails/0/handlingInstructions/")
            ]
            assert len(actions) == 2
            for action in actions:
                action["targetCase"] = "upper"
            paths, input_changes = ["/goodsItemDetails/0/handlingInstructions"], False
            reason = (
                "Retain the two reviewed operational clauses with policy-required target casing."
            )
        elif prefix == "341a94a0":
            overlay.setdefault("targetFromSourceSpans", []).append(
                {
                    "regions": [[0, 3]],
                    "target": "/parties/shipper/name",
                    "targetCase": "upper",
                }
            )
            paths, input_changes = ["/parties/shipper/name"], False
            reason = "Preserve the literal terminal name punctuation outside the sampled name span."
        else:
            overlay.setdefault("targetCountryCoreferences", []).append(
                {
                    "from": "/route/portOfLoading",
                    "to": "/placeOfIssue",
                }
            )
            paths, input_changes = ["/placeOfIssue/country"], False
            reason = (
                "Explicit same-place country-qualified loading occurrence supports issue country; "
            )
            reason += "Conditional co-reference requires equal locality names."
        folder = ART / "readback-revision-17"
        atomic_publish_json(folder / (sid + ".json"), template)
        entry["template"] = pin(folder / (sid + ".json"))
        changes[prefix] = {"targetPaths": paths, "inputChanges": input_changes, "reason": reason}
    if set(changes) != selected:
        raise ValueError("reviewed source membership changed")
    atomic_publish_json(folder / "catalog.json", catalog)
    atomic_publish_json(
        OUT / "readback-review-17.json",
        {
            "baseCatalog": base,
            "catalog": pin(folder / "catalog.json"),
            "sources": changes,
        },
    )


if __name__ == "__main__":
    main()
