"""Complete reviewed split-count owners and reconcile an already-bound equipment cell."""

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    folder = ART / "readback-revision-19"
    scopes = {}
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in {"08ec8a5f", "8bc1663b"}:
            continue
        template = read(verify_pin(entry["template"]))
        source = read(verify_pin(template["source"]))
        overlay = template["overlay"]
        if prefix == "08ec8a5f":
            for key, start, region, text in (
                ("v010", 1216, 61, "1O00"),
                ("v010", 1276, 65, "1OOD"),
                ("anchor:documentPatch.cargoPackages[0].typeCategory", 1221, 61, "BAGS"),
                ("anchor:documentPatch.cargoPackages[0].typeCategory", 1281, 65, "BAGS"),
            ):
                assert source["pages"][0]["regions"][region]["text"].count(text) == 1
                overlay["overrides"].append(
                    {"key": key, "byteStart": start, "fragments": [[0, region, text]]}
                )
            assert len(overlay["literalProjections"]) == 1
            assert overlay["literalProjections"][0]["target"] == "/goodsItemDetails/0/description"
            del overlay["literalProjections"]
            overlay.setdefault("targetFromSourceSpans", []).append(
                {
                    "fragments": [
                        [0, 61, "BENNI BRAND, SKIMMED MILK"],
                        [0, 65, source["pages"][0]["regions"][65]["text"]],
                        [0, 67, "POLY LINER INSIDE(LOW HEAT)"],
                    ],
                    "target": "/goodsItemDetails/0/description",
                    "targetCase": "upper",
                }
            )
            paths = ["/goodsItemDetails/0/description"]
            reason = "Two native count occurrences read 1O00/1OOD. Bind their exact owned "
            reason += "spans rather than borrowing the attachment's 1000 occurrence. Both "
            reason += "split and compound package owners project the literal product passage."
        else:
            key = "anchor:documentPatch.containers[0].typeDescription"
            extra = next(x for x in overlay["extras"] if x["owner"] == "new:equipment")
            assert extra["fragments"] == [[0, 66, "40 DRY 9'6"]]
            extra.update(existingKey=key, existingByteStart=1673)
            overlay["overrides"].append(
                {
                    "key": key,
                    "byteStart": 1673,
                    "fragments": extra["fragments"],
                    "valueRecipe": {"equipment": extra["equipment"]},
                }
            )
            paths = []
            reason = "The existing equipment edit and new native semantic owner cover the "
            reason += "same span. Use the existing-owner guard and one shared equipment recipe, "
            reason += "retaining the source slot's alias style and sampled equipment semantics."
        atomic_publish_json(folder / (sid + ".json"), template)
        entry["template"] = pin(folder / (sid + ".json"))
        scopes[prefix] = {"targetPaths": paths, "inputChanges": True, "reason": reason}
    if set(scopes) != {"08ec8a5f", "8bc1663b"}:
        raise ValueError("reviewed source membership differs")
    atomic_publish_json(folder / "catalog.json", catalog)
    atomic_publish_json(
        OUT / "readback-review-19.json",
        {
            "baseCatalog": base,
            "catalog": pin(folder / "catalog.json"),
            "sources": scopes,
        },
    )


if __name__ == "__main__":
    main()
