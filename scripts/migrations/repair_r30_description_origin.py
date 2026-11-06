"""Apply reviewed complete product passages and explicit FOB country co-reference."""

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.country_presentations import DEFAULT_REGISTRY_MANIFEST


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    changes = {}
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in {"0262ae40", "7b12b905"}:
            continue
        template = read(verify_pin(entry["template"]))
        overlay = template["overlay"]
        source = read(verify_pin(template["source"]))
        if prefix == "0262ae40":
            raw = verify_pin(entry["samplingSource"]).read_text()
            if "FOB SHENZHEN, CHINA" not in raw:
                raise ValueError("reviewed FOB country evidence changed")
            registry = pin(ROOT / DEFAULT_REGISTRY_MANIFEST)
            overlay.setdefault("literalProjections", []).append({
                "target": "/route/portOfLoading/country", "separator": "",
                "pieces": [{"owner": "reviewed-fob-country", "countryName": {
                    "codePath": "/origin/country_code", "manifestPath": registry["path"],
                    "sha256": registry["sha256"],
                }}],
            })
            overlay.setdefault("targetCountryCoreferences", []).append({
                "from": "/route/portOfLoading", "to": "/placeOfIssue",
            })
            paths = ["/route/portOfLoading/country", "/placeOfIssue/country"]
            reason = "Explicit FOB declaration qualifies the same printed loading place; "
            reason += "equal printed issue place shares that reviewed local country evidence."
        else:
            expected = {14: "ROTATION PRINTING MACHINE AND OVEN",
                        19: "(USED)(844319701000- 851419800011)",
                        21: "FLAME LAMINATION MACHINE (230 CM) (845180800000)",
                        22: "AUTOMATION SEWING MACHINE AND ACCESSORIES",
                        23: "(845180800000)"}
            for i, text in expected.items():
                if source["pages"][1]["regions"][i]["text"] != text:
                    raise ValueError("reviewed description continuation changed")
            overlay.setdefault("targetFromSourceSpans", []).append({
                "regions": [[1, i] for i in expected], "targetCase": "upper",
                "target": "/goodsItemDetails/0/description",
            })
            paths = ["/goodsItemDetails/0/description"]
            reason = "Copy complete product passages including attached parentheses and "
            reason += "inline codes, excluding interleaved tare/seal/count columns and duplicate copies."
        path = ART / "readback-revision-13" / (sid + ".json")
        atomic_publish_json(path, template)
        entry["template"] = pin(path)
        changes[prefix] = {"targetPaths": paths, "inputChanges": False, "reason": reason}
    destination = ART / "readback-revision-13/catalog.json"
    atomic_publish_json(destination, catalog)
    atomic_publish_json(OUT / "readback-review-13.json", {
        "baseCatalog": base, "catalog": pin(destination), "sources": changes,
    })


if __name__ == "__main__":
    main()
