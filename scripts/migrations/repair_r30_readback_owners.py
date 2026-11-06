"""Stage source-verified repeated owners and literal policy projections."""

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.country_presentations import DEFAULT_REGISTRY_MANIFEST


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    scopes = {
        "2ae5fe6d": ([], "Own the newly visible repeated package-total cell."),
        "bef02d48": ([], "Repeat the retained printed volume, not unrounded planning volume."),
        "89227fc3": ([], "Main-page and attachment issue declarations share sampled owners."),
        "0f03e468": ([], "Attachment vessel/voyage share the main-carriage owners."),
        "85b6388a": (["/parties/shipper/country"], "Use the pinned preferred country name."),
        "45386289": (
            ["/goodsItemDetails/0/handlingInstructions"],
            "Retain the literal shipment-specific payment-before-release clause.",
        ),
        "d69c24cb": (
            ["/goodsItemDetails/0/handlingInstructions"],
            "Retain the complete release instruction, excluding preceding costs alone.",
        ),
        "7b12b905": (["/freight"], "Project the explicitly selected FREIGHT PREPAID value."),
        "8a73dafe": (
            ["/goodsItemDetails/0/origin"],
            "Project the country in the explicitly captioned origin box.",
        ),
    }
    changed = {}
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in scopes:
            continue
        template = read(verify_pin(entry["template"]))
        overlay = template["overlay"]
        pages = read(verify_pin(template["source"]))["pages"]

        def expect(page, region, value, pages=pages, prefix=prefix):
            if pages[page]["regions"][region]["text"] != value:
                raise ValueError(f"reviewed native occurrence changed: {prefix} {page}.{region}")

        if prefix == "2ae5fe6d":
            expect(0, 59, "2455.00CARTONS")
            overlay["extras"].append(
                {
                    "owner": "audit:summary-package-rows",
                    "regions": [[0, 59]],
                    "packageRows": {"goods": 0, "style": "full"},
                }
            )
        elif prefix == "bef02d48":
            expect(0, 77, "18.923cu. m.")
            extra = next(x for x in overlay["extras"] if x["owner"] == "paddle:main-cube")
            del extra["physical"]
            extra["fromKey"] = "anchor:documentPatch.cargoGroups[0].volume.value"
        elif prefix == "89227fc3":
            for region, value, field in (
                (125, "GENOA", "placeOfIssue.name"),
                (128, "26 MAY 2025", "issueDate"),
            ):
                expect(0, region, value)
                overlay["extras"].append(
                    {
                        "owner": "audit:main-" + field,
                        "regions": [[0, region]],
                        "fromKey": "anchor:documentPatch." + field,
                    }
                )
        elif prefix == "0f03e468":
            for region, value, key in ((1, "YM TIPTOP", "vessel"), (2, "022W", "voyage")):
                expect(1, region, value)
                overlay["extras"].append(
                    {
                        "owner": "audit:attachment-" + key,
                        "regions": [[1, region]],
                        "fromKey": key,
                    }
                )
        elif prefix == "85b6388a":
            projection = next(
                x
                for x in overlay["literalProjections"]
                if x["target"] == "/parties/shipper/country"
            )
            registry = pin(ROOT / DEFAULT_REGISTRY_MANIFEST)
            projection["pieces"] = [
                {
                    "owner": "country",
                    "countryName": {
                        "codePath": "/origin/country_code",
                        "manifestPath": registry["path"],
                        "sha256": registry["sha256"],
                    },
                }
            ]
        elif prefix == "45386289":
            expect(2, 9, "ALL EXPENSES OF DISCHARGING INCLUDING RE-LOADING OF EMPTY")
            expect(2, 10, "CONTAINER(S) TO BE PAID BY RECEIVERS BEFORE RELEASE OF CARGO,")
            expect(2, 11, "INCLUDING LASHING AND UNLASHING.")
            overlay.setdefault("targetFromSourceSpans", []).append(
                {
                    "regions": [[2, 9], [2, 10], [2, 11]],
                    "target": "/goodsItemDetails/0/handlingInstructions",
                    "asList": True,
                    "targetCase": "upper",
                }
            )
        elif prefix == "d69c24cb":
            line = pages[0]["regions"][68]["text"]
            start = line.index("Any transportation")
            expect(
                0,
                69,
                "expss of containers at Alexandria terminal to be collected fm "
                "receiver's prior releasing the cargo.",
            )
            overlay.setdefault("targetFromSourceSpans", []).append(
                {
                    "spans": [
                        [0, 68, start, len(line)],
                        [0, 69, 0, len(pages[0]["regions"][69]["text"])],
                    ],
                    "target": "/goodsItemDetails/0/handlingInstructions",
                    "asList": True,
                    "targetCase": "upper",
                }
            )
        elif prefix == "7b12b905":
            expect(0, 114, "FREIGHT PREPAID")
            overlay["targetSets"]["/freight/paymentArrangement"] = "prepaid"
        elif prefix == "8a73dafe":
            expect(0, 31, "Point and Country of Origin")
            expect(0, 37, "* SAUDI ARABIA")
            overlay.setdefault("targetFromRenderedOwners", []).append(
                {
                    "owners": ["new:origin-country"],
                    "separator": " ",
                    "target": "/goodsItemDetails/0/origin/name",
                    "targetCase": "upper",
                }
            )
        path = ART / "readback-revision-12" / (sid + ".json")
        atomic_publish_json(path, template)
        entry["template"] = pin(path)
        targets, reason = scopes[prefix]
        changed[prefix] = {"targetPaths": targets, "inputChanges": True, "reason": reason}
    if changed.keys() != scopes.keys():
        raise ValueError("reviewed source membership differs")
    destination = ART / "readback-revision-12/catalog.json"
    atomic_publish_json(destination, catalog)
    atomic_publish_json(
        OUT / "readback-review-12.json",
        {
            "baseCatalog": base,
            "catalog": pin(destination),
            "sources": changed,
        },
    )


if __name__ == "__main__":
    main()
