"""Stage the reviewed native address/contact/brand corrections from batches 11-12."""

from migrate_r30_synthetic import DATA, OUT, read
from publish_r30_sources import verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    sources = {}
    selected = {
        "8923530a",
        "b60ce9db",
        "aa69d7fe",
        "6b76fc16",
        "7d44dcc4",
        "d383908f",
        "56f7785e",
        "ccfee841",
    }
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in selected:
            continue
        template = read(verify_pin(entry["template"]))
        overlay = template["overlay"]
        native = read(verify_pin(template["source"]))
        edits, targets = [], []

        def append(key, values, overlay=overlay, edits=edits):
            if key in overlay:
                edits.append(
                    {
                        "action": "append",
                        "path": "/overlay/" + key,
                        "beforeLength": len(overlay[key]),
                        "after": values,
                    }
                )
            else:
                edits.append({"action": "add", "path": "/overlay/" + key, "after": values})

        def passage(regions, target, fragments=(), native=native, append=append, targets=targets):
            spans = [[0, i, 0, len(native["pages"][0]["regions"][i]["text"])] for i in regions]
            for page, index, text in fragments:
                original = native["pages"][page]["regions"][index]["text"]
                if original.count(text) != 1:
                    raise ValueError("reviewed fragment is not unique")
                start = original.index(text)
                spans.append([page, index, start, start + len(text)])
            value = {"spans": spans, "target": target, "targetCase": "upper"}
            append("targetFromSourceSpans", [value])
            targets.append(target)

        if prefix == "8923530a":
            index, owner = next(
                (i, v) for i, v in enumerate(overlay["overrides"]) if v["key"] == "source_brand"
            )
            if owner["spans"] != [[1, 8, 8, 16]]:
                raise ValueError("reviewed brand slot changed")
            edits.append(
                {
                    "action": "add",
                    "path": f"/overlay/overrides/{index}/valueRecipe",
                    "after": {"wording": "brand"},
                }
            )
            edits.append(
                {
                    "action": "add",
                    "path": "/overlay/newWording",
                    "after": {
                        "brand": "One short fictional food-product brand, one or two UPPERCASE "
                        "words. Brand name only; no product description, country, company suffix, "
                        "caption or instructions. Do not copy AL TAMAM."
                    },
                }
            )
            targets.append("/goodsItemDetails/0/description")
            reason = (
                "The old source_brand edit repeated the complete goods passage. Own the brand "
                "independently with reviewed linguistic generation; project its literal occurrence "
                "into the existing bounded description. Do not change sampled goods."
            )
        elif prefix == "b60ce9db":
            assert (
                native["pages"][0]["regions"][12]["text"]
                == "NO:21 KAT:4-5 PK 34843 MALTEPE ISTANBUL"
            )
            passage([11, 12], "/parties/shipper/addressLine")
            reason = (
                "Preserve the complete owned postal block including the printed PK prefix between "
                "sampled subowners. No contacts or unrelated captions are included."
            )
        elif prefix == "aa69d7fe":
            assert native["pages"][0]["regions"][48]["text"] == "TAMER.AMIN@DABUR.COM, 11856"
            passage([43, 45, 47], "/parties/notifyParties/0/addressLine", [[0, 48, "11856"]])
            reason = (
                "The notify postal block continues after its email with an independently owned "
                "five-digit postal component. Preserve the already anonymized/sampled digits, "
                "excluding the email itself."
            )
        elif prefix == "6b76fc16":
            assert native["pages"][0]["regions"][34]["text"] == "ARROW EGYPT LTD."
            passage([34], "/parties/notifyParties/0/name")
            reason = (
                "The full company-name occurrence includes terminal legal-suffix punctuation "
                "outside the old sampled subspan; project the complete printed name."
            )
        elif prefix == "7d44dcc4":
            before = overlay["targetFromOwners"]
            after = [
                v
                for v in before
                if not v["target"].startswith("/parties/consignee/contactDetails/emailAddresses")
            ]
            if len(before) - len(after) != 1:
                raise ValueError("expected one goods-area consignee email projection")
            edits.append(
                {
                    "action": "replace",
                    "path": "/overlay/targetFromOwners",
                    "before": before,
                    "after": after,
                }
            )
            edits.append(
                {
                    "action": "add",
                    "path": (
                        "/overlay/targetSets/~1parties~1consignee~1contactDetails~1emailAddresses"
                    ),
                    "after": None,
                }
            )
            targets.append("/parties/consignee/contactDetails/emailAddresses")
            reason = (
                "The explicit role-specific policy excludes goods-area-only consignee contacts "
                "even when footnoted. Preserve the distinct eligible notify continuation and the "
                "printed/anonymized email."
            )
        elif prefix == "ccfee841":
            assert native["pages"][0]["regions"][42]["text"] == "QINGDAO.CHINA"
            append(
                "overrides",
                [
                    {"key": "f25", "byteStart": 865, "spans": [[0, 42, 0, 7]]},
                    {"key": "f24", "byteStart": 873, "spans": [[0, 42, 8, 13]]},
                ],
            )
            reason = "The retained receipt owns loading locality and country as two explicit "
            reason += "subspans rather than one compound frame. Bind both to the same native "
            reason += "loading occurrence without touching the separate receipt/discharge roles."
        else:
            destinations = ["/placeOfIssue"]
            if prefix == "d383908f":
                destinations.append("/freight/paymentPlace")
            append(
                "targetCountryCoreferences",
                [{"from": "/route/portOfLoading", "to": v} for v in destinations],
            )
            targets.extend(v + "/country" for v in destinations)
            reason = (
                "Guarded document-local same-name co-reference: a country-qualified loading "
                "occurrence supplies the country only when the independently printed "
                "issue/payment locality matches it."
            )
        sources[prefix] = {
            "targetPaths": targets,
            "inputChanges": prefix in {"8923530a", "ccfee841"},
            "reason": reason,
            "edits": edits,
        }
    if sources.keys() != selected:
        raise ValueError("review membership differs")
    atomic_publish_json(
        OUT / "readback-review-48-input.json",
        {
            "status": "MAIN_AGENT_REVIEWED_TEMPLATE_CORRECTIONS",
            "baseCatalog": base,
            "sources": sources,
        },
    )


if __name__ == "__main__":
    main()
