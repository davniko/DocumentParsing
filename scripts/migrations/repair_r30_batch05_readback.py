"""Correct confirmed repetitions and field boundaries found in batch-five readback."""

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    folder = ART / "readback-revision-21"
    scopes = {}
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in {"ff600393", "6018901a", "a3e3c1e3", "fdbc7d94", "0b7f3731"}:
            continue
        template = read(verify_pin(entry["template"]))
        source = read(verify_pin(template["source"]))
        overlay = template["overlay"]
        paths, input_changes = [], False
        if prefix == "ff600393":
            assert source["pages"][0]["regions"][44]["text"] == "CSCL URANUS 099W"
            assert source["pages"][1]["regions"][4]["text"] == "M.V. CSCL URANUS 099W"
            for key, text in (("vessel", "CSCL URANUS"), ("voyage", "099W")):
                overlay["extras"].append(
                    {
                        "owner": "audit:attachment-" + key,
                        "fromKey": key,
                        "fragments": [[1, 4, text]],
                    }
                )
            input_changes = True
            reason = "Attachment repeats the same main vessel/voyage in the source. Bind both "
            reason += "to the sampled main-carriage owners; retain the separate feeder vessel."
        elif prefix == "6018901a":
            assert source["pages"][0]["regions"][71]["text"] == "KOPER 2023-11-28"
            assert source["pages"][1]["regions"][41]["text"] == "KOPER 2023-11-28"
            for key, text in (("issue_place", "KOPER"), ("issue_date", "2023-11-28")):
                overlay["extras"].append(
                    {
                        "owner": "audit:attachment-" + key,
                        "fromKey": key,
                        "fragments": [[1, 41, text]],
                    }
                )
            input_changes = True
            reason = "Attachment issue place/date repeats the main issue block. Preserve that "
            reason += "co-reference under resampling instead of retaining KOPER/2023."
        elif prefix == "a3e3c1e3":
            copy = {
                "from": "/parties/notifyParties/0/contactDetails",
                "to": "/parties/consignee/contactDetails",
            }
            overlay["targetCopies"].remove(copy)
            paths = ["/parties/consignee/contactDetails"]
            reason = "Consignee contacts occur only in the goods-area continuation. The settled "
            reason += "role-specific exclusion overrides the general matching-marker rule. "
            reason += "Preserve notify contacts and all printed contact text."
        elif prefix == "fdbc7d94":
            action = next(
                a
                for a in overlay["targetFromSourceSpans"]
                if a["target"] == "/parties/shipper/contactDetails/contactName"
            )
            action["targetCase"] = "upper"
            paths = [action["target"]]
            reason = "Normalize the supported shipper contact name to the existing uppercase "
            reason += "target convention. Keep literal printed name and phone unchanged."
        else:
            assert source["pages"][0]["regions"][100]["text"] == "EASTMAN(TM) PROPIONIC ACID,"
            overlay.setdefault("targetFromSourceSpans", []).append(
                {
                    "regions": [[0, 100], [0, 102], [0, 104]],
                    "target": "/goodsItemDetails/0/description",
                    "targetCase": "upper",
                }
            )
            paths = ["/goodsItemDetails/0/description"]
            reason = "Project the complete owned product passage and embedded per-unit packing "
            reason += "from its three native regions. Join line breaks with spaces, avoiding "
            reason += "the old LIQUID60 fused target. Formal DG and Marks regions remain excluded."
        atomic_publish_json(folder / (sid + ".json"), template)
        entry["template"] = pin(folder / (sid + ".json"))
        scopes[prefix] = {"targetPaths": paths, "inputChanges": input_changes, "reason": reason}
    if len(scopes) != 5:
        raise ValueError("reviewed source membership differs")
    atomic_publish_json(folder / "catalog.json", catalog)
    atomic_publish_json(
        OUT / "readback-review-21.json",
        {
            "baseCatalog": base,
            "catalog": pin(folder / "catalog.json"),
            "sources": scopes,
        },
    )


if __name__ == "__main__":
    main()
