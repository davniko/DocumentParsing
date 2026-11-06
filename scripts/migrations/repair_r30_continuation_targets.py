"""Correct reviewed synthetic continuations without importing source-only absence."""

from migrate_r30_synthetic import ART, DATA, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = read(DATA / "manifest.json")["templateCatalog"]
    catalog = read(verify_pin(base))
    scope = {
        "73fdbd15": (
            ["/parties/deliveryAgent/addressLine", "/parties/deliveryAgent/country"],
            "Delivery address explicitly continues on the next sheet; retain its postal country.",
        ),
        "b2ba22f2": (
            ["/route/portOfDischarge/country"],
            "The real source's single SINGAPORE token has no separate country label. "
            "Synthetic bound locality phrases can explicitly print their sampled country; "
            "do not unconditionally remove the existing country target.",
        ),
        "2d5864aa": (
            ["/parties/consignee/contactDetails"],
            "Matching CN> continuation has the same phone/email as NP>. "
            "The correct explicit copy was subsequently deleted by a stale removal.",
        ),
        "ede7dfe3": (
            ["/negotiability"],
            "The source and sampled name owner contain a named consignee, "
            "not an order instruction. "
            "The conditional preprinted caption does not make negotiability unknown.",
        ),
    }
    changes = {}
    for sid, entry in catalog["templates"].items():
        prefix = sid[4:12]
        if prefix not in scope:
            continue
        template = read(verify_pin(entry["template"]))
        overlay = template["overlay"]
        if prefix == "73fdbd15":
            actions = [
                a
                for a in overlay["targetFromSourceSpans"]
                if a["target"] == "/parties/deliveryAgent/addressLine"
            ]
            if len(actions) != 1:
                raise ValueError("reviewed postal projection changed")
            actions[0]["regions"].append([1, 33])
            overlay["targetRemovals"].remove("/parties/deliveryAgent/country")
        elif prefix == "b2ba22f2":
            overlay["targetRemovals"].remove("/route/portOfDischarge/country")
        elif prefix == "2d5864aa":
            assert {
                "from": "/parties/notifyParties/0/contactDetails",
                "to": "/parties/consignee/contactDetails",
            } in overlay["targetCopies"]
            overlay["targetRemovals"].remove("/parties/consignee/contactDetails")
        else:
            overlay["targetSets"]["/negotiability"] = "non_negotiable"
        folder = ART / "readback-revision-15"
        atomic_publish_json(folder / (sid + ".json"), template)
        entry["template"] = pin(folder / (sid + ".json"))
        paths, reason = scope[prefix]
        changes[prefix] = {"targetPaths": paths, "inputChanges": False, "reason": reason}
    if set(changes) != set(scope):
        raise ValueError("reviewed source membership changed")
    atomic_publish_json(folder / "catalog.json", catalog)
    atomic_publish_json(
        OUT / "readback-review-15.json",
        {
            "baseCatalog": base,
            "catalog": pin(folder / "catalog.json"),
            "sources": changes,
        },
    )


if __name__ == "__main__":
    main()
