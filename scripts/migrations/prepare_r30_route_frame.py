"""Bind the two reviewed Manila receipt occurrences as complete locality frames."""

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json


def main():
    base = ART / "binding-revision-03/catalog.json"
    catalog = read(base)
    sid = next(s for s in catalog["templates"] if s[4:12] == "551657f3")
    entry = catalog["templates"][sid]
    template = read(verify_pin(entry["template"]))
    source = read(verify_pin(template["source"]))
    raw = verify_pin(entry["samplingSource"]).read_bytes()
    for byte, region, start in [(551, 35, 0), (1178, 71, 15)]:
        before = raw[byte : byte + 19].decode()
        expected = source["pages"][0]["regions"][region]["text"][start:]
        if "".join(before.split()) != expected or expected != "MANILA,PHILIPPINES":
            raise ValueError("reviewed route-frame evidence differs")
        template["overlay"]["overrides"].append(
            {
                "key": "reviewed:locality_frame:route.placeOfReceipt.country:0",
                "byteStart": byte,
                "spans": [[0, region, start, start + len(expected)]],
            }
        )
    destination = ART / "binding-revision-04" / (sid + ".json")
    atomic_publish_json(destination, template)
    entry["template"] = pin(destination)
    output = ART / "binding-revision-04/catalog.json"
    atomic_publish_json(output, catalog)
    atomic_publish_json(
        OUT / "binding-revision-04.json",
        {
            "baseCatalog": pin(base),
            "candidateCatalog": pin(output),
            "reason": "Receipt header and T/S CARGO FROM recurrence both have MANILA,PHILIPPINES with no post-comma space; complete locality-frame ownership is explicit, preserving current sampled name and printed country.",
        },
    )


if __name__ == "__main__":
    main()
