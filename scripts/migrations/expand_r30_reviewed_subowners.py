"""Compile exact subspans of reviewed complete owners, without fuzzy matching.

Whitespace-only differences in Paddle line segmentation are ignored for locating
characters, never for emitted content. An identical whole-owner character stream
and source-byte containment prove which repeated occurrence a subfield owns.
"""

import json

from migrate_r30_synthetic import ART, OUT, read
from prepare_r30_paddle_pilot import load_pages
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.paddle_render import spans_for


def project_child(parent, child, spans):
    if not parent["byteStart"] <= child["byteStart"] < child["byteEnd"] <= parent["byteEnd"]:
        return None
    raw = parent["before"].encode()
    lo, hi = child["byteStart"] - parent["byteStart"], child["byteEnd"] - parent["byteStart"]
    if raw[lo:hi].decode() != child["before"]:
        raise ValueError("subowner does not match its complete source owner")
    native = [
        (s.page, s.region, s.start + i, ch)
        for s in spans
        for i, ch in enumerate(s.expected)
        if not ch.isspace()
    ]
    source_chars = [c for c in parent["before"] if not c.isspace()]
    if source_chars != [x[3] for x in native]:
        return None
    a = len("".join(raw[:lo].decode().split()))
    b = len("".join(raw[:hi].decode().split()))
    if a == b or not child["before"].strip() or child["before"] != child["before"].strip():
        return None
    result = []
    for p, r, offset, _ in native[a:b]:
        if result and result[-1][:2] == [p, r]:
            result[-1][3] = offset + 1
        else:
            result.append([p, r, offset, offset + 1])
    return result


def main():
    base = ART / "binding-revision-01/catalog.json"
    catalog = read(base)
    proofs = []
    for sid, entry in catalog["templates"].items():
        template = read(verify_pin(entry["template"]))
        transfer = read(verify_pin(template["transfer"]))
        pages = load_pages(read(verify_pin(template["source"])))
        overrides = template["overlay"]["overrides"]
        parents = []
        for rule in overrides:
            if rule.keys() - {"key", "byteStart", "spans", "fragments", "regions"}:
                continue
            for slot in transfer["slots"]:
                if slot["key"] == rule["key"] and (
                    "byteStart" not in rule or slot["byteStart"] == rule["byteStart"]
                ):
                    parents.append((slot, spans_for(rule, pages)))
        additions = []
        for child in transfer["slots"]:
            if child["status"] == "exact_measured_owner_candidate":
                continue
            if any(
                r["key"] == child["key"]
                and ("byteStart" not in r or r["byteStart"] == child["byteStart"])
                for r in overrides
            ):
                continue
            proposals = {}
            for parent, spans in parents:
                result = project_child(parent, child, spans)
                if result:
                    proposals[json.dumps(result)] = (result, parent["owner"])
            if len(proposals) > 1:
                raise ValueError(f"contradictory reviewed parent spans for {sid}: {child['owner']}")
            if not proposals:
                continue
            result, parent = next(iter(proposals.values()))
            additions.append(
                {"key": child["key"], "byteStart": child["byteStart"], "spans": result}
            )
            proofs.append(
                {
                    "source": sid,
                    "owner": child["owner"],
                    "before": child["before"],
                    "reviewedParent": parent,
                    "spans": result,
                }
            )
        if additions:
            overrides.extend(additions)
            path = ART / "binding-revision-03" / (sid + ".json")
            atomic_publish_json(path, template)
            entry["template"] = pin(path)
    path = ART / "binding-revision-03/catalog.json"
    atomic_publish_json(path, catalog)
    atomic_publish_json(
        OUT / "binding-revision-03.json",
        {
            "baseCatalog": pin(base),
            "candidateCatalog": pin(path),
            "proofs": proofs,
            "status": "EXACT_CONTAINMENT_PROVED_REQUIRES_REPLAY",
        },
    )
    print(
        json.dumps(
            {
                "owners": len(proofs),
                "templates": len({r["source"] for r in proofs}),
                "catalog": pin(path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
