"""Read-only final real-data audit, distributions, novelty, flags and API accounting.

Lexical flags are a review inventory, not automatic semantic rejection/acceptance.
Source decisions and exact receipts remain separate from mechanical integrity.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from audit_real_v7_r10 import normalized
from filter_real_v7_starting_dataset import ROOT, read, tree_hashes, validate_records, write_json
from jsonschema import Draft202012Validator
from project_real_v7_r9 import records
from publish_real_v7_batch2 import validate_rows
from run_real_v7_batch import usage

from document_ocr.hashing import sha256_file
from document_ocr.labeling_agents.direct_grounding import (
    order_consignment_candidates,
    source_fidelity_findings,
)


def parties(patch):
    for role, p in patch.get("parties", {}).items():
        if isinstance(p, list):
            for i, q in enumerate(p):
                yield f"{role}/{i}", q
        else:
            yield role, p


def costs():
    result = []
    for name in (
        "direct-real-batch006-20261005",
        "direct-real-batch007-20261005",
        "direct-real-batch007-recovery-20261005",
        "direct-real-batch008-20261005",
    ):
        base = ROOT / "artifacts/kie-labeling" / name
        totals = Counter()
        dollars = 0.0
        unknown = []
        reused = 0
        calls = 0
        for path in sorted(base.glob("runs/*/*/calls/*-request.json")):
            receipt_path = path.with_name(path.name.replace("-request", "-result"))
            if not receipt_path.exists():
                unknown.append(
                    dict(request=str(path.relative_to(ROOT)), reason="interrupted_without_result")
                )
                continue
            receipt = read(receipt_path)
            if receipt.get("billingStatus") == "reused_without_request":
                assert (
                    sha256_file(ROOT / receipt["reusedReceipt"]) == receipt["reusedReceiptSha256"]
                )
                reused += 1
                continue
            calls += 1
            tokens, cost = usage(receipt)
            totals.update(tokens)
            dollars += cost
            if receipt.get("billingStatus") not in (
                "known",
                "reported",
                "usage_reported",
            ) and not receipt.get("responses"):
                unknown.append(
                    dict(
                        request=str(path.relative_to(ROOT)),
                        reason=receipt.get("billingStatus"),
                        status=receipt["status"],
                    )
                )
        result.append(
            dict(
                batch=name,
                receiptedNewCalls=calls,
                reusedCalls=reused,
                estimatedUsd=dollars,
                usage=dict(totals),
                unknownRequests=unknown,
            )
        )
    return dict(
        batches=result,
        estimatedKnownUsd=sum(b["estimatedUsd"] for b in result),
        unknownRequests=sum(len(b["unknownRequests"]) for b in result),
        scope=(
            "All attempts with usage receipts, including excluded candidates; "
            "recovery reuse charged once. Estimates are not settled invoices."
        ),
    )


def audit(dataset, full, out):
    start = time.monotonic()
    before = tree_hashes(dataset)
    manifest = read(dataset / "projection-manifest.json")
    assert before == manifest["files"] | {
        "projection-manifest.json": sha256_file(dataset / "projection-manifest.json")
    }
    info = {r["documentId"]: r for r in manifest["documents"]}
    fm = read(full / "manifest.json")
    new = {r["documentId"]: r for r in fm["documents"]}
    full_check = validate_rows(full, new)
    baseline = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r11-reduced-472"
    old = {r["documentId"]: r for _, r in records(baseline)}
    validators = [
        Draft202012Validator(read(dataset / x)) for x in ("schema.json", "prompt-schema.json")
    ]
    expected = {}
    inv = []
    literal = []
    flags = []
    counters = defaultdict(Counter)
    for split, row in records(dataset):
        doc = row["documentId"]
        p = row["target"]["documentPatch"]
        meta = info[doc]
        fresh = doc in new
        for validator in validators:
            validator.validate(row["target"])
        assert doc not in expected
        expected[doc] = (
            split,
            (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
        )
        if not fresh:
            assert row == old[doc]
        assert "forwardingAndExportReferences" not in p and "carrier" not in p.get("parties", {})
        assert "vesselFlagCountry" not in p.get("transport", {})
        goods = p.get("goodsItemDetails", [])
        assert len(goods) == 1 and goods[0].get("description", "").strip()
        g = goods[0]
        assert "marksAndNumbers" not in g and "additionalInformation" not in g
        containers = p.get("containerInformation", [])
        ids = {c["equipmentIdentifier"] for c in containers}
        assert len(ids) == len(containers)
        placements = g.get("splitGoodsPlacement", [])
        assert all(a["equipmentIdentifier"] in ids for a in placements)
        qty = [a.get("packageQuantity") for a in g.get("numberAndTypeOfPackages", [])]
        allocation = [a.get("packageQuantity") for a in placements]
        if placements:
            assert list(g)[-1] == "splitGoodsPlacement"
        if (
            allocation
            and all(v is not None for v in [*qty, *allocation])
            and qty
            and sum(qty) != sum(allocation)
        ):
            flags.append(
                dict(
                    sample=meta["sample"],
                    kind="partial_or_conflicting_package_coverage",
                    total=sum(qty),
                    allocated=sum(allocation),
                    new=fresh,
                )
            )
        for role, party in parties(p):
            for phone in party.get("contactDetails", {}).get("phoneNumbers", []):
                assert len(re.sub(r"\D", "", phone)) > 3, (meta["sample"], role, phone)
                if re.search(r"/\d{1,3}$", phone):
                    flags.append(
                        dict(
                            sample=meta["sample"],
                            kind="phone_suffix",
                            role=role,
                            value=phone,
                            new=fresh,
                        )
                    )
        with pdfium.PdfDocument(ROOT / meta["pdf"]) as pdf:
            pages = len(pdf)
        assert pages <= 5 and pages >= 1
        ocrpages = len(re.findall(r"(?m)^--- PAGE \d+ ---$", row["joinedRawText"]))
        assert ocrpages == pages, (meta["sample"], ocrpages, pages)
        orders = order_consignment_candidates(row["joinedRawText"])
        if orders and p.get("negotiability") != "negotiable":
            flags.append(
                dict(
                    sample=meta["sample"],
                    kind="order_wording",
                    value=p.get("negotiability"),
                    evidence=orders,
                    new=fresh,
                )
            )
        for section in ("cargo", "equipment", "route_transport"):
            found = source_fidelity_findings(row["target"], row["joinedRawText"], section)
            if found:
                literal.append(
                    dict(
                        sample=meta["sample"],
                        documentId=doc,
                        new=fresh,
                        section=section,
                        findings=[f.model_dump() for f in found],
                    )
                )
        ship = p.get("parties", {}).get("shipper", {}).get("name")
        consignee = p.get("parties", {}).get("consignee", {}).get("name")
        item = dict(
            documentId=doc,
            sample=meta["sample"],
            split=split,
            new=fresh,
            pages=pages,
            shipper=ship,
            consignee=consignee,
            goods=g["description"],
            bill=p.get("billOfLadingNumber"),
            masterBill=p.get("masterBillOfLadingNumber"),
            issueDate=p.get("issueDate"),
            vessel=p.get("transport", {}).get("vesselName"),
            containers=sorted(ids),
            equipmentCount=len(containers),
            packageQuantity=qty,
            allocations=allocation,
            dg=bool(g.get("dangerousGoods")),
            thermal=any(c.get("temperatureSetpoint") for c in containers),
            hsCodes=g.get("hsCodes", []),
            ocrPath=str((dataset / "samples" / doc / "ocr.txt").relative_to(ROOT)),
            labelPath=str((dataset / "samples" / doc / "labels.json").relative_to(ROOT)),
            pdf=meta["pdf"],
            historicalSplit=meta.get("historicalSplit"),
        )
        inv.append(item)
        for cohort in ("all", "new" if fresh else "baseline"):
            c = counters[cohort]
            c["documents"] += 1
            c[split] += 1
            c[f"pages_{pages}"] += 1
            c[f"containers_{len(containers)}"] += 1
            c["multiContainer"] += len(containers) > 1
            c["noContainer"] += not containers
            c["DG"] += item["dg"]
            c["thermal"] += item["thermal"]
            c["missingBillNumber"] += not item["bill"]
            c["negotiable"] += p.get("negotiability") == "negotiable"
            c["missingConsignee"] += not consignee
            c["missingShipper"] += not ship
            c["reassignedHistoricalTrainToValidation"] += (
                split == "validation" and meta.get("historicalSplit") == "train"
            )
    validate_records(dataset, expected)
    assert Counter(s for s, _ in expected.values()) == Counter(train=600, validation=60)
    assert len(new) == 188 and len(old) == 472 and set(old) <= expected.keys()
    groups = {}
    novelty = {}
    for fields in [
        ("shipper",),
        ("consignee",),
        ("goods",),
        ("shipper", "consignee"),
        ("shipper", "consignee", "goods"),
        ("bill",),
    ]:
        key = "+".join(fields)
        mapping = defaultdict(list)
        for r in inv:
            if all(r[f] for f in fields):
                mapping[tuple(normalized(r[f]) for f in fields)].append(r)
        groups[key] = [
            dict(
                key=k,
                documents=[r["sample"] for r in v],
                crossSplit=len({r["split"] for r in v}) > 1,
            )
            for k, v in mapping.items()
            if len(v) > 1
        ]
        baseline_keys = {k for k, v in mapping.items() if any(not r["new"] for r in v)}
        novelty[key] = dict(
            newDocumentsWithCompleteKey=sum(r["new"] and all(r[f] for f in fields) for r in inv),
            newDocumentsWithNovelKey=sum(
                r["new"] for k, v in mapping.items() if k not in baseline_keys for r in v
            ),
            distinctKeys=len(mapping),
            repeatedGroups=len(groups[key]),
            crossSplitGroups=sum(g["crossSplit"] for g in groups[key]),
        )
    assert tree_hashes(dataset) == before
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "inventory.json", inv)
    write_json(out / "lexical-review.json", literal)
    write_json(out / "review-flags.json", flags)
    write_json(out / "repetition.json", groups)
    write_json(out / "costs.json", costs())
    summary = dict(
        counts={k: dict(v) for k, v in counters.items()},
        novelty=novelty,
        fullReadback=full_check,
        reducedReadback=len(expected),
        baselineUnchanged=len(old),
        literalFlagGroups=len(literal),
        newLiteralFlagGroups=sum(r["new"] for r in literal),
        reviewFlags=len(flags),
        elapsedSeconds=round(time.monotonic() - start, 3),
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(ROOT / a.dataset, ROOT / a.full, ROOT / a.output)
