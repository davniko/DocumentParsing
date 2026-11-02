"""Scoped, native goods/packing review; never edits or admits a candidate.

Full rendered-document review remains mandatory. This separate small-context
pass checks disjoint/nested scope and numerical enrichment without distracting
party/address blocks. Findings require explicit main review and fresh rendering.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner, digest


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["packing_scope", "material", "quantity_or_mass", "physical_form", "other"]
    evidence: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    findings: list[Finding]


class Reviews(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[Review]


def response_model(items):
    """Require every requested record in the native schema, not only prose."""
    fields = {}
    for index, item in enumerate(items):
        description = item["description"]

        def literal_evidence(value, *, text=description):
            for finding in value.findings:
                if finding.evidence not in text:
                    raise ValueError("review evidence is not literal description text")
            return value

        record = create_model(
            f"PackingScopeRecord{index}",
            __config__=ConfigDict(extra="forbid"),
            __validators__={"literal_evidence": model_validator(mode="after")(literal_evidence)},
            findings=(list[Finding], ...),
        )
        fields[f"s{index}"] = (record, Field(description=item["sample_id"]))
    return create_model("PackingScopeRecords", __config__=ConfigDict(extra="forbid"), **fields)


SYSTEM = """Review synthetic bill-of-lading goods wording against supplied shipment facts.
Return one object for EVERY supplied key (s0, s1, ...), with an empty findings list when
there is no concrete inconsistency. You are an auditor, not a text generator.

The public package groups are DISJOINT counted groups: 12 rolls + 8 pallets is
20 packages, NOT twelve rolls loaded on eight pallets. A group can separately
have inner/support packaging, but only its supplied private packing establishes
that relation. Do not mistake an ordinary product form (fabric rolls in boxes)
for an outer package assertion. Do not assume that mentioning a package in one
product specification necessarily qualifies unrelated products.

Check specifically:
- Wording must not turn disjoint public groups into a shared containment relation.
- Universal package claims must cover all groups they qualify; generic materials
  may not be silently upgraded (a generic bag plus a paper bag is not all paper).
- Explicit per-package fill weights/counts must agree with the supplied values;
  intrinsic grades, models, dimensions, purities and small inner retail pack sizes
  are allowed when they do not contradict the actual cargo facts. Do not demand
  that every plausible enrichment was pre-specified.
- A generated form must suit packing: e.g. liquid resin in paper powder bags is
  contradictory, while bagged resin pellets are plausible. Do not rename chemicals
  or require exact tariff/product matching; classification context is not a complete
  product specification. Do not demand extra DG labels absent from this layout.
- An implied hierarchy must not add inner counts to declared totals.
- Keep separate sampled products separate: joining names must not invent one
  chemically or physically contradictory product. Include every sampled product
  in the supplied goods passage, rather than silently omitting difficult items.
- Respect the supplied thermal regime and selected product state. A tariff
  alternative is not permission to add fresh/chilled/live wording to an ambient
  shipment. A bare word such as freezer can name machinery, not refrigerated cargo.
- Check obvious whole-shipment absurdities, not commercial preference: live cattle
  inside ordinary closed dry-box hardware without livestock support, or loose gas
  inside sacks, require review. Do not demand operational regulatory certification.

Only flag a concrete inconsistency, not awkward grammar, plurality, capitalization,
commercial taste, or a merely absent optional detail. For each finding quote an
EXACT substring of the supplied description as evidence and explain the conflict.
"""


def packet(candidate):
    scenario = candidate["scenario"]
    provenance = scenario["provenance"]
    if "declaredPackages" in provenance:
        declared = provenance["declaredPackages"]
    else:
        rows = candidate["target"]["documentPatch"]["goodsItemDetails"][0][
            "numberAndTypeOfPackages"
        ]
        if len(rows) != 1:
            raise ValueError("multiple public package rows require their declared-package ledger")
        declared = {
            "singlePublicRow": rows[0],
            "physicalProfile": provenance["registryLoadProfile"],
        }
    return {
        "sample_id": candidate["documentId"],
        "regime": provenance["independentDraw"]["provenance"]["regime"],
        "description": candidate["target"]["documentPatch"]["goodsItemDetails"][0]["description"],
        "goods": [
            {"hs6": g["hs6"], "classificationContext": g["phrase"]}
            for g in scenario["goods_identities"]
        ],
        "declaredGroups": declared,
        "privatePacking": {
            k: v
            for k, v in provenance["freshPrivatePacking"].items()
            if k
            in {
                "relation",
                "relatedCategory",
                "relatedQuantity",
                "unitNetKg",
                "unitGrossKg",
                "facts",
                "surfaces",
                "wordingFacts",
                "scopeLedgers",
            }
        },
        "equipment": scenario["cargo"].get("containerInformation", []),
        "thermalProductContext": provenance.get("thermalCommodityContext"),
    }


def validate_reviews(answer, items):
    expected = {item["sample_id"]: item for item in items}
    ids = [row.sample_id for row in answer.records]
    if len(ids) != len(expected) or set(ids) != expected.keys():
        raise ValueError("scoped review omitted, duplicated or substituted a sample")
    for row in answer.records:
        for finding in row.findings:
            if finding.evidence not in expected[row.sample_id]["description"]:
                raise ValueError("review evidence is not literal description text")


async def run(
    config_path, candidates_path, output, *, retry_transient_refusals=False, concurrency=None,
    batch_size=6,
):
    if batch_size not in range(1, 7):
        raise ValueError("scoped review batch size must be between one and six")
    root = Path.cwd()
    config = json.loads(config_path.read_text())
    base = {key: config[key] for key in CuratedSynthesisConfig.model_fields}
    base["output"] = str(output.relative_to(root))
    if concurrency is not None:
        base["concurrency"] = concurrency
    runner = Runner(root, base)
    if retry_transient_refusals:
        from run_package_alias_batch import transient_refusal_pins

        runner.retry_failed_receipt_sha256s = frozenset(
            p["sha256"] for p in transient_refusal_pins(output)
        )
    paths = [root / value for value in json.loads(candidates_path.read_text())]
    packets, pins = [], {}
    already_reviewed = 0
    for path in paths:
        candidate = json.loads(path.read_text())
        item = packet(candidate)
        if item["sample_id"] in pins:
            raise ValueError("duplicate candidate in focused audit")
        pins[item["sample_id"]] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        previous = (
            output
            / "reviews"
            / (item["sample_id"] + "-" + pins[item["sample_id"]]["sha256"] + ".json")
        )
        if previous.exists():
            saved = json.loads(previous.read_text())
            if saved["candidate"] != pins[item["sample_id"]]:
                raise ValueError("existing scoped review has different candidate provenance")
            validate_reviews(Reviews(records=[saved["review"]]), [item])
            already_reviewed += 1
            continue
        packets.append(item)

    async def check(items):
        identity = digest([x["sample_id"] for x in items])
        native = await runner.call(
            "multirow-packing-scope-review",
            identity,
            response_model(items),
            SYSTEM,
            json.dumps({f"s{i}": item for i, item in enumerate(items)}, ensure_ascii=False),
            output_retries=1,
        )
        answer = Reviews(
            records=[
                Review(sample_id=item["sample_id"], findings=getattr(native, f"s{i}").findings)
                for i, item in enumerate(items)
            ]
        )
        validate_reviews(answer, items)
        for row in answer.records:
            atomic_publish_json(
                output
                / "reviews"
                / (row.sample_id + "-" + pins[row.sample_id]["sha256"] + ".json"),
                {
                    "status": "SCOPED_REVIEW_REQUIRES_MAIN_ADJUDICATION",
                    "candidate": pins[row.sample_id],
                    "review": row.model_dump(mode="json"),
                },
            )
        return answer

    results = await asyncio.gather(
        *(check(packets[i : i + batch_size]) for i in range(0, len(packets), batch_size)),
        return_exceptions=True,
    )
    errors = [str(x) for x in results if isinstance(x, BaseException)]
    report = {
        "status": "SCOPED_REVIEW_COMPLETE_NOT_ADMISSION"
        if not errors
        else "HELD_INCOMPLETE_REVIEW",
        "inputRecords": len(paths),
        "alreadyReviewed": already_reviewed,
        "newlyRequested": len(packets),
        "batchSize": batch_size,
        "errors": errors,
        "findings": sum(
            len(r.findings) for a in results if not isinstance(a, BaseException) for r in a.records
        ),
        "spentUsd": str(runner.spent),
    }
    atomic_publish_json(
        output / ("summary-" + digest([pins, batch_size, sha256_file(Path(__file__))]) + ".json"), report
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-transient-refusals", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9))
    parser.add_argument("--batch-size", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.config.resolve(),
            args.candidates.resolve(),
            args.output.resolve(),
            retry_transient_refusals=args.retry_transient_refusals,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
        )
    )
