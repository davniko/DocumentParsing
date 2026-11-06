"""Isolated correction of the one reviewed product-field instruction leak.

Historical campaigns remain immutable. Reconstruct both receipts from the
pinned sampling source and change only the approved product owner.
"""

from __future__ import annotations

import asyncio
import copy
import json
from decimal import Decimal
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner, digest
from document_ocr.synthesis.curated_descriptions import (
    CompiledDescriptionBlocks,
    CompiledDescriptionField,
    CompiledDescriptionSpan,
    project_rendered_descriptions,
)
from document_ocr.synthesis.curated_wording import (
    WORDING_PROMPT,
    WordingField,
    WordingRequest,
    native_wording_prompt,
    unpack_wording,
    wording_output_type,
)
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/paddle-migration/r30-independent-audit-20261103/product-repair"
IDENTITY = "syn_full_v7_9146bc816ae17430632360c3"
OLD = (
    ROOT
    / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
    / "batch-04/ordinary/campaign/candidates"
    / (IDENTITY + ".json")
)


def read(path):
    return json.loads(path.read_text())


def replay(source, edits):
    chunks, offset = [], 0
    for edit in sorted(edits, key=lambda e: e["byteStart"]):
        lo, hi = edit["byteStart"], edit["byteEnd"]
        if lo < offset or source[lo:hi].decode() != edit["before"]:
            raise ValueError("reviewed edit does not match pinned source")
        chunks.extend((source[offset:lo], edit["after"].encode()))
        offset = hi
    return b"".join([*chunks, source[offset:]]).decode()


async def main():
    destination = OUT / "candidate.json"
    if destination.exists():
        raise ValueError("reviewed candidate exists; no silent regeneration")
    old = read(OLD)
    wording_path = OLD.parent.parent / "wording" / OLD.name
    wording = read(wording_path)
    cp = (
        ROOT
        / "artifacts/paddle-migration/r30-independent-audit-20261103/template-repairs/catalog.json"
    )
    catalog = PaddleTemplateCatalog(
        ROOT, {"path": str(cp.relative_to(ROOT)), "sha256": sha256_file(cp)}
    )
    _, _, _, source = catalog._template(old["sourceDocumentId"])
    if replay(source, old["proof"]["edits"]) != old["joinedRawText"]:
        raise ValueError("historical proof does not reconstruct rejected receipt")
    edits = [e for e in old["proof"]["edits"] if e["key"] == "v011"]
    if len(edits) != 1 or edits[0]["after"] != wording["values"]["v011"]:
        raise ValueError("reviewed product is not one complete lexical owner")
    request = WordingRequest(
        sample_id=IDENTITY,
        context=json.dumps(
            {
                "goods": old["scenario"]["goods_identities"],
                "physicalForm": "ambient liquid",
                "packageType": "FIBRE DRUM",
                "netKgPerDrum": "66.591",
                "scope": (
                    "Generate product wording only. The previous result erroneously included "
                    "party-field keys and addresses. Do not include any party, prompt key, "
                    "instruction, HS code or unrelated shipment context. No product renaming "
                    "into the explicitly excluded same-heading goods. Keep the short "
                    "source-description scope."
                ),
            }
        ),
        fields=(
            WordingField(
                "v011",
                "goods description",
                edits[0]["before"],
                "Short commercial goods wording; retain the sampled product and physical form. "
                "Packing details are optional, but if mentioned must match the supplied facts.",
            ),
        ),
        gross_weight_kg=Decimal("30348.825"),
    )
    config = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    cfg = {k: config[k] for k in CuratedSynthesisConfig.model_fields}
    cfg.update(output=str((OUT / "provider").relative_to(ROOT)), concurrency=1, budget_usd=0.25)
    runner = Runner(ROOT, cfg)
    answer = await runner.call(
        "reviewed-product-only",
        IDENTITY,
        wording_output_type([request]),
        WORDING_PROMPT,
        native_wording_prompt([request]),
    )
    value = unpack_wording(answer, [request])[IDENTITY]["v011"]
    if value == edits[0]["after"]:
        raise ValueError("regeneration preserved rejected wording")
    new = copy.deepcopy(old)
    for edit in new["proof"]["edits"]:
        if edit["key"] == "v011":
            edit["after"] = value
    new["renderValues"]["v011"] = value
    wording["values"]["v011"] = value
    new["joinedRawText"] = replay(source, new["proof"]["edits"])
    new["joinedRawTextSha256"] = digest(new["joinedRawText"].encode())
    new["proof"]["renderedSha256"] = new["joinedRawTextSha256"]
    previous = old["proof"]["descriptionProjection"]
    blocks = CompiledDescriptionBlocks(
        previous["sourceSha256"],
        previous["contractSha256"],
        tuple(
            CompiledDescriptionField(
                field["path"],
                tuple(
                    CompiledDescriptionSpan(
                        s["sourceByteStart"],
                        s["sourceByteEnd"],
                        source[s["sourceByteStart"] : s["sourceByteEnd"]].decode(),
                    )
                    for s in field["spans"]
                ),
            )
            for field in previous["fields"]
        ),
    )
    _, projection = project_rendered_descriptions(
        blocks,
        source,
        new["joinedRawText"].encode(),
        new["proof"]["edits"],
        casing=previous["casing"],
    )
    new["proof"]["descriptionProjection"] = projection
    (field,) = projection["fields"]
    new["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = field["value"]
    new["wordingReceiptSha256"] = digest(wording)
    numeric = new["numericWordingProjection"]
    if any(numeric[k] for k in ("ownedClauses", "actions", "appendedClauses", "retainedClauses")):
        raise ValueError("this correction must not bypass numeric wording ownership")
    numeric["originalValuesSha256"] = numeric["projectedValuesSha256"] = digest(wording["values"])
    before_other, after_other = copy.deepcopy(old["target"]), copy.deepcopy(new["target"])
    for target in (before_other, after_other):
        target["documentPatch"]["goodsItemDetails"][0].pop("description")
    if before_other != after_other or old["scenario"] != new["scenario"]:
        raise ValueError("product correction changed other targets or shipment facts")
    atomic_publish_json(OUT / "wording.json", wording)
    atomic_publish_json(destination, new)
    result = catalog.render(destination)
    atomic_publish_json(OUT / "paddle-control.json", result)
    atomic_publish_json(
        OUT / "receipt.json",
        {
            "status": "REGENERATED_REQUIRES_MAIN_AGENT_READBACK",
            "before": {"path": str(OLD.relative_to(ROOT)), "sha256": sha256_file(OLD)},
            "after": {
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256_file(destination),
            },
            "newProduct": value,
            "unchangedOtherTargetsAndFacts": True,
            "allAttemptCostUsd": str(runner.spent),
        },
    )
    print(json.dumps({"product": value, "costUsd": str(runner.spent)}))


if __name__ == "__main__":
    asyncio.run(main())
