"""Propose newly visible linguistic owners for an explicitly selected Paddle batch."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from prepare_r30_paddle_pilot import ART, OUT, ROOT, read

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner
from document_ocr.synthesis.paddle_wording import SYSTEM, wording_model


def select_shipments(transfers, sample_ids):
    """An explicit subset permits fresh controls without replaying reviewed calls."""
    selected = [(transfer, row) for transfer in transfers for row in transfer["descendants"]]
    identities = [row["id"] for _, row in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate shipment identity in selected transfers")
    if sample_ids is None:
        return selected
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample selection must be nonempty and distinct")
    wanted = set(sample_ids)
    if not wanted <= set(identities):
        raise ValueError("sample selection is outside the selected sources")
    return [(transfer, row) for transfer, row in selected if row["id"] in wanted]


async def main(prefixes, *, artifact_root, overlay_path, report_root, sample_ids=None):
    base = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    cfg = {k: base[k] for k in CuratedSynthesisConfig.model_fields}
    cfg.update(
        output=str((artifact_root / "new-wording-provider").relative_to(ROOT)),
        concurrency=4,
        budget_usd=1,
    )
    runner = Runner(ROOT, cfg)
    overlays = read(overlay_path)["sources"]

    async def one(transfer, selected):
        identity = selected["id"]
        path = artifact_root / "new-wording" / (identity + ".json")
        if path.exists():
            raise ValueError("wording already exists; no silent regeneration")
        source = read(artifact_root / "sources" / (transfer["source"] + ".json"))
        candidate = read(selected["path"])
        spec = overlays[source["documentId"][4:12]]["newWording"]
        model = wording_model(
            spec, overlays[source["documentId"][4:12]].get("newWordingPatterns", {})
        )
        prompt = json.dumps(
            {
                "requested": spec,
                "origin": candidate["scenario"]["origin"],
                "destination": candidate["scenario"]["destination"],
                "goods": candidate["scenario"]["goods_identities"],
                "printedGoods": candidate["target"]["documentPatch"].get("goodsItemDetails", []),
                "equipment": candidate["target"]["documentPatch"].get("containerInformation", []),
                "parties": candidate["target"]["documentPatch"].get("parties", {}),
                "partyLocalities": candidate["scenario"]["party_localities"],
            },
            ensure_ascii=False,
        )
        result = await runner.call(
            "paddle-additional-wording", identity, model, SYSTEM, prompt, output_retries=1
        )
        atomic_write_json(
            path,
            {
                "status": "PROPOSED_REQUIRES_REVIEW",
                "documentId": identity,
                "sourceFingerprint": source["nativeFingerprint"],
                "candidateSha256": sha256_file(ROOT / selected["path"]),
                "prompt": prompt,
                "values": result.model_dump(),
            },
        )
        print(json.dumps({"id": identity, "values": result.model_dump()}), flush=True)

    transfers = []
    for prefix in prefixes:
        (path,) = (artifact_root / "transfers").glob("doc_" + prefix + "*.json")
        transfers.append(read(path))
    calls = [
        one(transfer, selected) for transfer, selected in select_shipments(transfers, sample_ids)
    ]
    results = await asyncio.gather(*calls, return_exceptions=True)
    errors = [str(r) for r in results if isinstance(r, BaseException)]
    atomic_write_json(
        report_root / "new-wording-proposals.json",
        {"calls": len(calls), "errors": errors, "allAttemptCostUsd": str(runner.spent)},
    )
    if errors:
        raise ValueError(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixes", nargs="+")
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--batch")
    parser.add_argument("--overlay-path", type=Path)
    parser.add_argument("--sample-ids", nargs="+")
    args = parser.parse_args()
    if args.scope == "full":
        if not args.batch:
            parser.error("full-scope wording requires its reviewed batch")
        from rebase_r30_sources import ART as artifact_root
        from rebase_r30_sources import OUT as full_report

        report_root = full_report / "batches" / args.batch
    else:
        if args.batch:
            parser.error("pilot has no full-rebase batch")
        artifact_root, report_root = ART, OUT
    asyncio.run(
        main(
            args.prefixes,
            artifact_root=artifact_root,
            overlay_path=args.overlay_path or report_root / "paddle-overlays.json",
            report_root=report_root,
            sample_ids=args.sample_ids,
        )
    )
