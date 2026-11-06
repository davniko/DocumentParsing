"""Replay certified privacy recipes and prove native-order-only corrections.

Does not overwrite pinned source decisions, staged records or certificates.
Publication is a separate, explicitly reviewed amendment operation.
"""

from __future__ import annotations

import copy
import json
import resource
import time
from functools import partial

import publish_r30_sources as publication
import rebase_r30_sources as full
import stage_r30_privacy as pilot

from document_ocr.atomic import atomic_publish_bytes, atomic_write_json
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.paddle_reflow import PaddleReflow


class OrderedEngine:
    def __init__(self, engine, order):
        self.engine, self.order = engine, order

    def render(self, *args, **kwargs):
        return self.engine.render(*args, **(kwargs | {"reading_order": self.order}))


def main():
    cfg = full.read(full.ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    engine = PaddleReflow(full.ROOT, ReflowPolicy.model_validate(cfg["positions"]["reflow"]))
    order_engines = {o: OrderedEngine(engine, o) for o in ("rectangles", "source")}
    privacy = full.read(full.PILOT / "real-privacy-overlays.json")["sources"]
    physical = full.read(full.PILOT / "paddle-overlays.json")["sources"]
    out = full.OUT / "native-order-amendment"
    timings = {o: 0.0 for o in order_engines}
    batches = []
    for batch, certificate_path in publication.active_source_certificates().items():
        cert = full.read(certificate_path)
        decisions_path = full.OUT / "batches" / batch / "main-agent-decisions.json"
        decisions = full.read(decisions_path) if batch != "pilot" else None
        evidence = (
            [publication.pin(decisions_path)]
            if decisions
            else [
                publication.pin(full.PILOT / "real-privacy-overlays.json"),
                publication.pin(full.PILOT / "paddle-overlays.json"),
            ]
        )
        records, changed, audited = {}, [], []
        for split, source_pin in cert["records"].items():
            output = []
            for old in full.rows(publication.verify_pin(source_pin)):
                did, candidate = old["documentId"], copy.deepcopy(old)
                if split == "real-train":
                    pre = did[4:12]
                    root = full.PILOT_ART if batch == "pilot" else full.ART
                    path = root / "sources" / (did + ".json")
                    source = full.read(path)
                    evidence.append(publication.pin(path))
                    if batch == "pilot":
                        transfer_path = root / "transfers" / (did + ".json")
                        transfer = full.read(transfer_path)
                        evidence.extend(
                            [
                                publication.pin(transfer_path),
                                publication.pin(root / "real-target-review" / (did + ".json")),
                            ]
                        )
                        run = partial(
                            pilot.stage,
                            source,
                            transfer,
                            privacy.get(pre, {}),
                            physical=physical.get(pre, {}),
                        )
                    else:
                        run = partial(full.stage_reviewed_record, source, decisions[pre])
                    outputs = {}
                    # Alternate order to avoid systematically warming only the new path.
                    orders = list(order_engines)
                    if len(audited) % 2:
                        orders.reverse()
                    for order in orders:
                        start = time.perf_counter()
                        outputs[order] = json.loads(json.dumps(run(engine=order_engines[order])))
                        timings[order] += time.perf_counter() - start
                    for order, fresh in outputs.items():
                        trial = copy.deepcopy(old)
                        for key in ("pages", "joinedRawText", "positionedText", "bboxText"):
                            trial[key] = fresh[key]
                        publication.order_only_record(old, trial)
                        if order == "rectangles" and trial != old:
                            raise ValueError(
                                f"previous published input no longer reproduces: {did}"
                            )
                        if order == "source":
                            candidate = trial
                    audited.append(did)
                if old != candidate:
                    changed.append(did)
                output.append(candidate)
            path = full.ART / "native-order-amendment" / batch / (split + ".jsonl")
            atomic_publish_bytes(path, publication.jsonl(output))
            records[split] = publication.pin(path)
        review = {
            "previous": publication.pin(certificate_path),
            "rationale": "Reviewed privacy recipes replayed with native source anchors; original "
            "rectangle mode reproduces published input exactly. Only region order "
            "changes; text, rectangles, targets and membership are identical.",
            "records": records,
            "sourceEvidence": evidence,
            "changedDocuments": sorted(changed),
            "auditedRealDocuments": audited,
        }
        review_path = out / ("zz-native-order-" + batch + ".json")
        atomic_write_json(review_path, review)
        entry = {
            "batch": batch,
            "audited": len(audited),
            "changed": len(changed),
            "review": publication.pin(review_path),
        }
        batches.append(entry)
        print(json.dumps(entry), flush=True)
    result = {
        "status": "AUDITED_NOT_PUBLISHED",
        "batches": batches,
        "replaySeconds": timings,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(out / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}), flush=True)


if __name__ == "__main__":
    main()
