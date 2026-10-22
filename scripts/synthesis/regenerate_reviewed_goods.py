"""Regenerate explicitly reviewed goods wording without changing other fields.

This is a bounded, paid correction operation, not an automatic classifier or
sampler. A pinned current receipt and an approved audit finding are required for
every requested sample. Rejected proposals never replace accepted receipts.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_scenarios import FilePin, _pin
from document_ocr.synthesis.curated_wording import (
    WORDING_PROMPT,
    native_wording_prompt,
    unpack_wording,
    wording_output_type,
)


class GoodsCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sourceDocumentId: str = Field(min_length=1)
    variant: int = Field(ge=1)
    sampleId: str = Field(min_length=1)
    wordingReceiptSha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    keys: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)
    auditEvidence: FilePin

    @model_validator(mode="after")
    def unique_keys(self):
        if len(set(self.keys)) != len(self.keys) or any(not key.strip() for key in self.keys):
            raise ValueError("reviewed goods correction requires unique nonblank keys")
        if not self.reason.strip():
            raise ValueError("reviewed goods correction requires an explicit reason")
        return self


class CorrectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["approved_for_regeneration"]
    corrections: tuple[GoodsCorrection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_samples(self):
        if len({row.sampleId for row in self.corrections}) != len(self.corrections):
            raise ValueError("duplicate reviewed sample")
        return self


async def regenerate(campaign, plan: CorrectionPlan) -> dict:
    """Stage all corrections, validate the full current wording, then commit."""
    root, output = campaign.root, campaign.output
    if (output / "manifest.json").exists():
        raise ValueError(
            "published campaigns are immutable; correct a separate inspection campaign"
        )
    plan_hash = digest(plan.model_dump(mode="json"))
    staged = []
    for row in plan.corrections:
        _pin(root, row.auditEvidence)
        if row.sourceDocumentId not in campaign.variant_counts:
            raise ValueError("reviewed source is outside the configured campaign")
        if row.variant > campaign.variant_counts[row.sourceDocumentId]:
            raise ValueError("reviewed variant is outside the configured campaign")
        blueprint, scenario, sample_id, target = campaign.plan(row.sourceDocumentId, row.variant)
        if sample_id != row.sampleId:
            raise ValueError("reviewed sample identity differs from the current plan")
        path = output / "wording" / f"{sample_id}.json"
        before_bytes = path.read_bytes()
        if digest(before_bytes) != row.wordingReceiptSha256:
            raise ValueError("reviewed wording receipt changed before correction")
        receipt = json.loads(before_bytes)
        candidate_path = output / "candidates" / f"{sample_id}.json"
        candidate_bytes = candidate_path.read_bytes() if candidate_path.exists() else None
        if candidate_bytes is not None:
            candidate = json.loads(candidate_bytes)
            if (
                candidate.get("documentId") != sample_id
                or candidate.get("sourceDocumentId") != row.sourceDocumentId
                or candidate.get("wordingReceiptSha256") != digest(receipt)
            ):
                raise ValueError("live candidate does not belong to the reviewed wording receipt")
        request = campaign.wording_request(blueprint, scenario, sample_id, target)
        campaign._validated_wording(receipt, request)
        products = {
            variable.key for variable in blueprint.contract.variables if variable.kind == "product"
        }
        fields = tuple(field for field in request.fields if field.key in row.keys)
        if not set(row.keys) <= products or {field.key for field in fields} != set(row.keys):
            raise ValueError("goods correction may own only current generated product fields")
        if any(
            set(group) & set(row.keys) and not set(group) <= set(row.keys)
            for group in request.description_groups
        ):
            raise ValueError("a jointly generated description group must be corrected together")
        scoped = replace(
            request,
            context=request.context
            + "\nREVIEWED GOODS-WORDING CORRECTION\n"
            + row.reason
            + "\nPrevious rejected wording:\n"
            + json.dumps({key: receipt["values"][key] for key in row.keys}, ensure_ascii=False)
            + "\nReturn new final commercial wording for only the requested product fields. "
            "Keep the sampled substance/product form, declared packing, shipment facts and "
            "source-conditioned description scope. Do not include review commentary.",
            fields=fields,
            description_groups=tuple(
                group for group in request.description_groups if set(group) <= set(row.keys)
            ),
        )
        staged.append((row, path, before_bytes, receipt, request, scoped, candidate_bytes))

    async def propose(item):
        row, path, before_bytes, receipt, request, scoped, candidate_bytes = item
        model = wording_output_type([scoped])
        prompt = native_wording_prompt([scoped])
        answer = await campaign.calls.call(
            "reviewed-goods-correction", row.sampleId, model, WORDING_PROMPT, prompt
        )
        replacement = unpack_wording(answer, [scoped])[row.sampleId]
        merged = copy.deepcopy(receipt)
        merged["values"].update(replacement)
        if merged["values"] == receipt["values"]:
            raise ValueError("reviewed correction did not change the rejected wording")
        if {
            key for key in merged["values"] if merged["values"][key] != receipt["values"][key]
        } - set(row.keys):
            raise ValueError("goods correction changed an unrelated lexical field")
        campaign.validate_generated_wording([request], {row.sampleId: merged["values"]})
        key = digest(
            [
                "reviewed-goods-correction",
                row.sampleId,
                model.model_json_schema(),
                WORDING_PROMPT,
                prompt,
                campaign.calls.config["provider"],
            ]
        )
        calls = sorted(
            (campaign.calls.output / "calls").glob(f"reviewed-goods-correction-{key}*.json")
        )
        if not calls:
            raise ValueError("goods correction has no immutable native call receipt")
        archive = output / "wording-history" / f"{row.sampleId}-{row.wordingReceiptSha256}.json"
        merged.setdefault("corrections", []).append(
            {
                "kind": "reviewed_goods_regeneration",
                "planSha256": plan_hash,
                "reason": row.reason,
                "auditEvidence": row.auditEvidence.model_dump(mode="json", exclude_none=True),
                "beforeReceipt": {
                    "path": str(archive.relative_to(output)),
                    "sha256": row.wordingReceiptSha256,
                },
                "calls": [
                    {"path": str(p.relative_to(root)), "sha256": sha256_file(p)} for p in calls
                ],
                "before": {k: receipt["values"][k] for k in row.keys},
                "after": replacement,
            }
        )
        return path, archive, before_bytes, merged, candidate_bytes

    results = await asyncio.gather(*(propose(item) for item in staged), return_exceptions=True)
    errors = [
        f"{type(value).__name__}: {value}" for value in results if isinstance(value, BaseException)
    ]
    directory = output / "reviewed-goods-corrections" / plan_hash
    if errors:
        report = {
            "status": "rejected",
            "errors": errors,
            "corrected": 0,
            "acceptedReceiptsUnchanged": True,
        }
        atomic_publish_json(directory / "rejection.json", report)
        raise ValueError("reviewed goods correction rejected: " + "; ".join(errors))
    # A concurrent edit must not be overwritten after a paid request returns.
    for path, _, before_bytes, _, candidate_bytes in results:
        if path.read_bytes() != before_bytes:
            raise ValueError("wording receipt changed during correction")
        candidate_path = output / "candidates" / path.name
        actual = candidate_path.read_bytes() if candidate_path.exists() else None
        if actual != candidate_bytes:
            raise ValueError("candidate changed during correction")
    candidate_archives = []
    for path, archive, before_bytes, merged, candidate_bytes in results:
        atomic_publish_bytes(archive, before_bytes)
        if candidate_bytes is not None:
            candidate_path = output / "candidates" / path.name
            candidate_archive = (
                output / "candidate-history" / (path.stem + "-" + digest(candidate_bytes) + ".json")
            )
            atomic_publish_bytes(candidate_archive, candidate_bytes)
            candidate_path.unlink()
            candidate_archives.append(str(candidate_archive.relative_to(output)))
        atomic_write_json(path, merged)
    report = {
        "status": "corrected_pending_full_render_review",
        "corrected": len(results),
        "sampleIds": [row.sampleId for row in plan.corrections],
        "unchangedUnrequestedFields": True,
        "supersededCandidateArchives": candidate_archives,
        "spentUsd": str(campaign.calls.spent),
    }
    atomic_publish_json(directory / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads((root / args.config).read_bytes())
    plan = CorrectionPlan.model_validate_json((root / args.plan).read_bytes())
    campaign = create_campaign(root, config)
    with (campaign.output / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.validate_source_readiness()
        print(json.dumps(asyncio.run(regenerate(campaign, plan)), indent=2))


if __name__ == "__main__":
    main()
