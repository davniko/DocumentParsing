"""Publish only complete, replayable and adjudicated curated synthesis pilots.

Publication consumes the existing campaign; it never generates or repairs data.
The manifest is the final commit record and binds the dataset, gallery, current
source files, candidates and semantic review receipts by content hash.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.curated import digest, runner_call_key
from document_ocr.synthesis.curated_casing import CasingPolicy, TargetCasing
from document_ocr.synthesis.curated_sampling import source_variant_counts
from document_ocr.synthesis.curated_wording import (
    RenderedReview,
    RenderedReviewRequest,
    rendered_review_prompt,
    review_output_type,
    unpack_review,
)
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

DECODER_TASK = "bill_of_lading_extraction_v7_reduced_direct"


def decoder_target(annotation: dict) -> dict:
    """Export only canonical facts; internal ownership receipts remain versioned."""
    historical = get_training_task("bill_of_lading_extraction_v7_reduced")
    canonical = historical.canonicalize(annotation)
    if canonical != annotation:
        raise ValueError("publication annotation is not canonical reduced V7")
    return get_training_task(DECODER_TASK).canonicalize(canonical["documentPatch"])


def decoder_constraints(annotation_constraints: dict) -> RelationExplicitTaskConstraints:
    """Reuse the frozen vocabulary, not the annotation envelope's schema pins."""
    original = RelationExplicitTaskConstraints.model_validate_json(
        json.dumps(annotation_constraints)
    )
    get_training_task("bill_of_lading_extraction_v7_reduced").bind_constraints(original)
    task = get_training_task(DECODER_TASK)
    value = original.model_dump(mode="json")
    value.update(
        task=DECODER_TASK,
        basePromptSchemaSha256=task.base_prompt_schema_sha256(),
        targetSchemaSha256=sha256_bytes(canonical_json_bytes(task.target_schema())),
    )
    constraints = RelationExplicitTaskConstraints.model_validate_json(json.dumps(value))
    task.bind_constraints(constraints)
    return constraints


def review_contract_hash(count: int, target_casing: TargetCasing = "uppercase") -> str:
    """Bind review acceptance to the active instructions and native output schema."""
    return digest(
        {
            "system": rendered_review_prompt(target_casing),
            "schema": review_output_type(count).model_json_schema(),
        }
    )


def review_request_hash(request: RenderedReviewRequest) -> str:
    """Bind a review to the exact assembled request and its ordered sample IDs."""
    return digest(
        {
            "sampleIds": request.sample_ids,
            "system": request.system,
            "prompt": request.prompt,
            "schema": request.output_type.model_json_schema(),
        }
    )


def review_runner_evidence(
    root: Path,
    output: Path,
    provider: dict[str, Any],
    sid: str,
    request: RenderedReviewRequest,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read the complete canonical Runner attempt chain for one native review."""
    key = runner_call_key(
        "rendered-review", sid, request.output_type, request.system, request.prompt, provider
    )
    filename = re.compile(
        rf"rendered-review-{re.escape(key)}(?:-attempt([2-9]|[1-9][0-9]+))?\.json"
    )
    attempts = []
    for path in (output / "calls").glob(f"rendered-review-{key}*.json"):
        match = filename.fullmatch(path.name)
        if match:
            attempts.append((int(match.group(1) or 1), path))
    attempts.sort()
    if [number for number, _ in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError(f"{sid}: incomplete native rendered-review attempt chain")
    if not attempts:
        raise ValueError(f"{sid}: missing native rendered-review Runner receipt")
    pins = []
    successful = []
    resolved_root = root.resolve()
    for number, path in attempts:
        if not path.resolve().is_relative_to(resolved_root):
            raise ValueError(f"{sid}: native rendered-review receipt escapes project root")
        payload = read_regular_file_bytes(path)
        row = json.loads(payload)
        if not isinstance(row, dict) or any(
            row.get(field) != value
            for field, value in {
                "stage": "rendered-review",
                "identity": sid,
                "promptSha256": digest(request.prompt),
                "schemaSha256": digest(request.output_type.model_json_schema()),
            }.items()
        ):
            raise ValueError(f"{sid}: native rendered-review receipt differs from request")
        if row.get("output") is None:
            if not isinstance(row.get("error"), str) or not row["error"]:
                raise ValueError(f"{sid}: invalid failed rendered-review receipt")
        else:
            if (
                number != len(attempts)
                or row.get("error") is not None
                or row.get("costStatus") != "provider_reported"
                or not isinstance(row.get("usage"), dict)
                or not isinstance(row.get("messages"), list)
                or not row["messages"]
            ):
                raise ValueError(f"{sid}: invalid successful rendered-review receipt")
            successful.append(
                request.output_type.model_validate(row["output"]).model_dump(mode="json")
            )
        pins.append({"path": path.relative_to(root).as_posix(), "sha256": digest(payload)})
    if len(successful) != 1:
        raise ValueError(f"{sid}: rendered-review requires one successful native receipt")
    return pins, successful[0]


class _Spend(Protocol):
    config: dict[str, Any]

    @property
    def spent(self) -> Decimal: ...


class PublicationCampaign(Protocol):
    """The read-only campaign surface needed to certify a publication."""

    root: Path
    output: Path
    config: dict[str, Any]
    rows: dict[str, dict]
    calls: _Spend

    def replay_candidate(self, candidate: dict) -> None: ...

    def review_requests(
        self, sid: str, candidates: Sequence[dict[str, Any]]
    ) -> tuple[RenderedReviewRequest, ...]: ...


class FindingDecision(BaseModel):
    """An explicit rejection of a review finding against the unchanged sample."""

    model_config = ConfigDict(extra="forbid")
    findingIndex: int = Field(ge=0)
    findingSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["rejected", "fixed"]
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)


class SourceAdjudication(BaseModel):
    """Hash-bound decisions for every finding in one final-text review.

    ``fixed`` deliberately cannot approve a sample: a repaired candidate needs
    a new semantic review. Rejection evidence quotes the current rendered OCR.
    """

    model_config = ConfigDict(extra="forbid")
    sourceDocumentId: str
    reviewSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidateHashes: dict[str, str]
    decisions: list[FindingDecision]


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("publication identity is not a safe filename")
    return value


def _read(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing publication prerequisite: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"publication prerequisite is not an object: {path}")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fence(value: str, language: str) -> str:
    width = max([3, *(len(match[0]) + 1 for match in re.finditer(r"`+", value))])
    fence = "`" * width
    return f"{fence}{language}\n{value.rstrip()}\n{fence}\n"


def _adjudicate(
    output: Path, sid: str, receipt: dict, candidates: dict[str, dict], review: RenderedReview
) -> str | None:
    if not review.findings:
        return None
    raw = _read(output / "adjudications" / f"{sid}.json")
    adjudication = SourceAdjudication.model_validate(raw)
    if (
        adjudication.sourceDocumentId != sid
        or adjudication.reviewSha256 != digest(receipt)
        or adjudication.candidateHashes != receipt["candidateHashes"]
    ):
        raise ValueError(f"{sid}: stale adjudication identity or review/candidate hashes")
    decisions = {decision.findingIndex: decision for decision in adjudication.decisions}
    if len(decisions) != len(adjudication.decisions) or set(decisions) != set(
        range(len(review.findings))
    ):
        raise ValueError(f"{sid}: unresolved or duplicate finding adjudications")
    for index, finding in enumerate(review.findings):
        decision = decisions[index]
        if decision.findingSha256 != digest(finding.model_dump(mode="json")):
            raise ValueError(f"{sid}: adjudication belongs to another finding")
        if decision.decision != "rejected":
            raise ValueError(
                f"{sid}: fixed findings require a new review of the repaired candidate"
            )
        text = candidates[finding.sample_id]["joinedRawText"]
        if not decision.reason.strip() or any(
            not quote.strip() or quote not in text for quote in decision.evidence
        ):
            raise ValueError(f"{sid}: adjudication lacks a reason or exact rendered evidence")
    return digest(raw)


def _source_snapshot(campaign: PublicationCampaign) -> tuple[dict, dict[str, dict]]:
    dataset = (campaign.root / campaign.config["dataset"]).resolve()
    output = campaign.output.resolve()
    if output == dataset or output in dataset.parents or dataset in output.parents:
        raise ValueError("publication output must be independent of the source dataset")
    files, train = {}, {}
    for split in ("train", "validation"):
        path = dataset / f"{split}.jsonl"
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if len({row["documentId"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate source identities in {split}")
        files[split] = {"path": str(path), "sha256": digest(payload), "records": len(rows)}
        if split == "train":
            train = {row["documentId"]: row for row in rows}
    return files, train


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    """Replays are idempotent; an existing different publication is preserved."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite a different published artifact: {path}")
        return
    temporary = path.with_name(path.name + ".pending")
    if temporary.exists() and temporary.read_bytes() != payload:
        raise ValueError(f"different interrupted publication exists: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def publish_campaign(campaign: PublicationCampaign, *, publish: bool = True) -> dict:
    """Validate every expected sample, then publish an independent plain-OCR set.

    Any missing, stale, tampered or unresolved record raises before publication.
    This function performs no paid calls and does not edit source datasets,
    candidates, reviews or adjudications. Call it under the campaign's lock.
    """
    config = campaign.config
    before, current = _source_snapshot(campaign)
    counts = source_variant_counts(config, current)
    selected = list(counts)
    constraints_path = campaign.root / config["task_constraints"]
    constraints_bytes = constraints_path.read_bytes()
    if digest(constraints_bytes) != config["task_constraints_sha256"]:
        raise ValueError("publication task constraints changed")
    constraints = decoder_constraints(json.loads(constraints_bytes))
    decoder = get_training_task(DECODER_TASK).bind_constraints(constraints)
    decoder_artifacts = {
        "schema.json": (decoder.target_schema()),
        "prompt-schema.json": json.loads(decoder.prompt_schema_json()),
        "task-constraints.json": constraints.model_dump(mode="json"),
    }
    decoder_payloads = {
        name: (_json(value) + "\n").encode() for name, value in decoder_artifacts.items()
    }
    records, entries, source_entries, seen_text = [], [], [], set()
    gallery = [
        "# Full curated synthesis pilot: source and rendered variants\n",
        "Each source is followed by every published variant. Text is plain OCR; "
        "position-enriched variants are published separately. Labels are in `dataset.jsonl`.\n",
    ]
    rejected_findings = 0
    contract_hashes = {
        sid: review_contract_hash(n, CasingPolicy.model_validate(config.get("casing", {})).target)
        for sid, n in counts.items()
    }
    expected_ids = set()
    for source_index, sid in enumerate(selected, 1):
        variants, contract_hash = counts[sid], contract_hashes[sid]
        _safe_id(sid)
        if sid not in current or campaign.rows.get(sid) != current[sid]:
            raise ValueError(f"{sid}: current source differs from the campaign snapshot")
        source = current[sid]
        source_hash = digest(source["joinedRawText"].encode())
        source_target_hash = digest(source["target"])
        candidates = {}
        for variant in range(1, variants + 1):
            sample_id = "syn_full_v7_" + digest([sid, config["seed"], variant])[:24]
            candidate = _read(campaign.output / "candidates" / f"{sample_id}.json")
            if (
                candidate["documentId"] != sample_id
                or candidate["sourceDocumentId"] != sid
                or candidate["variant"] != variant
                or candidate["seed"] != config["seed"]
            ):
                raise ValueError(f"{sample_id}: candidate scenario identity differs")
            campaign.replay_candidate(candidate)
            fingerprint = digest(candidate["joinedRawText"].encode())
            if candidate["joinedRawTextSha256"] != fingerprint:
                raise ValueError(f"{sample_id}: rendered text hash differs")
            if fingerprint in seen_text:
                raise ValueError(f"{sample_id}: duplicate generated OCR")
            seen_text.add(fingerprint)
            expected_ids.add(sample_id)
            candidates[sample_id] = candidate
        receipt = _read(campaign.output / "reviews" / f"{sid}.json")
        hashes = {sample_id: digest(candidate) for sample_id, candidate in candidates.items()}
        if receipt.get("sourceDocumentId") != sid or receipt.get("candidateHashes") != hashes:
            raise ValueError(f"{sid}: stale review or incomplete candidate hash coverage")
        if receipt.get("reviewContractSha256") != contract_hash:
            raise ValueError(f"{sid}: stale or missing review instruction/schema contract")
        requests = campaign.review_requests(sid, tuple(candidates.values()))
        request_evidence = receipt.get("reviewRequests")
        if not isinstance(request_evidence, list) or len(request_evidence) != len(requests):
            raise ValueError(f"{sid}: stale or missing exact review request contract")
        native_reviews = []
        for request, recorded in zip(requests, request_evidence, strict=True):
            if (
                not isinstance(recorded, dict)
                or set(recorded) != {"sampleIds", "sha256", "callReceipts"}
                or recorded["sampleIds"] != list(request.sample_ids)
                or recorded["sha256"] != review_request_hash(request)
            ):
                raise ValueError(f"{sid}: stale or missing exact review request contract")
            pins, native_output = review_runner_evidence(
                campaign.root,
                campaign.output,
                campaign.calls.config["provider"],
                sid,
                request,
            )
            if recorded["callReceipts"] != pins:
                raise ValueError(f"{sid}: rendered-review Runner receipt pin differs")
            native_reviews.append(
                unpack_review(
                    request.output_type.model_validate(native_output), list(request.sample_ids)
                )
            )
        native_review = RenderedReview(
            reviewed_ids=[sample for batch in native_reviews for sample in batch.reviewed_ids],
            findings=[finding for batch in native_reviews for finding in batch.findings],
        )
        if receipt.get("review") != native_review.model_dump(mode="json"):
            raise ValueError(f"{sid}: review findings differ from native Runner output")
        review = RenderedReview.model_validate(receipt["review"])
        if len(review.reviewed_ids) != len(candidates) or set(review.reviewed_ids) != set(
            candidates
        ):
            raise ValueError(f"{sid}: semantic review has incomplete or duplicate sample coverage")
        if any(finding.sample_id not in candidates for finding in review.findings):
            raise ValueError(f"{sid}: semantic review cites an unrequested sample")
        adjudication_hash = _adjudicate(campaign.output, sid, receipt, candidates, review)
        rejected_findings += len(review.findings)
        source_entries.append(
            {
                "documentId": sid,
                "sourceSha256": source_hash,
                "targetSha256": source_target_hash,
                "reviewSha256": digest(receipt),
                "adjudicationSha256": adjudication_hash,
            }
        )
        gallery.extend(
            [
                f"\n---\n\n## {source_index}. Source `{sid}`\n",
                _fence(source["joinedRawText"], "text"),
            ]
        )
        for sample_id, candidate in candidates.items():
            target = decoder.canonicalize(decoder_target(candidate["target"]))
            records.append(
                {
                    **{
                        key: candidate[key]
                        for key in (
                            "documentId",
                            "sourceDocumentId",
                            "joinedRawText",
                            "joinedRawTextSha256",
                        )
                    },
                    "target": target,
                }
            )
            entries.append(
                {
                    "documentId": sample_id,
                    "sourceDocumentId": sid,
                    "seed": candidate["seed"],
                    "variant": candidate["variant"],
                    "sourceSha256": source_hash,
                    "sourceTargetSha256": source_target_hash,
                    "renderedSha256": candidate["joinedRawTextSha256"],
                    "targetSha256": digest(target),
                    "candidateTargetSha256": digest(candidate["target"]),
                    "candidateSha256": hashes[sample_id],
                    "reviewSha256": digest(receipt),
                    "adjudicationSha256": adjudication_hash,
                }
            )
            gallery.extend(
                [
                    f"\n### Variant {candidate['variant']}: `{sample_id}`\n",
                    _fence(candidate["joinedRawText"], "text"),
                ]
            )
    actual_ids = {path.stem for path in (campaign.output / "candidates").glob("*.json")}
    if actual_ids != expected_ids:
        raise ValueError("candidate directory differs from exact configured publication scope")
    after, _ = _source_snapshot(campaign)
    if before != after:
        raise ValueError("source dataset changed during publication validation")
    dataset_bytes = ("".join(_json(record) + "\n" for record in records)).encode()
    gallery_bytes = "\n".join(gallery).encode()
    manifest = {
        "schemaVersion": 1,
        "task": DECODER_TASK,
        "targetSchemaSha256": constraints.targetSchemaSha256,
        "taskConstraintsSha256": digest(decoder_payloads["task-constraints.json"]),
        "annotationTaskConstraintsSha256": config["task_constraints_sha256"],
        "reviewContractSha256": contract_hash,
        "configSha256": digest(config),
        "inputRepresentation": "plain_ocr",
        "positionsSynthesized": False,
        "expected": sum(counts.values()),
        "valid": len(records),
        "sourceCount": len(selected),
        "variantsPerSource": config["variants_per_source"],
        "rejectedReviewFindings": rejected_findings,
        "unresolvedReviewFindings": 0,
        "costUsdAllCampaignCalls": str(campaign.calls.spent),
        "sourceDataset": before,
        "sources": source_entries,
        "samples": entries,
        "files": {
            **{name: digest(value) for name, value in decoder_payloads.items()},
            "dataset.jsonl": digest(dataset_bytes),
            "samples.md": digest(gallery_bytes),
        },
    }
    if "template_sampling" in config:
        manifest.pop("reviewContractSha256")
        manifest["reviewContractSha256BySource"] = contract_hashes
        manifest["variantsPerSource"] = counts
        manifest["templateSampling"] = config["template_sampling"]
    payloads = {
        **decoder_payloads,
        "dataset.jsonl": dataset_bytes,
        "samples.md": gallery_bytes,
        "manifest.json": (_json(manifest) + "\n").encode(),
    }
    if publish:
        # Check all existing files before creating any; the manifest commits last.
        for name, payload in payloads.items():
            path = campaign.output / name
            if path.exists() and path.read_bytes() != payload:
                raise ValueError(f"refusing to overwrite a different published artifact: {path}")
            pending = path.with_name(path.name + ".pending")
            if pending.exists() and pending.read_bytes() != payload:
                raise ValueError(f"different interrupted publication exists: {pending}")
        for name, payload in payloads.items():
            _write_new_or_identical(campaign.output / name, payload)
    return {
        "published": publish,
        "expected": manifest["expected"],
        "valid": len(records),
        "sources": len(selected),
        "rejectedReviewFindings": rejected_findings,
        "failures": [],
        "datasetSha256": digest(dataset_bytes),
        "manifestSha256": digest(payloads["manifest.json"]),
        "output": str(campaign.output),
    }
