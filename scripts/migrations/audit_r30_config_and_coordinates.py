"""Exact trainer token-length and complete native-coordinate preflight; no model/API calls."""

import hashlib
import json
import resource
import time
from pathlib import Path

from transformers import AutoTokenizer

from document_ocr.spatial_inputs.paddle import pages_from_record, serialize
from document_ocr.training.config import load_training_config
from document_ocr.training.data import _tokenize_batch
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import canonical_json, load_training_task

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic9400_"
    "r30_paddle_centroids_inputonly_v7_e10_compact_eva_a32_r64_local_schedulefree_v1.yaml"
)
OUT = ROOT / "docs/analysis/r30-training-readiness-20261106/config-and-coordinate-preflight.json"


def region_coverage(record):
    pages = pages_from_record(record)
    for mode, field in (("centroid", "positionedText"), ("bbox", "bboxText")):
        if serialize(pages, geometry=mode) != record[field]:
            raise ValueError(f"{record['documentId']}: mismatching {field}")
    total = sum(len(page.regions) for page in pages)
    nonempty = sum(bool(r.text.strip()) for p in pages for r in p.regions)
    return total, nonempty, len(pages)


def main():
    started = time.perf_counter()
    config = load_training_config(CONFIG)
    task = load_training_task(ROOT, config)
    prompt = load_prompt(ROOT, config.prompt, task)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        revision=config.model.tokenizer_revision,
        local_files_only=True,
    )
    summaries = {}
    for source in (*config.dataset.splits.train, *config.dataset.splits.validation):
        path = ROOT / source.path
        batch = {"document_id": [], "input_text": [], "target_text": []}
        summary = {
            "records": 0,
            "pages": 0,
            "region_boxes": 0,
            "nonempty_regions": 0,
            "empty_regions": 0,
            "missing_boxes": 0,
            "missing_centroid_suffixes": 0,
            "missing_bbox_suffixes": 0,
            "max_target_tokens_including_eos": 0,
            "max_input_tokens": 0,
        }

        def flush(batch, summary):
            if not batch["document_id"]:
                return
            encoded = _tokenize_batch(batch, tokenizer=tokenizer, config=config)
            for identity, length, input_length in zip(
                batch["document_id"], encoded["target_length"], encoded["input_length"], strict=True
            ):
                if length > summary["max_target_tokens_including_eos"]:
                    summary["max_target_tokens_including_eos"] = length
                    summary["longest_target_id"] = identity
                summary["max_input_tokens"] = max(summary["max_input_tokens"], input_length)
            for values in batch.values():
                values.clear()

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for encoded_line in stream:
                digest.update(encoded_line)
                record = json.loads(encoded_line)
                total, nonempty, pages = region_coverage(record)
                summary["records"] += 1
                summary["pages"] += pages
                summary["region_boxes"] += total
                summary["nonempty_regions"] += nonempty
                summary["empty_regions"] += total - nonempty
                # Every serialized region has one suffix, including empty OCR text.
                # The serializer above rejects missing/nonfinite/out-of-frame boxes.
                batch["document_id"].append(record["documentId"])
                batch["input_text"].append(prompt.render(record["positionedText"]))
                batch["target_text"].append(canonical_json(task.canonicalize(record["target"])))
                if len(batch["document_id"]) == 32:
                    flush(batch, summary)
        flush(batch, summary)
        if digest.hexdigest() != source.sha256 or summary["records"] != source.records:
            raise ValueError(f"dataset identity mismatch: {path}")
        summary["sha256"] = digest.hexdigest()
        summary["coordinate_coverage"] = 1.0
        summaries[path.stem] = summary
    longest = max(s["max_target_tokens_including_eos"] for s in summaries.values())
    expected = longest + 1 + 128  # One decoder-start token participates in max_length.
    if config.evaluation.generation_max_length != expected:
        raise ValueError(f"generation cap must equal longest target + start + 128: {expected}")
    report = {
        "status": "PASSED",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "splits": summaries,
        "max_label_tokens_including_eos": longest,
        "decoder_start_tokens": 1,
        "generation_headroom_tokens": 128,
        "generation_max_length": expected,
        "training_max_target_length": config.dataset.preprocessing.max_target_length,
        "page_headers_are_not_ocr_regions": True,
        "provider_calls": 0,
        "seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
