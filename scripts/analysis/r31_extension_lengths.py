"""Measure R31 target lengths with the training serializer and pinned tokenizer."""
from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.training.tasks import canonical_json
from scripts.synthesis.extend_r31_boundaries import DATA, REPORT


def main(tokenizer_path: Path):
    started = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    control_ids = {tokenizer.token_to_id(token) for token in ("<pad>", "<eos>", "<bos>")}
    assert None not in control_ids, "training control tokens absent from tokenizer"
    result = {}
    for split in ("real-train", "synthetic-train", "validation"):
        stats = Counter()
        maximum = None
        for index, row in enumerate(map(json.loads, (DATA / f"{split}.jsonl").open())):
            content = tokenizer.encode(canonical_json(row["target"]), add_special_tokens=False).ids
            assert not control_ids.intersection(content), "reserved PAD/EOS/BOS in target"
            length = len(content) + 1  # Training appends one EOS, never BOS.
            stats["documents"] += 1
            stats["targetTokensIncludingEos"] += length
            stats["exceeds3329"] += length > 3329
            stats["exceeds5500"] += length > 5500
            if maximum is None or length > maximum["tokens"]:
                maximum = {"id": row["documentId"], "tokens": length}
            if split == "synthetic-train" and index >= 9400:
                stats["extensionDocuments"] += 1
                stats["extensionTargetTokensIncludingEos"] += length
                stats["extensionMaximum"] = max(stats["extensionMaximum"], length)
        result[split] = {**stats, "maximum": maximum}
    result["tokenizerSha256"] = sha256_file(tokenizer_path)
    result["serialization"] = "training.tasks.canonical_json; no special tokens; append one EOS"
    result["seconds"] = time.perf_counter() - started
    result["peakRssKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_write_json(REPORT / "token-lengths.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    main(parser.parse_args().tokenizer)
