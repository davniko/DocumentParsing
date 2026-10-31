"""Bind the reviewed repeated machinery passages to their printed HS slots."""

import json
from pathlib import Path

import yaml
from prepare_multirow_batch import compile_batch
from prepare_multirow_marks_repair import replace_source

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    base = root / "completion-base-config.json"
    config = json.loads(base.read_text())
    sid = next(s for s in config["source_ids"] if s.startswith("doc_7b12b905"))
    contract = json.loads((Path(config["source_contracts"]) / sid / "contract.json").read_text())
    products = {
        v["key"]: v["value"] for v in contract["contract"]["variables"] if v["kind"] == "product"
    }
    if products != {
        "product_0": "ROTATION PRINTING MACHINE AND OVEN",
        "product_1": "USED",
        "product_2": "FLAME LAMINATION MACHINE (230 CM)",
        "product_3": "AUTOMATION SEWING MACHINE AND ACCESSORIES",
    }:
        raise ValueError("reviewed product owners changed")
    path = Path(config["ownership"])
    content = path.read_text()
    owner = yaml.safe_load(content)["sources"][sid]
    before = sha256_file(path)
    # The first two product passages precede a two-code list. The next two
    # passages both repeat the third HS code, on each of three container blocks.
    owner["product_hs_indices"] = {
        "product_0": 0,
        "product_1": 1,
        "product_2": 2,
        "product_3": 2,
    }
    repaired = root / "authority-repairs/product-hs/ownership.yaml"
    atomic_publish_bytes(repaired, replace_source(content, sid, owner).encode())
    atomic_publish_json(
        root / "authority-repairs/product-hs/review.json",
        {
            "status": "COMPILED_AWAITING_SYNTHESIS",
            "sourceId": sid,
            "beforeOwnershipSha256": before,
            "afterOwnershipSha256": sha256_file(repaired),
            "mapping": owner["product_hs_indices"],
            "sourceAndTargetUnchanged": True,
            "reason": "Preserve the printed two-code first passage and repeated third-code "
            "later passages. Reuse existing product-to-HS instruction ownership, "
            "not guessed joins.",
        },
    )
    config["ownership"] = str(repaired)
    revised = root / "product-hs-base-config.json"
    atomic_publish_json(revised, config)
    contracts = json.loads((root / "batch-35/source-contracts.json").read_text())["sources"]
    compile_batch(
        revised,
        root / "batch-37",
        [sid[4:12]],
        2,
        group_counts=[1, 2, 3],
        reviewed={sid: contracts[sid]},
        seed=2026110105,
    )


if __name__ == "__main__":
    main()
