"""Apply the two explicit source decisions and prepare their final live controls."""

import json
from pathlib import Path

import yaml
from prepare_multirow_batch import compile_batch
from prepare_multirow_marks_repair import replace_source

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign


def refresh_unchanged_postal_certificate(root, config, before, after):
    """The source's whole-ownership pin changed, but no postal owner changed."""
    if before.contract != after.contract or before.ownership_data.get(
        "related_actors"
    ) != after.ownership_data.get("related_actors"):
        raise ValueError("KOS repair unexpectedly changed a postal component")
    pin = config["independent"]["geography_certificates"]
    value = json.loads(Path(pin["path"]).read_text())
    certificate = value["sources"][after.document_id]
    if certificate["ownership_sha256"] != digest(dict(before.ownership_data)):
        raise ValueError("prior postal ownership certificate differs")
    certificate["ownership_sha256"] = digest(dict(after.ownership_data))
    path = root / "authority-repairs/user-resolutions/geography_certificates.json"
    atomic_publish_json(path, value)
    config["independent"]["geography_certificates"] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    config_path = root / "resolved-base-config-v2.json"
    atomic_publish_json(config_path, config)
    previous = json.loads((root / "batch-38/source-contracts.json").read_text())["sources"]
    compile_batch(
        config_path,
        root / "batch-39",
        [after.document_id[4:12]],
        2,
        group_counts=[1, 2, 3],
        seed=2026110106,
        reviewed={after.document_id: previous[after.document_id]},
    )


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    config = json.loads((root / "product-hs-base-config.json").read_text())
    ids = {sid[4:12]: sid for sid in config["source_ids"]}
    sid = ids["e18486c6"]
    source = _SourceOwnershipCampaign(Path.cwd(), config).blueprint(sid)
    old = Path(config["ownership"])
    text = old.read_text()
    owner = yaml.safe_load(text)["sources"][sid]
    for key, value in (
        ("reviewed:container:0:tare", "2030"),
        ("reviewed:shipment:gross", "7,737.00"),
    ):
        if source.source.count(value) != 1:
            raise ValueError("reviewed KOS numeric occurrence changed")
        owner.setdefault("add", []).append(
            {
                "key": key,
                "kind": "decimal_measurement",
                "paths": [],
                "occurrences": [{"text": value, "occurrence": 1}],
            }
        )
    owner.setdefault("measure_owners", {})["reviewed:shipment:gross"] = {
        "measure": "grossWeight",
        "unit": "kilogram",
        "baseline": "7737.00",
    }
    owner.setdefault("tare_owners", {})["reviewed:container:0:tare"] = [0]
    owner.setdefault("tare_baselines", {})["reviewed:container:0:tare"] = {
        "kg": "2030",
        "sourceSha256": digest(source.source.encode()),
        "reason": "User approved kilograms privately for the actual PDF KOS notation; "
        "retain KOS in rendering and do not add public weight targets.",
    }
    owner.setdefault("required_target_absences", {})[
        "documentPatch.goodsItemDetails[0].grossWeight"
    ] = "User decision: KOS is privately interpreted for synthesis only, not a public unit label."
    directory = root / "authority-repairs/user-resolutions"
    revised = directory / "ownership.yaml"
    atomic_publish_bytes(revised, replace_source(text, sid, owner).encode())
    config["ownership"] = str(revised)
    after = _SourceOwnershipCampaign(Path.cwd(), config).blueprint(sid)
    if after.source != source.source or after.target != source.target:
        raise ValueError("private KOS ownership changed source text or public target")
    numeric_pin = config["independent"]["numeric_certificates"]
    numeric = json.loads(Path(numeric_pin["path"]).read_text())
    numeric["sources"][sid]["numericReview"] = {
        "basis": "User decision: original PDF KOS spelling is retained; privately use kg "
        "for owned shipment gross and container tare; public weight targets remain absent.",
        "disposition": "source_only_measurement_owners",
    }
    numeric_path = directory / "numeric_certificates.json"
    atomic_publish_json(numeric_path, numeric)
    config["independent"]["numeric_certificates"] = {
        "path": str(numeric_path),
        "sha256": sha256_file(numeric_path),
    }
    config_path = root / "resolved-base-config.json"
    atomic_publish_json(config_path, config)
    contracts = {}
    for path in sorted(root.glob("batch-*/source-contracts.json")):
        contracts.update(json.loads(path.read_text())["sources"])
    compile_batch(
        config_path,
        root / "batch-38",
        ["e18486c6", "2763b891"],
        2,
        group_counts=[1, 2, 3],
        seed=2026110106,
        reviewed={ids[p]: contracts[ids[p]] for p in ("e18486c6", "2763b891")},
        source_group_counts={ids["2763b891"]: [1, 1, 1]},
    )
    atomic_publish_json(
        directory / "review.json",
        {
            "status": "USER_DECISIONS_APPLIED_AWAITING_FRESH_SYNTHESIS",
            "sourceIds": [sid, ids["2763b891"]],
            "decisions": {
                sid: "Use kilograms privately; preserve KOS and target absence.",
                ids["2763b891"]: "Keep this template single-row.",
            },
            "beforeOwnership": {"path": str(old), "sha256": sha256_file(old)},
            "afterOwnership": {"path": str(revised), "sha256": sha256_file(revised)},
            "sourceAndTargetUnchanged": True,
        },
    )
    refresh_unchanged_postal_certificate(root, config, source, after)


if __name__ == "__main__":
    main()
