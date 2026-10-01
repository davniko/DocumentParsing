"""The explicit rare-cargo hold cannot become an implicit source-copy path."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.template_compiler.cargo_frozen_policy import (
    CargoFrozenReview,
    CargoFrozenSourcePolicy,
    cargo_graph_sha256,
    load_case_policy,
    require_frozen_cargo,
    require_plan_mode,
)


def _target() -> dict:
    return {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "used ROPA TYPE RM5 with accessories",
                    "hsCodes": ["84336000"],
                    "marksAndNumbers": ["8K1541"],
                }
            ],
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 1}
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "allocations": [
                        {"containerNumber": "ABCU1234567", "packageId": "p1", "packageQuantity": 1}
                    ],
                }
            ],
            "containers": [
                {"containerNumber": "ABCU1234567", "sealNumbers": ["S123"]}
            ],
        }
    }


def _policy(target: dict) -> CargoFrozenSourcePolicy:
    return CargoFrozenSourcePolicy(
        schema_version=1,
        mode="cargo_frozen_source",
        document_id="doc_test",
        source_sha256="a" * 64,
        source_target_sha256="b" * 64,
        template_sha256="c" * 64,
        cargo_graph_sha256=cargo_graph_sha256(target),
        review_sha256="d" * 64,
    )


def test_frozen_cargo_allows_only_equipment_identity_replacements() -> None:
    original = _target()
    updated = deepcopy(original)
    updated["documentPatch"]["containers"][0]["containerNumber"] = "XYZU7654321"
    updated["documentPatch"]["containers"][0]["sealNumbers"][0] = "S987"
    updated["documentPatch"]["cargoAllocationGroups"][0]["allocations"][0][
        "containerNumber"
    ] = "XYZU7654321"
    require_frozen_cargo(original, updated, _policy(original))

    for mutation in ("description", "marksAndNumbers", "hsCodes"):
        bad = deepcopy(updated)
        bad["documentPatch"]["cargoGroups"][0][mutation] = "unrelated cargo"
        with pytest.raises(ValueError, match="changed a source-proven cargo fact"):
            require_frozen_cargo(original, bad, _policy(original))
    bad = deepcopy(updated)
    bad["documentPatch"]["cargoPackages"][0]["quantity"] = 2
    with pytest.raises(ValueError, match="changed a source-proven cargo fact"):
        require_frozen_cargo(original, bad, _policy(original))
    bad = deepcopy(updated)
    bad["documentPatch"]["containers"][0]["sealNumbers"] = []
    with pytest.raises(ValueError, match="changed a source-proven cargo fact"):
        require_frozen_cargo(original, bad, _policy(original))


def test_frozen_cargo_requires_explicit_sample_plan_mode() -> None:
    policy = _policy(_target())
    require_plan_mode(
        source_document_id="doc_test", scenario_mode="cargo_frozen_source", policy=policy
    )
    require_plan_mode(source_document_id="doc_normal", scenario_mode="sampled_route", policy=None)
    with pytest.raises(ValueError, match="sample plan cargo_frozen_source mode differs"):
        require_plan_mode(source_document_id="doc_test", scenario_mode=None, policy=policy)
    with pytest.raises(ValueError, match="sample plan cargo_frozen_source mode differs"):
        require_plan_mode(
            source_document_id="doc_normal", scenario_mode="cargo_frozen_source", policy=None
        )


def test_frozen_policy_requires_exact_source_and_review(tmp_path: Path) -> None:
    source = b"used ROPA TYPE RM5\nSERIALNO. : 8K1541\n"
    template = b"compiled template bytes"
    target = _target()
    facts = {
        "document_id": "doc_test",
        "source_sha256": sha256_bytes(source),
        "source_target_sha256": sha256_bytes(canonical_json_bytes(target)),
        "template_sha256": sha256_bytes(template),
        "cargo_graph_sha256": cargo_graph_sha256(target),
    }
    review = CargoFrozenReview(
        schema_version=1,
        decision="pass",
        rationale="Printed source supports one fixed machine and serial without a second item.",
        source_evidence=("used ROPA TYPE RM5", "SERIALNO. : 8K1541"),
        **facts,
    )
    review_bytes = canonical_json_bytes(review.model_dump(mode="json"))
    policy = CargoFrozenSourcePolicy(
        schema_version=1,
        mode="cargo_frozen_source",
        review_sha256=sha256_bytes(review_bytes),
        **facts,
    )
    (tmp_path / "cargo-frozen-source-review.json").write_bytes(review_bytes)
    (tmp_path / "cargo-frozen-source-policy.json").write_bytes(
        canonical_json_bytes(policy.model_dump(mode="json"))
    )
    assert load_case_policy(
        tmp_path,
        document_id="doc_test",
        source=source,
        source_target=target,
        latest_target=target,
        template_bytes=template,
    ) == policy
    with pytest.raises(ValueError, match="differs from the exact source case"):
        load_case_policy(
            tmp_path,
            document_id="doc_test",
            source=source.replace(b"RM5", b"RM6"),
            source_target=target,
            latest_target=target,
            template_bytes=template,
        )

    # The source-only fixed-binding list is itself reviewed and pinned, not an
    # implicit permission to preserve arbitrary generated auxiliary text.
    mismatched = review.model_copy(
        update={"fixed_source_binding_keys": ("agent:cargo:unreviewed",)}
    )
    mismatched_bytes = canonical_json_bytes(mismatched.model_dump(mode="json"))
    (tmp_path / "cargo-frozen-source-review.json").write_bytes(mismatched_bytes)
    (tmp_path / "cargo-frozen-source-policy.json").write_bytes(
        canonical_json_bytes(
            policy.model_copy(update={"review_sha256": sha256_bytes(mismatched_bytes)}).model_dump(
                mode="json"
            )
        )
    )
    with pytest.raises(ValueError, match="fixed source binding list differs"):
        load_case_policy(
            tmp_path,
            document_id="doc_test",
            source=source,
            source_target=target,
            latest_target=target,
            template_bytes=template,
        )
