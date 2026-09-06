from __future__ import annotations

import json
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.run_safety import (
    BehaviorEnvironmentFingerprint,
    FingerprintMismatchError,
    IdentifierCandidateContext,
    IdentifierReservationRequest,
    IdentifierSpaceError,
    LeakageError,
    StagedArtifactRun,
    StagedRunError,
    TrainOnlySourceScope,
    build_behavior_environment_fingerprint,
    identifier_corpus_sha256,
    reserve_global_identifiers,
    verify_behavior_environment_fingerprint,
)


def _hash(value: str) -> str:
    return sha256_bytes(value.encode())


def _scope(*, crossed_template: bool = False) -> TrainOnlySourceScope:
    templates = {
        "train-a": "template-shared" if crossed_template else "template-a",
        "train-b": "template-b",
        "eval-a": "template-shared" if crossed_template else "template-eval",
    }
    return TrainOnlySourceScope(
        corpus_document_ids=("eval-a", "train-b", "train-a"),
        fit_document_ids=("train-b", "train-a"),
        template_by_document=templates,
        source_sha256_by_document={document_id: _hash(document_id) for document_id in templates},
        fit_split="train",
    )


def test_train_scope_receipts_exact_donor_payload_and_audits_without_leakage() -> None:
    scope = _scope()
    payload = {"train-b": {"quantity": 12, "unit": "KGM"}}

    receipt = scope.create_donor_receipt(
        source_document_id="train-a",
        donor_payload_by_document=payload,
        purpose="cargo tuple resampling",
        field_paths=("cargo[].grossWeight", "cargo[].packages[].quantity"),
    )
    scope.validate_donor_receipt(receipt, donor_payload_by_document=payload)
    audit = scope.audit_donor_receipts(((receipt, payload),))

    assert receipt.source_sha256 == _hash("train-a")
    assert receipt.donors[0].document_id == "train-b"
    assert receipt.donors[0].selected_value_sha256 == sha256_bytes(
        canonical_json_bytes(payload["train-b"])
    )
    assert audit.operation_count == 1
    assert audit.donor_count == 1
    assert audit.leakage_detected is False


def test_train_scope_rejects_eval_unknown_self_and_cross_split_template_donors() -> None:
    scope = _scope()
    kwargs = {
        "source_document_id": "train-a",
        "purpose": "test",
        "field_paths": ("cargo[].quantity",),
    }

    with pytest.raises(LeakageError, match="outside fit split"):
        scope.create_donor_receipt(
            **kwargs,
            donor_payload_by_document={"eval-a": {"quantity": 2}},
        )
    with pytest.raises(LeakageError, match="outside the complete corpus"):
        scope.create_donor_receipt(
            **kwargs,
            donor_payload_by_document={"unknown": {"quantity": 2}},
        )
    with pytest.raises(LeakageError, match="own donor"):
        scope.create_donor_receipt(
            **kwargs,
            donor_payload_by_document={"train-a": {"quantity": 2}},
        )

    crossed = _scope(crossed_template=True)
    with pytest.raises(LeakageError, match="template crosses"):
        crossed.create_donor_receipt(
            source_document_id="train-a",
            donor_payload_by_document={"train-b": {"quantity": 2}},
            purpose="test",
            field_paths=("cargo[].quantity",),
        )
    assert crossed.isolated_fit_document_ids == frozenset({"train-b"})


def test_train_scope_rejects_modified_payload_receipt_and_duplicate_operations() -> None:
    scope = _scope()
    payload = {"train-b": [10, "KGM"]}
    receipt = scope.create_donor_receipt(
        source_document_id="train-a",
        donor_payload_by_document=payload,
        purpose="mass proposal",
        field_paths=("cargo[].grossWeight",),
    )

    with pytest.raises(LeakageError, match="does not match"):
        scope.validate_donor_receipt(
            receipt,
            donor_payload_by_document={"train-b": [11, "KGM"]},
        )
    with pytest.raises(LeakageError, match="duplicate donor operation"):
        scope.audit_donor_receipts(((receipt, payload), (receipt, payload)))


def test_fit_row_guard_rejects_cross_split_model_fit() -> None:
    scope = _scope()

    assert scope.assert_fit_rows(("train-b", "train-a"), purpose="sdv cargo view") == (
        "train-a",
        "train-b",
    )
    with pytest.raises(LeakageError, match="outside fit split"):
        scope.assert_fit_rows(("train-a", "eval-a"), purpose="sdv cargo view")


def test_statistical_fit_receipt_pins_exact_train_only_projected_rows() -> None:
    scope = _scope()
    rows = {
        "train-a": {"package": "PALLET", "quantity": 10},
        "train-b": {"package": "DRUM", "quantity": 20},
    }
    receipt = scope.create_statistical_fit_receipt(
        view_name="cargo_package_weight_v1",
        row_payload_by_document=rows,
        purpose="sdv candidate fit",
        field_paths=("package", "quantity"),
    )

    scope.validate_statistical_fit_receipt(receipt, row_payload_by_document=rows)
    assert len(receipt.inputs) == 2
    assert [row.document_id for row in receipt.inputs] == ["train-a", "train-b"]
    with pytest.raises(LeakageError, match="does not match"):
        scope.validate_statistical_fit_receipt(
            receipt,
            row_payload_by_document={**rows, "train-b": {"package": "DRUM", "quantity": 21}},
        )
    with pytest.raises(LeakageError, match="outside fit split"):
        scope.create_statistical_fit_receipt(
            view_name="cargo_package_weight_v1",
            row_payload_by_document={**rows, "eval-a": {"package": "BOX", "quantity": 1}},
            purpose="sdv candidate fit",
            field_paths=("package", "quantity"),
        )


def _identifier_factory(context: IdentifierCandidateContext) -> str:
    alphabet = string.ascii_uppercase + string.digits
    value = int.from_bytes(context.entropy[:10], "big")
    characters = []
    for _ in range(14):
        value, remainder = divmod(value, len(alphabet))
        characters.append(alphabet[remainder])
    return f"SYN-{''.join(characters)}"


def _reserve(requests: list[IdentifierReservationRequest]):
    real = ("REAL-0001", "REAL-0002", "real-0002")
    count, _unique, digest = identifier_corpus_sha256(real, canonicalize=str.casefold)
    return reserve_global_identifiers(
        real_identifiers=real,
        requests=requests,
        seed="frozen-seed",
        namespace="test-identifiers-v1",
        canonicalize=str.casefold,
        canonicalizer_name="unicode_casefold_v1",
        candidate_factory=_identifier_factory,
        expected_real_identifier_count=count,
        expected_real_corpus_sha256=digest,
    )


def test_global_identifier_reservation_is_unique_at_10k_scale_and_order_independent() -> None:
    requests = [
        IdentifierReservationRequest.model_validate(
            {"request_key": f"document-{index:05d}/seal-0", "identifier_kind": "seal"},
            strict=True,
        )
        for index in range(10_000)
    ]
    forward = _reserve(requests)
    shuffled = list(requests)
    random.Random(9127).shuffle(shuffled)
    reverse = _reserve(shuffled)

    identifiers = [row.canonical_identifier for row in forward.reservations]
    assert len(identifiers) == len(set(identifiers)) == 10_000
    assert not {"real-0001", "real-0002"} & set(identifiers)
    assert forward == reverse
    assert forward.request_count == 10_000


def test_identifier_reservation_matches_across_concurrent_invocations() -> None:
    requests = [
        IdentifierReservationRequest.model_validate(
            {"request_key": f"request-{index:04d}", "identifier_kind": "reference"},
            strict=True,
        )
        for index in range(1_000)
    ]
    variants = [requests, list(reversed(requests)), requests[::2] + requests[1::2]]

    with ThreadPoolExecutor(max_workers=3) as executor:
        outputs = list(executor.map(_reserve, variants))

    assert outputs[0] == outputs[1] == outputs[2]


def test_identifier_reservation_rejects_incomplete_corpus_and_nondeterministic_factory() -> None:
    request = IdentifierReservationRequest.model_validate(
        {"request_key": "one", "identifier_kind": "seal"}, strict=True
    )
    _count, _unique, digest = identifier_corpus_sha256(("real",), canonicalize=str.casefold)
    with pytest.raises(IdentifierSpaceError, match="count mismatch"):
        reserve_global_identifiers(
            real_identifiers=("real",),
            requests=(request,),
            seed="seed",
            namespace="namespace",
            canonicalize=str.casefold,
            canonicalizer_name="casefold",
            candidate_factory=_identifier_factory,
            expected_real_identifier_count=2,
            expected_real_corpus_sha256=digest,
        )

    calls = 0

    def stateful(_context: IdentifierCandidateContext) -> str:
        nonlocal calls
        calls += 1
        return f"candidate-{calls}"

    with pytest.raises(IdentifierSpaceError, match="stateful or nondeterministic"):
        reserve_global_identifiers(
            real_identifiers=("real",),
            requests=(request,),
            seed="seed",
            namespace="namespace",
            canonicalize=str.casefold,
            canonicalizer_name="casefold",
            candidate_factory=stateful,
            expected_real_identifier_count=1,
            expected_real_corpus_sha256=digest,
        )


def test_identifier_reservation_retries_real_and_generated_collisions() -> None:
    requests = tuple(
        IdentifierReservationRequest.model_validate(
            {"request_key": key, "identifier_kind": "reference"}, strict=True
        )
        for key in ("a", "b")
    )
    real = ("occupied",)
    count, _unique, digest = identifier_corpus_sha256(real, canonicalize=str.casefold)

    def colliding(context: IdentifierCandidateContext) -> str:
        if context.attempt == 0:
            return "OCCUPIED"
        if context.attempt == 1:
            return "SHARED-FIRST-FREE"
        return f"FREE-{context.request_key}"

    result = reserve_global_identifiers(
        real_identifiers=real,
        requests=requests,
        seed="seed",
        namespace="collision-test",
        canonicalize=str.casefold,
        canonicalizer_name="casefold",
        candidate_factory=colliding,
        expected_real_identifier_count=count,
        expected_real_corpus_sha256=digest,
    )

    assert [row.identifier for row in result.reservations] == [
        "SHARED-FIRST-FREE",
        "FREE-b",
    ]
    assert [row.collision_attempts for row in result.reservations] == [1, 2]


def test_behavior_fingerprint_covers_code_lock_image_runtime_and_environment(
    tmp_path: Path,
) -> None:
    (tmp_path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock-version = 1\n", encoding="utf-8")
    digest = "sha256:" + "a" * 64

    fingerprint = build_behavior_environment_fingerprint(
        project_root=tmp_path,
        behavior_files={"synthesis-source": Path("source.py")},
        lock_files={"uv-lock": Path("uv.lock")},
        image_reference="document-ocr@sha256:pinned",
        image_digest=digest,
        environment_identity={"CUDA_VERSION": "12.9", "SDV_VERSION": "1.38.2"},
    )
    serialized = BehaviorEnvironmentFingerprint.model_validate_json(
        fingerprint.model_dump_json(), strict=True
    )

    assert serialized == fingerprint
    assert fingerprint.behavior_files[0].sha256 == _hash("VALUE = 1\n")
    assert fingerprint.lock_files[0].sha256 == _hash("lock-version = 1\n")
    verify_behavior_environment_fingerprint(fingerprint, project_root=tmp_path)

    (tmp_path / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(FingerprintMismatchError, match="mismatch"):
        verify_behavior_environment_fingerprint(fingerprint, project_root=tmp_path)


def test_behavior_fingerprint_rejects_secrets_symlinks_and_floating_image(
    tmp_path: Path,
) -> None:
    (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock\n", encoding="utf-8")
    (tmp_path / "source-link.py").symlink_to(tmp_path / "source.py")
    (tmp_path / "real-directory").mkdir()
    (tmp_path / "real-directory/nested.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "linked-directory").symlink_to(tmp_path / "real-directory")
    common = {
        "project_root": tmp_path,
        "behavior_files": {"source": Path("source.py")},
        "lock_files": {"lock": Path("uv.lock")},
        "image_reference": "image:tag",
        "image_digest": "sha256:" + "b" * 64,
    }
    with pytest.raises(ValueError, match="secret-bearing"):
        build_behavior_environment_fingerprint(
            **common,
            environment_identity={"OPENAI_API_KEY": "must-not-be-hashed"},
        )
    with pytest.raises(ValueError, match="symbolic link"):
        build_behavior_environment_fingerprint(
            **{**common, "behavior_files": {"source": Path("source-link.py")}},
            environment_identity={},
        )
    with pytest.raises(ValueError, match="symbolic links"):
        build_behavior_environment_fingerprint(
            **{
                **common,
                "behavior_files": {"source": Path("linked-directory/nested.py")},
            },
            environment_identity={},
        )
    with pytest.raises(ValueError, match="immutable"):
        build_behavior_environment_fingerprint(
            **{**common, "image_digest": "latest"},
            environment_identity={},
        )


def test_staged_run_recovers_interruption_and_commits_exact_inventory(tmp_path: Path) -> None:
    transaction = _hash("transaction")
    first = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="synthesis-run",
        transaction_sha256=transaction,
    )
    first.publish_json("records/one.json", {"value": 1})
    interrupted = first.stage_root / "records/.two.deadbeef.tmp"
    interrupted.write_bytes(b"partial")

    resumed = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="synthesis-run",
        transaction_sha256=transaction,
    )
    assert resumed.recover_interrupted_temporary_files() == ("records/.two.deadbeef.tmp",)
    assert resumed.publish_json("records/one.json", {"value": 1}) is False
    assert resumed.publish_bytes("records/two.txt", b"complete\n") is True
    result = resumed.commit(
        expected_artifacts=("records/two.txt", "records/one.json"),
        metadata={"records": 2, "status": "complete"},
    )

    assert result.created is True
    assert not resumed.stage_root.exists()
    assert (resumed.final_root / "records/two.txt").read_bytes() == b"complete\n"
    assert resumed.validate_committed_run() == result.receipt
    assert [row.relative_path for row in result.receipt.artifacts] == [
        "records/one.json",
        "records/two.txt",
    ]


def test_staged_run_is_idempotent_after_commit_and_never_extends_final_tree(
    tmp_path: Path,
) -> None:
    transaction = _hash("transaction")
    run = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="run",
        transaction_sha256=transaction,
    )
    run.publish_bytes("artifact.bin", b"fixed")
    created = run.commit(expected_artifacts=("artifact.bin",), metadata={"records": 1})

    resumed = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="run",
        transaction_sha256=transaction,
    )
    assert resumed.completed is True
    assert resumed.publish_bytes("artifact.bin", b"fixed") is False
    repeated = resumed.commit(expected_artifacts=("artifact.bin",), metadata={"records": 1})
    assert repeated.created is False
    assert repeated.receipt == created.receipt
    with pytest.raises(StagedRunError, match="committed run"):
        resumed.publish_bytes("late.bin", b"must not be added")
    assert not (resumed.final_root / "late.bin").exists()


def test_staged_run_resumes_sealed_stage_after_final_rename_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = _hash("transaction")
    run = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="rename-recovery",
        transaction_sha256=transaction,
    )
    run.publish_bytes("artifact.bin", b"fixed")
    real_rename = os.rename

    def deny_final_rename(source: str | bytes | Path, target: str | bytes | Path) -> None:
        if Path(source) == run.stage_root and Path(target) == run.final_root:
            raise PermissionError("simulated external handle")
        real_rename(source, target)

    monkeypatch.setattr(os, "rename", deny_final_rename)
    with pytest.raises(PermissionError, match="simulated external handle"):
        run.commit(expected_artifacts=("artifact.bin",), metadata={"records": 1})
    assert (run.stage_root / "_COMMIT.json").is_file()
    assert not run.final_root.exists()

    monkeypatch.setattr(os, "rename", real_rename)
    resumed = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="rename-recovery",
        transaction_sha256=transaction,
    )

    assert resumed.completed is True
    assert not resumed.stage_root.exists()
    assert (resumed.final_root / "artifact.bin").read_bytes() == b"fixed"
    assert resumed.validate_committed_run().metadata == {"records": 1}


def test_concurrent_identical_stage_commits_publish_exactly_once(tmp_path: Path) -> None:
    transaction = _hash("transaction")
    runs = tuple(
        StagedArtifactRun(
            output_parent=tmp_path,
            run_name="concurrent",
            transaction_sha256=transaction,
        )
        for _ in range(2)
    )
    for run in runs:
        run.publish_bytes("artifact.bin", b"same")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda run: run.commit(
                    expected_artifacts=("artifact.bin",), metadata={"records": 1}
                ),
                runs,
            )
        )

    assert sum(result.created for result in results) == 1
    assert results[0].receipt == results[1].receipt
    assert (tmp_path / "concurrent/artifact.bin").read_bytes() == b"same"


def test_staged_run_refuses_partial_extra_conflicting_and_tampered_runs(
    tmp_path: Path,
) -> None:
    transaction = _hash("transaction")
    run = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="run",
        transaction_sha256=transaction,
    )
    run.publish_bytes("one.bin", b"one")
    with pytest.raises(StagedRunError, match="inventory mismatch"):
        run.commit(expected_artifacts=("one.bin", "two.bin"), metadata={})
    assert run.stage_root.exists()
    assert not run.final_root.exists()
    with pytest.raises(StagedRunError, match="conflicts"):
        run.publish_bytes("one.bin", b"changed")

    run.publish_bytes("two.bin", b"two")
    run.commit(expected_artifacts=("one.bin", "two.bin"), metadata={})
    (run.final_root / "one.bin").write_bytes(b"tampered")
    with pytest.raises(StagedRunError, match="inventory differs"):
        StagedArtifactRun(
            output_parent=tmp_path,
            run_name="run",
            transaction_sha256=transaction,
        )


def test_commit_receipt_contains_only_deterministic_json(tmp_path: Path) -> None:
    run = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="run",
        transaction_sha256=_hash("transaction"),
    )
    run.publish_json("data.json", {"z": 2, "a": 1})
    result = run.commit(expected_artifacts=("data.json",), metadata={"seed": 42})

    persisted = json.loads((run.final_root / "_COMMIT.json").read_bytes())
    assert persisted == result.receipt.model_dump(mode="json")
    assert "timestamp" not in canonical_json_bytes(persisted).decode()
