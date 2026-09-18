from __future__ import annotations

import re
from pathlib import Path

from document_ocr.hashing import canonical_json_sha256
from document_ocr.synthesis.template_compiler.pipeline import load_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = (
    _PROJECT_ROOT
    / "configs/synthesis/production/mpci_bl_template_compilation_remaining1420_v1_luna.yaml"
)
_HALF_A_CONFIG = (
    _PROJECT_ROOT
    / "configs/synthesis/production/mpci_bl_template_compilation_remaining710_a_v1_luna.yaml"
)
_HALF_B_CONFIG = (
    _PROJECT_ROOT
    / "configs/synthesis/production/mpci_bl_template_compilation_remaining710_b_v1_luna.yaml"
)
_HALF_B_PREFIX180_CONFIG = (
    _PROJECT_ROOT / "configs/synthesis/production/"
    "mpci_bl_template_compilation_remaining710_b_prefix180_v1_luna.yaml"
)
_HALF_B_FINAL190_CONFIG = (
    _PROJECT_ROOT / "configs/synthesis/production/"
    "mpci_bl_template_compilation_b_ordinals521_710_v1_luna.yaml"
)


def test_remaining_production_template_config_is_exact_and_launch_locked() -> None:
    config = load_config(_CONFIG)

    assert config.run_name == "mpci-bl-template-compilation-remaining1420-v1-luna-high"
    assert config.output_dir == "artifacts/kie-synthesis-production/template-base"
    assert config.workflow.documents == 1420
    assert config.workflow.max_concurrent_documents == 16
    assert config.workflow.max_concurrent_requests == 16
    assert config.workflow.provider_launch_authorized is False
    assert config.workflow.publish_training_records is False
    assert config.resume_from is None
    assert config.pinned_document_ids == ()

    exclusions = config.excluded_document_ids
    assert len(exclusions) == 230
    assert len(set(exclusions)) == 230
    assert exclusions == tuple(sorted(exclusions))
    assert all(re.fullmatch(r"doc_[0-9a-f]{64}", document_id) for document_id in exclusions)
    assert canonical_json_sha256(list(exclusions)) == (
        "d85c0a60b127efece0974ccffc897357a43d0ab193b30a7f9c25a71dc10ba4a4"
    )

    assert config.inputs.source_corpus.records == 2174
    assert config.inputs.source_corpus.sha256 == (
        "90d01f11ddf8cbca2bcc3f660c6990991c616cdaa70e8d549a565367ac570b1f"
    )
    assert config.compiler_provider.model == "gpt-5.6-luna"
    assert config.compiler_provider.reasoning_effort == "high"


def test_remaining_production_split_is_exact_balanced_and_launch_locked() -> None:
    half_a = load_config(_HALF_A_CONFIG)
    half_b = load_config(_HALF_B_CONFIG)

    assert half_a.run_name == "mpci-bl-template-compilation-remaining710-a-v1-luna-high"
    assert half_b.run_name == "mpci-bl-template-compilation-remaining710-b-v1-luna-high"
    for config in (half_a, half_b):
        assert config.output_dir == (
            "artifacts/kie-synthesis-production/template-base/remaining1420-split"
        )
        assert config.workflow.documents == 710
        assert config.workflow.max_concurrent_documents == 16
        assert config.workflow.max_concurrent_requests == 16
        assert config.workflow.provider_launch_authorized is False
        assert config.workflow.publish_training_records is False
        assert config.excluded_document_ids == ()
        assert len(config.pinned_document_ids) == 710
        assert len(set(config.pinned_document_ids)) == 710
        assert all(
            re.fullmatch(r"doc_[0-9a-f]{64}", document_id)
            for document_id in config.pinned_document_ids
        )

    assert half_a.resume_from is not None
    assert half_a.resume_from.path == (
        "artifacts/kie-synthesis-production/template-base/remaining1420-split/recovery/"
        "mpci-bl-template-compilation-remaining710-a-prefix150-recovery-v1"
    )
    assert half_a.resume_from.commit_sha256 == (
        "08a6ce45d59d4a2f82e714081a2ee118b535ff364f782e01741b9bcc5a57e22d"
    )
    assert half_a.resume_from.allow_prompt_change is False
    assert half_b.resume_from is None

    assert canonical_json_sha256(list(half_a.pinned_document_ids)) == (
        "ca99f213a1d7405e38a93c1334ebb336cb85da776962e8799a7a53af1a8f739f"
    )
    assert canonical_json_sha256(list(half_b.pinned_document_ids)) == (
        "3e3596781251ee0c4ce252c5f65098fbff37fccbe558b5045a2196cceffdf31d"
    )
    assert not set(half_a.pinned_document_ids) & set(half_b.pinned_document_ids)
    assert (
        canonical_json_sha256(sorted((*half_a.pinned_document_ids, *half_b.pinned_document_ids)))
        == "3b2c03f2b372ecd91e45d4398a42f083fbb3d35c58a912c518791dc829cf027a"
    )


def test_budgeted_half_b_prefix_is_exact_non_overlapping_and_launch_locked() -> None:
    half_a = load_config(_HALF_A_CONFIG)
    half_b = load_config(_HALF_B_CONFIG)
    prefix = load_config(_HALF_B_PREFIX180_CONFIG)

    assert prefix.workflow.documents == 180
    assert prefix.workflow.max_concurrent_documents == 16
    assert prefix.workflow.max_concurrent_requests == 16
    assert prefix.workflow.provider_launch_authorized is False
    assert prefix.resume_from is None
    assert prefix.pinned_document_ids == half_b.pinned_document_ids[:180]
    assert not set(prefix.pinned_document_ids) & set(half_a.pinned_document_ids)
    assert canonical_json_sha256(list(prefix.pinned_document_ids)) == (
        "47efdb2b21c0e44920ea9126865a6487535fd09048b642ae378ad87be8cbca84"
    )


def test_final_half_b_shard_is_exact_tail_non_overlapping_and_launch_locked() -> None:
    half_a = load_config(_HALF_A_CONFIG)
    half_b = load_config(_HALF_B_CONFIG)
    final = load_config(_HALF_B_FINAL190_CONFIG)

    assert final.run_name == "mpci-bl-template-compilation-b-ordinals521-710-v1-luna-high"
    assert final.output_dir == (
        "artifacts/kie-synthesis-production/template-base/remaining1420-split/budgeted-batches"
    )
    assert final.workflow.documents == 190
    assert final.workflow.max_concurrent_documents == 16
    assert final.workflow.max_concurrent_requests == 16
    assert final.workflow.provider_launch_authorized is False
    assert final.workflow.publish_training_records is False
    assert final.resume_from is None
    assert final.excluded_document_ids == ()
    assert final.pinned_document_ids == half_b.pinned_document_ids[520:710]
    assert len(set(final.pinned_document_ids)) == 190
    assert not set(final.pinned_document_ids) & set(half_a.pinned_document_ids)
    assert canonical_json_sha256(list(final.pinned_document_ids)) == (
        "6ae6c10ba4712c34ad906a15c03d5f3ce2a6c234a64f5b09b1d824c3fc7e9ef1"
    )
