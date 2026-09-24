"""A document-wide printed tariff field is one shared cargo fact."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    validate_compiled_single_printed_hs_scope,
    validate_single_printed_hs_scope,
)


def _fixture() -> tuple[str, dict, SpanDraft, tuple[str, ...]]:
    raw = "CONTAINERS A, B, C\nHS CODE : 27101999\n"
    code = "27101999"
    start = raw.index(code)
    paths = tuple(f"documentPatch.cargoGroups[{index}].hsCodes[0]" for index in range(3))
    target = {
        "documentPatch": {
            "cargoGroups": [{"hsCodes": [code]} for _ in paths],
        }
    }
    draft = SpanDraft(
        draft_id="printed_hs",
        logical_key="printed_hs",
        render_mode="target_binding",
        value_kind="identifier",
        group_kind="cargo",
        group_key="cargo:shared_hs",
        target_paths=(paths[-1],),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(code),
        source_text=code,
        evidence_origin="accepted_label_evidence",
        render_policy="opaque_identifier",
        rationale="Exact source tariff value.",
    )
    return raw, target, draft, paths


def test_single_printed_hs_requires_all_labeled_groups_to_share_one_owner() -> None:
    raw, target, draft, paths = _fixture()
    with pytest.raises(ValueError, match="shared-value target binding"):
        validate_single_printed_hs_scope(
            raw=raw,
            source_target=target,
            drafts=(draft,),
            semantic_only_target_facts=(),
        )

    validate_single_printed_hs_scope(
        raw=raw,
        source_target=target,
        drafts=(replace(draft, target_paths=paths),),
        semantic_only_target_facts=(),
    )


def test_hs_scope_guard_does_not_infer_equality_without_one_printed_field() -> None:
    raw, target, draft, _paths = _fixture()
    validate_single_printed_hs_scope(
        raw=raw + "HS CODE : 27101999\n",
        source_target=target,
        drafts=(draft,),
        semantic_only_target_facts=(),
    )
    target["documentPatch"]["cargoGroups"][0]["hsCodes"] = ["39202001"]
    validate_single_printed_hs_scope(
        raw=raw,
        source_target=target,
        drafts=(draft,),
        semantic_only_target_facts=(),
    )


def test_synthesis_preflight_rejects_old_one_group_hs_catalog() -> None:
    raw, target, _draft, paths = _fixture()
    owner = SimpleNamespace(
        target_paths=(paths[-1],),
        target_relationship="single_target",
        realization=SimpleNamespace(requires_agent=False),
    )
    template = SimpleNamespace(bindings=(owner,), semantic_only_target_facts=())
    with pytest.raises(ValueError, match="corrected catalog before synthesis"):
        validate_compiled_single_printed_hs_scope(
            raw=raw, source_target=target, template=template
        )
    shared = SimpleNamespace(
        target_paths=paths,
        target_relationship="shared_value_equality",
        realization=SimpleNamespace(requires_agent=False),
    )
    validate_compiled_single_printed_hs_scope(
        raw=raw,
        source_target=target,
        template=SimpleNamespace(bindings=(shared,), semantic_only_target_facts=()),
    )
