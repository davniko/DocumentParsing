from copy import deepcopy

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.template_compiler.descendant import _empty_usage
from document_ocr.synthesis.template_compiler.reviewed_generation import ReviewedGeneration


def fixture():
    context = dict(
        sample_id="sample",
        source_document_id="source",
        source_template_sha256="a" * 64,
        scenario={"new": "facts"},
        fields=[{"key": "party", "paths": ["name"]}],
    )
    review = ReviewedGeneration(
        kind="reviewed_linguistic_correction_v1",
        sample_id="sample",
        source_document_id="source",
        source_template_sha256="a" * 64,
        scenario_sha256=sha256_bytes(canonical_json_bytes(context["scenario"])),
        fields_sha256=sha256_bytes(canonical_json_bytes(context["fields"])),
        original_checkpoint_sha256="b" * 64,
        rationale="Exact reviewed correction",
        output={"party": "New Company"},
    )
    return context, review


def test_review_remains_distinct_from_provider_receipt():
    context, review = fixture()
    review.validate_context(**context)
    receipt = review.receipt(
        completed_at="2026-09-22", usage=_empty_usage().model_dump(mode="json")
    )
    assert receipt["manualReview"]["output"] == receipt["output"]
    assert not receipt["usage"]["requests"]
    assert not receipt["messages"]
    assert "batchProvenance" not in receipt and "hostOnly" not in receipt


@pytest.mark.parametrize(
    "key", ["sample_id", "source_document_id", "source_template_sha256", "scenario", "fields"]
)
def test_review_rejects_changed_context(key):
    context, review = fixture()
    context[key] = [] if key == "fields" else {} if key == "scenario" else "changed"
    with pytest.raises(ValueError, match="frozen scenario"):
        review.validate_context(**context)


@pytest.mark.parametrize(
    "output", [{}, {"party": "OK", "extra": "wrong"}, {"party": ""}, {"party": "a\nb"}]
)
def test_review_rejects_incomplete_or_malformed_output(output):
    context, review = fixture()
    review = review.model_copy(update={"output": output})
    with pytest.raises(ValueError):
        review.validate_context(**context)


def test_review_cannot_claim_model_usage():
    _, review = fixture()
    usage = deepcopy(_empty_usage().model_dump(mode="json"))
    usage["requests"] = 1
    with pytest.raises(ValueError, match="provider usage"):
        review.receipt(completed_at="2026-09-22", usage=usage)


def residual_fixture():
    from document_ocr.synthesis.template_compiler.reviewed_generation import ReviewedResidual

    context = dict(
        sample_id="s",
        source_document_id="d",
        source=b"source",
        template={"binding": "owned"},
        target={"new": "fact"},
        output={"slot": "fact"},
    )
    review = ReviewedResidual(
        kind="reviewed_residual_correction_v1",
        sample_id="s",
        source_document_id="d",
        source_sha256=sha256_bytes(context["source"]),
        template_sha256=sha256_bytes(canonical_json_bytes(context["template"])),
        target_sha256=sha256_bytes(canonical_json_bytes(context["target"])),
        original_checkpoint_sha256="a" * 64,
        rationale="Restore the complete owned value",
        output=context["output"],
    )
    return context, review


@pytest.mark.parametrize(
    "key", ["sample_id", "source_document_id", "source", "template", "target", "output"]
)
def test_residual_review_rejects_changed_facts_and_text(key):
    context, review = residual_fixture()
    review.validate_context(**context)
    context[key] = (
        b"changed"
        if key == "source"
        else {"slot": "changed"}
        if key in {"template", "target", "output"}
        else "changed"
    )
    with pytest.raises(ValueError, match="frozen"):
        review.validate_context(**context)


def test_manual_residual_is_not_a_fake_provider_response():
    from document_ocr.synthesis.template_compiler.descendant import _not_required_stage
    from document_ocr.synthesis.template_compiler.descendant_models import ResidualStageReceipt

    _, review = residual_fixture()
    payload = _not_required_stage(document_id="s", system_prompt_sha256="b" * 64).model_dump(
        mode="json"
    )
    payload.update(
        status="manual_review", output=review.output, manual_review=review.model_dump(mode="json")
    )
    assert (
        ResidualStageReceipt.model_validate_json(canonical_json_bytes(payload)).status
        == "manual_review"
    )
    for change in (
        dict(manual_review=None),
        dict(status="success"),
        dict(output={"slot": "different"}),
        dict(messages=["provider"]),
        dict(usage={**payload["usage"], "requests": 1}),
    ):
        with pytest.raises(ValueError):
            ResidualStageReceipt.model_validate_json(canonical_json_bytes({**payload, **change}))
