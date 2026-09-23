import pytest

from document_ocr.synthesis.template_compiler import host


@pytest.mark.parametrize(
    "caption", ["ORIGINALS TO BE RELEASED AT", "PORT OF LOADING", "VOYAGE NO."]
)
def test_empty_terminal_field_does_not_own_the_following_form_heading(caption):
    raw = "LOADING PIER/TERMINAL\n" + caption + "\nSingapore"
    assert host.normalize_explicit_loading_terminal_locality(raw=raw, drafts=()) == ()


def test_an_actual_terminal_value_is_still_owned():
    raw = "LOADING PIER/TERMINAL\nMAHER TERMINAL\nPORT OF LOADING\nNEW YORK"
    result = host.normalize_explicit_loading_terminal_locality(raw=raw, drafts=())
    assert len(result) == 1
    assert result[0].source_text == "MAHER TERMINAL"
    assert result[0].group_key == "route:loading_terminal"


def test_previously_misclassified_caption_becomes_literal_and_stays_literal():
    text = "ORIGINALS TO BE RELEASED AT"
    raw = "LOADING PIER/TERMINAL\n" + text + "\nSingapore"
    draft = host.SpanDraft(
        draft_id="caption",
        logical_key="agent:route:loading_terminal:caption",
        render_mode="deterministic_auxiliary",
        value_kind="location",
        group_kind="route",
        group_key="route:loading_terminal",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=21,
        char_end=21 + len(text),
        source_text=text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Old classification.",
    )
    fixed = host.normalize_explicit_loading_terminal_locality(raw=raw, drafts=(draft,))
    assert fixed[0].render_mode == "literal_static"
    assert fixed[0].source_text == text
    assert host.normalize_explicit_loading_terminal_locality(raw=raw, drafts=fixed) == fixed
