"""Native Qwen thinking-boundary handling for rewards and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ThinkingMode = Literal["disabled", "enabled"]

THINKING_OPEN_TAG = "<think>"
THINKING_CLOSE_TAG = "</think>"
THINKING_PROMPT_SUFFIX = f"{THINKING_OPEN_TAG}\n"


@dataclass(frozen=True, slots=True)
class CompletionParts:
    raw_text: str
    reasoning_trace: str | None
    final_answer: str | None
    boundary_valid: bool
    boundary_error: str | None


def split_completion(value: str, *, thinking: ThinkingMode) -> CompletionParts:
    """Separate Qwen's native reasoning continuation from its final answer.

    The official Qwen chat template places ``<think>\n`` at the end of the
    prompt. Consequently, a generated continuation contains the reasoning,
    one closing ``</think>`` tag, and then the final answer. Fail closed when
    that native boundary is absent or ambiguous; never search for or salvage a
    JSON-looking substring from malformed output.
    """

    raw_text = value.strip()
    if thinking == "disabled":
        return CompletionParts(
            raw_text=raw_text,
            reasoning_trace=None,
            final_answer=raw_text,
            boundary_valid=True,
            boundary_error=None,
        )
    if thinking != "enabled":
        raise ValueError(f"unsupported thinking mode: {thinking!r}")
    if THINKING_OPEN_TAG in raw_text:
        return CompletionParts(raw_text, None, None, False, "unexpected_open_tag")
    close_count = raw_text.count(THINKING_CLOSE_TAG)
    if close_count == 0:
        return CompletionParts(raw_text, None, None, False, "missing_close_tag")
    if close_count != 1:
        return CompletionParts(raw_text, None, None, False, "multiple_close_tags")
    reasoning, final_answer = raw_text.split(THINKING_CLOSE_TAG, maxsplit=1)
    final_answer = final_answer.strip()
    if not final_answer:
        return CompletionParts(raw_text, reasoning.strip(), None, False, "empty_final_answer")
    return CompletionParts(
        raw_text=raw_text,
        reasoning_trace=reasoning.strip(),
        final_answer=final_answer,
        boundary_valid=True,
        boundary_error=None,
    )
