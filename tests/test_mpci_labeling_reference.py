from __future__ import annotations

import re
from pathlib import Path
from types import UnionType
from typing import Annotated, Union, get_args, get_origin

from pydantic import BaseModel

from document_ocr.label_schemas.mpci_bill_of_lading import MpciBillOfLadingDocumentPatch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPOSITORY_ROOT / "MPCI_BILL_OF_LADING_LABELING_REFERENCE.md"
SESSION_PROMPT_PATH = REPOSITORY_ROOT / "LABELING_SESSION_PROMPT.md"


def _model_leaf_paths(annotation: object, prefix: str) -> set[str]:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _model_leaf_paths(get_args(annotation)[0], prefix)
    if origin in (Union, UnionType):
        return set().union(
            *(
                _model_leaf_paths(member, prefix)
                for member in get_args(annotation)
                if member is not type(None)
            )
        )
    if origin in (tuple, list):
        return _model_leaf_paths(get_args(annotation)[0], f"{prefix}[]")
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return set().union(
            *(
                _model_leaf_paths(
                    field.annotation,
                    f"{prefix}.{name}",
                )
                for name, field in annotation.model_fields.items()
            )
        )
    return {prefix}


def _documented_leaf_paths(reference: str) -> set[str]:
    appendix = reference.split("## Appendix A: complete target-leaf inventory", maxsplit=1)[1]
    code_block = appendix.split("```text", maxsplit=1)[1].split("```", maxsplit=1)[0]
    return {line.strip() for line in code_block.splitlines() if line.strip()}


def test_reference_inventory_exactly_covers_pydantic_target_leaves() -> None:
    reference = REFERENCE_PATH.read_text(encoding="utf-8")
    expected = _model_leaf_paths(MpciBillOfLadingDocumentPatch, "documentPatch")

    assert _documented_leaf_paths(reference) == expected


def test_session_prompt_requires_every_worker_to_read_reference() -> None:
    prompt = SESSION_PROMPT_PATH.read_text(encoding="utf-8")
    worker_contract = prompt.split("## Exact worker prompt contract", maxsplit=1)[1]

    assert "MPCI_BILL_OF_LADING_LABELING_REFERENCE.md" in worker_contract
    assert "rawOcrEvidence" in worker_contract
    assert "pre-mapping raw OCR value" in worker_contract
    assert "frozen semantic conversion authority" in prompt


def test_reference_local_markdown_links_resolve() -> None:
    reference = REFERENCE_PATH.read_text(encoding="utf-8")
    link_targets = re.findall(r"\]\(([^):]+\.md)\)", reference)

    assert link_targets
    for target in link_targets:
        assert (REPOSITORY_ROOT / target).is_file(), target
