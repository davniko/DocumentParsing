"""Auditable prompt loading and literal document-text injection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from document_ocr.training.config import PromptConfig, resolve_config_path
from document_ocr.training.tasks import TrainingTask


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One validated prompt artifact with a content identity."""

    path: Path
    encoded: bytes
    text: str
    placeholder: str
    sha256: str
    template_sha256: str
    output_schema_sha256: str

    def render(self, document_text: str) -> str:
        """Inject one non-empty OCR document by literal replacement."""

        if not document_text or not document_text.strip():
            raise ValueError("document text must contain a non-whitespace character")
        if "\x00" in document_text:
            raise ValueError("document text must not contain a NUL character")
        return self.text.replace(self.placeholder, document_text)


def load_prompt(
    project_root: Path,
    config: PromptConfig,
    task: TrainingTask,
) -> PromptTemplate:
    """Load a UTF-8 template and bind its task-derived output schema."""

    configured_path = resolve_config_path(project_root, config.path)
    if configured_path.is_symlink():
        raise ValueError(f"prompt path must not be a symbolic link: {configured_path}")
    try:
        path = configured_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"prompt file does not exist: {configured_path}") from error
    if not path.is_file():
        raise ValueError(f"prompt path is not a regular file: {path}")

    template_encoded = path.read_bytes()
    try:
        template_text = template_encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"prompt is not valid UTF-8: {path}") from error
    if template_text.startswith("\ufeff"):
        raise ValueError("prompt must not contain a UTF-8 byte-order mark")
    if "\x00" in template_text:
        raise ValueError("prompt must not contain a NUL character")
    if template_text.count(config.placeholder) != 1:
        raise ValueError(f"prompt must contain exactly one {config.placeholder!r} placeholder")
    if template_text.count(config.schema_placeholder) != 1:
        raise ValueError(
            f"prompt must contain exactly one {config.schema_placeholder!r} placeholder"
        )
    residual = template_text.replace(config.placeholder, "").replace(
        config.schema_placeholder, ""
    )
    if "{{" in residual or "}}" in residual:
        raise ValueError("prompt contains an unsupported template expression")
    if not residual.strip():
        raise ValueError("prompt must contain instructions outside its placeholders")

    output_schema = task.prompt_schema_json()
    output_schema_encoded = output_schema.encode("utf-8")
    text = template_text.replace(config.schema_placeholder, output_schema)
    encoded = text.encode("utf-8")

    return PromptTemplate(
        path=path,
        encoded=encoded,
        text=text,
        placeholder=config.placeholder,
        sha256=hashlib.sha256(encoded).hexdigest(),
        template_sha256=hashlib.sha256(template_encoded).hexdigest(),
        output_schema_sha256=hashlib.sha256(output_schema_encoded).hexdigest(),
    )
