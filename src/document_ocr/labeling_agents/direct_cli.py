"""Single-document pilot commands for direct extraction; paid work is always explicit."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from document_ocr.atomic import atomic_publish_json
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.labeling_agents.direct import DirectLabelingFlow, draft_value, encoded
from document_ocr.labeling_agents.direct_models import DirectLabelingConfig
from document_ocr.semantic_v3.transform import CategoryRegistry


def add_commands(commands: Any) -> None:
    for name, help_text in (
        ("extract", "one OCR-only direct extraction call; save draft labels without review"),
        ("refine", "review a saved V7 draft with at most two section correction waves"),
        ("schema", "export described extraction schema without any provider request"),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("--config", required=True, type=Path)
        parser.add_argument("--project-root", type=Path, default=Path.cwd())
        parser.add_argument(
            "--output",
            required=True,
            type=Path,
            help="New run directory (extract/refine), or schema JSON file (schema)",
        )
        if name != "schema":
            parser.add_argument("--ocr", required=True, type=Path, help="UTF-8 OCR text file")
        if name == "refine":
            parser.add_argument(
                "--candidate",
                required=True,
                type=Path,
                help="Direct V7 target JSON, e.g. extract output target.json",
            )
            parser.add_argument(
                "--pdf",
                required=True,
                type=Path,
                help=(
                    "Complete source PDF, supplied upfront to cargo relationship agents; "
                    "layout-only authority"
                ),
            )


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    root = arguments.project_root.resolve(strict=True)
    config = DirectLabelingConfig.model_validate_json(
        encoded(load_strict_yaml_mapping(arguments.config))
    )
    registry_payload = (root / config.package_registry).read_bytes()
    if sha256_bytes(registry_payload) != config.package_registry_sha256:
        raise ValueError("package registry hash mismatch")
    registry = CategoryRegistry.model_validate_json(registry_payload)
    if registry.registryKind != "package":
        raise ValueError("expected package registry")
    if arguments.command == "schema":
        from document_ocr.labeling_agents.direct import described_schema

        atomic_publish_json(
            arguments.output, described_schema(BillOfLadingExtractionV7Label, registry)
        )
        return {"status": "schema_exported", "path": str(arguments.output), "paid_requests": 0}

    ocr = arguments.ocr.read_text(encoding="utf-8")
    candidate = (
        draft_value(json.loads(arguments.candidate.read_bytes()))
        if arguments.command == "refine"
        else None
    )
    pdf = getattr(arguments, "pdf", None)
    if pdf is not None:
        pdf = pdf.resolve(strict=True)
        if not pdf.is_file():
            raise ValueError("PDF path must identify a file")
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(root / ".env").get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing from environment and project .env")

    async def run() -> dict[str, Any]:
        async with AsyncOpenAI(
            api_key=key, max_retries=0, timeout=config.timeout_seconds
        ) as client:
            model = OpenAIResponsesModel(
                config.model, provider=OpenAIProvider(openai_client=client)
            )
            flow = DirectLabelingFlow(
                model=model,
                config=config,
                project_root=root,
                output_dir=arguments.output,
                ocr=ocr,
                pdf_path=pdf,
            )
            try:
                if candidate is None:
                    await flow.extract()
                    status = "draft"
                else:
                    status = (await flow.refine(candidate))["status"]
            except BaseException as error:
                atomic_publish_json(
                    arguments.output / "failure.json",
                    {
                        "status": "failed",
                        "errorType": type(error).__name__,
                        "diagnostic": str(error),
                        "recordedCalls": len(flow.receipts),
                    },
                )
                raise
            return {"status": status, "output": str(arguments.output), "calls": len(flow.receipts)}

    return asyncio.run(run())
