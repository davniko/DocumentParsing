"""Configured entry points for DG registry compilation and semantic planning."""

from __future__ import annotations

from pathlib import Path

from document_ocr.synthesis.config import SynthesisDangerousGoodsRegistryConfig
from document_ocr.synthesis.dangerous_goods_registry import compile_dangerous_goods_registry
from document_ocr.training.config import resolve_config_path


def run_dangerous_goods_registry_build(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisDangerousGoodsRegistryConfig,
) -> dict[str, object]:
    """Compile the exact source manifest named by one validated YAML config."""

    del config_path
    source_root = resolve_config_path(project_root, config.inputs.source_root)
    manifest = resolve_config_path(project_root, config.inputs.source_manifest.path)
    output_parent = resolve_config_path(project_root, config.run.output_dir)
    result = compile_dangerous_goods_registry(
        source_root=source_root,
        source_manifest_path=manifest,
        expected_manifest_sha256=config.inputs.source_manifest.sha256,
        output_parent=output_parent,
        run_name=config.run.run_id,
    )
    return {
        "output_dir": str(result.root),
        "created": result.created,
        "hmt_records": result.receipt.audit.hmt_records,
        "maritime_records": result.receipt.audit.hmt_maritime_records,
        "exact_hs_links": result.receipt.audit.ecics_exact_hs_links,
    }
