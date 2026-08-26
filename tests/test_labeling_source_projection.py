from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_exporter import _record

from document_ocr.hashing import sha256_bytes
from document_ocr.labeling_agents.source_projection import (
    LabelingSourceProjectionError,
    _QualityDocument,
    _QualityPageReference,
    _work_item,
)
from document_ocr.models import PageExtractionRecord, SourceObject


def _fixture(tmp_path: Path) -> tuple[_QualityDocument, PageExtractionRecord]:
    document_id = "doc_" + "a" * 64
    pdf = tmp_path / "source.pdf"
    pdf_payload = b"%PDF-1.7\nfixture\n%%EOF\n"
    pdf.write_bytes(pdf_payload)
    source_sha256 = sha256_bytes(pdf_payload)
    source = SourceObject(
        document_id=document_id,
        source_type="local",
        source_uri=pdf.as_uri(),
        source_dataset_version="fixture-v1",
        source_object_version=source_sha256,
        source_size_bytes=len(pdf_payload),
        source_last_modified=datetime(2026, 8, 23, tzinfo=UTC),
        source_sha256=source_sha256,
        local_canonical_path=str(pdf),
        local_relative_key="source.pdf",
        local_device=1,
        local_inode=2,
        local_mtime_ns=3,
    )
    values = _record(0, 1).model_dump(mode="python")
    raster_path = f"page-images/{document_id}/{values['page_id']}/{values['raster_sha256']}.png"
    values.update(
        {
            **source.model_dump(mode="python"),
            "run_id": "fixture-run",
            "document_id": document_id,
            "raster_path": raster_path,
        }
    )
    page = PageExtractionRecord.model_validate(values, strict=True)
    reference = _QualityPageReference(
        extraction_id=page.extraction_id,
        page_id=page.page_id,
        page_index=page.page_index,
        page_number=page.page_number,
        raw_ocr_text_sha256=page.raw_ocr_text_sha256,
        raw_response_path=page.raw_response_path,
        raster_path=page.raster_path,
        raster_sha256=page.raster_sha256,
    )
    document = _QualityDocument(
        schema_version=1,
        run_id="fixture-run",
        document_id=document_id,
        document_page_count=1,
        source=source,
        pages=(reference,),
    )
    return document, page


def test_quality_projection_builds_page_ordered_ocr_conditioned_work_item(
    tmp_path: Path,
) -> None:
    document, page = _fixture(tmp_path)

    item = _work_item(document, (page,))

    assert item.source.documentId == document.document_id
    assert item.source.extractionRunId == "fixture-run"
    assert item.joinedRawText == f"--- PAGE 1 ---\n{page.raw_ocr_text}"
    assert item.source.pages[0].rawResponseSha256 == page.raw_response_sha256


def test_quality_projection_rejects_local_pdf_identity_drift(tmp_path: Path) -> None:
    document, page = _fixture(tmp_path)
    Path(document.source.local_canonical_path or "").write_bytes(b"changed")

    with pytest.raises(LabelingSourceProjectionError, match="PDF identity changed"):
        _work_item(document, (page,))
