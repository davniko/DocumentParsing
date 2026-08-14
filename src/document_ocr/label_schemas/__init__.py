"""Document-type-specific, OCR-conditioned KIE label contracts."""

from document_ocr.label_schemas.common import (
    ExtractionPageReference,
    ExtractionSourceReference,
    FieldEvidence,
    LabelWarning,
    RawOcrValueEvidence,
)
from document_ocr.label_schemas.mpci_bill_of_lading import (
    MpciBillOfLadingAnnotation,
    MpciBillOfLadingDocumentPatch,
    MpciBillOfLadingLabel,
)

__all__ = [
    "ExtractionPageReference",
    "ExtractionSourceReference",
    "FieldEvidence",
    "LabelWarning",
    "MpciBillOfLadingAnnotation",
    "MpciBillOfLadingDocumentPatch",
    "MpciBillOfLadingLabel",
    "RawOcrValueEvidence",
]
