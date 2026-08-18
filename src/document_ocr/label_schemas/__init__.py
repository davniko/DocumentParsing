"""Document-type-specific, OCR-conditioned KIE label contracts."""

from document_ocr.label_schemas.bill_of_lading import (
    BillOfLadingAnnotation,
    BillOfLadingDocumentPatch,
    BillOfLadingExclusion,
    BillOfLadingLabel,
)
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
from document_ocr.label_schemas.mpci_projection import (
    MpciProjectionError,
    project_bill_of_lading_to_mpci,
)

__all__ = [
    "BillOfLadingAnnotation",
    "BillOfLadingDocumentPatch",
    "BillOfLadingExclusion",
    "BillOfLadingLabel",
    "ExtractionPageReference",
    "ExtractionSourceReference",
    "FieldEvidence",
    "LabelWarning",
    "MpciBillOfLadingAnnotation",
    "MpciBillOfLadingDocumentPatch",
    "MpciBillOfLadingLabel",
    "MpciProjectionError",
    "RawOcrValueEvidence",
    "project_bill_of_lading_to_mpci",
]
