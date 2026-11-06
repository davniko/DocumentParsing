import importlib.util
import sys
from pathlib import Path

from document_ocr.spatial_inputs.paddle import RegionSpan

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
SPEC = importlib.util.spec_from_file_location(
    "r30_subowners",
    Path(__file__).resolve().parents[1] / "scripts/migrations/expand_r30_reviewed_subowners.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_split_quantity_type_preserves_exact_occurrence():
    parent = {"byteStart": 10, "byteEnd": 22, "before": "12\n\nCARTONS "}
    child = {"byteStart": 14, "byteEnd": 21, "before": "CARTONS"}
    spans = [RegionSpan(0, 8, 3, 5, "12"), RegionSpan(0, 9, 0, 7, "CARTONS")]
    assert MODULE.project_child(parent, child, spans) == [[0, 9, 0, 7]]


def test_nonidentical_complete_owner_cannot_transfer():
    parent = {"byteStart": 0, "byteEnd": 9, "before": "12 PALLET"}
    child = {"byteStart": 0, "byteEnd": 2, "before": "12"}
    assert MODULE.project_child(parent, child, [RegionSpan(0, 0, 0, 9, "12 CARTON")]) is None


def test_outside_parent_is_not_a_candidate():
    parent = {"byteStart": 8, "byteEnd": 10, "before": "12"}
    child = {"byteStart": 2, "byteEnd": 4, "before": "12"}
    assert MODULE.project_child(parent, child, [RegionSpan(0, 0, 0, 2, "12")]) is None
