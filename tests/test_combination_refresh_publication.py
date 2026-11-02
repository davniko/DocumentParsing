import importlib.util
import sys
from pathlib import Path

import pytest

from document_ocr.synthesis.curated import digest


@pytest.fixture(scope="module")
def publication():
    scripts = Path(__file__).resolve().parents[1] / "scripts/synthesis"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "combination_publication_test", scripts / "finalize_combination_refresh.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(scripts))


def test_approved_owner_omission_preserves_other_lines_targets_and_anchors(publication):
    raw = "NOTIFY PARTY\n" + publication.OWNER + "\nADDRESS\n"
    positioned = (
        "NOTIFY PARTY || 100,100\n" + publication.OWNER + " || 100,130\nADDRESS || 100,160\n"
    )
    row = dict(
        documentId="test",
        sourceDocumentId=publication.OWNER_SOURCE,
        joinedRawText=raw,
        positionedText=positioned,
        target={"parties": {"notify": {"name": "EXAMPLE"}}},
    )
    after, receipt = publication.omit_owner(row)
    assert after["joinedRawText"] == "NOTIFY PARTY\n\nADDRESS\n"
    assert after["positionedText"] == "NOTIFY PARTY || 100,100\n ||\nADDRESS || 100,160\n"
    assert after["target"] == row["target"]
    assert row["joinedRawText"] == raw
    assert after["joinedRawTextSha256"] == digest(after["joinedRawText"].encode())
    assert receipt["lines"] == [1]


@pytest.mark.parametrize(
    "mutation", ["different_source", "label_owner", "partial_line", "duplicate"]
)
def test_owner_cleanup_rejects_unapproved_scope(publication, mutation):
    text = publication.OWNER
    row = dict(
        documentId="test",
        sourceDocumentId=publication.OWNER_SOURCE,
        joinedRawText=text,
        positionedText=text + " ||",
        target={},
    )
    if mutation == "different_source":
        row["sourceDocumentId"] = "another"
    if mutation == "label_owner":
        row["target"] = {"description": text}
    if mutation == "partial_line":
        row["joinedRawText"] = text + " ADDITIONAL TEXT"
    if mutation == "duplicate":
        row["joinedRawText"] = text + "\n" + text
        row["positionedText"] = text + " ||\n" + text + " ||"
    with pytest.raises(AssertionError):
        publication.omit_owner(row)


def test_unknown_equipment_does_not_establish_mixed_equipment(publication):
    row = {
        "target": {
            "containerInformation": [{"sizeCategory": "TWENTY"}, {}],
            "goodsItemDetails": [{"numberAndTypeOfPackages": [{}, {}]}],
        }
    }
    assert not publication.mixed_multi(row)
    row["target"]["containerInformation"][1] = {"sizeCategory": "FORTY"}
    assert publication.mixed_multi(row)
