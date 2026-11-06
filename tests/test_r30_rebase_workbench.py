import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/migrations"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "r30_rebase_workbench", SCRIPTS / "r30_rebase_workbench.py"
)
workbench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = workbench
spec.loader.exec_module(workbench)


def source(*lines):
    return {
        "pages": [
            {
                "index": 0,
                "width": 1000,
                "height": 1000,
                "regions": [
                    {"index": i, "text": line, "box": [10, 10 + 20 * i, 100, 25 + 20 * i]}
                    for i, line in enumerate(lines)
                ],
            }
        ]
    }


def replacement(spans, text):
    return {
        "spans": [dict(zip(("page", "region", "start", "end"), s, strict=True)) for s in spans],
        "text": text,
    }


def test_compile_preserves_exact_reviewed_public_lexical_witness(monkeypatch, tmp_path):
    import rebase_r30_sources

    did = "doc_12345678"
    witness = {
        "planIndex": 20,
        "page": 0,
        "text": "HS 845140000013",
        "reason": "Public HS digits overlap private reference suffix, not identity.",
    }
    review = {"notes": ["Reviewed native source"], "retainedLexemes": [witness]}
    monkeypatch.setattr(workbench, "OUT", tmp_path)
    monkeypatch.setattr(
        workbench, "batch_records", lambda _: [{"documentId": did, "activeTemplate": False}]
    )
    original_read = workbench.read

    def read(path):
        if path.name == "workbench-review.json":
            return {"12345678": review}
        if path.name == did + ".json":
            return {"sourceSha256": "pin", "nativeFingerprint": "native"}
        return original_read(path)

    seen = []
    monkeypatch.setattr(workbench, "read", read)
    monkeypatch.setattr(workbench, "sha256_file", lambda _: "pin")
    monkeypatch.setattr(
        rebase_r30_sources,
        "stage_reviewed_record",
        lambda source, decision, engine: seen.append(decision) or {"proof": []},
    )
    workbench.compile_review("test")
    assert len(seen) == 2
    assert all(d["retainedLexemes"] == [witness] for d in seen)
    assert seen[0]["retainedLexemes"] is not review["retainedLexemes"]


def test_private_literal_keeps_native_punctuation_and_current_private_value():
    s = source("OLD LTD,CAIRO", "TEL 1234")
    edits = [replacement([[0, 0, 0, 7]], "NEW LTD")]
    assert workbench.private_literal(s, edits, [[0, 0, "OLD LTD,CAIRO"]]) == "NEW LTD,CAIRO"
    assert workbench.private_literal(s, edits, [[0, 1, "1234"]]) == "1234"


def test_private_literal_multiline_owner_is_projected_once():
    s = source("OLD", "COMPANY", "Egypt")
    edits = [replacement([[0, 0, 0, 3], [0, 1, 0, 7]], "New Firm")]
    assert (
        workbench.private_literal(
            s, edits, [[0, 0, "OLD"], [0, 1, "COMPANY"], [0, 2, "Egypt"]], uppercase=True
        )
        == "NEW FIRM EGYPT"
    )


@pytest.mark.parametrize("fragments", [[], [[0, 0, "OLD"]], [[0, 0, "OLD LTD"], [0, 0, "LTD"]]])
def test_private_literal_rejects_missing_partial_or_overlapping_ownership(fragments):
    with pytest.raises(ValueError):
        workbench.private_literal(
            source("OLD LTD"), [replacement([[0, 0, 0, 7]], "NEW LTD")], fragments
        )


@pytest.mark.parametrize(
    "q",
    [
        dict(page=-1, region=0, quote="A"),
        dict(page=0, region=3, quote="A"),
        dict(page=0, region=0, quote="Z"),
        dict(page=0, region=0, quote="A"),
    ],
)
def test_quotes_reject_invalid_or_nonunique_evidence(q):
    with pytest.raises(ValueError):
        workbench.quote_spans(source("A A"), [q])


def test_bad_proposal_does_not_hide_other_findings():
    result = workbench.FieldOnlyReview(
        sets=[
            workbench.SetProposal(
                path="/billOfLadingNumber",
                value_json='"BL12"',
                evidence=[workbench.Quote(page=0, region=0, quote="BL12")],
                reason="printed",
            ),
            workbench.SetProposal(
                path="/issueDate",
                value_json="not-json",
                evidence=[workbench.Quote(page=0, region=0, quote="absent")],
                reason="bad",
            ),
        ],
        removals=[],
        concerns=[],
    )
    issues = workbench.proposal_checks(source("BL12"), result, "fields", [])
    assert len(issues) == 1
    assert issues[0]["index"] == 1
    assert len(result.sets) == 2


def test_binding_draft_does_not_promote_absence_or_bad_quotes():
    bundle = {
        "owners": [
            dict(owner="bill@0:4", key="bill", status="manual_span_review"),
            dict(owner="seal@5:9", key="seal", status="manual_span_review"),
        ]
    }
    draft = workbench.binding_draft(
        source("BL12"),
        bundle,
        {
            "owners": [
                dict(
                    owner="bill@0:4",
                    disposition="mapped",
                    spans=[dict(page=0, region=0, quote="BL12")],
                ),
                dict(owner="seal@5:9", disposition="absent", spans=[]),
            ]
        },
    )
    assert draft["status"] == "UNREVIEWED_BINDING_DRAFT"
    assert len(draft["overrides"]) == len(draft["issues"]) == 1


def test_new_wording_patterns_are_part_of_schema_and_validation():
    from propose_r30_new_wording import wording_model

    model = wording_model(
        {"signature": "Fictional signature"}, {"signature": r"^[A-Z][a-z]{2,6}\.$"}
    )
    assert model.model_json_schema()["properties"]["signature"]["pattern"] == r"^[A-Z][a-z]{2,6}\.$"
    assert model(signature="Doran.").signature == "Doran."
    with pytest.raises(ValueError):
        model(signature="D. Rana")
    with pytest.raises(ValueError, match="unrequested"):
        wording_model({"signature": "Name"}, {"other": ".*"})


def test_preflight_collects_all_geometry_conflicts_without_relaxing_them():
    from prepare_r30_paddle_pilot import load_pages

    pages = load_pages(source("A", "foreign one", "B", "C", "foreign two", "D"))
    overlay = {
        "overrides": [],
        "extras": [
            {"owner": "first", "regions": [[0, 0], [0, 2]]},
            {"owner": "second", "regions": [[0, 3], [0, 5]]},
        ],
    }
    issues = workbench.preflight_owners(pages, [], {"slots": []}, overlay)
    assert [i["foreignRegions"] for i in issues] == [[[0, 1]], [[0, 4]]]
    overlay["extras"][0]["separateRegionFrames"] = True
    issues = workbench.preflight_owners(pages, [], {"slots": []}, overlay)
    assert [i["foreignRegions"] for i in issues] == [[[0, 4]]]
    overlay["extras"][1]["separateRegionFrames"] = True
    assert workbench.preflight_owners(pages, [], {"slots": []}, overlay) == []


def test_preflight_requires_explicit_overlay_and_complete_controls(tmp_path, monkeypatch):
    import json

    did = "doc_12345678"
    monkeypatch.setattr(workbench, "OUT", tmp_path)
    monkeypatch.setattr(workbench, "ART", tmp_path)
    monkeypatch.setattr(
        workbench, "batch_records", lambda _: [{"documentId": did, "activeTemplate": True}]
    )
    (tmp_path / "transfers").mkdir()
    (tmp_path / "transfers" / (did + ".json")).write_text(json.dumps({"descendants": []}))
    overlay = tmp_path / "reviewed.json"
    overlay.write_text(json.dumps({"sources": {"12345678": {"overrides": [], "extras": []}}}))
    workbench.preflight("test", overlay)
    receipt = json.loads((tmp_path / "batches/test/workbench-preflight.json").read_text())
    assert receipt["results"][0]["issues"] == ["requires three distinct controls; found 0"]
    assert receipt["overlaySha256"] == workbench.sha256_file(overlay)
    with pytest.raises(FileNotFoundError):
        workbench.preflight("test", tmp_path / "missing.json")


def test_target_field_shapes_ignore_cardinality_not_missing_roles():
    one = {"goods": [{"packages": 1}], "notify": {"sameAs": "consignee"}}
    two = {"goods": [{"packages": 3}, {"packages": 4}], "notify": {"sameAs": None}}
    assert workbench.target_field_shapes(one) - workbench.target_field_shapes(two) == {
        "/notify/sameAs"
    }
    assert workbench.target_field_shapes(two) - workbench.target_field_shapes(one) == set()
    assert workbench.target_field_shapes({"items": [], "none": None, "nested": {}}) == set()


def test_control_review_rejects_stale_source_before_reading_controls(tmp_path, monkeypatch):
    import json

    did = "doc_12345678"
    monkeypatch.setattr(workbench, "OUT", tmp_path)
    monkeypatch.setattr(workbench, "ART", tmp_path)
    monkeypatch.setattr(workbench, "rows", lambda _: [])
    monkeypatch.setattr(
        workbench, "batch_records", lambda _: [{"documentId": did, "activeTemplate": True}]
    )
    directory = tmp_path / "batches/test"
    directory.mkdir(parents=True)
    (directory / "paddle-overlays.json").write_text(json.dumps({"sources": {"12345678": {}}}))
    (directory / "main-agent-decisions.json").write_text("{}")
    (tmp_path / "staged-real").mkdir()
    (tmp_path / "staged-real" / (did + ".json")).write_text(json.dumps({"reviewSha256": "stale"}))
    with pytest.raises(ValueError, match="predates current decisions"):
        workbench.review_controls("test", None)
