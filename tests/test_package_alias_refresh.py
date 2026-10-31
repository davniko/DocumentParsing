from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.package_alias_refresh import (
    apply_reviewed_cleanup,
    locate,
    owners,
    project_literal_targets,
    refresh,
)
from document_ocr.synthesis.package_presentations import PackageVocabulary

PATH = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].typeCategory"


def setup(raw="--- PAGE 1 ---\n20 STEEL DRUMS\nContainer A: 20 STEEL DRUMS"):
    row = dict(
        documentId="sample",
        sourceDocumentId="source",
        joinedRawText=raw,
        joinedRawTextSha256=digest(raw.encode()),
        positionedText="\n".join(
            s if s.startswith("--- PAGE") else s + " || 200,300" for s in raw.splitlines()
        ),
        target={
            "goodsItemDetails": [
                {
                    "description": "OIL",
                    "numberAndTypeOfPackages": [{"typeCategory": "STEEL", "packageQuantity": 20}],
                }
            ]
        },
    )
    row["positionedTextSha256"] = digest(row["positionedText"].encode())
    vocabulary = PackageVocabulary(
        dict(
            version=1,
            excludedUnscopedTokens=[],
            entries={
                "STEEL": dict(
                    family="DRUM", full=["STEEL DRUMS", "STEEL DRUM"], short=[], optional=[]
                ),
                "DRUM": dict(
                    family="DRUM", full=["DRUMS", "DRUM"], short=["DRM"], optional=["DRUM(S)"]
                ),
            },
        )
    )
    presenter = SimpleNamespace(
        vocabulary=vocabulary,
        metrics=SimpleNamespace(length=len),
        policy=SimpleNamespace(
            weights=lambda: {"full": 0.5, "short": 0.3, "optional": 0.2},
            vocabulary=SimpleNamespace(sha256="v"),
            certificates=SimpleNamespace(sha256="c"),
        ),
        require_certified=lambda _: None,
    )
    slots, start = [], 0
    while (start := raw.find("STEEL DRUMS", start)) != -1:
        a = len(raw[:start].encode())
        slots.append(
            dict(
                key="packages",
                occurrence=len(slots),
                categoryPaths=[PATH],
                mode="scalar",
                role="declaration" if not slots else "allocation",
                currentStart=a,
                currentEnd=a + 11,
                currentText="STEEL DRUMS",
            )
        )
        start += 11
    return row, slots, presenter


def test_locate_supports_current_privacy_edits_and_id_format_only():
    old = b"--- PAGE 1 ---\nOld name\nABC123: 20 DRUMS"
    new = b"--- PAGE 1 ---\nNew private name\nABC 123: 20 DRUMS"
    a = old.index(b"DRUMS")
    x, y = locate(old, new, a, a + 5)
    assert new[x:y] == b"DRUMS"
    with pytest.raises(ValueError, match="ambiguous"):
        locate(old, new.replace(b"20", b"21"), a, a + 5)


def test_locate_rejects_cross_page_and_multiline():
    old = b"--- PAGE 1 ---\nDRUMS\nBOXES"
    a = old.index(b"DRUMS")
    with pytest.raises(ValueError, match="page"):
        locate(old, b"--- PAGE 1 ---\n--- PAGE 2 ---\nDRUMS\nBOXES", a, a + 5)
    with pytest.raises(ValueError, match="multiline"):
        locate(old, old, a, len(old))


def test_refresh_is_deterministic_and_preserves_specificity_quantities_and_coordinates():
    row, slots, presenter = setup()
    before = deepcopy(row)
    new, receipt = refresh(row, slots, presenter, seed=31)
    assert refresh(row, slots, presenter, seed=31) == (new, receipt)
    assert row == before
    assert new["target"] == row["target"]
    assert "STEEL DRUM" in new["joinedRawText"]
    assert all(d["afterAdvance"] <= d["beforeAdvance"] for d in receipt["decisions"])
    assert receipt["allCurrentAnchorsPreserved"]
    assert new["positionedText"].count(" || 200,300") == 2


def test_protected_current_wording_never_restores_original_source():
    row, slots, presenter = setup()
    slots[0].update(protectedSourceDetail=True, source="ORIGINAL PRIVATE TEXT")
    new, receipt = refresh(row, slots, presenter, seed=31)
    assert receipt["decisions"][0]["after"] == "STEEL DRUMS"
    assert "ORIGINAL" not in new["joinedRawText"]


def test_finer_unregistered_wording_is_not_replaced():
    row, slots, presenter = setup()
    slots[0]["currentText"] = "NEW STEEL DRUMS"
    with pytest.raises(ValueError, match="stale"):
        refresh(row, slots, presenter, seed=31)
    row, slots, presenter = setup()
    presenter.vocabulary.entries["STEEL"]["full"] = ["STEEL DRUM"]
    with pytest.raises(ValueError, match="specificity"):
        refresh(row, slots, presenter, seed=31)


def test_literal_projection_uses_owned_occurrence_not_global_noun_replacement():
    raw = "--- PAGE 1 ---\n20 DRUMS\nOIL IN DRUMS\nAgent: Drums Company"
    a = raw.index("DRUMS", raw.index("OIL"))
    target = {
        "goodsItemDetails": [{"description": "OIL IN DRUMS"}],
        "parties": {"shipper": {"name": "Drums Company"}},
    }
    new, changes = project_literal_targets(
        target, raw, [dict(byteStart=a, byteEnd=a + 5, after="DRM")]
    )
    assert new["goodsItemDetails"][0]["description"] == "OIL IN DRM"
    assert new["parties"] == target["parties"]
    assert len(changes) == 1


def test_unicode_and_wrapped_literal_projection():
    raw = "é\nOIL IN\nSTEEL DRUMS"
    a = len(raw[: raw.index("STEEL")].encode())
    target = {"goodsItemDetails": [{"description": "OIL IN STEEL DRUMS"}]}
    new, _ = project_literal_targets(
        target, raw, [dict(byteStart=a, byteEnd=a + 11, after="STEEL DRUM")]
    )
    assert new["goodsItemDetails"][0]["description"] == "OIL IN STEEL DRUM"


def test_uppercase_target_with_mixed_case_ocr_is_projected_literally():
    raw = "Straße origin: Barbados\nSTEEL DRUMS"
    a = len(raw[: raw.index("STEEL")].encode())
    target = {"goodsItemDetails": [{"description": "STRASSE ORIGIN: BARBADOS STEEL DRUMS"}]}
    new, _ = project_literal_targets(
        target, raw, [dict(byteStart=a, byteEnd=a + 11, after="STEEL DRUM")]
    )
    assert new["goodsItemDetails"][0]["description"] == "STRASSE ORIGIN: BARBADOS STEEL DRUM"


def test_noncontiguous_target_is_held_instead_of_silently_leaving_stale_alias():
    raw = "OIL\nSEAL 123\nPACKED IN DRUMS"
    a = raw.index("DRUMS")
    with pytest.raises(ValueError, match="noncontiguous"):
        project_literal_targets(
            {"goodsItemDetails": [{"description": "OIL PACKED IN DRUMS"}]},
            raw,
            [dict(byteStart=a, byteEnd=a + 5, after="DRM")],
        )


def test_repeated_literal_disagreement_and_nonliteral_fields_are_rejected():
    raw = "OIL IN DRUMS\nOIL IN DRUMS"
    with pytest.raises(ValueError, match="differently"):
        project_literal_targets(
            {"goodsItemDetails": [{"description": "OIL IN DRUMS"}]},
            raw,
            [dict(byteStart=7, byteEnd=12, after="DRM")],
        )
    with pytest.raises(ValueError, match="nonliteral"):
        project_literal_targets(
            {"parties": {"shipper": {"name": "OIL IN DRUMS"}}},
            "OIL IN DRUMS",
            [dict(byteStart=7, byteEnd=12, after="DRM")],
        )


def test_owners_rejects_stale_candidate_hash():
    row, _, _ = setup()
    candidate = {**row, "joinedRawTextSha256": "bad"}
    with pytest.raises(ValueError, match="hash"):
        owners(candidate, row, {"slots": []})


def test_identity_replay_preserves_entire_current_record():
    row, slots, presenter = setup()
    for slot in slots:
        slot["protectedSourceDetail"] = True
    new, receipt = refresh(row, slots, presenter, seed=1)
    assert new == row
    assert not receipt["proof"]["edits"]


def test_reviewed_cleanup_has_strict_preconditions_and_keeps_line_positions():
    row, _, _ = setup("--- PAGE 1 ---\nOIL\nWRONG ECHO\n20 STEEL DRUMS")
    row["target"]["goodsItemDetails"][0]["description"] = "OIL WRONG ECHO"
    plan = dict(
        documentId="sample",
        beforeRecordSha256=digest(row),
        rawEdits=[dict(before="WRONG ECHO", after="")],
        targetEdits=[dict(path="goodsItemDetails[0].description", before=" WRONG ECHO", after="")],
    )
    new, receipt = apply_reviewed_cleanup(row, plan)
    assert new["joinedRawText"] == "--- PAGE 1 ---\nOIL\n\n20 STEEL DRUMS"
    assert new["positionedText"].splitlines()[-1] == row["positionedText"].splitlines()[-1]
    assert new["target"]["goodsItemDetails"][0]["description"] == "OIL"
    assert receipt["beforeRecordSha256"] == digest(row)
    with pytest.raises(ValueError, match="precondition"):
        apply_reviewed_cleanup(new, plan)


def test_reviewed_cleanup_explicit_repeated_occurrences_keep_all_anchors():
    row, _, _ = setup("--- PAGE 1 ---\nold@example.com\nold@example.com")
    plan = dict(
        documentId="sample",
        beforeRecordSha256=digest(row),
        targetEdits=[],
        rawEdits=[
            dict(before="old@example.com", after="a@example.com", occurrence=1),
            dict(before="old@example.com", after="b@example.com", occurrence=2),
        ],
    )
    new, _ = apply_reviewed_cleanup(row, plan)
    assert new["joinedRawText"].endswith("a@example.com\nb@example.com")
    assert [line.rsplit(" ||", 1)[-1] for line in new["positionedText"].splitlines()] == [
        line.rsplit(" ||", 1)[-1] for line in row["positionedText"].splitlines()
    ]


@pytest.mark.parametrize("occurrence", [None, 0, 3, True])
def test_reviewed_cleanup_rejects_ambiguous_or_invalid_occurrence(occurrence):
    row, _, _ = setup("--- PAGE 1 ---\nold@example.com\nold@example.com")
    edit = dict(before="old@example.com", after="a@example.com")
    if occurrence is not None:
        edit["occurrence"] = occurrence
    with pytest.raises(ValueError, match="reviewed_cleanup"):
        apply_reviewed_cleanup(
            row,
            dict(
                documentId="sample", beforeRecordSha256=digest(row), rawEdits=[edit], targetEdits=[]
            ),
        )
