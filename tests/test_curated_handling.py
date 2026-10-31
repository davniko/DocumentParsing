from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated_handling import validate_rendered_handling

PATH = "documentPatch.goodsItemDetails[0].handlingInstructions[0]"


def target(value):
    return {"documentPatch": {"goodsItemDetails": [{"handlingInstructions": [value]}]}}


def blueprint(source, fragments=None):
    return SimpleNamespace(source=source, ownership_data={"handling_fragments": fragments or {}})


def test_literal_join_and_case_without_punctuation_rewriting():
    bp = blueprint('"Release goods"\nwithout original B/L.')
    validate_rendered_handling(bp, bp.source, target('"RELEASE GOODS" WITHOUT ORIGINAL B/L.'), [])
    with pytest.raises(ValueError, match="literal"):
        validate_rendered_handling(
            bp, bp.source, target("RELEASE GOODS. WITHOUT ORIGINAL B/L."), []
        )


def test_summaries_and_unsupported_values_fail_not_silently_repaired():
    bp = blueprint("Keep cool and dry; do not stack.")
    for value in ("KEEP IN GOOD CONDITION", "KEEP COOL AND DRY. DO NOT STACK."):
        with pytest.raises(ValueError, match="literal"):
            validate_rendered_handling(bp, bp.source, target(value), [])


def test_reviewed_separated_ventilation_replays_changed_rate_and_byte_shift():
    source = "ÉAB VENTILATION:\nPRIVATE REF 123\n20 CBM/HR"
    selectors = {
        PATH: [{"text": "VENTILATION:", "occurrence": 1}, {"text": "20 CBM/HR", "occurrence": 1}]
    }
    bp = blueprint(source, selectors)
    raw = source.encode()
    start = raw.index(b"20 CBM/HR")
    edits = [
        {"byteStart": 0, "byteEnd": len("ÉAB".encode()), "before": "ÉAB", "after": "X"},
        {"byteStart": start, "byteEnd": len(raw), "before": "20 CBM/HR", "after": "0 CBM/HR"},
    ]
    rendered = "X VENTILATION:\nPRIVATE REF 123\n0 CBM/HR"
    validate_rendered_handling(bp, rendered, target("VENTILATION: 0 CBM/HR"), edits)
    with pytest.raises(ValueError, match="projection"):
        validate_rendered_handling(bp, rendered, target("VENTILATION: 20 CBM/HR"), edits)
    with pytest.raises(ValueError, match="rendered bytes"):
        validate_rendered_handling(
            bp, rendered.replace("0 CBM", "1 CBM"), target("VENTILATION: 0 CBM/HR"), edits
        )


def test_unreviewed_separated_fields_and_crossing_edits_fail():
    source = "VENTILATION:\nREF 123\n20 CBM/HR"
    with pytest.raises(ValueError, match="literal"):
        validate_rendered_handling(blueprint(source), source, target("VENTILATION: 20 CBM/HR"), [])
    selectors = {
        PATH: [{"text": "VENTILATION:", "occurrence": 1}, {"text": "20 CBM/HR", "occurrence": 1}]
    }
    with pytest.raises(ValueError, match="crosses"):
        validate_rendered_handling(
            blueprint(source, selectors),
            "VENTILATION: 0 CBM/HR",
            target("VENTILATION: 0 CBM/HR"),
            [
                {
                    "byteStart": 0,
                    "byteEnd": len(source),
                    "before": source,
                    "after": "VENTILATION: 0 CBM/HR",
                }
            ],
        )


def test_absent_handling_has_no_text_inference_and_stale_join_fails():
    bp = blueprint("21 DAYS FREE TIME\nTemperature: -18 C")
    validate_rendered_handling(bp, bp.source, {"documentPatch": {}}, [])
    with pytest.raises(ValueError, match="existing handling"):
        validate_rendered_handling(
            blueprint(bp.source, {PATH: []}), bp.source, {"documentPatch": {}}, []
        )
