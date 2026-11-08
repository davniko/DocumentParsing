"""Ensure the read-only audit distinguishes evidence screens from defects."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "party_audit", Path(__file__).parents[1] / "scripts/analysis/audit_r30_party_boundaries.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_party_target_linebreak_and_marker_are_detected():
    assert "line_break_or_tab" in audit.label_flags("shipper.name", "EXAMPLE\nLIMITED")
    assert "serialization_marker" in audit.label_flags("shipper.addressLine", "ROAD || 123,456")
    assert "control_character" in audit.label_flags("shipper.name", "BAD\x00NAME")


def test_contact_caption_is_not_postal_care_of():
    assert "contact_registration_caption_screen" in audit.label_flags(
        "shipper.addressLine", "ROAD TEL: 12345"
    )
    assert not audit.label_flags("shipper.addressLine", "CARE OF WAREHOUSE LTD, ROAD 4")


def test_interleaving_is_not_declared_unsupported():
    raw = "EXAMPLE\nBOOKING NO\nLIMITED"
    assert (
        audit.support("EXAMPLE LIMITED", raw, audit.compact(raw), set(audit.words(raw)))
        == "tokens_present_noncontiguous"
    )


def test_missing_lexeme_is_a_screen_not_silent_match():
    raw = "EXAMPLE LIMITED"
    assert (
        audit.support("UNRELATED LIMITED", raw, audit.compact(raw), set(audit.words(raw)))
        == "missing_lexeme_screen"
    )


def test_taxonomy_exhausts_saved_party_errors():
    result = audit.taxonomy()
    assert len(result) == 87
    assert len({(r["document_id"], r["path"]) for r in result}) == 87
