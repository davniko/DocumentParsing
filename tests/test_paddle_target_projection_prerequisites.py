import pytest

from document_ocr.synthesis.paddle_render import target_projection_enabled

ACTION = {"target": "/parties/shipper/contactDetails/phoneNumbers",
          "requiresTarget": "/parties/shipper/addressLine"}


def test_reviewed_postal_ownership_gates_new_contact_without_changing_target():
    target = {"parties": {"shipper": {"name": "A"}}}
    assert not target_projection_enabled(target, ACTION)
    assert target == {"parties": {"shipper": {"name": "A"}}}
    target["parties"]["shipper"]["addressLine"] = "OWNED ADDRESS"
    assert target_projection_enabled(target, ACTION)


def test_existing_fact_is_never_silently_removed_when_prerequisite_is_absent():
    target = {"parties": {"shipper": {"contactDetails": {"phoneNumbers": ["123"]}}}}
    with pytest.raises(ValueError, match="conflicts"):
        target_projection_enabled(target, ACTION)


@pytest.mark.parametrize("value", ["", "   ", [], 0, False])
def test_invalid_reviewed_prerequisite_is_not_treated_as_absence(value):
    with pytest.raises(ValueError, match="nonempty"):
        target_projection_enabled({"parties": {"shipper": {"addressLine": value}}}, ACTION)


def test_ordinary_projection_and_explicit_null_absence():
    assert target_projection_enabled({}, {"target": "/unconditional"})
    assert not target_projection_enabled({"parties": {"shipper": {"addressLine": None}}}, ACTION)
    with pytest.raises(ValueError, match="pointer"):
        target_projection_enabled({}, {**ACTION, "requiresTarget": True})
