import re
import uuid
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated_auxiliary import (
    _render_recipe,
    _validate_recipe,
    validate_identifier_privacy,
)
from document_ocr.synthesis.generators import DeterministicStream, surface_pattern


@pytest.mark.parametrize(
    "source",
    [
        "SHIN-ELQ-240522_2415",
        "000123",
        "xZ9/ab-123",
        "133.848.188-65",
        "A1",
        "2050557452024051307",
        "S&P-23224",
        "4700 DL7A 029732",
    ],
)
def test_source_format_codes_change_identity_not_format(source):
    recipe = {"identifier": "private commercial reference", "format": "source"}
    _validate_recipe(recipe)
    values = []
    for variant in range(50):
        stream = DeterministicStream(13, "private-identifiers", str(variant))
        rendered = _render_recipe(recipe, None, {}, stream, source, {30})
        assert rendered != source
        assert surface_pattern(rendered) == surface_pattern(source)
        assert rendered == _render_recipe(recipe, None, {}, stream, source, {30})
        values.append(rendered)
    assert len(set(values)) > 1


def test_punctuation_and_case_variants_share_one_code():
    recipe = {"identifier": "private registration", "format": "source"}
    stream = DeterministicStream(13, "private-identifiers", "sample")
    variants = ["AB-12.345", "ab12345", "AB 12 345"]
    values = [_render_recipe(recipe, None, {}, stream, v, {30}) for v in variants]
    assert len({re.sub(r"[^A-Z0-9]", "", v.upper()) for v in values}) == 1


def test_uuid_and_unpunctuated_aliases_stay_hexadecimal():
    source = "9E853AC1-7A97-4BEA-8F4D-\nC41528C68EBE"
    plain = re.sub(r"[^A-Z0-9]", "", source)
    recipe = {"identifier": "private cargo reference", "format": "source"}
    for variant in range(20):
        stream = DeterministicStream(13, "private-uuid", str(variant))
        formatted = _render_recipe(recipe, None, {}, stream, source, {30})
        compact = _render_recipe(recipe, None, {}, stream, plain, {30})
        assert uuid.UUID(formatted.replace("\n", "")).hex.upper() == compact
        assert surface_pattern(formatted) == surface_pattern(source)
        assert compact != plain


def test_mixed_owner_preserves_caption_date_and_every_other_byte():
    recipe = {"identifier": "invoice", "format": "source", "source_value": "S&P-23224"}
    _validate_recipe(recipe)
    original = "INVOICE NO. S&P-23224 DATED: 27.11.2023"
    value = _render_recipe(
        recipe, None, {}, DeterministicStream(1, "audit", "sample"), original, {30}
    )
    assert value.startswith("INVOICE NO. ") and value.endswith(" DATED: 27.11.2023")
    assert "S&P-23224" not in value
    assert surface_pattern(value) == surface_pattern(original)
    with pytest.raises(ValueError, match="absent"):
        _render_recipe(
            recipe, None, {}, DeterministicStream(1, "audit", "sample"), "OTHER CODE", {30}
        )


@pytest.mark.parametrize(
    "recipe",
    [
        {"identifier": "invoice", "source_value": "ABC123"},
        {"identifier": "invoice", "format": "source", "source_value": ""},
        {"identifier": "invoice", "format": "source", "source_value": "É123"},
        {"identifier": "invoice", "format": "source", "source_value": 123},
    ],
)
def test_invalid_source_identifier_metadata_surfaces(recipe):
    with pytest.raises(ValueError, match="source_value"):
        _validate_recipe(recipe)


def test_incompatible_canonical_alias_is_rejected():
    recipe = {"identifier": "reference", "format": "source", "canonical_identity": "AB1234"}
    with pytest.raises(ValueError, match="incompatible character classes"):
        _render_recipe(recipe, None, {}, DeterministicStream(1, "audit", "sample"), "123456", {30})


def private_blueprint(policy, recipe):
    return SimpleNamespace(
        ownership_data={"identifier_privacy": policy},
        target={"documentPatch": {}},
        regions=[
            SimpleNamespace(key="invoice", kind="identifier", curated_key=None, target_paths=())
        ],
        historical_bindings={"invoice": {"auxiliary_recipe": recipe}},
    )


def test_identifier_privacy_requires_complete_classification_before_paid_work():
    bp = private_blueprint({}, {})
    with pytest.raises(ValueError, match="unresolved"):
        validate_identifier_privacy(bp)
    bp.ownership_data = {}
    with pytest.raises(ValueError, match="explicit privacy inventory"):
        validate_identifier_privacy(bp)
    for recipe in ({}, {"identifier": "invoice"}, {"text": "SOURCE INVOICE"}):
        with pytest.raises(ValueError, match="source-format generator"):
            validate_identifier_privacy(private_blueprint({"invoice": "private"}, recipe))
    validate_identifier_privacy(
        private_blueprint({"invoice": "private"}, {"identifier": "invoice", "format": "source"})
    )
    validate_identifier_privacy(private_blueprint({"invoice": "public"}, {}))
    validate_identifier_privacy(
        private_blueprint({"invoice": "dependent"}, {"target": "documentPatch.billOfLadingNumber"})
    )


@pytest.mark.parametrize("policy", [{"invoice": "unknown"}, {"other": "private"}, {1: "private"}])
def test_identifier_privacy_cannot_silently_approve_invalid_metadata(policy):
    with pytest.raises(ValueError):
        validate_identifier_privacy(private_blueprint(policy, {}))
