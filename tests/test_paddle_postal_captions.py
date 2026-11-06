import pytest

from document_ocr.synthesis.paddle_render import extra_targets, strip_postal_captions


def test_reviewed_caption_does_not_remove_value_or_uncaptioned_postal_text():
    action = {"target": "/parties/consignee/addressLine", "stripPostalCaptions": ["POST CODE:"]}
    assert strip_postal_captions("ROAD 12 Post Code: 38204", action) == "ROAD 12 38204"
    assert strip_postal_captions("POST CODE ROAD 12 38204", action) == "POST CODE ROAD 12 38204"
    assert strip_postal_captions("ROAD 12 38204", action) == "ROAD 12 38204"
    assert strip_postal_captions("COMPOST CODE: 38204", action) == "COMPOST CODE: 38204"


@pytest.mark.parametrize("caption", ["", ":", "POST CODE", "POST.*:", None])
def test_reviewed_caption_must_be_literal_label(caption):
    with pytest.raises(ValueError, match="caption"):
        strip_postal_captions("TEXT", {"target": "/addressLine", "stripPostalCaptions": [caption]})


def test_projection_cannot_clean_product_description():
    with pytest.raises(ValueError, match="addressLine"):
        strip_postal_captions("TEXT", {"target": "/description", "stripPostalCaptions": ["CODE:"]})
    assert strip_postal_captions("  LITERAL\nTEXT  ", {}) == "  LITERAL\nTEXT  "


def test_reviewed_unpunctuated_numeric_postcode_caption():
    action = {
        "target": "/parties/deliveryAgent/addressLine",
        "stripPostalCaptions": ["POSTAL CODE "],
    }
    assert strip_postal_captions("AL HAMRA Postal Code 14709 UAE", action) == "AL HAMRA 14709 UAE"
    assert strip_postal_captions("Postal Code 00124", action) == "00124"
    assert strip_postal_captions("POSTAL CODE ROAD 12", action) == "POSTAL CODE ROAD 12"
    assert strip_postal_captions("XPOSTAL CODE 14709", action) == "XPOSTAL CODE 14709"
    assert strip_postal_captions("POSTAL CODE: 14709", action) == "POSTAL CODE: 14709"


def test_extra_postal_projection_keeps_value_and_rejects_unowned_use():
    action = {
        "target": "/parties/deliveryAgent/addressLine",
        "stripPostalCaptions": ["POSTAL CODE "],
    }
    assert extra_targets(action, "AL HAMRA POSTAL CODE 14709 UAE") == {
        action["target"]: "AL HAMRA 14709 UAE"
    }
    for invalid, value in [({"stripPostalCaptions": ["CODE "]}, "CODE 12"), (action, 12)]:
        with pytest.raises(ValueError, match="scalar text target"):
            extra_targets(invalid, value)


@pytest.mark.parametrize("postcode", ["BD1727", "P96 X221", "K1A 0B1", "999-04"])
def test_reviewed_unpunctuated_alphanumeric_postcodes(postcode):
    action = {"target": "/parties/shipper/addressLine", "stripPostalCaptions": ["POST CODE "]}
    assert strip_postal_captions(f"ROAD 12 Post Code {postcode}", action) == f"ROAD 12 {postcode}"
    assert strip_postal_captions("POST CODE ROAD 12", action) == "POST CODE ROAD 12"
    assert strip_postal_captions("COMPOST CODE BD1727", action) == "COMPOST CODE BD1727"
