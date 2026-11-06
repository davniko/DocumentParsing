import pytest

from document_ocr.synthesis.paddle_render import literal_owner_value


@pytest.mark.parametrize(
    ("address", "code", "expected"),
    [
        ("SAINTE-ROSE 97115 GUADELOUPE", "97115", "SAINTE-ROSE 97115 GUADELOUPE"),
        ("ROAD 197115 GUADELOUPE", "97115", "ROAD 197115 GUADELOUPE 97115"),
        ("ROAD X97115 GUADELOUPE", "97115", "ROAD X97115 GUADELOUPE 97115"),
        ("CANADA K1A\n0B1", "K1A 0B1", "CANADA K1A 0B1"),
    ],
)
def test_only_the_explicit_complete_postal_code_is_deduplicated(address, code, expected):
    rows = [{"owner": "address@1:2", "after": address}, {"owner": "postcode@3:4", "after": code}]
    assert (
        literal_owner_value(rows, ["address", "postcode"], " ", postal_code_key="postcode")
        == expected
    )
    assert literal_owner_value(rows, ["address", "postcode"], " ") == " ".join(
        (address + " " + code).split()
    )


def test_postcode_owner_cannot_be_guessed():
    with pytest.raises(ValueError):
        literal_owner_value(
            [{"owner": "address@1:2", "after": "York"}],
            ["address"],
            " ",
            postal_code_key="postcode",
        )
