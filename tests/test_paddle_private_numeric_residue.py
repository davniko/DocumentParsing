import pytest

from document_ocr.synthesis.paddle_validation import check_private_residue


def privacy(value):
    return {"proof": [{"owner": "reviewed-private", "spans": [{"expected": value}]}]}


@pytest.mark.parametrize(
    ("old", "text"),
    [
        ("52510", "IMPORT REFERENCE1484931875251072029"),
        ("001827", "CPIU1001827/20HC\nCPIU1001827/20HC"),
        ("001827", "SEAL: 0018270"),
        ("001827", "SEAL: 1001827"),
        ("001827", "SEAL: 10018270"),
    ],
)
def test_numeric_substring_is_not_a_complete_original_identity(old, text):
    check_private_residue(privacy(old), {"joinedRawText": text})


@pytest.mark.parametrize(
    ("old", "text"),
    [
        ("52510", "P.O BOX 52510, NEW ROAD"),
        ("001827", "001827"),
        ("001827", "SEAL:001827/OTHER"),
        ("001827", "SEAL:AB001827"),
        ("001827", "SEAL:001827AB"),
        ("001 827", "SEAL: 001\n827"),
        ("001827", "CPIU1001827/20HC\nSEAL:001827"),
        ("P.O BOX 8330", "P.O BOX 8330, SAIF ZONE, SHARJAH"),
        ("Original Trading", "ORIGINAL\nTRADING LTD"),
        ("AB1827", "PREFIXAB1827SUFFIX"),
    ],
)
def test_complete_values_and_existing_non_numeric_screen_still_fail(old, text):
    with pytest.raises(ValueError, match="original private owner survives"):
        check_private_residue(privacy(old), {"joinedRawText": text})


def test_unrelated_numbers_and_short_screen_scope_are_unchanged():
    check_private_residue(privacy("001827"), {"joinedRawText": "SEAL 006834"})
    check_private_residue(privacy("1234"), {"joinedRawText": "1234"})
