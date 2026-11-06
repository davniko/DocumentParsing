import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/migrations/r30_owner_frames.py"
spec = importlib.util.spec_from_file_location("r30_owner_frames", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_changed_prefix_belongs_to_complete_country_owner():
    old = b"Exporter: MALDIVES\nTEL: +599 318 1998\n"
    new = b"Exporter: REPUBLIC OF MALDIVES\nTEL: +5993181998\n"
    spans = module.unique_owner_spans(old, new, [(10, 18), (24, 37)])
    assert [new[a:b] for a, b in spans] == [b"REPUBLIC OF MALDIVES", b"+5993181998"]


def test_ambiguous_spaces_are_not_silently_assigned():
    with pytest.raises(ValueError, match="ambiguous owner partitions"):
        module.unique_owner_spans(b"A: ONE TWO!", b"A: THREE FOUR FIVE!", [(3, 6), (7, 10)])


def test_current_literal_target_authority_resolves_space_partition():
    new = b"A: THREE FOUR FIVE!"
    spans = module.unique_owner_spans(
        b"A: ONE TWO!",
        new,
        [(3, 6), (7, 10)],
        expected_literals=["THREE FOUR", "FIVE"],
    )
    assert [new[a:b] for a, b in spans] == [b"THREE FOUR", b"FIVE"]


def test_protected_literal_change_fails():
    with pytest.raises(ValueError, match="inter-owner literals"):
        module.unique_owner_spans(b"A: ONE\nTEL TWO!", b"A: ONE\nFAX TWO!", [(3, 6), (11, 14)])


def test_native_bytes_not_unicode_character_offsets():
    old, new = "Nom: Émile; Pays: France!".encode(), "Nom: Jörg; Pays: F.R.!".encode()
    spans = module.unique_owner_spans(old, new, [(5, 11), (19, 25)])
    assert [new[a:b].decode() for a, b in spans] == ["Jörg", "F.R."]


def test_private_identifier_format_disambiguates_company_boundary():
    old, new = b"Invoice: OLD NAME 1234!", b"Invoice: NEW COMPANY LLC 5678!"
    spans = module.unique_owner_spans(old, new, [(9, 17), (18, 22)],
                                     expected_patterns=[None, r"[0-9]{4}"])
    assert [new[a:b] for a, b in spans] == [b"NEW COMPANY LLC", b"5678"]
    with pytest.raises(ValueError, match="inter-owner literals"):
        module.unique_owner_spans(old, new, [(9, 17), (18, 22)],
                                 expected_patterns=[None, r"[0-9]{5}"])


def test_touching_owners_require_sufficient_explicit_authority():
    old, new = b"ONE1234!", b"NEW NAME5678!"
    spans = module.unique_owner_spans(old, new, [(0, 3), (3, 7)],
                                     expected_patterns=[None, r"[0-9]{4}"])
    assert [new[a:b] for a, b in spans] == [b"NEW NAME", b"5678"]
    with pytest.raises(ValueError, match="unseparated owners"):
        module.unique_owner_spans(old, new, [(0, 3), (3, 7)])


def test_empty_owner_itself_is_exact_zero_length_authority():
    spans = module.unique_owner_spans(b"3!", b"9!", [(0,1),(1,1)])
    assert spans == ((0,1),(1,1))
