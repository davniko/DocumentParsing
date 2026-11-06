from pathlib import Path

import pytest

from document_ocr.synthesis.paddle_render import extra_value


def test_sequence_excludes_every_original_member():
    for identity in range(100):
        values = []
        for offset in (0, 1):
            recipe = {
                "sourceIdentifier": str(4 + offset),
                "identifierSequence": {
                    "canonical_identity": "4",
                    "sequence_length": 2,
                    "sequence_offset": offset,
                },
            }
            value, _ = extra_value(recipe, {}, {}, str(identity), None, project_root=Path.cwd())
            values.append(int(value))
        assert values[1] == values[0] + 1
        assert not set(values) & {4, 5}


@pytest.mark.parametrize(
    "bad",
    [
        {"canonical_identity": "4", "sequence_length": 2, "sequence_offset": 2},
        {"canonical_identity": "4", "sequence_length": 2, "sequence_offset": True},
        {"canonical_identity": "4", "sequence_length": 2, "sequence_offset": 0, "other": 1},
    ],
)
def test_sequence_requires_exact_valid_recipe(bad):
    with pytest.raises(ValueError, match="identifier sequence"):
        extra_value(
            {"sourceIdentifier": "4", "identifierSequence": bad},
            {},
            {},
            "x",
            None,
            project_root=Path.cwd(),
        )
