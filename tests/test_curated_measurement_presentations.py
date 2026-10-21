"""Reviewed unusual unit spellings stay source-local and independently corroborated."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import pytest

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_ownership import build_owned_blueprint, ownership_surfaces
from document_ocr.synthesis.curated_templates import render_sampling_blueprint

P = "documentPatch.goodsItemDetails[0].volume"


def fixture(*, token="CM", alias="44.461", hint="CM", alias_unit="CF"):
    raw = f"MEASUREMENT\n1.259 {token}\n{alias} {alias_unit}\n1.259 {token}\n"
    row = {
        "documentId": "reviewed_unit",
        "joinedRawText": raw,
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": None,
                "goodsItemDetails": [{"volume": {"value": 1.259, "unit": "cubic_metre"}}],
            },
        },
    }
    contract = SourceContract(
        variables=[
            dict(
                key="cube",
                kind="volume",
                value="1.259",
                meaning=P + ".value",
                required_literals=[],
                occurrences=[
                    dict(text="1.259", occurrence=n, presentation="number") for n in (1, 2)
                ],
            )
        ],
        targets=[dict(path=P + ".value", expression="{cube}")],
        fixed_context=(
            "Reviewed paired metric/imperial measurement column; source-local spelling."
        ),
    )
    start = raw.index(alias)
    historical = {
        "document_id": "reviewed_unit",
        "source_sha256": digest(raw.encode()),
        "bindings": [
            dict(
                logical_key="cube_ft",
                target_paths=[],
                value_kind="decimal_measurement",
                occurrences=[
                    dict(byte_start=start, byte_end=start + len(alias), source_text=alias)
                ],
            )
        ],
    }
    recipe = {"measure_path": P, "render_unit": "cubic_foot"}
    if hint is not None:
        recipe["canonical_unit_text"] = hint
    return row, build_owned_blueprint(row, historical, contract, {"surfaces": {"cube_ft": recipe}})


@pytest.mark.parametrize("value", [1.259, 0.4, 10, 23.761])
def test_source_local_spelling_preserves_identity_and_couples_all_three_numbers(value):
    row, bp = fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["volume"]["value"] = value
    surfaces = ownership_surfaces(bp, target, None)
    expected = (
        "44.461"
        if value == 1.259
        else str((Decimal(str(value)) / Decimal("0.028316846592")).quantize(Decimal("0.001")))
    )
    assert surfaces == {"cube_ft": [expected]}
    rendered, _, _ = render_sampling_blueprint(
        bp, target, {"cube": str(value)}, surface_values=surfaces
    )
    assert rendered.count(f"{Decimal(str(value)):.3f} CM") == 2
    assert f"{expected} CF" in rendered
    if value == 1.259:
        assert rendered == row["joinedRawText"]


@pytest.mark.parametrize("hint", [None, "CBM", "CM|KG", "CM."])
def test_unrecognized_or_wrong_literal_is_not_a_global_unit_fallback(hint):
    row, bp = fixture(hint=hint)
    with pytest.raises(ValueError, match="unit is not printed"):
        ownership_surfaces(bp, row["target"], None)


@pytest.mark.parametrize("hint", ["", " CM", "CM\n", "CM\t", 7, None])
def test_bad_literal_field_rejected(hint):
    row, bp = fixture()
    declarations = deepcopy(bp.ownership_data)
    declarations["surfaces"]["cube_ft"]["canonical_unit_text"] = hint
    with pytest.raises(ValueError, match="literal printed token"):
        ownership_surfaces(replace(bp, ownership_data=declarations), row["target"], None)


def test_literal_is_escaped_not_treated_as_regex():
    row, bp = fixture(token="C.M", hint="C.M")
    assert ownership_surfaces(bp, row["target"], None) == {"cube_ft": ["44.461"]}
    row, bp = fixture(token="CXM", hint="C.M")
    with pytest.raises(ValueError, match="unit is not printed"):
        ownership_surfaces(bp, row["target"], None)


def test_known_wrong_dimension_unit_cannot_be_reinterpreted():
    row, bp = fixture(token="KG", hint="KG")
    with pytest.raises(ValueError, match="known different unit"):
        ownership_surfaces(bp, row["target"], None)


def test_missing_known_alias_unit_and_incompatible_precision_are_rejected():
    row, bp = fixture(alias_unit="XX")
    with pytest.raises(ValueError, match="converted unit is not printed"):
        ownership_surfaces(bp, row["target"], None)
    row, bp = fixture(alias="45.461")
    with pytest.raises(ValueError, match="canonical printed interval"):
        ownership_surfaces(bp, row["target"], None)


def test_wrong_source_bytes_and_wrong_field_are_rejected():
    row, bp = fixture()
    historical = deepcopy(bp.historical_bindings)
    historical["cube_ft"]["occurrences"][0]["source_text"] = "99.999"
    with pytest.raises(ValueError, match="source bytes"):
        ownership_surfaces(replace(bp, historical_bindings=historical), row["target"], None)
    declarations = deepcopy(bp.ownership_data)
    declarations["surfaces"]["cube_ft"]["unknown_unit_text"] = "CM"
    with pytest.raises(ValueError, match="only optional"):
        ownership_surfaces(replace(bp, ownership_data=declarations), row["target"], None)


def test_same_unit_alias_is_not_independent_corroboration():
    row, bp = fixture()
    declarations = deepcopy(bp.ownership_data)
    declarations["surfaces"]["cube_ft"]["render_unit"] = "cubic_metre"
    with pytest.raises(ValueError, match="independent different-unit"):
        ownership_surfaces(replace(bp, ownership_data=declarations), row["target"], None)
