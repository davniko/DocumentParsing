import copy
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
module = importlib.import_module("r30_literal_target_corrections")


def fixture(field="origin"):
    rendered = {
        "documentId": "synthetic-1",
        "sourceDocumentId": "source-1",
        "pages": [{"regions": [{"text": "PRODUCT OF PORTUGAL. KEEP COOL"}]}],
        "target": {"goodsItemDetails": [{"description": "PRODUCT OF PORTUGAL. KEEP COOL"}]},
    }
    review = {
        "status": "MAIN_AGENT_REVIEWED_LITERAL_TARGET_CORRECTION",
        "documentId": "synthetic-1",
        "sourceDocumentId": "source-1",
        "path": "/goodsItemDetails/0/" + field,
        "before": module.ABSENT,
        "after": {"name": "PORTUGAL"} if field == "origin" else ["KEEP COOL"],
        "descriptionQuote": "PRODUCT OF PORTUGAL. KEEP COOL",
        "reason": "Reviewed whole-product origin or literal storage instruction.",
    }
    return rendered, review


@pytest.mark.parametrize("field", ["origin", "handlingInstructions"])
def test_only_reviewed_target_changes(field):
    rendered, review = fixture(field)
    original = copy.deepcopy(rendered)
    result = module.apply_review(rendered, review)
    assert rendered == original
    assert result["pages"] == original["pages"]
    assert result["target"]["goodsItemDetails"][0].pop(field) == review["after"]
    assert result == original


def test_noop_does_not_copy_or_read_files():
    rendered, _ = fixture()
    assert module.apply_literal_target_correction(rendered, None) is rendered
    assert module.correction_index({}) == {}


@pytest.mark.parametrize(
    "group,category", [("I", "HIGH_DANGER"), ("II", "MEDIUM_DANGER"), ("III", "LOW_DANGER")]
)
def test_explicit_packing_group_addition_preserves_other_dg_facts(group, category):
    rendered, review = fixture()
    goods = rendered["target"]["goodsItemDetails"][0]
    quote = f"HEXACHLOROACETONE (UN 2661, CLASS 6.1, PG {group})"
    goods.update(description=quote, dangerousGoods=[{"unNumber": "2661"}])
    review.update(
        path="/goodsItemDetails/0/dangerousGoods/0/packingGroupCategory",
        descriptionQuote=quote,
        after=category,
    )
    result = module.apply_review(rendered, review)
    assert result["target"]["goodsItemDetails"][0]["dangerousGoods"] == [
        {"unNumber": "2661", "packingGroupCategory": category}
    ]
    assert "packingGroupCategory" not in goods["dangerousGoods"][0]


@pytest.mark.parametrize(
    "quote,after",
    [
        ("PRODUCT PG III", "LOW_DANGER"),
        ("PRODUCT UN 1234 PG III", "LOW_DANGER"),
        ("UN 2661 PG III, UN 1234 PG II", "LOW_DANGER"),
        ("UN 2661 PG III", "HIGH_DANGER"),
    ],
)
def test_packing_group_review_rejects_missing_conflicting_or_wrong_identity(quote, after):
    rendered, review = fixture()
    rendered["target"]["goodsItemDetails"][0].update(
        description=quote, dangerousGoods=[{"unNumber": "2661"}]
    )
    review.update(
        path="/goodsItemDetails/0/dangerousGoods/0/packingGroupCategory",
        descriptionQuote=quote,
        after=after,
    )
    with pytest.raises(ValueError, match="matching UN/PG"):
        module.apply_review(rendered, review)


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "PROPOSED"),
        ("documentId", "another"),
        ("sourceDocumentId", "another"),
        ("path", "/parties/shipper/country"),
        ("before", {"name": "SPAIN"}),
        ("descriptionQuote", "A DIFFERENT PASSAGE"),
        ("after", {"name": "SPAIN"}),
        ("after", {"name": ""}),
        ("after", {"name": "PORTUGAL", "code": "PT"}),
        ("reason", " "),
    ],
)
def test_unreviewed_stale_or_nonliteral_changes_fail(key, value):
    rendered, review = fixture()
    review[key] = value
    with pytest.raises(ValueError):
        module.apply_review(rendered, review)


def test_existing_handling_is_preserved_and_not_duplicated():
    rendered, review = fixture("handlingInstructions")
    rendered["target"]["goodsItemDetails"][0]["handlingInstructions"] = ["NON STACKABLE"]
    review.update(before=["NON STACKABLE"], after=["NON STACKABLE", "KEEP COOL"])
    assert module.apply_review(rendered, review)["target"]["goodsItemDetails"][0][
        "handlingInstructions"
    ] == ["NON STACKABLE", "KEEP COOL"]
    for after in (["KEEP COOL"], ["NON STACKABLE", "INVENTED"], ["NON STACKABLE"]):
        review["after"] = after
        with pytest.raises(ValueError):
            module.apply_review(rendered, review)


def test_description_addition_composes_with_new_template_clause():
    rendered, review = fixture("handlingInstructions")
    rendered["target"]["goodsItemDetails"][0]["handlingInstructions"] = [
        "NON STACKABLE", "EXPRESS RELEASE"
    ]
    review.update(before=["NON STACKABLE"], after=["NON STACKABLE", "KEEP COOL"])
    result = module.apply_review(rendered, review)
    assert result["target"]["goodsItemDetails"][0]["handlingInstructions"] == [
        "NON STACKABLE", "EXPRESS RELEASE", "KEEP COOL"
    ]
    assert rendered["target"]["goodsItemDetails"][0]["handlingInstructions"] == [
        "NON STACKABLE", "EXPRESS RELEASE"
    ]


@pytest.mark.parametrize("retained", [[], ["EXPRESS RELEASE"], ["STACKABLE"],
    ["NON STACKABLE", "KEEP COOL"], ["NON STACKABLE", ""],
    ["NON STACKABLE", None], "NON STACKABLE"])
def test_handling_composition_rejects_removed_changed_or_already_applied_base(retained):
    rendered, review = fixture("handlingInstructions")
    rendered["target"]["goodsItemDetails"][0]["handlingInstructions"] = retained
    review.update(before=["NON STACKABLE"], after=["NON STACKABLE", "KEEP COOL"])
    with pytest.raises(ValueError, match="baseline changed"):
        module.apply_review(rendered, review)


def test_handling_composition_still_requires_literal_quote():
    rendered, review = fixture("handlingInstructions")
    rendered["target"]["goodsItemDetails"][0]["handlingInstructions"] = ["EXPRESS RELEASE"]
    rendered["target"]["goodsItemDetails"][0]["description"] = "OTHER GOODS"
    with pytest.raises(ValueError, match="retained description"):
        module.apply_review(rendered, review)


@pytest.mark.parametrize("changed", [None, "target", "pages", "positionedText", "bboxText"])
def test_published_amendment_requires_exact_replay(monkeypatch, changed):
    amendment = importlib.import_module("amend_r30_literal_targets")
    row, review = fixture()
    row.update(joinedRawText="PRODUCT OF PORTUGAL", positionedText="A", bboxText="B")
    replay = copy.deepcopy(row)
    if changed:
        replay[changed] = "changed"
    monkeypatch.setattr(amendment, "check_training_target", lambda target: [])
    monkeypatch.setattr(
        amendment,
        "apply_literal_target_correction",
        lambda value, pin: module.apply_review(value, review),
    )
    if changed:
        with pytest.raises(ValueError, match="exact pinned replay"):
            amendment.checked_repair(row, replay, {})
    else:
        fixed = amendment.checked_repair(row, replay, {})
        assert fixed["target"]["goodsItemDetails"][0]["origin"] == {"name": "PORTUGAL"}
        assert fixed["pages"] == row["pages"]
        assert "origin" not in row["target"]["goodsItemDetails"][0]
