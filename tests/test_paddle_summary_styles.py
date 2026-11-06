from pathlib import Path

import pytest

from document_ocr.synthesis.package_presentations import PackageVocabulary
from document_ocr.synthesis.paddle_render import extra_value


def evaluate(spec, category="PACKAGE_BALE", rows=None):
    vocabulary = PackageVocabulary({"version": 1, "excludedUnscopedTokens": [], "entries": {
        "PACKAGE_BALE": {"full": ["BALES"], "short": [], "optional": [], "family": "PACKAGE_BALE"},
        "PACKAGE_PACKAGE": {"full": ["PACKAGES"], "short": ["PKG"], "optional": [], "family": "PACKAGE_PACKAGE"},
    }})
    current = {"target": {"goodsItemDetails": [{"numberAndTypeOfPackages": rows or [
        {"typeCategory": category, "packageQuantity": 10}
    ]}]}}
    return extra_value({"packageSummary": {"goods": 0, "part": "type", **spec}}, {}, current,
                       "test", None, vocabulary, project_root=Path("."))


def test_full_word_is_used_only_when_explicitly_eligible():
    with pytest.raises(ValueError, match="no reviewed alias"):
        evaluate({"style": "short"})
    assert evaluate({"styles": ["short", "full"]}) == ("BALES", None)
    assert evaluate({"styles": ["short", "full"]}, "PACKAGE_PACKAGE") == ("PKG", None)


def test_mixed_group_summary_remains_compact_generic_package():
    rows = [{"typeCategory": "PACKAGE_BALE", "packageQuantity": 8},
            {"typeCategory": "PACKAGE_PACKAGE", "packageQuantity": 2}]
    assert evaluate({"styles": ["short", "full"]}, rows=rows) == ("PKG", None)


@pytest.mark.parametrize("spec", [
    {"style": "short", "styles": ["full"]}, {"styles": []},
    {"styles": ["short", "short"]}, {"styles": ["invented"]}, {"styles": "full"},
])
def test_ambiguous_or_invalid_style_contract_fails(spec):
    with pytest.raises(ValueError, match="explicitly eligible"):
        evaluate(spec)
