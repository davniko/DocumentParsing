from types import SimpleNamespace as NS
from typing import ClassVar

import pytest

from document_ocr.synthesis.thermal_goods import build_thermal_goods_support


class Registry:
    receipt = NS(snapshot_date=None)
    rows: ClassVar[dict[str, tuple[str, str]]] = {
        "020110": ("Fresh or chilled beef", "Meat"),
        "020210": ("Frozen beef", "Meat"),
        "071310": ("Dried peas", "Dried legumes"),
        "070200": ("Fresh tomatoes", "Tomatoes"),
        "392410": ("Plastic tableware", "Plastic articles"),
    }
    global_codes = tuple(rows)

    def require_global(self, code, *, on_date):
        description, heading = self.rows[code]
        return NS(
            description=description,
            heading_description=heading,
            chapter_description="Registry chapter",
            chapter_code=code[:2],
        )


def test_heading_policy_does_not_admit_the_rest_of_the_fresh_produce_chapter():
    support = build_thermal_goods_support(
        registry=Registry(), ambient_chapters=["39"], ambient_headings=["0713"]
    )
    assert {i.hs6 for i in support.ambient} == {"071310", "392410"}
    assert {i.hs6 for i in support.frozen} == {"020210"}
    assert {i.hs6 for i in support.chilled} == {"020110"}


def test_explicit_thermal_identity_is_not_changed_to_ambient_by_heading_policy():
    support = build_thermal_goods_support(
        registry=Registry(), ambient_chapters=["39"], ambient_headings=["0202"]
    )
    assert {i.hs6 for i in support.ambient} == {"392410"}


@pytest.mark.parametrize("heading", ["07", "071", "07130", "07AB"])
def test_invalid_heading_is_rejected(heading):
    with pytest.raises(ValueError, match="four-digit"):
        build_thermal_goods_support(
            registry=Registry(), ambient_chapters=["39"], ambient_headings=[heading]
        )
