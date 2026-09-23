import pytest
from test_synthesis_dangerous_goods_realization import fact

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.template_compiler import dangerous_goods_packaging as packaging


def chemical(section="202"):
    return fact().record.model_copy(update={"source_fields": {"Non-bulk Packaging": section}})


@pytest.mark.parametrize("section", ["220", "219", "301", "185", "None", ""])
def test_special_articles_and_gases_never_enter_generic_drums(section):
    with pytest.raises(ValueError, match="specialized packaging"):
        packaging.profile(chemical(section))


def test_contradictory_pg_or_gas_record_cannot_inherit_a_liquid_profile():
    with pytest.raises(ValueError, match="specialized packaging"):
        packaging.profile(chemical("201"))
    with pytest.raises(ValueError, match="specialized packaging"):
        packaging.profile(chemical().model_copy(update={"exact_hazard_class": "2.1"}))


def test_large_unit_mass_requires_overpack_not_a_two_tonne_carton():
    signature, receipt = packaging.sample_packages(
        records=[chemical()],
        group={"grossWeight": {"value": 2000, "unit": "kilogram"}},
        packages=[{"quantity": 1, "typeCategory": "PACKAGE_PALLET"}],
        stream=DeterministicStream(5, "packaging-test", "case"),
    )
    assert signature == ("PACKAGE_PALLET",)
    assert receipt["minimumInnerUnitsPerOverpack"] == 5


def test_mixed_chemicals_without_package_allocation_are_explicitly_rejected():
    with pytest.raises(ValueError, match="separate-package allocation"):
        packaging.sample_packages(
            records=[chemical(), chemical()],
            group={},
            packages=[{"quantity": 10}],
            stream=DeterministicStream(5, "packaging-test", "case"),
        )


def test_liquid_boxes_have_inner_receptacles_not_uncontained_chemicals():
    signature, receipt = packaging.sample_packages(
        records=[chemical()],
        group={},
        packages=[{"quantity": 10, "typeCategory": "PACKAGE_DRUM"}],
        stream=DeterministicStream(5, "packaging-test", "case"),
    )
    assert len(signature) == 1
    assert "inner receptacles" in receipt["packagingScope"]
    assert receipt["sections"] == ["173.202"]


def test_individual_packages_are_not_promoted_to_pallet_overpacks():
    for seed in range(50):
        signature, _ = packaging.sample_packages(
            records=[chemical()],
            group={"grossWeight": {"value": 16000, "unit": "kilogram"}},
            packages=[{"quantity": 603, "typeCategory": "PACKAGE_DRUM"}],
            stream=DeterministicStream(seed, "packaging-test", "case"),
        )
        assert "PACKAGE_PALLET" not in signature
    with pytest.raises(ValueError, match="explicit source overpack"):
        packaging.sample_packages(
            records=[chemical()],
            group={"grossWeight": {"value": 2000, "unit": "kilogram"}},
            packages=[{"quantity": 1, "typeCategory": "PACKAGE_DRUM"}],
            stream=DeterministicStream(5, "packaging-test", "case"),
        )
