from copy import deepcopy

import pytest

from document_ocr.synthesis.template_compiler import source_only_package_assertions as assertions


def _target(*quantities: int, category: str) -> dict:
    return {
        "documentPatch": {
            "cargoPackages": [
                {"quantity": quantity, "typeCategory": category} for quantity in quantities
            ]
        }
    }


def test_source_proved_product_counts_follow_changed_package_quantity() -> None:
    source = b"24 INTERMEDIATE BULK CONTAINERS\nA (5 IBC)\nB (6 IBC)\nC (13 IBC)\n"
    source_target = _target(24, category="PACKAGE_INTERMEDIATE_BULK_CONTAINER")
    target = _target(18, category="PACKAGE_INTERMEDIATE_BULK_CONTAINER")
    assertions.validate(
        source=source, rendered=source, source_target=source_target, target=source_target
    )
    with pytest.raises(ValueError, match="product package subtotals"):
        assertions.validate(
            source=source,
            rendered=source.replace(b"24 INTERMEDIATE", b"18 INTERMEDIATE"),
            source_target=source_target,
            target=target,
        )
    assertions.validate(
        source=source,
        rendered=source.replace(b"24 INTERMEDIATE", b"18 INTERMEDIATE").replace(
            b"13 IBC", b"7 IBC"
        ),
        source_target=source_target,
        target=target,
    )
    changed_category = deepcopy(target)
    changed_category["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PALLET"
    with pytest.raises(ValueError, match="product package subtotals"):
        assertions.validate(
            source=source, rendered=source, source_target=source_target, target=changed_category
        )


def test_retained_source_proved_equation_follows_sampled_inner_count() -> None:
    source = (
        b"TOTAL 44 PALLETS ONLY\n42 PALLETS X 165 PCS = 6930 PCS\n2 PALLET X 80 PCS = 160 PCS\n"
    )
    source_target = _target(7090, category="PACKAGE_PIECE")
    target = _target(5651, category="PACKAGE_PIECE")
    assertions.validate(
        source=source, rendered=source, source_target=source_target, target=source_target
    )
    with pytest.raises(ValueError, match="package equation"):
        assertions.validate(
            source=source, rendered=source, source_target=source_target, target=target
        )
    # A descendant may remove both optional equations while retaining a physically
    # possible outer pallet declaration; the guard must not invent a new equation.
    assertions.validate(
        source=source,
        rendered=b"TOTAL 44 PALLETS ONLY\n5651 PCS\n",
        source_target=source_target,
        target=target,
    )
    with pytest.raises(ValueError, match="outer package count"):
        assertions.validate(
            source=source,
            rendered=b"TOTAL 44 PALLETS ONLY\n21 PCS\n",
            source_target=source_target,
            target=_target(21, category="PACKAGE_PIECE"),
        )
    with pytest.raises(ValueError, match="pallet target"):
        assertions.validate(
            source=source,
            rendered=b"TOTAL 44 PALLETS ONLY\n4289 PALLETS\n",
            source_target=source_target,
            target=_target(4289, category="PACKAGE_PALLET"),
        )
    assertions.validate(
        source=source,
        rendered=b"TOTAL 44 PALLETS ONLY\n",
        source_target=source_target,
        target=_target(44, category="PACKAGE_PALLET"),
    )


@pytest.mark.parametrize("malformed", [b"PALLET NO:5-2", b"PALLET NO:0-9", b"PALLET NO:1-0"])
def test_source_proved_pallet_mark_range_must_remain_well_formed(malformed: bytes) -> None:
    source = b"PALLET NO:1-12\n12 CARTONS\n"
    target = _target(12, category="PACKAGE_CARTON")
    with pytest.raises(ValueError, match="pallet mark range"):
        assertions.validate(
            source=source,
            rendered=malformed + b"\n12 CARTONS\n",
            source_target=target,
            target=target,
        )
    assertions.validate(
        source=source,
        rendered=b"PALLET NO:5-12\n12 CARTONS\n",
        source_target=target,
        target=target,
    )
    assertions.validate(
        source=source,
        rendered=b"PALLET NO:1-08\n12 CARTONS\n",
        source_target=target,
        target=target,
    )


def test_unproved_count_does_not_impose_product_subtotal_contract() -> None:
    source = b"24 INTERMEDIATE BULK CONTAINERS\nA (5 IBC)\n"
    source_target = _target(24, category="PACKAGE_INTERMEDIATE_BULK_CONTAINER")
    target = _target(18, category="PACKAGE_INTERMEDIATE_BULK_CONTAINER")
    assertions.validate(
        source=source,
        rendered=b"18 INTERMEDIATE BULK CONTAINERS\nA (5 IBC)\n",
        source_target=source_target,
        target=target,
    )


def test_counted_outer_pallet_clause_must_not_render_incompletely() -> None:
    source = b"72 DRUMS\nLOADED ONTO\n28 PALLETS LOADED INTO\n1 CONTAINER\n"
    source_target = _target(72, category="PACKAGE_DRUM")
    sampled_target = _target(67, category="PACKAGE_DRUM")
    malformed = b"67 DRUMS\nLOADED\nONTO PALLETS\nINTO 1 CONTAINER\n"
    with pytest.raises(ValueError, match="outer pallet clause lost its count"):
        assertions.validate(
            source=source,
            rendered=malformed,
            source_target=source_target,
            target=sampled_target,
        )
    assertions.validate(
        source=source,
        rendered=b"67 DRUMS\nLOADED\nINTO 1 CONTAINER\n",
        source_target=source_target,
        target=sampled_target,
    )
    assertions.validate(
        source=b"72 DRUMS\nON PALLETS\n",
        rendered=b"67 DRUMS\nON PALLETS\n",
        source_target=source_target,
        target=sampled_target,
    )


def test_source_box_pallet_hierarchy_does_not_justify_new_carton_level() -> None:
    source = b"1 x 40RH 1494 BOXES\nH.S. CODE: 08081080\nON 21 PALLETS\n"
    source_target = _target(1494, category="PACKAGE_BOX")
    sampled_target = _target(1052, category="PACKAGE_BOX")
    with pytest.raises(ValueError, match="carton overpack is unsupported"):
        assertions.validate(
            source=source,
            rendered=(b"1 x 40RH 1052 BOXES\nH.S. CODE: 08081080\nPACKED IN EXPORT CARTONS\n"),
            source_target=source_target,
            target=sampled_target,
        )
    assertions.validate(
        source=source,
        rendered=b"1 x 40RH 1052 BOXES\nH.S. CODE: 08081080\nPACKED FOR EXPORT\n",
        source_target=source_target,
        target=sampled_target,
    )


def test_source_proved_total_items_follow_one_package_without_new_carton_level() -> None:
    source = (
        b"5760 BOX(ES) of FRESH AVOCADOS\nTotal Items: 5760\n"
        b"5760 BOX(ES) of FRESH AVOCADOS\nTotal Items: 5760\n"
    )
    original = _target(5760, category="PACKAGE_BOX")
    sampled = _target(5443, category="PACKAGE_BOX")
    assertions.validate(source=source, rendered=source, source_target=original, target=original)
    with pytest.raises(ValueError, match="Total Items disagree"):
        assertions.validate(
            source=source,
            rendered=source.replace(b"5760 BOX(ES)", b"5443 BOX(ES)"),
            source_target=original,
            target=sampled,
        )
    revised = source.replace(b"5760", b"5443")
    assertions.validate(source=source, rendered=revised, source_target=original, target=sampled)
    with pytest.raises(ValueError, match="unsupported counted cartons"):
        assertions.validate(
            source=source,
            rendered=revised.replace(b"FRESH AVOCADOS", b"FRESH AVOCADOS, 5443 CARTONS"),
            source_target=original,
            target=sampled,
        )
    # Package-category conversion remains legal when it is reflected in print.
    assertions.validate(
        source=source,
        rendered=revised.replace(b"BOX(ES)", b"CARTON(S)"),
        source_target=original,
        target=_target(5443, category="PACKAGE_CARTON"),
    )
    # A bare matching "Total Items" line without a package-bearing source line
    # is not enough evidence to create a package dependency.
    assertions.validate(
        source=b"Total Items: 5760\n",
        rendered=b"Total Items: 17\n",
        source_target=original,
        target=sampled,
    )
