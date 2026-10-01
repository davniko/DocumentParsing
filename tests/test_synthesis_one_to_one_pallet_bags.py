"""Source and descendant proofs for a printed private pallet/bag level."""

from __future__ import annotations

import re
from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.one_to_one_pallet_bags import (
    certify,
    validate_descendant,
)

SOURCE = b"Description of Packages and Goods\n60 PACKAGES\n60 PALLETS ( 60 BAGS )\n"


def _binding(
    key: str,
    text: str,
    *,
    mode: str,
    paths: tuple[str, ...] = (),
    derivation: str | None = None,
    dependencies: tuple[str, ...] = (),
    group_key: str = "package:0",
) -> SimpleNamespace:
    row = re.search(rb"(60) (PALLETS) \( (60) (BAGS) \)", SOURCE)
    assert row is not None
    group = (
        1
        if key == "derived:outer_pallet_count_equals_bag_quantity"
        else 2
        if key == "static:outer_pallet_category"
        else 3
        if key.endswith(".quantity")
        else 4
    )
    start = row.start(group)
    slot = SimpleNamespace(byte_start=start, byte_end=start + len(text), source_text=text)
    return SimpleNamespace(
        logical_key=key,
        render_mode=mode,
        derivation=derivation,
        dependency_bindings=dependencies,
        target_paths=paths,
        dependency_paths=(),
        group_key=group_key,
        occurrences=(slot,),
    )


def _fixture() -> tuple[list[SimpleNamespace], dict]:
    quantity = "documentPatch.cargoPackages[0].quantity"
    category = "documentPatch.cargoPackages[0].typeCategory"
    bindings = [
        _binding(
            "derived:outer_pallet_count_equals_bag_quantity",
            "60",
            mode="deterministic_derived",
            derivation="same_as_binding",
            dependencies=("anchor:" + quantity,),
            group_key="package:0:outer_pallets",
        ),
        _binding(
            "static:outer_pallet_category",
            "PALLETS",
            mode="literal_static",
            group_key="package:0:outer_pallets",
        ),
        _binding("anchor:" + quantity, "60", mode="target_binding", paths=(quantity,)),
        _binding("anchor:" + category, "BAGS", mode="target_binding", paths=(category,)),
    ]
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1"}],
            "cargoPackages": [
                {"packageId": "p1", "groupId": "g1", "quantity": 60, "typeCategory": "PACKAGE_BAG"}
            ],
        }
    }
    return bindings, target


def test_source_and_changed_descendant_have_exact_private_packing_contract() -> None:
    bindings, source_target = _fixture()
    constraint = certify(source=SOURCE, bindings=bindings, source_target=source_target)
    assert constraint is not None
    assert (constraint.package_index, constraint.source_quantity, constraint.inner_category) == (
        0,
        60,
        "PACKAGE_BAG",
    )
    changed = deepcopy(source_target)
    changed["documentPatch"]["cargoPackages"][0]["quantity"] = 73
    validate_descendant(
        source=SOURCE,
        rendered=SOURCE.replace(b"60 PACKAGES", b"73 PACKAGES").replace(
            b"60 PALLETS ( 60 BAGS )", b"73 PALLETS ( 73 BAGS )"
        ),
        bindings=bindings,
        source_target=source_target,
        target=changed,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_category", "not certified"),
        ("stale_outer", "contradicts target"),
        ("stale_inner", "contradicts target"),
    ],
)
def test_invalid_descendant_fails_closed(mutation: str, message: str) -> None:
    bindings, source_target = _fixture()
    changed = deepcopy(source_target)
    changed["documentPatch"]["cargoPackages"][0]["quantity"] = 73
    rendered = SOURCE.replace(b"60 PACKAGES", b"73 PACKAGES").replace(
        b"60 PALLETS ( 60 BAGS )", b"73 PALLETS ( 73 BAGS )"
    )
    if mutation == "wrong_category":
        changed["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_DRUM"
    elif mutation == "stale_outer":
        rendered = rendered.replace(b"73 PALLETS", b"60 PALLETS")
    else:
        rendered = rendered.replace(b"73 BAGS", b"60 BAGS")
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_descendant(
            source=SOURCE,
            rendered=rendered,
            bindings=bindings,
            source_target=source_target,
            target=changed,
        )


def test_missing_inner_owner_and_mismatched_printed_counts_fail_closed() -> None:
    bindings, source_target = _fixture()
    with pytest.raises(ValueError, match="required binding"):
        certify(source=SOURCE, bindings=bindings[:-1], source_target=source_target)
    broken = deepcopy(source_target)
    broken["documentPatch"]["cargoPackages"][0]["quantity"] = 61
    with pytest.raises(ValueError, match="do not equal"):
        certify(source=SOURCE, bindings=bindings, source_target=broken)
