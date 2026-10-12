"""Complete equipment alias selection and source-owned render framing."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.container_semantics import review_source_equipment_surface
from document_ocr.synthesis.curated_equipment_aliases import (
    EquipmentAliasRegistry,
    apply_equipment_aliases,
    validate_equipment_alias,
)
from document_ocr.synthesis.generators import DeterministicStream

S20 = "TWENTY_FOOT_STANDARD_HEIGHT"
H20 = "TWENTY_FOOT_HIGH_CUBE"
S40 = "FORTY_FOOT_STANDARD_HEIGHT"
H40 = "FORTY_FOOT_HIGH_CUBE"
H45 = "FORTY_FIVE_FOOT_HIGH_CUBE"
GP = "GENERAL_PURPOSE"
RF = "REFRIGERATED"


@pytest.mark.parametrize(
    "surface,size,kind",
    [
        ("20HC", H20, GP),
        ("20HQ", H20, GP),
        ("3X20HC", H20, GP),
        ("2 x 25G1", H20, GP),
        ("1X55G1", H45, GP),
        ("2 x L5G1", H45, GP),
        ("2 x 25 G1\tfcl/fcl  ", H20, GP),
        ("2 X HC40", H40, GP),
        ("2 x RF40", S40, RF),
        ("25G1 FCL/FCL CY/CY", H20, GP),
        ("40RF", S40, RF),
        ("40RF 9'6", H40, RF),
        ("40 DRY 9x6", H40, GP),
        (" 2 x 40 DRY 9x6  ", H40, GP),
        ("(2 x 40 DRY 9x6)", H40, GP),
        ("40/DRY 9 x 6", H40, GP),
        ("40DRY 9x6", H40, GP),
        ("40RF 9x6", H40, RF),
        ("20 DRY 9x6", H20, GP),
        ("20RF 9x6", H20, RF),
        ("45 REEFER 9x6", H45, RF),
        ("40 DRY 8x6", S40, GP),
        ("40RF 8x6", S40, RF),
        ("20 DRY 8x6", S20, GP),
        ("20RF 8x6", S20, RF),
        ("40 FLAT 9x6", H40, "PLATFORM"),
        ("40 DRY 9FT6IN", H40, GP),
        ("40 9x6 REEFER", H40, RF),
        ("40'9'6\" DRY CONTAINER", H40, GP),
        ("40\u20199\u20196\u201d REEFER", H40, RF),
        ("40X9'6\" REEFER CONTAINER", H40, RF),
        ("20 X 9'6\" REEFER CONTAINER", H20, RF),
        ("20'", S20, GP),
        ("40'", S40, GP),
    ],
)
def test_complete_aliases_apply_standard_default_and_explicit_height(surface, size, kind):
    reviewed = review_source_equipment_surface(surface, temperature_present=False)
    assert (reviewed.size_category, reviewed.type_category) == (size, kind)
    assert reviewed.printed_surface == surface
    validate_equipment_alias(surface, size, kind)


@pytest.mark.parametrize(
    "surface",
    [
        "20HC 8'6",
        "20HQ STANDARD HEIGHT",
        "40RF 8'6 9'6",
        "40' STANDARD HEIGHT HIGH CUBE",
        "40HO 8'6",
        "L5G1 8'6",
        "25G1 8'6",
        "42G1 HIGH CUBE",
        "40RQ 8'6",
        "22GO 9'6",
        "40HC 8x6",
        "20HQ 8 x 6",
        "40RF 8x6 9x6",
        "40RF 9x6 8x6",
        "40HO 8x6",
        "L5G1 8x6",
        "25G1 8x6",
        "22GO 9x6",
        "45 DRY 8x6",
        "45RF 8'6",
        " 45 DRY 8x6 ",
        " 2 x 45RF 8x6 ",
        "(2 x 40HC 8x6)",
        "45\u00d78x6 DRY",
        "40X9'6\" REEFER HEIGHT 8'6",
        "40X9'6\" DRY 8x6",
        "40'X8'X9'6\" DRY HEIGHT 8x6",
    ],
)
def test_conflicting_explicit_height_rejects_every_target(surface):
    reviewed = review_source_equipment_surface(surface, temperature_present=False)
    assert reviewed.size_category is None and reviewed.type_category is None
    assert reviewed.review_rule == "conflicting_explicit_height_signals"
    with pytest.raises(ValueError, match="conflicts"):
        validate_equipment_alias(surface, H40, GP)


@pytest.mark.parametrize(
    "surface",
    [
        "1X45R9",
        "1X55G1SEAL",
        "2X20HQ1234567",
        "120HC",
        "20HCX",
        "SEAL20HC",
        "1X25G1 UNKNOWN",
        "1X25G1 FCL/FCL EXTRA",
        "1X25G1 FCL/FCL FCL/FCL",
        "1X25G1 CY/UNKNOWN",
        "40X9'67 REEFER",
        "40X9x60 DRY",
        "20X7'6 REEFER",
    ],
)
def test_opaque_or_malformed_equipment_is_not_unwrapped_into_a_pair(surface):
    reviewed = review_source_equipment_surface(surface, temperature_present=False)
    assert reviewed.size_category is None and reviewed.type_category is None


@pytest.mark.parametrize("length,size", [("20", H20), ("40", H40), ("45", H45)])
@pytest.mark.parametrize("noun,kind", [("DRY", GP), ("REEFER", RF)])
@pytest.mark.parametrize(
    "height", ["9'6", "9\u20196\u201d", "9\u20326\u2033", "9\u20186\u201c", "9 x 6", "9\u00d76"]
)
def test_printed_height_spelling_variants_keep_the_same_equipment_pair(
    length, size, noun, kind, height
):
    value = review_source_equipment_surface(f"{length} {noun} {height}", temperature_present=False)
    assert (value.size_category, value.type_category) == (size, kind)
    assert "default_standard_height" not in value.review_rule


@pytest.mark.parametrize(
    "surface,size",
    [
        ("40 DRY WIDTH 9x6 HEIGHT 8x6", S40),
        ("40 DRY WIDTH 8x6 HEIGHT 9x6", H40),
        ("40 DRY WIDTH 9'6 HEIGHT 8'6", S40),
        ("40 DRY HEIGHT 9x6 WIDTH 8x6", H40),
        ("40 HIGH CUBE WIDTH 8x6", H40),
        ("40'X9'6\"X8'6\" DRY CONTAINER", S40),
        ("40'X8'X9'6\" DRY CONTAINER", H40),
        ("40'X8'6\"X9x6 REEFER CONTAINER", H40),
        ("40X9'6\" REEFER WIDTH 8x6", H40),
        ("40 DRY 9x6 WIDTH 8x6", H40),
    ],
)
def test_width_does_not_impersonate_height_or_create_a_conflict(surface, size):
    reviewed = review_source_equipment_surface(surface, temperature_present=False)
    assert reviewed.size_category == size
    assert reviewed.type_category == (RF if "REEFER" in surface else GP)


@pytest.mark.parametrize(
    "surface",
    [
        "40 DRY PIPE SIZE 9x6",
        "40 DRY PIPE HEIGHT 9x6",
        "40 DRY WIDTH 9x6",
        "40 DRY 9x6 PIPE",
        "40 DRY 9x6 PRODUCT DIMENSIONS",
        "PRODUCT 9x6 40 DRY",
    ],
)
def test_unrelated_product_dimensions_cannot_promote_equipment_height(surface):
    reviewed = review_source_equipment_surface(surface, temperature_present=False)
    assert (reviewed.size_category, reviewed.type_category) == (S40, GP)
    assert "default_standard_height" in reviewed.review_rule


@pytest.mark.parametrize("surface", ["9x6", "8'6", "9\u20326\u2033", "HEIGHT 9x6", "PIPE 9x6"])
def test_height_only_and_product_dimensions_do_not_invent_length_or_type(surface):
    reviewed = review_source_equipment_surface(surface, temperature_present=False)
    assert reviewed.size_category is None and reviewed.type_category is None


@pytest.mark.parametrize(
    "surface,size",
    [
        ("20'X8x6", S20),
        ("20'X9x6", H20),
        ("20X9x6", H20),
        ("40 X 9'6", H40),
        ("40\u2032\u00d79\u20326\u2033", H40),
        ("40'X8'X9'6\"", H40),
        ("40'X9'6\"X8'6\"", S40),
        ("45'X9x6", H45),
    ],
)
def test_dimensional_size_uses_last_axis_without_inventing_equipment_type(surface, size):
    from document_ocr.synthesis.container_semantics import dimensional_equipment_size

    assert dimensional_equipment_size(surface) == size
    observed = review_source_equipment_surface(surface, temperature_present=False)
    assert observed.type_category is None


@pytest.mark.parametrize("surface,kind", [("40 DRY 9x6", GP), ("40RF 9x6", RF)])
def test_label_normalization_uses_owned_ocr_height_without_mutating_source(surface, kind):
    from document_ocr.labeling_agents.equipment_normalization import reconcile_equipment_categories

    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "containerInformation": [
                {"equipmentIdentifier": "ABCU1234567", "typeDescription": surface}
            ]
        },
    }
    before = deepcopy(target)
    raw = f"ABCU1234567 SEAL0001 {surface} 20 CARTONS\nPRODUCT DIMENSIONS 8x6\n"
    repaired, decisions = reconcile_equipment_categories(target, source_text=raw)
    assert repaired["documentPatch"]["containerInformation"] == [
        {"equipmentIdentifier": "ABCU1234567", "sizeCategory": H40, "typeCategory": kind}
    ]
    assert decisions[0]["source_evidence"] == [surface]
    assert target == before


@pytest.mark.parametrize("temperature", [False, True])
@pytest.mark.parametrize("surface", ["20RF 9x6", "40RF 9x6", "45 REEFER 9x6"])
def test_height_alias_does_not_invent_a_temperature_instruction(surface, temperature):
    observed = review_source_equipment_surface(surface, temperature_present=temperature)
    assert observed.type_category == RF
    assert observed.thermal_operation == ("active" if temperature else "not_indicated")


def registry_data():
    return {
        "version": 1,
        "entries": [
            {"sizeCategory": H20, "typeCategory": GP, "aliases": ["20HC", "20HQ", "25G1"]},
            {"sizeCategory": S40, "typeCategory": RF, "aliases": ["40RF", "42R0"]},
        ],
    }


def test_registry_is_complete_pair_checked_and_deterministic():
    registry = EquipmentAliasRegistry.from_data(registry_data())
    draws = [registry.select(H20, GP, DeterministicStream(17, "alias", str(i))) for i in range(100)]
    assert set(draws) == {"20HC", "20HQ", "25G1"}
    assert draws == [
        registry.select(H20, GP, DeterministicStream(17, "alias", str(i))) for i in range(100)
    ]
    with pytest.raises(ValueError, match="no entry"):
        registry.select(H40, GP, DeterministicStream(17, "alias", "missing"))


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_pair",
        "duplicate_alias",
        "duplicate_pair",
        "unknown_field",
        "empty",
        "newline",
        "counted",
        "movement",
    ],
)
def test_registry_rejects_unreviewed_or_invalid_data(defect):
    data = registry_data()
    if defect == "wrong_pair":
        data["entries"][0]["aliases"] = ["20GP"]
    elif defect == "duplicate_alias":
        data["entries"][0]["aliases"] = ["20HC", "20HC"]
    elif defect == "duplicate_pair":
        data["entries"].append(deepcopy(data["entries"][0]))
    elif defect == "unknown_field":
        data["unreviewedFallback"] = "40HC"
    elif defect == "empty":
        data["entries"][0]["aliases"] = []
    elif defect == "newline":
        data["entries"][0]["aliases"] = ["20HC\n25G1"]
    else:
        data["entries"][0]["aliases"] = ["2X20HC" if defect == "counted" else "20HC FCL/FCL"]
    with pytest.raises(ValueError):
        EquipmentAliasRegistry.from_data(data)


def test_production_registry_all_aliases_are_valid_complete_pairs():
    path = Path(__file__).parents[1] / "artifacts/sampling-registries/equipment-aliases.json"
    registry = EquipmentAliasRegistry.from_data(json.loads(path.read_text()))
    assert len(registry.entries) == 95
    assert sum(len(entry.aliases) for entry in registry.entries) == 152
    for entry in registry.entries:
        for alias in entry.aliases:
            validate_equipment_alias(alias, entry.sizeCategory, entry.typeCategory)


def blueprint(
    originals=("(2X40HC FCL/FCL)",), indices=(0, 1), fields=("sizeCategory", "typeCategory")
):
    old = [{"sizeCategory": H40, "typeCategory": GP} for _ in range(3)]
    old[2] = {"sizeCategory": H40, "typeCategory": RF}
    paths = [
        f"documentPatch.containerInformation[{i}].{field}" for i in indices for field in fields
    ]
    bindings = {
        "equipment": {
            "target_paths": paths,
            "occurrences": [{"source_text": value} for value in originals],
        }
    }
    return SimpleNamespace(
        target={"documentPatch": {"containerInformation": old}},
        historical_bindings=bindings,
        regions=[SimpleNamespace(key="equipment", curated_key=None)],
    )


def sampled_target(bp):
    target = deepcopy(bp.target)
    target["documentPatch"]["containerInformation"][0] = {"sizeCategory": H20, "typeCategory": GP}
    target["documentPatch"]["containerInformation"][1] = {"sizeCategory": H20, "typeCategory": GP}
    return target


@pytest.mark.parametrize(
    "source",
    [
        "(2X40HC FCL/FCL)",
        " 2 x 45G1 CY/CY ",
        "40HC FCL",
        "40HC LCL",
        "40HC CY",
        "2 X HC40",
        "(2 x HC40 CY/CY)",
    ],
)
def test_alias_override_keeps_count_movement_and_documentary_framing(source):
    bp = blueprint((source,))
    target = sampled_target(bp)
    before = deepcopy(target)
    surfaces = {"equipment": ["ordinary surface"], "count_only": ["2"], "unrelated": ["keep"]}
    result, receipt = apply_equipment_aliases(bp, target, surfaces, "20HQ")
    assert result["equipment"] == [
        source.replace("40HC", "20HQ").replace("45G1", "20HQ").replace("HC40", "20HQ")
    ]
    assert result["count_only"] == ["2"] and result["unrelated"] == ["keep"]
    assert surfaces["equipment"] == ["ordinary surface"]
    assert target == before
    assert receipt[0]["referencedContainerIndices"] == [0, 1]
    observed = review_source_equipment_surface(result["equipment"][0], temperature_present=False)
    assert (observed.size_category, observed.type_category) == (H20, GP)


def test_dimensional_alias_rendering_preserves_owned_count_and_movement():
    bp = blueprint(("2 x 40 DRY 9x6 FCL/FCL",))
    target = sampled_target(bp)
    before = deepcopy(target)
    surfaces = {"equipment": ["ordinary"], "unrelated": ["PRODUCT DIMENSIONS 8x6"]}
    rendered, receipts = apply_equipment_aliases(bp, target, surfaces, "20 DRY 9x6")
    assert rendered == {
        "equipment": ["2 x 20 DRY 9x6 FCL/FCL"],
        "unrelated": ["PRODUCT DIMENSIONS 8x6"],
    }
    assert receipts[0]["changedOccurrences"] == [0]
    assert target == before and surfaces["equipment"] == ["ordinary"]


def test_partial_occurrence_and_unrelated_reefer_do_not_constrain_complete_owner():
    bp = blueprint(("40HC", "40'"), indices=(0,))
    target = sampled_target(bp)
    result, receipts = apply_equipment_aliases(
        bp, target, {"equipment": ["ordinary", "20'"]}, "25G1"
    )
    assert result == {"equipment": ["25G1", "20'"]}
    assert receipts[0]["changedOccurrences"] == [0]
    assert receipts[0]["preservedPartialOccurrences"] == [1]
    assert receipts[0]["referencedContainerIndices"] == [0]


def test_scalar_surface_is_not_split_into_characters():
    bp = blueprint()
    result, _ = apply_equipment_aliases(
        bp, sampled_target(bp), {"equipment": "ordinary", "unrelated": "keep as text"}, "20HQ"
    )
    assert result["equipment"] == ["(2X20HQ FCL/FCL)"]
    assert result["unrelated"] == "keep as text"


def test_size_only_owner_is_not_upgraded_to_full_alias():
    bp = blueprint(("40'",), fields=("sizeCategory",))
    with pytest.raises(ValueError, match="no complete owned"):
        apply_equipment_aliases(bp, sampled_target(bp), {"equipment": ["20'"]}, "20HQ")


@pytest.mark.parametrize("source", ["2 40HC", "2 x (40HC)", "2 HC40", "2 x (HC40)"])
def test_unreviewed_count_framing_is_not_silently_deleted(source):
    bp = blueprint((source,))
    with pytest.raises(ValueError, match="unsupported source framing"):
        apply_equipment_aliases(bp, sampled_target(bp), {"equipment": ["ordinary"]}, "20HQ")


@pytest.mark.parametrize("source", ["2 X INV40", "2 X PO40", "2 X SEAL40"])
def test_foreign_codes_cannot_authorize_equipment_count_framing(source):
    bp = blueprint((source,))
    with pytest.raises(ValueError, match="no complete owned"):
        apply_equipment_aliases(bp, sampled_target(bp), {"equipment": ["ordinary"]}, "20HQ")


def test_alias_sampling_preserves_reverse_count_and_leaves_unowned_lookalikes():
    registry = EquipmentAliasRegistry.from_data(
        json.loads(Path("artifacts/sampling-registries/equipment-aliases.json").read_text())
    )
    bp = blueprint(("2 X HC40",))
    target = sampled_target(bp)
    foreign = {"reference": "2 X HC40", "phone": "40 X 9", "product": "9x6"}
    for seed in range(20):
        alias = registry.select(H20, GP, DeterministicStream(17, "reverse-count", str(seed)))
        result, receipt = apply_equipment_aliases(
            bp, target, {"equipment": ["ordinary"], **foreign}, alias
        )
        assert result["equipment"] == ["2 X " + alias]
        assert all(result[key] == value for key, value in foreign.items())
        observed = review_source_equipment_surface(
            result["equipment"][0], temperature_present=False
        )
        assert (observed.size_category, observed.type_category) == (H20, GP)
        assert receipt[0]["referencedContainerIndices"] == [0, 1]


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_occurrence",
        "different_source_pair",
        "wrong_source_label",
        "missing_surface",
        "count_mismatch",
        "wrong_alias",
        "mixed_target",
        "bad_index",
        "inactive_owner",
    ],
)
def test_alias_override_fails_loudly_when_owner_or_pair_is_unsound(defect):
    bp = blueprint()
    target = sampled_target(bp)
    surfaces = {"equipment": ["ordinary"]}
    alias = "20HC"
    if defect in {"unknown_occurrence", "different_source_pair"}:
        bp.historical_bindings["equipment"]["occurrences"].append(
            {"source_text": "opaque" if defect == "unknown_occurrence" else "40RF"}
        )
        surfaces["equipment"].append("ordinary")
    elif defect == "wrong_source_label":
        bp.target["documentPatch"]["containerInformation"][0]["typeCategory"] = RF
    elif defect == "missing_surface":
        surfaces = {}
    elif defect == "count_mismatch":
        surfaces["equipment"].append("extra")
    elif defect == "wrong_alias":
        alias = "20GP"
    elif defect == "mixed_target":
        target["documentPatch"]["containerInformation"][1]["sizeCategory"] = S20
    elif defect == "bad_index":
        bp.historical_bindings["equipment"]["target_paths"] = [
            f"documentPatch.containerInformation[9].{field}"
            for field in ("sizeCategory", "typeCategory")
        ]
    else:
        bp.regions[0].curated_key = "replaced"
    with pytest.raises(ValueError):
        apply_equipment_aliases(bp, target, surfaces, alias)
