"""Current-data receipt composition must preserve boundaries and fail on gaps."""

import importlib.util
import sys
from pathlib import Path

import pytest

from document_ocr.synthesis.curated import digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
SPEC = importlib.util.spec_from_file_location(
    "r30_composition",
    Path(__file__).resolve().parents[1] / "scripts/migrations/compose_r30_current_receipts.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    source = b"Name: Original\n10 CARTONS\nFooter"
    rendered = "Name: Invented\n10 CARTONS\nFooter"
    candidate = {
        "sourceDocumentId": "source",
        "documentId": "sample",
        "joinedRawText": rendered,
        "scenario": {"goods": "retained"},
        "target": {"schemaVersion": "7.0.0", "documentPatch": {}},
        "proof": {
            "sourceSha256": digest(source),
            "renderedSha256": digest(rendered.encode()),
            "edits": [
                {
                    "key": "name",
                    "byteStart": 6,
                    "byteEnd": 14,
                    "before": "Original",
                    "after": "Invented",
                    "targetPaths": [],
                }
            ],
        },
    }
    current = {
        "sourceDocumentId": "source",
        "documentId": "sample",
        "joinedRawText": rendered,
        "target": {},
    }
    return candidate, current


def test_current_owned_value_replays_without_restoring_old_text():
    candidate, current = fixture()
    current["joinedRawText"] = current["joinedRawText"].replace("Invented", "Replacement")
    result = MODULE.compose(candidate, current)
    assert result["proof"]["edits"][0]["after"] == "Replacement"
    assert result["joinedRawText"] == current["joinedRawText"]
    assert result["scenario"] == candidate["scenario"]
    assert candidate["proof"]["edits"][0]["after"] == "Invented"


def test_unowned_context_change_is_held_not_discarded():
    candidate, current = fixture()
    current["joinedRawText"] = current["joinedRawText"].replace("Footer", "Other context")
    with pytest.raises(ValueError, match="outside the historical complete owners"):
        MODULE.compose(candidate, current)


def test_explicit_retained_id_mapping_is_bijective_and_does_not_allow_quantity_change():
    candidate, current = fixture()
    candidate["target"]["documentPatch"]["containerInformation"] = [{"equipmentIdentifier": "OLD123"}]
    current["target"]["containerInformation"] = [{"equipmentIdentifier": "KEPT456"}]
    with pytest.raises(ValueError, match="cargo facts changed"):
        MODULE.compose(candidate, current)
    result = MODULE.compose(candidate, current, retained_equipment_ids={"OLD123": "KEPT456"})
    assert result["target"]["documentPatch"] == current["target"]
    with pytest.raises(ValueError, match="bijection"):
        MODULE.compose(candidate, current, retained_equipment_ids={"OLD123": "WRONG"})
    candidate["target"]["documentPatch"]["goodsItemDetails"] = [{"numberAndTypeOfPackages": [{"packageQuantity": 10}]}]
    current["target"]["goodsItemDetails"] = [{"numberAndTypeOfPackages": [{"packageQuantity": 20}]}]
    with pytest.raises(ValueError, match="cargo facts changed"):
        MODULE.compose(candidate, current, retained_equipment_ids={"OLD123": "KEPT456"})


@pytest.mark.parametrize("field,value,surface", [
    ("goodsItemDetails[0].grossWeight.value", 77977.6, "77,977.60"),
    ("goodsItemDetails[0].volume.value", 21.8, "21.8000"),
    ("goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity", 660, "SIX HUNDRED SIXTY"),
    ("shippedOnBoardDate", "2024-03-13", "13-MAR-2024"),
])
def test_retained_semantic_surface_has_independent_target_witness(field,value,surface):
    assert MODULE.retained_semantic_literal(field,value,value,surface) == surface.upper()
    assert MODULE.retained_semantic_literal(field,value,None,surface) is None
    assert MODULE.retained_semantic_literal(field,value,value,"OTHER " + surface) is None


def test_word_count_cannot_absorb_noun_or_conflicting_number():
    field = "packages[0].packageQuantity"
    assert MODULE.retained_semantic_literal(field,660,660,"SIX HUNDRED SIXTY CARTONS") is None
    assert MODULE.retained_semantic_literal(field,660,661,"SIX HUNDRED SIXTY") is None
    assert MODULE.retained_semantic_literal(field,660,660,"SIX HUNDRED SIXTY ONE") is None


def deleted_owner_fixture():
    candidate, current = fixture()
    source = MODULE.reconstruct_source(candidate)
    candidate["proof"]["edits"].append({
        "key": "removed_statement", "byteStart": 15, "byteEnd": 25,
        "before": "10 CARTONS", "after": "", "targetPaths": [],
    })
    candidate["joinedRawText"] = "Name: Invented\n\nFooter"
    candidate["proof"]["renderedSha256"] = digest(candidate["joinedRawText"].encode())
    assert MODULE.reconstruct_source(candidate) == source
    current["joinedRawText"] = "Name: Replacement\n\nFooter"
    return candidate, current


def test_intentional_deleted_owner_survives_unrelated_current_repair():
    candidate, current = deleted_owner_fixture()
    result = MODULE.compose(candidate, current)
    assert result["proof"]["edits"][1]["after"] == ""
    assert result["joinedRawText"] == current["joinedRawText"]
    framed = MODULE.compose(candidate, current, method="literal_frames")
    assert framed["proof"]["edits"] == result["proof"]["edits"]


def test_deleted_owner_cannot_absorb_new_unowned_text():
    candidate, current = deleted_owner_fixture()
    current["joinedRawText"] = "Name: Replacement\nNEW STATEMENT\nFooter"
    with pytest.raises(ValueError):
        MODULE.compose(candidate, current)
    with pytest.raises(ValueError):
        MODULE.compose(candidate, current, method="literal_frames")


def test_changed_cargo_cannot_reuse_stale_scenario():
    candidate, current = fixture()
    candidate["target"]["documentPatch"] = {"goodsItemDetails": [{"grossWeight": {"value": 100}}]}
    current["target"] = {"goodsItemDetails": [{"grossWeight": {"value": 200}}]}
    with pytest.raises(ValueError, match="cargo facts changed"):
        MODULE.compose(candidate, current)


def compiled_fixture():
    candidate, current = fixture()
    source = MODULE.reconstruct_source(candidate)
    binding = {
        "logical_key": "private_reference", "value_kind": "identifier", "target_paths": [],
        "occurrences": [{"byte_start": source.index(b"Footer"), "byte_end": len(source), "source_text": "Footer", "target_paths": []}],
    }
    return candidate, current, {
        "document_id": "source", "source_sha256": digest(source), "bindings": [binding]
    }


def test_compiled_identifier_replays_retained_private_repair_without_mutation():
    import copy
    candidate, current, compiled = compiled_fixture()
    before = copy.deepcopy(candidate)
    current["joinedRawText"] = current["joinedRawText"].replace("Footer", "NewRef")
    result = MODULE.compose_with_compiled_identifiers(candidate, current, compiled)
    assert candidate == before
    assert result["joinedRawText"] == current["joinedRawText"]
    assert result["scenario"] == candidate["scenario"]
    assert result["target"]["documentPatch"] == current["target"]
    assert result["migrationComposition"]["addedCompiledIdentifierOwners"][0]["after"] == "NewRef"


@pytest.mark.parametrize("key,value", [("document_id", "other"), ("source_sha256", "wrong")])
def test_compiled_identifier_rejects_other_authority(key, value):
    candidate, current, compiled = compiled_fixture()
    compiled[key] = value
    with pytest.raises(ValueError, match="different source"):
        MODULE.compose_with_compiled_identifiers(candidate, current, compiled)


@pytest.mark.parametrize("level", ["binding", "occurrence"])
def test_compiled_identifier_does_not_admit_target_owners(level):
    candidate, current, compiled = compiled_fixture()
    binding = compiled["bindings"][0]
    (binding if level == "binding" else binding["occurrences"][0])["target_paths"] = ["some.target"]
    with pytest.raises(ValueError, match="no eligible"):
        MODULE.compose_with_compiled_identifiers(candidate, current, compiled)


def test_compiled_identifier_rejects_changed_protected_context():
    candidate, current, compiled = compiled_fixture()
    current["joinedRawText"] = current["joinedRawText"].replace("Footer", "NewRef").replace("CARTONS", "BARRELS")
    with pytest.raises(ValueError, match="outside the historical"):
        MODULE.compose_with_compiled_identifiers(candidate, current, compiled)


def test_compiled_identifier_source_equal_addition_is_not_new_native_owner():
    candidate, current, compiled = compiled_fixture()
    with pytest.raises(ValueError, match="do not explain"):
        MODULE.compose_with_compiled_identifiers(candidate, current, compiled)


def test_compiled_identifier_rejects_bad_source_span():
    candidate, current, compiled = compiled_fixture()
    compiled["bindings"][0]["occurrences"][0]["byte_start"] += 1
    with pytest.raises(ValueError, match="span differs"):
        MODULE.compose_with_compiled_identifiers(candidate, current, compiled)
