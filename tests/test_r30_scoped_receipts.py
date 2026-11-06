import copy
import importlib.util
from pathlib import Path

import pytest

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits

spec = importlib.util.spec_from_file_location(
    "r30_scoped_receipts",
    Path(__file__).resolve().parents[1] / "scripts/migrations/r30_scoped_receipts.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    source = b"NAME|1 PALLET|GOODS"

    def edit(key, lo, hi, after, path):
        return {
            "key": key,
            "occurrence": 0,
            "byteStart": lo,
            "byteEnd": hi,
            "before": source[lo:hi].decode(),
            "after": after,
            "targetPaths": [path],
        }

    root = "documentPatch.goodsItemDetails[0]."

    def candidate(identity, name, equipment, rows, edits):
        quantity = sum(r["packageQuantity"] for r in rows)
        target = {
            "parties": {"shipper": {"name": name}},
            "containerInformation": [
                {
                    "equipmentIdentifier": equipment,
                    "sealNumbers": [identity],
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ],
            "goodsItemDetails": [
                {
                    "description": "BOLTS",
                    "origin": {"name": "FRANCE"},
                    "numberAndTypeOfPackages": rows,
                    "splitGoodsPlacement": [
                        {"equipmentIdentifier": equipment, "packageQuantity": quantity}
                    ],
                }
            ],
        }
        raw = apply_edits(source, edits).decode()
        return {
            "documentId": identity,
            "sourceDocumentId": "source",
            "joinedRawText": raw,
            "target": {"documentPatch": target},
            "proof": {
                "edits": edits,
                "sourceSha256": digest(source),
                "renderedSha256": digest(raw.encode()),
            },
            "physical": {},
            "scenario": {
                "cargo": {
                    "goodsItemDetails": [
                        {
                            **copy.deepcopy(target["goodsItemDetails"][0]),
                            "splitGoodsPlacement": [
                                {"equipmentIdentifier": "SOURCE123", "packageQuantity": quantity}
                            ],
                        }
                    ],
                    "containerInformation": [
                        {
                            **copy.deepcopy(target["containerInformation"][0]),
                            "equipmentIdentifier": "SOURCE123",
                            "sealNumbers": ["SOURCE-SEAL"],
                        }
                    ],
                },
                "goods_identities": ["bolts"],
                "origin": {"country": "FRANCE"},
                "provenance": {},
            },
        }

    old = candidate(
        "old",
        "RETAINED",
        "OLD123",
        [
            {"packageQuantity": 2, "typeCategory": "PACKAGE_CRATE"},
            {"packageQuantity": 3, "typeCategory": "PACKAGE_CASE"},
        ],
        [
            edit("name", 0, 4, "RETAINED", "documentPatch.parties.shipper.name"),
            edit(
                "compound",
                5,
                13,
                "2 CRATES + 3 CASES",
                root + "numberAndTypeOfPackages[0].typeCategory",
            ),
            edit("product", 14, 19, "BOLTS", root + "description"),
        ],
    )
    fresh = candidate(
        "fresh",
        "NOT COPIED",
        "NEW123",
        [{"packageQuantity": 4, "typeCategory": "PACKAGE_BOX"}],
        [
            edit("name", 0, 4, "NOT COPIED", "documentPatch.parties.shipper.name"),
            edit("count", 5, 6, "4", root + "numberAndTypeOfPackages[0].packageQuantity"),
            edit("noun", 7, 13, "BOXES", root + "numberAndTypeOfPackages[0].typeCategory"),
            edit("product", 14, 19, "BOLTS", root + "description"),
        ],
    )
    current = {
        "documentId": "old",
        "sourceDocumentId": "source",
        "joinedRawText": old["joinedRawText"],
        "target": old["target"]["documentPatch"],
    }
    return current, old, fresh


def test_split_cargo_owners_replace_compound_without_copying_other_fields():
    current, old, fresh = fixture()
    before = copy.deepcopy((current, old, fresh))
    result, after = module.compose_cargo(current, old, fresh, set())
    assert result["joinedRawText"] == "RETAINED|4 BOXES|BOLTS"
    assert after["target"]["parties"] == current["target"]["parties"]
    assert after["target"]["containerInformation"] == current["target"]["containerInformation"]
    assert after["target"]["goodsItemDetails"][0]["splitGoodsPlacement"] == [
        {"equipmentIdentifier": "OLD123", "packageQuantity": 4}
    ]
    assert (current, old, fresh) == before
    assert (
        result["scenario"]["cargo"]["containerInformation"][0]["equipmentIdentifier"] == "SOURCE123"
    )
    assert (
        result["scenario"]["cargo"]["goodsItemDetails"][0]["splitGoodsPlacement"][0][
            "equipmentIdentifier"
        ]
        == "SOURCE123"
    )


def test_source_equipment_inventory_cannot_change():
    current, old, fresh = fixture()
    fresh["scenario"]["cargo"]["containerInformation"][0]["equipmentIdentifier"] = "WRONG"
    with pytest.raises(ValueError, match="source equipment ownership"):
        module.compose_cargo(current, old, fresh, set())


def test_fresh_source_equal_value_replaces_retained_changed_value():
    current, old, fresh = fixture()
    fresh["proof"]["edits"] = [e for e in fresh["proof"]["edits"] if e["key"] != "product"]
    raw = apply_edits(b"NAME|1 PALLET|GOODS", fresh["proof"]["edits"]).decode()
    fresh["joinedRawText"] = raw
    fresh["proof"]["renderedSha256"] = digest(raw.encode())
    fresh["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "GOODS"
    fresh["surfaceValues"] = {"product": ["GOODS"]}
    result, _ = module.compose_cargo(current, old, fresh, set())
    assert result["joinedRawText"] == "RETAINED|4 BOXES|GOODS"
    assert result["surfaceValues"]["product"] == ["GOODS"]
    del fresh["surfaceValues"]
    result, _ = module.compose_cargo(current, old, fresh, set())
    assert result["joinedRawText"] == "RETAINED|4 BOXES|GOODS"
    fresh["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "DIFFERENT"
    with pytest.raises(ValueError, match="unchanged sampled authority"):
        module.compose_cargo(current, old, fresh, set())


def test_mixed_cargo_and_private_owner_cannot_be_replaced():
    current, old, fresh = fixture()
    old["proof"]["edits"][1]["targetPaths"].append("documentPatch.parties.shipper.name")
    with pytest.raises(ValueError, match="protected retained owner"):
        module.compose_cargo(current, old, fresh, set())


def test_explicit_mixed_owner_preserves_its_text_and_cargo_target():
    current, old, fresh = fixture()
    path = 'documentPatch.goodsItemDetails[0].handlingInstructions[0]'
    for candidate, value in [(old, 'RETAINED'), (fresh, 'NOT COPIED')]:
        candidate['proof']['edits'][0]['targetPaths'].append(path)
        candidate['target']['documentPatch']['goodsItemDetails'][0]['handlingInstructions'] = [value]
        candidate['scenario']['cargo']['goodsItemDetails'][0]['handlingInstructions'] = [value]
    result, after = module.compose_cargo(current, old, fresh, set(), preserved_mixed_owners=('name',))
    assert result['joinedRawText'] == 'RETAINED|4 BOXES|BOLTS'
    assert after['target']['goodsItemDetails'][0]['handlingInstructions'] == ['RETAINED']
    assert result['scenario']['cargo']['goodsItemDetails'][0]['handlingInstructions'] == ['RETAINED']
    assert result['migrationCargoRefresh']['preservedMixedOwners'] == ['name']
    for keys in [('missing',), ('name', 'name'), ('product',)]:
        with pytest.raises(ValueError):
            module.compose_cargo(current, old, fresh, set(), preserved_mixed_owners=keys)
    fresh['proof']['edits'][0]['targetPaths'].remove(path)
    with pytest.raises(ValueError, match='identical source and target ownership'):
        module.compose_cargo(current, old, fresh, set(), preserved_mixed_owners=('name',))


def test_unchanged_alias_is_verified_at_its_own_occurrence():
    edit = {"key": "type", "occurrence": 1, "before": "PALLET", "targetPaths": []}
    fresh = {"surfaceValues": {"type": ["WOODEN PALLETS", "PALLET"]}}
    assert module.unchanged_cargo_authority(edit, fresh, {})
    edit["occurrence"] = 0
    assert not module.unchanged_cargo_authority(edit, fresh, {})
    edit["occurrence"] = 2
    with pytest.raises(ValueError, match="occurrence"):
        module.unchanged_cargo_authority(edit, fresh, {})


def test_source_equal_number_words_require_quantity_authority():
    path = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"
    edit = {"key": "count", "occurrence": 1, "before": "ONE", "targetPaths": [path]}
    fresh = {"renderValues": {"count": "1"}}
    assert module.unchanged_cargo_authority(edit, fresh, {path: 1})
    fresh["renderValues"]["count"] = "2"
    assert not module.unchanged_cargo_authority(edit, fresh, {path: 2})
    edit["targetPaths"] = ["documentPatch.goodsItemDetails[0].description"]
    fresh["renderValues"]["count"] = "1"
    assert not module.unchanged_cargo_authority(edit, fresh, {})


def test_later_unreconciled_repairs_cannot_be_reverted():
    current, old, fresh = fixture()
    current["joinedRawText"] += " additional repaired text"
    with pytest.raises(ValueError, match="exact retained text"):
        module.compose_cargo(current, old, fresh, set())


def origin_description_fixture():
    from document_ocr.synthesis.curated_descriptions import (
        CompiledDescriptionBlocks, CompiledDescriptionField,
        CompiledDescriptionSpan, project_rendered_descriptions,
    )

    current, old, fresh = fixture()
    source = b"NAME|1 PALLET|GOODS ORIGIN: CANADA"
    path = "documentPatch.goodsItemDetails[0].description"
    blocks = CompiledDescriptionBlocks(digest(source), "a" * 64, (
        CompiledDescriptionField(path, (
            CompiledDescriptionSpan(14, len(source), source[14:].decode()),
        )),
    ))
    for candidate, country in ((old, "FRANCE"), (fresh, "SPAIN")):
        candidate["proof"]["edits"][-1]["after"] = "SPAIN VARIETY BOLTS"
        candidate["proof"]["edits"].append({
            "key": "origin", "occurrence": 0,
            "byteStart": source.index(b"CANADA"), "byteEnd": len(source),
            "before": "CANADA", "after": country,
            "targetPaths": ["documentPatch.goodsItemDetails[0].origin.name"],
        })
        raw = apply_edits(source, candidate["proof"]["edits"])
        values, projection = project_rendered_descriptions(
            blocks, source, raw, candidate["proof"]["edits"]
        )
        candidate["joinedRawText"] = raw.decode()
        candidate["proof"].update(sourceSha256=digest(source),
                                  renderedSha256=digest(raw),
                                  descriptionProjection=projection)
        goods = candidate["target"]["documentPatch"]["goodsItemDetails"][0]
        goods.update(description=values[path], origin={"name": country})
    current.update(joinedRawText=old["joinedRawText"],
                   target=copy.deepcopy(old["target"]["documentPatch"]))
    return current, old, fresh


def test_scoped_origin_in_continuous_description_follows_retained_text():
    current, old, fresh = origin_description_fixture()
    before = copy.deepcopy((current, old, fresh))
    result, after = module.compose_cargo(current, old, fresh, set())
    goods = after["target"]["goodsItemDetails"][0]
    assert goods["description"] == "SPAIN VARIETY BOLTS ORIGIN: FRANCE"
    assert goods["origin"] == {"name": "FRANCE"}
    proof = result["proof"]["descriptionProjection"]
    assert proof["fields"][0]["value"] == goods["description"]
    assert proof["fields"][0]["spans"][0]["text"] in result["joinedRawText"]
    assert (current, old, fresh) == before


@pytest.mark.parametrize("corruption", ["target", "proof", "bounds", "source", "duplicate"])
def test_scoped_description_rejects_stale_or_ambiguous_authority(corruption):
    current, old, fresh = origin_description_fixture()
    p = fresh["proof"]["descriptionProjection"]
    if corruption == "target":
        fresh["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "WRONG"
    elif corruption == "proof":
        p["fields"][0]["value"] = "WRONG"
    elif corruption == "bounds":
        p["fields"][0]["spans"][0]["sourceByteEnd"] = 10000
    elif corruption == "source":
        p["sourceSha256"] = "0" * 64
    else:
        p["fields"].append(copy.deepcopy(p["fields"][0]))
    with pytest.raises(ValueError, match="description"):
        module.compose_cargo(current, old, fresh, set())


def saved_stage():
    current, old, fresh = fixture()
    composed, after = module.compose_cargo(current, old, fresh, set())
    raw = current["joinedRawText"].encode()
    stage = {"before": current, "after": after, "evidence": {
        "beforeRecordSha256": digest(current), "nativeCandidateId": fresh["documentId"],
        "nativeSha256": digest(fresh), "privateOwnerKeys": [],
        "geometryProof": copy.deepcopy(composed["proof"]),
        "currentProof": {"sourceSha256": digest(raw),
                         "renderedSha256": digest(after["joinedRawText"].encode()),
                         "edits": [{"key": "cargo-refresh", "byteStart": 0, "byteEnd": len(raw),
                                    "before": raw.decode(), "after": after["joinedRawText"]}]},
    }}
    return stage, old, fresh


def test_saved_stage_preserves_complete_proof_and_cargo_authority():
    stage, old, fresh = saved_stage()
    before = copy.deepcopy((stage, old, fresh))
    result = module.compose_saved_scoped_stage(stage, old, fresh)
    assert result["joinedRawText"] == "RETAINED|4 BOXES|BOLTS"
    assert result["target"]["documentPatch"] == stage["after"]["target"]
    assert result["scenario"]["origin"] == old["scenario"]["origin"]
    assert result["proof"] == stage["evidence"]["geometryProof"]
    assert (stage, old, fresh) == before


@pytest.mark.parametrize("case", ["fresh", "before", "sourceProof", "currentProof", "protectedTarget", "goods"])
def test_saved_stage_rejects_changed_evidence(case):
    stage, old, fresh = saved_stage()
    if case == "fresh":
        fresh["physical"]["changed"] = True
    elif case == "before":
        stage["before"]["joinedRawText"] += "CHANGED"
    elif case == "sourceProof":
        stage["evidence"]["geometryProof"]["edits"][0]["after"] = "CHANGED"
    elif case == "currentProof":
        stage["evidence"]["currentProof"]["edits"][0]["after"] = "CHANGED"
    elif case == "protectedTarget":
        stage["after"]["target"]["parties"]["shipper"]["name"] = "CHANGED"
    else:
        stage["after"]["target"]["goodsItemDetails"][0]["description"] = "CHANGED"
    with pytest.raises(ValueError):
        module.compose_saved_scoped_stage(stage, old, fresh)
