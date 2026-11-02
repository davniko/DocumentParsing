from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.scoped_cargo_refresh import (
    apply_edits,
    cargo_auxiliary_keys,
    cargo_path,
    complete_cargo_edits,
    current_interval,
    frozen_geometry_edits,
    migrate,
    reconstruct_source,
    rendered_interval,
)


def candidate(source, edits):
    raw = apply_edits(source, edits)
    return {
        "joinedRawText": raw.decode(),
        "proof": {"sourceSha256": digest(source), "renderedSha256": digest(raw), "edits": edits},
    }


def test_current_hs_extension_is_not_left_as_stale_suffix():
    old = b"HS CODE 610910\n"
    now = b"HS CODE 61091042\n"
    a, b = current_interval(old, now, 8, 14)
    assert now[a:b] == b"61091042"


def test_current_internal_repair_preserves_surroundings():
    old = b"party OLD\nHS 610910\nPHONE 1234"
    now = b"party NEW LONG NAME\nHS 6109.10\nPHONE 5678"
    a, b = current_interval(old, now, old.index(b"610910"), old.index(b"610910") + 6)
    assert now[a:b] == b"6109.10"
    edits = [dict(byteStart=a, byteEnd=b, before=now[a:b].decode(), after="1511.10.22")]
    assert apply_edits(now, edits) == b"party NEW LONG NAME\nHS 1511.10.22\nPHONE 5678"


def test_unchanged_native_owner_still_migrates():
    native = candidate(b"HS 610910\n", [])
    region = SimpleNamespace(
        key="hs",
        occurrence=0,
        start=3,
        end=9,
        source="610910",
        target_paths=["documentPatch.goodsItemDetails[0].hsCodes[0]"],
    )
    assert complete_cargo_edits(native, [region], set())[0]["after"] == "610910"


def test_source_and_render_receipts_are_verified():
    native = candidate(b"123", [dict(byteStart=0, byteEnd=3, before="123", after="5678")])
    assert reconstruct_source(native) == b"123"
    native["proof"]["sourceSha256"] = "0" * 64
    with pytest.raises(ValueError, match="source reconstruction"):
        reconstruct_source(native)


def test_narrow_new_owner_cannot_split_old_compound_owner():
    with pytest.raises(ValueError, match="cuts through"):
        rendered_interval({"edits": [dict(byteStart=0, byteEnd=8, after="new")]}, 3, 8)


@pytest.mark.parametrize(
    ("source", "current"),
    [
        (b"a\nb\n", b"a\nnew\nb\n"),
        (b"a\nb\n", b"new\na\nb\n"),
        (b"a\nb\n", b"a\nb\nnew\n"),
        (b"a\nb\n", b"x\nb\n"),
    ],
)
def test_frozen_geometry_is_anchored_and_replays(source, current):
    edits = frozen_geometry_edits(source, current, 0, "test")
    assert all(e["byteStart"] < e["byteEnd"] for e in edits)
    assert apply_edits(source, edits) == current


def test_protected_paths_never_owned_by_cargo():
    for path in [
        "documentPatch.parties.shipper.name",
        "documentPatch.goodsItemDetails[0].origin.name",
        "documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].equipmentIdentifier",
        "documentPatch.containerInformation[0].equipmentIdentifier",
    ]:
        assert not cargo_path(path)


def test_auxiliary_dependency_owns_product_repeat_not_party_repeat():
    assert cargo_auxiliary_keys(
        {
            "attached": {"target": "documentPatch.goodsItemDetails[0].description"},
            "mark": {"target": "documentPatch.parties.shipper.name"},
            "country": {"target": "documentPatch.goodsItemDetails[0].origin.name"},
            "constant": {"text": "Owner: protected party"},
        }
    ) == {"attached"}


def test_private_tare_replays_even_when_native_equals_source_and_party_was_repaired():
    source = b"PARTY X\nTARE 2000\n"
    old = candidate(
        source,
        [
            dict(
                key="party",
                occurrence=0,
                byteStart=6,
                byteEnd=7,
                before="X",
                after="OLD",
                targetPaths=["documentPatch.parties.shipper.name"],
            ),
            dict(
                key="tare",
                occurrence=0,
                byteStart=13,
                byteEnd=17,
                before="2000",
                after="3000",
                targetPaths=[],
            ),
        ],
    )
    native = candidate(
        source,
        [
            dict(
                key="party",
                occurrence=0,
                byteStart=6,
                byteEnd=7,
                before="X",
                after="NEW",
                targetPaths=["documentPatch.parties.shipper.name"],
            )
        ],
    )
    target = {
        "parties": {"shipper": {"name": "NOW"}},
        "containerInformation": [
            {
                "equipmentIdentifier": "ABC",
                "sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT",
                "typeCategory": "GENERAL_PURPOSE",
            }
        ],
        "goodsItemDetails": [{"description": "RICE"}],
    }
    old.update(sourceDocumentId="s", documentId="old")
    native.update(sourceDocumentId="s", documentId="native", target={"documentPatch": target})
    current = {
        "documentId": "current",
        "sourceDocumentId": "s",
        "joinedRawText": "PARTY NOW\nTARE 3000\n",
        "target": target,
    }
    current["joinedRawTextSha256"] = digest(current["joinedRawText"].encode())
    region = SimpleNamespace(
        key="tare", occurrence=0, start=13, end=17, source="2000", target_paths=[]
    )
    after, evidence = migrate(current, old, native, private_keys={"tare"}, regions=[region])
    assert after["joinedRawText"] == "PARTY NOW\nTARE 2000\n"
    assert after["target"] == current["target"]
    assert (
        apply_edits(source, evidence["geometryProof"]["edits"]) == after["joinedRawText"].encode()
    )
