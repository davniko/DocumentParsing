"""Reviewed block membership, rather than lexical slots, owns description labels."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_curated_templates import source_fixture

from document_ocr.synthesis.curated import Occurrence, TargetBinding, Variable, digest
from document_ocr.synthesis.curated_campaign import Campaign, validate_sample
from document_ocr.synthesis.curated_descriptions import (
    compile_description_blocks,
    project_rendered_descriptions,
    validate_description_regions,
)
from document_ocr.synthesis.curated_templates import (
    LexicalOwnership,
    compile_sampling_blueprint,
    render_sampling_blueprint,
)

PATH = "documentPatch.goodsItemDetails[0].description"


def declaration(row, *quotes):
    return {
        "source_sha256": digest(row["joinedRawText"].encode()),
        "fields": [{"path": PATH, "spans": [{"text": q, "occurrence": 1} for q in quotes]}],
    }


def block_fixture():
    row, contract, historical = source_fixture()
    row["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "STEEL BOLTS 24 BOXES"
    blocks = declaration(row, "STEEL BOLTS\n24 BOXES")
    return row, contract, historical, blocks


def test_projection_retains_host_count_package_and_source_order_without_rewriting_ocr():
    row, contract, historical, blocks = block_fixture()
    # Deliberately wrong old assembly: the block projection must be authoritative.
    contract.targets[2] = TargetBinding(path=PATH, expression="{quantity} {product}")
    blueprint = compile_sampling_blueprint(row, historical, contract, description_blocks=blocks)
    sampled = deepcopy(row["target"])
    goods = sampled["documentPatch"]["goodsItemDetails"][0]
    goods.update(description="CERAMIC TILES", hsCodes=["690721"])
    goods["numberAndTypeOfPackages"][0].update(packageQuantity=36, typeCategory="PACKAGE_CARTON")
    text, target, proof = render_sampling_blueprint(
        blueprint, sampled, {"product": "Ceramic tiles\nGRADE A"}
    )
    assert "Ceramic tiles\nGRADE A\n36 CARTONS\nHS: 690721" in text
    assert target["documentPatch"]["goodsItemDetails"][0]["description"] == (
        "CERAMIC TILES GRADE A 36 CARTONS"
    )
    assert "HS:" not in proof["descriptionProjection"]["fields"][0]["value"]
    assert proof["changedTargetPaths"] == proof["coveredTargetPaths"]
    assert render_sampling_blueprint(blueprint, target, {"product": "Ceramic tiles\nGRADE A"}) == (
        text,
        target,
        proof,
    )
    assert sampled["documentPatch"]["goodsItemDetails"][0]["description"] == "CERAMIC TILES"


def test_whole_product_overlay_keeps_source_lexical_baseline_separate_from_full_target():
    row, contract, historical, blocks = block_fixture()
    override = LexicalOwnership(
        "whole_goods",
        (PATH,),
        (Occurrence(text="STEEL BOLTS", occurrence=1, presentation="text"),),
        "product",
    )
    blueprint = compile_sampling_blueprint(
        row, historical, contract, description_blocks=blocks, ownership_overrides=(override,)
    )
    assert next(v.value for v in blueprint.contract.variables if v.key == "whole_goods") == (
        "STEEL BOLTS"
    )
    assert render_sampling_blueprint(blueprint, row["target"], {})[0] == row["joinedRawText"]
    text, target, _ = render_sampling_blueprint(
        blueprint, row["target"], {"whole_goods": "COPPER NUTS"}
    )
    assert "COPPER NUTS\n24 BOXES" in text
    assert target["documentPatch"]["goodsItemDetails"][0]["description"] == "COPPER NUTS 24 BOXES"


def test_separate_marks_not_imported_by_old_expression_and_selected_repeat_is_not_duplicated():
    row, contract, historical, _ = block_fixture()
    row["joinedRawText"] += "MARKS\nVIN ABC123\nGOODS COPY\nSTEEL BOLTS\n"
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods["description"] = "STEEL BOLTS"
    product = next(v for v in contract.variables if v.key == "product")
    product.occurrences.append(Occurrence(text="STEEL BOLTS", occurrence=2, presentation="text"))
    contract.variables.append(
        Variable(
            key="vin",
            kind="identifier",
            value="ABC123",
            meaning="Marks VIN",
            required_literals=[],
            occurrences=[Occurrence(text="ABC123", occurrence=1, presentation="text")],
        )
    )
    contract.targets[2] = TargetBinding(path=PATH, expression="{product} {vin}")
    blocks = declaration(row, "STEEL BOLTS")
    blocks["fields"][0]["spans"][0]["occurrence"] = 2
    blueprint = compile_sampling_blueprint(row, historical, contract, description_blocks=blocks)
    text, target, proof = render_sampling_blueprint(
        blueprint, row["target"], {"product": "COPPER NUTS", "vin": "XYZ456"}
    )
    assert text.count("COPPER NUTS") == 2 and "VIN XYZ456" in text
    assert target["documentPatch"]["goodsItemDetails"][0]["description"] == "COPPER NUTS"
    assert len(proof["descriptionProjection"]["fields"][0]["spans"]) == 1


def test_split_continuation_preserves_punctuation_capacity_origin_and_configured_casing():
    source = "ÉTÉ\nGOODS\n10 BOXES OF PARTS 10KG/CARTON\nHS 123456\nGRADE B-EGYPT\n"
    description = "10 BOXES OF PARTS 10KG/CARTON GRADE B-EGYPT"
    row = {
        "joinedRawText": source,
        "target": {
            "documentPatch": {
                "goodsItemDetails": [
                    {"description": description},
                ]
            }
        },
    }
    blocks = compile_description_blocks(
        source, row["target"], declaration(row, "10 BOXES OF PARTS 10KG/CARTON", "GRADE B-EGYPT")
    )
    start = len(source[: source.index("PARTS")].encode())
    edits = [
        {"byteStart": start, "byteEnd": start + 5, "before": "PARTS", "after": "New parts\nXL"}
    ]
    rendered = source.replace("PARTS", "New parts\nXL").encode()
    values, proof = project_rendered_descriptions(blocks, source.encode(), rendered, edits)
    assert values[PATH] == "10 BOXES OF NEW PARTS XL 10KG/CARTON GRADE B-EGYPT"
    assert (
        project_rendered_descriptions(blocks, source.encode(), rendered, edits, casing="preserve")[
            0
        ][PATH]
        == "10 BOXES OF New parts XL 10KG/CARTON GRADE B-EGYPT"
    )
    for span in proof["fields"][0]["spans"]:
        assert (
            rendered[span["renderedByteStart"] : span["renderedByteEnd"]].decode() == span["text"]
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("missing", "require reviewed"),
        ("hash", "SHA-256"),
        ("path", "every description"),
        ("occurrence", "occurrence absent"),
        ("order", "source order"),
        ("duplicate", "source order"),
        ("source_label", "differs from approved block"),
    ],
)
def test_contract_fails_closed(mutation, error):
    row, _, _, blocks = block_fixture()
    if mutation == "missing":
        blocks = None
    elif mutation == "hash":
        blocks["source_sha256"] = "0" * 64
    elif mutation == "path":
        blocks["fields"][0]["path"] = PATH.replace("[0]", "[1]")
    elif mutation == "occurrence":
        blocks["fields"][0]["spans"][0]["occurrence"] = 2
    elif mutation == "order":
        blocks = declaration(row, "24 BOXES", "STEEL BOLTS")
    elif mutation == "duplicate":
        blocks["fields"][0]["spans"] *= 2
    else:
        row["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "STEEL BOLTS"
    with pytest.raises(ValueError, match=error):
        compile_description_blocks(row["joinedRawText"], row["target"], blocks)


def test_partial_product_region_and_unselected_product_fail_before_render():
    row, contract, history = source_fixture()
    row["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "BOLTS"
    with pytest.raises(ValueError, match="boundary crosses mutable"):
        compile_sampling_blueprint(
            row, history, contract, description_blocks=declaration(row, "BOLTS")
        )
    row["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "24 BOXES"
    with pytest.raises(ValueError, match="lack an approved description occurrence"):
        compile_sampling_blueprint(
            row, history, contract, description_blocks=declaration(row, "24 BOXES")
        )


def test_mutated_blueprint_and_tampered_candidate_do_not_silently_pass():
    row, contract, history = source_fixture()
    bp = compile_sampling_blueprint(
        row, history, contract, description_blocks=declaration(row, "STEEL BOLTS")
    )
    with pytest.raises(ValueError, match="require complete"):
        render_sampling_blueprint(replace(bp, description_blocks=None), row["target"], {})
    with pytest.raises(ValueError, match="explicit product wording"):
        modified = deepcopy(row["target"])
        modified["documentPatch"]["goodsItemDetails"][0]["description"] = "NEW PRODUCT"
        render_sampling_blueprint(bp, modified, {})
    text, target, proof = render_sampling_blueprint(bp, row["target"], {"product": "NEW PRODUCT"})
    target["documentPatch"]["goodsItemDetails"][0]["description"] = "WRONG LABEL"
    candidate = dict(
        documentId="sample",
        seed=1,
        joinedRawText=text,
        target=target,
        proof=proof,
        renderValues={"product": "NEW PRODUCT"},
        surfaceValues={},
        countryCodes={},
    )
    with pytest.raises(ValueError, match="independent source/edit replay"):
        validate_sample(bp, SimpleNamespace(party_localities={}), candidate, None)


def test_source_readiness_precedes_any_concurrent_paid_generation():
    campaign = Campaign.__new__(Campaign)
    campaign.source_ids = ["good", "missing"]
    checked, called = [], []

    def blueprint(sid):
        checked.append(sid)
        if sid == "missing":
            raise ValueError("missing reviewed description_blocks")

    async def generate(sid):
        called.append(sid)

    campaign.blueprint = blueprint
    campaign.generate = generate
    with pytest.raises(ValueError, match="missing reviewed"):
        asyncio.run(campaign.generate_all())
    assert checked == ["good", "missing"] and called == []


def test_projection_rejects_crossing_edits_invalid_receipts_and_missing_product_contract():
    row, contract, history = source_fixture()
    bp = compile_sampling_blueprint(
        row, history, contract, description_blocks=declaration(row, "STEEL BOLTS")
    )
    raw = bp.source.encode()
    span = bp.description_blocks.fields[0].spans[0]
    crossing = {
        "byteStart": span.start - 1,
        "byteEnd": span.end,
        "before": raw[span.start - 1 : span.end].decode(),
        "after": "NEW",
    }
    with pytest.raises(ValueError, match="crosses approved block"):
        project_rendered_descriptions(bp.description_blocks, raw, raw, [crossing])
    bad = {"byteStart": span.start, "byteEnd": span.end, "before": "WRONG", "after": "NEW"}
    with pytest.raises(ValueError, match="differs from source"):
        project_rendered_descriptions(bp.description_blocks, raw, raw, [bad])
    bad["before"] = "STEEL BOLTS"
    with pytest.raises(ValueError, match="differs from rendered"):
        project_rendered_descriptions(bp.description_blocks, raw, raw, [bad])
    with pytest.raises(ValueError, match="owners require reviewed"):
        validate_description_regions(None, bp.regions)


def test_goods_blocks_cannot_share_bytes_or_exchange_product_ownership():
    row, contract, history = source_fixture()
    blocks = declaration(row, "STEEL BOLTS")
    bp = compile_sampling_blueprint(row, history, contract, description_blocks=blocks)
    wrong_owner = tuple(
        replace(r, target_paths=(PATH.replace("[0]", "[1]"),)) if r.kind == "product" else r
        for r in bp.regions
    )
    with pytest.raises(ValueError, match="belongs to another description"):
        validate_description_regions(bp.description_blocks, wrong_owner)
    row["target"]["documentPatch"]["goodsItemDetails"].append({"description": "STEEL BOLTS"})
    blocks["fields"].append({**blocks["fields"][0], "path": PATH.replace("[0]", "[1]")})
    with pytest.raises(ValueError, match="different goods overlap"):
        compile_description_blocks(row["joinedRawText"], row["target"], blocks)


def test_expansion_before_block_and_all_text_deletion_are_checked():
    row, contract, history = source_fixture()
    bp = compile_sampling_blueprint(
        row, history, contract, description_blocks=declaration(row, "STEEL BOLTS")
    )
    target = deepcopy(row["target"])
    target["documentPatch"]["parties"]["shipper"]["name"] = "A MUCH LONGER COMPANY NAME"
    text, _, proof = render_sampling_blueprint(bp, target, {})
    span = proof["descriptionProjection"]["fields"][0]["spans"][0]
    assert span["renderedByteStart"] > span["sourceByteStart"]
    assert text.encode()[span["renderedByteStart"] : span["renderedByteEnd"]] == b"STEEL BOLTS"
    raw = bp.source.encode()
    start, end = span["sourceByteStart"], span["sourceByteEnd"]
    with pytest.raises(ValueError, match="block is empty"):
        project_rendered_descriptions(
            bp.description_blocks,
            raw,
            raw[:start] + raw[end:],
            [{"byteStart": start, "byteEnd": end, "before": "STEEL BOLTS", "after": ""}],
        )
