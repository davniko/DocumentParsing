from copy import deepcopy
from dataclasses import replace
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import (
    Occurrence,
    SourceContract,
    TargetBinding,
    Variable,
    digest,
)
from document_ocr.synthesis.curated_templates import (
    RenderRegion,
    SamplingBlueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.declared_package_rendering import (
    PREFIX,
    prepare_lists,
    project_geography_contract,
    public_rows,
)
from document_ocr.synthesis.declared_packages import CountedPackage, DeclaredPackages
from document_ocr.synthesis.package_presentations import PackageVocabulary


def fixture():
    source = "BL123\nTOTAL 10 BOXES\nEND"
    target = {
        "documentPatch": {
            "billOfLadingNumber": "BL123",
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 10, "typeCategory": "PACKAGE_BOX"}
                    ]
                }
            ],
        }
    }
    regions = (
        RenderRegion(
            "bill",
            0,
            0,
            5,
            "BL123",
            ("documentPatch.billOfLadingNumber",),
            "identifier",
            "bill",
            baseline="BL123",
        ),
        RenderRegion(
            "quantity",
            0,
            12,
            14,
            "10",
            (PREFIX + "[0].packageQuantity",),
            "count",
            "quantity",
            "number",
            "10",
        ),
        RenderRegion("noun", 0, 15, 20, "BOXES", (PREFIX + "[0].typeCategory",), "owned_text"),
    )
    contract = SourceContract(
        variables=[
            Variable(
                key="bill",
                kind="identifier",
                value="BL123",
                meaning="bill",
                required_literals=[],
                occurrences=[Occurrence(text="BL123", occurrence=1, presentation="text")],
            ),
            Variable(
                key="quantity",
                kind="count",
                value="10",
                meaning="declared package total",
                required_literals=[],
                occurrences=[Occurrence(text="10", occurrence=1, presentation="number")],
            ),
        ],
        targets=[
            TargetBinding(path="documentPatch.billOfLadingNumber", expression="{bill}"),
            TargetBinding(path=PREFIX + "[0].packageQuantity", expression="{quantity}"),
        ],
        fixed_context="Goods",
    )
    bp = SamplingBlueprint(
        "test",
        source,
        target,
        contract,
        regions,
        {
            "noun": {
                "occurrences": [{"source_text": "BOXES"}],
                "target_paths": [PREFIX + "[0].typeCategory"],
            }
        },
        {},
        (),
    )
    vocabulary = PackageVocabulary(
        {
            "version": 1,
            "excludedUnscopedTokens": [],
            "entries": {
                "PACKAGE_BOX": {
                    "family": "PACKAGE_BOX",
                    "full": ["BOXES"],
                    "short": ["BX"],
                    "optional": ["BOX(ES)"],
                },
                "PACKAGE_BOX_FIBREBOARD": {
                    "family": "PACKAGE_BOX",
                    "full": ["FIBREBOARD BOXES"],
                    "short": [],
                    "optional": ["FIBREBOARD BOX(ES)"],
                },
            },
        }
    )
    presenter = SimpleNamespace(
        vocabulary=vocabulary,
        policy=SimpleNamespace(weights=lambda: {"full": 1, "short": 1, "optional": 1}),
    )
    declaration = {
        "sourceSha256": digest(source.encode()),
        "sourceTargetSha256": digest(target),
        "projection": "shipment",
        "declarations": [
            {
                "owners": [["quantity", 0], ["noun", 0]],
                "start": 12,
                "end": 20,
                "source": "10 BOXES",
                "scope": "shipment",
            }
        ],
    }
    ledger = DeclaredPackages(
        (
            CountedPackage("PACKAGE_BOX", D(2), D(3), D(".1")),
            CountedPackage("PACKAGE_BOX_FIBREBOARD", D(4), D(5), D(".2")),
        ),
        ((7, 8),),
        ((D(1000), D(10)),),
    )
    new = deepcopy(target)
    new["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"] = ledger.public_rows()
    return bp, new, ledger, declaration, presenter


@pytest.mark.parametrize("seed", range(30))
def test_full_renderer_declares_groups_before_byte_projection(seed):
    bp, target, ledger, certificate, presenter = fixture()
    plan = prepare_lists(bp, target, ledger, certificate, presenter, seed=seed, variant=1)
    text, after, proof = render_sampling_blueprint(
        bp, target, {}, package_lists=plan, package_vocabulary=presenter.vocabulary
    )
    assert after == target
    assert "7 " in text and "8 FIBREBOARD" in text
    assert "15 BOXES" not in text
    assert "BL123" in text and text.endswith("END")
    assert set(proof["changedTargetPaths"]) == set(proof["coveredTargetPaths"])
    assert len(proof["edits"]) == 1
    assert proof["edits"][0]["before"] == "10 BOXES"
    assert len(proof["edits"][0]["targetPaths"]) == 4
    assert proof["declaredPackageLists"] == plan.evidence


def test_topology_change_without_explicit_plan_is_rejected():
    bp, target, *_ = fixture()
    with pytest.raises(ValueError, match="topology"):
        render_sampling_blueprint(bp, target, {})


def test_different_public_families_need_no_fabricated_generic_package_noun():
    bp, target, old, certificate, presenter = fixture()
    presenter.vocabulary.entries["PACKAGE_BAG"] = {
        "family": "PACKAGE_BAG",
        "full": ["BAGS"],
        "short": [],
        "optional": [],
    }
    ledger = DeclaredPackages(
        (old.groups[0], CountedPackage("PACKAGE_BAG", D(4), D(5), D(".2"))),
        old.matrix,
        old.capacities,
    )
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"] = ledger.public_rows()
    plan = prepare_lists(bp, target, ledger, certificate, presenter, seed=1, variant=1)
    assert plan.aggregate_category == "PACKAGE_PACKAGE"
    text, after, _ = render_sampling_blueprint(
        bp, target, {}, package_lists=plan, package_vocabulary=presenter.vocabulary
    )
    assert "8 BAGS" in text and "PACKAGE" not in text
    assert after == target
    damaged = replace(plan, regions=tuple((r, "7 BOXES") for r, _ in plan.regions))
    with pytest.raises(ValueError, match="own printed type"):
        render_sampling_blueprint(
            bp, target, {}, package_lists=damaged, package_vocabulary=presenter.vocabulary
        )


def test_missing_owner_or_stale_span_is_rejected():
    bp, target, ledger, certificate, presenter = fixture()
    certificate["declarations"][0]["source"] = "11 BOXES"
    with pytest.raises(ValueError, match="span changed"):
        prepare_lists(bp, target, ledger, certificate, presenter, seed=1, variant=1)


@pytest.mark.parametrize(
    "style,quantity", [("words", "Seven"), ("words_digits", "Seven (7)"), ("decimal2", "7.00")]
)
def test_reviewed_quantity_style_is_preserved(style, quantity):
    bp, target, ledger, certificate, presenter = fixture()
    certificate["declarations"][0]["quantityFormat"] = style
    plan = prepare_lists(bp, target, ledger, certificate, presenter, seed=1, variant=1)
    assert quantity.casefold() in plan.regions[0][1].casefold()
    assert plan.target_rows == ledger.public_rows()


def test_word_count_source_does_not_silently_turn_into_digits():
    bp, target, ledger, certificate, presenter = fixture()
    bp = replace(
        bp,
        regions=tuple(
            replace(r, presentation="words") if r.kind == "count" else r for r in bp.regions
        ),
    )
    plan = prepare_lists(bp, target, ledger, certificate, presenter, seed=1, variant=1)
    assert plan.regions[0][1].casefold().startswith("seven ")


def test_complete_private_declaration_preserves_inner_and_support_counts():
    bp, target, ledger, certificate, presenter = fixture()
    whole = RenderRegion("held-numeric:0", 0, 12, 20, "10 BOXES", (), "auxiliary")
    bp = replace(bp, regions=(bp.regions[0], whole))
    owner = certificate["declarations"][0]
    owner.update(owners=[[whole.key, 0]], privateSurface=whole.key)
    text = "7 BOXES STC 70 BAGS\n8 FIBREBOARD BOXES STC 160 BAGS"
    packing = {"surfaces": {whole.key: [text]}}
    plan = prepare_lists(
        bp, target, ledger, certificate, presenter, seed=1, variant=1, private_packing=packing
    )
    assert plan.regions[0][1] == text
    assert plan.target_rows == ledger.public_rows()
    with pytest.raises(ValueError, match="complete numeric owner"):
        prepare_lists(bp, target, ledger, certificate, presenter, seed=1, variant=1)
    packing["surfaces"][whole.key] = ["15 BOXES STC 230 BAGS"]
    with pytest.raises(ValueError, match="category witness"):
        prepare_lists(
            bp, target, ledger, certificate, presenter, seed=1, variant=1, private_packing=packing
        )


def test_local_occurrences_do_not_publish_hidden_group_column_sums():
    _, _, ledger, _, _ = fixture()
    value = DeclaredPackages(ledger.groups, ((7, 8), (2, 3)), ledger.capacities * 2)
    assert [r["packageQuantity"] for r in public_rows(value, "container_occurrences")] == [
        7,
        8,
        2,
        3,
    ]
    assert [r["packageQuantity"] for r in public_rows(value, "shipment")] == [9, 11]


def test_extra_unprinted_row_rejected():
    bp, target, ledger, certificate, presenter = fixture()
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"].append(
        {"packageQuantity": 1, "typeCategory": "PACKAGE_BOX"}
    )
    with pytest.raises(ValueError, match="private ledger projection"):
        prepare_lists(bp, target, ledger, certificate, presenter, seed=1, variant=1)


def test_geography_collapse_preserves_strict_cargo_contract():
    bp, _, _, certificate, _ = fixture()
    original = deepcopy(bp.target)
    original["documentPatch"]["route"] = {
        "portOfLoading": {"name": "Singapore", "country": "Singapore"}
    }
    bp = replace(bp, target=original)
    certificate = {**certificate, "sourceTargetSha256": digest(original)}
    projected = deepcopy(original)
    del projected["documentPatch"]["route"]["portOfLoading"]["country"]
    new_bp = replace(bp, target=projected)
    current = project_geography_contract(bp, new_bp, certificate)
    assert current["sourceTargetSha256"] == digest(projected)
    assert current["declarations"] == certificate["declarations"]
    assert certificate["sourceTargetSha256"] == digest(original)
    projected["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "packageQuantity"
    ] = 99
    with pytest.raises(ValueError, match="only omit authorized locality"):
        project_geography_contract(bp, new_bp, certificate)


def test_geography_cannot_repin_changed_source_text():
    bp, _, _, certificate, _ = fixture()
    with pytest.raises(ValueError, match="original pinned"):
        project_geography_contract(bp, replace(bp, source=bp.source + "!"), certificate)
