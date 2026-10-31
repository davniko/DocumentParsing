from copy import deepcopy
from random import Random
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.package_presentations import PackagePresentations, PackageVocabulary


def vocabulary():
    return PackageVocabulary(
        {
            "version": 1,
            "excludedUnscopedTokens": ["PC"],
            "entries": {
                "PACKAGE_BOX": {
                    "full": ["BOX", "BOXES"],
                    "optional": ["BOX(ES)"],
                    "short": ["BX"],
                    "family": "PACKAGE_BOX",
                },
                "PACKAGE_BOX_FIBREBOARD": {
                    "full": ["FIBREBOARD BOX", "FIBREBOARD BOXES"],
                    "optional": ["FIBREBOARD BOX(ES)"],
                    "short": [],
                    "family": "PACKAGE_BOX",
                },
            },
        }
    )


def test_specificity_is_not_inferred_from_bare_family():
    v = vocabulary()
    assert v.supports("20 BX", "PACKAGE_BOX")
    assert not v.supports("20 BOXES", "PACKAGE_BOX_FIBREBOARD")
    assert v.supports("20 FIBREBOARD BOX(ES)", "PACKAGE_BOX_FIBREBOARD")


def test_inspection_and_incomplete_certificates_cannot_publish():
    presenter = PackagePresentations.__new__(PackagePresentations)
    presenter.certification_status = "compiled_pending_batch_review_and_synthesis"
    presenter.acceptance = {}
    with pytest.raises(ValueError, match="not production certified"):
        presenter.require_certified(["source"])
    presenter.certification_status = "certified_native_synthesis_and_geometry"
    with pytest.raises(ValueError, match="lacks package presentation acceptance"):
        presenter.require_certified(["source"])
    presenter.acceptance = {"source": {"status": "PASS"}}
    presenter.require_certified(["source"])


def test_registry_has_no_quantity_gate_and_seed_is_reproducible():
    v = vocabulary()
    weights = {"full": 1, "short": 1, "optional": 1}
    outputs = {v.choose("PACKAGE_BOX", tuple(weights), weights, str(n))[0] for n in range(60)}
    assert outputs == {"BOX", "BOXES", "BOX(ES)", "BX"}
    assert v.choose("PACKAGE_BOX", tuple(weights), weights, "same") == v.choose(
        "PACKAGE_BOX", tuple(weights), weights, "same"
    )


def test_empty_eligible_pool_is_error_not_baseline_fallback():
    with pytest.raises(ValueError, match="no eligible"):
        vocabulary().choose("PACKAGE_BOX_FIBREBOARD", ("short",), {"short": 1}, "seed")


def test_specific_category_without_short_forms_reports_conditional_weights():
    _, evidence = vocabulary().choose(
        "PACKAGE_BOX_FIBREBOARD",
        ("full", "short", "optional"),
        {"full": 2, "short": 9, "optional": 2},
        "seed",
    )
    assert evidence["eligibleProbabilities"] == {"full": 0.5, "optional": 0.5}
    assert evidence["excludedStyles"] == {"short": "no_reviewed_form"}


@pytest.mark.parametrize("value", ["PC", "BX"])
def test_quarantined_and_cross_category_collisions_fail(value):
    v = vocabulary()
    entries = deepcopy(v.entries)
    entries["PACKAGE_BOX_FIBREBOARD"]["short"] = [value]
    with pytest.raises(ValueError, match=r"quarantined|conflicting"):
        PackageVocabulary({"version": 1, "entries": entries, "excludedUnscopedTokens": ["PC"]})


def test_numeric_constructor_uses_explicit_alias_without_changing_arithmetic():
    from document_ocr.synthesis.independent_numeric import _noun
    from document_ocr.synthesis.package_aliases import select_package_alias

    assert _noun("PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY", 2) == "WOODEN BOXES"
    assert (
        select_package_alias(
            "PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY", 1, "BOX(ES)", mode="source", rng=Random(1)
        )
        == "WOODEN BOX(ES)"
    )


def scalar_presenter():
    from document_ocr.synthesis.curated import digest

    path = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].typeCategory"
    target = {
        "documentPatch": {
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 20, "typeCategory": "PACKAGE_BOX_FIBREBOARD"}
                    ]
                }
            ]
        }
    }
    source = "FIBREBOARD BOXES\nBOXES"
    slots = [
        {
            "key": "noun",
            "occurrence": 0,
            "start": 0,
            "end": 16,
            "source": "FIBREBOARD BOXES",
            "categoryPaths": [path],
            "mode": "scalar",
            "role": "declaration",
            "geometry": {},
        },
        {
            "key": "noun",
            "occurrence": 1,
            "start": 17,
            "end": 22,
            "source": "BOXES",
            "categoryPaths": [path],
            "mode": "scalar",
            "role": "allocation",
            "geometry": {},
        },
    ]
    bp = SimpleNamespace(
        document_id="source",
        source=source,
        target=target,
        regions=[
            SimpleNamespace(
                key=s["key"],
                occurrence=s["occurrence"],
                start=s["start"],
                end=s["end"],
                source=s["source"],
                target_paths=[path],
            )
            for s in slots
        ],
        historical_bindings={
            "noun": {"occurrences": [{"source_text": s["source"]} for s in slots]}
        },
    )
    p = PackagePresentations.__new__(PackagePresentations)
    p.vocabulary = vocabulary()
    p.certificates = {
        "source": {
            "sourceSha256": digest(source.encode()),
            "sourceTargetSha256": digest(target),
            "slots": slots,
        }
    }
    p.policy = SimpleNamespace(
        weights=lambda: {"full": 1, "short": 100, "optional": 1},
        vocabulary=SimpleNamespace(sha256="v"),
        certificates=SimpleNamespace(sha256="c"),
    )
    return p, bp


def test_scalar_projection_preserves_specific_witness_and_all_numeric_targets():
    p, bp = scalar_presenter()
    original = deepcopy(bp.target)
    surfaces, receipt = p.apply(bp, bp.target, {}, seed=7, variant=1)
    assert p.vocabulary.supports(surfaces["noun"][0], "PACKAGE_BOX_FIBREBOARD")
    assert receipt["slots"][1]["presentedCategory"] == "PACKAGE_BOX"
    assert bp.target == original
    assert p.apply(bp, bp.target, {}, seed=7, variant=1) == (surfaces, receipt)


def test_changed_source_target_invalidates_package_certificate():
    p, bp = scalar_presenter()
    bp.target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "packageQuantity"
    ] = 21
    with pytest.raises(ValueError, match="source target certificate"):
        p.certificate(bp)


def test_locality_projection_keeps_package_certificate_and_owners_pinned():
    from document_ocr.synthesis.curated import digest

    presenter, original = scalar_presenter()
    original.target["documentPatch"]["placeOfIssue"] = {
        "name": "Hong Kong", "country": "Hong Kong"
    }
    presenter.certificates["source"]["sourceTargetSha256"] = digest(original.target)
    projected = deepcopy(original)
    del projected.target["documentPatch"]["placeOfIssue"]["country"]
    view = presenter.locality_projection_view(original, projected)
    assert view.certificate(projected)["sourceTargetSha256"] == digest(projected.target)
    assert presenter.certificate(original)["sourceTargetSha256"] == digest(original.target)
    changed_cargo = deepcopy(projected)
    changed_cargo.target["documentPatch"]["goodsItemDetails"][0][
        "numberAndTypeOfPackages"
    ][0]["packageQuantity"] = 21
    with pytest.raises(ValueError):
        presenter.locality_projection_view(original, changed_cargo)
    changed_owner = deepcopy(projected)
    changed_owner.regions[0].end -= 1
    with pytest.raises(ValueError, match="span"):
        presenter.locality_projection_view(original, changed_owner)


def test_finer_source_detail_is_preserved_for_same_category():
    p, bp = scalar_presenter()
    p.certificates["source"]["slots"][0]["protectedSourceDetail"] = True
    surfaces, receipt = p.apply(bp, bp.target, {}, seed=7, variant=1)
    assert surfaces["noun"][0] == "FIBREBOARD BOXES"
    assert receipt["slots"][0]["style"] == "protected"
