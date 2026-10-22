from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_wording import (
    WordingField,
    WordingRequest,
    native_wording_prompt,
)
from document_ocr.synthesis.independent_presentation import (
    DistinctActor,
    DistinctActorReview,
    LocalityAssembly,
    SourceOccurrenceLocalityReview,
    apply_scoped_values,
    load_distinct_actor_reviews,
    load_locality_reviews,
    resolve_reviewed_distinct_actors,
    resolve_reviewed_locality_assemblies,
    semantic_private_packing,
    strengthen_presentation_request,
    validate_distinct_actors,
    validate_locality_owners,
    wording_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = (
    ROOT
    / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/independent-flat-v1"
)
LOCALITY_REVIEWS_SHA256 = (
    "0aea95e7aa790cc6a58cd4d035b721474f84c2e2151fc2de258399c7e39ad5b8"
)
DISTINCT_ACTORS_SHA256 = (
    "b5cb808ed5818114890d9cb2e057d9c5eb80204242b20635c569981526985c8f"
)


def _field(key="name", kind="individual"):
    return WordingField(
        key,
        "documentPatch.parties.shipper name",
        "EXAMPLE",
        "identity only",
        kind,
    )


def test_pinned_existing_review_artifacts_load_into_typed_indices():
    locality = load_locality_reviews(
        (AUTHORITY / "locality_owner_reviews.json").read_bytes(),
        expected_sha256=LOCALITY_REVIEWS_SHA256,
    )
    actors = load_distinct_actor_reviews(
        (AUTHORITY / "distinct_actor_reviews.json").read_bytes(),
        expected_sha256=DISTINCT_ACTORS_SHA256,
    )
    assert len(locality.reviews) == 3
    assert len(actors.reviews) == 1
    occurrence = next(
        review for review in locality.reviews.values() if review.kind == "source_occurrence"
    )
    assert isinstance(occurrence.keys, tuple)
    source_only = next(
        review
        for review in locality.reviews.values()
        if review.kind == "source_only_certificate"
    )
    assert source_only.source_document_id.startswith("doc_")
    assert actors.for_document(next(iter(actors.reviews))) is not None
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        load_locality_reviews(
            (AUTHORITY / "locality_owner_reviews.json").read_bytes(),
            expected_sha256="0" * 64,
        )


def test_review_loader_rejects_duplicate_keys_and_schema_drift():
    duplicate = b'{"doc_a":{},"doc_a":{}}'
    with pytest.raises(ValueError, match="duplicate key"):
        load_locality_reviews(duplicate, expected_sha256=sha256_bytes(duplicate))
    row = {
        "doc_a": {
            "kind": "source_occurrence",
            "source_document_id": "doc_a",
            "source_sha256": "0" * 64,
            "role": "shipper",
            "expression": "{postal}",
            "keys": ["postal"],
            "source_locality_text": "ROTTERDAM",
            "unreviewed_field": True,
        }
    }
    payload = json.dumps(row, sort_keys=True).encode()
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_locality_reviews(payload, expected_sha256=sha256_bytes(payload))


def test_composed_request_preserves_fields_and_adds_all_authorities():
    request = WordingRequest(
        "sample",
        "CURRENT GOODS FACTS",
        (
            _field("primary", "individual"),
            _field("related", "organization"),
            WordingField("postal", "postal", "OLD CITY", "postal only"),
            WordingField("goods", "goods description", "GOODS", "goods only"),
        ),
    )
    result = strengthen_presentation_request(
        request,
        locality_assemblies=(
            LocalityAssembly(
                key="shipper",
                component_keys=("postal",),
                owner_key="postal",
                locality="ROTTERDAM",
                source_evidence="Reviewed source locality",
            ),
        ),
        distinct_actors=(
            DistinctActor(
                related_key="related",
                primary_expression="{primary}",
                source_evidence="Reviewed separate identities",
            ),
        ),
        regime="frozen",
    )
    prompt = native_wording_prompt([result])
    assert "natural person's full personal name only" in prompt
    assert "REVIEWED LOCALITY OWNER shipper" in prompt
    assert "DISTINCT ACTOR {shipper_name_component_2}" in prompt
    assert "primary identity is {shipper_name_component_1}" in prompt
    assert "Selected operational product state: FROZEN" in prompt
    assert "PER THAT COUNTED UNIT" in prompt
    assert request.context == "CURRENT GOODS FACTS"
    assert result.fields[-1] is request.fields[-1]


def test_locality_owner_is_explicit_and_never_last_component_default():
    assembly = LocalityAssembly(
        key="party",
        component_keys=("top", "tail"),
        owner_key="top",
        locality="CITY",
        source_evidence="CITY occurs in source top",
    )
    assert validate_locality_owners(
        {"top": "CITY MAIN ROAD", "tail": "COUNTRY"}, (assembly,)
    ) == []
    finding = validate_locality_owners(
        {"top": "MAIN ROAD", "tail": "CITY"}, (assembly,)
    )[0]
    assert finding.owner == "top" and finding.component_counts == {"top": 0, "tail": 1}
    with pytest.raises(ValueError, match="missing value"):
        validate_locality_owners({"top": "CITY"}, (assembly,))


def test_scoped_values_cannot_change_unreviewed_fields():
    parent = {"postal": "OLD", "goods": "UNCHANGED", "name": "UNCHANGED"}
    assert apply_scoped_values(parent, {"postal": "NEW"}, {"postal"}) == {
        **parent,
        "postal": "NEW",
    }
    assert parent["postal"] == "OLD"
    with pytest.raises(ValueError, match="exact authorized"):
        apply_scoped_values(
            parent,
            {"postal": "NEW", "goods": "CHANGED"},
            {"postal"},
        )


def _review_blueprint():
    return SimpleNamespace(
        document_id="source",
        source="EXACT SOURCE",
        contract=SimpleNamespace(
            targets=[
                SimpleNamespace(path="shipper.addressLine", expression="{top} {tail}"),
                SimpleNamespace(path="shipper.name", expression="{primary}"),
            ],
            variables=[
                SimpleNamespace(
                    key="top",
                    kind="postal",
                    occurrences=[SimpleNamespace(text="CITY MAIN ROAD")],
                ),
                SimpleNamespace(
                    key="tail",
                    kind="postal",
                    occurrences=[SimpleNamespace(text="COUNTRY")],
                ),
                SimpleNamespace(key="primary", kind="name", occurrences=[]),
                SimpleNamespace(key="related", kind="name", occurrences=[]),
            ],
        ),
        ownership_data={
            "related_actors": {
                "related": {"context_party": "shipper", "relationship": "agent_for"}
            }
        },
    )


def test_typed_source_locality_review_binds_exact_source_and_kind():
    blueprint = _review_blueprint()
    review = SourceOccurrenceLocalityReview(
        kind="source_occurrence",
        source_document_id="source",
        source_sha256=digest(blueprint.source.encode()),
        role="shipper",
        expression="{top} {tail}",
        keys=("top", "tail"),
        source_locality_text="CITY",
    )
    scenario = SimpleNamespace(
        party_localities={"shipper": SimpleNamespace(name="NEW CITY")}
    )
    resolved = resolve_reviewed_locality_assemblies(blueprint, scenario, review)
    assert resolved[0].owner_key == "top" and resolved[0].locality == "NEW CITY"
    with pytest.raises(ValueError, match="another source"):
        resolve_reviewed_locality_assemblies(
            blueprint,
            scenario,
            review.model_copy(update={"source_document_id": "other"}),
        )
    blueprint.contract.variables[0].kind = "name"
    with pytest.raises(ValueError, match="not postal"):
        resolve_reviewed_locality_assemblies(blueprint, scenario, review)


def test_typed_distinct_actor_review_binds_exact_relation():
    blueprint = _review_blueprint()
    review = DistinctActorReview(
        source_document_id="source",
        source_sha256=digest(blueprint.source.encode()),
        primary_role="shipper",
        primary_expression="{primary}",
        related_key="related",
        source_evidence="Source shows two legal names",
    )
    (resolved,) = resolve_reviewed_distinct_actors(blueprint, review)
    assert resolved.related_key == "related"
    blueprint.ownership_data["related_actors"]["related"]["context_party"] = "consignee"
    with pytest.raises(ValueError, match="reviewed related identity"):
        resolve_reviewed_distinct_actors(blueprint, review)


def test_distinct_actor_validation_rejects_full_primary_copy_only():
    boundary = DistinctActor(
        related_key="related",
        primary_expression="{primary}",
        source_evidence="Reviewed separate identities",
    )
    assert validate_distinct_actors(
        {"primary": "ALPHA TRADING LTD", "related": "Alpha Trading Ltd Export Division"},
        (boundary,),
    )
    assert validate_distinct_actors(
        {"primary": "ALPHA TRADING LTD", "related": "ALPHA PRODUCTS SA"},
        (boundary,),
    ) == []
    assert validate_distinct_actors(
        {"primary": "AB", "related": "FABRICS LIMITED"}, (boundary,)
    ) == []


def _ledger():
    return {
        "countedCategory": "PACKAGE_PALLET",
        "publicQuantity": 3,
        "relation": "CONTAINS",
        "relatedCategory": "PACKAGE_CARTON",
        "relatedQuantity": 12,
        "unitNetKg": "100",
        "unitGrossKg": "125",
        "netKg": "300",
        "grossKg": "375",
        "facts": {"innerQuantity": 12},
        "surfaces": {"private": "12 CARTONS IN 3 PALLETS"},
        "wordingFacts": ["12 CARTONS IN 3 PALLETS"],
        "wordingAppends": {"product": "12 CARTONS IN 3 PALLETS"},
        "presentationDecisions": [],
        "version": 2,
        "sourceDocumentId": "source",
        "certificateSha256": "one-source-cert",
        "certificateFileSha256": "all-source-cert-file",
        "publicAllocationVisibilityUnchanged": True,
        "invariants": {"publicTargetBeforeSha256": "before"},
        "scopeLedgers": {},
    }


@pytest.mark.parametrize(
    "key",
    [
        "countedCategory",
        "publicQuantity",
        "relation",
        "relatedCategory",
        "relatedQuantity",
        "unitNetKg",
        "unitGrossKg",
        "netKg",
        "grossKg",
        "facts",
        "surfaces",
        "wordingFacts",
        "wordingAppends",
        "presentationDecisions",
    ],
)
def test_every_semantic_private_packing_fact_remains_visible(key):
    before = _ledger()
    after = copy.deepcopy(before)
    after[key] = "different"
    assert semantic_private_packing(before) != semantic_private_packing(after)


def test_private_packing_rejects_missing_or_unclassified_facts():
    missing = _ledger()
    del missing["relatedCategory"]
    with pytest.raises(ValueError, match="semantic fields missing"):
        semantic_private_packing(missing)
    unknown = _ledger()
    unknown["newPackingFact"] = 5
    with pytest.raises(ValueError, match="unclassified"):
        semantic_private_packing(unknown)


def test_wording_scenario_is_a_nonmutating_prompt_view():
    class Scenario(BaseModel):
        provenance: dict
        quantity: int

    original = Scenario(
        provenance={"freshPrivatePacking": _ledger(), "other": 3}, quantity=3
    )
    witness = original.model_dump()
    view = wording_scenario(original)
    assert original.model_dump() == witness
    assert view.quantity == 3 and view.provenance["other"] == 3
    assert "certificateFileSha256" not in view.provenance["freshPrivatePacking"]
    assert "certificateFileSha256" in original.provenance["freshPrivatePacking"]


@pytest.mark.parametrize("regime", ["ambient", "chilled", "frozen", "dg"])
def test_all_supported_regimes_preserve_request_structure(regime):
    request = WordingRequest(
        "sample",
        "context",
        (WordingField("goods", "goods description", "OLD", "new"),),
    )
    result = strengthen_presentation_request(
        request,
        locality_assemblies=(),
        distinct_actors=(),
        regime=regime,
    )
    assert result.sample_id == request.sample_id
    assert result.fields == request.fields
    assert result.description_groups == request.description_groups
    assert result.gross_weight_kg == request.gross_weight_kg
    assert "COUNTED UNIT" in result.context


def test_unknown_regime_has_no_ambient_fallback():
    with pytest.raises(ValueError, match="unknown sampled"):
        strengthen_presentation_request(
            WordingRequest("sample", "", ()),
            locality_assemblies=(),
            distinct_actors=(),
            regime="unknown",
        )
