"""Source-only reference privacy follows accepted roles, not value-kind guesses."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated_auxiliary import validate_identifier_privacy
from document_ocr.synthesis.curated_templates import RenderRegion

REFERENCE_ROLE = "documentPatch.forwardingAndExportReferences[0]"
OWNER = "reviewed-commercial-reference"


@pytest.fixture
def reference():
    return SimpleNamespace(
        ownership_data={"identifier_privacy": {}},
        target={"documentPatch": {}},
        regions=[RenderRegion(OWNER, 0, 0, 9, "123456789", (), "other_text")],
        historical_bindings={
            OWNER: {
                "target_paths": [REFERENCE_ROLE],
                "realization": {"target_values": []},
                "auxiliary_recipe": {},
            }
        },
    )


@pytest.mark.parametrize("paths_type", [list, tuple], ids=["json-list", "typed-tuple"])
@pytest.mark.parametrize("provenance", ["binding", "accepted", "both"])
def test_mistyped_reference_requires_explicit_privacy(reference, paths_type, provenance):
    binding = reference.historical_bindings[OWNER]
    binding["target_paths"] = paths_type(
        [REFERENCE_ROLE] if provenance in {"binding", "both"} else []
    )
    binding["realization"]["target_values"] = paths_type(
        [{"target_path": REFERENCE_ROLE, "source_value": "123456789"}]
        if provenance in {"accepted", "both"}
        else []
    )
    with pytest.raises(ValueError, match="privacy is unresolved"):
        validate_identifier_privacy(reference)


@pytest.mark.parametrize(
    "disposition,recipe",
    [
        ("private", {"identifier": "commercial reference", "format": "source"}),
        ("public", {}),
        ("dependent", {"date": "2025-05-03", "format": "%d/%m/%Y"}),
        ("dependent", {"target": "documentPatch.billOfLadingNumber"}),
    ],
)
@pytest.mark.parametrize("paths_type", [list, tuple], ids=["json-list", "typed-tuple"])
def test_reviewed_reference_dispositions_need_no_retyping(
    reference, disposition, recipe, paths_type
):
    reference.ownership_data["identifier_privacy"][OWNER] = disposition
    binding = reference.historical_bindings[OWNER]
    binding["target_paths"] = paths_type(binding["target_paths"])
    binding["auxiliary_recipe"] = recipe
    validate_identifier_privacy(reference)
    assert reference.regions[0].kind == "other_text"


@pytest.mark.parametrize("recipe", [{}, {"identifier": "invoice"}, {"text": "123456789"}])
def test_private_reference_requires_source_format_generation(reference, recipe):
    reference.ownership_data["identifier_privacy"][OWNER] = "private"
    reference.historical_bindings[OWNER]["auxiliary_recipe"] = recipe
    with pytest.raises(ValueError, match="requires a source-format generator"):
        validate_identifier_privacy(reference)


def test_dependent_reference_requires_explicit_sampled_owner(reference):
    reference.ownership_data["identifier_privacy"][OWNER] = "dependent"
    with pytest.raises(ValueError, match="lacks its explicit sampled owner"):
        validate_identifier_privacy(reference)


@pytest.mark.parametrize(
    "owner_state", ["curated_forwarder", "curated_via_company", "active_bl", "active_forwarder"]
)
def test_current_target_and_generated_company_owners_are_excluded(reference, owner_state):
    binding = reference.historical_bindings[OWNER]
    binding["target_paths"] = []
    binding["realization"]["target_values"] = [
        {"target_path": REFERENCE_ROLE, "source_value": "123456789"}
    ]
    region = reference.regions[0]
    if owner_state.startswith("curated_"):
        reference.regions = [replace(region, curated_key=owner_state)]
    elif owner_state == "active_bl":
        reference.target["documentPatch"]["billOfLadingNumber"] = "SYN012345"
        reference.regions = [replace(region, target_paths=("documentPatch.billOfLadingNumber",))]
    else:
        reference.target["documentPatch"]["parties"] = {
            "forwardingAgent": {"name": "SYNTHETIC FORWARDING LLC"}
        }
        reference.regions = [
            replace(region, target_paths=("documentPatch.parties.forwardingAgent.name",))
        ]
    validate_identifier_privacy(reference)


def test_retired_reference_without_a_live_region_is_excluded(reference):
    reference.regions = []
    validate_identifier_privacy(reference)


@pytest.mark.parametrize(
    "path",
    ["documentPatch.forwardingAndExportReferencesOther[0]", "documentPatch.otherField"],
)
def test_reference_like_names_do_not_replace_semantic_role_provenance(reference, path):
    # Even an anchor-looking logical key cannot classify otherwise unrelated text.
    key = "anchor:" + REFERENCE_ROLE
    binding = reference.historical_bindings.pop(OWNER)
    binding["target_paths"] = [path]
    binding["realization"]["target_values"] = [{"target_path": path, "source_value": "123456789"}]
    reference.historical_bindings[key] = binding
    reference.regions = [replace(reference.regions[0], key=key)]
    validate_identifier_privacy(reference)


def test_whole_reference_field_provenance_is_checked(reference):
    reference.historical_bindings[OWNER]["target_paths"] = [
        "documentPatch.forwardingAndExportReferences"
    ]
    with pytest.raises(ValueError, match="privacy is unresolved"):
        validate_identifier_privacy(reference)


def test_unrelated_typed_identifier_still_requires_privacy(reference):
    reference.historical_bindings[OWNER]["target_paths"] = []
    reference.regions = [replace(reference.regions[0], kind="identifier")]
    with pytest.raises(ValueError, match="privacy is unresolved"):
        validate_identifier_privacy(reference)
