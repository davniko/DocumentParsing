from dataclasses import replace

import pytest

from document_ocr.synthesis.template_compiler import contact_values, host


def fixture(*, split=False):
    raw = "Amr Ali Abbas Elsayed <contact@example.com> ATTN: Amr\nAli Abbas Elsayed TEL:123"
    path = "documentPatch.parties.consignee.contactDetails."
    name = "Amr Ali Abbas Elsayed"
    draft = host.SpanDraft(
        draft_id="name",
        logical_key="contact",
        render_mode="target_binding",
        value_kind="contact_name",
        group_kind="party",
        group_key="party:consignee:0",
        target_paths=(path + "contactName",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=0,
        char_end=len(name),
        source_text=name,
        evidence_origin="accepted_label_evidence",
        render_policy="natural_text",
        rationale="test",
    )
    email = "contact@example.com"
    start = raw.index(email)
    drafts = [
        draft,
        replace(
            draft,
            draft_id="email",
            logical_key="email",
            value_kind="email",
            target_paths=(path + "emailAddresses[0]",),
            char_start=start,
            char_end=start + len(email),
            source_text=email,
        ),
    ]
    if split:
        start = raw.index("Amr", len(name))
        drafts.extend(
            [
                replace(
                    draft,
                    draft_id="first",
                    logical_key="attention",
                    target_paths=(),
                    render_mode="deterministic_auxiliary",
                    char_start=start,
                    char_end=start + 3,
                    source_text="Amr",
                ),
                replace(
                    draft,
                    draft_id="rest",
                    char_start=start + 4,
                    char_end=start + len(name),
                    source_text="Ali Abbas Elsayed",
                ),
            ]
        )
    return dict(
        raw=raw,
        drafts=tuple(drafts),
        source_target={
            "documentPatch": {
                "parties": {
                    "consignee": {
                        "contactDetails": {"contactName": name, "emailAddresses": [email]}
                    }
                }
            }
        },
    )


@pytest.mark.parametrize("split", [False, True])
def test_complete_attention_repeat_is_one_name_value_and_idempotent(split):
    args = fixture(split=split)
    result = contact_values.normalize_attention_repeats(**args)
    names = [d for d in result if d.logical_key == "contact"]
    assert len(result) == 3 and len(names) == 2
    assert names[1].source_text == "Amr\nAli Abbas Elsayed"
    assert names[1].target_paths == names[0].target_paths
    host.validate_draft_source_alignment(raw=args["raw"], drafts=result)
    host.validate_binding_realizations(**{**args, "drafts": result})
    assert contact_values.normalize_attention_repeats(**{**args, "drafts": result}) == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("group_key", "party:shipper:0"),
        ("target_paths", ("documentPatch.parties.shipper.contactDetails.emailAddresses[0]",)),
    ],
)
def test_mailbox_from_another_party_cannot_supply_name_ownership(field, value):
    args = fixture()
    args["drafts"] = (args["drafts"][0], replace(args["drafts"][1], **{field: value}))
    assert contact_values.normalize_attention_repeats(**args) == args["drafts"]


@pytest.mark.parametrize(
    "change",
    [
        {"group_key": "party:notify:0"},
        {"target_paths": ("documentPatch.parties.shipper.contactDetails.contactName",)},
        {"value_kind": "organization"},
        {"dependency_paths": ("documentPatch.parties.consignee.name",)},
    ],
)
def test_other_owned_facts_cannot_be_absorbed(change):
    args = fixture(split=True)
    args["drafts"] = (*args["drafts"][:2], replace(args["drafts"][2], **change), args["drafts"][3])
    assert contact_values.normalize_attention_repeats(**args) == args["drafts"]


def test_bound_reference_to_partial_name_prevents_removal():
    args = fixture(split=True)
    args["drafts"] = (
        *args["drafts"],
        replace(
            args["drafts"][0],
            draft_id="reference",
            logical_key="ref",
            char_start=len(args["raw"]) - 3,
            char_end=len(args["raw"]),
            source_text="123",
            target_paths=(),
            dependency_bindings=("attention",),
        ),
    )
    assert contact_values.normalize_attention_repeats(**args) == args["drafts"]


def test_invalid_source_alignment_fails_instead_of_normalizing():
    args = fixture()
    args["drafts"] = (replace(args["drafts"][0], source_text="Other Name"), args["drafts"][1])
    with pytest.raises(host.DraftSourceAlignmentError):
        contact_values.normalize_attention_repeats(**args)
