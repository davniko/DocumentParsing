"""Offline tests of the actual PydanticAI direct-labeling path; no paid requests."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from document_ocr.config import load_strict_yaml_mapping
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.labeling_agents.direct import (
    DirectLabelingFlow,
    _review_changes,
    connected_sections,
    correction_change_error,
    correction_model,
    correction_paths,
    encoded,
    merge_sections,
    normalize_optional_objects,
)
from document_ocr.labeling_agents.direct_cargo import (
    CargoFactFinding,
    CargoRelationFinding,
    CargoSourceMap,
    cargo_numeric_findings,
    derived_amounts,
    map_grounding_errors,
    measure_interpretations,
    numeric_values,
)
from document_ocr.labeling_agents.direct_cli import execute
from document_ocr.labeling_agents.direct_grounding import source_fidelity_findings
from document_ocr.labeling_agents.direct_models import (
    SECTION_FIELDS,
    SECTION_MODELS,
    DirectLabelingConfig,
    SectionReview,
)

ROOT = Path(__file__).resolve().parents[1]
OCR = (
    "SHIPPER\nEXPORTER LTD\nUNIT 4\nNEWBERG OR 97132\nUSA\nB/L NO: BL001\n"
    "TTNU8111930 24 BAGS YARN\n"
)


def label() -> dict:
    return {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "billOfLadingNumber": "BL001",
            "parties": {
                "shipper": {
                    "name": "EXPORTER LTD",
                    "addressLine": "UNIT 4 NEWBERG OR 97132 USA",
                    "country": "USA",
                }
            },
            "containerInformation": [{"equipmentIdentifier": "TTNU8111930"}],
            "goodsItemDetails": [
                {
                    "description": "YARN",
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 24, "typeCategory": "PACKAGE_BAG"}
                    ],
                    "splitGoodsPlacement": [
                        {"equipmentIdentifier": "TTNU8111930", "packageQuantity": 24}
                    ],
                }
            ],
        },
    }


def config() -> DirectLabelingConfig:
    return DirectLabelingConfig.model_validate_json(
        encoded(load_strict_yaml_mapping(ROOT / "configs/labeling_agents/mpci_bl_direct.yaml"))
    )


def flow(tmp_path, responder, *, relations_responder=None, source_map=None, ocr=OCR, **kwargs):
    from pypdf import PdfWriter

    if "pdf_path" not in kwargs:
        tmp_path.mkdir(parents=True, exist_ok=True)
        pdf = tmp_path / "source.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        with pdf.open("wb") as file:
            writer.write(file)
        kwargs["pdf_path"] = pdf

    async def dispatch(messages, info):
        name = info.model_request_parameters.output_object.json_schema["title"]
        if name == "CargoSourceMap":
            assert "Candidate" not in text(messages)
            return response(
                source_map
                or {
                    "layoutInterpretation": "One printed cargo row.",
                    "products": [{"key": "g1", "description": "YARN"}],
                    "statements": [
                        {
                            "products": ["g1"],
                            "containers": ["TTNU8111930"],
                            "scope": "portion",
                            "packages": [
                                {"quantity": 24, "packageType": "BAGS", "level": "target"}
                            ],
                            "explanation": "Single row links the bags and container.",
                        }
                    ],
                    "uncertainties": [],
                }
            )
        if name == "CargoRelationResponse":
            if relations_responder is None:
                return response({"response": {"status": "pass", "findings": []}})
            value = relations_responder(messages, info)
        else:
            value = responder(messages, info)
        return await value if inspect.isawaitable(value) else value

    return DirectLabelingFlow(
        model=FunctionModel(dispatch, profile=ModelProfile(supports_json_schema_output=True)),
        config=config(),
        project_root=ROOT,
        output_dir=tmp_path / "run",
        ocr=ocr,
        **kwargs,
    )


def parts(messages):
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def text(messages):
    return "\n".join(
        item
        for part in parts(messages)
        for item in ([part.content] if isinstance(part.content, str) else part.content)
        if isinstance(item, str)
    )


def response(value):
    if isinstance(value.get("response"), dict) and "status" in value["response"]:
        value["response"]["explanation"] = "Fixture source and field policy support this verdict."
    return ModelResponse(parts=[TextPart(encoded(value))])


def correction_fields(messages, values):
    candidates = text(messages).split("Candidate extraction (untrusted draft):\n")[1:]
    before = {}
    for candidate in candidates:
        before.update(json.JSONDecoder().raw_decode(candidate)[0])
    return [c["field"] for c in _review_changes(before, values)]


def correction(messages, values):
    findings = json.JSONDecoder().raw_decode(text(messages).split("Findings:\n", 1)[1])[0]
    return response(
        {
            "response": {
                "decisions": [
                    {
                        "findingId": f["id"],
                        "disposition": "accept",
                        "explanation": "The fixture OCR supports the proposed correction.",
                        "changedFields": correction_fields(messages, values) if i == 0 else [],
                    }
                    for i, f in enumerate(findings)
                ],
                "values": values,
            }
        }
    )


def schema_objects(value):
    if isinstance(value, dict):
        if "properties" in value:
            yield value
        for child in value.values():
            yield from schema_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from schema_objects(child)


@pytest.mark.asyncio
async def test_extraction_sends_literal_ocr_descriptions_and_returns_plain_draft(tmp_path):
    calls = []

    def responder(messages, info):
        calls.append(messages)
        assert any(item == OCR for p in parts(messages) for item in p.content)
        schema = info.model_request_parameters.output_object.json_schema
        assert info.model_request_parameters.output_object.description
        # Reusable models stay in $defs instead of duplicating their schemas per role.
        assert "$defs" in schema
        assert all(model.get("description") for model in schema["$defs"].values())
        definitions = list(schema_objects(schema))
        for definition in definitions:
            for field in definition.get("properties", {}).values():
                assert field.get("description")
        by_title = {definition.get("title"): definition for definition in definitions}
        package = by_title["PackagesV7"]["properties"]["typeCategory"]
        assert "PACKAGE_BAG" in encoded(package)
        assert "Vocabulary:" in package["description"]
        canonical, printed = schema["$defs"]["ContainerInformationV7"]["anyOf"]
        for name in ("sizeCategory", "typeCategory"):
            assert canonical["properties"][name]["type"] == "string"
            assert None not in canonical["properties"][name]["enum"]
            assert "anyOf" not in canonical["properties"][name]
            assert printed["properties"][name]["type"] == "null"
        assert canonical["properties"]["typeDescription"]["type"] == "null"
        assert {"type": "null"} in printed["properties"]["typeDescription"]["anyOf"]
        assert "additionalInformation" not in by_title["GoodsItemDetailsV7"]["properties"]
        assert "city" not in by_title["ExtractionPartyV7"]["properties"]
        return response(label())

    subject = flow(tmp_path, responder)
    result = await subject.extract()
    assert result.canonical_target() == label()
    assert len(calls) == 1
    assert json.loads((subject.output_dir / "status.json").read_text())["status"] == "draft"
    assert subject.receipts[0]["status"] == "completed"
    assert (subject.output_dir / "ocr.txt").read_text() == OCR


@pytest.mark.asyncio
async def test_real_sdk_request_has_plain_ocr_strict_described_schema_and_usage(tmp_path):
    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/responses"
        assert payload["reasoning"]["effort"] == config().reasoning_effort
        assert payload["max_output_tokens"] == config().max_output_tokens
        texts = [
            part["text"]
            for message in payload["input"]
            for part in message.get("content", [])
            if part.get("type") == "input_text"
        ]
        assert OCR in texts
        native = payload["text"]["format"]
        assert native["strict"] is True
        assert native["description"]
        for model in schema_objects(native["schema"]):
            assert set(model["required"]) == set(model["properties"])
            assert model["additionalProperties"] is False
            assert all(field.get("description") for field in model["properties"].values())
        return httpx.Response(
            200,
            json={
                "id": "resp_offline_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": config().model,
                "output": [
                    {
                        "id": "msg_offline_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": encoded(label()), "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 300, "output_tokens": 50, "total_tokens": 350},
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client,
        AsyncOpenAI(api_key="offline-test-key", http_client=http_client, max_retries=0) as sdk,
    ):
        subject = DirectLabelingFlow(
            model=OpenAIResponsesModel(config().model, provider=OpenAIProvider(openai_client=sdk)),
            config=config(),
            project_root=ROOT,
            output_dir=tmp_path / "sdk",
            ocr=OCR,
        )
        result = await subject.extract()
    assert result.canonical_target() == label()
    assert len(requests) == 1
    assert subject.receipts[0]["usage"]["input_tokens"] == 300
    assert subject.receipts[0]["responses"][0]["responseId"] == "resp_offline_test"


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["invalid_category", "orphan", "legacy_party", "empty"])
async def test_invalid_drafts_are_not_published_or_silently_retried(tmp_path, problem):
    value = label()
    if problem == "invalid_category":
        value["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
            "typeCategory"
        ] = "PACKAGE_MADE_UP"
    elif problem == "orphan":
        del value["documentPatch"]["containerInformation"]
    elif problem == "legacy_party":
        value["documentPatch"]["parties"]["shipper"]["city"] = "NEWBERG"
    else:
        value["documentPatch"] = {}
    subject = flow(tmp_path, lambda messages, info: response(value))
    with pytest.raises((ValidationError, UnexpectedModelBehavior, ValueError)):
        await subject.extract()
    assert not (subject.output_dir / "target.json").exists()
    assert len(subject.receipts) == 1
    assert subject.receipts[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_absent_optional_objects_are_lossless_audited_and_shared_with_correction(tmp_path):
    value = label()
    value["documentPatch"]["freight"] = {"paymentArrangement": None, "paymentPlace": None}
    value["documentPatch"]["parties"]["shipper"]["contactDetails"] = {
        "contactName": None,
        "phoneNumbers": None,
        "emailAddresses": None,
        "websiteUrls": None,
    }
    value["documentPatch"]["goodsItemDetails"][0]["origin"] = {"name": None, "identifier": None}
    value["documentPatch"]["containerInformation"][0]["temperatureSetpoint"] = {
        "value": 0,
        "unit": "celsius",
    }
    before = copy.deepcopy(value)
    subject = flow(tmp_path, lambda messages, info: response(value))
    result = await subject.extract()
    expected = label()
    expected["documentPatch"]["containerInformation"][0]["temperatureSetpoint"] = {
        "value": 0,
        "unit": "celsius",
    }
    assert result.canonical_target() == expected
    assert value == before
    receipt = subject.receipts[0]
    assert json.loads(receipt["responses"][0]["text"][0]) == before
    assert {c["path"] for c in receipt["nullObjectNormalizations"]} == {
        "documentPatch.freight",
        "documentPatch.parties.shipper.contactDetails",
        "documentPatch.goodsItemDetails[0].origin",
    }
    for section, names in SECTION_FIELDS.items():
        raw = {key: before["documentPatch"][key] for key in names if key in before["documentPatch"]}
        envelope = {"response": {"decisions": [], "values": raw}}
        model = correction_model((section,))
        fixed = model.model_validate_json(encoded(normalize_optional_objects(model, envelope)))
        assert fixed.response.values.model_dump(mode="json", exclude_none=True) == {
            key: expected["documentPatch"][key] for key in names if key in expected["documentPatch"]
        }


@pytest.mark.parametrize(
    "problem",
    [
        "unknown_null_key",
        "null_required_mass",
        "empty_list",
        "empty_string",
        "empty_list_member",
        "invalid_identifier",
        "no_facts",
    ],
)
def test_null_normalization_cannot_hide_invalid_facts(problem):
    value = label()
    patch = value["documentPatch"]
    if problem == "unknown_null_key":
        patch["freight"] = {"unknown": None}
    elif problem == "null_required_mass":
        patch["goodsItemDetails"][0]["grossWeight"] = {"value": None, "unit": None}
    elif problem == "empty_list":
        patch["parties"]["shipper"]["contactDetails"] = {"phoneNumbers": []}
    elif problem == "empty_string":
        patch["parties"]["shipper"]["contactDetails"] = {"contactName": ""}
    elif problem == "empty_list_member":
        patch["goodsItemDetails"].append({"description": None})
    elif problem == "invalid_identifier":
        patch["containerInformation"][0]["equipmentIdentifier"] = "INLLU4102226"
    else:
        value["documentPatch"] = {"freight": {"paymentArrangement": None}}
    original = copy.deepcopy(value)
    with pytest.raises(ValidationError):
        BillOfLadingExtractionV7Label.model_validate_json(
            encoded(normalize_optional_objects(BillOfLadingExtractionV7Label, value))
        )
    assert value == original


def test_section_models_cover_schema_once_and_merge_protects_other_sections():
    fields = [field for section in SECTION_FIELDS.values() for field in section]
    assert len(fields) == len(set(fields))
    assert set(fields) == set(
        BillOfLadingExtractionV7Label.model_fields["documentPatch"].annotation.model_fields
    )
    candidate = BillOfLadingExtractionV7Label.model_validate_json(encoded(label()))
    replacement = SECTION_MODELS["metadata_freight"].model_validate_json(
        '{"billOfLadingNumber":"BL002"}'
    )
    result = merge_sections(
        candidate, {"metadata_freight": replacement}, validation_scope=tuple(SECTION_FIELDS)
    )
    expected = label()
    expected["documentPatch"]["billOfLadingNumber"] = "BL002"
    assert result == expected
    with pytest.raises(ValueError):
        SECTION_MODELS["equipment"].model_validate_json('{"parties":null}')
    with pytest.raises(ValueError, match="absent container"):
        merge_sections(
            candidate,
            {"equipment": SECTION_MODELS["equipment"]()},
            validation_scope=("equipment", "cargo"),
        )
    with pytest.raises(ValueError, match="outside its validation scope"):
        merge_sections(candidate, {"metadata_freight": replacement}, validation_scope=("parties",))


@pytest.mark.asyncio
async def test_reviews_check_absent_sections_and_clean_result_never_becomes_gold(tmp_path):
    def responder(messages, info):
        assert OCR in text(messages)
        assert "Section field definitions:" in text(messages)
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    candidate = BillOfLadingExtractionV7Label.model_validate_json(
        '{"schemaVersion":"7.0.0","documentPatch":{"billOfLadingNumber":"BL001"}}'
    )
    result = await subject.refine(candidate)
    assert len(subject.receipts) == 7
    assert set(result["reviews"]) == set(SECTION_FIELDS)
    assert result["status"] == "reviewed_candidate"
    assert result["goldApproved"] is False


@pytest.mark.asyncio
async def test_review_scope_and_shared_request_limit_across_documents(tmp_path):
    active = peak = 0

    async def responder(messages, info):
        nonlocal active, peak
        source = text(messages)
        section = next(s for s in SECTION_FIELDS if f"Assigned section: {s}." in source)
        context = source.split("Candidate extraction (untrusted draft):\n", 1)[1]
        assigned, *related = context.split(
            "\nRead-only related extraction (for association checks):\n"
        )
        assert json.JSONDecoder().raw_decode(assigned)[0].keys() <= set(SECTION_FIELDS[section])
        assert bool(related) == (section in ("cargo", "equipment"))
        if related:
            companion = "equipment" if section == "cargo" else "cargo"
            assert json.JSONDecoder().raw_decode(related[0])[0].keys() <= set(
                SECTION_FIELDS[companion]
            )
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.002)
        active -= 1
        return response({"response": {"status": "pass", "findings": []}})

    semaphore = asyncio.Semaphore(2)
    subjects = [flow(tmp_path / str(i), responder, request_semaphore=semaphore) for i in range(3)]
    await asyncio.gather(*(subject.refine(label()) for subject in subjects))
    assert peak == 2
    assert sum(len(s.receipts) for s in subjects) == 21


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_identifier", [False, True])
async def test_invalid_draft_is_reviewable_but_never_exported_as_valid(tmp_path, repair_identifier):
    counts = {}

    def responder(messages, info):
        source = text(messages)
        section = next(s for s in SECTION_FIELDS if f"Assigned section: {s}." in source)
        counts[section] = counts.get(section, 0) + 1
        if counts[section] == 1:
            marker = "Candidate schema diagnostics (not correction evidence):\n"
            assert (marker in source) == (section in ("equipment", "cargo"))
            if marker in source:
                diagnostics = json.JSONDecoder().raw_decode(source.split(marker, 1)[1])[0]
                assert all("input" not in d and "ctx" not in d for d in diagnostics)
                assert any("equipmentIdentifier" in d["loc"] for d in diagnostics)
        if "Findings:" in source:
            if not repair_identifier and section in ("equipment", "cargo"):
                return response(
                    {"response": {"reason": "Identifier remains unresolved in this fixture."}}
                )
            return correction(
                messages,
                {k: v for k, v in label()["documentPatch"].items() if k in SECTION_FIELDS[section]},
            )
        actionable = ("parties", "equipment", "cargo") if repair_identifier else ("parties",)
        if section in actionable and counts[section] == 1:
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": "description" if section == "cargo" else section,
                                "issue": "wrong_value",
                                "explanation": "Known test defect",
                                "suggestedCorrection": "Restore the printed value",
                            }
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    candidate = label()
    candidate["documentPatch"]["containerInformation"][0]["equipmentIdentifier"] = "INVALID"
    candidate["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"][0][
        "equipmentIdentifier"
    ] = "INVALID"
    candidate["documentPatch"]["parties"]["shipper"]["addressLine"] = "UNIT 4 USA"
    before = copy.deepcopy(candidate)
    subject = flow(tmp_path, responder)
    result = await subject.refine(candidate)
    assert candidate == before
    if repair_identifier:
        assert result["target"] == label()
        assert result["validationError"] is None
        assert result["status"] == "reviewed_candidate"
    else:
        assert result["status"] == "needs_adjudication"
        assert result["target"] is None
        assert result["validationError"]
        assert not (subject.output_dir / "target.json").exists()
        draft = json.loads((subject.output_dir / "reviewed-draft.json").read_text())
        assert draft["documentPatch"]["parties"] == label()["documentPatch"]["parties"]
        assert (
            draft["documentPatch"]["containerInformation"]
            == before["documentPatch"]["containerInformation"]
        )


@pytest.mark.asyncio
async def test_section_constraint_cannot_be_overridden_by_semantic_pass(tmp_path):
    candidate = label()
    candidate["documentPatch"]["forwardingAndExportReferences"] = ["INV: 123", "INV: 123"]
    original = copy.deepcopy(candidate)
    calls = []

    def responder(messages, info):
        source = text(messages)
        if "Findings:" in source:
            calls.append(source)
            assert "references must be unique" in source
            return correction(
                messages,
                {"billOfLadingNumber": "BL001", "forwardingAndExportReferences": ["INV: 123"]},
            )
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder, ocr=OCR + "\nINV: 123\n")
    result = await subject.refine(candidate)
    assert candidate == original
    assert len(calls) == 1
    assert result["target"]["documentPatch"]["forwardingAndExportReferences"] == ["INV: 123"]
    assert result["status"] == "reviewed_candidate"


@pytest.mark.asyncio
async def test_correction_is_typed_bounded_and_rechecks_dependencies(tmp_path):
    counts = {}

    def responder(messages, info):
        source = text(messages)
        section = next(s for s in SECTION_FIELDS if f"Assigned section: {s}." in source)
        counts[section] = counts.get(section, 0) + 1
        marker = "Changes in the last correction wave (audit both directions):\n"
        if marker in source:
            changes = json.JSONDecoder().raw_decode(source.split(marker, 1)[1])[0]
            assert "Findings:" not in source  # Fresh review, not a vote on a prior reviewer.
            if section == "parties":
                assert changes == [
                    {
                        "field": "parties.shipper.addressLine",
                        "before": "UNIT 4 NEWBERG OR USA",
                        "after": "UNIT 4 NEWBERG OR 97132 USA",
                    }
                ]
            else:
                assert section == "metadata_freight" and changes == []
        elif counts[section] > 1:
            assert "Findings:" in source
        if "Findings:" in source:
            fixed = copy.deepcopy(label()["documentPatch"]["parties"])
            fixed["shipper"]["addressLine"] = "UNIT 4 NEWBERG OR 97132 USA"
            return correction(messages, {"parties": fixed})
        if section == "parties" and counts[section] == 1:
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": "shipper.addressLine",
                                "issue": "missing",
                                "explanation": "Postcode omitted",
                                "suggestedCorrection": "Restore 97132",
                            }
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    before = label()
    before["documentPatch"]["parties"]["shipper"]["addressLine"] = "UNIT 4 NEWBERG OR USA"
    subject = flow(tmp_path, responder)
    result = await subject.refine(
        BillOfLadingExtractionV7Label.model_validate_json(encoded(before))
    )
    assert result["target"] == label()
    assert counts == {
        "parties": 3,
        "route_transport": 1,
        "metadata_freight": 2,
        "equipment": 1,
        "cargo": 1,
    }


def test_review_changes_preserve_added_removed_and_regrouped_entities():
    before = {"parties": {"shipper": {"country": "USA"}}, "goods": [{"description": "A"}]}
    after = {"parties": {"shipper": {"name": "ACME"}}, "goods": [{"description": "A B"}]}
    assert _review_changes(before, after) == [
        {"field": "goods", "before": [{"description": "A"}], "after": [{"description": "A B"}]},
        {"field": "parties.shipper.country", "before": "USA", "after": None},
        {"field": "parties.shipper.name", "before": None, "after": "ACME"},
    ]
    assert _review_changes(before, copy.deepcopy(before)) == []


@pytest.mark.parametrize("mutation", ["declared", "undeclared", "extra", "duplicate", "rejected"])
def test_correction_change_contract_detects_side_effects(mutation):
    before = {"freight": {"paymentArrangement": "prepaid"}, "issueDate": "2024-03-04"}
    after = {"issueDate": "2024-03-04"}
    fields = ["freight.paymentArrangement"]
    if mutation == "undeclared":
        fields = []
    elif mutation == "extra":
        fields += ["issueDate"]
    elif mutation == "duplicate":
        fields *= 2
    decisions = [
        {
            "findingId": "metadata_freight:0",
            "changedFields": fields,
            "disposition": "reject" if mutation == "rejected" else "accept",
        }
    ]
    error = correction_change_error(before, after, decisions)
    assert (error is None) == (mutation in ("declared", "extra", "duplicate"))


def test_multiple_findings_can_change_one_list_without_authorizing_other_fields():
    before = {"goodsItemDetails": [{"description": "YARN"}], "issueDate": "2024-03-04"}
    after = {**before, "goodsItemDetails": [{"description": "COTTON YARN"}]}
    decisions = [
        {"findingId": f"cargo:{i}", "changedFields": ["goodsItemDetails"], "disposition": "accept"}
        for i in range(2)
    ]
    assert correction_change_error(before, after, decisions) is None
    assert "undeclaredChanges" in correction_change_error(
        before, {**after, "issueDate": "2024-03-05"}, decisions
    )


def test_container_and_placement_identifiers_share_wire_format():
    from document_ocr.label_schemas.bill_of_lading_v7 import ContainerInformationV7, PlacementV7

    for model in (ContainerInformationV7, PlacementV7):
        field = model.model_json_schema()["properties"]["equipmentIdentifier"]
        assert field["pattern"] == r"^[A-Z]{3}[UJZ][0-9]{7}$"
        assert model.model_validate_json('{"equipmentIdentifier":"MCLU5074498"}')
        with pytest.raises(ValidationError):
            model.model_validate_json('{"equipmentIdentifier":"MCLU 507449-8"}')


def test_correction_scope_choices_follow_typed_values():
    assert correction_paths(SECTION_MODELS["cargo"]) == ["goodsItemDetails"]
    paths = correction_paths(SECTION_MODELS["route_transport"])
    assert "route.placeOfDelivery.name" in paths
    assert "route.placeOfDelivery" not in paths
    schema = correction_model(("route_transport",)).model_json_schema()
    fields = schema["$defs"]["ScopedCorrectionDecision"]["properties"]["changedFields"]
    assert fields["items"]["enum"] == paths


@pytest.mark.asyncio
async def test_source_map_only_repair_cannot_modify_target_facts(tmp_path):
    def reviewer(messages, info):
        if "Findings:" in text(messages):
            goods = copy.deepcopy(label()["documentPatch"]["goodsItemDetails"])
            goods[0]["marksAndNumbers"] = ["YARN"]
            return correction(messages, {"goodsItemDetails": goods})
        return response({"response": {"status": "pass", "findings": []}})

    def accounting(messages, info):
        return response(
            {
                "response": {
                    "status": "corrections_needed",
                    "findings": [
                        {
                            "field": "cargoSourceMap",
                            "issue": "wrong_owner",
                            "explanation": "Fixture source-map ownership correction only.",
                            "suggestedCorrection": "Repair source map only.",
                        }
                    ],
                }
            }
        )

    subject = flow(tmp_path, reviewer, relations_responder=accounting)
    result = await subject.refine(label())
    assert result["status"] == "needs_adjudication"
    assert result["target"] == label()
    assert "map-only" in result["reviews"]["cargo"]["findings"][0]["explanation"]


@pytest.mark.asyncio
@pytest.mark.parametrize("declare_removal", [False, True])
async def test_correction_reversal_is_declared_and_reviewed_against_last_wave(
    tmp_path, declare_removal
):
    wave = 0
    checked = []

    def responder(messages, info):
        nonlocal wave
        source = text(messages)
        if "Findings:" in source:
            wave += 1
            values = {"billOfLadingNumber": "BL001"}
            if wave == 1:
                values["freight"] = {"paymentArrangement": "prepaid"}
            result = correction(messages, values)
            if wave == 2 and not declare_removal:
                raw = json.loads(result.parts[0].content)
                raw["response"]["decisions"][0]["changedFields"] = []
                result = response(raw)
            return result
        if "Assigned section: metadata_freight." in source:
            marker = "Changes in the last correction wave (audit both directions):\n"
            if marker in source:
                checked.append(json.JSONDecoder().raw_decode(source.split(marker)[1])[0])
            if wave < 2:
                return response(
                    {
                        "response": {
                            "status": "corrections_needed",
                            "findings": [
                                {
                                    "field": "freight.paymentArrangement",
                                    "issue": "normalization",
                                    "explanation": "Fixture tests a reversal, not freight policy.",
                                    "suggestedCorrection": "Add" if wave == 0 else "Remove",
                                }
                            ],
                        }
                    }
                )
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    result = await subject.refine(label())
    if declare_removal:
        assert result["status"] == "reviewed_candidate"
        assert checked[-1] == [
            {"field": "freight.paymentArrangement", "before": "prepaid", "after": None}
        ]
    else:
        assert result["status"] == "needs_adjudication"
        assert result["target"]["documentPatch"]["freight"]["paymentArrangement"] == "prepaid"
        assert (
            "changed-field mismatch"
            in result["reviews"]["metadata_freight"]["findings"][0]["explanation"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision_mode", ["accept", "reject", "revise", "missing", "duplicate", "foreign", "hold"]
)
async def test_cross_section_relocation_is_joint_audited_and_preserves_other_facts(
    tmp_path,
    decision_mode,
):
    candidate = label()
    candidate["documentPatch"]["forwardingAndExportReferences"] = ["ITEM CODE: ZX-42"]
    counts = {}

    def responder(messages, info):
        if "focused cargo-association reviewer" in (info.instructions or ""):
            return response({"response": {"status": "pass", "findings": []}})
        source = text(messages)
        section = next(s for s in SECTION_FIELDS if f"Assigned section: {s}." in source)
        if "Findings:" in source:
            assert "Joint correction scope: metadata_freight, cargo" in source
            assert "Assigned section: cargo." in source
            if decision_mode == "hold":
                return response({"response": {"reason": "Destination goods ownership ambiguous"}})
            values = {
                k: copy.deepcopy(v)
                for k, v in candidate["documentPatch"].items()
                if k in (*SECTION_FIELDS["metadata_freight"], *SECTION_FIELDS["cargo"])
            }
            if decision_mode != "reject":
                values["forwardingAndExportReferences"] = None
                values["goodsItemDetails"][0]["description"] += " ITEM CODE: ZX-42"
            decisions = [
                {
                    "findingId": "metadata_freight:0",
                    "disposition": decision_mode
                    if decision_mode in ("accept", "reject", "revise")
                    else "accept",
                    "explanation": "Fixture adjudication of source ownership.",
                    "changedFields": correction_fields(messages, values),
                }
            ]
            if decision_mode == "missing":
                decisions = []
            if decision_mode == "duplicate":
                decisions *= 2
            if decision_mode == "foreign":
                decisions[0]["findingId"] = "parties:99"
            return response({"response": {"decisions": decisions, "values": values}})
        counts[section] = counts.get(section, 0) + 1
        if section == "metadata_freight" and counts[section] == 1:
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": "forwardingAndExportReferences",
                                "issue": "wrong_owner",
                                "explanation": "Product code has a goods owner.",
                                "reassignTo": "cargo",
                                "suggestedCorrection": (
                                    "Move ITEM CODE: ZX-42 to the YARN description."
                                ),
                            }
                        ],
                    }
                }
            )
        if counts[section] > 1:
            assert "Changes in the last correction wave" in source
            if section in ("cargo", "metadata_freight"):
                assert "Audited correction record" not in source
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    result = await subject.refine(candidate)
    actual = result["target"]["documentPatch"]
    assert actual["parties"] == candidate["documentPatch"]["parties"]
    assert actual["containerInformation"] == candidate["documentPatch"]["containerInformation"]
    if decision_mode in ("accept", "revise"):
        assert "forwardingAndExportReferences" not in actual
        assert actual["goodsItemDetails"][0]["description"] == "YARN ITEM CODE: ZX-42"
    else:
        assert actual == candidate["documentPatch"]
    if decision_mode in ("missing", "duplicate", "foreign", "hold"):
        assert result["status"] == "needs_adjudication"
        assert result["reviews"]["cargo"]["status"] == "unresolved"
    else:
        assert result["correctionDecisions"][0]["decisions"][0]["disposition"] == decision_mode


def test_connected_corrections_cover_transitive_dependencies_without_unrelated_scope():
    assert connected_sections(
        [
            ("cargo", "equipment"),
            ("parties",),
            ("metadata_freight", "route_transport"),
            ("cargo", "metadata_freight"),
        ]
    ) == [("parties",), ("route_transport", "metadata_freight", "equipment", "cargo")]


def test_source_map_retains_packing_levels_unknowns_and_shared_ownership():
    source = CargoSourceMap.model_validate_json(
        encoded(
            {
                "layoutInterpretation": "Two portions, with outer pallets recorded separately.",
                "products": [
                    {"key": "a", "description": "YARN"},
                    {"key": "b", "description": "CLOTH"},
                ],
                "statements": [
                    {
                        "products": ["a"],
                        "containers": ["TTNU8111930"],
                        "scope": "portion",
                        "packages": [
                            {"quantity": q, "packageType": "BAGS", "level": "target"},
                            {"quantity": 1, "packageType": "PALLET", "level": "outer"},
                        ],
                        "explanation": "Separate printed portion.",
                    }
                    for q in (2, 3)
                ],
                "uncertainties": [],
            }
        )
    )
    assert derived_amounts(source, "YARN", "packageQuantity") == {5}
    assert not derived_amounts(source, "CLOTH", "packageQuantity")
    assert not map_grounding_errors(source, "TTNU 8111930 2 BAGS 3 BAGS 1 PALLET")
    candidate = label()
    goods = candidate["documentPatch"]["goodsItemDetails"][0]
    goods["numberAndTypeOfPackages"][0]["packageQuantity"] = 5
    goods["splitGoodsPlacement"] = [{"equipmentIdentifier": "TTNU8111930"}]
    assert not cargo_numeric_findings(candidate, "2 BAGS 3 BAGS", source)
    source.statements.append(
        source.statements[0].model_copy(update={"products": ["a", "b"], "scope": "shared_total"})
    )
    assert not derived_amounts(source, "YARN", "packageQuantity")
    assert cargo_numeric_findings(candidate, "2 BAGS 3 BAGS", source)
    source.statements.pop()
    source.statements[1].packages[0] = (
        source.statements[1].packages[0].model_copy(update={"quantity": None})
    )
    assert not derived_amounts(source, "YARN", "packageQuantity")
    broken = source.model_dump(mode="json")
    broken["statements"][0]["products"] = ["absent"]
    with pytest.raises(ValidationError, match="absent product"):
        CargoSourceMap.model_validate_json(encoded(broken))


def test_review_contracts_enforce_disjoint_cargo_responsibilities():
    common = {"issue": "wrong_value", "explanation": "Source-owned value differs."}
    CargoFactFinding(field="description", **common)
    CargoRelationFinding(field="goodsItemDetails.splitGoodsPlacement", **common)
    CargoRelationFinding(field="goodsItemDetails.grossWeight", **common)
    with pytest.raises(ValidationError):
        CargoRelationFinding(field="marksAndNumbers", **common)
    with pytest.raises(ValidationError):
        CargoFactFinding(field="goodsItemDetails.splitGoodsPlacement", **common)
    with pytest.raises(ValidationError):
        CargoFactFinding(field="grossWeight", **common)


def test_numeric_screen_recognizes_grouping_without_arbitrary_concatenation():
    from decimal import Decimal

    assert 227910 in numeric_values("GROSS WEIGHT: 227 910 KOS")
    assert Decimal("1234567.50") in numeric_values("1\u202f234\u202f567,50 KGS")
    assert 14625 not in numeric_values("14 6 25 PACKAGES")
    assert 227910 not in numeric_values("227\n910")
    assert measure_interpretations("7.740") == {Decimal("7.74"), Decimal(7740)}
    assert measure_interpretations("7.74") == {Decimal("7.74")}
    assert measure_interpretations("1\u202f234\u202f567,50") == {Decimal("1234567.50")}
    assert not measure_interpretations("14 6 25")


def test_derived_totals_handle_unicode_without_editing_labels_or_guessing_identity():
    source = CargoSourceMap.model_validate_json(
        encoded(
            {
                "layoutInterpretation": "One product in two complete portions.",
                "products": [{"key": "g", "description": "GRANITE COOKWARE"}],
                "statements": [
                    {
                        "products": ["g"],
                        "containers": [],
                        "scope": "portion",
                        "packages": [{"quantity": q, "packageType": "BOXES", "level": "target"}],
                        "explanation": "Explicit portion.",
                    }
                    for q in (408, 438)
                ],
                "uncertainties": [],
            }
        )
    )
    assert derived_amounts(source, "GRANİTE COOKWARE", "packageQuantity") == {846}
    assert not derived_amounts(source, "GRANITE 12PCS COOKWARE", "packageQuantity")
    source.products.append(
        source.products[0].model_copy(update={"key": "other", "description": "GRANİTE COOKWARE"})
    )
    assert not derived_amounts(source, "GRANITE COOKWARE", "packageQuantity")


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["container_id", "amount", "amount_relocated"])
async def test_correction_cannot_commit_new_pdf_only_cargo_facts(tmp_path, defect):
    candidate = label()
    if defect == "amount_relocated":
        candidate["documentPatch"]["goodsItemDetails"][0]["grossWeight"] = {
            "value": 21446.13,
            "unit": "kilogram",
        }

    def responder(messages, info):
        source = text(messages)
        if "Findings:" in source:
            patch = copy.deepcopy(candidate["documentPatch"])
            if defect == "container_id":
                patch["containerInformation"][0]["equipmentIdentifier"] = "TGHU8973609"
                patch["goodsItemDetails"][0]["splitGoodsPlacement"][0]["equipmentIdentifier"] = (
                    "TGHU8973609"
                )
            elif defect == "amount":
                patch["goodsItemDetails"][0]["grossWeight"] = {
                    "value": 21446.13,
                    "unit": "kilogram",
                }
            else:
                patch["goodsItemDetails"][0]["volume"] = {"value": 21446.13, "unit": "cubic_metre"}
            scopes = ("equipment", "cargo") if defect == "container_id" else ("cargo",)
            return correction(
                messages,
                {k: v for k, v in patch.items() if any(k in SECTION_FIELDS[s] for s in scopes)},
            )
        if "Assigned section: equipment." in source and defect == "container_id":
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": "equipmentIdentifier",
                                "issue": "wrong_owner",
                                "explanation": "Deliberately false PDF-only suggestion.",
                                "suggestedCorrection": (
                                    "Replace equipment and placements with TGHU8973609."
                                ),
                                "reassignTo": "cargo",
                            }
                        ],
                    }
                }
            )
        if "Assigned section: cargo." in source and defect != "container_id":
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": "grossWeight",
                                "issue": "missing",
                                "explanation": "Deliberately false PDF-only suggestion.",
                                "suggestedCorrection": "Add gross weight 21446.13 kg.",
                            }
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    result = await subject.refine(candidate)
    assert result["target"] == candidate
    assert result["status"] == "needs_adjudication"
    assert result["reviews"]["cargo"]["status"] == "unresolved"
    assert sum(r["stage"] == "cargo_mapper" for r in subject.receipts) == 1


@pytest.mark.asyncio
async def test_malformed_cargo_review_keeps_independent_sections_and_source_map(tmp_path):
    def responder(messages, info):
        if "Assigned section: cargo." in text(messages):
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": "not_a_cargo_fact",
                                "issue": "missing",
                                "explanation": "Schema violation",
                            }
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    reviews = await subject.review(label(), tuple(SECTION_FIELDS))
    assert reviews["cargo"].status == "unresolved"
    assert all(r.status == "pass" for s, r in reviews.items() if s != "cargo")
    assert (subject.output_dir / "cargo-source-map.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["broken_reference", "semantic_hold"])
async def test_unresolved_correction_does_not_discard_independent_valid_changes(tmp_path, failure):
    counts = {}

    def responder(messages, info):
        source = text(messages)
        section = next(s for s in SECTION_FIELDS if f"Assigned section: {s}." in source)
        counts[section] = counts.get(section, 0) + 1
        if "Findings:" in source:
            if section == "parties":
                return correction(messages, {"parties": label()["documentPatch"]["parties"]})
            if failure == "semantic_hold":
                return response({"response": {"reason": "Container identity remains ambiguous."}})
            return correction(messages, {"containerInformation": None})
        if section in ("parties", "equipment") and counts[section] == 1:
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": section,
                                "issue": "wrong_value",
                                "explanation": "Test finding",
                                "suggestedCorrection": "Restore source facts",
                            }
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    before = label()
    before["documentPatch"]["parties"]["shipper"]["addressLine"] = "UNIT 4 USA"
    subject = flow(tmp_path, responder)
    result = await subject.refine(
        BillOfLadingExtractionV7Label.model_validate_json(encoded(before))
    )
    assert result["target"] == label()
    assert result["status"] == "needs_adjudication"
    assert result["reviews"]["parties"]["status"] == "pass"
    assert result["reviews"]["equipment"]["status"] == "unresolved"


@pytest.mark.asyncio
async def test_layout_missing_is_explicit_and_does_not_hold_siblings(tmp_path):
    def responder(messages, info):
        if "Assigned section: parties." in text(messages):
            return response({"response": {"pages": [1], "question": "Who owns the continuation?"}})
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder, pdf_path=None)
    result = await subject.refine(
        BillOfLadingExtractionV7Label.model_validate_json(encoded(label()))
    )
    assert result["status"] == "needs_adjudication"
    assert result["reviews"]["parties"]["status"] == "unresolved"
    assert result["reviews"]["cargo"]["status"] == "unresolved"
    assert all(
        result["reviews"][s]["status"] == "pass"
        for s in SECTION_FIELDS
        if s not in ("parties", "cargo")
    )


@pytest.mark.asyncio
async def test_layout_request_returns_image_and_stays_bounded(tmp_path):
    from pydantic_ai import BinaryContent
    from pypdf import PdfWriter

    pdf = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with pdf.open("wb") as file:
        writer.write(file)
    seen = []

    def responder(messages, info):
        images = [
            item for p in parts(messages) for item in p.content if isinstance(item, BinaryContent)
        ]
        seen.append(len(images))
        return response({"response": {"pages": [1], "question": "Which block?"}})

    subject = flow(tmp_path, responder, pdf_path=pdf)
    candidate = BillOfLadingExtractionV7Label.model_validate_json(encoded(label()))
    result = await subject.review(candidate, ("parties",))
    assert seen == [0, 1]
    assert result["parties"].status == "unresolved"
    assert subject.layout_sections == {"parties"}
    assert len(subject.receipts) == 2


@pytest.mark.asyncio
async def test_topology_review_gets_complete_pdf_not_unrelated_party_layout(tmp_path):
    from pydantic_ai import BinaryContent
    from pypdf import PdfWriter

    pdf = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.add_blank_page(width=595, height=842)
    with pdf.open("wb") as file:
        writer.write(file)
    topology_images = []

    def responder(messages, info):
        images = [x for p in parts(messages) for x in p.content if isinstance(x, BinaryContent)]
        if "Assigned section: equipment." in text(messages) and not images:
            return response({"response": {"pages": [1], "question": "Container row boundary?"}})
        if "focused cargo-association reviewer" in (info.instructions or ""):
            assert images[0].media_type == "application/pdf"
            assert images[0].data == pdf.read_bytes()
            assert "marksAndNumbers" not in text(messages)
            topology_images.append(len(images))
        if "Assigned section: parties." in text(messages):
            assert not images
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder, pdf_path=pdf, relations_responder=responder)
    result = await subject.refine(label())
    assert result["status"] == "reviewed_candidate"
    assert topology_images == [1]


@pytest.mark.asyncio
async def test_layout_context_survives_correction_and_rereview_without_affecting_other_scopes(
    tmp_path,
):
    from pydantic_ai import BinaryContent
    from pypdf import PdfWriter

    pdf = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with pdf.open("wb") as file:
        writer.write(file)
    seen = []

    def responder(messages, info):
        source = text(messages)
        images = [x for p in parts(messages) for x in p.content if isinstance(x, BinaryContent)]
        if "Assigned section: parties." not in source:
            assert bool(images) == ("Assigned section: route_transport." in source)
            return response({"response": {"status": "pass", "findings": []}})
        seen.append(len(images))
        if len(seen) == 1:
            return response(
                {"response": {"pages": [1], "question": "Who owns the postal continuation?"}}
            )
        if len(seen) == 2:
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": "shipper.addressLine",
                                "issue": "missing",
                                "explanation": "Owned postcode",
                                "suggestedCorrection": "Restore 97132",
                            }
                        ],
                    }
                }
            )
        if "Findings:" in source:
            return correction(messages, {"parties": label()["documentPatch"]["parties"]})
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder, pdf_path=pdf)
    candidate = label()
    candidate["documentPatch"]["parties"]["shipper"]["addressLine"] = "UNIT 4 USA"
    result = await subject.refine(candidate)
    assert result["target"] == label()
    assert seen == [0, 1, 1, 1]
    assert subject._layout_pages == {"parties": [1], "route_transport": [1]}


@pytest.mark.asyncio
async def test_provider_failure_preserves_other_reviews_and_no_success_export(tmp_path):
    async def responder(messages, info):
        if "Assigned section: cargo." in text(messages):
            raise RuntimeError("provider failure")
        await asyncio.sleep(0.005)
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    with pytest.raises(ExceptionGroup, match="completed siblings preserved"):
        await subject.refine(BillOfLadingExtractionV7Label.model_validate_json(encoded(label())))
    assert len(list((subject.output_dir / "reviews").glob("*.json"))) == 4
    assert len(subject.receipts) == 6
    assert not (subject.output_dir / "target.json").exists()


def test_review_status_and_schema_command_without_credentials(tmp_path, monkeypatch):
    import argparse

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        SectionReview(
            status="pass",
            explanation="Incoherent fixture verdict",
            findings=[{"field": "x", "issue": "missing", "explanation": "x"}],
        )
    output = tmp_path / "schema.json"
    result = execute(
        argparse.Namespace(
            command="schema",
            project_root=ROOT,
            config=ROOT / "configs/labeling_agents/mpci_bl_direct.yaml",
            output=output,
        )
    )
    assert result["paid_requests"] == 0
    assert output.is_file()


@pytest.mark.parametrize("printed", ["001234560000", "00.12.34.56.00.00", "00 12 34\n56 00 00"])
def test_literal_fidelity_accepts_formatting_but_not_changed_digits(printed):
    candidate = label()
    goods = candidate["documentPatch"]["goodsItemDetails"][0]
    for code, expected in [
        ("001234560000", 0),
        ("00123456000000", 1),
        ("001234", 1),
        ("1234560000", 1),
    ]:
        goods["hsCodes"] = [code]
        # Separated groups could also be separate codes: ownership still needs review.
        if " " in printed and code in ("001234", "1234560000"):
            continue
        assert (
            len(source_fidelity_findings(candidate, "TTNU8111930 HS CODE: " + printed, "cargo"))
            == expected
        )
    candidate["documentPatch"]["transport"] = {
        "vesselName": "EF OLIVIA",
        "voyageNumber": "0NVC3S1MA",
    }
    assert not source_fidelity_findings(candidate, "EF-OLIVIA / 0NVC3S1MA", "route_transport")
    assert len(source_fidelity_findings(candidate, OCR, "route_transport")) == 2
    assert (
        len(source_fidelity_findings(candidate, "EF OLIVIA / X0NVC3S1MA", "route_transport")) == 1
    )


@pytest.mark.asyncio
async def test_model_pass_cannot_override_literal_gate_and_unchanged_issue_is_not_retried(tmp_path):
    candidate = label()
    candidate["documentPatch"]["goodsItemDetails"][0]["hsCodes"] = ["00123456000000"]

    def responder(messages, info):
        if "Findings:" in text(messages):
            return correction(
                messages, {"goodsItemDetails": candidate["documentPatch"]["goodsItemDetails"]}
            )
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    subject.ocr += "HS: 00.12.34.56.00.00"
    result = await subject.refine(candidate)
    assert result["status"] == "needs_adjudication"
    assert result["reviews"]["cargo"]["findings"][0]["issue"] == "unsupported"
    assert sum(r["stage"] == "corrector" for r in subject.receipts) == 1
    assert result["target"] == candidate  # Diagnostics are not automatic edits.


@pytest.mark.asyncio
async def test_new_final_review_finding_gets_one_more_wave_and_then_stops(tmp_path):
    count = 0

    def responder(messages, info):
        nonlocal count
        source = text(messages)
        if "Findings:" in source:
            count += 1
            return correction(messages, {"parties": label()["documentPatch"]["parties"]})
        if "Assigned section: parties." in source:
            return response(
                {
                    "response": {
                        "status": "corrections_needed",
                        "findings": [
                            {
                                "field": f"parties.shipper.new_defect_{count}",
                                "issue": "missing",
                                "explanation": "New supported fact found after correction.",
                                "suggestedCorrection": "Restore the newly discovered fact.",
                            }
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    result = await subject.refine(label())
    assert count == 2
    assert result["status"] == "needs_adjudication"
    assert [r["wave"] for r in result["correctionDecisions"]] == [1, 2]
    assert (subject.output_dir / "wave-1-target.json").exists()
    assert (subject.output_dir / "wave-2-target.json").exists()


@pytest.mark.asyncio
async def test_mixed_ambiguity_preserves_independent_correction_without_retry(tmp_path):
    def responder(messages, info):
        source = text(messages)
        if "Findings:" in source:
            findings = json.JSONDecoder().raw_decode(source.split("Findings:\n", 1)[1])[0]
            return response(
                {
                    "response": {
                        "decisions": [
                            {
                                "findingId": f["id"],
                                "disposition": "unresolved"
                                if f["issue"] == "ambiguous"
                                else "accept",
                                "explanation": "Address clear; identity undecidable.",
                                "changedFields": correction_fields(
                                    messages, {"parties": label()["documentPatch"]["parties"]}
                                )
                                if i == 0
                                else [],
                            }
                            for i, f in enumerate(findings)
                        ],
                        "values": {"parties": label()["documentPatch"]["parties"]},
                    }
                }
            )
        if (
            "Assigned section: parties." in source
            and "Changes in the last correction wave" not in source
        ):
            return response(
                {
                    "response": {
                        "status": "unresolved",
                        "findings": [
                            {
                                "field": "shipper.addressLine",
                                "issue": "missing",
                                "explanation": "Postcode omitted",
                                "suggestedCorrection": "Restore 97132",
                            },
                            {
                                "field": "consignee",
                                "issue": "ambiguous",
                                "explanation": "Two possible roles",
                            },
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    candidate = label()
    candidate["documentPatch"]["parties"]["shipper"]["addressLine"] = "UNIT 4 USA"
    subject = flow(tmp_path, responder)
    result = await subject.refine(candidate)
    assert result["target"] == label()
    assert result["status"] == "needs_adjudication"
    assert len(result["reviews"]["parties"]["findings"]) == 1
    assert result["reviews"]["parties"]["findings"][0]["field"] == "consignee"
    assert sum(r["stage"] == "corrector" for r in subject.receipts) == 1


@pytest.mark.asyncio
async def test_topology_review_can_block_a_clean_general_cargo_review(tmp_path):
    def responder(messages, info):
        if "focused cargo-association reviewer" in (info.instructions or ""):
            return response(
                {
                    "response": {
                        "status": "unresolved",
                        "findings": [
                            {
                                "field": "goodsItemDetails.splitGoodsPlacement",
                                "issue": "ambiguous",
                                "explanation": "Equal totals cannot establish assignments.",
                            }
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder, relations_responder=responder)
    result = await subject.refine(label())
    assert result["status"] == "needs_adjudication"
    assert result["reviews"]["cargo"]["status"] == "unresolved"
    assert len(subject.receipts) == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_at", ["review", "correction"])
@pytest.mark.parametrize("valid_map", [True, False])
async def test_source_map_repair_is_grounded_atomic_and_used_for_totals(
    tmp_path, repair_at, valid_map
):
    source = {
        "layoutInterpretation": "Two rows of one goods item; dots group thousands.",
        "products": [{"key": "g1", "description": "YARN"}],
        "statements": [
            {
                "products": ["g1"],
                "containers": ["TTNU8111930"],
                "scope": "portion",
                "packages": [{"quantity": 12, "packageType": "BAGS", "level": "target"}],
                "grossWeight": {"printedValue": f"{n}.000", "value": n, "unit": "KG"},
                "explanation": "An explicitly printed row owned by yarn.",
            }
            for n in (5, 6)
        ],
        "uncertainties": [],
    }
    repaired = copy.deepcopy(source)
    for statement in repaired["statements"]:
        statement["grossWeight"]["value"] *= 1000
    if not valid_map:
        repaired["statements"][0]["grossWeight"]["printedValue"] = "5000.999"
    candidate = label()
    candidate["documentPatch"]["goodsItemDetails"][0]["grossWeight"] = {
        "value": 11,
        "unit": "kilogram",
    }
    fixed = copy.deepcopy(candidate)
    fixed["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["value"] = 11000
    reviews = 0

    def relations(messages, info):
        nonlocal reviews
        reviews += 1
        finding = {
            "field": "goodsItemDetails.grossWeight",
            "issue": "normalization",
            "explanation": "The printed masses use thousands separators.",
            "suggestedCorrection": "Interpret both row masses as thousands and sum to 11000.",
        }
        return response(
            {
                "response": {
                    "status": "corrections_needed" if reviews == 1 else "pass",
                    "findings": [finding] if reviews == 1 else [],
                },
                "mapCorrection": repaired if repair_at == "review" and reviews == 1 else None,
            }
        )

    def responder(messages, info):
        if "Findings:" in text(messages):
            reply = correction(
                messages, {"goodsItemDetails": fixed["documentPatch"]["goodsItemDetails"]}
            )
            raw = json.loads(reply.parts[0].content)
            raw["response"]["mapCorrection"] = repaired if repair_at == "correction" else None
            return response(raw)
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder, relations_responder=relations, source_map=source)
    subject.ocr += "\n12 BAGS 5.000 KG\n12 BAGS 6.000 KG"
    result = await subject.refine(candidate)
    assert result["target"] == (fixed if valid_map else candidate)
    assert result["status"] == ("reviewed_candidate" if valid_map else "needs_adjudication")
    assert subject._cargo_source.statements[0].grossWeight.value == (5000 if valid_map else 5)
    assert sum(r["stage"] == "cargo_mapper" for r in subject.receipts) == 1


@pytest.mark.asyncio
async def test_ambiguous_value_allows_safe_omission_without_losing_source_hold(tmp_path):
    candidate = label()
    candidate["documentPatch"]["issueDate"] = "2025-04-10"

    def responder(messages, info):
        source = text(messages)
        if "Findings:" in source:
            values = {
                k: v
                for k, v in candidate["documentPatch"].items()
                if k in SECTION_FIELDS["metadata_freight"]
            }
            values["issueDate"] = None
            reply = correction(messages, values)
            raw = json.loads(reply.parts[0].content)
            raw["response"]["decisions"][0]["disposition"] = "unresolved"
            return response(raw)
        if "Assigned section: metadata_freight." in source:
            return response(
                {
                    "response": {
                        "status": "unresolved",
                        "findings": [
                            {
                                "field": "issueDate",
                                "issue": "ambiguous",
                                "explanation": "Date order is unresolved; no supported ISO value.",
                                "suggestedCorrection": (
                                    "Omit the assumed date under the field's policy."
                                    if "Changes in the last correction wave" not in source
                                    else None
                                ),
                            }
                        ],
                    }
                }
            )
        return response({"response": {"status": "pass", "findings": []}})

    subject = flow(tmp_path, responder)
    subject.ocr += " ISSUE DATE: 10-04-2025"
    result = await subject.refine(candidate)
    assert "issueDate" not in result["target"]["documentPatch"]
    assert result["status"] == "needs_adjudication"
    assert result["target"]["documentPatch"]["billOfLadingNumber"] == "BL001"
    assert sum(r["stage"] == "corrector" for r in subject.receipts) == 1
