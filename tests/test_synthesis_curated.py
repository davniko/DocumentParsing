import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from document_ocr.synthesis.curated import (
    LexicalBatch,
    Runner,
    SourceContract,
    _number_style,
    compile_contract,
    digest,
    render,
    save,
    scenario_values,
    validate_candidate,
)


def example():
    row = {
        "documentId": "test",
        ("joinedRawText"): (
            "--- PAGE 1 ---\nSHIPPER\nACME LTD\n17 INDUSTRIAL ROAD\nDELHI, "
            "INDIA\nTAX: 9917\nGOODS\nSTEEL BOLTS\nGROSS: 1,200.00 KG\n24 "
            "BOXES\n"
        ),
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": "non_negotiable",
                "parties": {
                    "shipper": {
                        "name": "ACME LTD",
                        "addressLine": "17 INDUSTRIAL ROAD DELHI, INDIA",
                        "country": "INDIA",
                    }
                },
                "goodsItemDetails": [
                    {
                        "description": "STEEL BOLTS",
                        "grossWeight": {"value": 1200.0, "unit": "kilogram"},
                        "numberAndTypeOfPackages": [
                            {"packageQuantity": 24, "typeCategory": "PACKAGE_BOX"}
                        ],
                    }
                ],
            },
        },
    }
    variables = []

    def var(key, kind, value, text, presentation="text", required=()):
        variables.append(
            dict(
                key=key,
                kind=kind,
                value=value,
                meaning=key,
                required_literals=list(required),
                occurrences=[dict(text=text, occurrence=1, presentation=presentation)],
            )
        )

    var("name", "name", "ACME LTD", "ACME LTD")
    var(
        "postal",
        "postal",
        "17 INDUSTRIAL ROAD DELHI, INDIA",
        "17 INDUSTRIAL ROAD\nDELHI, INDIA",
        required=["INDIA"],
    )
    var("product", "product", "STEEL BOLTS", "STEEL BOLTS", required=["STEEL"])
    var("gross", "mass", "1200", "1,200.00", "number")
    var("count", "count", "24", "24", "number")
    targets = [
        ("documentPatch.parties.shipper.name", "{name}"),
        ("documentPatch.parties.shipper.addressLine", "{postal}"),
        ("documentPatch.goodsItemDetails[0].description", "{product}"),
        ("documentPatch.goodsItemDetails[0].grossWeight.value", "{gross}"),
        ("documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity", "{count}"),
    ]
    contract = SourceContract(
        variables=variables,
        targets=[dict(path=p, expression=e) for p, e in targets],
        fixed_context="country and classification fixed",
    )
    return row, contract


def test_identity_and_changed_postal_replay_protect_unrelated_text():
    row, contract = example()
    original = deepcopy(row)
    values = {v.key: v.value for v in contract.variables}
    text, target, proof = render(row, contract, values)
    assert text == row["joinedRawText"] and target == row["target"]
    values.update(postal="203 RIVER PARK MUMBAI, INDIA", name="RIVER TRADING LTD")
    text, target, _ = render(row, contract, values)
    assert "TAX: 9917" in text and text.count("INDIA") == 1
    assert target["documentPatch"]["parties"]["shipper"]["addressLine"] == values["postal"]
    assert row == original
    assert proof["exact_literal_regions"]


@pytest.mark.parametrize(
    "text,old,new,expected",
    [
        ("1,200.00", "1200", "960", "960.00"),
        ("1.200,00", "1200", "960", "960,00"),
        ("1 200", "1200", "960", "960"),
        ("024", "24", "12", "012"),
        ("11870.000", "11870", "7100", "7100.000"),
        ("18.280", "18280", "10968", "10.968"),
    ],
)
def test_numeric_presentation_is_source_verified(text, old, new, expected):
    assert _number_style(text, old, new) == expected


def test_contract_refuses_wrong_span_wrong_value_overlap_or_changed_gold():
    row, contract = example()
    broken = contract.model_copy(deep=True)
    broken.variables[0].value = "OTHER LTD"
    with pytest.raises(ValueError, match="differs from baseline"):
        compile_contract(row, broken)
    broken = contract.model_copy(deep=True)
    broken.variables[0].occurrences[0].occurrence = 2
    with pytest.raises(ValueError, match="missing occurrence"):
        compile_contract(row, broken)
    broken = contract.model_copy(deep=True)
    broken.variables[1].occurrences.append(broken.variables[0].occurrences[0])
    with pytest.raises(ValueError):
        compile_contract(row, broken)
    broken = contract.model_copy(deep=True)
    broken.targets[0].expression += " EXTRA"
    with pytest.raises(ValueError, match="identity replay"):
        compile_contract(row, broken)


def test_scenario_is_reproducible_and_integral_and_renderable():
    row, contract = example()
    a = scenario_values(contract, "sample", 42, 0)
    assert a == scenario_values(contract, "sample", 42, 0)
    assert int(a["count"]) < 24 and float(a["gross"]) < 1200
    render(row, contract, a)
    with pytest.raises(ValueError, match="lost required literal"):
        render(row, contract, {**a, "postal": "123 NEW ROAD"})
    with pytest.raises(ValueError, match="numeric specifications"):
        render(row, contract, {**a, "product": "STEEL BOLTS 999"})


def test_publication_replay_rejects_tampering_even_when_artifacts_are_self_consistent():
    row, contract = example()
    sid = "syn_v7_" + digest([row["documentId"], 42, 0])[:24]
    values = scenario_values(contract, sid, 42, 0)
    text, target, proof = render(row, contract, values)
    record = dict(
        documentId=sid,
        sourceDocumentId=row["documentId"],
        values=values,
        joinedRawText=text,
        target=target,
        proof=proof,
        joinedRawTextSha256=digest(text.encode()),
        contractSha256=digest(contract.model_dump(mode="json")),
        seed=42,
        variant=0,
        variantCount=3,
    )
    validate_candidate(row, contract, record)
    bad = deepcopy(record)
    bad["values"]["count"] = "9917"  # A tax ID is not a shipment quantity.
    text, target, proof = render(row, contract, bad["values"])
    bad.update(
        joinedRawText=text, target=target, proof=proof, joinedRawTextSha256=digest(text.encode())
    )
    with pytest.raises(ValueError, match="host-owned"):
        validate_candidate(row, contract, bad)
    bad = deepcopy(record)
    bad["target"]["documentPatch"]["parties"]["shipper"]["addressLine"] += " TAX: 9917"
    with pytest.raises(ValueError, match="exact source/scenario replay"):
        validate_candidate(row, contract, bad)
    bad = deepcopy(record)
    bad["joinedRawText"] = bad["joinedRawText"].replace("TAX: 9917", "TAX: 0000")
    with pytest.raises(ValueError, match="exact source/scenario replay"):
        validate_candidate(row, contract, bad)


def test_country_anchor_is_a_token_not_a_substring_and_is_not_duplicated():
    row, contract = example()
    values = {v.key: v.value for v in contract.variables}
    with pytest.raises(ValueError, match="lost required literal"):
        render(row, contract, {**values, "postal": "17 INDIANA ROAD"})
    with pytest.raises(ValueError, match="repeated postal anchor"):
        render(row, contract, {**values, "postal": "17 ROAD DELHI, INDIA INDIA"})


def test_template_owned_terminal_delimiter_is_not_generated_twice():
    row, contract = example()
    row["joinedRawText"] = row["joinedRawText"].replace("INDIA\nTAX", "INDIA.\nTAX")
    values = {v.key: v.value for v in contract.variables}
    render(row, contract, values)
    with pytest.raises(ValueError, match="template-owned delimiter"):
        render(row, contract, {**values, "postal": "29 RIVER ROAD DELHI, INDIA."})


def test_failed_review_preserves_candidates_but_publication_requires_exact_approval(tmp_path):
    row, contract = example()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/synthesis/mpci_bl_curated_v7_pilot24.yaml").read_text()
    )
    config.update(output=str(tmp_path), seed=42, variants_per_source=3)
    # Exercise the actual serialized-config/provider/frozen-schema boundary.
    # Mock only paid calls, not the initialization path used by the CLI.
    runner = Runner(root, config)
    source_dir = tmp_path / "sources" / row["documentId"]
    save(
        source_dir / "contract.json",
        dict(
            sourceSha256=digest(row["joinedRawText"].encode()),
            targetSha256=digest(row["target"]),
            contract=contract.model_dump(mode="json"),
        ),
    )

    async def call(stage, *args):
        if stage == "lexical-review":
            raise RuntimeError("provider unavailable")
        return LexicalBatch(
            variants=[
                dict(
                    values=[
                        dict(key=v.key, value=v.value)
                        for v in contract.variables
                        if v.kind in {"postal", "name", "product"}
                    ]
                )
                for _ in range(3)
            ]
        )

    runner.call = call
    result = asyncio.run(runner.source(row, "generate"))
    assert result["status"] == "review"
    records = [json.loads(p.read_text()) for p in (tmp_path / "samples").glob("*.json")]
    assert len(records) == 3 and all(r["reviewError"] for r in records)
    report = runner.validate_and_publish(
        {row["documentId"]: row}, [row["documentId"]], publish=True
    )
    assert not report["published"] and report["failures"]
    save(
        source_dir / "adjudication.json",
        dict(
            decision="approved",
            rationale="Reviewed fixture",
            contractSha256=digest(contract.model_dump(mode="json")),
            sampleSha256={r["documentId"]: digest(r) for r in records},
        ),
    )
    report = runner.validate_and_publish(
        {row["documentId"]: row}, [row["documentId"]], publish=True
    )
    assert report["published"] and report["valid"] == 3
    bad = records[0]
    bad["joinedRawText"] += " WRONG"
    save(tmp_path / "samples" / (bad["documentId"] + ".json"), bad)
    report = runner.validate_and_publish(
        {row["documentId"]: row}, [row["documentId"]], publish=True
    )
    assert not report["published"] and report["failures"]


def test_new_quantity_cannot_leave_an_existing_placement_stale():
    row, contract = example()
    row["target"]["documentPatch"]["containerInformation"] = [
        dict(equipmentIdentifier="MSCU1234566")
    ]
    row["target"]["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"] = [
        dict(equipmentIdentifier="MSCU1234566", packageQuantity=24)
    ]
    values = scenario_values(contract, "test", 42, 0)
    with pytest.raises(ValueError, match="allocation no longer sums"):
        render(row, contract, values)
@pytest.mark.parametrize(
    "source,value",
    [
        ("39 Banni El-Abbas, Bab Sharqi ,\nAlexandria , Egypt",
         "22 SULTAN HUSSEIN ST., ANTOUKHY , ALEXANDRIA , EGYPT"),
        ("FIRST LINE\nSECOND LINE\nTHIRD LINE", "SHORT ADDRESS"),
        ("FIRST LINE\nSECOND LINE", "BUILDING A ; DISTRICT B , CITY C , COUNTRY D"),
    ],
)
def test_layout_keeps_punctuation_with_words_without_changing_content(source, value):
    from document_ocr.synthesis.curated import layout_surface

    rendered = layout_surface(source, value)
    assert " ".join(rendered.split()) == " ".join(value.split())
    assert all(line and line[0] not in ",.;:!?" for line in rendered.splitlines())
