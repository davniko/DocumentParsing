from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/bill_of_lading_v2_full_run.py"
_SPEC = importlib.util.spec_from_file_location("bill_of_lading_v2_full_run", _TOOL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to import semantic-v2 full-run tool")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _work_item(document_id: str, raw_hash: str, raster_hash: str) -> dict[str, object]:
    return {
        "source": {
            "documentId": document_id,
            "joinedRawTextSha256": raw_hash,
            "pages": [{"rasterSha256": raster_hash}],
        }
    }


def test_page_splitter_preserves_order() -> None:
    joined = "--- PAGE 1 ---\nFIRST\n\n--- PAGE 2 ---\nSECOND"

    assert _MODULE._page_texts(joined) == {1: "FIRST", 2: "SECOND"}


def _raw_evidence(page: int, raw_value: str, excerpt: str) -> SimpleNamespace:
    return SimpleNamespace(pageNumber=page, rawValue=raw_value, ocrExcerpt=excerpt)


def test_evidence_order_uses_raw_value_position_inside_shared_excerpt() -> None:
    excerpt = "GROSS WEIGHT\nCargo\nKGS\n27256.030"
    pages = {1: excerpt}
    unit = _raw_evidence(1, "KGS", excerpt)
    value = _raw_evidence(1, "27256.030", excerpt)

    _MODULE._verify_evidence(((unit, value),), pages, "doc_order")
    with pytest.raises(RuntimeError, match="not verbatim/in source order"):
        _MODULE._verify_evidence(((value, unit),), pages, "doc_order")


def test_evidence_order_rejects_decreasing_page_numbers() -> None:
    pages = {1: "FIRST", 2: "SECOND"}

    with pytest.raises(RuntimeError, match="not verbatim/in source order"):
        _MODULE._verify_evidence(
            ((_raw_evidence(2, "SECOND", "SECOND"), _raw_evidence(1, "FIRST", "FIRST")),),
            pages,
            "doc_pages",
        )


@pytest.mark.parametrize("raw_value", ["APR-29-2025", "April-29-2025", "APR-29-25"])
def test_date_evidence_supports_month_first_hyphenated_dates(raw_value: str) -> None:
    assert _MODULE._date_evidence_supports("2025-04-29", raw_value)
    assert not _MODULE._date_evidence_supports("2025-04-28", raw_value)


@pytest.mark.parametrize("raw_value", ["APR.29.2024", "APR/29/2024", "APR 29.2024"])
def test_date_evidence_supports_month_first_mixed_punctuation(raw_value: str) -> None:
    assert _MODULE._date_evidence_supports("2024-04-29", raw_value)
    assert not _MODULE._date_evidence_supports("2024-04-28", raw_value)


def test_date_evidence_supports_bill_of_lading_day_of_prose() -> None:
    raw_value = "17TH day of ___ MARCH 2024"

    assert _MODULE._date_evidence_supports("2024-03-17", raw_value)
    assert not _MODULE._date_evidence_supports("2024-03-18", raw_value)


def test_date_evidence_supports_ocr_split_two_digit_day() -> None:
    raw_value = "1 4 FEB 2024"

    assert _MODULE._date_evidence_supports("2024-02-14", raw_value)
    assert not _MODULE._date_evidence_supports("2024-02-04", raw_value)


@pytest.mark.parametrize(
    "raw_value",
    ["27.JUL.2023", "27/JUL/2023", "27-JUL-2023", "27.JUL 2023"],
)
def test_date_evidence_supports_alphabetic_month_with_punctuation(
    raw_value: str,
) -> None:
    assert _MODULE._date_evidence_supports("2023-07-27", raw_value)
    assert not _MODULE._date_evidence_supports("2023-07-28", raw_value)


@pytest.mark.parametrize(
    "raw_value",
    ["2024-FEB-15", "2024/FEB/15", "2024.FEB.15", "2024 FEB 15"],
)
def test_date_evidence_supports_year_first_alphabetic_month(raw_value: str) -> None:
    assert _MODULE._date_evidence_supports("2024-02-15", raw_value)
    assert not _MODULE._date_evidence_supports("2024-02-14", raw_value)


@pytest.mark.parametrize("raw_value", ["14-08- 2023", "14-08-\n2023"])
def test_date_evidence_supports_numeric_date_wrapped_after_separator(
    raw_value: str,
) -> None:
    assert _MODULE._date_evidence_supports("2023-08-14", raw_value)
    assert not _MODULE._date_evidence_supports("2023-08-15", raw_value)


@pytest.mark.parametrize("raw_value", ["26-03-24", "26/03/24", "26.03.24"])
def test_date_evidence_supports_day_first_numeric_two_digit_year(
    raw_value: str,
) -> None:
    assert _MODULE._date_evidence_supports("2024-03-26", raw_value)
    assert not _MODULE._date_evidence_supports("2024-03-25", raw_value)


@pytest.mark.parametrize("raw_value", ["03-26-24", "03/26/24", "03.26.24"])
def test_date_evidence_supports_month_first_numeric_two_digit_year(
    raw_value: str,
) -> None:
    assert _MODULE._date_evidence_supports("2024-03-26", raw_value)
    assert not _MODULE._date_evidence_supports("2024-03-25", raw_value)


@pytest.mark.parametrize(
    "raw_value",
    ["17- th June, 2023", "17 -TH JUNE 2023", "17th June 2023"],
)
def test_date_evidence_supports_detached_ordinal_suffix(raw_value: str) -> None:
    assert _MODULE._date_evidence_supports("2023-06-17", raw_value)
    assert not _MODULE._date_evidence_supports("2023-06-18", raw_value)


@pytest.mark.parametrize("raw_value", ["1000,00 mt", "1000.00 MT", "1000 metric tonnes"])
def test_evidence_supports_exact_metric_tonne_to_kilogram_mass_conversion(
    raw_value: str,
) -> None:
    value_path = "documentPatch.goodsItems[0].netWeight.value"
    unit_path = "documentPatch.goodsItems[0].netWeight.unit"

    assert _MODULE._target_leaf_is_supported(value_path, 1_000_000, raw_value)
    assert _MODULE._target_leaf_is_supported(unit_path, "kilogram", raw_value)
    assert not _MODULE._target_leaf_is_supported(value_path, 999_000, raw_value)


def test_tonne_token_does_not_authorize_conversion_outside_a_mass_path() -> None:
    assert not _MODULE._target_leaf_is_supported(
        "documentPatch.goodsItems[0].volume.value", 1_000_000, "1000 mt"
    )


@pytest.mark.parametrize("raw_value", ["252.412.420", "252,412,420"])
def test_numeric_evidence_supports_grouped_integer_with_final_decimal(
    raw_value: str,
) -> None:
    path = "documentPatch.goodsItems[0].netWeight.value"

    assert _MODULE._target_leaf_is_supported(path, 252_412.420, raw_value)
    assert _MODULE._target_leaf_is_supported(path, 252_412_420, raw_value)
    assert not _MODULE._target_leaf_is_supported(path, 252_412.421, raw_value)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("4 656KG", 4_656),
        ("4\n527KG", 4_527),
        ("14 600 KGS", 14_600),
        ("14\r\n201 kilograms", 14_201),
        ("Kgs 200 541", 200_541),
        ("KILOGRAMS: 200 541", 200_541),
    ],
)
def test_mass_evidence_supports_ocr_space_grouped_thousands(
    raw_value: str,
    expected: int,
) -> None:
    path = "documentPatch.goodsItems[0].grossWeight.value"

    assert _MODULE._target_leaf_is_supported(path, expected, raw_value)
    assert not _MODULE._target_leaf_is_supported(path, expected + 1, raw_value)


def test_mass_evidence_does_not_join_independent_unit_bound_numbers() -> None:
    path = "documentPatch.goodsItems[0].grossWeight.value"

    assert not _MODULE._target_leaf_is_supported(path, 4_527, "4 KG / 527 KG")
    assert not _MODULE._target_leaf_is_supported(
        "documentPatch.goodsItems[0].packages[0].quantity",
        4_527,
        "4 527",
    )


@pytest.mark.parametrize("raw_value", ["pieces 36 071", "36 071 PIECES"])
def test_package_evidence_supports_unit_bound_space_grouped_thousands(
    raw_value: str,
) -> None:
    path = "documentPatch.goodsItems[0].packages[0].quantity"

    assert _MODULE._target_leaf_is_supported(path, 36_071, raw_value)
    assert not _MODULE._target_leaf_is_supported(path, 36_072, raw_value)


def test_package_evidence_does_not_join_bare_or_independent_numbers() -> None:
    path = "documentPatch.goodsItems[0].packages[0].quantity"

    assert not _MODULE._target_leaf_is_supported(path, 36_071, "36 071")
    assert not _MODULE._target_leaf_is_supported(path, 36_071, "36 PCS / 071 PCS")


def test_run_configuration_selects_a_second_immutable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "DEST",
        "MAX_CONCURRENT_WORKERS",
        "PLATFORM_TOTAL_SLOTS",
        "QUALITY_FILTER_MANIFEST_SHA256",
        "QUALITY_FILTER_ROOT",
        "REQUIRE_INDEPENDENT_REVIEW",
        "REVIEWER_REASONING_EFFORT",
        "RUN_CONFIG_PATH",
        "RUN_ID",
        "RUN_KIND",
        "SOURCE_RUN",
        "SOURCE_RUN_ID",
        "V1_RUN",
        "WORKER_REASONING_EFFORT",
        "_EXPECTED",
    ):
        monkeypatch.setattr(_MODULE, name, getattr(_MODULE, name))
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)
    for relative in ("source", "quality", "v1", "artifacts/kie-labels"):
        (tmp_path / relative).mkdir(parents=True)
    config = tmp_path / "run.yaml"
    config.write_text(
        "\n".join(
            (
                "run_id: test-followup-run",
                "run_kind: test_quality_projection",
                "source_run: source",
                "source_run_id: extraction-test",
                "quality_filter_root: quality",
                f"quality_filter_manifest_sha256: {'a' * 64}",
                "v1_run: v1",
                "platform_total_slots: 9",
                "max_concurrent_workers: 8",
                "worker_reasoning_effort: max",
                "reviewer_reasoning_effort: max",
                "require_independent_review: true",
                "expected:",
                "  selectedDocuments: 2",
                "  selectedPages: 3",
                "  completeDocuments: 2",
                "  incompleteDocuments: 0",
                "  successfulPages: 3",
                "  failedPages: 0",
                "  nonLatinExcludedDocuments: 0",
                "  eligibleDocuments: 2",
                "  eligiblePages: 3",
                "  preLabelExcludedDocuments: 0",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    _MODULE._configure_run(config)

    assert _MODULE.RUN_ID == "test-followup-run"
    assert (tmp_path / "source").resolve() == _MODULE.SOURCE_RUN
    assert (tmp_path / "quality").resolve() == _MODULE.QUALITY_FILTER_ROOT
    assert _MODULE.PLATFORM_TOTAL_SLOTS == 9
    assert _MODULE.MAX_CONCURRENT_WORKERS == 8
    assert _MODULE.WORKER_REASONING_EFFORT == "max"
    assert _MODULE.REVIEWER_REASONING_EFFORT == "max"
    assert _MODULE.REQUIRE_INDEPENDENT_REVIEW is True
    assert _MODULE._EXPECTED["eligibleDocuments"] == 2


def test_quality_projection_is_bound_to_exact_joined_ocr_and_page_identity() -> None:
    raw_text = "RAW OCR VALUE"
    raw_digest = _MODULE._digest(raw_text.encode())
    document_id = f"doc_{'1' * 64}"
    projected_page = {
        "page_index": 0,
        "page_number": 1,
        "page_id": "page-1",
        "extraction_id": "extract-1",
        "raw_ocr_text": raw_text,
        "raw_ocr_text_sha256": raw_digest,
        "raw_response_path": "raw/page.json",
        "raw_response_sha256": "2" * 64,
        "raster_path": "images/page.png",
        "raster_sha256": "3" * 64,
    }
    projected_reference = {
        key: value
        for key, value in projected_page.items()
        if key
        in {
            "page_index",
            "page_number",
            "page_id",
            "extraction_id",
            "raw_ocr_text_sha256",
            "raw_response_path",
            "raster_path",
            "raster_sha256",
        }
    }
    quality_document = {
        "document_id": document_id,
        "document_page_count": 1,
        "run_id": "extract-run",
        "source": {
            "source_uri": "file:///source.pdf",
            "local_canonical_path": "/source.pdf",
            "source_sha256": "4" * 64,
        },
        "pages": [projected_reference],
    }
    joined = f"--- PAGE 1 ---\n{raw_text}"
    item = {
        "source": {
            "documentId": document_id,
            "extractionRunId": "extract-run",
            "sourceUri": "file:///source.pdf",
            "localCanonicalPath": "/source.pdf",
            "sourceSha256": "4" * 64,
            "documentPageCount": 1,
            "joinedRawTextSha256": _MODULE._digest(joined.encode()),
            "pages": [
                {
                    "pageIndex": 0,
                    "pageNumber": 1,
                    "pageId": "page-1",
                    "extractionId": "extract-1",
                    "rawOcrTextSha256": raw_digest,
                    "rawResponsePath": "raw/page.json",
                    "rawResponseSha256": "2" * 64,
                    "rasterPath": "images/page.png",
                    "rasterSha256": "3" * 64,
                }
            ],
        },
        "joinedRawText": joined,
    }

    _MODULE._verify_quality_work_item(item, quality_document, [projected_page])

    corrupted = dict(projected_page, raw_ocr_text="DIFFERENT")
    with pytest.raises(RuntimeError, match="raw OCR identity mismatch"):
        _MODULE._verify_quality_work_item(item, quality_document, [corrupted])


def test_country_resolver_uses_frozen_iso_snapshot_and_explicit_aliases() -> None:
    index, digest = _MODULE._country_index()

    assert len(digest) == 64
    assert index[_MODULE._normalize_country("Netherlands")] == "NL"
    assert index[_MODULE._normalize_country("The Netherlands")] == "NL"
    assert index[_MODULE._normalize_country("U.K")] == "GB"
    assert index[_MODULE._normalize_country("Türkiye")] == "TR"
    assert index[_MODULE._normalize_country("A.R. Egypt")] == "EG"
    assert index[_MODULE._normalize_country("Al Yemen")] == "YE"
    assert index[_MODULE._normalize_country("China(CN)")] == "CN"
    assert index[_MODULE._normalize_country("Egypt(EG)")] == "EG"
    assert _MODULE._normalize_country("China(EG)") not in index
    assert index[_MODULE._normalize_country("Egitto")] == "EG"
    assert index[_MODULE._normalize_country("Egipto")] == "EG"
    assert index[_MODULE._normalize_country("Egypte")] == "EG"
    assert index[_MODULE._normalize_country("España")] == "ES"
    assert index[_MODULE._normalize_country("ESPANA")] == "ES"
    assert index[_MODULE._normalize_country("Republic of Egypt")] == "EG"
    assert index[_MODULE._normalize_country("Sierra Lione")] == "SL"
    assert index[_MODULE._normalize_country("P. R. China")] == "CN"
    assert index[_MODULE._normalize_country("Hongkong")] == "HK"
    assert index[_MODULE._normalize_country("Kingdom of Sudia Arabia")] == "SA"
    assert index[_MODULE._normalize_country("KSA")] == "SA"


def test_duplicate_groups_join_exact_ocr_and_first_page_raster_components() -> None:
    items = [
        _work_item("doc_a", "1" * 64, "a" * 64),
        _work_item("doc_b", "2" * 64, "a" * 64),
        _work_item("doc_c", "2" * 64, "c" * 64),
        _work_item("doc_d", "4" * 64, "d" * 64),
    ]

    groups = _MODULE._duplicate_groups(items)

    assert len(groups) == 1
    assert groups[0]["documentIds"] == ["doc_a", "doc_b", "doc_c"]
    assert {row["kind"] for row in groups[0]["identityEvidence"]} == {
        "first_page_raster",
        "joined_raw_text",
    }


def test_semantic_gate_checks_flavor_text_and_address_redundancy() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "TAX ID: 123",
                    "address": "11 MAIN STREET, CAIRO",
                    "city": "CAIRO",
                }
            },
            "goodsItems": [{"marksAndNumbers": ["SHIPPER'S LOAD & COUNT"]}],
        }
    }

    findings = _MODULE._semantic_findings(target)

    assert any("prohibited tax/boilerplate" in finding for finding in findings)
    assert any("address repeats separately emitted city" in finding for finding in findings)


def test_semantic_gate_distinguishes_swiss_canton_from_turkish_tax_office() -> None:
    canton_address = {
        "documentPatch": {
            "parties": {"shipper": {"address": "PIGNETS 2 VD 1028"}}
        }
    }
    dotted_tax_office = {
        "documentPatch": {
            "parties": {"shipper": {"address": "ERENKOY V.D. 2111507266"}}
        }
    }
    labeled_tax_office = {
        "documentPatch": {
            "parties": {"shipper": {"address": "ESENYURT VD NO: 600 042 8579"}}
        }
    }

    assert _MODULE._semantic_findings(canton_address) == []
    assert any(
        "prohibited tax/boilerplate" in finding
        for finding in _MODULE._semantic_findings(dotted_tax_office)
    )
    assert any(
        "prohibited tax/boilerplate" in finding
        for finding in _MODULE._semantic_findings(labeled_tax_office)
    )


def test_semantic_gate_rejects_explicit_null_in_sparse_target() -> None:
    target = {
        "documentPatch": {
            "masterBillOfLadingNumber": None,
            "parties": {"shipper": {"name": "EXAMPLE LTD", "city": None}},
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "documentPatch.masterBillOfLadingNumber: explicit null must be omitted from sparse target",
        "documentPatch.parties.shipper.city: explicit null must be omitted from sparse target",
    ]


def test_semantic_gate_rejects_postal_label_but_keeps_postal_value() -> None:
    contaminated = {
        "documentPatch": {"parties": {"shipper": {"address": "P.O. BOX 4 POSTAL CODE 107"}}}
    }
    clean = {"documentPatch": {"parties": {"shipper": {"address": "P.O. BOX 4 107"}}}}

    assert _MODULE._semantic_findings(contaminated) == [
        "documentPatch.parties.shipper.address: postal value retains its field label "
        "'P.O. BOX 4 POSTAL CODE 107'"
    ]
    assert _MODULE._semantic_findings(clean) == []


@pytest.mark.parametrize(
    ("package_quantities", "allocation_quantities"),
    [
        ([457, 19], [457]),
        ([2383, 633], [2383, 633]),
        ([22, 21], [22, 21]),
    ],
)
def test_semantic_gate_accepts_exact_container_allocation_coverage(
    package_quantities: list[int], allocation_quantities: list[int]
) -> None:
    target = {
        "documentPatch": {
            "goodsItems": [
                {
                    "packages": [
                        {"quantity": quantity} for quantity in package_quantities
                    ],
                    "containerAllocations": [
                        {"packageQuantity": quantity}
                        for quantity in allocation_quantities
                    ],
                }
            ]
        }
    }

    assert _MODULE._semantic_findings(target) == []


def test_semantic_gate_requires_allocation_quantities_for_quantified_packages() -> None:
    target = {
        "documentPatch": {
            "goodsItems": [
                {
                    "packages": [{"quantity": 1099}],
                    "containerAllocations": [{}, {"packageQuantity": 89}],
                }
            ]
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "documentPatch.goodsItems[0].containerAllocations: allocation quantities are "
        "required when the goods item has an emitted package quantity"
    ]


def test_semantic_gate_rejects_partial_or_mismatched_allocation_coverage() -> None:
    target = {
        "documentPatch": {
            "goodsItems": [
                {
                    "packages": [{"quantity": 100}],
                    "containerAllocations": [
                        {"packageQuantity": 40},
                        {"packageQuantity": 50},
                    ],
                }
            ]
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "documentPatch.goodsItems[0].containerAllocations: allocation quantities must "
        "exactly cover one emitted package level or the total of the emitted package levels"
    ]


@pytest.mark.parametrize("package_type", ["40' CNTR(S)", "1 CONTAINER", "CONTAINERS"])
def test_semantic_gate_rejects_container_counts_as_goods_package_levels(
    package_type: str,
) -> None:
    target = {
        "documentPatch": {
            "goodsItems": [{"packages": [{"quantity": 1, "type": package_type}]}]
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "documentPatch.goodsItems[0].packages[0].type: container count/type must not be "
        "emitted as a goods package level"
    ]


@pytest.mark.parametrize(
    "heading",
    [
        "Delivery Agent at place of delivery",
        "9. Destination Agent",
        "AGENT AT PORT OF DISCHARGE",
        "AGENT AT PORT OF DESTINATION",
        "PORT OF DISCHARGE AGENT: ARKAS EGYPT",
        "Agent's Address at Destination:",
        "Agent to contact at Destination",
        "FOR DELIVERY OF GOODS PLEASE APPLY TO",
        "FOR DELIVERY OF GOODS, PLEASE APPLY TO:",
        "For cargo Delivery, Please Contact:",
        "FOR RELEASE OF CARGO APPLY TO",
        "TO OBTAIN DELIVERY CONTACT:",
        "(AS FRT FWDRS DELIVERY AGENT ONLY)",
        "(AS FREIGHT FORWARDERS DELIVERY AGENT ONLY)",
        (
            "IN CASE CONTAINER HAS NOT BEEN RETURNED EMPTY TO THE CARRIER'S AGENT. "
            "DETAILS OF THE AGENT AT P.O.D.: PAN MARINE SHIPPING-EGYPT"
        ),
    ],
)
def test_delivery_agent_role_gate_accepts_explicit_role_heading(heading: str) -> None:
    target = {"documentPatch": {"parties": {"deliveryAgent": {"name": "AGENT"}}}}

    assert _MODULE._source_role_findings(target, heading) == []


@pytest.mark.parametrize(
    "heading",
    [
        "SHIPPING AGENT DETAILS",
        "DESTINATION OFFICE ADDRESS",
        "NAME AND FULL ADDRESS OF SHIPPING AGENT IN EGYPT",
        "AGENT ADDRESS",
    ],
)
def test_delivery_agent_role_gate_rejects_generic_agent_heading(heading: str) -> None:
    target = {"documentPatch": {"parties": {"deliveryAgent": {"name": "AGENT"}}}}

    assert _MODULE._source_role_findings(target, heading) == [
        "deliveryAgent requires an explicit destination/discharge/delivery-agent source heading"
    ]


@pytest.mark.parametrize(
    ("target_path", "role"),
    [
        ("documentPatch.goodsItems[0].origin.name", "goods origin"),
        ("documentPatch.parties.shipper.country", "party-block geography"),
        ("documentPatch.parties.notifyParties[0].country", "party-block geography"),
    ],
)
def test_semantic_gate_rejects_foreign_exporter_country_evidence(
    target_path: str, role: str
) -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath=target_path,
                rawOcrEvidence=(
                    SimpleNamespace(ocrExcerpt="FOREIGN EXPORTER COUNTRY:CHINA"),
                ),
            ),
        )
    )

    assert _MODULE._foreign_exporter_country_evidence_findings(annotation) == [
        f"{target_path}: FOREIGN EXPORTER COUNTRY identifies exporter metadata, not {role}"
    ]


def test_semantic_gate_accepts_explicit_country_of_origin_evidence() -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath="documentPatch.goodsItems[0].origin.name",
                rawOcrEvidence=(SimpleNamespace(ocrExcerpt="COUNTRY OF ORIGIN: CHINA"),),
            ),
        )
    )

    assert _MODULE._foreign_exporter_country_evidence_findings(annotation) == []


def test_evidence_normalization_gate_accepts_exact_verbatim_string() -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath="documentPatch.billOfLadingNumber",
                evidenceKind="verbatim",
                rawOcrEvidence=(
                    SimpleNamespace(rawValue="BILL OF LADING NO."),
                    SimpleNamespace(rawValue="ABC123"),
                ),
            ),
        )
    )
    target = {"documentPatch": {"billOfLadingNumber": "ABC123"}}

    assert _MODULE._evidence_normalization_findings(annotation, target) == []


@pytest.mark.parametrize(
    ("target_path", "target_value", "raw_value"),
    [
        ("documentPatch.goodsItems[0].grossWeight.unit", "kilogram", "KGS"),
        ("documentPatch.parties.notifyParties[0].sameAs", "consignee", "SAME AS CONSIGNEE"),
        ("documentPatch.freight.paymentArrangement", "prepaid", "PREPAID"),
    ],
)
def test_evidence_normalization_gate_rejects_transformation_marked_verbatim(
    target_path: str, target_value: str, raw_value: str
) -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath=target_path,
                evidenceKind="verbatim",
                rawOcrEvidence=(SimpleNamespace(rawValue=raw_value),),
            ),
        )
    )
    target: dict[str, object]
    if target_path.endswith(".unit"):
        target = {
            "documentPatch": {"goodsItems": [{"grossWeight": {"unit": target_value}}]}
        }
    elif target_path.endswith(".sameAs"):
        target = {
            "documentPatch": {"parties": {"notifyParties": [{"sameAs": target_value}]}}
        }
    else:
        target = {"documentPatch": {"freight": {"paymentArrangement": target_value}}}

    assert _MODULE._evidence_normalization_findings(annotation, target) == [
        f"{target_path}: verbatim evidence does not contain the exact emitted string; "
        "use a non-verbatim evidence kind with a normalization rule"
    ]


def test_evidence_normalization_gate_leaves_numeric_verbatim_to_support_check() -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath="documentPatch.goodsItems[0].packages[0].quantity",
                evidenceKind="verbatim",
                rawOcrEvidence=(SimpleNamespace(rawValue="2484"),),
            ),
        )
    )
    target = {"documentPatch": {"goodsItems": [{"packages": [{"quantity": 2484}]}]}}

    assert _MODULE._evidence_normalization_findings(annotation, target) == []


def test_conditional_non_negotiable_gate_requires_named_consignee_evidence() -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath="documentPatch.negotiability",
                rawOcrEvidence=(
                    SimpleNamespace(rawValue="NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER"),
                ),
            ),
        )
    )
    target = {
        "documentPatch": {
            "negotiability": "non_negotiable",
            "parties": {"consignee": {"name": "NAMED IMPORTER LTD"}},
        }
    }

    assert _MODULE._negotiability_evidence_findings(
        annotation,
        target,
        "NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER",
    ) == ["conditional non-negotiable target must cite the emitted named consignee"]


def test_conditional_non_negotiable_gate_accepts_named_consignee_evidence() -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath="documentPatch.negotiability",
                rawOcrEvidence=(
                    SimpleNamespace(rawValue="NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER"),
                    SimpleNamespace(rawValue="NAMED IMPORTER LTD"),
                ),
            ),
        )
    )
    target = {
        "documentPatch": {
            "negotiability": "non_negotiable",
            "parties": {"consignee": {"name": "NAMED IMPORTER LTD"}},
        }
    }

    assert (
        _MODULE._negotiability_evidence_findings(
            annotation,
            target,
            "NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER",
        )
        == []
    )


def test_conditional_receivable_to_order_gate_accepts_named_consignee_evidence() -> None:
    conditional = (
        'Consignee (receivable if assigned "to order" or "to order of a named person"'
        ' or "to order of bearer")'
    )
    consignee = "DELEMAR ALUMINIUM PROFILESPRODUCTION S.A.E"
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath="documentPatch.negotiability",
                rawOcrEvidence=(
                    SimpleNamespace(rawValue=conditional),
                    SimpleNamespace(rawValue=consignee),
                ),
            ),
        )
    )
    target = {
        "documentPatch": {
            "negotiability": "non_negotiable",
            "parties": {"consignee": {"name": consignee}},
        }
    }

    assert _MODULE._target_leaf_is_supported(
        "documentPatch.negotiability",
        "non_negotiable",
        f"{conditional}\n{consignee}",
    )
    assert _MODULE._negotiability_evidence_findings(annotation, target, conditional) == []


def test_semantic_gate_distinguishes_chemical_acid_from_customs_acid_id() -> None:
    chemical = {
        "documentPatch": {"goodsItems": [{"description": "CORROSIVE LIQUID (CAPRYLIC ACID)"}]}
    }
    customs_identifier = {
        "documentPatch": {"goodsItems": [{"description": "ACID: 1004977722024020145"}]}
    }

    assert _MODULE._semantic_findings(chemical) == []
    assert any(
        "prohibited tax/boilerplate" in finding
        for finding in _MODULE._semantic_findings(customs_identifier)
    )


def test_address_redundancy_gate_does_not_match_city_inside_facility_name() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": (
                        "ROOM 1702, INTERNATIONAL TRADE CENTER OF YUYAO PLASTIC CITY, ZHEJIANG"
                    ),
                    "city": "YUYAO",
                    "country": "CHINA",
                }
            }
        }
    }

    assert _MODULE._semantic_findings(target) == []


def test_address_redundancy_gate_does_not_split_hyphenated_facility_name() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": (
                        "NO.16&18, CHANGJIANG ROAD, CHINA-SHANGHAI COOPERATION "
                        "(SCO) DEMONSTRATION ZONE 266300 JIAOZHOU"
                    ),
                    "city": "QINGDAO",
                    "country": "CHINA",
                }
            }
        }
    }

    assert _MODULE._semantic_findings(target) == []


def test_address_redundancy_gate_matches_city_before_state_and_postal_code() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": "37 EDISON AVE. WEST BABYLON NY 11704",
                    "city": "WEST BABYLON",
                    "country": "USA",
                }
            }
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "party address repeats separately emitted city 'WEST BABYLON'"
    ]


def test_address_redundancy_gate_keeps_city_name_used_as_street_name() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": "ATHENS ST 10446",
                    "city": "ATHENS",
                    "country": "GREECE",
                }
            }
        }
    }

    assert _MODULE._semantic_findings(target) == []


def test_address_redundancy_gate_allows_city_name_inside_distinct_locality() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "address": (
                            "BLOCK NO. 4, ABDEL RAHMAN BISAR ST., FIRST SETTLEMENT, NEW CAIRO"
                        ),
                        "city": "CAIRO",
                        "country": "EGYPT",
                    }
                ]
            }
        }
    }

    assert _MODULE._semantic_findings(target) == []


def test_address_redundancy_gate_matches_terminal_city_after_postal_code() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": "BLEICHERSTR. 28 DE-73066 UHINGEN",
                    "city": "UHINGEN",
                    "country": "GERMANY",
                }
            }
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "party address repeats separately emitted city 'UHINGEN'"
    ]


def test_semantic_gate_rejects_generic_freight_payment_place() -> None:
    target = {
        "documentPatch": {
            "freight": {
                "paymentArrangement": "collect",
                "paymentPlace": {"name": "DESTINATION"},
            }
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "freight payment place is a generic direction 'DESTINATION'"
    ]


def test_warning_gate_rejects_valid_container_claimed_as_invalid() -> None:
    annotation = SimpleNamespace(
        warnings=(
            SimpleNamespace(
                code="invalid_identifier_omitted",
                message="Printed GCXU2131234 fails ISO 6346 validation.",
            ),
        )
    )

    assert _MODULE._warning_findings(annotation) == [
        "invalid_identifier_omitted warning names valid ISO 6346 identifier 'GCXU2131234'"
    ]


@pytest.mark.parametrize(
    "raw_line",
    [
        "FAX NO. +2066 3228896",
        "TEL: +420 596 270 052-58 FAX: +420 596 270 051",
        "Fax : +20572292206",
    ],
)
def test_fax_warning_gate_requires_warning_for_standalone_value(raw_line: str) -> None:
    annotation = SimpleNamespace(warnings=())

    assert _MODULE._fax_warning_findings(
        annotation, f"--- PAGE 1 ---\n{raw_line}"
    ) == [
        "standalone FAX value on page 1 requires a schema_cannot_represent warning"
    ]


@pytest.mark.parametrize(
    "raw_line",
    [
        "TEL/FAX: +902164404050",
        "Tel / Fax :+201003300211",
        "TEL-FAX 086-25-85698168",
        "TELEPHONE – FAX: +20 3 4833755",  # noqa: RUF001 - intentional OCR fixture
        "FAX :- - -",
    ],
)
def test_fax_warning_gate_ignores_joint_or_blank_value(raw_line: str) -> None:
    annotation = SimpleNamespace(warnings=())

    assert _MODULE._fax_warning_findings(
        annotation, f"--- PAGE 1 ---\n{raw_line}"
    ) == []


def test_fax_warning_gate_accepts_page_bound_warning() -> None:
    annotation = SimpleNamespace(
        warnings=(
            SimpleNamespace(
                code="schema_cannot_represent",
                message="The explicit shipper fax value is omitted.",
                pageNumbers=(1,),
            ),
        )
    )

    assert _MODULE._fax_warning_findings(
        annotation, "--- PAGE 1 ---\nFAX: +201 000041924"
    ) == []


def test_phone_evidence_gate_distinguishes_tel_and_fax_on_same_line() -> None:
    path = "documentPatch.parties.deliveryAgent.contactDetails.phoneNumbers[0]"
    excerpt = "Tel.+20 3 4833755 Fax.+20 3 4836774"

    assert not _MODULE._phone_evidence_is_fax(path, "+20 3 4833755", excerpt)
    assert _MODULE._phone_evidence_is_fax(path, "+20 3 4836774", excerpt)


def test_phone_evidence_gate_allows_joint_tel_fax_value_as_phone() -> None:
    path = "documentPatch.parties.consignee.contactDetails.phoneNumbers[0]"
    excerpt = "Tel / Fax :+201003300211 TAX ID NO : 599640898"

    assert not _MODULE._phone_evidence_is_fax(path, "+201003300211", excerpt)


def test_semantic_gate_rejects_contact_value_with_joint_phone_fax_label() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {"contactDetails": {"phoneNumbers": ["PHONE & FAX 002 02 2575 2002"]}}
            }
        }
    }

    assert any(
        "communication value retains its field label" in finding
        for finding in _MODULE._semantic_findings(target)
    )


def test_semantic_gate_rejects_forwarding_reference_with_field_label() -> None:
    target = {"documentPatch": {"forwardingAndExportReferences": ["Svc Contract 299232432"]}}

    assert any(
        "reference value retains its field label" in finding
        for finding in _MODULE._semantic_findings(target)
    )


def test_semantic_gate_allows_invoice_prefix_that_is_part_of_identifier() -> None:
    target = {"documentPatch": {"forwardingAndExportReferences": ["INV/2023/00186"]}}

    assert _MODULE._semantic_findings(target) == []


def test_semantic_gate_rejects_contact_footnote_and_mark_field_label() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [{"contactDetails": {"emailAddresses": ["OPS@EXAMPLE.COM***"]}}]
            },
            "goodsItems": [{"marksAndNumbers": ["C/NO. 30 /3,5,8"]}],
        }
    }

    findings = _MODULE._semantic_findings(target)

    assert any("communication value retains a footnote marker" in row for row in findings)
    assert any("mark value retains its field label" in row for row in findings)


@pytest.mark.parametrize(
    ("target_path", "target", "raw_value"),
    [
        ("documentPatch.issueDate", "2023-09-30", "SEP/30/2023"),
        ("documentPatch.issueDate", "2024-03-29", "MARCH 29TH, 2024"),
        ("documentPatch.issueDate", "2024-06-28", "JUN.28,2024"),
        ("documentPatch.goodsItems[0].grossWeight.value", 60960, "60,960KG"),
        ("documentPatch.goodsItems[0].grossWeight.value", 7740, "7.740 KG"),
        ("documentPatch.goodsItems[0].grossWeight.value", 5370.153, "5370,153"),
        ("documentPatch.goodsItems[0].packages[0].quantity", 6, "SIX"),
        (
            "documentPatch.goodsItems[0].packages[0].quantity",
            52,
            "TOTAL: FIFTY-TWO IBC ONLY",
        ),
        (
            "documentPatch.goodsItems[0].packages[0].quantity",
            5272,
            "FIVE THOUSAND TWO HUNDRED AND SEVENTY TWO ONLY",
        ),
        ("documentPatch.goodsItems[0].grossWeight.unit", "kilogram", "KGS"),
        ("documentPatch.goodsItems[0].volume.unit", "cubic_metre", "cu. m."),
        ("documentPatch.negotiability", "non_negotiable", "NON-NEGOTIABLE"),
        (
            "documentPatch.negotiability",
            "non_negotiable",
            "Bill of Lading Type EXPRESS / EXPRESS BILL OF LADING",
        ),
        ("documentPatch.negotiability", "non_negotiable", "B/L EXPRESS"),
        ("documentPatch.negotiability", "non_negotiable", "EXPRESS BL"),
        (
            "documentPatch.negotiability",
            "non_negotiable",
            "In accepting this Waybill the Merchant expressly accepts its terms",
        ),
        (
            "documentPatch.negotiability",
            "non_negotiable",
            ("Consignee (negotiable only if consigned to order) AL MAJIC FOR IMPORT EXPORT"),
        ),
        ("documentPatch.negotiability", "negotiable", "TO THE ORDER"),
        ("documentPatch.parties.notifyParties[0].sameAs", "consignee", "SAME AS CNEE"),
        ("documentPatch.parties.notifyParties[0].sameAs", "shipper", "SAME AS SHPR"),
        (
            "documentPatch.freight.paymentArrangement",
            "collect",
            "FREIGHT PAYABLE AT DESTINATION",
        ),
        ("documentPatch.parties.shipper.name", "Solucoes", "Soluções"),
        ("documentPatch.containers[0].containerNumber", "YMLU3540304", "YMLU 3540304"),
    ],
)
def test_target_leaf_support_accepts_authorized_normalizations(
    target_path: str, target: object, raw_value: str
) -> None:
    assert _MODULE._target_leaf_is_supported(target_path, target, raw_value)


@pytest.mark.parametrize(
    ("target_path", "target", "raw_value"),
    [
        ("documentPatch.billOfLadingNumber", "VASHYDSOK001653", "BL Barcode"),
        (
            "documentPatch.parties.deliveryAgent.name",
            "EGYPTIAN INTERNATIONAL SHIPPING AGENCIES AND SERVICES",
            "Delivery",
        ),
        ("documentPatch.goodsItems[0].grossWeight.value", 13884.09, "weight"),
        ("documentPatch.goodsItems[0].grossWeight.unit", "kilogram", "weight"),
        (
            "documentPatch.negotiability",
            "non_negotiable",
            "EXPRESS RELEASE REQUEST",
        ),
        ("documentPatch.parties.notifyParties[0].sameAs", "consignee", "SAME AS SHPR"),
    ],
)
def test_target_leaf_support_rejects_unrelated_raw_ocr(
    target_path: str, target: object, raw_value: str
) -> None:
    assert not _MODULE._target_leaf_is_supported(target_path, target, raw_value)


def test_duplicate_comparison_surfaces_shared_path_contradictions() -> None:
    groups = [{"groupId": "duplicate-001", "documentIds": ["doc_a", "doc_b"]}]
    targets = {
        "doc_a": {"documentPatch": {"billOfLadingNumber": "A"}},
        "doc_b": {"documentPatch": {"billOfLadingNumber": "B"}},
    }

    comparisons = _MODULE._duplicate_comparisons(groups, targets)

    assert comparisons[0]["pairwise"][0]["contradictions"] == [
        {"path": "documentPatch.billOfLadingNumber", "first": "A", "second": "B"}
    ]


def test_duplicate_consistency_separates_same_ocr_contradictions_from_variants() -> None:
    groups = [
        {"groupId": "duplicate-001", "documentIds": ["doc_a", "doc_b", "doc_c"]}
    ]
    targets = {
        "doc_a": {"documentPatch": {"billOfLadingNumber": "A"}},
        "doc_b": {"documentPatch": {"billOfLadingNumber": "B"}},
        "doc_c": {"documentPatch": {"billOfLadingNumber": "C"}},
    }
    raw_hashes = {"doc_a": "1" * 64, "doc_b": "1" * 64, "doc_c": "2" * 64}

    report = _MODULE._duplicate_consistency_report(
        groups, targets, raw_hashes, {"doc_a"}
    )

    assert report["status"] == "failed"
    assert report["sameRawOcrContradictionCount"] == 1
    assert report["rawOcrVariantPairCount"] == 2
    assert report["rawOcrVariantDifferenceCount"] == 2
    assert report["trainingContradictionCount"] == 0
    classifications = {
        tuple(pair["documentIds"]): pair["classification"]
        for pair in report["acceptedGroups"][0]["pairwise"]
    }
    assert classifications == {
        ("doc_a", "doc_b"): "same_raw_ocr_contradiction",
        ("doc_a", "doc_c"): "raw_ocr_variant_difference",
        ("doc_b", "doc_c"): "raw_ocr_variant_difference",
    }


def test_duplicate_consistency_blocks_different_retained_training_targets() -> None:
    groups = [{"groupId": "duplicate-001", "documentIds": ["doc_a", "doc_b"]}]
    targets = {
        "doc_a": {"documentPatch": {"billOfLadingNumber": "A"}},
        "doc_b": {"documentPatch": {"billOfLadingNumber": "B"}},
    }

    report = _MODULE._duplicate_consistency_report(
        groups,
        targets,
        {"doc_a": "1" * 64, "doc_b": "2" * 64},
        {"doc_a", "doc_b"},
    )

    assert report["sameRawOcrContradictionCount"] == 0
    assert report["rawOcrVariantDifferenceCount"] == 1
    assert report["trainingContradictionCount"] == 1
    assert report["status"] == "failed"


def test_manifest_digest_recovers_after_two_artifact_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    monkeypatch.setattr(_MODULE, "RUN_ID", "test-run")
    (tmp_path / "payload.json").write_text('{"ok":true}\n', encoding="utf-8")

    _MODULE._publish_manifest()
    (tmp_path / "manifest.json.sha256").unlink()
    _MODULE._recover_manifest_digest()

    digest_line = (tmp_path / "manifest.json.sha256").read_text(encoding="utf-8")
    assert digest_line.endswith("  manifest.json\n")


def test_worker_start_rejects_unknown_document_without_writing_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)

    with pytest.raises(RuntimeError, match="work item does not exist"):
        _MODULE.record_worker_start("doc_missing", 1, "/root/worker")

    assert not (tmp_path / "worker-logs").exists()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_omit_explicit_nulls_preserves_sparse_values_and_list_order() -> None:
    value = {
        "present": "value",
        "absent": None,
        "nested": {"keep": 0, "drop": None},
        "items": [{"keep": False, "drop": None}, None, "tail"],
    }

    assert _MODULE._omit_explicit_nulls(value) == {
        "present": "value",
        "nested": {"keep": 0},
        "items": [{"keep": False}, "tail"],
    }


def test_check_attempt_records_exact_candidate_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_digest"
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(attempt_path, {"candidate": "exact bytes"})
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {
            "status": "completed",
            "candidateSha256": _MODULE._file_digest(attempt_path),
        },
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_attempt",
        lambda _document_id, _path: ("excluded", {"rawOcrEvidence": []}, None, {}),
    )

    _MODULE.check_attempt(document_id, 1)

    check = json.loads(
        (tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json").read_text(
            encoding="utf-8"
        )
    )
    assert check["attemptSha256"] == _MODULE._file_digest(attempt_path)


def test_check_attempt_does_not_seal_lifecycle_precondition_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_still_running"
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(attempt_path, {"candidate": "complete but not yet sealed"})
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {"status": "running", "candidateSha256": None},
    )

    with pytest.raises(RuntimeError, match="only a completed worker attempt"):
        _MODULE.check_attempt(document_id, 1)

    assert not (
        tmp_path
        / "validation/attempt-checks"
        / f"{document_id}.attempt-1.json"
    ).exists()


def test_worker_finish_seals_completed_candidate_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_sealed"
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(attempt_path, {"candidate": "published"})
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {"status": "running", "completedAt": None},
    )

    _MODULE.record_worker_finish(document_id, 1, "completed")

    worker = json.loads(
        (tmp_path / "worker-logs" / f"{document_id}.attempt-1.json").read_text(encoding="utf-8")
    )
    assert worker["candidatePath"] == (f"candidate-attempts/{document_id}.attempt-1.json")
    assert worker["candidateSha256"] == _MODULE._file_digest(attempt_path)
    assert worker["status"] == "completed"
    assert worker["completedAt"] is not None


def test_independent_reviewer_lifecycle_binds_exact_candidate_and_work_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    monkeypatch.setattr(_MODULE, "REVIEWER_REASONING_EFFORT", "max")
    document_id = f"doc_{'a' * 64}"
    work_path = tmp_path / "work-items" / f"{document_id}.json"
    candidate_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(
        work_path,
        {
            "source": {"documentId": document_id},
            "joinedRawText": "--- PAGE 1 ---\nONE RAW VALUE",
        },
    )
    _write_json(candidate_path, {"candidate": "sealed"})
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json",
        {
            "status": "passed",
            "attemptSha256": _MODULE._file_digest(candidate_path),
        },
    )

    _MODULE.record_reviewer_start(document_id, 1, 1, "/root/reviewer")
    review_path = tmp_path / "independent-reviews" / f"{document_id}.attempt-1.review-1.json"
    _write_json(
        review_path,
        {
            "reviewSchemaVersion": "1.0.0",
            "taskType": "bill_of_lading_kie_semantic_review",
            "documentId": document_id,
            "candidateAttempt": 1,
            "reviewNumber": 1,
            "workItemSha256": _MODULE._file_digest(work_path),
            "candidateSha256": _MODULE._file_digest(candidate_path),
            "reviewerModel": "gpt-5.6-luna",
            "reviewerReasoningEffort": "max",
            "result": "pass",
            "checks": {
                "rawOcrTruthBoundary": "pass",
                "evidenceIntegrity": "pass",
                "semanticCompleteness": "pass",
                "semanticCorrectness": "pass",
                "contaminationAndRedundancy": "pass",
                "documentUnitAndRelationships": "pass",
            },
            "findings": [],
            "summary": "Candidate is complete and grounded in the assigned raw OCR.",
        },
    )

    _MODULE.record_reviewer_finish(document_id, 1, 1, "completed")
    _MODULE.check_independent_review(document_id, 1, 1)

    check = json.loads(
        (
            tmp_path
            / "validation/independent-review-checks"
            / f"{document_id}.attempt-1.review-1.json"
        ).read_text(encoding="utf-8")
    )
    assert check["status"] == "passed"
    assert check["semanticResult"] == "pass"
    assert check["candidateSha256"] == _MODULE._file_digest(candidate_path)


def test_required_independent_review_cannot_be_bypassed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    monkeypatch.setattr(_MODULE, "REQUIRE_INDEPENDENT_REVIEW", True)
    document_id = "doc_required_review"
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(attempt_path, {"candidate": "checked"})
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {
            "status": "completed",
            "taskName": "/root/worker",
            "model": "gpt-5.6-luna",
            "reasoningEffort": "max",
            "startedAt": "2026-08-17T00:00:00Z",
            "completedAt": "2026-08-17T00:01:00Z",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json",
        {
            "status": "passed",
            "kind": "approved",
            "attemptSha256": _MODULE._file_digest(attempt_path),
        },
    )

    with pytest.raises(RuntimeError, match="requires an independent review number"):
        _MODULE.record_review(document_id, 1, "accepted", "include", ["overseer accepts candidate"])

    assert not (tmp_path / "validation/decisions" / f"{document_id}.json").exists()


def test_check_attempt_seals_failure_and_rejects_mutated_completed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_mutated_before_check"
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(attempt_path, {"candidate": "sealed"})
    sealed_digest = _MODULE._file_digest(attempt_path)
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {"status": "completed", "candidateSha256": sealed_digest},
    )
    _write_json(attempt_path, {"candidate": "mutated"})

    with pytest.raises(RuntimeError, match="changed after worker completion"):
        _MODULE.check_attempt(document_id, 1)

    check_path = tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json"
    failed = json.loads(check_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["attemptSha256"] == _MODULE._file_digest(attempt_path)

    with pytest.raises(RuntimeError, match="attempt check is immutable"):
        _MODULE.check_attempt(document_id, 1)


def test_review_rejects_candidate_mutated_after_passing_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_mutated"
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(attempt_path, {"candidate": "checked"})
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {
            "status": "completed",
            "taskName": "/root/worker",
            "model": "gpt-5.6-luna",
            "reasoningEffort": "high",
            "startedAt": "2026-08-17T00:00:00Z",
            "completedAt": "2026-08-17T00:01:00Z",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json",
        {
            "status": "passed",
            "kind": "approved",
            "attemptSha256": _MODULE._file_digest(attempt_path),
        },
    )
    _write_json(attempt_path, {"candidate": "changed after check"})

    with pytest.raises(RuntimeError, match="changed after automated validation"):
        _MODULE.record_review(document_id, 1, "accepted", "include", ["looks correct"])

    assert not (tmp_path / "validation/attempt-reviews" / f"{document_id}.attempt-1.json").exists()


def test_review_rejects_implicit_decision_overwrite_before_writing_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_a"
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-2.json",
        {
            "status": "completed",
            "taskName": "/root/worker",
            "model": "gpt-5.6-luna",
            "reasoningEffort": "high",
            "startedAt": "2026-08-17T00:00:00Z",
            "completedAt": "2026-08-17T00:01:00Z",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-2.json",
        {"status": "passed", "kind": "annotation"},
    )
    _write_json(
        tmp_path / "validation/decisions" / f"{document_id}.json",
        {"documentId": document_id, "attempt": 1, "decision": "approve"},
    )

    with pytest.raises(RuntimeError, match="already has an accepted decision"):
        _MODULE.record_review(document_id, 2, "accepted", "include", ["corrected candidate"])

    assert not (tmp_path / "validation/attempt-reviews" / f"{document_id}.attempt-2.json").exists()


def test_review_can_reject_failed_check_without_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_failed"
    _write_json(
        tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json",
        {"candidate": "semantically contaminated"},
    )
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {
            "status": "completed",
            "taskName": "/root/worker",
            "model": "gpt-5.6-luna",
            "reasoningEffort": "high",
            "startedAt": "2026-08-17T00:00:00Z",
            "completedAt": "2026-08-17T00:01:00Z",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json",
        {"status": "failed", "error": "semantic contamination"},
    )

    _MODULE.record_review(document_id, 1, "rejected", None, ["remove duplicated country"])

    review = json.loads(
        (tmp_path / "validation/attempt-reviews" / f"{document_id}.attempt-1.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["result"] == "rejected"
    assert not (tmp_path / "validation/decisions" / f"{document_id}.json").exists()


def test_review_explicitly_supersedes_and_preserves_decision_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_a"
    worker = {
        "status": "completed",
        "taskName": "/root/worker",
        "model": "gpt-5.6-luna",
        "reasoningEffort": "high",
        "startedAt": "2026-08-17T00:00:00Z",
        "completedAt": "2026-08-17T00:01:00Z",
        "usage": "unavailable_from_collaboration_runtime",
    }
    _write_json(tmp_path / "worker-logs" / f"{document_id}.attempt-2.json", worker)
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-2.json"
    _write_json(attempt_path, {"candidate": "corrected"})
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-2.json",
        {
            "status": "passed",
            "kind": "approved",
            "attemptSha256": _MODULE._file_digest(attempt_path),
        },
    )
    previous = {
        "documentId": document_id,
        "attempt": 1,
        "attemptPath": f"candidate-attempts/{document_id}.attempt-1.json",
        "decision": "approve",
        "trainingDisposition": "include",
        "reviewPath": f"validation/attempt-reviews/{document_id}.attempt-1.json",
        "decidedAt": "2026-08-17T00:02:00Z",
    }
    decision_path = tmp_path / "validation/decisions" / f"{document_id}.json"
    _write_json(decision_path, previous)

    _MODULE.record_review(
        document_id,
        2,
        "accepted",
        "include",
        ["corrected candidate"],
        supersede_accepted=True,
    )

    history_path = tmp_path / "validation/decision-history" / f"{document_id}.attempt-1.json"
    assert json.loads(history_path.read_text(encoding="utf-8")) == previous
    current = json.loads(decision_path.read_text(encoding="utf-8"))
    assert current["attempt"] == 2
    assert current["supersedesDecisionPath"] == (
        f"validation/decision-history/{document_id}.attempt-1.json"
    )


def test_training_redisposition_archives_decision_without_relabeling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    monkeypatch.setattr(_MODULE, "REQUIRE_INDEPENDENT_REVIEW", False)
    document_id = "doc_duplicate"
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(attempt_path, {"candidate": "unchanged"})
    candidate_digest = _MODULE._file_digest(attempt_path)
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json",
        {
            "status": "passed",
            "kind": "approved",
            "attemptSha256": candidate_digest,
        },
    )
    _write_json(
        tmp_path / "validation/duplicate-candidates.json",
        {
            "groups": [
                {
                    "groupId": "duplicate-001",
                    "documentIds": [document_id, "doc_representative"],
                }
            ]
        },
    )
    decision = {
        "documentId": document_id,
        "attempt": 1,
        "decision": "approve",
        "trainingDisposition": "include",
        "decidedAt": "2026-08-20T00:00:00Z",
    }
    decision_path = tmp_path / "validation/decisions" / f"{document_id}.json"
    _write_json(decision_path, decision)
    previous_digest = _MODULE._file_digest(decision_path)
    findings = ["retain the stronger duplicate as the sole training representative"]

    _MODULE.record_training_disposition(document_id, "duplicate_suppressed", findings)

    history_relative = (
        f"validation/decision-history/{document_id}.decision-{previous_digest}.json"
    )
    history = json.loads((tmp_path / history_relative).read_text(encoding="utf-8"))
    current = json.loads(decision_path.read_text(encoding="utf-8"))
    assert history == decision
    assert current["attempt"] == 1
    assert current["trainingDisposition"] == "duplicate_suppressed"
    assert current["trainingDispositionChange"] | {"changedAt": "ignored"} == {
        "from": "include",
        "to": "duplicate_suppressed",
        "duplicateGroupId": "duplicate-001",
        "findings": findings,
        "changedAt": "ignored",
    }
    assert current["supersedesDecisionPath"] == history_relative
    assert _MODULE._file_digest(attempt_path) == candidate_digest

    # An interrupted caller can safely replay the exact disposition command.
    _MODULE.record_training_disposition(document_id, "duplicate_suppressed", findings)
    assert json.loads(decision_path.read_text(encoding="utf-8")) == current


def test_training_redisposition_rejects_non_duplicate_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    monkeypatch.setattr(_MODULE, "REQUIRE_INDEPENDENT_REVIEW", False)
    document_id = "doc_unique"
    attempt_path = tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    _write_json(attempt_path, {"candidate": "unchanged"})
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json",
        {
            "status": "passed",
            "kind": "approved",
            "attemptSha256": _MODULE._file_digest(attempt_path),
        },
    )
    _write_json(tmp_path / "validation/duplicate-candidates.json", {"groups": []})
    _write_json(
        tmp_path / "validation/decisions" / f"{document_id}.json",
        {
            "documentId": document_id,
            "attempt": 1,
            "decision": "approve",
            "trainingDisposition": "include",
        },
    )

    with pytest.raises(RuntimeError, match="exactly one frozen duplicate group"):
        _MODULE.record_training_disposition(
            document_id,
            "duplicate_suppressed",
            ["not a duplicate"],
        )


def test_contract_revision_preserves_hash_history_and_seals_current_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(_MODULE, "DEST", tmp_path / "run")
    monkeypatch.setattr(_MODULE, "RUN_ID", "test-run")
    monkeypatch.setattr(_MODULE, "_CONTRACT_PATHS", ("contract.md",))
    contract_path = tmp_path / "contract.md"
    contract_path.write_text("original\n", encoding="utf-8")
    original = _MODULE._current_contract_hashes()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: test\n", encoding="utf-8")
    monkeypatch.setattr(_MODULE, "RUN_CONFIG_PATH", config_path)
    _write_json(
        tmp_path / "run/run-metadata.json",
        {
            "runId": "test-run",
            "status": "prepared",
            "runConfiguration": {
                "path": "config.yaml",
                "sha256": _MODULE._file_digest(config_path),
            },
            "frozenContract": original,
        },
    )
    contract_path.write_text("revised\n", encoding="utf-8")
    findings = ["record an intentional prompt-policy revision"]

    _MODULE.record_contract_revision(findings)

    metadata = json.loads(
        (tmp_path / "run/run-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["frozenContract"] == _MODULE._current_contract_hashes()
    assert metadata["contractRevisionCount"] == 1
    revision_path = tmp_path / "run" / metadata["contractRevisionPaths"][0]
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    assert revision["previousContract"] == original
    assert revision["currentContract"] == metadata["frozenContract"]
    assert revision["changedFiles"] == ["contract.md"]
    assert revision["findings"] == findings
    _MODULE._verify_frozen_contract(metadata)

    # Exact replay after a crash is idempotent and does not create another revision.
    _MODULE.record_contract_revision(findings)
    replayed = json.loads(
        (tmp_path / "run/run-metadata.json").read_text(encoding="utf-8")
    )
    assert replayed == metadata
    assert list((tmp_path / "run/validation/contract-revisions").glob("*.json")) == [
        revision_path
    ]


def test_frozen_contract_verifier_rejects_unrecorded_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(_MODULE, "_CONTRACT_PATHS", ("contract.md",))
    contract_path = tmp_path / "contract.md"
    contract_path.write_text("sealed\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: test\n", encoding="utf-8")
    monkeypatch.setattr(_MODULE, "RUN_CONFIG_PATH", config_path)
    metadata = {
        "runConfiguration": {
            "path": "config.yaml",
            "sha256": _MODULE._file_digest(config_path),
        },
        "frozenContract": _MODULE._current_contract_hashes(),
    }
    contract_path.write_text("drifted\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="labeling contract changed"):
        _MODULE._verify_frozen_contract(metadata)


def test_size_comparison_summary_reports_exact_totals_and_delta() -> None:
    rows = [
        {
            "v1Leaves": 10,
            "v2Leaves": 7,
            "v1CanonicalBytes": 100,
            "v2CanonicalBytes": 80,
            "v1Tokens": 20,
            "v2Tokens": 15,
        },
        {
            "v1Leaves": 6,
            "v2Leaves": 5,
            "v1CanonicalBytes": 60,
            "v2CanonicalBytes": 50,
            "v1Tokens": 12,
            "v2Tokens": 10,
        },
    ]

    assert _MODULE._size_comparison_summary(rows) == {
        "documents": 2,
        "leaves": {
            "v1Total": 16,
            "v2Total": 12,
            "absoluteDelta": -4,
            "percentDelta": -25.0,
        },
        "canonicalBytes": {
            "v1Total": 160,
            "v2Total": 130,
            "absoluteDelta": -30,
            "percentDelta": -18.75,
        },
        "tokens": {
            "v1Total": 32,
            "v2Total": 25,
            "absoluteDelta": -7,
            "percentDelta": -21.875,
        },
    }


def test_size_comparison_summary_requires_overlap() -> None:
    with pytest.raises(RuntimeError, match="overlapping validated documents"):
        _MODULE._size_comparison_summary([])


def test_eda_counts_leaf_paths_instead_of_treating_values_as_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)

    def annotation(target: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            label=SimpleNamespace(canonical_target=lambda: {"documentPatch": target}),
            warnings=(SimpleNamespace(code="schema_cannot_represent"),),
            source=SimpleNamespace(documentPageCount=1),
            documentType="bill_of_lading",
        )

    tokenizer = SimpleNamespace(encode=lambda value, add_special_tokens: value.split())
    result = _MODULE._eda(
        {
            "doc_a": annotation({"billOfLadingNumber": "A", "issueDate": "2024-01-01"}),
            "doc_b": annotation({"billOfLadingNumber": "B"}),
        },
        {},
        tokenizer,
    )

    field_rows = (
        (tmp_path / "eda/tables/field_presence.csv").read_text(encoding="utf-8").splitlines()
    )
    assert "documentPatch.billOfLadingNumber,2" in field_rows
    assert "documentPatch.issueDate,1" in field_rows
    assert result["validatedDocuments"] == 2
