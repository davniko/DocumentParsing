from __future__ import annotations

import io
import math

import pyarrow.parquet as pq
import pytest
from PIL import Image

from document_ocr.training.dataset_eda import (
    DatasetEdaError,
    _csv_bytes,
    _js_divergence,
    _page_texts,
    _parquet_bytes,
    _percentile,
    _template_clusters,
    _vessel_present,
    carrier_family,
    country_group,
)
from document_ocr.training.eda_plots import heatmap, horizontal_bar


def test_carrier_and_country_grouping_is_conservative() -> None:
    assert carrier_family("Maersk A/S") == ("MAERSK", True)
    assert carrier_family("Ocean Network Express Pte. Ltd.") == ("ONE", True)
    assert carrier_family("Unlisted Example Shipping") == (
        "OTHER::UNLISTED EXAMPLE SHIPPING",
        False,
    )
    assert carrier_family(None) == ("<MISSING>", False)
    assert country_group("U.S.A.") == "UNITED STATES"
    assert country_group("Republic of Egypt") == "EGYPT"
    assert country_group("São Tomé") == "SAO TOME"
    assert country_group(None) == "<MISSING>"


def test_page_texts_requires_contiguous_order() -> None:
    assert _page_texts("--- PAGE 1 ---\nfirst\n--- PAGE 2 ---\nsecond") == (
        "first",
        "second",
    )
    with pytest.raises(DatasetEdaError, match="non-contiguous"):
        _page_texts("--- PAGE 1 ---\nfirst\n--- PAGE 3 ---\nthird")


def test_vessel_presence_uses_the_schema_field_names() -> None:
    assert _vessel_present({"vesselName": "EXAMPLE"})
    assert _vessel_present({"vesselImoNumber": "IMO1234567"})
    assert not _vessel_present({"voyageNumber": "V001"})


def _feature(document_id: str, vector: tuple[float, ...]) -> dict[str, object]:
    return {
        "document_id": document_id,
        "carrier_family": "TEST CARRIER",
        "document_type": "bill_of_lading",
        "first_page_anchor_sequence": ["shipper", "consignee", "packages"],
        "_visual_vector": vector,
        "visual_layout_hash": document_id,
        "source_filename": f"{document_id}.pdf",
        "source_corpus": "current_main680",
        "page_count": 1,
    }


def test_template_proxy_is_deterministic_and_separates_layouts() -> None:
    near = 0.99 / math.sqrt(0.99**2 + 0.01**2)
    small = 0.01 / math.sqrt(0.99**2 + 0.01**2)
    rows = [
        _feature("doc_a", (1.0, 0.0)),
        _feature("doc_b", (near, small)),
        _feature("doc_c", (0.0, 1.0)),
    ]
    clusters = _template_clusters(
        rows,
        visual_minimum=0.90,
        ocr_minimum=0.70,
        combined_minimum=0.88,
        visual_weight=0.60,
        annotate=True,
    )
    assert sorted(row["document_count"] for row in clusters) == [1, 2]
    assert rows[0]["template_proxy_id"] == rows[1]["template_proxy_id"]
    assert rows[2]["template_proxy_id"] != rows[0]["template_proxy_id"]
    repeated = _template_clusters(
        rows,
        visual_minimum=0.90,
        ocr_minimum=0.70,
        combined_minimum=0.88,
        visual_weight=0.60,
        annotate=False,
    )
    assert clusters == repeated


def test_statistical_primitives_have_expected_values() -> None:
    assert _percentile((0.0, 10.0), 0.25) == 2.5
    assert _percentile((1.0, 2.0, 3.0), 0.5) == 2.0
    assert _js_divergence(
        __import__("collections").Counter({"a": 10}),
        __import__("collections").Counter({"b": 10}),
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "payload",
    [
        horizontal_bar(
            title="Test",
            subtitle="Subtitle",
            items=(("a", 3.0), ("b", 1.0)),
        ),
        heatmap(
            title="Correlation",
            subtitle="Centered",
            x_labels=("a", "b"),
            y_labels=("a", "b"),
            matrix=((1.0, -0.5), (-0.5, 1.0)),
            value_format="float",
            center_zero=True,
        ),
    ],
)
def test_plot_is_a_deterministic_png(payload: bytes) -> None:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        assert image.format == "PNG"
        assert image.size == (1600, 1000)


def test_tabular_serializers_preserve_nested_values() -> None:
    rows = [{"document_id": "doc_a", "count": 2, "values": ["x", "y"]}]
    csv_payload = _csv_bytes(rows)
    assert b'"[""x"",""y""]"' in csv_payload
    parquet_payload = _parquet_bytes(rows)
    table = pq.read_table(io.BytesIO(parquet_payload))
    assert table.to_pylist() == rows
