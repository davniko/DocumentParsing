from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis import dangerous_goods_registry as dangerous_goods_registry_module
from document_ocr.synthesis.dangerous_goods_registry import (
    DangerousGoodsEcicsLink,
    DangerousGoodsHmtRecord,
    DangerousGoodsRegistryError,
    LoadedDangerousGoodsRegistry,
    compile_dangerous_goods_registry,
)
from document_ocr.synthesis.generators import DeterministicStream


def _hmt(
    record_id_suffix: str,
    *,
    un: str,
    exact_class: str,
    category: str,
    subsidiaries: tuple[str, ...] = (),
    semantic_subsidiaries: tuple[str, ...] = (),
) -> DangerousGoodsHmtRecord:
    return DangerousGoodsHmtRecord.model_validate(
        {
            "record_id": f"hmt_{record_id_suffix * 64}",
            "source_row": 2,
            "un_number": un,
            "proper_shipping_name": f"TEST MATERIAL {un}",
            "proper_shipping_name_markup": f"TEST MATERIAL {un}",
            "optional_qualifiers": (),
            "exact_hazard_class": exact_class,
            "hazard_category": category,
            "exact_label_codes": (exact_class, *subsidiaries),
            "exact_subsidiary_hazards": subsidiaries,
            "subsidiary_hazard_categories": semantic_subsidiaries,
            "packing_group_code": "II",
            "packing_group_category": "MEDIUM_DANGER",
            "symbols": (),
            "technical_name_required": False,
            "nos_entry": False,
            "vessel_stowage_location": "A",
            "vessel_stowage_other": (),
            "maritime_eligible": True,
            "maritime_disposition": "eligible",
            "source_fields": {"UN ID Number": f"UN{un}"},
        },
        strict=True,
    )


def _registry() -> LoadedDangerousGoodsRegistry:
    corrosive = _hmt(
        "a",
        un="1789",
        exact_class="8",
        category="CORROSIVE_SUBSTANCES",
    )
    flammable = _hmt(
        "b",
        un="1993",
        exact_class="3",
        category="FLAMMABLE_LIQUIDS",
        subsidiaries=("8",),
        semantic_subsidiaries=("CORROSIVE_SUBSTANCES",),
    )
    link = DangerousGoodsEcicsLink.model_validate(
        {
            "cus_number": "0000001-0",
            "un_number": "1993",
            "cn_code": "29011000",
            "hs6": "290110",
            "cas_numbers": ("1-11-1",),
            "ec_number": "200-000-0",
            "nomenclature": "IUPAC",
            "name": "test chemical",
            "iupac_description": "test chemical",
            "first_nomenclature": None,
            "first_nomenclature_description": None,
            "synonyms": (),
            "hmt_record_id": flammable.record_id,
            "chemical_identity_match": "normalized_exact_official_name_v1",
            "disposition": "eligible_unique_maritime_hmt",
        },
        strict=True,
    )
    return LoadedDangerousGoodsRegistry(
        hmt_records=(corrosive, flammable),
        ecics_links=(link,),
        hmt_sha256="1" * 64,
        ecics_sha256="2" * 64,
    )


def test_general_branch_emits_one_atomic_tuple_and_no_hs_or_flashpoint() -> None:
    sampled = _registry().sample(
        stream=DeterministicStream(seed=1, namespace="test", identity="one"),
        method="general_regulatory_tuple",
        category_weights={"CORROSIVE_SUBSTANCES": 1},
        maximum_subsidiary_hazards=1,
    )
    assert sampled.hmt_record.un_number == "1789"
    assert sampled.target.model_dump(mode="json", exclude_none=True) == {
        "unNumber": "1789",
        "hazardCategory": "CORROSIVE_SUBSTANCES",
        "packingGroupCategory": "MEDIUM_DANGER",
    }
    assert sampled.hs_codes == ()
    assert sampled.target.flashPoint is None


def test_exact_chemical_branch_keeps_hmt_and_ecics_hs_together() -> None:
    sampled = _registry().sample(
        stream=DeterministicStream(seed=1, namespace="test", identity="two"),
        method="hs_linked_exact_chemical",
        category_weights={"FLAMMABLE_LIQUIDS": 1},
        maximum_subsidiary_hazards=1,
    )
    assert sampled.target.unNumber == sampled.ecics_link.un_number == "1993"
    assert sampled.target.hazardCategory == "FLAMMABLE_LIQUIDS"
    assert sampled.target.subsidiaryHazardCategories == ("CORROSIVE_SUBSTANCES",)
    assert sampled.hs_codes == ("290110",)


def test_sampler_rejects_unavailable_requested_category() -> None:
    with pytest.raises(DangerousGoodsRegistryError, match="unavailable categories"):
        _registry().sample(
            stream=DeterministicStream(seed=1, namespace="test", identity="three"),
            method="hs_linked_exact_chemical",
            category_weights={"RADIOACTIVE_MATERIAL": 1},
            maximum_subsidiary_hazards=1,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("xlrd") is None or importlib.util.find_spec("openpyxl") is None,
    reason="real registry compilation runs only in the pinned synthesis environment",
)
def test_real_registry_compiles_to_expected_audited_counts(tmp_path: Path) -> None:
    source_root = Path("data/registries/dangerous-goods")
    manifest = source_root / "source-manifest-20260831.json"
    if not manifest.is_file():
        pytest.skip("pinned DG source snapshot is not present")
    result = compile_dangerous_goods_registry(
        source_root=source_root,
        source_manifest_path=manifest,
        expected_manifest_sha256=(
            "839c08a3502e473d3136e3c86680510249cc8c2c5a7ff45b188d2255118402f9"
        ),
        output_parent=tmp_path,
        run_name="real-dg-registry-test",
    )
    assert result.receipt.audit.hmt_records == 2934
    assert result.receipt.audit.hmt_maritime_records == 2909
    assert result.receipt.audit.ecics_exact_hs_links == 470
    assert result.receipt.audit.ecics_exact_hs_distinct_un_numbers == 468
    assert result.receipt.audit.ecics_exact_hs_distinct_hs6 == 152
    assert result.receipt.audit.ecics_dispositions == {
        "ambiguous_maritime_hmt_tuple": 1593,
        "chemical_identity_name_mismatch": 1362,
        "cn_shorter_than_hs6": 12,
        "eligible_unique_maritime_hmt": 470,
        "no_hmt_record": 36,
        "no_maritime_hmt_record": 7,
    }
    assert result.receipt.implementation_sha256 == sha256_file(
        Path(dangerous_goods_registry_module.__file__)
    )
