"""Party-shaped cargo marks must share their generated party facts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import descendant
from document_ocr.synthesis.template_compiler.models import CertifiedSemanticTemplate
from document_ocr.synthesis.template_compiler.party_mark_contract import (
    validate_party_mark_preflight,
)


def _fixture(
    *,
    source_name: str = "MED CARE EGYPT",
    source_city: str = "Cairo",
    source_country: str = "Egypt",
    marks: tuple[str, ...] = ("MED CARE EGYPT", "CAIRO, EGYPT"),
    description: str = "MEDICAL PARTS",
) -> tuple[NS, dict, dict]:
    source = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "name": source_name,
                    "city": source_city,
                    "country": source_country,
                },
                "notifyParties": [
                    {"name": source_name, "city": source_city, "country": source_country}
                ],
            },
            "cargoGroups": [{"description": description, "marksAndNumbers": list(marks)}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["parties"]["consignee"] = {
        "name": "Lakeshore BioSupply LLC",
        "city": "Chicago",
        "country": "United States",
    }
    target["documentPatch"]["parties"]["notifyParties"] = [{"sameAs": "consignee"}]
    bindings = tuple(
        NS(
            logical_key=f"mark:{index}",
            group_kind="cargo",
            target_paths=(f"documentPatch.cargoGroups[0].marksAndNumbers[{index}]",),
            dependency_paths=(),
            dependency_bindings=(),
            occurrences=(NS(source_text=mark),),
        )
        for index, mark in enumerate(marks)
    )
    return NS(bindings=bindings, auxiliary_semantic_plan=NS(entities=())), source, target


def _check(template: NS, source: dict, target: dict, auxiliary: dict | None = None) -> None:
    validate_party_mark_preflight(
        template=template,
        source_target=source,
        target=target,
        auxiliary_values={} if auxiliary is None else auxiliary,
    )


def test_party_name_and_locality_need_explicit_segment_ownership() -> None:
    template, source, target = _fixture()
    with pytest.raises(ValueError, match=r"explicit segment ownership.*mark:0"):
        _check(template, source, target)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [
        "LAKESHORE BIOSUPPLY LLC",
        "CHICAGO, UNITED STATES",
    ]
    with pytest.raises(ValueError, match=r"explicit segment ownership.*mark:0"):
        _check(template, source, target)
    template.bindings[0].dependency_paths = ("documentPatch.parties.consignee.name",)
    with pytest.raises(ValueError, match=r"explicit segment ownership.*mark:1"):
        _check(template, source, target)
    template.bindings[1].dependency_paths = (
        "documentPatch.parties.consignee.city",
        "documentPatch.parties.consignee.country",
    )
    _check(template, source, target)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"][0] = "MED CARE EGYPT"
    with pytest.raises(ValueError, match="differs from its declared generated party"):
        _check(template, source, target)


def test_country_only_mark_is_linked_only_beside_a_party_name() -> None:
    template, source, target = _fixture(
        source_name="FUTURE PIPE INDUSTRIES SAE",
        source_city="6TH OCTOBER CITY",
        marks=("FUTURE PIPE INDUSTRIES SAE", "EGYPT"),
    )
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [
        "LAKESHORE BIOSUPPLY LLC",
        "UNITED STATES",
    ]
    template.bindings[0].dependency_paths = ("documentPatch.parties.consignee.name",)
    with pytest.raises(ValueError, match=r"explicit segment ownership.*mark:1"):
        _check(template, source, target)
    template.bindings[1].dependency_paths = ("documentPatch.parties.consignee.country",)
    _check(template, source, target)

    # Country coincidence without the party-name mark is not a dependency.
    independent, independent_source, independent_target = _fixture(
        marks=("EGYPT",), description="FROZEN GOODS"
    )
    _check(independent, independent_source, independent_target)


def test_prefix_and_brand_distinction() -> None:
    prefix, source, target = _fixture(
        source_name="REAL MAN MOHAMED YEHYA MOHAMED AL-JABALI",
        marks=("REAL MAN",),
    )
    with pytest.raises(ValueError, match="explicit segment ownership"):
        _check(prefix, source, target)
    prefix.bindings[0].dependency_paths = ("documentPatch.parties.consignee.name",)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["LAKESHORE BIOSUPPLY LLC"]
    _check(prefix, source, target)

    one_word, source, target = _fixture(
        source_name="ASKOLL (CHINA) MOTOR TECHNOLOGIES CO., LTD",
        marks=("ASKOLL",),
        description="ASKOLL DRAIN PUMPS",
    )
    _check(one_word, source, target)
    exact_one_word, source, target = _fixture(
        source_name="JEHACO", marks=("JEHACO",), description="PUMPS SPARE PARTS"
    )
    with pytest.raises(ValueError, match="explicit segment ownership"):
        _check(exact_one_word, source, target)
    product_brand, source, target = _fixture(
        source_name="PLSTONE INTERNATIONAL LTD",
        marks=("PLSTONE INTERNATIONAL LTD",),
        description="PLSTONE INTERNATIONAL LTD BRAND TYRES",
    )
    _check(product_brand, source, target)


def test_compound_party_mark_and_source_only_repeat_share_owner() -> None:
    template, source, target = _fixture(marks=("MED CARE EGYPT",))
    template.bindings[0].target_paths += ("documentPatch.parties.consignee.name",)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["Lakeshore BioSupply LLC"]
    template.bindings += (
        NS(
            logical_key="repeated-mark",
            group_kind="cargo",
            target_paths=(),
            dependency_paths=(),
            dependency_bindings=("mark:0",),
            occurrences=(NS(source_text="MED CARE EGYPT"),),
        ),
    )
    _check(template, source, target)

    template.bindings[-1].dependency_bindings = ()
    with pytest.raises(ValueError, match=r"explicit segment ownership.*repeated-mark"):
        _check(template, source, target)

    template.bindings[-1].dependency_bindings = ("mark:0",)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["MED CARE EGYPT"]
    with pytest.raises(ValueError, match="differs from its declared generated party"):
        _check(template, source, target)


def test_party_name_embedded_in_captioned_mark_needs_owned_segment() -> None:
    template, source, target = _fixture(
        marks=("SHIP TO: MED CARE EGYPT / REF 71322",), description="MEDICAL PARTS"
    )
    with pytest.raises(ValueError, match="explicit segment ownership"):
        _check(template, source, target)
    template.bindings[0].dependency_paths = ("documentPatch.parties.consignee.name",)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [
        "SHIP TO: Lakeshore BioSupply LLC / REF 71322"
    ]
    _check(template, source, target)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [
        "MED CARE EGYPT / Lakeshore BioSupply LLC"
    ]
    with pytest.raises(ValueError, match="differs from its declared generated party"):
        _check(template, source, target)

    template.bindings[0].occurrences = (NS(source_text="MED CARE EGYPT"),)
    _check(template, source, source)
    with pytest.raises(ValueError, match="lacks physical source proof"):
        _check(template, source, target)


def test_independent_source_only_cargo_entity_is_not_assumed_to_be_a_mark() -> None:
    template, source, target = _fixture(marks=(), description="CHEMICALS")
    template.bindings = (
        NS(
            logical_key="agent:cargo_manufacturer_text",
            group_kind="cargo",
            target_paths=(),
            dependency_paths=(),
            dependency_bindings=(),
            occurrences=(NS(source_text="MED CARE EGYPT"),),
        ),
    )
    _check(template, source, target)
    template.bindings[0].logical_key = "agent:cargo:0:marks0:repeat"
    with pytest.raises(ValueError, match="explicit segment ownership"):
        _check(template, source, target)


def test_same_party_registration_requires_binding_provenance() -> None:
    template, source, target = _fixture(marks=("TAX NO: 613700163",))
    template.bindings += (
        NS(
            logical_key="party-tax",
            group_kind="party",
            target_paths=(),
            occurrences=(NS(source_text="613700163"),),
        ),
    )
    template.auxiliary_semantic_plan.entities = (
        NS(
            relationship="same_as_target_party",
            target_party_path="documentPatch.parties.consignee",
            members=(NS(field="tax_identifier", logical_key="party-tax"),),
        ),
    )
    with pytest.raises(ValueError, match="registration mark lacks explicit binding ownership"):
        _check(template, source, target, {"party-tax": "862433927"})
    template.bindings[0].dependency_bindings = ("party-tax",)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["TAX NO: 862433927"]
    _check(template, source, target, {"party-tax": "862433927"})
    template.bindings[0].occurrences = (NS(source_text="613700163"),)
    with pytest.raises(ValueError, match="registration mark lacks physical source proof"):
        _check(template, source, target, {"party-tax": "862433927"})
    template.bindings[0].occurrences = (NS(source_text="TAX NO: 613700163"),)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["TAX NO: 8624339279"]
    with pytest.raises(ValueError, match="differs from generated entity value"):
        _check(template, source, target, {"party-tax": "862433927"})


def test_formatted_registration_surfaces_share_the_same_identifier() -> None:
    template, source, target = _fixture(marks=("TAX: 05.735.193/0004-95",))
    template.bindings += (
        NS(
            logical_key="party-tax",
            group_kind="party",
            target_paths=(),
            occurrences=(
                NS(source_text="05.735.193/0004-95"),
                NS(source_text="05735193000495"),
            ),
        ),
    )
    template.auxiliary_semantic_plan.entities = (
        NS(
            relationship="same_as_target_party",
            target_party_path="documentPatch.parties.consignee",
            members=(NS(field="tax_identifier", logical_key="party-tax"),),
        ),
    )
    template.bindings[0].dependency_bindings = ("party-tax",)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["TAX: 87.654.321/0001-22"]
    _check(template, source, target, {"party-tax": "87654321000122"})


def test_source_only_registration_mark_requires_same_party_binding() -> None:
    template, source, target = _fixture(marks=(), description="MACHINERY")
    template.bindings = (
        NS(
            logical_key="agent:cargo:marks:tax_id",
            group_kind="cargo",
            target_paths=(),
            dependency_paths=(),
            dependency_bindings=(),
            occurrences=(NS(source_text="728 - 401 - 266"),),
        ),
        NS(
            logical_key="party-tax",
            group_kind="party",
            target_paths=(),
            dependency_paths=(),
            dependency_bindings=(),
            occurrences=(NS(source_text="728401266"),),
        ),
    )
    template.auxiliary_semantic_plan.entities = (
        NS(
            relationship="same_as_target_party",
            target_party_path="documentPatch.parties.consignee",
            members=(NS(field="tax_identifier", logical_key="party-tax"),),
        ),
    )
    with pytest.raises(ValueError, match="source-only party registration mark"):
        _check(template, source, target, {"party-tax": "135792468"})
    template.bindings[0].dependency_bindings = ("party-tax",)
    _check(template, source, target, {"party-tax": "135792468"})
    template.auxiliary_semantic_plan.entities[0].relationship = "independent"
    template.bindings[0].dependency_bindings = ()
    _check(template, source, target, {"party-tax": "135792468"})


_REAL_CASES = {
    "doc_c559693f": "doc_c559693f7c4556d8d7823d90bfa230e503d5fe94bfbf46c0a606386dfad9d13a",
    "doc_1e9d32e5": "doc_1e9d32e5bed47af2601329c7fbb2f15649300f58e2b20962a1de99acc9fddf2c",
}


def _catalog_case(prefix: str) -> tuple[CertifiedSemanticTemplate, dict]:
    cases = Path(
        "artifacts/kie-synthesis-production/template-base/catalogs/"
        "mpci-bl-production-template-catalog1507-v34-discharge-country/cases"
    )
    case = cases / _REAL_CASES[prefix]
    template = CertifiedSemanticTemplate.model_validate_json((case / "template.json").read_bytes())
    source = json.loads((case / "source-label.json").read_bytes())
    return template, source


def test_real_compiled_mark_replay_and_pre_provider_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    template, source = _catalog_case("doc_c559693f")
    target = deepcopy(source)
    target["documentPatch"]["parties"]["consignee"]["name"] = "KEDMA FOUNDRY MATERIALS LTD."
    with pytest.raises(ValueError, match="explicit segment ownership"):
        _check(template, source, target)

    case = NS(template=template, source_target=source, target=target, auxiliary_values={})
    monkeypatch.setattr(descendant, "_require_frozen_target", lambda _case: None)
    monkeypatch.setattr(
        descendant,
        "_dangerous_goods_outputs",
        lambda _case: pytest.fail("provider-related planning must not run"),
    )
    with pytest.raises(ValueError, match="explicit segment ownership"):
        descendant._build_initial_plan(case, seed=7, country_codes={})

    brand_template, brand_source = _catalog_case("doc_1e9d32e5")
    brand_target = deepcopy(brand_source)
    brand_target["documentPatch"]["parties"]["shipper"]["name"] = "NEW APPLIANCES LTD"
    _check(brand_template, brand_source, brand_target)
