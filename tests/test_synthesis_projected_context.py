from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import projected_context as context
from document_ocr.synthesis.template_compiler.descendant import (
    BindingOutput,
    _render_target_binding,
    _validate_projected_context,
)


def fixture():
    path = "documentPatch.parties.shipper"
    name = NS(
        logical_key="name",
        target_paths=(path + ".name",),
        value_kind="organization",
        derivation=None,
        dependency_paths=(path + ".country",),
        dependency_bindings=("country",),
        occurrences=(NS(slot_id="n", source_text="OLD COMPANY"),),
        realization=NS(
            mode="token_projected_surface",
            adapter="natural_text",
            target_values=(NS(source_value="OLD COMPANY EGYPT"),),
            slots=(NS(required_target_prefix_tokens=(), required_target_suffix_tokens=("egypt",)),),
        ),
    )
    country = NS(
        logical_key="country",
        target_paths=(path + ".country",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        occurrences=(NS(slot_id="c", source_text="EGYPT"),),
        realization=NS(mode="single_surface", target_values=(NS(source_value="Egypt"),)),
    )
    return NS(bindings=(name, country)), {
        "documentPatch": {"parties": {"shipper": {"name": "OLD COMPANY EGYPT", "country": "Egypt"}}}
    }


@pytest.mark.parametrize("country", ["Canada", "United Kingdom"])
def test_same_party_shared_country_varies_without_new_labels_or_old_country(country):
    template, source = fixture()
    target = deepcopy(source)
    party = target["documentPatch"]["parties"]["shipper"]
    party.update(name="New Trading Group " + country, country=country)
    output = _render_target_binding(template.bindings[0], target, template=template)
    assert output.replacements == {"n": "NEW TRADING GROUP"}
    _validate_projected_context(
        template,
        {
            "name": output,
            "country": BindingOutput(replacements={"c": country.upper()}, canonical_value=country),
        },
    )
    field = dict(
        paths=["documentPatch.parties.shipper.name"],
        source="OLD COMPANY EGYPT",
        constraints=[dict(fixedLiteralTokens=[["egypt"]])],
        hostAssembly=dict(prefix="", suffix=" EGYPT", mutableSource="OLD COMPANY", minimumWords=2),
    )
    revised = context.condition_requests(NS(template=template, target=source), [field], target)[0]
    assert revised["hostAssembly"]["suffix"] == " " + country
    assert revised["constraints"][0]["fixedLiteralTokens"] == [country.lower().split()]
    assert field["hostAssembly"]["suffix"] == " EGYPT"
    assert set(party) == {"name", "country"}


def test_identical_text_in_another_party_does_not_create_a_context_dependency():
    template, _ = fixture()
    template.bindings[1].target_paths = ("documentPatch.parties.consignee.country",)
    assert context.eligible_edges(template.bindings[0], template.bindings) == ()


def test_one_dependency_annotation_is_not_enough_to_waive_context_validation():
    template, _ = fixture()
    template.bindings[0].dependency_bindings = ()
    assert context.declared_edges(template.bindings[0], template.bindings) == ()


def test_unchanged_company_target_cannot_hide_a_changed_printed_country():
    template, _ = fixture()
    outputs = {
        "name": BindingOutput(
            replacements={"n": "OLD COMPANY"}, canonical_value="OLD COMPANY EGYPT"
        ),
        "country": BindingOutput(replacements={"c": "CANADA"}, canonical_value="Canada"),
    }
    with pytest.raises(ValueError, match="omits or contradicts"):
        _validate_projected_context(template, outputs)


def test_offline_preview_applies_declared_context_before_prose_is_generated():
    from document_ocr.synthesis.template_compiler.lexical_facts import HostLexicalPlan

    template, original = fixture()
    target = deepcopy(original)
    target["documentPatch"]["parties"]["shipper"]["country"] = "United Kingdom"
    fields = [
        dict(
            key="name",
            paths=["documentPatch.parties.shipper.name"],
            sampledContext={"Egypt": "United Kingdom"},
        )
    ]
    preview = HostLexicalPlan({}, {}).preview(
        NS(template=template, target=original), target, fields
    )
    assert preview["documentPatch"]["parties"]["shipper"]["name"] == "OLD COMPANY United Kingdom"
    assert target["documentPatch"]["parties"]["shipper"]["name"] == "OLD COMPANY EGYPT"


@pytest.mark.parametrize("origin", ["Canada", "United Kingdom"])
def test_made_in_mark_uses_its_own_cargo_origin(origin):
    template, _ = fixture()
    mark, country = template.bindings
    path = "documentPatch.cargoGroups[0]"
    mark.target_paths = (path + ".marksAndNumbers[0]",)
    mark.value_kind = "cargo_text"
    mark.dependency_paths = (path + ".origin.name",)
    mark.occurrences[0].source_text = "BRAND MADE IN"
    mark.realization.target_values[0].source_value = "BRAND MADE IN EGYPT"
    country.target_paths = (path + ".origin.name",)
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"origin": {"name": origin}, "marksAndNumbers": ["NEW BRAND MADE IN " + origin]}
            ]
        }
    }
    assert len(context.eligible_edges(mark, template.bindings)) == 1
    output = _render_target_binding(mark, target, template=template)
    assert output.replacements == {"n": "NEW BRAND MADE IN"}
    _validate_projected_context(
        template,
        {
            "name": output,
            "country": BindingOutput(replacements={"c": origin.upper()}, canonical_value=origin),
        },
    )
    country.target_paths = ("documentPatch.cargoGroups[1].origin.name",)
    assert context.eligible_edges(mark, template.bindings) == ()


def test_uncaptioned_cargo_mark_cannot_infer_origin_from_equal_text():
    template, _ = fixture()
    mark, country = template.bindings
    mark.target_paths = ("documentPatch.cargoGroups[0].marksAndNumbers[0]",)
    country.target_paths = ("documentPatch.cargoGroups[0].origin.name",)
    assert context.eligible_edges(mark, template.bindings) == ()


def test_declared_context_does_not_join_an_unrelated_equal_country():
    template, original = fixture()
    other = deepcopy(template.bindings[1])
    other.logical_key = "unrelated"
    other.target_paths = ("documentPatch.parties.consignee.country",)
    other.occurrences[0].slot_id = "u"
    template.bindings = (*template.bindings, other)
    original["documentPatch"]["parties"]["shipper"].update(
        name="New Trading Canada", country="Canada"
    )
    output = _render_target_binding(template.bindings[0], original, template=template)
    _validate_projected_context(
        template,
        {
            "name": output,
            "country": BindingOutput(replacements={"c": "CANADA"}, canonical_value="Canada"),
            "unrelated": BindingOutput(replacements={"u": "GERMANY"}, canonical_value="Germany"),
        },
    )
