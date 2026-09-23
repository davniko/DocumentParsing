from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.descendant import _render_agent_target_binding


def fixture(surfaces):
    path = "documentPatch.parties.shipper.country"
    source = {"documentPatch": {"parties": {"shipper": {"country": "UAE"}}}}
    target = {"documentPatch": {"parties": {"shipper": {"country": "United Kingdom"}}}}
    binding = NS(
        value_kind="location",
        target_paths=(path,),
        derivation=None,
        logical_key="country",
        group_kind="party",
        occurrences=tuple(
            NS(slot_id=str(i), source_text=v, render_policy="natural_text")
            for i, v in enumerate(surfaces)
        ),
        realization=NS(target_values=(NS(source_value="UAE"),)),
    )
    return binding, source, target


def test_complete_country_aliases_are_not_partitioned_into_incomplete_words():
    binding, source, target = fixture(("UAE", "UNITED ARAB EMIRATES"))
    output = _render_agent_target_binding(
        binding,
        source_target=source,
        target=target,
        country_codes={"uae": "AE", "unitedarabemirates": "AE", "unitedkingdom": "GB", "gbr": "GB"},
    )
    assert output.replacements == {"0": "GBR", "1": "UNITED KINGDOM"}
    assert output.canonical_value == "United Kingdom"


def test_repeated_country_resolution_requires_registry_evidence():
    binding, source, target = fixture(("UAE", "UNITED ARAB EMIRATES"))
    with pytest.raises(ValueError, match="pinned country registry"):
        _render_agent_target_binding(binding, source_target=source, target=target)


def test_genuinely_segmented_country_is_not_repeated_in_each_fragment():
    binding, source, target = fixture(("UNITED ARAB", "EMIRATES"))
    output = _render_agent_target_binding(
        binding,
        source_target=source,
        target=target,
        country_codes={"uae": "AE", "unitedarabemirates": "AE", "unitedkingdom": "GB"},
    )
    assert output.replacements == {"0": "UNITED", "1": "KINGDOM"}


@pytest.mark.parametrize(
    ("source", "expected"), [("U.A.E", "G.B.R"), ("CN", "GB"), ("China", "United Kingdom")]
)
def test_country_occurrence_preserves_code_width_and_punctuation(source, expected):
    from document_ocr.synthesis.template_compiler.descendant import _country_occurrence_surface

    assert (
        _country_occurrence_surface(
            source, "United Kingdom", {"gb": "GB", "gbr": "GB", "unitedkingdom": "GB"}
        )
        == expected
    )
