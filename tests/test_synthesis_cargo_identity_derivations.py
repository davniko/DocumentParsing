from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import cargo_identity_derivations as aliases
from document_ocr.synthesis.template_compiler import lexical_facts

PATH = "documentPatch.cargoGroups[0].hsCodes[0]"


def binding(**updates):
    return NS(
        **{
            "logical_key": "goods-alias",
            "derivation": "sampled_cargo_identity",
            "dependency_paths": (PATH,),
            "dependency_bindings": (),
            "target_paths": (),
            "render_mode": "deterministic_derived",
            "group_kind": "cargo",
            "value_kind": "cargo_text",
            **updates,
        }
    )


def scenario():
    return NS(
        target={"documentPatch": {"cargoGroups": [{"groupId": "g1", "hsCodes": ["1234.5612"]}]}},
        identities={"g1": [{"hs6": "123456", "requiredDescription": "Sampled goods"}]},
        receipt={},
    )


def test_alias_uses_generated_identity_not_source_or_added_label():
    value = scenario()
    original = deepcopy(value.target)
    assert aliases.description(PATH, value) == "Sampled goods"
    value.identities["g1"][0]["properShippingName"] = "Sampled dangerous goods"
    assert aliases.description(PATH, value) == "Sampled dangerous goods"
    assert original == value.target
    aliases.validate_source([binding()], original)
    assert aliases.owner(binding()) == (0, 0)


@pytest.mark.parametrize(
    "updates",
    [
        {"target_paths": (PATH,)},
        {"dependency_paths": ()},
        {"dependency_paths": (PATH, PATH)},
        {"dependency_paths": (PATH.replace("hsCodes[0]", "description"),)},
        {"group_kind": "equipment"},
        {"value_kind": "identifier"},
        {"dependency_bindings": ("other",)},
        {"render_mode": "literal_static"},
    ],
)
def test_invalid_alias_contract_is_rejected(updates):
    with pytest.raises(ValueError, match="explicit printed HS owner"):
        aliases.owner(binding(**updates))


def test_missing_or_ambiguous_identity_is_not_fallback_text():
    value = scenario()
    value.identities["g1"] = []
    with pytest.raises(ValueError, match="unique generated"):
        aliases.description(PATH, value)
    value.identities["g1"] = [
        {"hs6": "123456", "requiredDescription": name} for name in ("One", "Two")
    ]
    with pytest.raises(ValueError, match="unique generated"):
        aliases.description(PATH, value)
    with pytest.raises(ValueError, match="absent printed"):
        aliases.validate_source([binding()], {"documentPatch": {"cargoGroups": []}})


def test_host_owns_alias_and_provider_cannot_overwrite_it():
    source = NS(target={"documentPatch": {}}, template=NS(bindings=()))
    field = {"key": "alias", "paths": [], "source": "Old goods", "cargoIdentityPath": PATH}
    plan = lexical_facts.prepare(
        source, [field], scenario=scenario(), projection=None, contract=None, sample_id="s", seed=1
    )
    assert plan.values == {"alias": "Sampled goods"}
    with pytest.raises(ValueError, match="overwrite"):
        plan.merge({"alias": "Unrelated goods"})


@pytest.mark.parametrize("prefix", ["", "Étiquette\n"])
def test_unowned_hs_for_goods_is_screened_in_drafts_and_byte_templates(prefix):
    raw = prefix + "HS CODE 4011.4000 FOR TYRES\n& 4013.9020 FOR TUBES."
    spans = [(raw.index(code), raw.index(code) + len(code)) for code in ("4011.4000", "4013.9020")]
    drafts = [
        NS(target_paths=(PATH,), render_mode="target_binding", char_start=s, char_end=e)
        for s, e in spans
    ]
    expected = tuple((raw.index(word), raw.index(word) + len(word)) for word in ("TYRES", "TUBES"))
    assert aliases.unowned_alias_spans(raw, drafts) == expected
    # Exercise the real compiler risk inventory as well as its span detector:
    # the strict wire schema must accept the risk kind emitted by the host.
    from document_ocr.synthesis.template_compiler import host

    risks = host.all_risk_candidates(raw, drafts, {})
    commodity_risks = [r for r in risks if r.kind == "cargo_identity_alias"]
    assert [r.source_text for r in commodity_risks] == ["TYRES", "TUBES"]
    for risk in commodity_risks:
        assert raw.encode()[risk.byte_start : risk.byte_end].decode() == risk.source_text
    certified = [
        NS(
            target_paths=d.target_paths,
            render_mode=d.render_mode,
            occurrences=(
                NS(
                    byte_start=len(raw[: d.char_start].encode()),
                    byte_end=len(raw[: d.char_end].encode()),
                ),
            ),
        )
        for d in drafts
    ]
    assert aliases.unowned_alias_spans(raw, certified) == expected
    owners = [
        NS(target_paths=(), render_mode="deterministic_derived", char_start=s, char_end=e)
        for s, e in expected
    ]
    assert aliases.unowned_alias_spans(raw, drafts + owners) == ()
    owners[0].render_mode = "literal_static"
    assert aliases.unowned_alias_spans(raw, drafts + owners) == expected[:1]
