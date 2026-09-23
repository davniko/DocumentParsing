"""An explicitly shared carrier role is fixed, not a source-copy fallback."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.generation_contract import (
    fixed_carrier_role_paths,
    require_complete_variation,
)


def _target():
    return {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "CARRIER LTD"},
                "forwardingAgent": {"name": "CARRIER LTD"},
                "shipper": {"name": "OLD SHIPPER"},
            }
        }
    }


def _binding(*paths, mode="carrier_static"):
    return SimpleNamespace(render_mode=mode, target_paths=paths)


_CARRIER = "documentPatch.parties.carrier.name"
_FORWARDER = "documentPatch.parties.forwardingAgent.name"


def test_explicit_shared_role_is_fixed_but_other_parties_must_vary():
    source = _target()
    target = deepcopy(source)
    target["documentPatch"]["parties"]["shipper"]["name"] = "NEW SHIPPER"
    bindings = (_binding(_CARRIER, _FORWARDER),)
    assert fixed_carrier_role_paths(source, bindings) == frozenset({_FORWARDER})
    assert require_complete_variation(source, target, bindings=bindings) == (
        "documentPatch.parties.shipper.name",
    )
    with pytest.raises(ValueError, match="synthesis did not occur"):
        require_complete_variation(source, source, bindings=bindings)


@pytest.mark.parametrize(
    "bindings",
    [
        (),
        (_binding(_FORWARDER),),
        (_binding(_CARRIER, _FORWARDER, mode="target_binding"),),
        (_binding(_CARRIER), _binding(_FORWARDER)),
    ],
)
def test_matching_names_or_static_mode_alone_do_not_establish_shared_identity(bindings):
    source = _target()
    target = deepcopy(source)
    target["documentPatch"]["parties"]["shipper"]["name"] = "NEW SHIPPER"
    with pytest.raises(ValueError, match="synthesis did not occur"):
        require_complete_variation(source, target, bindings=bindings)


@pytest.mark.parametrize("path", [_CARRIER, _FORWARDER])
def test_shared_carrier_identity_cannot_change(path):
    source = _target()
    target = deepcopy(source)
    target["documentPatch"]["parties"][path.split(".")[2]]["name"] = "CHANGED"
    target["documentPatch"]["parties"]["shipper"]["name"] = "NEW SHIPPER"
    with pytest.raises(ValueError, match="fixed carrier-role"):
        require_complete_variation(source, target, bindings=(_binding(_CARRIER, _FORWARDER),))


def test_shared_binding_requires_equal_corresponding_source_fields():
    source = _target()
    source["documentPatch"]["parties"]["forwardingAgent"]["name"] = "OTHER COMPANY"
    with pytest.raises(ValueError, match="shared carrier-role"):
        fixed_carrier_role_paths(source, (_binding(_CARRIER, _FORWARDER),))
    with pytest.raises(ValueError, match="shared carrier-role"):
        fixed_carrier_role_paths(
            _target(), (_binding(_CARRIER, "documentPatch.parties.shipper.address"),)
        )
