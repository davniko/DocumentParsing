from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.cargo_identity_derivations import (
    require_tariff_owners,
)
from document_ocr.synthesis.template_compiler.measurement_prose import require_product_count_owners


def owner(text, *, target=(), dependency=()):
    start = text.index(b"20)")
    return NS(
        target_paths=target,
        dependency_paths=dependency,
        occurrences=(NS(byte_start=start, byte_end=start + 2),),
    )


def test_unbound_or_source_fixed_inner_count_cannot_pass():
    text = b"(1000KG X 20)20,000KG"
    for bindings in ((), (owner(text),)):
        with pytest.raises(ValueError, match="lacks package ownership"):
            require_product_count_owners(text, bindings)


@pytest.mark.parametrize("dependency", [False, True])
def test_explicit_quantity_ownership_with_utf8_offsets(dependency):
    text = "Poids échantillon (1000KG \u00d7 20) 20,000KG".encode()
    path = ("documentPatch.cargoPackages[0].quantity",)
    binding = owner(text, dependency=path if dependency else (), target=() if dependency else path)
    require_product_count_owners(text, (binding,))


def test_non_equation_text_does_not_require_an_invented_count():
    require_product_count_owners(b"1000 KG; 20 DRY; no printed multiplication", ())


@pytest.mark.parametrize("caption", ["COMMODITY CODE: ", "HS CODE: ", "H.S. NO. "])
def test_tariff_reference_requires_real_goods_ownership(caption):
    text = (caption + "03035510").encode()
    slot = NS(byte_start=len(caption), source_text="03035510")
    binding = NS(target_paths=(), dependency_paths=(), logical_key="code", occurrences=(slot,))
    with pytest.raises(ValueError, match="tariff code lacks"):
        require_tariff_owners(text, (binding,))
    binding.target_paths = ("documentPatch.cargoGroups[0].hsCodes[0]",)
    require_tariff_owners(text, (binding,))


def test_invoice_number_does_not_become_tariff_code():
    require_tariff_owners(
        b"INVOICE NUMBER: 03035510",
        (
            NS(
                target_paths=(),
                dependency_paths=(),
                logical_key="invoice",
                occurrences=(NS(byte_start=16, source_text="03035510"),),
            ),
        ),
    )
