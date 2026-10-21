"""Keep typed identity components separate from a party's postal geography.

An address expression can legitimately contain a related company name, for
example ``C/O {related_name} {postal}``. Its country-like words are identity,
not an independently printed postal country. Only explicit contract kinds
establish that boundary; generated wording is never used to guess a company.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from typing import TYPE_CHECKING

from document_ocr.synthesis.curated_wording import postal_geography_counts

if TYPE_CHECKING:
    from document_ocr.synthesis.curated import SourceContract

_COMPONENT = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def postal_geography_expression(contract: SourceContract, expression: str) -> str:
    """Exclude certified name slots, leaving every postal slot/literal unchanged.

    Call before interpolation, including before fixed-component accounting.
    A second copy inside a postal variable is deliberately not masked.
    """
    kinds = {variable.key: variable.kind for variable in contract.variables}
    return _COMPONENT.sub(lambda match: " " if kinds[match[1]] == "name" else match[0], expression)


def validate_postal_component_identities(
    contract: SourceContract,
    expression: str,
    values: dict[str, str],
    *,
    locality: str,
    country: str,
) -> None:
    """Reject a separately supplied identity copied into this assembly's postal slot.

    This checks only complete, certified name components referenced by the same
    expression. It does not classify arbitrary organizations or reject an
    ordinary care-of/site phrase that has no separately supplied identity slot.
    A name equal to the locality/country is not distinguishable from geography
    by this lexical check and is not treated as an identity-copy proof.
    """
    kinds = {variable.key: variable.kind for variable in contract.variables}
    keys = sorted(set(_COMPONENT.findall(expression)))
    names = [key for key in keys if kinds[key] == "name"]
    if not names:
        return

    def normalized(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    geographic_names = {normalized(locality), normalized(country)}
    postal = {key: normalized(values[key]) for key in keys if kinds[key] == "postal"}
    for name_key in sorted(names):
        # A terminal full stop/comma is presentation, not a different company;
        # internal punctuation and the complete name wording remain required.
        identity = normalized(values[name_key]).rstrip(".,")
        if identity in geographic_names:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(identity) + r"(?!\w)")
        for postal_key, text in postal.items():
            if pattern.search(text):
                raise ValueError(
                    f"generated postal component {postal_key} repeats separately owned "
                    f"identity component {name_key}"
                )


def postal_field_geography_requirement(
    contract: SourceContract,
    expression: str,
    field_key: str,
    *,
    locality: str,
    country: str,
    country_code: str,
    country_labelled: bool,
    country_owner_keys: Collection[str] = (),
) -> str:
    """Make country ownership explicit in one generated postal field's schema.

    The full address context is still authoritative. This repeated field-level
    instruction prevents a street/locality component from independently becoming
    a complete address when its country belongs elsewhere in the same assembly.
    """
    kinds = {variable.key: variable.kind for variable in contract.variables}
    if kinds[field_key] != "postal":
        raise ValueError("postal geography instruction requires a certified postal field")
    keys = set(_COMPONENT.findall(expression))
    if field_key not in keys:
        return ""
    if country_labelled and field_key in country_owner_keys:
        return ""
    if not country_labelled:
        return (
            " This party omits a postal-country declaration. Do not add "
            f"{country!r} or {country_code!r} as a country line or suffix in this component. "
            "Keep the required locality once, even when its spelling contains or equals "
            "the country name. Separately owned company identities are not postal geography."
        )
    literal = _COMPONENT.sub("", postal_geography_expression(contract, expression))
    counts = postal_geography_counts(
        literal, locality=locality, country=country, country_code=country_code
    )
    if counts.country or counts.terminal_code:
        owner = "The unchanged host-owned address literal already supplies the country"
    elif keys.intersection(country_owner_keys):
        owner = "A separately owned component in this address assembly supplies the country"
    else:
        return ""
    return (
        f" {owner}. Do not output {country!r} or {country_code!r} as a country "
        "declaration in this component; do not append a country line or suffix. "
        "Return only this component, not a complete independent address. "
        "The required locality spelling is unaffected."
    )
