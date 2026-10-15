"""Explicit ownership of private people and source-only contact mailboxes.

Carrier companies remain public context. A signatory or mailbox is a separate
identity, with exact source ownership and no invented extraction-party role.
"""

from __future__ import annotations

import re

from document_ocr.synthesis.curated import flat, locate
from document_ocr.synthesis.curated_contacts import ContactField, ContactParty
from document_ocr.synthesis.curated_templates import SamplingBlueprint
from document_ocr.synthesis.template_compiler.contact_values import validate_mailbox


def private_contact_contract(blueprint: SamplingBlueprint) -> dict:
    spec = blueprint.ownership_data.get("private_contacts", {})
    if not isinstance(spec, dict) or set(spec) - {
        "names",
        "email_paths",
        "generic_email_paths",
        "replace_domain_paths",
        "provider_domain_paths",
        "emails",
    }:
        raise ValueError("private contacts require explicit names/email paths/source mailboxes")
    if not spec:
        return spec
    names = spec.get("names", {})
    emails = spec.get("emails", {})
    paths = spec.get("email_paths", [])
    generic_paths = spec.get("generic_email_paths", [])
    domain_paths = spec.get("replace_domain_paths", [])
    provider_paths = spec.get("provider_domain_paths", [])
    if (
        not isinstance(names, dict)
        or not isinstance(emails, dict)
        or not isinstance(paths, list)
        or not isinstance(generic_paths, list)
        or not isinstance(domain_paths, list)
        or not isinstance(provider_paths, list)
        or any(
            not isinstance(path, str)
            for path in (*paths, *generic_paths, *domain_paths, *provider_paths)
        )
    ):
        raise ValueError("invalid private contact ownership sections")
    leaves = flat(blueprint.target)
    declared = paths + generic_paths
    if len(set(declared)) != len(declared) or any(
        path not in leaves or ".emailAddresses[" not in path for path in declared
    ):
        raise ValueError("private mailbox paths must name distinct existing email targets")
    domain_declared = domain_paths + provider_paths
    if len(set(domain_declared)) != len(domain_declared) or not set(domain_declared) <= set(
        declared
    ):
        raise ValueError("private company domains require distinct classified email target paths")
    variables = {v.key: v for v in blueprint.contract.variables}
    for key, declaration in names.items():
        if (
            key not in variables
            or variables[key].kind != "name"
            or not isinstance(declaration, dict)
            or set(declaration) != {"role"}
            or not isinstance(declaration["role"], str)
            or not declaration["role"].strip()
        ):
            raise ValueError("private person needs a known name variable and explicit printed role")
        if any("{" + key + "}" in b.expression for b in blueprint.contract.targets):
            raise ValueError("source-only private person cannot own an extraction target")
        owned = [locate(blueprint.source, occurrence) for occurrence in variables[key].occurrences]
        pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, variables[key].value.split())) + r"(?!\w)"
        for match in re.finditer(pattern, blueprint.source, re.I):
            start = len(blueprint.source[: match.start()].encode())
            end = len(blueprint.source[: match.end()].encode())
            if not any(a <= start and end <= b for a, b in owned):
                raise ValueError("private person has an unowned repeated source occurrence")
    present_keys = {r.key for r in blueprint.regions if r.curated_key is None}
    for key, declaration in emails.items():
        if (
            key not in present_keys
            or key not in blueprint.historical_bindings
            or not isinstance(declaration, dict)
            or set(declaration) - {"company", "country", "role", "person_key", "domain_policy"}
            or not {"company", "country", "role"} <= declaration.keys()
            or not isinstance(declaration["role"], str)
            or not declaration["role"].strip()
            or not isinstance(declaration["country"], str)
            or (
                declaration["company"] is not None
                and (
                    not isinstance(declaration["company"], str)
                    or not declaration["company"].strip()
                )
            )
            or declaration.get("person_key") not in (None, *names)
        ):
            raise ValueError("source mailbox requires an exact source-only owner and context")
        if not isinstance(declaration.get("domain_policy"), str) or declaration[
            "domain_policy"
        ] not in {"private_company", "public_provider", "public_carrier"}:
            raise ValueError(
                "source-only mailbox requires explicit private-company, public-provider "
                "or public-carrier domain policy"
            )
        binding = blueprint.historical_bindings[key]
        if any(path in leaves for path in binding.get("target_paths", [])):
            raise ValueError("labelled mailbox must use its target email path")
        originals = {"".join(o["source_text"].split()) for o in binding["occurrences"]}
        if len(originals) != 1:
            raise ValueError("one private mailbox owner has conflicting source identities")
        validate_mailbox(next(iter(originals)))
    return spec


def source_contact_parties(
    blueprint: SamplingBlueprint, sample_id: str, wording: dict[str, str]
) -> list[ContactParty]:
    """Reuse company/person context without adding these actors to the target."""
    result = []
    for key, declaration in private_contact_contract(blueprint).get("emails", {}).items():
        original = "".join(
            blueprint.historical_bindings[key]["occurrences"][0]["source_text"].split()
        )
        person_key = declaration.get("person_key")
        if person_key is not None and (
            person_key not in wording or not wording[person_key].strip()
        ):
            raise ValueError("private mailbox requires its generated person identity")
        result.append(
            ContactParty(
                sample_id,
                declaration["company"],
                declaration["country"],
                (
                    ContactField(
                        "c0",
                        "email",
                        original,
                        (),
                        personal=True,
                        bindings=(key,),
                        replace_domain=declaration["domain_policy"] == "private_company",
                    ),
                ),
                declaration["role"],
                (wording[person_key],) if person_key is not None else (),
            )
        )
    return result


def validate_private_person_coverage(blueprint: SamplingBlueprint) -> None:
    """A public carrier does not make its named employees public identities."""
    spec = private_contact_contract(blueprint)
    email_paths = {p for p in flat(blueprint.target) if ".emailAddresses[" in p}
    declared = set(spec.get("email_paths", [])) | set(spec.get("generic_email_paths", []))
    if email_paths != declared:
        raise ValueError(
            "every source email requires explicit private or generic recipient classification"
        )
    domain_declared = set(spec.get("replace_domain_paths", [])) | set(
        spec.get("provider_domain_paths", [])
    )
    if email_paths != domain_declared:
        raise ValueError(
            "every source email requires explicit private-company or "
            "public-provider domain classification"
        )
    for region in blueprint.regions:
        if region.curated_key is not None:
            continue
        binding = blueprint.historical_bindings[region.key]
        if binding.get("render_mode") == "carrier_static" and binding["value_kind"] in {
            "person",
            "contact_name",
        }:
            raise ValueError("private carrier person requires a generated source-only name owner")
