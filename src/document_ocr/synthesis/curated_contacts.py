"""Company-conditioned email and website wording, separate from shipment facts."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from document_ocr.synthesis.curated import assign, digest, flat
from document_ocr.synthesis.template_compiler.contact_values import validate_mailbox

CONTACT_PROMPT = (
    "Write realistic fictional business emails and websites for the supplied new companies. "
    "The original contacts are style examples; create new contacts belonging to the new name. "
    "For personal mailboxes, use the supplied new contact names where available and replace "
    "the original person's mailbox identity, not merely its domain. "
    "Private source recipients are omitted from style examples. If no new contact name is "
    "supplied, invent a fictional recipient; do not reconstruct one from company knowledge. "
    "Use plausible company domains or free-mail providers and natural mailbox names. "
    "Email and website domains are independent. Retain the website form "
    "(such as WWW. or https://). Return only the requested contact values. "
    "For unnamed parties, generate contacts in the supplied role and original contact style. "
    "These are synthetic training text, not verified or contactable businesses."
)

_MAILBOX_ATOM = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
_DOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_MAILBOX_PATTERN = rf"^{_MAILBOX_ATOM}(?:\.{_MAILBOX_ATOM})*@{_DOMAIN_LABEL}(?:\.{_DOMAIN_LABEL})+$"


@dataclass(frozen=True)
class ContactField:
    key: str
    kind: str
    original: str
    paths: tuple[str, ...]
    personal: bool = False
    bindings: tuple[str, ...] = ()
    replace_domain: bool = False


@dataclass(frozen=True)
class ContactParty:
    sample_id: str
    name: str | None
    country: str
    fields: tuple[ContactField, ...]
    role: str | None = None
    contact_names: tuple[str, ...] = ()


def contact_parties(
    source: dict,
    target: dict,
    sample_id: str,
    *,
    personal_paths: frozenset[str] = frozenset(),
    replace_domain_paths: frozenset[str] = frozenset(),
) -> list[ContactParty]:
    """Share company context, including explicit notify references, without copying labels."""
    before, after = flat(source), flat(target)
    if any(
        path not in before or ".emailAddresses[" not in path
        for path in personal_paths | replace_domain_paths
    ):
        raise ValueError("personal email declaration requires a present email target")
    groups = {}
    for path, original in before.items():
        kind = (
            "email"
            if ".emailAddresses[" in path
            else ("website" if ".websiteUrls[" in path else None)
        )
        if kind is None:
            continue
        role = path.split(".contactDetails.")[0]
        contact_name = after.get(role + ".contactDetails.contactName")
        name = after.get(role + ".name")
        country = after.get(role + ".country", "")
        reference = after.get(role + ".sameAs")
        if reference is not None:
            if reference not in ("shipper", "consignee") or not re.fullmatch(
                r"documentPatch\.parties\.notifyParties\[\d+\]", role
            ):
                raise ValueError(f"invalid explicit contact party reference: {role}/{reference}")
            referenced = target["documentPatch"]["parties"].get(reference)
            if (
                not isinstance(referenced, dict)
                or not referenced
                or referenced.get("sameAs") is not None
            ):
                raise ValueError(f"contact reference requires a concrete party: {reference}")
            # sameAs omits repeated identity fields by design. Supply the named
            # party's current context while retaining the notify contact paths.
            name, country = referenced.get("name"), referenced.get("country")
            if any(
                value is not None and (not isinstance(value, str) or not value.strip())
                for value in (name, country)
            ):
                raise ValueError(f"contact reference has invalid identity context: {reference}")
            country = "" if country is None else country
            if contact_name is None:
                contact_name = referenced.get("contactDetails", {}).get("contactName")
        # Contact generation follows company identity, not the configurable
        # casing of its label/presentation. Keep repeats and request hashes
        # stable when only capitalization changes; endpoints themselves remain
        # case-preserved.
        identity = (
            name.upper() if name else None,
            country.upper(),
            None if name else role,
            tuple(sorted({n.strip().upper() for n in contact_name.split(";") if n.strip()}))
            if isinstance(contact_name, str)
            else (),
        )
        groups.setdefault(identity, {}).setdefault((kind, original), []).append(path)
    return [
        ContactParty(
            sample_id,
            name,
            country,
            tuple(
                ContactField(
                    f"c{i}",
                    kind,
                    old,
                    tuple(paths),
                    any(p in personal_paths for p in paths),
                    replace_domain=any(p in replace_domain_paths for p in paths),
                )
                for i, ((kind, old), paths) in enumerate(fields.items())
            ),
            role,
            contact_names,
        )
        for (name, country, role, contact_names), fields in groups.items()
    ]


def _validated_mailbox(value: str) -> str:
    validate_mailbox(value)
    return value


def retains_mailbox_identity(original: str, value: str) -> bool:
    """Separator changes do not anonymize a recipient or a copied account ID."""
    old = re.sub(r"[^a-z0-9]", "", original.split("@", 1)[0].casefold())
    new = re.sub(r"[^a-z0-9]", "", value.split("@", 1)[0].casefold())
    return old == new or (len(old) >= 6 and any(c.isdigit() for c in old) and old in new)


def contact_output_type(parties: list[ContactParty]) -> type[BaseModel]:
    if not parties:
        raise ValueError("empty contact request")
    fields = {}
    for i, party in enumerate(parties):
        nested = create_model(
            f"CompanyContacts{i}",
            __config__=ConfigDict(extra="forbid"),
            __validators__={
                f"validate_{f.key}": field_validator(f.key)(_validated_mailbox)
                for f in party.fields
                if f.kind == "email"
            },
            **{
                f.key: (
                    str,
                    Field(
                        min_length=1,
                        **(
                            {"pattern": _MAILBOX_PATTERN, "max_length": 254}
                            if f.kind == "email"
                            else {"pattern": r"^\S+$"}
                        ),
                        description=(
                            "One ASCII dot-atom email address for this party, following the "
                            "original contact style; local part up to 64 characters."
                            + (
                                " New personal mailbox identity; "
                                "do not retain the original local part."
                                if f.personal
                                else ""
                            )
                            + (
                                " Use a new business domain; "
                                "the original company domain must change."
                                if f.replace_domain
                                else ""
                            )
                            if f.kind == "email"
                            else (
                                "New fictional website for the unnamed party; "
                                "the original hostname is not a supplied identity. "
                                if party.name is None
                                else "New company website. "
                            )
                            + (
                                "Include the original URL scheme."
                                if "://" in f.original
                                else "Start with www.; omit the URL scheme."
                                if f.original.lower().startswith("www.")
                                else "Use a bare hostname without a URL scheme."
                            )
                        ),
                    ),
                )
                for f in party.fields
            },
        )
        fields[f"p{i}"] = (
            nested,
            Field(
                description="Contacts for the supplied company or explicitly unnamed party role."
            ),
        )
    return create_model("SyntheticCompanyContacts", __config__=ConfigDict(extra="forbid"), **fields)


def contact_prompt(parties: list[ContactParty]) -> str:
    return "\n\n".join(
        (
            f"COMPANY p{i}: {p.name}"
            if p.name
            else f"PARTY p{i}: unnamed {p.role}\n"
            "Company identity is not supplied. Invent a fictional business domain for the "
            "new contacts; the original hostname is a style example, not an identity to retain."
        )
        + f"\nCOUNTRY: {p.country or 'not printed'}\n"
        + ("NEW CONTACT NAMES: " + "; ".join(p.contact_names) + "\n" if p.contact_names else "")
        + "\n".join(_contact_style(f, unnamed=p.name is None) for f in p.fields)
        for i, p in enumerate(parties)
    )


def _contact_style(field: ContactField, *, unnamed: bool = False) -> str:
    if unnamed and field.kind == "website":
        form = (
            "scheme " + field.original.split("://", 1)[0] + "://"
            if "://" in field.original
            else "WWW. prefix, without scheme"
            if field.original.casefold().startswith("www.")
            else "bare hostname, without scheme"
        )
        return f"{field.key} — new fictional website, {form}. Original hostname omitted."
    if not field.personal:
        if field.replace_domain:
            return (
                f"{field.key} — generic email recipient style: {field.original.split('@', 1)[0]}. "
                "Use a new fictional business domain for the generated company."
            )
        return f"{field.key} — original {field.kind}: {field.original}"
    local, domain = field.original.split("@", 1)
    separators = [s for s in (".", "_", "-") if s in local]
    style = "recipient words" + (" separated with " + " ".join(separators) if separators else "")
    if any(c.isdigit() for c in local):
        style += "; may include synthetic digits"
    # A source-only carrier/issuer mailbox keeps its public company context.
    # Labelled party contacts instead belong to the newly generated company;
    # showing its predecessor's private corporate domain adds no useful signal.
    domain_hint = (
        "; new fictional business domain"
        if field.bindings and not field.paths and field.replace_domain
        else f"; public company domain {domain}"
        if field.bindings and not field.paths
        else ""
    )
    return f"{field.key} — new private email: {style}{domain_hint}. Use a new recipient identity."


def request_hash(parties: list[ContactParty]) -> str:
    return digest(
        {
            "parties": [
                {k: v for k, v in p.__dict__.items() if k != "role" or v is not None}
                | {"fields": [f.__dict__ for f in p.fields]}
                for p in parties
            ],
            "system": CONTACT_PROMPT,
            "prompt": contact_prompt(parties),
            "schema": contact_output_type(parties).model_json_schema(),
        }
    )


def contact_host(value: str, kind: str) -> str:
    if kind == "email":
        return value.rsplit("@", 1)[-1].lower()
    parsed = urlsplit(value if "://" in value else "https://" + value)
    return (parsed.hostname or "").lower().removeprefix("www.")


def unpack_contacts(output: BaseModel, parties: list[ContactParty]) -> dict[str, dict[str, str]]:
    payload = contact_output_type(parties).model_validate(output.model_dump()).model_dump()
    results = {}
    array_values: dict[tuple[str, str], dict[str, str]] = {}
    for i, party in enumerate(parties):
        for field in party.fields:
            value = payload[f"p{i}"][field.key]
            if value != value.strip() or any(c.isspace() for c in value):
                raise ValueError("contact must be one complete value without whitespace")
            if field.kind == "email":
                validate_mailbox(value)
            else:
                parsed = urlsplit(value if "://" in value else "https://" + value)
                if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
                    raise ValueError("invalid website URL")
                # Reuse the mailbox domain validator; no DNS lookup or invented correction.
                validate_mailbox("validation@" + (parsed.hostname or ""))
                if ("://" in field.original) != ("://" in value) or (
                    field.original.lower().startswith("www.")
                    and not value.lower().startswith("www.")
                ):
                    raise ValueError("website form differs from source example")
            host = contact_host(value, field.kind)
            if field.replace_domain and host == contact_host(field.original, field.kind):
                raise ValueError(
                    f"contact retained the original private company domain: p{i}/{field.key}"
                )
            if (
                host.endswith((".example", ".invalid", ".test"))
                or value.casefold() == field.original.casefold()
            ):
                raise ValueError("contact retained a source identity or placeholder domain")
            if field.personal and retains_mailbox_identity(field.original, value):
                raise ValueError(
                    f"personal contact retained the source mailbox identity: p{i}/{field.key}"
                )
            for path in field.paths:
                # A shared contact can populate several roles. Uniqueness belongs
                # to each target list, not the company, request, or domain.
                array_path = path.rsplit("[", 1)[0]
                values = array_values.setdefault((party.sample_id, array_path), {})
                previous_path = values.get(value)
                if previous_path is not None and previous_path != path:
                    raise ValueError(
                        f"duplicate generated contact in {party.sample_id}/{array_path}: "
                        f"{previous_path} and {path}"
                    )
                values[value] = path
                results.setdefault(party.sample_id, {})[path] = value
            for key in field.bindings:
                path = "binding:" + key
                values = results.setdefault(party.sample_id, {})
                if path in values and values[path] != value:
                    raise ValueError("one source contact binding has conflicting generated values")
                values[path] = value
    return results


def apply_contacts(target: dict, values: dict[str, str], parties: list[ContactParty]) -> dict:
    expected = {
        path
        for p in parties
        for f in p.fields
        for path in (*f.paths, *("binding:" + b for b in f.bindings))
    }
    if set(values) != expected:
        raise ValueError("contact receipt has incomplete or unrelated fields")
    result = deepcopy(target)
    for path, value in values.items():
        if not path.startswith("binding:"):
            assign(result, path, value)
    return result


def contact_surfaces(values: dict[str, str]) -> dict[str, str]:
    """Source-only contacts are rendered into their owners, never extraction targets."""
    return {
        key.removeprefix("binding:"): value
        for key, value in values.items()
        if key.startswith("binding:")
    }
