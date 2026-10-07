"""Company-conditioned email and website wording, separate from shipment facts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, create_model

from document_ocr.synthesis.curated import assign, digest, flat
from document_ocr.synthesis.template_compiler.contact_values import validate_mailbox

CONTACT_PROMPT = (
    "Write realistic fictional business emails and websites for the supplied new companies. "
    "The original contacts are style examples; create new contacts belonging to the new name. "
    "Match the examples' contact style: free-mail examples use a free-mail provider; "
    "corporate examples use plausible company domains and natural mailbox names. "
    "Retain the website form (such as WWW. or https://). When examples share a company domain, "
    "the new contacts share one too. Return only the requested contact values. "
    "These are synthetic training text, not verified or contactable businesses."
)


@dataclass(frozen=True)
class ContactField:
    key: str
    kind: str
    original: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ContactParty:
    sample_id: str
    name: str
    country: str
    fields: tuple[ContactField, ...]


def contact_parties(source: dict, target: dict, sample_id: str) -> list[ContactParty]:
    """Repeated roles for the same named company share their contact values."""
    before, after = flat(source), flat(target)
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
        name = after.get(role + ".name")
        if not name:
            raise ValueError(f"contact generation requires a company name: {role}")
        # Contact generation follows company identity, not the configurable
        # casing of its label/presentation. Keep repeats and request hashes
        # stable when only capitalization changes; endpoints themselves remain
        # case-preserved.
        identity = (name.upper(), after.get(role + ".country", "").upper())
        groups.setdefault(identity, {}).setdefault((kind, original), []).append(path)
    return [
        ContactParty(
            sample_id,
            name,
            country,
            tuple(
                ContactField(f"c{i}", kind, old, tuple(paths))
                for i, ((kind, old), paths) in enumerate(fields.items())
            ),
        )
        for (name, country), fields in groups.items()
    ]


def contact_output_type(parties: list[ContactParty]) -> type[BaseModel]:
    if not parties:
        raise ValueError("empty contact request")
    fields = {}
    for i, party in enumerate(parties):
        nested = create_model(
            f"CompanyContacts{i}",
            __config__=ConfigDict(extra="forbid"),
            **{
                f.key: (
                    str,
                    Field(
                        min_length=1,
                        pattern=r"^\S+$",
                        description=(
                            "New mailbox for the named party, "
                            "following the original contact's style."
                            if f.kind == "email"
                            else "New company website. "
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
        fields[f"p{i}"] = (nested, Field(description="Contacts for the supplied company."))
    return create_model("SyntheticCompanyContacts", __config__=ConfigDict(extra="forbid"), **fields)


def contact_prompt(parties: list[ContactParty]) -> str:
    return "\n\n".join(
        f"COMPANY p{i}: {p.name}\nCOUNTRY: {p.country or 'not printed'}\n"
        + "\n".join(f"{f.key} — original {f.kind}: {f.original}" for f in p.fields)
        for i, p in enumerate(parties)
    )


def request_hash(parties: list[ContactParty]) -> str:
    return digest(
        {
            "parties": [p.__dict__ | {"fields": [f.__dict__ for f in p.fields]} for p in parties],
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
    for i, party in enumerate(parties):
        domains = {}
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
            if (
                host.endswith((".example", ".invalid", ".test"))
                or value.casefold() == field.original.casefold()
            ):
                raise ValueError("contact retained a source identity or placeholder domain")
            old_host = contact_host(field.original, field.kind)
            if old_host in domains and domains[old_host] != host:
                raise ValueError("contacts that shared a source domain diverged")
            domains[old_host] = host
            for path in field.paths:
                results.setdefault(party.sample_id, {})[path] = value
    return results


def apply_contacts(target: dict, values: dict[str, str], parties: list[ContactParty]) -> dict:
    expected = {path for p in parties for f in p.fields for path in f.paths}
    if set(values) != expected:
        raise ValueError("contact receipt has incomplete or unrelated fields")
    result = deepcopy(target)
    for path, value in values.items():
        assign(result, path, value)
    return result
