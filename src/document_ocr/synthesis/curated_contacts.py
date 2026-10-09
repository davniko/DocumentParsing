"""Company-conditioned email and website wording, separate from shipment facts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from document_ocr.synthesis.curated import assign, digest, flat
from document_ocr.synthesis.template_compiler.contact_values import validate_mailbox

CONTACT_PROMPT = (
    "Write realistic fictional business emails and websites for the supplied new companies. "
    "The original contacts are style examples; create new contacts belonging to the new name. "
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


@dataclass(frozen=True)
class ContactParty:
    sample_id: str
    name: str | None
    country: str
    fields: tuple[ContactField, ...]
    role: str | None = None


def contact_parties(source: dict, target: dict, sample_id: str) -> list[ContactParty]:
    """Share named-company contacts; keep unnamed party roles independently owned."""
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
        # Contact generation follows company identity, not the configurable
        # casing of its label/presentation. Keep repeats and request hashes
        # stable when only capitalization changes; endpoints themselves remain
        # case-preserved.
        identity = (
            name.upper() if name else None,
            after.get(role + ".country", "").upper(),
            None if name else role,
        )
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
            role,
        )
        for (name, country, role), fields in groups.items()
    ]


def _validated_mailbox(value: str) -> str:
    validate_mailbox(value)
    return value


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
            "Generate only contact values in the printed style; company identity is not supplied."
        )
        + f"\nCOUNTRY: {p.country or 'not printed'}\n"
        + "\n".join(f"{f.key} — original {f.kind}: {f.original}" for f in p.fields)
        for i, p in enumerate(parties)
    )


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
            if (
                host.endswith((".example", ".invalid", ".test"))
                or value.casefold() == field.original.casefold()
            ):
                raise ValueError("contact retained a source identity or placeholder domain")
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
