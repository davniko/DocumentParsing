"""PII-free route, freight, and party modeling views for B/L synthesis.

The prepared relation tables remain the semantic authority.  These views expose
only low-cardinality topology, missingness, cardinality, and coarse relationship
features to statistical models.  Source identifiers remain in ``BenchmarkView``
lineage arrays; names, addresses, cities, printed countries, and contact values
never enter the model DataFrames.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol, cast

from document_ocr.synthesis.sdv_harness import BenchmarkView

PartyRelation = Literal[
    "origin",
    "destination",
    "third_country",
    "same_as",
    "missing",
]
CityMode = Literal[
    "endpoint_port",
    "other_same_country",
    "third_country",
    "missing",
]
PartyGeographyPresence = Literal[
    "neither",
    "city_only",
    "country_only",
    "city_and_country",
]
PaymentSide = Literal["origin", "destination", "third_country", "missing"]


class CountryResolver(Protocol):
    """Strict country resolver boundary used by scenario view construction.

    Resolvers return one ISO alpha-2 code for a known printed value and
    ``None`` when that value is unresolved.  Returning ``None`` lets the
    builder publish one exhaustive audit covering every unresolved source
    cell instead of stopping at the first miss.
    """

    def resolve(self, value: str | None) -> str | None: ...


CountryResolverInput = Mapping[str, str] | CountryResolver | Callable[[str | None], str | None]

_ISO_COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")
_WORD = re.compile(r"\w+", re.UNICODE)
_CONTACT_COUNT_PROFILE = re.compile(
    r"^phone=(?P<phone>0|[1-9][0-9]*)\|email=(?P<email>0|[1-9][0-9]*)"
    r"\|website=(?P<website>0|[1-9][0-9]*)$"
)
_PARTY_ROLES = frozenset(
    {
        "shipper",
        "consignee",
        "notifyParties",
        "carrier",
        "forwardingAgent",
        "deliveryAgent",
        "consolidator",
    }
)
_ORIGIN_LOCATION_ROLES = ("portOfLoading", "placeOfReceipt")
_DESTINATION_LOCATION_ROLES = (
    "finalDestination",
    "placeOfDelivery",
    "portOfDischarge",
)
_PORT_LOCATION_ROLES = frozenset({"portOfLoading", "transshipmentPort", "portOfDischarge"})
_LOCATION_ROLES = frozenset(
    {
        "placeOfIssue",
        "placeOfReceipt",
        "portOfLoading",
        "transshipmentPort",
        "portOfDischarge",
        "placeOfDelivery",
        "finalDestination",
        "paymentPlace",
    }
)
_ORIGIN_PREFERRED_PARTY_ROLES = frozenset({"shipper", "forwardingAgent"})
_DESTINATION_PREFERRED_PARTY_ROLES = frozenset({"consignee", "notifyParties", "deliveryAgent"})

PARTY_STRUCTURE_COLUMNS = (
    "role",
    "relation_to_route",
    "city_mode",
    "geography_presence",
    "name_present",
    "contact_name_present",
    "contact_count_profile",
    "address_character_count",
    "address_word_count",
)

PARTY_STRUCTURE_PROJECTED_COLUMNS = (
    "role",
    "relation_to_route",
    "city_mode",
    "same_as_present",
    "name_present",
    "address_present",
    "city_present",
    "country_present",
    "contact_name_present",
    "phone_present",
    "email_present",
    "website_present",
    "phone_count",
    "email_count",
    "website_count",
    "address_character_count",
    "address_word_count",
)

ROUTE_FREIGHT_DOCUMENT_COLUMNS = (
    "origin_country_code",
    "destination_country_code",
    "freight_payment_arrangement",
    "freight_payment_side",
    "negotiability",
    "has_place_of_issue",
    "has_place_of_receipt",
    "has_port_of_loading",
    "has_transshipment_port",
    "has_port_of_discharge",
    "has_place_of_delivery",
    "has_final_destination",
    "has_payment_place",
    "has_bill_of_lading_number",
    "has_original_bill_of_lading_number",
    "has_master_bill_of_lading_number",
    "has_issue_date",
    "has_shipped_on_board_date",
    "has_vessel_name",
    "has_vessel_imo_number",
    "has_voyage_number",
    "has_vessel_flag_country",
    "party_count",
    "notify_party_count",
    "party_contact_count",
    "container_count",
    "cargo_group_count",
    "package_count",
    "allocation_group_count",
    "allocation_count",
    "dangerous_goods_count",
    "hs_code_count",
)

_PARTY_CATEGORICAL_COLUMNS = frozenset(
    {
        "role",
        "relation_to_route",
        "city_mode",
        "geography_presence",
        "contact_count_profile",
    }
)
_PARTY_BOOLEAN_COLUMNS = frozenset({"name_present", "contact_name_present"})
_PARTY_RELATIONS = frozenset({"origin", "destination", "third_country", "same_as", "missing"})
_CITY_MODES = frozenset({"endpoint_port", "other_same_country", "third_country", "missing"})
_GEOGRAPHY_PRESENCE = frozenset({"neither", "city_only", "country_only", "city_and_country"})
_ROUTE_CATEGORICAL_COLUMNS = frozenset(
    {
        "origin_country_code",
        "destination_country_code",
        "freight_payment_arrangement",
        "freight_payment_side",
        "negotiability",
    }
)
_ROUTE_BOOLEAN_COLUMNS = frozenset(
    column for column in ROUTE_FREIGHT_DOCUMENT_COLUMNS if column.startswith("has_")
)

_PRIMARY_KEYS: Mapping[str, str] = {
    "documents": "document_id",
    "document_locations": "location_id",
    "parties": "party_id",
    "party_contacts": "party_contact_id",
    "containers": "container_id",
    "cargo_groups": "cargo_group_row_id",
    "cargo_hs_codes": "cargo_group_value_id",
    "packages": "package_row_id",
    "allocation_groups": "allocation_group_id",
    "allocations": "allocation_id",
    "dangerous_goods": "dangerous_goods_id",
}


class ScenarioViewError(ValueError):
    """A scenario-view input violates scope, schema, or relationship contracts."""


@dataclass(frozen=True, slots=True)
class UnresolvedCountry:
    """One exact printed country that the injected resolver could not resolve."""

    document_id: str
    table: str
    row_id: str
    column: str
    value: str


class CountryResolutionError(ScenarioViewError):
    """All unresolved country cells from one otherwise valid build attempt."""

    def __init__(self, issues: Sequence[UnresolvedCountry]) -> None:
        ordered = tuple(
            sorted(
                issues,
                key=lambda item: (
                    item.document_id,
                    item.table,
                    item.row_id,
                    item.column,
                    item.value,
                ),
            )
        )
        if not ordered:
            raise ValueError("CountryResolutionError requires at least one issue")
        self.issues = ordered
        preview = ", ".join(
            f"{item.document_id}:{item.table}:{item.row_id}:{item.value!r}" for item in ordered[:5]
        )
        super().__init__(f"country resolver has {len(ordered)} unresolved modeled cells: {preview}")


@dataclass(frozen=True, slots=True)
class ScenarioViewsAudit:
    fit_document_count: int
    fit_template_count: int
    party_structure_rows: int
    party_structure_inverse_projection_rows: int
    route_freight_document_rows: int
    country_cells_present: int
    country_cells_resolved: int
    country_cells_missing: int
    raw_pii_columns_exposed: Literal[False] = False


@dataclass(frozen=True, slots=True)
class ScenarioBenchmarkViews:
    party_structure: BenchmarkView
    route_freight_document: BenchmarkView
    audit: ScenarioViewsAudit


@dataclass(frozen=True, slots=True)
class _CountryResolutionCounts:
    present: int
    resolved: int
    missing: int


def _driver_scalar(value: Any) -> str | int | bool:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        return value
    raise ScenarioViewError(
        f"party structure contains an unsupported scalar: {type(value).__name__}"
    )


def party_structure_driver_violations(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the independent statistical drivers without derived duplicates."""

    if set(row) != set(PARTY_STRUCTURE_COLUMNS):
        return ("column_contract",)
    violations: list[str] = []
    scalar: dict[str, str | int | bool | None] = {}
    for column in PARTY_STRUCTURE_COLUMNS:
        try:
            scalar[column] = _driver_scalar(row[column])
        except ScenarioViewError:
            scalar[column] = None
            violations.append(f"{column}_scalar")
    if scalar["role"] not in _PARTY_ROLES:
        violations.append("role_domain")
    relation = scalar["relation_to_route"]
    if relation not in _PARTY_RELATIONS:
        violations.append("relation_domain")
    city_mode = scalar["city_mode"]
    if city_mode not in _CITY_MODES:
        violations.append("city_mode_domain")
    geography = scalar["geography_presence"]
    if geography not in _GEOGRAPHY_PRESENCE:
        violations.append("geography_presence_domain")
    for column in ("name_present", "contact_name_present"):
        if type(scalar[column]) is not bool:
            violations.append(f"{column}_boolean")
    profile = scalar["contact_count_profile"]
    if not isinstance(profile, str) or _CONTACT_COUNT_PROFILE.fullmatch(profile) is None:
        violations.append("contact_count_profile_format")
    counts: dict[str, int] = {}
    for column in (
        "address_character_count",
        "address_word_count",
    ):
        value = scalar[column]
        if type(value) is not int or value < 0:
            violations.append(f"{column}_nonnegative_integer")
        else:
            counts[column] = value
    if counts.get("address_character_count") == 0 and counts.get("address_word_count") != 0:
        violations.append("address_absent_words")
    if counts.get("address_word_count", 0) > counts.get("address_character_count", -1):
        violations.append("address_word_character_order")
    city_present = geography in {"city_only", "city_and_country"}
    country_present = geography in {"country_only", "city_and_country"}
    if not city_present and city_mode != "missing":
        violations.append("city_mode_requires_city")
    if city_mode != "missing" and not country_present:
        violations.append("city_mode_requires_country")
    if city_present and country_present and city_mode == "missing":
        violations.append("city_country_require_mode")
    if not country_present and relation not in {"same_as", "missing"}:
        violations.append("relation_requires_country")
    if country_present and relation == "missing":
        violations.append("country_requires_relation")
    return tuple(sorted(set(violations)))


def project_party_structure_driver(row: Mapping[str, Any]) -> dict[str, str | int | bool]:
    """Reconstruct all deterministic party topology fields from one valid driver row."""

    violations = party_structure_driver_violations(row)
    if violations:
        raise ScenarioViewError(f"invalid party structure driver row: {violations}")
    scalar = {column: _driver_scalar(row[column]) for column in PARTY_STRUCTURE_COLUMNS}
    geography = cast(str, scalar["geography_presence"])
    profile = _CONTACT_COUNT_PROFILE.fullmatch(cast(str, scalar["contact_count_profile"]))
    if profile is None:
        raise RuntimeError("validated contact count profile did not parse")
    counts: dict[str, int] = {
        "phone_count": int(profile.group("phone")),
        "email_count": int(profile.group("email")),
        "website_count": int(profile.group("website")),
        **{
            column: cast(int, scalar[column])
            for column in (
                "address_character_count",
                "address_word_count",
            )
        },
    }
    projected: dict[str, str | int | bool] = {
        "role": scalar["role"],
        "relation_to_route": scalar["relation_to_route"],
        "city_mode": scalar["city_mode"],
        "same_as_present": scalar["relation_to_route"] == "same_as",
        "name_present": scalar["name_present"],
        "address_present": counts["address_character_count"] > 0,
        "city_present": geography in {"city_only", "city_and_country"},
        "country_present": geography in {"country_only", "city_and_country"},
        "contact_name_present": scalar["contact_name_present"],
        "phone_present": counts["phone_count"] > 0,
        "email_present": counts["email_count"] > 0,
        "website_present": counts["website_count"] > 0,
        **counts,
    }
    if tuple(projected) != PARTY_STRUCTURE_PROJECTED_COLUMNS:
        raise RuntimeError("party structure projection columns changed unexpectedly")
    return projected


class _CountryResolver:
    def __init__(self, values: CountryResolverInput) -> None:
        self._resolve: Callable[[str | None], str | None]
        if isinstance(values, Mapping):
            mapping: dict[str, str] = {}
            for printed, code in values.items():
                if (
                    not isinstance(printed, str)
                    or not printed
                    or printed != printed.strip()
                    or not isinstance(code, str)
                    or _ISO_COUNTRY_CODE.fullmatch(code) is None
                ):
                    raise ScenarioViewError(
                        "country resolver requires exact non-empty printed keys and "
                        "ISO alpha-2 values"
                    )
                mapping[printed] = code
            if not mapping:
                raise ScenarioViewError("country resolver must not be empty")
            self._resolve = cast(Callable[[str | None], str | None], mapping.get)
        else:
            method = getattr(values, "resolve", None)
            if callable(method):
                self._resolve = method
            elif callable(values):
                self._resolve = values
            else:
                raise ScenarioViewError(
                    "country resolver must be a mapping, resolver object, or callable"
                )
        self._issues: list[UnresolvedCountry] = []
        self._present = 0
        self._resolved = 0
        self._missing = 0

    def resolve(
        self,
        value: Any,
        *,
        document_id: str,
        table: str,
        row_id: str,
        column: str,
    ) -> str | None:
        if value is None:
            self._missing += 1
            return None
        if not isinstance(value, str) or not value or value != value.strip():
            raise ScenarioViewError(
                f"{table}:{row_id}:{column} must be null or an exact non-empty string"
            )
        self._present += 1
        try:
            code = self._resolve(value)
        except Exception as error:
            raise ScenarioViewError(
                f"country resolver failed at {table}:{row_id}:{column}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if code is None:
            self._issues.append(
                UnresolvedCountry(
                    document_id=document_id,
                    table=table,
                    row_id=row_id,
                    column=column,
                    value=value,
                )
            )
            return None
        if not isinstance(code, str) or _ISO_COUNTRY_CODE.fullmatch(code) is None:
            raise ScenarioViewError(
                f"country resolver returned a non-ISO alpha-2 value at "
                f"{table}:{row_id}:{column}: {code!r}"
            )
        self._resolved += 1
        return code

    def finish(self) -> _CountryResolutionCounts:
        if self._issues:
            raise CountryResolutionError(self._issues)
        return _CountryResolutionCounts(
            present=self._present,
            resolved=self._resolved,
            missing=self._missing,
        )


def _validate_scope(
    *,
    document_ids: Sequence[str],
    fit_document_ids: Sequence[str],
    partition_by_document: Mapping[str, str],
    template_by_document: Mapping[str, str],
    allowed_partition: str,
) -> tuple[str, ...]:
    corpus = tuple(document_ids)
    if not corpus or len(corpus) != len(set(corpus)):
        raise ScenarioViewError("document table IDs must be non-empty and unique")
    corpus_set = set(corpus)
    for label, values in (
        ("partition", partition_by_document),
        ("template", template_by_document),
    ):
        if set(values) != corpus_set:
            missing = sorted(corpus_set - set(values))
            extra = sorted(set(values) - corpus_set)
            raise ScenarioViewError(
                f"{label} map coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ScenarioViewError(f"{label} map values must be non-empty strings")
    if not allowed_partition.strip():
        raise ScenarioViewError("allowed_partition must be non-empty")
    fit = tuple(fit_document_ids)
    if not fit or len(fit) != len(set(fit)):
        raise ScenarioViewError("fit_document_ids must be non-empty and unique")
    unknown = sorted(set(fit) - corpus_set)
    if unknown:
        raise ScenarioViewError(f"fit scope references absent documents: {unknown[:5]}")
    wrong_partition = sorted(
        document_id
        for document_id in fit
        if partition_by_document[document_id] != allowed_partition
    )
    if wrong_partition:
        raise ScenarioViewError(
            f"fit documents are outside partition {allowed_partition!r}: {wrong_partition[:5]}"
        )
    members_by_template: dict[str, set[str]] = defaultdict(set)
    for document_id in corpus:
        members_by_template[template_by_document[document_id]].add(document_id)
    crossed: list[tuple[str, str, tuple[str, ...]]] = []
    checked_templates: set[str] = set()
    for document_id in fit:
        template_id = template_by_document[document_id]
        if template_id in checked_templates:
            continue
        checked_templates.add(template_id)
        foreign = tuple(
            sorted(
                member
                for member in members_by_template[template_id]
                if partition_by_document[member] != allowed_partition
            )
        )
        if foreign:
            crossed.append((document_id, template_id, foreign))
    if crossed:
        document_id, template_id, foreign = crossed[0]
        raise ScenarioViewError(
            "fit template crosses the allowed partition: "
            f"{document_id} ({template_id}); non-{allowed_partition} members={list(foreign[:5])}"
        )
    return fit


def _validate_and_group_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    document_ids: Sequence[str],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, tuple[Mapping[str, Any], ...]]],
]:
    missing_tables = sorted(set(_PRIMARY_KEYS) - set(tables))
    if missing_tables:
        raise ScenarioViewError(f"prepared tables are missing: {missing_tables}")
    document_set = set(document_ids)
    by_document: dict[str, dict[str, tuple[Mapping[str, Any], ...]]] = defaultdict(dict)
    document_index: dict[str, Mapping[str, Any]] = {}
    for table, primary_key in _PRIMARY_KEYS.items():
        seen: set[str] = set()
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in tables[table]:
            row_id = row.get(primary_key)
            document_id = row.get("document_id")
            if (
                not isinstance(row_id, str)
                or not row_id
                or row_id in seen
                or not isinstance(document_id, str)
                or document_id not in document_set
            ):
                raise ScenarioViewError(
                    f"{table} contains an invalid/duplicate key or unknown document"
                )
            seen.add(row_id)
            grouped[document_id].append(row)
            if table == "documents":
                document_index[document_id] = row
        for document_id, rows in grouped.items():
            by_document[document_id][table] = tuple(rows)
    if set(document_index) != document_set:
        raise ScenarioViewError("documents table index differs from its declared IDs")
    return document_index, by_document


def _text_present(value: Any, *, label: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, str) or not value or value != value.strip():
        raise ScenarioViewError(f"{label} must be null or an exact non-empty string")
    return True


def _normalized_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_WORD.findall(normalized))


def _location_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    document_id: str,
    resolver: _CountryResolver,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str | None]]:
    locations: dict[str, Mapping[str, Any]] = {}
    codes: dict[str, str | None] = {}
    for row in rows:
        row_id = row.get("location_id")
        role = row.get("location_role")
        if (
            not isinstance(row_id, str)
            or not isinstance(role, str)
            or role not in _LOCATION_ROLES
            or role in locations
        ):
            raise ScenarioViewError(f"invalid or duplicate location role in {document_id}")
        locations[role] = row
        if role in (
            set(_ORIGIN_LOCATION_ROLES)
            | set(_DESTINATION_LOCATION_ROLES)
            | {"transshipmentPort", "paymentPlace"}
        ):
            codes[role] = resolver.resolve(
                row.get("country"),
                document_id=document_id,
                table="document_locations",
                row_id=row_id,
                column="country",
            )
    return locations, codes


def _first_code(codes: Mapping[str, str | None], roles: Sequence[str]) -> str | None:
    return next((codes[role] for role in roles if codes.get(role) is not None), None)


def _relation(
    *,
    role: str,
    same_as_present: bool,
    country_code: str | None,
    origin_codes: frozenset[str],
    destination_codes: frozenset[str],
) -> PartyRelation:
    if same_as_present:
        return "same_as"
    if country_code is None:
        return "missing"
    in_origin = country_code in origin_codes
    in_destination = country_code in destination_codes
    if in_origin and in_destination:
        if role in _DESTINATION_PREFERRED_PARTY_ROLES:
            return "destination"
        if role in _ORIGIN_PREFERRED_PARTY_ROLES:
            return "origin"
        return "third_country"
    if in_origin:
        return "origin"
    if in_destination:
        return "destination"
    return "third_country"


def _city_mode(
    *,
    city: Any,
    country_code: str | None,
    origin_codes: frozenset[str],
    destination_codes: frozenset[str],
    port_names: frozenset[tuple[str, str]],
) -> CityMode:
    if city is None:
        return "missing"
    if not isinstance(city, str) or not city or city != city.strip():
        raise ScenarioViewError("party city must be null or an exact non-empty string")
    if country_code is None:
        return "missing"
    if (country_code, _normalized_surface(city)) in port_names:
        return "endpoint_port"
    if country_code in origin_codes or country_code in destination_codes:
        return "other_same_country"
    return "third_country"


def _metadata(
    name: str,
    columns: Sequence[str],
    *,
    categorical: frozenset[str],
    boolean: frozenset[str],
) -> dict[str, Any]:
    definitions: dict[str, dict[str, str]] = {}
    for column in columns:
        if column in categorical:
            definitions[column] = {"sdtype": "categorical"}
        elif column in boolean:
            definitions[column] = {"sdtype": "boolean"}
        else:
            definitions[column] = {
                "sdtype": "numerical",
                "computer_representation": "Int64",
            }
    return {
        "METADATA_SPEC_VERSION": "V1",
        "tables": {name: {"columns": definitions}},
        "relationships": [],
    }


def _presence(document: Mapping[str, Any], column: str) -> bool:
    return _text_present(document.get(column), label=f"documents:{column}")


def build_scenario_views(
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_document_ids: Sequence[str],
    partition_by_document: Mapping[str, str],
    template_by_document: Mapping[str, str],
    country_resolver: CountryResolverInput,
    allowed_partition: str = "train",
) -> ScenarioBenchmarkViews:
    """Build two low-cardinality, train/template-isolated benchmark views."""

    document_rows = tables.get("documents")
    if document_rows is None:
        raise ScenarioViewError("prepared tables are missing: ['documents']")
    document_ids = tuple(
        cast(str, row.get("document_id"))
        for row in document_rows
        if isinstance(row.get("document_id"), str)
    )
    if len(document_ids) != len(document_rows):
        raise ScenarioViewError("documents table contains an invalid document_id")
    fit = _validate_scope(
        document_ids=document_ids,
        fit_document_ids=fit_document_ids,
        partition_by_document=partition_by_document,
        template_by_document=template_by_document,
        allowed_partition=allowed_partition,
    )
    document_index, by_document = _validate_and_group_tables(tables, document_ids)
    resolver = _CountryResolver(country_resolver)
    party_model_rows: list[dict[str, Any]] = []
    party_row_ids: list[str] = []
    party_group_ids: list[str] = []
    party_partitions: list[str] = []
    route_model_rows: list[dict[str, Any]] = []
    route_row_ids: list[str] = []
    route_group_ids: list[str] = []
    route_partitions: list[str] = []

    for document_id in fit:
        rows = by_document.get(document_id, {})
        locations, location_codes = _location_index(
            rows.get("document_locations", ()),
            document_id=document_id,
            resolver=resolver,
        )
        origin_codes = frozenset(
            code
            for role in _ORIGIN_LOCATION_ROLES
            if (code := location_codes.get(role)) is not None
        )
        destination_codes = frozenset(
            code
            for role in _DESTINATION_LOCATION_ROLES
            if (code := location_codes.get(role)) is not None
        )
        origin_code = _first_code(location_codes, _ORIGIN_LOCATION_ROLES)
        destination_code = _first_code(location_codes, _DESTINATION_LOCATION_ROLES)
        port_names = frozenset(
            (cast(str, location_codes[role]), _normalized_surface(cast(str, row["name"])))
            for role, row in locations.items()
            if role in _PORT_LOCATION_ROLES
            and location_codes.get(role) is not None
            and _text_present(
                row.get("name"), label=f"document_locations:{row['location_id']}:name"
            )
        )
        contacts_by_party: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for contact in rows.get("party_contacts", ()):
            party_id = contact.get("party_id")
            contact_type = contact.get("contact_type")
            if not isinstance(party_id, str) or contact_type not in {"phone", "email", "website"}:
                raise ScenarioViewError(f"invalid party contact in {document_id}")
            if not _text_present(
                contact.get("value"),
                label=f"party_contacts:{contact.get('party_contact_id')}:value",
            ):
                raise ScenarioViewError("party contact values cannot be null")
            contacts_by_party[party_id].append(contact)

        parties = rows.get("parties", ())
        known_party_ids = {cast(str, party.get("party_id")) for party in parties}
        orphan_contacts = sorted(set(contacts_by_party) - known_party_ids)
        if orphan_contacts:
            raise ScenarioViewError(f"orphan party contacts in {document_id}: {orphan_contacts}")
        for party in parties:
            party_id = party.get("party_id")
            role = party.get("role")
            if not isinstance(party_id, str) or role not in _PARTY_ROLES:
                raise ScenarioViewError(f"invalid party row in {document_id}")
            same_as_present = _text_present(
                party.get("same_as"), label=f"parties:{party_id}:same_as"
            )
            country_code = resolver.resolve(
                party.get("country"),
                document_id=document_id,
                table="parties",
                row_id=party_id,
                column="country",
            )
            contact_counts = {kind: 0 for kind in ("phone", "email", "website")}
            for contact in contacts_by_party.get(party_id, ()):
                contact_counts[cast(str, contact["contact_type"])] += 1
            address = party.get("address")
            address_present = _text_present(address, label=f"parties:{party_id}:address")
            address_text = cast(str, address) if address_present else ""
            city_present = _text_present(party.get("city"), label=f"parties:{party_id}:city")
            country_present = country_code is not None
            geography_presence: PartyGeographyPresence
            if city_present and country_present:
                geography_presence = "city_and_country"
            elif city_present:
                geography_presence = "city_only"
            elif country_present:
                geography_presence = "country_only"
            else:
                geography_presence = "neither"
            relation = _relation(
                role=cast(str, role),
                same_as_present=same_as_present,
                country_code=country_code,
                origin_codes=origin_codes,
                destination_codes=destination_codes,
            )
            city_mode = _city_mode(
                city=party.get("city"),
                country_code=country_code,
                origin_codes=origin_codes,
                destination_codes=destination_codes,
                port_names=port_names,
            )
            name_present = _text_present(party.get("name"), label=f"parties:{party_id}:name")
            contact_name_present = _text_present(
                party.get("contact_name"), label=f"parties:{party_id}:contact_name"
            )
            model_row = {
                "role": role,
                "relation_to_route": relation,
                "city_mode": city_mode,
                "geography_presence": geography_presence,
                "name_present": name_present,
                "contact_name_present": contact_name_present,
                "contact_count_profile": (
                    f"phone={contact_counts['phone']}|email={contact_counts['email']}"
                    f"|website={contact_counts['website']}"
                ),
                "address_character_count": len(address_text),
                "address_word_count": len(_WORD.findall(address_text)),
            }
            if tuple(model_row) != PARTY_STRUCTURE_COLUMNS:
                raise RuntimeError("party structure row columns changed unexpectedly")
            source_projection = {
                "role": role,
                "relation_to_route": relation,
                "city_mode": city_mode,
                "same_as_present": same_as_present,
                "name_present": name_present,
                "address_present": address_present,
                "city_present": city_present,
                "country_present": country_present,
                "contact_name_present": contact_name_present,
                "phone_present": contact_counts["phone"] > 0,
                "email_present": contact_counts["email"] > 0,
                "website_present": contact_counts["website"] > 0,
                "phone_count": contact_counts["phone"],
                "email_count": contact_counts["email"],
                "website_count": contact_counts["website"],
                "address_character_count": len(address_text),
                "address_word_count": len(_WORD.findall(address_text)),
            }
            if project_party_structure_driver(model_row) != source_projection:
                raise RuntimeError("party structure driver projection differs from source topology")
            party_model_rows.append(model_row)
            party_row_ids.append(party_id)
            party_group_ids.append(template_by_document[document_id])
            party_partitions.append(partition_by_document[document_id])

        document = document_index[document_id]
        payment_code = location_codes.get("paymentPlace")
        payment_side: PaymentSide
        if payment_code is None:
            payment_side = "missing"
        elif payment_code in origin_codes:
            payment_side = "origin"
        elif payment_code in destination_codes:
            payment_side = "destination"
        else:
            payment_side = "third_country"
        route_row = {
            "origin_country_code": origin_code,
            "destination_country_code": destination_code,
            "freight_payment_arrangement": document.get("freight_payment_arrangement"),
            "freight_payment_side": payment_side,
            "negotiability": document.get("negotiability"),
            "has_place_of_issue": "placeOfIssue" in locations,
            "has_place_of_receipt": "placeOfReceipt" in locations,
            "has_port_of_loading": "portOfLoading" in locations,
            "has_transshipment_port": "transshipmentPort" in locations,
            "has_port_of_discharge": "portOfDischarge" in locations,
            "has_place_of_delivery": "placeOfDelivery" in locations,
            "has_final_destination": "finalDestination" in locations,
            "has_payment_place": "paymentPlace" in locations,
            "has_bill_of_lading_number": _presence(document, "bill_of_lading_number"),
            "has_original_bill_of_lading_number": _presence(
                document, "original_bill_of_lading_number"
            ),
            "has_master_bill_of_lading_number": _presence(document, "master_bill_of_lading_number"),
            "has_issue_date": _presence(document, "issue_date"),
            "has_shipped_on_board_date": _presence(document, "shipped_on_board_date"),
            "has_vessel_name": _presence(document, "transport_vessel_name"),
            "has_vessel_imo_number": _presence(document, "transport_vessel_imo_number"),
            "has_voyage_number": _presence(document, "transport_voyage_number"),
            "has_vessel_flag_country": _presence(document, "transport_vessel_flag_country"),
            "party_count": len(parties),
            "notify_party_count": sum(party.get("role") == "notifyParties" for party in parties),
            "party_contact_count": len(rows.get("party_contacts", ())),
            "container_count": len(rows.get("containers", ())),
            "cargo_group_count": len(rows.get("cargo_groups", ())),
            "package_count": len(rows.get("packages", ())),
            "allocation_group_count": len(rows.get("allocation_groups", ())),
            "allocation_count": len(rows.get("allocations", ())),
            "dangerous_goods_count": len(rows.get("dangerous_goods", ())),
            "hs_code_count": len(rows.get("cargo_hs_codes", ())),
        }
        if tuple(route_row) != ROUTE_FREIGHT_DOCUMENT_COLUMNS:
            raise RuntimeError("route/freight document row columns changed unexpectedly")
        route_model_rows.append(route_row)
        route_row_ids.append(document_id)
        route_group_ids.append(template_by_document[document_id])
        route_partitions.append(partition_by_document[document_id])

    counts = resolver.finish()
    pandas = import_module("pandas")
    party_data = pandas.DataFrame.from_records(party_model_rows, columns=PARTY_STRUCTURE_COLUMNS)
    route_data = pandas.DataFrame.from_records(
        route_model_rows, columns=ROUTE_FREIGHT_DOCUMENT_COLUMNS
    )
    party_view = BenchmarkView(
        name="party_structure",
        data=party_data,
        metadata=_metadata(
            "party_structure",
            PARTY_STRUCTURE_COLUMNS,
            categorical=_PARTY_CATEGORICAL_COLUMNS,
            boolean=_PARTY_BOOLEAN_COLUMNS,
        ),
        row_ids=tuple(party_row_ids),
        group_ids=tuple(party_group_ids),
        partition_labels=tuple(party_partitions),
        allowed_partition=allowed_partition,
    )
    route_view = BenchmarkView(
        name="route_freight_document",
        data=route_data,
        metadata=_metadata(
            "route_freight_document",
            ROUTE_FREIGHT_DOCUMENT_COLUMNS,
            categorical=_ROUTE_CATEGORICAL_COLUMNS,
            boolean=_ROUTE_BOOLEAN_COLUMNS,
        ),
        row_ids=tuple(route_row_ids),
        group_ids=tuple(route_group_ids),
        partition_labels=tuple(route_partitions),
        allowed_partition=allowed_partition,
    )
    return ScenarioBenchmarkViews(
        party_structure=party_view,
        route_freight_document=route_view,
        audit=ScenarioViewsAudit(
            fit_document_count=len(fit),
            fit_template_count=len(set(route_group_ids)),
            party_structure_rows=len(party_data),
            party_structure_inverse_projection_rows=len(party_data),
            route_freight_document_rows=len(route_data),
            country_cells_present=counts.present,
            country_cells_resolved=counts.resolved,
            country_cells_missing=counts.missing,
        ),
    )
