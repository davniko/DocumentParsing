"""Atomic, provider-constrained rewriting of synthetic Bill of Lading OCR text.

The model submits one terminal line-range patch.  The output function validates
and applies the complete patch atomically, so successful edits are not sent back
through another model turn.  A separate compact semantic reviewer may request a
bounded correction.  Every provider message and exact before/after artifact is
retained, while no training record is published by this probe.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import resource
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from difflib import SequenceMatcher
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    create_model,
    model_validator,
)
from pydantic_ai import (
    Agent,
    ModelRetry,
    NativeOutput,
    RunContext,
    ToolOutput,
    capture_run_messages,
)
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.messages import CachePoint, ModelResponse, UserContent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas import bill_of_lading_v3, bill_of_lading_v5
from document_ocr.labeling_agents.deterministic_annotation import (
    explicit_hs_codes_from_text,
)
from document_ocr.synthesis.config import (
    RawTextRewriteInputsConfig,
    RawTextRewriteTargetIntegrityConfig,
    SynthesisRawTextRewriteCycleProbeConfig,
)
from document_ocr.synthesis.container_semantics import (
    canonical_equipment_surface,
    review_source_equipment_surface,
)
from document_ocr.synthesis.country_registry import CountryRegistry, load_iso_country_registry
from document_ocr.synthesis.drafts import set_target_value, target_value
from document_ocr.synthesis.generators import DeterministicStream, largest_remainder_allocation
from document_ocr.synthesis.linguistic_completion_pipeline import DocumentLinguisticPlan
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_provider_key,
    model_messages,
    openai_responses_settings,
    usage_receipt,
)
from document_ocr.synthesis.package_registry import LoadedPackageRegistry, load_package_registry
from document_ocr.synthesis.raw_text_rewrite_probe import (
    ChangedLeaf,
    _load_jsonl,
    _resolve_pinned_file,
    _validate_linguistic_run,
    changed_leaves,
    unified_text_diff,
)
from document_ocr.synthesis.route_registry import (
    UnlocodeLocation,
    load_pinned_unlocode_locations,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.task_adapter import (
    BILL_OF_LADING_TASK_ADAPTER,
    BILL_OF_LADING_V5_TASK_ADAPTER,
)
from document_ocr.synthesis.thermal_goods import render_synthetic_hs_code
from document_ocr.synthesis.transport_capacity import (
    EquipmentCapacity,
    EquipmentFamily,
    TransportCapacityLimits,
    capacity_limits,
    equipment_capacity,
    reproject_measures_for_semantic_equipment,
)
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceText = Annotated[str, StringConstraints(min_length=1, max_length=600)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")
_NEWLINE = re.compile(r"\r\n|\r|\n")
_PLACEHOLDER = re.compile(
    r"(?<![A-Z0-9])(?:UNAVAILABLE|NOT[ \t]+AVAILABLE|UNKNOWN|TBD|TBA|N[./]A[.]?)(?![A-Z0-9])",
    re.IGNORECASE,
)
_HTML_ENTITY = re.compile(r"&(?:#[0-9]+|#x[0-9A-F]+|[A-Z][A-Z0-9]+);", re.IGNORECASE)
_INLINE_SLOT_LABEL = re.compile(
    r"(?<![A-Z0-9])(?P<label>PHONE|FAX|MOBILE|TEL(?:EPHONE)?|PH|E-?MAIL|WEBSITE|WEB|T)"
    r"(?P<spacing>[ \t]*):",
    re.IGNORECASE,
)
_ELECTRONIC_CONTACT_SURFACE = re.compile(
    r"(?:https?://|www\.)[^\s]+|\S*@\S+",
    re.IGNORECASE,
)
_PRESERVABLE_STATUS_LINE = re.compile(
    r"^[ \t]*(?:EXPRESS[ \t]+RELEASE|NON-NEGOTIABLE(?:[ \t]+SEA[ \t]+WAYBILL)?|"
    r"DELIVERY[ \t]+STATUS[ \t]+FREE[ \t]+OUT|FREE[ \t]+OUT|FCL/FCL|"
    r"FREIGHT[ \t]+(?:PREPAID|COLLECT))[ \t]*$",
    re.IGNORECASE,
)
_PRESERVABLE_LEGAL_BOILERPLATE_LINE = re.compile(
    r"^[ \t]*(?:"
    r"NOT[ \t]+NEGOTIABLE[ \t]+UNLESS[ \t]+CONSIGNED[ \t]+TO[ \t]+ORDER|"
    r"\(?TERMS?[ \t]+OF[ \t]+(?:THE[ \t]+)?BILL[ \t]+OF[ \t]+LADING\b.*\)?|"
    r"THE[ \t]+SURRENDER[ \t]+OF\b.*\bBILL[ \t]+OF[ \t]+LADING\b.*|"
    r"IN[ \t]+WITNESS[ \t]+WHEREOF\b.*\bBILLS?[ \t]+OF[ \t]+LADING\b.*|"
    r"ATTENTION[ \t]+OF[ \t]+SHIPPER\b.*\bTERMS?[ \t]+AND[ \t]+CONDITIONS?\b.*"
    r")[ \t]*$",
    re.IGNORECASE,
)
_PARTY_HEADING_LINE = re.compile(
    r"^[ \t]*(?:\([0-9]+\)[ \t]*)?"
    r"(?:SHIPPER(?:/EXPORTER)?|EXPORTER|CONSIGNEE|NOTIFY(?:[ \t]+PARTY)?)"
    r"(?:[ \t]*(?:/[ \t]*)?(?:\([^\r\n()]*\)|[0-9]+))*[ \t:.-]*$",
    re.IGNORECASE,
)
_PARTY_COUNTRY_METADATA_LINE = re.compile(
    r"(?ix)^"
    r"(?P<prefix>[ \t]*(?P<label>"
    r"(?:SHIPPER|(?:FOREIGN[ \t]+)?EXPORTER)(?:[ \t]+REGISTRATION)?[ \t]+COUNTRY"
    r"(?:[ \t]+CODE)?|"
    r"(?:CONSIGNEE|IMPORTER)(?:[ \t]+REGISTRATION)?[ \t]+COUNTRY(?:[ \t]+CODE)?"
    r")[ \t]*(?::|=)?[ \t]*)"
    r"(?P<value>[A-Z][A-Z0-9 .,'()/-]*?[A-Z0-9])"
    r"(?P<suffix>[ \t]*[,;.]?[ \t]*)$"
)


class RewritePreparationConfig(Protocol):
    """The source/target controls required before any rewrite model is constructed."""

    inputs: RawTextRewriteInputsConfig
    target_integrity: RawTextRewriteTargetIntegrityConfig


_STRUCTURAL_LINE_PREFIX = re.compile(r"^(?P<prefix>[ \t]*(?:\([0-9]+\)|[0-9]+(?:\.[0-9]+)+\.)\s+)")
_POSITIVE_REEFER_OPERATION = re.compile(
    r"\b(?:TEMPERATURE[ \t]+(?:IS[ \t]+)?(?:TO[ \t]+BE[ \t]+)?SET|"
    r"PLUGGING[ \t]+FOR[ \t]+(?:THE[ \t]+)?ACCOUNT)\b",
    re.IGNORECASE,
)
_NON_OPERATING_REEFER = re.compile(
    r"\b(?:NON[- ]OPERATING|SWITCHED[ \t]+OFF|COOLING[ \t]+NOT[ \t]+OPERATING)\b",
    re.IGNORECASE,
)
_COMPOUND_PARTY_RELATIONS: tuple[
    tuple[Literal["on_behalf_of", "trading_as"], re.Pattern[str]], ...
] = (
    ("on_behalf_of", re.compile(r"\b(?:ON\s+BEHALF\s+OF|O\s*/?\s*B\s*/?\s*O)\b", re.I)),
    (
        "trading_as",
        re.compile(r"\b(?:TRADING\s+AS|DOING\s+BUSINESS\s+AS|T\s*/\s*A|D\s*/\s*B\s*/\s*A)\b", re.I),
    ),
)
_COMPOUND_PARTY_ROLES = (
    "shipper",
    "consignee",
    "carrier",
    "forwardingAgent",
    "deliveryAgent",
    "consolidator",
)
_CONTAINER_TYPE_DESCRIPTION_PATH = re.compile(
    r"^documentPatch\.containers\[([0-9]+)\]\.typeDescription$"
)
_CONTAINER_SIZE_CATEGORY_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.sizeCategory$")
_CONTAINER_TYPE_CATEGORY_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.typeCategory$")
_CONTAINER_NUMBER_TARGET_PATH = re.compile(
    r"^documentPatch\.(?:containers\[[0-9]+\]|"
    r"cargoAllocationGroups\[[0-9]+\]\.allocations\[[0-9]+\])\.containerNumber$"
)
_CANONICAL_CONTAINER_NUMBER = re.compile(r"^[A-Z]{4}[0-9]{7}$")
_NUMERIC_DATE = re.compile(
    r"(?<![0-9])(?P<a>[0-9]{1,4})(?P<s>[./-])(?P<b>[0-9]{1,2})(?P=s)(?P<c>[0-9]{1,4})(?![0-9])"
)
_DAY_NAMED_MONTH_DATE = re.compile(
    r"(?<![A-Z0-9])(?P<d>[0-9]{1,2})(?P<s>[-./ ])(?P<m>[A-Z]{3,9})"
    r"(?P=s)(?P<y>[0-9]{2,4})(?![A-Z0-9])",
    re.IGNORECASE,
)
_NAMED_MONTH_DAY_DELIMITED_DATE = re.compile(
    r"(?<![A-Z0-9])(?P<m>[A-Z]{3,9})(?P<s>[-/.])(?P<d>[0-9]{1,2})"
    r"(?P<s2>(?P=s)|,)(?P<gap>[ ]*)(?P<y>[0-9]{2,4})(?![A-Z0-9])",
    re.IGNORECASE,
)
_NAMED_MONTH_DAY_DATE = re.compile(
    r"(?<![A-Z0-9])(?P<m>[A-Z]{3,9})(?P<month_punct>\.?)[ ]+"
    r"(?P<d>[0-9]{1,2})"
    r"(?P<comma>,?)[ ]+(?P<y>[0-9]{2,4})(?![A-Z0-9])",
    re.IGNORECASE,
)
_YEAR_NAMED_MONTH_DAY_DATE = re.compile(
    r"(?<![A-Z0-9])(?P<y>[0-9]{4})(?P<s1>[-./ ])(?P<m>[A-Z]{3,9})"
    r"(?P<s2>[-./ ])(?P<d>[0-9]{1,2})(?![A-Z0-9])",
    re.IGNORECASE,
)
_DATE_SURFACE_PATTERNS = (
    _NUMERIC_DATE,
    _DAY_NAMED_MONTH_DATE,
    _NAMED_MONTH_DAY_DELIMITED_DATE,
    _NAMED_MONTH_DAY_DATE,
    _YEAR_NAMED_MONTH_DAY_DATE,
)
# HS surfaces may contain internal punctuation (``3403 19 10`` or ``2401.100.010``),
# but a spaced hyphen separates two codes (``52094200 - 52114200``).  The former generic
# numeric scanner consumed the complete two-code list as one number and made both codes
# unlocatable.  Keep this grammar deliberately HS-specific.
_HS_NUMERIC_SURFACE = re.compile(
    r"(?<![0-9])[0-9]+(?:(?:[./]|-(?![ \t]))[0-9]+|[ \t]+[0-9]+)*(?![0-9])"
)
_ISSUE_DATE_CONTEXT = re.compile(
    r"(?:PLACE\s+(?:(?:AND|&)\s+DATE\s+OF\s+ISSUE|"
    r"OF\s+(?:B(?:\(S\)|S)?/?L|BILL(?:\(S\)|S)?)\s+ISSUE(?:/DATE)?|"
    r"OF\s+ISSUE(?:\s+DATE)?|ISSUED)|"
    r"DATE\s+(?:ISSUED|OF\s+ISSUE|ISSUE\s+OF\s+(?:WAYBILL|BILL(?:\(S\)|S)?))|\bDATED\b)",
    re.I,
)
_SHIPPED_DATE_CONTEXT = re.compile(
    r"(?:(?:SHIP|SHIPPED)\s+ON\s+BOARD(?:\s+DATE(?:D)?)?|ON\s+BOARD(?:\s+DATE)?|"
    r"DATE\s+(?:LADEN|SHIPPED)\s+ON\s+BOARD|LADEN\s+ON\s+BOARD)",
    re.I,
)
_ANONYMOUS_EQUIPMENT = re.compile(
    r"(?:/|X\s+)?(?:20|40|45)[\u2019'\"]?(?:\s*FT)?\s*"
    r"(?:HC|HQ|HIGH\s+CUBE|DRY|GP|DC|REEFER|RF)?"
    r"\s*CONTAINERS?\s+SAID\s+TO\s+CONTAIN",
    re.I,
)
_SPLIT_ANONYMOUS_EQUIPMENT_LINE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<count>[1-9][0-9,]*)(?P<x>[ \t]*X[ \t]*)"
    r"(?P<size>20|40|45)(?P<quote>[\u2019'\"]?)(?:[ \t]*FT)?[ \t]*"
    r"(?P<container>CONTAINERS?)(?P<suffix>[ \t]+SAID[ \t]+TO)(?P<trailing>[ \t]*)$",
    re.IGNORECASE,
)
_COMPACT_AGGREGATE_EQUIPMENT_LINE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<count>[1-9][0-9,]*)(?P<x>[ \t]*X[ \t]*)"
    r"(?P<equipment>(?:20|40|45)[^\r\n]{0,72}?)(?P<trailing>[ \t]*)$",
    re.IGNORECASE,
)
_COUNTED_AGGREGATE_EQUIPMENT_LINE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<count>[1-9][0-9,]*)(?P<x>[ \t]*X[ \t]*)"
    r"(?P<equipment>(?:20|40|45)[^\r\n]{0,36}?)"
    r"(?P<space>[ \t]+)(?P<container>CONTAINER(?:\(S\)|S)?)"
    r"(?P<suffix>[^\r\n]*?)(?P<trailing>[ \t]*)$",
    re.IGNORECASE,
)
_INLINE_EQUIPMENT_ALIAS = re.compile(
    r"(?<![A-Za-z0-9])(?:20|40|45)[ \t]*['\u2019\"]?[ \t]*(?:FT[ \t]*)?"
    r"(?:HC|HQ|HR|RH|RQ|RF|RE|GP|DR|DV|DC|ST|HW|HO)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_COUNT_PREFIXED_INLINE_EQUIPMENT_ALIAS = re.compile(
    r"(?<![A-Za-z0-9])[1-9][0-9,]*[ \t]*X[ \t]*"
    r"(?P<equipment>(?:20|40|45)[ \t]*['\u2019\"]?[ \t]*(?:FT[ \t]*)?"
    r"(?:HC|HQ|HR|RH|RQ|RF|RE|GP|DR|DV|DC|ST|HW|HO))"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_PARENTHETICAL_AGGREGATE_EQUIPMENT_LINE = re.compile(
    r"^(?P<prefix>[ \t]*SAY[ \t]*:[ \t]*)"
    r"(?P<written_count>[A-Za-z]+(?:[- \t]+[A-Za-z]+)*)"
    r"(?P<open>[ \t]*\([ \t]*)"
    r"(?P<body>[^()\r\n]{1,160}?)"
    r"(?P<close>[ \t]*\)[ \t]*)"
    r"(?P<container>CONTAINERS?)"
    r"(?P<suffix>[ \t]+ONLY\.[ \t]*)$",
    re.IGNORECASE,
)
_PARENTHETICAL_EQUIPMENT_COMPONENT = re.compile(
    r"(?P<equipment>(?:20|40|45)[A-Za-z'\u2019\"]{0,24}?)"
    r"(?P<x>[ \t]*X[ \t]*)(?P<count>[1-9][0-9,]*)",
    re.IGNORECASE,
)
_DENSE_AGGREGATE_MARKER = re.compile(
    r"\bWEIGHT[ \t]+IN[ \t]+KGS[ \t]+TOTAL[ \t]*:[ \t]*"
    r"(?P<count>[1-9][0-9,]*)[ \t]+CONTAINER(?:S|\(S\))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DENSE_AGGREGATE_CAPTION = re.compile(
    r"^[ \t]*(?:"
    r"CONTINUED[ \t]+FROM[ \t]+PREVIOUS[ \t]+SHEET"
    r"(?:[ \t]+SHEET[ \t]+[0-9]+[ \t]+OF[ \t]+[0-9]+)?|"
    r"CONTINUED[ \t]+ON[ \t]+(?:THE[ \t]+)?NEXT[ \t]+SHEET"
    r"(?:[ \t]+SHEET[ \t]+[0-9]+[ \t]+OF[ \t]+[0-9]+)?|"
    r"SHEET[ \t]+[0-9]+[ \t]+OF[ \t]+[0-9]+|"
    r"ADDITIONAL[ \t]+CHARGES[ \t]+OF[ \t]+[0-9]+"
    r")[ \t]*$",
    re.IGNORECASE,
)
_DENSE_AGGREGATE_POST_TUPLE_CAPTION = re.compile(
    r"[ \t]+ABOVE[ \t]+PARTICULARS[ \t]+DECLARED[ \t]+BY[ \t]+SHIPPER\."
    r"[ \t]+CARRIER[ \t]+NOT[ \t]+RESPONSIBLE\.[ \t]*$",
    re.IGNORECASE,
)
_EXACT_DG_CLASS_PATTERN = r"(?:1(?:\.[1-6](?:[A-HJ-LN-S])?)?|[2-9](?:\.[1-9])?)"
_DG_CLASS_BEFORE_UN = re.compile(
    r"(?P<class_prefix>(?:IMDG[ \t]+)?CLASS[ \t]*(?::[ \t]*)?)"
    rf"(?P<class>{_EXACT_DG_CLASS_PATTERN})(?P<middle>[^\r\n]{{0,36}}?)"
    r"(?P<un_prefix>UN(?:DG)?(?:[ \t]+NUMBER|[ \t]+NO\.?|[ \t]*#)?[ \t]*:?[ \t]*"
    r"(?:UN[ \t]*)?)"
    r"(?P<un>[0-9]{4})"
    r"(?P<pg_clause>[ \t]*(?:-[ \t]*)?(?:PG|PACKAG(?:E|ING)[ \t]+GROUP)"
    r"[ \t]*:[ \t]*"
    r"(?P<pg>I{1,3}|NOT[ \t]+ASSIGNED))?",
    re.IGNORECASE,
)
_DG_UN_BEFORE_CLASS = re.compile(
    r"(?P<un_prefix>UN(?:DG)?(?:[ \t]+NUMBER|[ \t]+NO\.?|[ \t]*#)?[ \t]*:?[ \t]*"
    r"(?:UN[ \t]*)?)"
    r"(?P<un>[0-9]{4})(?P<middle>[^\r\n]{0,36}?)"
    r"(?P<class_prefix>(?:IMDG[ \t]+)?CLASS[ \t]*(?::[ \t]*)?)"
    rf"(?P<class>{_EXACT_DG_CLASS_PATTERN})"
    r"(?P<pg_clause>[ \t]*(?:-[ \t]*)?(?:PG|PACKAG(?:E|ING)[ \t]+GROUP)"
    r"[ \t]*:[ \t]*"
    r"(?P<pg>I{1,3}|NOT[ \t]+ASSIGNED))?",
    re.IGNORECASE,
)
_CARRIER_RECEIPT_COUNT = re.compile(
    r"(?P<prefix>TOTAL[ \t]+(?:NUMBER|NO\.)[ \t]+OF[ \t]+"
    r"(?:CONTAINERS?[ \t]+OR[ \t]+PACKAGES?|PACKAGES?|CONTAINERS?)[ \t]+)"
    r"(?P<count>[0-9][0-9,]*)"
    r"(?P<suffix>[ \t]+RECEIVED[ \t]+BY[ \t]+(?:THE[ \t]+)?CARRIER[ \t]*:?)",
    re.IGNORECASE,
)
_CARRIER_RECEIPT_HEADING = re.compile(r"\bCARRIER'?S[ \t]+RECEIPT\b", re.IGNORECASE)
_EQUIPMENT_BREAKDOWN_LINE = re.compile(
    r"^[ \t]*[0-9][0-9,]*[ \t]+X[ \t]+(?:20|40|45)'"
    r"(?:[ \t]*\+[ \t]*[0-9][0-9,]*[ \t]+X[ \t]+(?:20|40|45)')*[ \t]*$",
    re.IGNORECASE,
)
_RAW_AGENT_FOR_CARRIER_LINE = re.compile(
    r"^[ \t]*AS[ \t]+AGENTS?[ \t]+FOR[ \t]+THE[ \t]+CARRIER"
    r"(?:[ \t]+AND[ \t]+SERVICE[ \t]+PROVIDER)?[ \t:]+"
    r"(?P<principal>[^\r\n]{2,160})[ \t]*$",
    re.IGNORECASE,
)
_RAW_ISSUED_AGENT_BY_LINE = re.compile(
    r"^[ \t]*(?:\([0-9]+\)[ \t]+)?ISSUED[ \t]+AS[ \t]+AGENTS?[ \t]+FOR[ \t]+"
    r"(?P<principal>[^\r\n]{2,160}?)[ \t]+AS[ \t]+CARRIER[ \t]+BY:[ \t]*$",
    re.IGNORECASE,
)
_RAW_INLINE_AGENT_FOR_CARRIER_LINE = re.compile(
    r"^[ \t]*(?P<identity>[^\r\n]{2,160}?),?[ \t]+AS[ \t]+AGENTS?[ \t]+FOR[ \t]+"
    r"THE[ \t]+CARRIER(?:[ \t]+(?P<principal>[^\r\n]{2,160}))?[ \t]*$",
    re.IGNORECASE,
)
_RAW_SIGNED_ON_BEHALF_LINE = re.compile(
    r"^[ \t]*SIGNED[ \t]+ON[ \t]+BEHALF[ \t]+OF[ \t]+"
    r"(?:THE[ \t]+CARRIER[ \t]+)?(?P<principal>[^\r\n]{2,160})[ \t]*$",
    re.IGNORECASE,
)
_RAW_SIGNED_FOR_CARRIER_LINE = re.compile(
    r"^[ \t]*SIGNED[ \t]+FOR[ \t]+THE[ \t]+CARRIER[ \t:]+"
    r"(?P<principal>[^\r\n]{2,160}?)(?P<trailing_by>[ \t]+BY)?[ \t]*$",
    re.IGNORECASE,
)
_RAW_SHIPPED_ON_BOARD = re.compile(
    r"\bSHIPPED[ \t]+ON[ \t]+BOARD\b",
    re.IGNORECASE,
)
_RAW_SHIPPED_ON_BOARD_DATE = re.compile(
    r"(?:[0-3]?[0-9][./-](?:[A-Z]{3}|[01]?[0-9])[./-][0-9]{2,4}|"
    r"[0-9]{4}[./-][01]?[0-9][./-][0-3]?[0-9])",
    re.IGNORECASE,
)
_RAW_BY_AGENT_LINE = re.compile(
    r"^[ \t]*BY[ \t]+(?P<identity>[^\r\n]{2,160}?)[ \t]+AS[ \t]+AGENT"
    r"(?:[ \t]+[^\r\n]{1,160})?[ \t]*$",
    re.IGNORECASE,
)
_RAW_GROSS_WEIGHT = re.compile(
    r"(?<![A-Z0-9])(?:G\.?[ \t]*W\.?|GROSS[ \t]+(?:WEIGHT|WT\.?))"
    r"[ \t]*[:#-]?[ \t]*(?P<value>[0-9]+(?:[.,][0-9]+)*)",
    re.IGNORECASE,
)
_RAW_VOLUME = re.compile(
    r"(?<![A-Z0-9])(?:CBM|CUBIC[ \t]+MET(?:ER|RE)S?|VOLUME)"
    r"[ \t]*[:#-]?[ \t]*(?P<value>[0-9]+(?:[.,][0-9]+)*)",
    re.IGNORECASE,
)
_RAW_TRAILING_GROSS_WEIGHT = re.compile(
    r"(?<![0-9.,])(?P<value>[0-9]+(?:[.,][0-9]+)*)[ \t]*"
    r"(?:KGS?|KGM|KILOGRAMS?)(?![A-Z])",
    re.IGNORECASE,
)
_RAW_TRAILING_VOLUME = re.compile(
    r"(?<![0-9.,])(?P<value>[0-9]+(?:[.,][0-9]+)*)[ \t]*"
    r"(?:CBM|M(?:3|³)|CU\.?[ \t]*M\.?)\b",
    re.IGNORECASE,
)
_RAW_TARE_WEIGHT = re.compile(
    r"(?<![A-Z0-9])TARE(?:[ \t]+WEIGHT)?[ \t]*[:#-]?[ \t]*"
    r"(?P<value>[0-9]+(?:[.,][0-9]+)*)",
    re.IGNORECASE,
)
_RAW_PACKAGE_QUANTITY = re.compile(
    r"(?<![0-9.,])(?P<value>[0-9][0-9,]*(?:[.,]00)?)[ \t]*"
    r"(?:(?:WOODEN|WOOD|PLYWOOD|CARDBOARD|FIBREBOARD|FIBERBOARD|CORRUGATED|"
    r"PLASTIC|STEEL|METAL)[ \t]+)?"
    r"(?P<package>PACKAGES?|PKGS?|PCS?|PIECES?|PALLETS?|CARTONS?|DRUMS?|BAGS?|"
    r"BOX(?:ES)?|BALES?|ROLLS?|CRATES?|CASES?|BUNDLES?|SETS?|LOTS?|UNITS?|"
    r"SACKS?|JERRICANS?|TINS?|CANS?|BARRELS?|VEHICLES?)(?![A-Z])",
    re.IGNORECASE,
)
_RAW_PACKAGE_TYPE_SURFACE = re.compile(
    r"(?<![0-9.,])(?P<value>[0-9][0-9,]*(?:[.,]00)?)[ \t]*"
    r"(?P<package>"
    r"(?:(?:WOODEN|WOOD|PLYWOOD|CARDBOARD|FIBREBOARD|FIBERBOARD|CORRUGATED|"
    r"PLASTIC|STEEL|METAL)[ \t]+)?"
    r"(?:PACKAGES?|PKGS?|PKG|PCS?|PIECES?|PALLETS?|PLTS?|CARTONS?|CTNS?|"
    r"DRUMS?|BAGS?|BOX(?:ES)?|BALES?|ROLLS?|REELS?|CRATES?|CASES?|BUNDLES?|"
    r"SETS?|LOTS?|UNITS?|SACKS?|JERRICANS?|TINS?|CANS?|BARRELS?|VEHICLES?|"
    r"SHEETS?|IBCS?))(?P<parenthesized_plural>[ \t]*\((?:E)?S\))?(?![A-Z])",
    re.IGNORECASE,
)
_RAW_PACKAGE_NOUN_ONLY = re.compile(
    r"(?<![A-Z])(?P<package>"
    r"(?:(?:WOODEN|WOOD|PLYWOOD|CARDBOARD|FIBREBOARD|FIBERBOARD|CORRUGATED|"
    r"PLASTIC|STEEL|METAL)[ \t]+)?"
    r"(?:PACKAGES?|PKGS?|PKG|PCS?|PIECES?|PALLETS?|PLTS?|CARTONS?|CTNS?|"
    r"DRUMS?|BAGS?|BOX(?:ES)?|BALES?|ROLLS?|REELS?|CRATES?|CASES?|BUNDLES?|"
    r"SETS?|LOTS?|UNITS?|SACKS?|JERRICANS?|TINS?|CANS?|BARRELS?|VEHICLES?|"
    r"SHEETS?|IBCS?))(?:[ \t]*\((?:E)?S\))?(?![A-Z])",
    re.IGNORECASE,
)
_RAW_WRITTEN_PACKAGE_QUANTITY = re.compile(
    r"(?i)\bSAY[ \t]*:?[ \t]+(?P<value>[A-Z]+(?:[ -]+[A-Z]+)*)[ \t]+"
    r"(?:PACKAGES?|PKGS?)(?:\(S\))?(?![A-Z])"
)
_RAW_WRITTEN_TOTAL_PACKAGE_TYPE = re.compile(
    r"(?is)(?<![A-Z0-9])\(?[ \t]*TOTAL[ \t]+"
    r"(?P<written_count>[A-Z]+(?:[ \t\r\n-]+[A-Z]+){0,20}?)"
    r"[ \t\r\n]+(?P<package>"
    r"(?:(?:WOODEN|WOOD|PLYWOOD|CARDBOARD|FIBREBOARD|FIBERBOARD|CORRUGATED|"
    r"PLASTIC|STEEL|METAL)[ \t]+)?"
    r"(?:PACKAGES?|PKGS?|PKG|PCS?|PIECES?|PALLETS?|PLTS?|CARTONS?|CTNS?|"
    r"DRUMS?|BAGS?|BOX(?:ES)?|BALES?|ROLLS?|REELS?|CRATES?|CASES?|BUNDLES?|"
    r"SETS?|LOTS?|UNITS?|SACKS?|JERRICANS?|TINS?|CANS?|BARRELS?|VEHICLES?|"
    r"SHEETS?|IBCS?))(?P<parenthesized_plural>[ \t]*\((?:E)?S\))?[ \t]*ONLY\b"
)
_CARDINAL_SMALL = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
}
_CARDINAL_TENS = {
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
}
_CARDINAL_SCALES = {"THOUSAND": 1_000, "MILLION": 1_000_000}
_RAW_INLINE_CONTAINER_MEASURES = re.compile(
    r"\((?P<gross>[0-9]+(?:[.,][0-9]+)*)[ \t]*KG[ \t]*/[ \t]*"
    r"(?P<volume>[0-9]+(?:[.,][0-9]+)*)[ \t]*M3[ \t]*/[ \t]*"
    r"(?P<packages>[0-9][0-9,]*)[ \t]*PK\)",
    re.IGNORECASE,
)
_RAW_MEASUREMENT_NUMBER = re.compile(r"(?<![0-9.,])(?P<value>[0-9]+(?:[.,][0-9]+)*)(?![0-9.,])")
_RAW_DENSE_MEASUREMENT_LINE = re.compile(r"^[ \t]*(?P<value>[0-9]+(?:[.,][0-9]+)*)[ \t]*$")
_RAW_DENSE_MEASUREMENT_TUPLE = re.compile(
    r"^[ \t]*(?P<gross>[0-9]+(?:[.,][0-9]+)*)[ \t]+"
    r"(?P<tare>[0-9]+(?:[.,][0-9]+)*)[ \t]+"
    r"(?P<volume>[0-9]+(?:[.,][0-9]+)*)[ \t]*$"
)
OperationalKind = Literal[
    "gross_weight_kg",
    "tare_weight_kg",
    "volume_m3",
    "package_quantity",
]
OperationalGrammar = Literal[
    "labeled_measurement",
    "inline_container_breakdown",
    "dense_container_table",
]
_CONTAINER_EQUIPMENT_REVIEW_PATH = re.compile(
    r"documentPatch\.containers\[([0-9]+)\]\."
    r"(?:typeDescription|printedEquipmentSurface|sizeCategory|typeCategory)"
)
_ANCHORABLE_IDENTIFIER_PATH = re.compile(
    r"^documentPatch\.(?:"
    r"billOfLadingNumber|"
    r"forwardingAndExportReferences\[[0-9]+\]|"
    r"transport\.voyageNumber|"
    r"containers\[[0-9]+\]\.(?:containerNumber|sealNumbers\[[0-9]+\])|"
    r"cargoAllocationGroups\[[0-9]+\]\.allocations\[[0-9]+\]\.containerNumber|"
    r"cargoGroups\[[0-9]+\]\.marksAndNumbers\[[0-9]+\]|"
    r"cargoGroups\[[0-9]+\]\.dangerousGoods\[[0-9]+\]\.unNumber"
    r")$"
)
_MARKS_AND_NUMBERS_PATH = re.compile(
    r"^documentPatch\.cargoGroups\[[0-9]+\]\.marksAndNumbers\[[0-9]+\]$"
)
_INLINE_MARKS_VALUE_PREFIX = re.compile(
    r"(?:^|\b)(?:MKS|MARKS?)(?:\s*(?:&|AND|/)\s*(?:NOS?|NUMBERS?))?"
    r"\s*[:#=-]?\s*$",
    re.IGNORECASE,
)
_ANCHORABLE_CONTACT_PATH = re.compile(
    r"^documentPatch\.parties\.(?:shipper|consignee|carrier|deliveryAgent|"
    r"forwardingAgent|notifyParties\[[0-9]+\])\.contactDetails\."
    r"(?:contactName|phoneNumbers\[[0-9]+\]|emailAddresses\[[0-9]+\]|"
    r"websiteUrls\[[0-9]+\])$"
)
_SPECIAL_USE_TLDS = frozenset({"example", "invalid", "localhost", "test"})
_SPECIAL_USE_DOMAINS = frozenset(
    {
        "example.com",
        "example.net",
        "example.org",
        "localhost",
    }
)
_COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")
_DOMAIN_TOKEN = re.compile(r"[^a-z0-9]+")
_MONTHS = {
    "JAN": 1,
    "JANUARY": 1,
    "FEB": 2,
    "FEBRUARY": 2,
    "MAR": 3,
    "MARCH": 3,
    "APR": 4,
    "APRIL": 4,
    "MAY": 5,
    "JUN": 6,
    "JUNE": 6,
    "JUL": 7,
    "JULY": 7,
    "AUG": 8,
    "AUGUST": 8,
    "SEP": 9,
    "SEPT": 9,
    "SEPTEMBER": 9,
    "OCT": 10,
    "OCTOBER": 10,
    "NOV": 11,
    "NOVEMBER": 11,
    "DEC": 12,
    "DECEMBER": 12,
}


class RawTextRewriteCycleError(RuntimeError):
    """The atomic rewrite probe cannot establish its artifact contract."""


class LineRangeReplacement(BaseModel):
    """One inclusive line-ID range replaced in the current OCR text."""

    model_config = _STRICT

    startLineId: Annotated[
        str,
        StringConstraints(pattern=r"^L[0-9]{5}$"),
        Field(description="Exact ID preceding the first line to replace."),
    ]
    endLineId: Annotated[
        str,
        StringConstraints(pattern=r"^L[0-9]{5}$"),
        Field(description="Exact ID preceding the inclusive final line to replace."),
    ]
    newText: Annotated[
        str,
        StringConstraints(max_length=50_000),
        Field(
            description=(
                "Complete replacement text without a trailing newline. Emit exactly one output "
                "line per selected source line and preserve which selected lines are blank. "
                "Never use unavailable, unknown, or N/A placeholders; a target-absent assertion "
                "must be replaced in its occupied slot with coherent non-extractable flavor."
            )
        ),
    ]


class AppliedLineRangeReplacement(BaseModel):
    model_config = _STRICT

    startLine: Annotated[int, Field(ge=1)]
    endLine: Annotated[int, Field(ge=1)]
    startLineId: str
    endLineId: str
    oldText: str
    newText: str
    oldTextSha256: Sha256
    sourceLinesReplaced: Annotated[int, Field(ge=1)]
    outputLinesInserted: Annotated[int, Field(ge=0)]


class AppliedDeterministicPrefill(BaseModel):
    """One exact line-bound value or derived surface completed without a provider call."""

    model_config = _STRICT

    lineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    targetPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    beforeLine: str
    afterLine: str


class CompoundPartyFlavorRealization(BaseModel):
    """One target-primary identity plus fictional source-topology flavor."""

    model_config = _STRICT

    targetPath: Annotated[
        str,
        StringConstraints(
            pattern=(
                r"^documentPatch\.parties\."
                r"(?:shipper|consignee|carrier|forwardingAgent|deliveryAgent|consolidator)\.name$"
            )
        ),
        Field(description="Exact party-name path from compoundPartyFlavorRequirements."),
    ]
    targetPrimaryName: NonEmptyText
    renderedName: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=5, max_length=360),
        Field(
            description=(
                "Complete raw-text realization beginning with the exact target primary name, "
                "then preserving the source relationship with a distinct fictional identity. "
                "This auxiliary flavor never changes the target label."
            )
        ),
    ]


class AppliedCompoundPartyFlavorRealization(BaseModel):
    model_config = _STRICT

    targetPath: NonEmptyText
    relationships: tuple[Literal["on_behalf_of", "trading_as"], ...]
    targetPrimaryName: NonEmptyText
    renderedName: NonEmptyText


class CompoundPartyFlavorRequirement(BaseModel):
    model_config = _STRICT

    targetPath: NonEmptyText
    relationships: tuple[Literal["on_behalf_of", "trading_as"], ...] = Field(min_length=1)
    sourceLabelName: NonEmptyText
    targetPrimaryName: NonEmptyText


class LabelChangeDirective(BaseModel):
    """One authoritative model-facing semantic change between reviewed labels."""

    model_config = _STRICT

    path: NonEmptyText
    action: Literal[
        "replace",
        "deactivate_extractable_fact_preserve_slot",
        "add",
        "preserve_equivalent_equipment_surface",
        "replace_equipment_surface",
        "add_equipment_surface",
    ]
    sourceValue: JsonValue
    targetValue: JsonValue


class CompactLabelChangeDirective(BaseModel):
    """One deduplicated raw-text change shared by one or more schema paths."""

    model_config = _STRICT

    paths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    action: Literal[
        "replace",
        "deactivate_extractable_fact_preserve_slot",
        "add",
        "preserve_equivalent_equipment_surface",
        "replace_equipment_surface",
        "add_equipment_surface",
    ]
    sourceValue: JsonValue
    targetValue: JsonValue


class SurfaceRenderingRequirement(BaseModel):
    """One exact document surface derived from a canonical target value."""

    model_config = _STRICT

    kind: Literal[
        "date",
        "date_global",
        "hs_code",
        "hs_code_block",
        "cargo_origin",
        "dangerous_goods_tuple",
        "dangerous_goods_proper_shipping_name",
        "carrier_header_identity",
        "carrier_principal_identity",
        "carrier_receipt_count",
        "carrier_receipt_equipment_breakdown",
        "aggregate_equipment_breakdown",
    ]
    targetPath: NonEmptyText
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    sourceOccurrences: Annotated[int, Field(ge=1)]
    contextEvidence: Annotated[str, StringConstraints(min_length=1, max_length=600)]
    sourceLineIds: tuple[Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")], ...] = ()


class SourceSemanticRoleHint(BaseModel):
    """A deterministic role for flattened OCR text that resembles another field."""

    model_config = _STRICT

    role: Literal["anonymous_equipment_count", "anonymous_equipment_surface"]
    sourceLineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    sourceSurface: NonEmptyText
    requiredOutputSurface: NonEmptyText
    sourceEvidence: Annotated[str, StringConstraints(min_length=1, max_length=600)]
    forbiddenTargetPathPrefixes: tuple[NonEmptyText, ...]


class InlineSlotTopologyRequirement(BaseModel):
    """One labeled inline source slot whose occupied state is part of the template."""

    model_config = _STRICT

    lineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    ordinal: Annotated[int, Field(ge=0)]
    labelSurface: NonEmptyText
    sourcePopulated: bool


class SourceStatusPreservationRequirement(BaseModel):
    """One unchanged operational/document status that must remain byte-identical."""

    model_config = _STRICT

    sourceSurface: NonEmptyText
    sourceOccurrences: Annotated[int, Field(ge=1)]


class TargetLiteralRequirement(BaseModel):
    """One changed free-text target value that must occur in the rewritten OCR."""

    model_config = _STRICT

    targetPath: NonEmptyText
    targetValue: NonEmptyText
    matchPolicy: Literal[
        "semantic_literal",
        "alphanumeric_identifier",
        "ordered_semantic_atoms",
    ] = "semantic_literal"


class TargetValueOccurrenceRequirement(BaseModel):
    """Exact occurrence count for a changed scalar whose duplication changes semantics."""

    model_config = _STRICT

    targetPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    targetValue: NonEmptyText
    requiredOccurrences: Annotated[int, Field(ge=1)]


class AnchoredScalarReplacementRequirement(BaseModel):
    """Exact changed label scalar that must remain on every source line where it was printed."""

    model_config = _STRICT

    targetPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceLineIds: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    surfaceKind: Literal["scalar", "measurement", "package_noun"] = "scalar"


class RawAuxiliaryIdentityRequirement(BaseModel):
    """A raw-only agency identity that must be anonymized without changing its relationship."""

    model_config = _STRICT

    requirementId: NonEmptyText
    relationship: Literal["agent_for_carrier"]
    sourceIdentity: NonEmptyText
    sourceIdentityLineCount: Annotated[int, Field(ge=1, le=3)]
    gapLineCount: Annotated[int, Field(ge=0, le=2)]
    consistencyGroupId: NonEmptyText
    targetPrincipalName: NonEmptyText
    sourceEvidence: EvidenceText


class OperationalFlavorRequirement(BaseModel):
    """One exact container-row fact derived from the synthetic shipment semantics."""

    model_config = _STRICT

    requirementId: NonEmptyText
    kind: OperationalKind
    sourceGrammar: OperationalGrammar
    sourceLineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    sourceMeasurementStartColumn: Annotated[int, Field(ge=0)]
    sourceValueSurface: NonEmptyText
    targetValueSurface: NonEmptyText
    sourceCanonicalValue: NonEmptyText
    targetCanonicalValue: NonEmptyText
    consistencyGroupId: NonEmptyText
    targetContainerNumber: NonEmptyText
    targetEquipmentFamily: EquipmentFamily
    maximumValue: NonEmptyText | None
    sameLineFollowingContainerNumber: NonEmptyText | None
    empiricalProfileDocumentId: NonEmptyText | None
    samplingMethod: Literal[
        "empirical_capacity_utilization_resample_v1",
        "empirical_same_family_tare_resample_v1",
        "target_measure_allocation_v1",
        "target_package_allocation_v1",
        "target_membership_package_projection_v1",
        "target_single_container_aggregate_projection_v1",
    ]
    sourceEvidence: EvidenceText


class CargoFlavorRewriteRequirement(BaseModel):
    """Every occupied OCR line belonging to one changed cargo-description block."""

    model_config = _STRICT

    requirementId: NonEmptyText
    lineRole: Literal["label_grounded", "source_only_auxiliary_packaging"] = (
        "label_grounded"
    )
    targetPath: NonEmptyText
    targetDescription: NonEmptyText
    sourceLineIds: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceSurfaces: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    enforcePackageSurfaceGuard: bool = False
    allowedPackageSurfaces: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def line_ids_match_surfaces(self) -> CargoFlavorRewriteRequirement:
        if len(self.sourceLineIds) != len(self.sourceSurfaces):
            raise ValueError("cargo-flavor source line IDs and surfaces must have equal length")
        line_numbers = tuple(int(line_id[1:]) for line_id in self.sourceLineIds)
        if line_numbers != tuple(sorted(set(line_numbers))):
            raise ValueError("cargo-flavor source line IDs must be unique and ascending")
        normalized_allowed = tuple(
            sorted({_semantic_normalize(value) for value in self.allowedPackageSurfaces})
        )
        if normalized_allowed != tuple(
            _semantic_normalize(value) for value in self.allowedPackageSurfaces
        ):
            raise ValueError(
                "allowed cargo package surfaces must be normalized, unique, and sorted"
            )
        return self


class AppliedRawAuxiliaryIdentityRealization(BaseModel):
    model_config = _STRICT

    requirementId: NonEmptyText
    relationship: Literal["agent_for_carrier"]
    sourceIdentity: NonEmptyText
    sourceIdentityLineCount: Annotated[int, Field(ge=1, le=3)]
    gapLineCount: Annotated[int, Field(ge=0, le=2)]
    consistencyGroupId: NonEmptyText
    targetPrincipalName: NonEmptyText
    renderedIdentity: NonEmptyText


class TargetIntegrityChange(BaseModel):
    """One audited repair applied before the raw-text model sees a target."""

    model_config = _STRICT

    path: NonEmptyText
    reason: Literal[
        "special_use_domain",
        "country_incoherent_cctld",
        "package_surface_category_mismatch",
        "overlapping_source_scalar_topology",
        "shared_source_party_identity_topology",
        "unprinted_equipment_topology",
    ]
    before: NonEmptyText
    after: NonEmptyText


class RecoveredExplicitHsTargetFact(BaseModel):
    """One task-extractable HS value restored from pinned semantic-plan provenance."""

    model_config = _STRICT

    targetPath: NonEmptyText
    sourceEvidenceCodes: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    semanticPlanPath: NonEmptyText
    hs6: Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]
    renderedCode: Annotated[str, StringConstraints(pattern=r"^[0-9]{6,18}$")]
    outputDigits: Annotated[int, Field(ge=6, le=18)]
    method: Literal["explicit_raw_hs_slot_from_pinned_semantic_plan_v1"]


class CustomsProgramEntry(BaseModel):
    """One authoritative jurisdiction-bound document-program surface."""

    model_config = _STRICT

    program_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]+$")]
    official_name: NonEmptyText
    authority: NonEmptyText
    official_source_url: Annotated[str, StringConstraints(pattern=r"^https://[^\s]+$")]
    jurisdiction_country_code: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    trade_direction: Literal["import", "export"]
    source_surfaces: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    generic_replacement_surface: NonEmptyText

    @model_validator(mode="after")
    def surfaces_are_unique_and_longest_first(self) -> CustomsProgramEntry:
        # Punctuation is part of a customs selector (``ACID:`` is not the cargo word
        # ``acid``), so registry identity normalizes only case and whitespace.
        normalized = tuple(" ".join(value.casefold().split()) for value in self.source_surfaces)
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("customs-program source surfaces must be nonempty and unique")
        if tuple(sorted(self.source_surfaces, key=len, reverse=True)) != self.source_surfaces:
            raise ValueError("customs-program source surfaces must be longest-first")
        return self


class CustomsProgramRegistry(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    entries: Annotated[tuple[CustomsProgramEntry, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def entries_are_unique(self) -> CustomsProgramRegistry:
        identities = tuple(row.program_id for row in self.entries)
        if len(identities) != len(set(identities)):
            raise ValueError("customs-program registry IDs must be unique")
        surfaces = [
            (row.trade_direction, " ".join(surface.casefold().split()))
            for row in self.entries
            for surface in row.source_surfaces
        ]
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("customs-program surfaces overlap within a trade direction")
        return self


class JurisdictionalSurfaceRequirement(BaseModel):
    """A stale named customs program that must be generalized for a new route."""

    model_config = _STRICT

    requirementId: NonEmptyText
    programId: NonEmptyText
    tradeDirection: Literal["import", "export"]
    programJurisdictionCountryCode: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    targetRouteCountryCode: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    sourceLineIds: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    sourceOccurrences: Annotated[int, Field(ge=1)]
    authority: NonEmptyText
    officialSourceUrl: Annotated[str, StringConstraints(pattern=r"^https://[^\s]+$")]


@dataclass(frozen=True, slots=True)
class TargetIntegrityResources:
    countries: CountryRegistry
    packages: LoadedPackageRegistry
    route_countries_by_name: Mapping[str, frozenset[str]]
    route_port_countries_by_name: Mapping[str, frozenset[str]]
    customs_programs: tuple[CustomsProgramEntry, ...]
    operational_profiles: tuple[EmpiricalOperationalProfile, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedMeasurementSurface:
    value: Decimal
    decimal_separator: str | None
    grouping_separator: str | None
    decimal_places: int


@dataclass(frozen=True, slots=True)
class DenseAggregateMeasurementRow:
    """One uniquely anchored gross/tare/volume tuple owned by a shipment-total marker."""

    container_count: int
    value_line_indexes: tuple[int, int, int]
    value_spans: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    value_surfaces: tuple[str, str, str]
    evidence: str


@dataclass(frozen=True, slots=True)
class CargoPackageProjectionRow:
    """Source/target package semantics for one stable group/package identity."""

    key: tuple[str, str]
    target_index: int
    source_quantity: int
    target_quantity: int
    source_category: str
    target_category: str


@dataclass(frozen=True, slots=True)
class EmpiricalOperationalProfile:
    document_id: str
    container_number: str
    equipment_family: EquipmentFamily
    gross_weight_kg: Decimal | None
    gross_utilization: Decimal | None
    volume_m3: Decimal | None
    volume_utilization: Decimal | None
    tare_weight_kg: Decimal | None = None


class AtomicRewriteCommit(BaseModel):
    model_config = _STRICT

    beforeTextSha256: Sha256
    afterTextSha256: Sha256
    beforeTargetLabelSha256: Sha256
    afterTargetLabelSha256: Sha256
    compoundPartyFlavorRealizations: tuple[AppliedCompoundPartyFlavorRealization, ...]
    rawAuxiliaryIdentityRealizations: tuple[AppliedRawAuxiliaryIdentityRealization, ...]
    appliedReplacements: tuple[AppliedLineRangeReplacement, ...]


class ResidualCandidate(BaseModel):
    model_config = _STRICT

    targetPath: NonEmptyText
    sourceValue: JsonValue
    currentOccurrences: Annotated[int, Field(ge=1)]
    currentEvidence: EvidenceText


class DeterministicRewriteAudit(BaseModel):
    model_config = _STRICT

    currentTextSha256: Sha256
    textChanged: bool
    pageMarkersPreserved: bool
    newlineConventionPreserved: bool
    lineCountPreserved: bool
    blankLineTopologyPreserved: bool
    inlineSlotTopologyPreserved: bool
    structuralLinePrefixesPreserved: bool
    isolatedFormattingChangesAbsent: bool
    sourceStatusSurfacesPreserved: bool
    blankLineDepthBounded: bool
    noPlaceholderIntroduced: bool
    requiredSurfacesRendered: bool
    jurisdictionalSurfacesRendered: bool
    requiredTargetLiteralsRendered: bool
    targetValueOccurrencesExact: bool
    anchoredScalarReplacementsRendered: bool
    rawAuxiliaryIdentitiesReplaced: bool
    operationalFlavorRendered: bool
    deactivatedReeferOperationCoherent: bool
    cargoFlavorRewritten: bool
    residualCandidates: tuple[ResidualCandidate, ...]

    @property
    def core_passed(self) -> bool:
        return all(
            (
                self.textChanged,
                self.pageMarkersPreserved,
                self.newlineConventionPreserved,
                self.lineCountPreserved,
                self.blankLineTopologyPreserved,
                self.inlineSlotTopologyPreserved,
                self.structuralLinePrefixesPreserved,
                self.isolatedFormattingChangesAbsent,
                self.sourceStatusSurfacesPreserved,
                self.blankLineDepthBounded,
                self.noPlaceholderIntroduced,
                self.requiredSurfacesRendered,
                self.jurisdictionalSurfacesRendered,
                self.requiredTargetLiteralsRendered,
                self.targetValueOccurrencesExact,
                self.anchoredScalarReplacementsRendered,
                self.rawAuxiliaryIdentitiesReplaced,
                self.operationalFlavorRendered,
                self.deactivatedReeferOperationCoherent,
                self.cargoFlavorRewritten,
            )
        )


class SemanticReviewFinding(BaseModel):
    model_config = _STRICT

    category: Literal[
        "missing_or_wrong_target_fact",
        "stale_source_fact",
        "incomplete_auxiliary_anonymization",
        "incoherent_synthetic_flavor",
        "derived_fact_mismatch",
        "legal_relationship_damage",
        "format_or_layout_damage",
        "unnecessary_change",
    ]
    affectedPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1, max_length=8)]
    currentEvidence: Annotated[tuple[EvidenceText, ...], Field(min_length=1, max_length=4)]
    correction: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=800)
    ]


class SemanticReviewReceipt(BaseModel):
    """Compact independent judgment over a complete rewritten OCR sample."""

    model_config = _STRICT

    verdict: Literal["pass", "revise"]
    targetFacts: Literal["pass", "fail"]
    staleSourceFacts: Literal["pass", "fail"]
    auxiliaryFlavor: Literal["pass", "fail"]
    cargoAndOperationalRealism: Literal["pass", "fail"]
    legalTopology: Literal["pass", "fail"]
    formattingAndMinimality: Literal["pass", "fail"]
    findings: Annotated[tuple[SemanticReviewFinding, ...], Field(max_length=24)]

    @model_validator(mode="after")
    def verdict_matches_findings(self) -> SemanticReviewReceipt:
        checks = (
            self.targetFacts,
            self.staleSourceFacts,
            self.auxiliaryFlavor,
            self.cargoAndOperationalRealism,
            self.legalTopology,
            self.formattingAndMinimality,
        )
        if self.verdict == "pass" and (self.findings or any(row != "pass" for row in checks)):
            raise ValueError("pass requires every checklist item to pass and no findings")
        if self.verdict == "revise" and (not self.findings or all(row == "pass" for row in checks)):
            raise ValueError("revise requires a finding and at least one failed checklist item")
        return self


class RewriteStageRecord(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[17]
    sequence: Annotated[int, Field(ge=1)]
    stage: Literal["draft_editor", "correction_editor", "semantic_reviewer"]
    correctionCycle: Annotated[int, Field(ge=0, le=3)]
    providerModel: NonEmptyText
    reasoningEffort: NonEmptyText
    startedAt: datetime
    completedAt: datetime
    inputPayloadSha256: Sha256
    currentTextSha256Before: Sha256
    currentTextSha256After: Sha256
    editorCommit: AtomicRewriteCommit | None
    semanticReview: SemanticReviewReceipt | None
    deterministicAudit: DeterministicRewriteAudit | None
    usage: LinguisticUsageReceipt
    modelMessages: JsonValue
    errorType: NonEmptyText | None
    errorMessage: NonEmptyText | None


class RewriteCycleResult(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[17]
    caseNumber: Annotated[int, Field(ge=1)]
    documentId: NonEmptyText
    scenarioId: NonEmptyText
    status: Literal["quality_validated", "needs_review", "call_failed", "cost_limited"]
    reason: NonEmptyText
    sourceTextSha256: Sha256
    outputTextSha256: Sha256
    sourceLabelSha256: Sha256
    upstreamTargetLabelSha256: Sha256
    targetLabelSha256: Sha256
    targetIntegritySha256: Sha256
    targetCapacityReprojected: bool
    changedLeavesSha256: Sha256
    labelChangeContractSha256: Sha256
    editorOutputSchemaSha256: Sha256
    reviewerOutputSchemaSha256: Sha256
    correctionCycles: Annotated[int, Field(ge=0, le=3)]
    editorPasses: Annotated[int, Field(ge=0)]
    reviewPasses: Annotated[int, Field(ge=0)]
    compoundPartyFlavorRealizations: Annotated[int, Field(ge=0)]
    rawAuxiliaryIdentityRealizations: Annotated[int, Field(ge=0)]
    deterministicPrefills: Annotated[int, Field(ge=0)]
    replacements: Annotated[int, Field(ge=0)]
    sourceLinesReplaced: Annotated[int, Field(ge=0)]
    outputLinesInserted: Annotated[int, Field(ge=0)]
    durationMs: Annotated[float, Field(ge=0)]
    totalUsage: LinguisticUsageReceipt


class RewriteCycleBundle(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[17]
    result: RewriteCycleResult
    feature: dict[str, JsonValue]
    sourceLabel: dict[str, JsonValue]
    upstreamTargetLabel: dict[str, JsonValue]
    targetLabel: dict[str, JsonValue]
    targetIntegrity: dict[str, JsonValue]
    changedLeaves: tuple[ChangedLeaf, ...]
    labelChangeContract: tuple[LabelChangeDirective, ...]
    compactLabelChangeContract: tuple[CompactLabelChangeDirective, ...]
    surfaceRenderingRequirements: tuple[SurfaceRenderingRequirement, ...]
    targetLiteralRequirements: tuple[TargetLiteralRequirement, ...]
    targetValueOccurrenceRequirements: tuple[TargetValueOccurrenceRequirement, ...]
    anchoredScalarReplacementRequirements: tuple[AnchoredScalarReplacementRequirement, ...]
    sourceSemanticRoleHints: tuple[SourceSemanticRoleHint, ...]
    inlineSlotTopologyRequirements: tuple[InlineSlotTopologyRequirement, ...]
    sourceStatusPreservationRequirements: tuple[SourceStatusPreservationRequirement, ...]
    compoundPartyFlavorRequirements: tuple[CompoundPartyFlavorRequirement, ...]
    rawAuxiliaryIdentityRequirements: tuple[RawAuxiliaryIdentityRequirement, ...]
    operationalFlavorRequirements: tuple[OperationalFlavorRequirement, ...]
    cargoFlavorRewriteRequirements: tuple[CargoFlavorRewriteRequirement, ...]
    jurisdictionalSurfaceRequirements: tuple[JurisdictionalSurfaceRequirement, ...]
    sourceText: str
    outputText: str
    deterministicPrefills: tuple[AppliedDeterministicPrefill, ...]
    appliedReplacements: tuple[AppliedLineRangeReplacement, ...]
    deterministicAudits: tuple[DeterministicRewriteAudit, ...]
    semanticReviews: tuple[SemanticReviewReceipt, ...]
    stageRecords: tuple[RewriteStageRecord, ...]
    diff: str


@dataclass(slots=True)
class RewriteWorkspace:
    original_text: str
    current_text: str
    scenario_id: str = "test-scenario"
    source_label: dict[str, JsonValue] = field(default_factory=dict)
    upstream_target_label: dict[str, JsonValue] = field(default_factory=dict)
    current_target_label: dict[str, JsonValue] = field(default_factory=dict)
    surface_requirements: tuple[SurfaceRenderingRequirement, ...] = ()
    target_literal_requirements: tuple[TargetLiteralRequirement, ...] = ()
    target_value_occurrence_requirements: tuple[TargetValueOccurrenceRequirement, ...] = ()
    anchored_scalar_replacement_requirements: tuple[AnchoredScalarReplacementRequirement, ...] = ()
    source_role_hints: tuple[SourceSemanticRoleHint, ...] = ()
    inline_slot_requirements: tuple[InlineSlotTopologyRequirement, ...] = ()
    source_status_requirements: tuple[SourceStatusPreservationRequirement, ...] = ()
    jurisdictional_requirements: tuple[JurisdictionalSurfaceRequirement, ...] = ()
    raw_auxiliary_identity_requirements: tuple[RawAuxiliaryIdentityRequirement, ...] = ()
    operational_flavor_requirements: tuple[OperationalFlavorRequirement, ...] = ()
    cargo_flavor_rewrite_requirements: tuple[CargoFlavorRewriteRequirement, ...] = ()
    deterministic_prefills: list[AppliedDeterministicPrefill] = field(default_factory=list)
    commits: list[AtomicRewriteCommit] = field(default_factory=list)

    @property
    def current_sha256(self) -> str:
        return sha256_bytes(self.current_text.encode("utf-8"))

    @property
    def current_target_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.current_target_label))


@dataclass(slots=True)
class RewriteState:
    case_number: int
    document_id: str
    scenario_id: str
    feature: dict[str, JsonValue]
    source_label: dict[str, JsonValue]
    target_integrity: dict[str, JsonValue]
    workspace: RewriteWorkspace
    started_at: datetime
    stages: list[RewriteStageRecord] = field(default_factory=list)
    audits: list[DeterministicRewriteAudit] = field(default_factory=list)
    reviews: list[SemanticReviewReceipt] = field(default_factory=list)

    @property
    def target_label(self) -> dict[str, JsonValue]:
        return self.workspace.current_target_label

    @property
    def changed_leaves(self) -> tuple[ChangedLeaf, ...]:
        return rewrite_changed_leaves(self.source_label, self.target_label)


def _page_markers(value: str) -> tuple[str, ...]:
    return tuple(line for line in value.splitlines() if _PAGE_MARKER.fullmatch(line))


def _newline_convention(value: str) -> frozenset[str]:
    return frozenset(match.group(0) for match in _NEWLINE.finditer(value))


def _all_occurrences(text: str, needle: str) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    while True:
        offset = text.find(needle, cursor)
        if offset < 0:
            return tuple(offsets)
        offsets.append(offset)
        cursor = offset + len(needle)


def _evidence_occurs(evidence: str, text: str) -> bool:
    if (
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(evidence)}(?![A-Za-z0-9])",
            text,
        )
        is not None
    ):
        return True
    normalized_evidence = _semantic_normalize(evidence)
    if not normalized_evidence:
        return False
    normalized_text = _semantic_normalize(text)
    return (
        re.search(
            rf"(?<![A-Z0-9]){re.escape(normalized_evidence)}(?![A-Z0-9])",
            normalized_text,
        )
        is not None
    )


def _max_consecutive_blank_lines(value: str) -> int:
    maximum = current = 0
    for line in value.splitlines():
        if line.strip():
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _blank_line_topology(value: str) -> tuple[bool, ...]:
    """Return the source template's occupied-versus-blank line structure."""

    return tuple(not line.strip() for line in value.splitlines())


def _inline_slots(line: str) -> tuple[tuple[str, bool], ...]:
    matches = tuple(_INLINE_SLOT_LABEL.finditer(line))
    slots: list[tuple[str, bool]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        value = line[match.end() : end]
        slots.append((match.group(0), re.search(r"[A-Z0-9]", value, re.I) is not None))
    return tuple(slots)


def inline_slot_topology_requirements(raw_text: str) -> tuple[InlineSlotTopologyRequirement, ...]:
    """Describe inline contact slots without exposing a guessed semantic value."""

    requirements: list[InlineSlotTopologyRequirement] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        for ordinal, (label, populated) in enumerate(_inline_slots(line)):
            requirements.append(
                InlineSlotTopologyRequirement(
                    lineId=_line_id(line_number, line),
                    ordinal=ordinal,
                    labelSurface=label,
                    sourcePopulated=populated,
                )
            )
    return tuple(requirements)


def _inline_slot_topology_preserved(source: str, output: str) -> bool:
    return not _inline_slot_topology_mismatches(source, output)


def _inline_slot_topology_mismatches(source: str, output: str) -> tuple[str, ...]:
    source_lines = source.splitlines()
    output_lines = output.splitlines()
    if len(source_lines) != len(output_lines):
        return ("line-count-mismatch",)
    return tuple(
        f"L{line_number:05d}"
        for line_number, (source_line, output_line) in enumerate(
            zip(source_lines, output_lines, strict=True), start=1
        )
        if _inline_slots(source_line) != _inline_slots(output_line)
    )


def _structural_line_prefixes_preserved(source: str, output: str) -> bool:
    """Protect numbered clause/field prefixes while allowing their values to change."""

    return not _structural_line_prefix_mismatches(source, output)


def _structural_line_prefix_mismatches(source: str, output: str) -> tuple[str, ...]:
    """Return exact line IDs whose numbered template prefix changed."""

    source_lines = source.splitlines()
    output_lines = output.splitlines()
    if len(source_lines) != len(output_lines):
        return ("line-count-mismatch",)
    mismatches: list[str] = []
    for line_number, (source_line, output_line) in enumerate(
        zip(source_lines, output_lines, strict=True), start=1
    ):
        source_match = _STRUCTURAL_LINE_PREFIX.match(source_line)
        if source_match is None:
            continue
        output_match = _STRUCTURAL_LINE_PREFIX.match(output_line)
        if output_match is None or output_match.group("prefix") != source_match.group("prefix"):
            mismatches.append(f"L{line_number:05d}")
    return tuple(mismatches)


def _numeric_surface_matches_value(surface: str, expected: int | float) -> bool:
    """Compare a signed OCR number with a numeric label value without losing its sign."""

    normalized = surface.strip().replace("\N{MINUS SIGN}", "-")
    sign = Decimal(-1) if normalized.startswith("-") else Decimal(1)
    unsigned = normalized.lstrip("+-")
    parsed = _parse_measurement_surface(unsigned, maximum=None)
    return parsed is not None and sign * parsed.value == Decimal(str(expected))


def _isolated_formatting_changes(
    workspace: RewriteWorkspace, output: str
) -> tuple[tuple[int, str, str], ...]:
    """Find punctuation/whitespace edits isolated from any lexical value change.

    Synthetic rendering may replace words, identifiers, and numbers, but it must not
    opportunistically clean up punctuation elsewhere on the same mostly-preserved line.
    Fully regenerated party and cargo values are exempt. A formatting-only diff is
    legitimate only when no unchanged
    alphanumeric token separates it from a lexical value edit.
    """

    source_lines = workspace.original_text.splitlines()
    output_lines = output.splitlines()
    if len(source_lines) != len(output_lines):
        return ((0, "line-count-mismatch", "line-count-mismatch"),)
    isolated: list[tuple[int, str, str]] = []
    for line_number, (source_line, output_line) in enumerate(
        zip(source_lines, output_lines, strict=True), start=1
    ):
        if source_line == output_line:
            continue
        source_comparison = source_line
        output_comparison = output_line
        controlled_value_line = False
        for leaf in rewrite_changed_leaves(workspace.source_label, workspace.current_target_label):
            if (
                isinstance(leaf.sourceValue, (int, float))
                and not isinstance(leaf.sourceValue, bool)
                and isinstance(leaf.targetValue, (int, float))
                and not isinstance(leaf.targetValue, bool)
            ):
                source_match = next(
                    (
                        match
                        for match in re.finditer(
                            r"(?<![0-9.,])[-+]?\d+(?:[.,]\d+)*(?![0-9.,])",
                            source_comparison,
                        )
                        if _numeric_surface_matches_value(match.group(0), leaf.sourceValue)
                    ),
                    None,
                )
                target_match = next(
                    (
                        match
                        for match in re.finditer(
                            r"(?<![0-9.,])[-+]?\d+(?:[.,]\d+)*(?![0-9.,])",
                            output_comparison,
                        )
                        if _numeric_surface_matches_value(match.group(0), leaf.targetValue)
                    ),
                    None,
                )
                if source_match is not None and target_match is not None:
                    source_comparison = (
                        source_comparison[: source_match.start()]
                        + "VALUE"
                        + source_comparison[source_match.end() :]
                    )
                    output_comparison = (
                        output_comparison[: target_match.start()]
                        + "VALUE"
                        + output_comparison[target_match.end() :]
                    )
                continue
            if not isinstance(leaf.sourceValue, str) or not isinstance(leaf.targetValue, str):
                continue
            source_line_semantic = _semantic_normalize(source_comparison)
            output_line_semantic = _semantic_normalize(output_comparison)
            source_value_semantic = _semantic_normalize(leaf.sourceValue)
            target_value_semantic = _semantic_normalize(leaf.targetValue)
            # A long physical-line fragment of one changed free-text scalar is itself the value
            # slot.  Punctuation such as STRETCH-WRAPPED is target content, not template cleanup.
            if (
                len(source_line_semantic) >= 12
                and len(output_line_semantic) >= 12
                and source_line_semantic in source_value_semantic
                and output_line_semantic in target_value_semantic
            ):
                controlled_value_line = True
                break
            if (
                len(_semantic_normalize(leaf.sourceValue)) >= 4
                and leaf.sourceValue in source_comparison
            ) or (
                len(_semantic_normalize(leaf.targetValue)) >= 4
                and leaf.targetValue in output_comparison
            ):
                controlled_value_line = True
                break
            if leaf.sourceValue in source_comparison:
                source_comparison = source_comparison.replace(leaf.sourceValue, "VALUE")
            if leaf.targetValue in output_comparison:
                output_comparison = output_comparison.replace(leaf.targetValue, "VALUE")
        if controlled_value_line:
            continue
        matcher = SequenceMatcher(None, source_comparison, output_comparison, autojunk=False)
        opcodes = matcher.get_opcodes()
        equal_alphanumeric = sum(
            sum(character.isalnum() for character in source_comparison[source_start:source_end])
            for tag, source_start, source_end, _, _ in opcodes
            if tag == "equal"
        )
        source_alphanumeric = sum(character.isalnum() for character in source_comparison)
        output_alphanumeric = sum(character.isalnum() for character in output_comparison)
        # A line that remains at least 90% the same is an existing prose template with sparse
        # value slots. The integer ratio keeps this check deterministic across Python versions.
        # Completely regenerated party/cargo values may legitimately use different punctuation.
        if (
            equal_alphanumeric * 10 < source_alphanumeric * 9
            or equal_alphanumeric * 10 < output_alphanumeric * 9
        ):
            continue
        changed = [row for row in opcodes if row[0] != "equal"]
        lexical = [
            row
            for row in changed
            if re.search(
                r"[A-Za-z0-9]",
                source_comparison[row[1] : row[2]] + output_comparison[row[3] : row[4]],
            )
        ]
        for _, source_start, source_end, output_start, output_end in changed:
            before = source_comparison[source_start:source_end]
            after = output_comparison[output_start:output_end]
            if re.search(r"[A-Za-z0-9]", before + after):
                continue
            associated = any(
                not re.search(
                    r"[A-Za-z0-9]",
                    source_comparison[min(source_end, row[2]) : max(source_start, row[1])]
                    + output_comparison[min(output_end, row[4]) : max(output_start, row[3])],
                )
                for row in lexical
            )
            if not associated:
                isolated.append((line_number, before, after))
    return tuple(isolated)


def _label_has_temperature_setpoint(label: Mapping[str, Any]) -> bool:
    patch = label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    return isinstance(containers, list) and any(
        isinstance(container, Mapping) and container.get("temperatureSetpoint") is not None
        for container in containers
    )


def _deactivated_reefer_operation_coherent(workspace: RewriteWorkspace, output: str) -> bool:
    """Reject positive source operation claims after every target setpoint is removed."""

    if (
        _NON_OPERATING_REEFER.search(output) is not None
        and re.search(r"\bPLUGGING[ \t]+FOR[ \t]+(?:THE[ \t]+)?ACCOUNT\b", output, re.IGNORECASE)
        is not None
    ):
        return False
    source_has_setpoint = _label_has_temperature_setpoint(workspace.source_label)
    target_has_setpoint = _label_has_temperature_setpoint(workspace.current_target_label)
    if not source_has_setpoint or target_has_setpoint:
        return True
    source_lines = workspace.original_text.splitlines()
    output_lines = output.splitlines()
    if len(source_lines) != len(output_lines):
        return False
    return all(
        _POSITIVE_REEFER_OPERATION.search(output_line) is None
        for source_line, output_line in zip(source_lines, output_lines, strict=True)
        if _POSITIVE_REEFER_OPERATION.search(source_line) is not None
    )


def source_status_preservation_requirements(
    raw_text: str, leaves: Sequence[ChangedLeaf]
) -> tuple[SourceStatusPreservationRequirement, ...]:
    """Protect unchanged generic status lines while allowing target-controlled changes."""

    changed_source_values = tuple(
        (leaf.path, leaf.sourceValue, _semantic_normalize(leaf.sourceValue))
        for leaf in leaves
        if leaf.requiresTextEdit
        and leaf.sourcePresent
        and isinstance(leaf.sourceValue, str)
        and leaf.sourceValue != leaf.targetValue
    )
    legal_document_semantics_changed = any(
        leaf.requiresTextEdit
        and leaf.sourceValue != leaf.targetValue
        and leaf.path.rsplit(".", 1)[-1] in {"negotiability", "numberOfOriginals"}
        for leaf in leaves
    )
    counts: dict[str, int] = {}
    for line in raw_text.splitlines():
        status_line = _PRESERVABLE_STATUS_LINE.fullmatch(line) is not None
        legal_line = (
            not legal_document_semantics_changed
            and _PRESERVABLE_LEGAL_BOILERPLATE_LINE.fullmatch(line) is not None
        )
        party_heading = (
            not legal_document_semantics_changed and _PARTY_HEADING_LINE.fullmatch(line) is not None
        )
        if not status_line and not legal_line and not party_heading:
            continue
        normalized = _semantic_normalize(line)
        if legal_line:
            # Legal clauses are template text, but a clause can also contain a mutable named
            # principal (for example, ``The legal provider ACME ... does business as ...``).
            # Protecting that complete line would make anonymization impossible.  The one party
            # value that commonly occurs as legal grammar is ``TO ORDER``; preserve it only when
            # it appears in the clause's explicit consignment syntax.  Every other changed
            # source scalar releases the line to the normal path-owned rewrite contract.
            changed_nonstatus = any(
                surface
                and surface in normalized
                and not (
                    path.startswith("documentPatch.parties.")
                    and surface == "TO ORDER"
                    and re.search(r"\bCONSIGNED[ \t]+TO[ \t]+ORDER\b", line, re.I)
                    is not None
                )
                for path, _source, surface in changed_source_values
            )
            if changed_nonstatus:
                legal_line = False
        if party_heading or legal_line:
            # A party value can occur inside heading metadata (for example ``TO ORDER`` in a
            # consignee negotiability caption) or immutable B/L boilerplate. That is template/
            # legal wording, not the changed party value printed in the semantic field. A real
            # negotiability/original-count change disables ``legal_line`` above and still makes
            # the clause available to the corresponding document-semantic rewrite.
            counts[line] = counts.get(line, 0) + 1
            continue
        if any(
            surface and surface in normalized
            for _path, _source, surface in changed_source_values
        ):
            continue
        counts[line] = counts.get(line, 0) + 1
    return tuple(
        SourceStatusPreservationRequirement(
            sourceSurface=surface,
            sourceOccurrences=occurrences,
        )
        for surface, occurrences in sorted(counts.items())
    )


def _source_status_surfaces_preserved(
    value: str, requirements: Sequence[SourceStatusPreservationRequirement]
) -> bool:
    lines = value.splitlines()
    return all(lines.count(row.sourceSurface) == row.sourceOccurrences for row in requirements)


def _placeholder_count(value: str) -> int:
    return sum(1 for _ in _PLACEHOLDER.finditer(value))


def _line_body(value: str) -> str:
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith(("\r", "\n")):
        return value[:-1]
    return value


def _line_ending(value: str) -> str:
    if value.endswith("\r\n"):
        return "\r\n"
    if value.endswith("\r"):
        return "\r"
    if value.endswith("\n"):
        return "\n"
    return ""


def _line_id(index: int, value: str) -> str:
    del value
    return f"L{index:05d}"


def _line_number(line_id: str) -> int:
    return int(line_id[1:6])


def _default_newline(value: str) -> str:
    conventions = _newline_convention(value)
    if len(conventions) > 1:
        raise ValueError("mixed newline conventions are not supported by line-range editing")
    return next(iter(conventions), "\n")


def indexed_ocr_lines(value: str) -> str:
    """Return compact line addresses without changing the underlying OCR."""

    return "\n".join(
        f"{_line_id(index, line)}|{_line_body(line)}"
        for index, line in enumerate(value.splitlines(keepends=True), start=1)
    )


def editable_indexed_ocr_lines(value: str) -> str:
    """Expose only occupied, non-marker lines while retaining absolute line addresses.

    Blank lines and page markers are immutable template structure. Omitting them from the editor's
    model-visible workspace prevents accidental selection without hiding adjacency: gaps in the
    absolute IDs show exactly where protected lines occur.
    """

    return "\n".join(
        f"{_line_id(index, line)}|{body}"
        for index, line in enumerate(value.splitlines(keepends=True), start=1)
        if (body := _line_body(line)).strip() and _PAGE_MARKER.fullmatch(body) is None
    )


def _semantic_normalize(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[A-Z0-9]+", ascii_value.upper()))


def _introduced_html_entities(source: str, output: str) -> tuple[str, ...]:
    """Return markup escapes introduced into a plain-text OCR transcript."""

    remaining = Counter(match.group(0).casefold() for match in _HTML_ENTITY.finditer(output))
    remaining.subtract(match.group(0).casefold() for match in _HTML_ENTITY.finditer(source))
    return tuple(
        sorted(entity for entity, count in remaining.items() for _ in range(max(0, count)))
    )


def _party_scalar_occurrence_count(text: str, surface: str) -> int:
    """Count printed party scalars without mistaking email/URL tokens for identities.

    Party names commonly reappear in email domains (for example, ``MLH SHIPPING`` and
    ``ops@mlh-shipping.com``).  Domain tokens are contact metadata, not an additional printed
    party occurrence.  Remove only those electronic-contact spans, then count the complete scalar
    with tolerant OCR punctuation boundaries across the document. Counting after whitespace
    normalization also recognizes a name or address that the source OCR wraps over several lines.
    """

    normalized_surface = _semantic_normalize(surface.replace("&", " AND "))
    if not normalized_surface:
        return 0
    pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(normalized_surface)}(?![A-Z0-9])")
    # Structural field headings can contain a party value as legal guidance rather than data.
    # ``CONSIGNEE (3) (NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER)`` must not count as a second
    # occurrence of the actual value ``TO ORDER`` printed on the following line.  Drop only
    # complete, grammar-recognized heading lines; party names that merely contain words such as
    # ``Consignee`` remain countable.
    data_text = "\n".join(
        line for line in text.splitlines() if _PARTY_HEADING_LINE.fullmatch(line) is None
    )
    normalized_text = _semantic_normalize(
        _ELECTRONIC_CONTACT_SURFACE.sub(" ", data_text).replace("&", " AND ")
    )
    return sum(1 for _ in pattern.finditer(normalized_text))


def _party_scalar_occurrence_line_sets(text: str, surface: str) -> tuple[frozenset[int], ...]:
    """Locate each party-scalar occurrence without losing cross-line provenance.

    The cardinality guard intentionally normalizes OCR whitespace, so a party value split over
    two physical lines counts as one occurrence.  Repair selection needs the inverse mapping back
    to every physical line; searching one line at a time cannot find such a duplicate and can
    repeatedly ask the model to rewrite the already-correct role block instead.  Tokenizing each
    retained line with the exact same normalization/contact/heading policy preserves that mapping.
    """

    target_tokens = tuple(_semantic_normalize(surface.replace("&", " AND ")).split())
    if not target_tokens:
        return ()
    tokens: list[tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _PARTY_HEADING_LINE.fullmatch(line) is not None:
            continue
        scrubbed = _ELECTRONIC_CONTACT_SURFACE.sub(" ", line).replace("&", " AND ")
        tokens.extend((token, number) for token in _semantic_normalize(scrubbed).split())
    width = len(target_tokens)
    return tuple(
        frozenset(line_number for _, line_number in tokens[start : start + width])
        for start in range(0, len(tokens) - width + 1)
        if tuple(token for token, _ in tokens[start : start + width]) == target_tokens
    )


_CARGO_MATCH_STOPWORDS = frozenset(
    {
        "AND",
        "AS",
        "BY",
        "CARGO",
        "DESCRIPTION",
        "FOR",
        "FROM",
        "GOODS",
        "IN",
        "ITEM",
        "OF",
        "ON",
        "OR",
        "PRODUCT",
        "PRODUCTS",
        "THE",
        "TO",
        "WITH",
    }
)
_CARGO_QUANTITY_CONTINUATION = re.compile(
    r"^[ \t]*[0-9][0-9., /-]*[ \t]+(?:EA|PCS?|PIECES?|PKGS?|PACKAGES?|"
    r"BAGS?|BALES?|BOXES?|CARTONS?|CASES?|CRATES?|DRUMS?|PALLETS?|ROLLS?|SETS?|"
    r"UNITS?)[ \t]*$",
    re.IGNORECASE,
)


def _cargo_match_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _semantic_normalize(value).split()
        if token not in _CARGO_MATCH_STOPWORDS
        and len(token) >= 3
        and any("A" <= ch <= "Z" for ch in token)
    )


def _line_matches_cargo_description(line: str, source_description: str) -> bool:
    """Match a source-description fragment without relying on a document-template regex."""

    line_normalized = _semantic_normalize(line)
    source_normalized = _semantic_normalize(source_description)
    if not line_normalized or not source_normalized:
        return False
    source_in_line = re.search(
        rf"(?<![A-Z0-9]){re.escape(source_normalized)}(?![A-Z0-9])",
        line_normalized,
    )
    line_in_source = re.search(
        rf"(?<![A-Z0-9]){re.escape(line_normalized)}(?![A-Z0-9])",
        source_normalized,
    )
    if len(source_normalized) >= 4 and (source_in_line is not None or line_in_source is not None):
        return True
    line_tokens = _cargo_match_tokens(line)
    source_tokens = _cargo_match_tokens(source_description)
    if not line_tokens or not source_tokens:
        return False
    shared = set(line_tokens) & set(source_tokens)
    if len(shared) >= 2:
        return True
    source_bigrams = set(pairwise(source_tokens))
    line_bigrams = set(pairwise(line_tokens))
    if source_bigrams & line_bigrams:
        return True
    longest = SequenceMatcher(
        None, line_normalized, source_normalized, autojunk=False
    ).find_longest_match()
    return longest.size >= 10 and longest.size / len(line_normalized) >= 0.60


def _cargo_cluster_score(
    lines: Sequence[str], source_description: str, start: int, end: int
) -> float:
    candidate = " ".join(lines[start : end + 1])
    source_normalized = _semantic_normalize(source_description)
    candidate_normalized = _semantic_normalize(candidate)
    source_tokens = set(_cargo_match_tokens(source_description))
    candidate_tokens = set(_cargo_match_tokens(candidate))
    shared = source_tokens & candidate_tokens
    precision = len(shared) / len(candidate_tokens) if candidate_tokens else 0.0
    recall = len(shared) / len(source_tokens) if source_tokens else 0.0
    token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    character_ratio = SequenceMatcher(
        None, candidate_normalized, source_normalized, autojunk=False
    ).ratio()
    containment = float(
        source_normalized in candidate_normalized or candidate_normalized in source_normalized
    )
    return (0.55 * token_f1) + (0.35 * character_ratio) + (0.10 * containment)


def _cargo_candidate_clusters(
    lines: Sequence[str], source_description: str
) -> tuple[tuple[int, int, float], ...]:
    fuzzy_hits = [
        index
        for index, line in enumerate(lines)
        if line.strip()
        and _PAGE_MARKER.fullmatch(line) is None
        and _line_matches_cargo_description(line, source_description)
    ]
    normalized_description = _semantic_normalize(source_description)
    informative_token_count = len(_cargo_match_tokens(source_description))
    # When a sufficiently descriptive source value occurs literally, each complete occurrence is
    # its own cargo-row anchor. Fuzzy matching can otherwise collapse adjacent near-identical rows
    # (for example model 966 F and model 966 D) into one span and assign both package relations to
    # the first group. Short generic descriptions retain the scored fuzzy path below because an
    # exact token such as ``FABRIC`` may be a heading rather than a product row.
    exact_hits = [
        index
        for index in fuzzy_hits
        if informative_token_count >= 2
        and normalized_description in _semantic_normalize(lines[index])
    ]
    clusters: list[list[int]] = [[hit] for hit in exact_hits]
    if clusters:
        cluster_by_anchor = {cluster[0]: cluster for cluster in clusters}
        for fragment in fuzzy_hits:
            if fragment in cluster_by_anchor:
                continue
            fragment_surface = _semantic_normalize(lines[fragment])
            if not fragment_surface or fragment_surface not in normalized_description:
                continue
            anchor = min(exact_hits, key=lambda hit: (abs(fragment - hit), hit))
            if abs(fragment - anchor) > 3:
                continue
            between = range(min(fragment, anchor) + 1, max(fragment, anchor))
            if any(
                not lines[index].strip() or _PAGE_MARKER.fullmatch(lines[index]) is not None
                for index in between
            ):
                continue
            cluster_by_anchor[anchor].append(fragment)
        clusters = [sorted(cluster) for cluster in clusters]
    if not clusters:
        for hit in fuzzy_hits:
            if (
                not clusters
                or hit - clusters[-1][-1] > 3
                or any(
                    not lines[index].strip()
                    for index in range(clusters[-1][-1] + 1, hit)
                )
                or any(
                    _PAGE_MARKER.fullmatch(lines[index]) is not None
                    for index in range(clusters[-1][-1] + 1, hit)
                )
            ):
                clusters.append([hit])
            else:
                clusters[-1].append(hit)
    candidates: list[tuple[int, int, float]] = []
    for cluster in clusters:
        start = cluster[0]
        end = cluster[-1]
        while start > 0 and _CARGO_QUANTITY_CONTINUATION.fullmatch(lines[start - 1]):
            start -= 1
        while end + 1 < len(lines) and _CARGO_QUANTITY_CONTINUATION.fullmatch(lines[end + 1]):
            end += 1
        candidates.append((start, end, _cargo_cluster_score(lines, source_description, start, end)))
    return tuple(candidates)


def _cargo_group_description_pairs(
    source_label: Mapping[str, Any], target_label: Mapping[str, Any]
) -> tuple[tuple[int, str, str], ...]:
    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    if not isinstance(source_patch, Mapping) or not isinstance(target_patch, Mapping):
        return ()
    source_groups = source_patch.get("cargoGroups")
    target_groups = target_patch.get("cargoGroups")
    if not isinstance(source_groups, Sequence) or isinstance(source_groups, (str, bytes)):
        return ()
    if not isinstance(target_groups, Sequence) or isinstance(target_groups, (str, bytes)):
        return ()
    target_by_group_id = {
        row.get("groupId"): row
        for row in target_groups
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    pairs: list[tuple[int, str, str]] = []
    for index, source_group in enumerate(source_groups):
        if not isinstance(source_group, Mapping):
            continue
        group_id = source_group.get("groupId")
        target_group = target_by_group_id.get(group_id)
        if target_group is None and index < len(target_groups):
            candidate = target_groups[index]
            target_group = candidate if isinstance(candidate, Mapping) else None
        if target_group is None:
            continue
        source_description = source_group.get("description")
        target_description = target_group.get("description")
        if not isinstance(source_description, str) or not isinstance(target_description, str):
            continue
        if _semantic_normalize(source_description) == _semantic_normalize(target_description):
            continue
        pairs.append((index, source_description, target_description))
    return tuple(pairs)


def cargo_flavor_rewrite_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[CargoFlavorRewriteRequirement, ...]:
    """Locate complete source cargo-description spans that must be rewritten.

    Reviewed labels commonly concatenate a visual cargo table while the OCR keeps product rows and
    quantity continuations on separate lines. Strong lexical anchors locate the rows; only occupied
    intervening slots are included. This prevents an editor from changing the headline while
    leaving a trailing source product row or quantity behind.
    """

    lines = raw_text.splitlines()
    pairs = _cargo_group_description_pairs(source_label, target_label)
    candidates_by_group = {
        group_index: _cargo_candidate_clusters(lines, source_description)
        for group_index, source_description, _ in pairs
    }
    assigned: dict[int, tuple[int, int]] = {}
    claimed: set[tuple[int, int]] = set()

    repeated_descriptions: dict[str, list[int]] = {}
    for group_index, source_description, _ in pairs:
        repeated_descriptions.setdefault(_semantic_normalize(source_description), []).append(
            group_index
        )
    for group_indices in repeated_descriptions.values():
        if len(group_indices) < 2:
            continue
        # A short generic form heading (for example ``FABRIC``) can lexically match every group
        # while the actual repeated cargo rows are printed later.  Selecting the earliest spans
        # assigns the heading to the first group and shifts every remaining group onto the wrong
        # row.  Retain the strongest N distinct candidates, then restore document order so label
        # order maps to physical row order.  Ties are stable and exact; insufficient distinct
        # evidence remains unassigned and therefore fails closed downstream.
        scored_by_span: dict[tuple[int, int], float] = {}
        for group_index in group_indices:
            for start, end, score in candidates_by_group[group_index]:
                scored_by_span[(start, end)] = max(score, scored_by_span.get((start, end), -1.0))
        strongest = sorted(
            scored_by_span,
            key=lambda span: (-scored_by_span[span], span[0], span[1]),
        )[: len(group_indices)]
        available = sorted(strongest)
        for group_index, span in zip(sorted(group_indices), available, strict=False):
            assigned[group_index] = span
            claimed.add(span)

    scored_options = sorted(
        (
            (score, group_index, start, end)
            for group_index, candidates in candidates_by_group.items()
            if group_index not in assigned
            for start, end, score in candidates
        ),
        key=lambda row: (-row[0], row[1], row[2], row[3]),
    )
    for _, group_index, start, end in scored_options:
        span = (start, end)
        if group_index in assigned or span in claimed:
            continue
        assigned[group_index] = span
        claimed.add(span)

    target_description_by_group = {
        group_index: target_description for group_index, _, target_description in pairs
    }
    requirements: list[CargoFlavorRewriteRequirement] = []
    for group_index, (start, end) in sorted(assigned.items()):
        source_indices = tuple(
            index
            for index in range(start, end + 1)
            if lines[index].strip() and _PAGE_MARKER.fullmatch(lines[index]) is None
        )
        if not source_indices:
            continue
        requirements.append(
            CargoFlavorRewriteRequirement(
                requirementId=f"cargo-group-{group_index + 1}-span-1",
                targetPath=f"documentPatch.cargoGroups[{group_index}].description",
                targetDescription=target_description_by_group[group_index],
                sourceLineIds=tuple(_line_id(index + 1, lines[index]) for index in source_indices),
                sourceSurfaces=tuple(lines[index] for index in source_indices),
            )
        )
    return tuple(requirements)


def _cargo_flavor_rewrite_requirements_rendered(
    value: str, requirements: Sequence[CargoFlavorRewriteRequirement]
) -> bool:
    return not _cargo_flavor_rewrite_failures(value, requirements)


def _cargo_flavor_rewrite_failures(
    value: str, requirements: Sequence[CargoFlavorRewriteRequirement]
) -> tuple[dict[str, JsonValue], ...]:
    """Describe only the cargo lines that still violate the rewrite contract.

    Corrective model calls are deliberately line-local. Returning an entire multi-line cargo
    requirement here caused a single unchanged auxiliary product line to regenerate every already
    accepted line in the group. The structured diagnostics retain enough information to select an
    exact repair surface while still falling back to the complete target path when the target
    description itself is absent and no source line remains unchanged.
    """

    lines = value.splitlines()
    normalized_value = _semantic_normalize(value)
    failures: list[dict[str, JsonValue]] = []
    for requirement in requirements:
        target_missing = _semantic_normalize(requirement.targetDescription) not in normalized_value
        missing_line_ids: list[str] = []
        unchanged_source_lines: list[dict[str, JsonValue]] = []
        unexpected_package_surfaces: list[dict[str, JsonValue]] = []
        for line_id, source_surface in zip(
            requirement.sourceLineIds, requirement.sourceSurfaces, strict=True
        ):
            line_number = _line_number(line_id)
            if line_number > len(lines):
                missing_line_ids.append(line_id)
                continue
            if _semantic_normalize(lines[line_number - 1]) == _semantic_normalize(source_surface):
                unchanged_source_lines.append(
                    {
                        "lineId": line_id,
                        "sourceSurface": source_surface,
                    }
                )
            if requirement.enforcePackageSurfaceGuard:
                for package_surface in unexpected_cargo_package_surfaces(
                    lines[line_number - 1], requirement
                ):
                    unexpected_package_surfaces.append(
                        {
                            "lineId": line_id,
                            "surface": package_surface,
                            "allowedSurfaces": list(requirement.allowedPackageSurfaces),
                        }
                    )
        if (
            target_missing
            or missing_line_ids
            or unchanged_source_lines
            or unexpected_package_surfaces
        ):
            failure: dict[str, JsonValue] = {
                "requirementId": requirement.requirementId,
                "targetPath": requirement.targetPath,
                "targetDescriptionMissing": target_missing,
                "missingLineIds": cast(JsonValue, missing_line_ids),
                "unchangedSourceLines": cast(JsonValue, unchanged_source_lines),
            }
            if unexpected_package_surfaces:
                failure["unexpectedPackageSurfaces"] = cast(
                    JsonValue,
                    unexpected_package_surfaces,
                )
            failures.append(failure)
    return tuple(failures)


def rewrite_changed_leaves(
    source: Mapping[str, JsonValue], target: Mapping[str, JsonValue]
) -> tuple[ChangedLeaf, ...]:
    """Return label changes that require a different semantic OCR surface.

    Raw OCR retains the template's capitalization and punctuation. A target that differs only in
    those presentation details (for example ``EGYPT`` versus ``Egypt``) is already rendered and
    must not be rewritten or later reported as a stale source fact.
    """

    return tuple(
        leaf
        for leaf in changed_leaves(source, target)
        if not (
            isinstance(leaf.sourceValue, str)
            and isinstance(leaf.targetValue, str)
            and _semantic_normalize(leaf.sourceValue) == _semantic_normalize(leaf.targetValue)
        )
    )


def _load_customs_program_registry(path: Path, *, expected_entries: int) -> CustomsProgramRegistry:
    registry = CustomsProgramRegistry.model_validate_json(
        read_regular_file_bytes(path), strict=True
    )
    if len(registry.entries) != expected_entries:
        raise ValueError(
            "customs-program registry count differs from its pin: "
            f"expected={expected_entries}, actual={len(registry.entries)}"
        )
    return registry


def _route_country_index(
    locations: Sequence[UnlocodeLocation],
    *,
    required_function: str | None = None,
) -> Mapping[str, frozenset[str]]:
    countries: dict[str, set[str]] = {}
    for location in locations:
        if required_function is not None and required_function not in location.function_codes:
            continue
        for name in (location.name, location.name_without_diacritics):
            normalized = _semantic_normalize(name)
            if normalized:
                countries.setdefault(normalized, set()).add(location.country_code)
    return MappingProxyType({name: frozenset(values) for name, values in countries.items()})


def _location_country_code(
    value: Any,
    resources: TargetIntegrityResources,
    *,
    country_hints: frozenset[str] = frozenset(),
    maritime_port: bool = False,
) -> str | None:
    if not isinstance(value, Mapping):
        return None
    explicit: str | None = None
    country = value.get("country")
    if isinstance(country, str):
        explicit = resources.countries.resolve(country)
        if explicit is None:
            raise ValueError(
                f"target route country is absent from the pinned ISO registry: {country!r}"
            )
    inferred: str | None = None
    name = value.get("name")
    if isinstance(name, str):
        index = (
            resources.route_port_countries_by_name
            if maritime_port
            else resources.route_countries_by_name
        )
        candidates = index.get(_semantic_normalize(name), frozenset())
        if len(candidates) == 1:
            inferred = next(iter(candidates))
        elif len(candidates) > 1:
            # Port/locality names are not globally unique. A unique intersection with countries
            # already printed in the synthetic parties is still evidence-backed; an empty or
            # multi-country intersection remains unresolved rather than guessed.
            hinted_candidates = candidates & country_hints
            if len(hinted_candidates) == 1:
                inferred = next(iter(hinted_candidates))
    if explicit is not None and inferred is not None and explicit != inferred:
        raise ValueError(
            "target route location country conflicts with pinned UN/LOCODE: "
            f"name={name!r}, explicit={explicit}, unlocode={inferred}"
        )
    return explicit or inferred


_ROUTE_FIELDS_BY_DIRECTION: Mapping[Literal["import", "export"], tuple[str, ...]] = {
    "import": ("placeOfDelivery", "finalDestination", "portOfDischarge"),
    "export": ("portOfLoading", "placeOfReceipt"),
}


def _target_party_country_codes(
    target_label: Mapping[str, Any], resources: TargetIntegrityResources
) -> frozenset[str]:
    patch = target_label.get("documentPatch")
    parties = patch.get("parties") if isinstance(patch, Mapping) else None
    if not isinstance(parties, Mapping):
        return frozenset()
    output: set[str] = set()
    party_values = (
        row
        for value in parties.values()
        for row in (value if isinstance(value, list) else (value,))
    )
    for party in party_values:
        country = party.get("country") if isinstance(party, Mapping) else None
        if not isinstance(country, str):
            continue
        code = resources.countries.resolve(country)
        if code is None:
            raise ValueError(
                f"target party country is absent from the pinned ISO registry: {country!r}"
            )
        output.add(code)
    return frozenset(output)


def _target_route_country_code(
    target_label: Mapping[str, Any],
    *,
    direction: Literal["import", "export"],
    resources: TargetIntegrityResources,
) -> str | None:
    patch = target_label.get("documentPatch")
    route = patch.get("route") if isinstance(patch, Mapping) else None
    if not isinstance(route, Mapping):
        return None
    country_hints = _target_party_country_codes(target_label, resources)
    for field_name in _ROUTE_FIELDS_BY_DIRECTION[direction]:
        country = _location_country_code(
            route.get(field_name),
            resources,
            country_hints=country_hints,
            maritime_port=field_name in {"portOfLoading", "portOfDischarge"},
        )
        if country is not None:
            return country
    return None


def _target_route_jurisdictions(
    target_label: Mapping[str, Any], resources: TargetIntegrityResources
) -> dict[str, JsonValue]:
    """Expose authoritative route jurisdictions for raw-only legal/customs flavor."""

    output: dict[str, JsonValue] = {}
    directions: tuple[Literal["export", "import"], ...] = ("export", "import")
    for direction in directions:
        code = _target_route_country_code(
            target_label,
            direction=direction,
            resources=resources,
        )
        output[direction] = (
            None
            if code is None
            else {
                "isoAlpha2": code,
                "name": resources.countries.printable_name(code),
            }
        )
    return output


def _literal_phrase_pattern(surface: str) -> re.Pattern[str]:
    """Match a registry surface without discarding meaningful punctuation.

    Customs acronyms can collide with ordinary cargo words (for example ``ACID`` in a chemical
    description). Registry entries therefore describe exact printed selectors such as
    ``ACID NO`` or ``ACID#``. Whitespace may vary, but punctuation is semantic and must not be
    normalized away.
    """

    stripped = surface.strip()
    if not stripped:
        raise ValueError("registry surface is empty")
    tokens = re.split(r"(\s+|[-:#])", stripped)
    body_parts: list[str] = []
    for index, token in enumerate(tokens):
        if not token:
            continue
        if token.isspace():
            previous = next((value for value in reversed(tokens[:index]) if value), "")
            following = next((value for value in tokens[index + 1 :] if value), "")
            if previous in "-:#" or following in "-:#":
                continue
            body_parts.append(r"\s+")
        elif token in "-:#":
            body_parts.append(r"\s*" + re.escape(token) + r"\s*")
        else:
            body_parts.append(re.escape(token))
    prefix = r"(?<![A-Z0-9])" if stripped[0].isalnum() else ""
    suffix = r"(?![A-Z0-9])" if stripped[-1].isalnum() else ""
    return re.compile(prefix + "".join(body_parts) + suffix, re.IGNORECASE)


def _customs_selector_match_is_safe(raw_text: str, match: re.Match[str], surface: str) -> bool:
    """Disambiguate punctuation-only selectors from ordinary cargo prose.

    ``ACID:`` is a customs field only when it begins the OCR line (allowing punctuation such as
    ``**``). A mid-line phrase such as ``FATTY ACID:`` remains immutable cargo text. Longer
    registered selectors such as ``ACID NO`` are already semantically self-identifying.
    """

    if not surface.endswith((":", "#")):
        return True
    line_start = raw_text.rfind("\n", 0, match.start()) + 1
    return not any(character.isalnum() for character in raw_text[line_start : match.start()])


def _anchored_scalar_pattern(surface: str) -> re.Pattern[str]:
    """Match an identifier/reference surface without discarding meaningful punctuation."""

    stripped = surface.strip()
    if not stripped:
        raise ValueError("anchored scalar source surface is empty")
    parts = re.split(r"(\s+)", stripped)
    body = "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)
    prefix = r"(?<![A-Z0-9])" if stripped[0].isalnum() else ""
    suffix = r"(?![A-Z0-9])" if stripped[-1].isalnum() else ""
    return re.compile(f"{prefix}{body}{suffix}", re.IGNORECASE)


def _formatted_container_pattern(surface: str) -> re.Pattern[str]:
    if _CANONICAL_CONTAINER_NUMBER.fullmatch(surface) is None:
        raise ValueError(f"container number is not canonical: {surface!r}")
    body = r"[ \t-]*".join(re.escape(character) for character in surface)
    return re.compile(rf"(?<![A-Z0-9]){body}(?![A-Z0-9])", re.IGNORECASE)


def _preserve_alphanumeric_surface_shape(source: str, target: str) -> str:
    target_characters = tuple(character for character in target if character.isalnum())
    if sum(character.isalnum() for character in source) != len(target_characters):
        raise ValueError("source and target alphanumeric surfaces have different widths")
    output: list[str] = []
    cursor = 0
    for character in source:
        if character.isalnum():
            output.append(target_characters[cursor])
            cursor += 1
        else:
            output.append(character)
    return "".join(output)


def _anchored_requirement_pattern(
    requirement: AnchoredScalarReplacementRequirement,
    surface: str,
) -> re.Pattern[str]:
    if requirement.surfaceKind == "scalar":
        return _anchored_scalar_pattern(surface)
    if requirement.surfaceKind == "package_noun":
        stripped = surface.strip()
        if not stripped:
            raise ValueError("anchored package noun surface is empty")
        return re.compile(
            rf"(?<![A-Z]){re.escape(stripped)}(?![A-Z])",
            re.IGNORECASE,
        )
    stripped = surface.strip()
    if not stripped:
        raise ValueError("anchored measurement surface is empty")
    return re.compile(
        rf"(?<![0-9.,]){re.escape(stripped)}(?![0-9.,])",
        re.IGNORECASE,
    )


def jurisdictional_surface_requirements(
    raw_text: str,
    target_label: Mapping[str, Any],
    resources: TargetIntegrityResources,
) -> tuple[JurisdictionalSurfaceRequirement, ...]:
    """Require named customs programs to agree with the synthetic route jurisdiction."""

    output: list[JurisdictionalSurfaceRequirement] = []
    raw_lines = raw_text.splitlines()
    for program in resources.customs_programs:
        matched: list[tuple[str, tuple[str, ...], int]] = []
        occupied: list[tuple[int, int]] = []
        for surface in program.source_surfaces:
            matches = tuple(
                match
                for match in _literal_phrase_pattern(surface).finditer(raw_text)
                if _customs_selector_match_is_safe(raw_text, match, surface)
                if not any(match.start() < end and match.end() > start for start, end in occupied)
            )
            count = len(matches)
            if count:
                line_ids = tuple(
                    _line_id(
                        raw_text.count("\n", 0, match.start()) + 1,
                        raw_lines[raw_text.count("\n", 0, match.start())],
                    )
                    for match in matches
                )
                matched.append((surface, tuple(dict.fromkeys(line_ids)), count))
                occupied.extend((match.start(), match.end()) for match in matches)
        if not matched:
            continue
        target_country = _target_route_country_code(
            target_label,
            direction=program.trade_direction,
            resources=resources,
        )
        if target_country is None:
            raise ValueError(
                "cannot safely rewrite a jurisdiction-bound customs program without a resolved "
                f"target {program.trade_direction} route country: {program.program_id}"
            )
        if target_country == program.jurisdiction_country_code:
            continue
        output.extend(
            JurisdictionalSurfaceRequirement(
                requirementId=(
                    "jurisdiction-"
                    + program.program_id
                    + "-"
                    + sha256_bytes(
                        canonical_json_bytes(
                            {
                                "surface": surface,
                                "lineIds": list(line_ids),
                                "target": program.generic_replacement_surface,
                            }
                        )
                    )[:12]
                ),
                programId=program.program_id,
                tradeDirection=program.trade_direction,
                programJurisdictionCountryCode=program.jurisdiction_country_code,
                targetRouteCountryCode=target_country,
                sourceLineIds=line_ids,
                sourceSurface=surface,
                targetSurface=program.generic_replacement_surface,
                sourceOccurrences=occurrences,
                authority=program.authority,
                officialSourceUrl=program.official_source_url,
            )
            for surface, line_ids, occurrences in matched
        )
    return tuple(output)


def _jurisdictional_surfaces_rendered(
    value: str, requirements: Sequence[JurisdictionalSurfaceRequirement]
) -> bool:
    lines = value.splitlines()
    for requirement in requirements:
        source_pattern = _literal_phrase_pattern(requirement.sourceSurface)
        target_pattern = _literal_phrase_pattern(requirement.targetSurface)
        for line_id in requirement.sourceLineIds:
            line_index = _line_number(line_id) - 1
            if not 0 <= line_index < len(lines):
                return False
            line = lines[line_index]
            if source_pattern.search(line) is not None or target_pattern.search(line) is None:
                return False
    return True


_NON_LITERAL_TARGET_FIELD = re.compile(
    r"(?:^|\.)(?:"
    r"typeCategory|sizeCategory|hazardCategory|subsidiaryHazardCategories|"
    r"packingGroupCategory|paymentArrangement|negotiability|unit|sameAsConsignee"
    r")(?:\[[0-9]+\])?$"
)
_FORMATTED_IDENTIFIER_TARGET_FIELD = re.compile(
    r"^documentPatch\.(?:containers\[[0-9]+\]|cargoAllocationGroups\[[0-9]+\]\."
    r"allocations\[[0-9]+\])\.containerNumber$"
)
_FORWARDING_OR_EXPORT_REFERENCE_CONTEXT = re.compile(
    r"\b(?:EXPORT(?:[ \t]+REFERENCES?)?|INVOICE|SHIPPING[ \t]+BILL|"
    r"SB[ \t]*(?:NO\.?|NUMBER)|CUSTOMS|REFERENCE|REF(?:ERENCE)?[ \t]*NO|"
    r"F/?AGENT[^\r\n]{0,30}\bREF|SHIPPER'?S[ \t]+REF)\b",
    re.IGNORECASE,
)


def target_literal_requirements(
    leaves: Sequence[ChangedLeaf],
    anchored_replacements: Sequence[AnchoredScalarReplacementRequirement] = (),
) -> tuple[TargetLiteralRequirement, ...]:
    """Require changed human-readable strings while excluding schema/category encodings."""

    rows: list[TargetLiteralRequirement] = []
    seen: set[tuple[str, str]] = set()
    anchored_paths = {
        path for requirement in anchored_replacements for path in requirement.targetPaths
    }
    for leaf in leaves:
        value = leaf.targetValue
        if (
            not leaf.requiresTextEdit
            or not leaf.targetPresent
            or not isinstance(value, str)
            or len(value.strip()) < 4
            or leaf.path.endswith("Date")
            or ".hsCodes[" in leaf.path
            or leaf.path.startswith("documentPatch.parties.")
            or _NON_LITERAL_TARGET_FIELD.search(leaf.path) is not None
            or leaf.path in anchored_paths
        ):
            continue
        normalized = _semantic_normalize(value)
        if not normalized or (leaf.path, normalized) in seen:
            continue
        rows.append(
            TargetLiteralRequirement(
                targetPath=leaf.path,
                targetValue=value,
                matchPolicy=(
                    "alphanumeric_identifier"
                    if _FORMATTED_IDENTIFIER_TARGET_FIELD.fullmatch(leaf.path) is not None
                    else "ordered_semantic_atoms"
                    if re.fullmatch(
                        r"documentPatch\.forwardingAndExportReferences\[[0-9]+\]", leaf.path
                    )
                    and len(value.split()) > 1
                    else "semantic_literal"
                ),
            )
        )
        seen.add((leaf.path, normalized))
    return tuple(rows)


def _missing_target_literals(
    value: str, requirements: Sequence[TargetLiteralRequirement]
) -> tuple[TargetLiteralRequirement, ...]:
    normalized = _semantic_normalize(value)
    alphanumeric = re.sub(r"[^A-Z0-9]", "", value.upper())

    def rendered(requirement: TargetLiteralRequirement) -> bool:
        if requirement.matchPolicy == "alphanumeric_identifier":
            return re.sub(r"[^A-Z0-9]", "", requirement.targetValue.upper()) in alphanumeric
        if requirement.matchPolicy == "semantic_literal":
            return _semantic_normalize(requirement.targetValue) in normalized
        atoms = tuple(
            atom
            for piece in requirement.targetValue.split()
            if (atom := _semantic_normalize(piece))
        )
        if len(atoms) < 2:
            return False
        # Printed field labels may occur between a reference number and its date. Keep the
        # allowed gap and physical window finite, and require explicit reference grammar, so
        # unrelated identifiers and dates elsewhere in the document cannot satisfy it.
        pattern = r"(?:[ \t]+[A-Z0-9]+){0,6}[ \t]+".join(re.escape(atom) for atom in atoms)
        lines = value.splitlines()
        return any(
            _FORWARDING_OR_EXPORT_REFERENCE_CONTEXT.search(window) is not None
            and re.search(pattern, _semantic_normalize(window)) is not None
            for start in range(len(lines))
            for width in (1, 2)
            if start + width <= len(lines)
            for window in ("\n".join(lines[start : start + width]),)
        )

    return tuple(row for row in requirements if not rendered(row))


def target_value_occurrence_requirements(
    raw_text: str,
    leaves: Sequence[ChangedLeaf],
    raw_auxiliary_identities: Sequence[RawAuxiliaryIdentityRequirement] = (),
    surface_requirements: Sequence[SurfaceRenderingRequirement] = (),
    anchored_replacements: Sequence[AnchoredScalarReplacementRequirement] = (),
    *,
    defer_ambiguous_party_scalar_cardinality: bool = False,
) -> tuple[TargetValueOccurrenceRequirement, ...]:
    """Preserve role-bound party scalar cardinality without duplicating shared values.

    A source document can deliberately print one contact value for several semantic roles. For
    example, consignee and notify party can share one phone number and print it once in each party
    block. Synthetic labels normally give those roles different replacement numbers. Counting all
    raw occurrences independently for every changed leaf would require both replacements in both
    slots. Instead, divide the printed occurrences of a shared source value across the label leaves
    that own it, then aggregate those per-leaf multiplicities by replacement value. Party names and
    addresses use the source party-name repetition count, so surplus address lines must receive
    distinct auxiliary detail rather than duplicate the labeled address.
    """

    contact_leaves: list[ChangedLeaf] = []
    party_scalar_leaves: list[ChangedLeaf] = []
    party_path = re.compile(
        r"^(?P<prefix>documentPatch\.parties\.(?:notifyParties\[[0-9]+\]|[A-Za-z]+))\."
        r"(?P<field>name|address)$"
    )
    for leaf in leaves:
        if (
            not leaf.requiresTextEdit
            or not leaf.sourcePresent
            or not leaf.targetPresent
            or not isinstance(leaf.sourceValue, str)
            or not isinstance(leaf.targetValue, str)
            or len(leaf.targetValue.strip()) < 4
            or leaf.sourceValue == leaf.targetValue
        ):
            continue
        if ".contactDetails." in leaf.path:
            contact_leaves.append(leaf)
        elif party_path.fullmatch(leaf.path) is not None:
            party_scalar_leaves.append(leaf)

    by_source: dict[str, list[ChangedLeaf]] = {}
    for leaf in contact_leaves:
        source_value = cast(str, leaf.sourceValue)
        by_source.setdefault(source_value, []).append(leaf)

    contributions_by_path: dict[str, int] = {}
    for source_value, source_leaves in sorted(by_source.items()):
        printed_occurrences = raw_text.count(source_value)
        if printed_occurrences == 0:
            continue
        semantic_owners = len(source_leaves)
        if printed_occurrences % semantic_owners:
            paths = ", ".join(sorted(leaf.path for leaf in source_leaves))
            raise ValueError(
                "printed contact occurrences cannot be assigned exactly across semantic owners: "
                f"{source_value!r} occurs {printed_occurrences} time(s) for {semantic_owners} "
                f"owner(s): {paths}"
            )
        per_owner = printed_occurrences // semantic_owners
        for leaf in source_leaves:
            contributions_by_path[leaf.path] = per_owner

    party_owners_by_source: dict[tuple[str, str], list[ChangedLeaf]] = {}
    for leaf in party_scalar_leaves:
        match = party_path.fullmatch(leaf.path)
        if match is not None:
            party_owners_by_source.setdefault(
                (match.group("field"), cast(str, leaf.sourceValue)), []
            ).append(leaf)
    auxiliary_name_occurrences: dict[str, int] = {}
    for requirement in raw_auxiliary_identities:
        key = _semantic_normalize(requirement.sourceIdentity)
        auxiliary_name_occurrences[key] = auxiliary_name_occurrences.get(key, 0) + 1
    for (field_name, source_value), owners in sorted(party_owners_by_source.items()):
        normalized_source = _semantic_normalize(source_value)
        source_paths = {leaf.path for leaf in owners}
        carrier_principal = field_name == "name" and source_paths == {
            "documentPatch.parties.carrier.name"
        }
        printed_occurrences = (
            carrier_principal_template_slot_count(raw_text, source_value)
            if carrier_principal
            else _party_scalar_occurrence_count(raw_text, source_value)
        )
        if field_name == "name" and not carrier_principal:
            printed_occurrences -= auxiliary_name_occurrences.get(normalized_source, 0)
        if field_name == "name":
            printed_occurrences += sum(
                requirement.sourceOccurrences
                for requirement in surface_requirements
                if requirement.kind == "carrier_header_identity"
                and requirement.targetPath in source_paths
                and _semantic_normalize(requirement.sourceSurface) != normalized_source
                and _semantic_normalize(requirement.targetSurface)
                == _semantic_normalize(cast(str, owners[0].targetValue))
            )
        if printed_occurrences < 0:
            raise ValueError(
                f"raw auxiliary identities exceed printed party-name occurrences: {source_value!r}"
            )
        if printed_occurrences == 0:
            continue
        quotient, remainder = divmod(printed_occurrences, len(owners))
        target_values = {_semantic_normalize(cast(str, leaf.targetValue)) for leaf in owners}
        if remainder and len(target_values) != 1 and not defer_ambiguous_party_scalar_cardinality:
            raise ValueError(
                "printed party scalar occurrences cannot be assigned exactly across semantic "
                f"owners: {source_value!r} occurs {printed_occurrences} time(s) for "
                f"{len(owners)} owner(s): {', '.join(sorted(source_paths))}"
            )
        # Several semantic roles may intentionally project to one target identity (for example,
        # consignee plus notify party), while an auxiliary ``IMPORTER:`` line prints a third copy
        # of the same source identity.  Only the aggregate target occurrence count matters in
        # that case.  The inventory compiler can also defer a non-divisible count for distinct
        # targets: these provisional contributions exist only long enough for its role-owned line
        # evidence to replace them with exact per-role cardinalities before model execution.
        for index, leaf in enumerate(sorted(owners, key=lambda row: row.path)):
            contributions_by_path[leaf.path] = quotient + (1 if index < remainder else 0)

    # Some OCR values are printed across a physical line break and therefore have zero exact
    # whole-text occurrences (for example a telephone extension split onto the next line). The
    # anchored compiler has already proven the exact owned line and target rendering. Use that
    # proof for occurrence cardinality instead of silently under-counting a target value shared
    # by another party role.
    for leaf in (*contact_leaves, *party_scalar_leaves):
        if leaf.path in contributions_by_path:
            continue
        matching = tuple(
            requirement
            for requirement in anchored_replacements
            if leaf.path in requirement.targetPaths
            and requirement.targetSurface == leaf.targetValue
        )
        if matching:
            contributions_by_path[leaf.path] = sum(
                len(requirement.sourceLineIds) for requirement in matching
            )

    grouped: dict[str, dict[str, Any]] = {}
    eligible = (*contact_leaves, *party_scalar_leaves)
    incomplete_target_values = {
        _semantic_normalize(cast(str, leaf.targetValue))
        for leaf in eligible
        if leaf.path not in contributions_by_path
    }
    changed_source_values = frozenset(cast(str, leaf.sourceValue) for leaf in eligible)
    for leaf in eligible:
        contribution = contributions_by_path.get(leaf.path)
        if contribution is None:
            continue
        target_value = cast(str, leaf.targetValue)
        normalized = _semantic_normalize(target_value)
        row = grouped.setdefault(
            normalized,
            {
                "targetValue": target_value,
                "paths": [],
                "requiredOccurrences": (
                    raw_text.count(target_value) if target_value not in changed_source_values else 0
                ),
            },
        )
        if row["targetValue"] != target_value:
            raise ValueError("contact values normalize identically but have different surfaces")
        row["paths"].append(leaf.path)
        row["requiredOccurrences"] += contribution

    output: list[TargetValueOccurrenceRequirement] = []
    for normalized in sorted(grouped):
        # An exact count derived from only some semantic owners is worse than no count: it can
        # make two legitimate party fields with the same synthetic value mutually impossible.
        # Other literal, anchored, and role-block checks still require every located target.
        if normalized in incomplete_target_values:
            continue
        row = grouped[normalized]
        target_value = cast(str, row["targetValue"])
        required = cast(int, row["requiredOccurrences"])
        if required < 1:
            continue
        output.append(
            TargetValueOccurrenceRequirement(
                targetPaths=tuple(sorted(cast(list[str], row["paths"]))),
                targetValue=target_value,
                requiredOccurrences=required,
            )
        )
    return tuple(output)


def _target_value_occurrences_exact(
    value: str, requirements: Sequence[TargetValueOccurrenceRequirement]
) -> bool:
    return all(
        _target_value_occurrence_count(value, row) == row.requiredOccurrences
        for row in requirements
    )


def _target_value_occurrence_count(
    value: str, requirement: TargetValueOccurrenceRequirement
) -> int:
    if any(".contactDetails." in path for path in requirement.targetPaths):
        return value.count(requirement.targetValue)
    if any(
        path.startswith("documentPatch.parties.") and path.rsplit(".", 1)[-1] in {"name", "address"}
        for path in requirement.targetPaths
    ):
        return _party_scalar_occurrence_count(value, requirement.targetValue)
    normalized_value = _semantic_normalize(value)
    normalized_target = _semantic_normalize(requirement.targetValue)
    if not normalized_target:
        return 0
    # Flattened OCR often concatenates a heading and its value (``Notify PartyACME LTD``). Exact
    # token boundaries would reject a value that is visibly present in the correct source style.
    # Cardinality is still exact because the complete normalized target scalar is counted.
    return normalized_value.count(normalized_target)


def anchored_scalar_replacement_requirements(
    raw_text: str, leaves: Sequence[ChangedLeaf]
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Anchor unambiguous exact scalar replacements to their original printed lines.

    This contract is intentionally limited to schema fields whose values are identifiers,
    references, or exact contact scalars and therefore cannot be legitimately paraphrased. Party
    and route prose remains under role-aware occurrence and semantic-review contracts. Dates and
    HS codes have dedicated source-format renderers.
    """

    semantic_source_targets: dict[str, set[str]] = {}
    for leaf in leaves:
        if not isinstance(leaf.sourceValue, str) or not isinstance(leaf.targetValue, str):
            continue
        normalized_source = _semantic_normalize(leaf.sourceValue)
        normalized_target = _semantic_normalize(leaf.targetValue)
        if normalized_source and normalized_target:
            semantic_source_targets.setdefault(normalized_source, set()).add(normalized_target)

    grouped: dict[str, list[ChangedLeaf]] = {}
    for leaf in leaves:
        if (
            not leaf.requiresTextEdit
            or not leaf.sourcePresent
            or not leaf.targetPresent
            or not isinstance(leaf.sourceValue, str)
            or not isinstance(leaf.targetValue, str)
            or leaf.sourceValue == leaf.targetValue
            or len(leaf.sourceValue.strip()) < 4
            or "\n" in leaf.sourceValue
            or "\n" in leaf.targetValue
            or leaf.path.endswith("Date")
            or ".hsCodes[" in leaf.path
            or _NON_LITERAL_TARGET_FIELD.search(leaf.path) is not None
            or (
                _ANCHORABLE_IDENTIFIER_PATH.fullmatch(leaf.path) is None
                and _ANCHORABLE_CONTACT_PATH.fullmatch(leaf.path) is None
            )
        ):
            continue
        # The same OCR surface can represent different fields. For example, a reviewed mark
        # ``*EGYPT`` can also appear in a notify-party country line. Replacing every literal
        # occurrence would move the new mark into an address. Only anchor a surface globally when
        # all semantically equivalent changed source values resolve to the same target surface.
        normalized_source = _semantic_normalize(leaf.sourceValue)
        if len(semantic_source_targets.get(normalized_source, set())) > 1:
            continue
        grouped.setdefault(leaf.sourceValue, []).append(leaf)

    source_lines = raw_text.splitlines()
    source_patterns = {
        source_surface: _anchored_scalar_pattern(source_surface) for source_surface in grouped
    }
    requirements: list[AnchoredScalarReplacementRequirement] = []
    for source_surface, source_leaves in sorted(
        grouped.items(), key=lambda row: (-len(row[0]), row[0])
    ):
        target_surfaces = {cast(str, leaf.targetValue) for leaf in source_leaves}
        if len(target_surfaces) != 1:
            continue
        target_surface = next(iter(target_surfaces))
        if (
            _CANONICAL_CONTAINER_NUMBER.fullmatch(source_surface) is not None
            and _CANONICAL_CONTAINER_NUMBER.fullmatch(target_surface) is not None
            and all(
                _CONTAINER_NUMBER_TARGET_PATH.fullmatch(leaf.path) is not None
                for leaf in source_leaves
            )
        ):
            printed_pattern = _formatted_container_pattern(source_surface)
            printed: dict[str, list[str]] = {}
            for index, line in enumerate(source_lines, start=1):
                for match in printed_pattern.finditer(line):
                    printed.setdefault(match.group(0), []).append(_line_id(index, line))
            for printed_source, printed_line_ids in sorted(printed.items()):
                requirements.append(
                    AnchoredScalarReplacementRequirement(
                        targetPaths=tuple(sorted(leaf.path for leaf in source_leaves)),
                        sourceLineIds=tuple(printed_line_ids),
                        sourceSurface=printed_source,
                        targetSurface=_preserve_alphanumeric_surface_shape(
                            printed_source, target_surface
                        ),
                    )
                )
            if printed:
                continue
        pattern = source_patterns[source_surface]
        source_line_ids: tuple[str, ...] = tuple(
            _line_id(index, line)
            for index, line in enumerate(source_lines, start=1)
            if any(
                not any(
                    len(other_surface) > len(source_surface)
                    and other_match.start() <= match.start()
                    and match.end() <= other_match.end()
                    for other_surface, other_pattern in source_patterns.items()
                    for other_match in other_pattern.finditer(line)
                )
                for match in pattern.finditer(line)
            )
        )
        if len(source_line_ids) > 1 and all(
            _MARKS_AND_NUMBERS_PATH.fullmatch(leaf.path) is not None for leaf in source_leaves
        ):
            # A mark is often copied from a party identity.  Replacing every literal match can
            # silently write the synthetic mark into the consignee/notify block.  Multiple
            # occurrences are deterministic only when the field label on that same line proves
            # which copy is the mark.  Unique and page-split high-information marks retain their
            # existing exact paths below.
            explicit_mark_lines: list[str] = []
            for index, line in enumerate(source_lines, start=1):
                for match in pattern.finditer(line):
                    if _INLINE_MARKS_VALUE_PREFIX.search(line[: match.start()]) is not None:
                        explicit_mark_lines.append(_line_id(index, line))
                        break
            source_line_ids = tuple(explicit_mark_lines)
        if not source_line_ids:
            # A marks-and-numbers value can be interrupted by a physical page break while its
            # semantic value remains one label scalar (for example ``Q40043311 -`` on page 1 and
            # ``Q40043782`` after the repeated page-2 heading).  When every atom is a unique,
            # high-information identifier and source/target arity agrees, each atom is a safe
            # deterministic anchor.  Anything less remains unlocated and fail-closed.
            if all(
                _MARKS_AND_NUMBERS_PATH.fullmatch(leaf.path) is not None for leaf in source_leaves
            ):
                source_atoms = tuple(re.findall(r"[A-Za-z0-9]+", source_surface))
                target_atoms = tuple(re.findall(r"[A-Za-z0-9]+", target_surface))
                high_information = all(
                    len(atom) >= 4 and any(character.isdigit() for character in atom)
                    for atom in source_atoms
                )
                source_atom_patterns = tuple(
                    _anchored_scalar_pattern(atom) for atom in source_atoms
                )
                if (
                    source_atoms
                    and len(source_atoms) == len(target_atoms)
                    and len(set(source_atoms)) == len(source_atoms)
                    and high_information
                    and all(
                        sum(1 for _ in pattern.finditer(raw_text)) == 1
                        for pattern in source_atom_patterns
                    )
                ):
                    target_paths = tuple(sorted(leaf.path for leaf in source_leaves))
                    for source_atom, target_atom, atom_pattern in zip(
                        source_atoms, target_atoms, source_atom_patterns, strict=True
                    ):
                        atom_line_ids = tuple(
                            _line_id(index, line)
                            for index, line in enumerate(source_lines, start=1)
                            if atom_pattern.search(line) is not None
                        )
                        if len(atom_line_ids) != 1:
                            raise AssertionError("unique mark atom did not resolve to one line")
                        requirements.append(
                            AnchoredScalarReplacementRequirement(
                                targetPaths=target_paths,
                                sourceLineIds=atom_line_ids,
                                sourceSurface=source_atom,
                                targetSurface=target_atom,
                            )
                        )
            continue
        requirements.append(
            AnchoredScalarReplacementRequirement(
                targetPaths=tuple(sorted(leaf.path for leaf in source_leaves)),
                sourceLineIds=source_line_ids,
                sourceSurface=source_surface,
                targetSurface=target_surface,
            )
        )
    return tuple(requirements)


def _equipment_surface_with_source_case(source: str, target: str) -> str:
    """Render canonical equipment semantics without changing the source's letter-case style."""

    letters = tuple(character for character in source if character.isalpha())
    if letters and all(character.isupper() for character in letters):
        return target.upper()
    if letters and all(character.islower() for character in letters):
        return target.lower()
    words = re.findall(r"[A-Za-z]+", source)
    if words and all(word[0].isupper() and word[1:].islower() for word in words):
        return target.title()
    return target


def container_equipment_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Bind changed equipment semantics to the exact source container row.

    Container size/type is categorical and has no linguistic degrees of freedom.  A repeated
    source description such as ``40' Dry Hi-Cube`` may map to different target categories for
    different containers, so global replacement is unsafe.  This compiler instead requires the
    exact source container identifier on the same line, or (for a one-container template) a
    unique equipment surface elsewhere in the document.  Unbound summaries remain outside this
    function and fail closed or use their dedicated aggregate renderer.
    """

    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    source_containers = (
        source_patch.get("containers") if isinstance(source_patch, Mapping) else None
    )
    target_containers = (
        target_patch.get("containers") if isinstance(target_patch, Mapping) else None
    )
    if not isinstance(source_containers, Sequence) or isinstance(source_containers, (str, bytes)):
        return ()
    if not isinstance(target_containers, Sequence) or isinstance(target_containers, (str, bytes)):
        return ()
    if len(source_containers) != len(target_containers):
        raise ValueError("cannot bind equipment surfaces across different container counts")

    source_lines = raw_text.splitlines()
    recovered_single_container_surface: str | None = None
    if len(source_containers) == 1:
        recovered_candidates = tuple(
            match.group(0)
            for line in source_lines
            for match in _INLINE_EQUIPMENT_ALIAS.finditer(line)
            if (
                reviewed := review_source_equipment_surface(
                    match.group(0), temperature_present=False
                )
            ).size_category
            is not None
            and reviewed.type_category is not None
        )
        if len(recovered_candidates) == 1:
            recovered_single_container_surface = recovered_candidates[0]
    container_number_lines: dict[int, frozenset[int]] = {}
    for container_index, container in enumerate(source_containers):
        if not isinstance(container, Mapping):
            raise ValueError("source container is not an object")
        number = container.get("containerNumber")
        if not isinstance(number, str):
            continue
        number_pattern = _formatted_container_pattern(number)
        container_number_lines[container_index] = frozenset(
            line_number
            for line_number, line in enumerate(source_lines, start=1)
            if number_pattern.search(line) is not None
        )
    requirements: list[AnchoredScalarReplacementRequirement] = []
    table_occurrences: (
        Mapping[
            int,
            Sequence[tuple[OperationalKind, int, str, str, OperationalGrammar]],
        ]
        | None
    ) = None
    for index, (source_container, target_container) in enumerate(
        zip(source_containers, target_containers, strict=True)
    ):
        if not isinstance(source_container, Mapping) or not isinstance(target_container, Mapping):
            raise ValueError("container equipment binding received a non-object container")
        source_number = source_container.get("containerNumber")
        source_surface = next(
            (
                source_container.get(key)
                for key in ("typeDescription", "typeCode")
                if isinstance(source_container.get(key), str)
            ),
            recovered_single_container_surface,
        )
        target_size = target_container.get("sizeCategory")
        target_type = target_container.get("typeCategory")
        if not all(
            isinstance(value, str)
            for value in (source_number, source_surface, target_size, target_type)
        ):
            continue
        reviewed = review_source_equipment_surface(
            cast(str, source_surface),
            temperature_present=target_container.get("temperatureSetpoint") is not None,
        )
        if reviewed.resolution == "reviewed_source_grammar" and (
            reviewed.size_category,
            reviewed.type_category,
        ) == (target_size, target_type):
            continue

        target_surface = canonical_equipment_surface(
            cast(bill_of_lading_v5.ContainerSizeCategory, target_size),
            cast(bill_of_lading_v5.ContainerTypeCategory, target_type),
        )
        surface_pattern = _anchored_scalar_pattern(cast(str, source_surface))
        number_pattern = _formatted_container_pattern(cast(str, source_number))
        matches: list[tuple[str, str]] = []
        for line_number, line in enumerate(source_lines, start=1):
            if number_pattern.search(line) is None:
                continue
            surface_matches = tuple(surface_pattern.finditer(line))
            if len(surface_matches) != 1:
                continue
            printed_surface = surface_matches[0].group(0)
            matches.append((_line_id(line_number, line), printed_surface))

        if not matches:
            if table_occurrences is None:
                table_occurrences = _container_measurement_occurrences(
                    raw_text, cast(Sequence[Mapping[str, Any]], source_containers)
                )
            for kind, line_number, _value, _evidence, _grammar in table_occurrences.get(index, ()):
                if kind != "package_quantity":
                    continue
                line = source_lines[line_number - 1]
                surface_matches = tuple(surface_pattern.finditer(line))
                if len(surface_matches) != 1:
                    continue
                matches.append(
                    (
                        _line_id(line_number, line),
                        surface_matches[0].group(0),
                    )
                )

        if not matches:
            # Some flattened manifests print all container identifiers first and then one
            # equipment/measurement row per container in the same order.  When one exact source
            # surface is shared by a known subset of containers and occurs exactly that many
            # times on distinct physical lines, positional alignment is complete evidence.  Any
            # extra boilerplate or aggregate occurrence defeats the equality and falls through
            # to the relation-local binders below.
            peer_indices = tuple(
                peer_index
                for peer_index, peer in enumerate(source_containers)
                if isinstance(peer, Mapping)
                and peer.get("typeDescription") == source_surface
            )
            positional_occurrences = tuple(
                (line_number, line, line_matches[0].group(0))
                for line_number, line in enumerate(source_lines, start=1)
                if (line_matches := tuple(surface_pattern.finditer(line)))
                and len(line_matches) == 1
                and not (
                    (aggregate_match := _COMPACT_AGGREGATE_EQUIPMENT_LINE.fullmatch(line))
                    is not None
                    and len(source_containers) > 1
                    and int(aggregate_match.group("count").replace(",", ""))
                    == len(source_containers)
                )
            )
            if (
                re.search(r"[A-Za-z]", cast(str, source_surface)) is not None
                and peer_indices
                and len(positional_occurrences) == len(peer_indices)
                and len({row[0] for row in positional_occurrences}) == len(peer_indices)
            ):
                position = peer_indices.index(index)
                line_number, line, printed_surface = positional_occurrences[position]
                matches.append((_line_id(line_number, line), printed_surface))

        if not matches:
            # Columnar manifests often print the container identifier, seal, and equipment type
            # on consecutive physical lines.  Bind an exact type surface only when its closest
            # container identifier in the same nonblank paragraph is unique and no more than three
            # lines away.  Lines that themselves describe an anonymous/aggregate container count
            # are not per-container slots and are intentionally excluded.
            for line_number, line in enumerate(source_lines, start=1):
                surface_matches = tuple(surface_pattern.finditer(line))
                if len(surface_matches) != 1:
                    continue
                aggregate_match = _COMPACT_AGGREGATE_EQUIPMENT_LINE.fullmatch(line)
                if (
                    len(source_containers) > 1
                    and aggregate_match is not None
                    and int(aggregate_match.group("count").replace(",", ""))
                    == len(source_containers)
                ):
                    continue
                distances = {
                    container_index: distance
                    for container_index, owned_lines in container_number_lines.items()
                    if (
                        distance := _nearby_equipment_line_distance(
                            source_lines,
                            owned_lines,
                            line_number,
                        )
                    )
                    is not None
                    and distance <= 3
                }
                if not distances:
                    continue
                minimum = min(distances.values())
                owners = tuple(
                    container_index
                    for container_index, distance in distances.items()
                    if distance == minimum
                )
                if owners != (index,):
                    continue
                matches.append(
                    (
                        _line_id(line_number, line),
                        surface_matches[0].group(0),
                    )
                )

        if not matches and len(source_containers) == 1:
            all_surface_matches = [
                (line_number, line, tuple(surface_pattern.finditer(line)))
                for line_number, line in enumerate(source_lines, start=1)
                if surface_pattern.search(line) is not None
            ]
            if all(len(line_matches) == 1 for _, _, line_matches in all_surface_matches):
                matches.extend(
                    (_line_id(line_number, line), line_matches[0].group(0))
                    for line_number, line, line_matches in all_surface_matches
                )

        # A container-local cargo row can repeat the same equipment semantics in a different
        # compact spelling (for example ``20 Package(s) of 1x40 'HQ``).  It is part of the same
        # target container fact and must change with that container; otherwise a 20-foot target
        # can retain a contradictory 40-foot assertion.  The already relation-proven operational
        # row supplies ownership, while the reviewed equipment parser proves semantic identity.
        if table_occurrences is None:
            table_occurrences = _container_measurement_occurrences(
                raw_text, cast(Sequence[Mapping[str, Any]], source_containers)
            )
        source_pair = (reviewed.size_category, reviewed.type_category)
        counted_aliases: list[tuple[str, str, str]] = []
        directly_owned_lines = {line_id for line_id, _surface in matches}
        for kind, line_number, _value, _evidence, _grammar in table_occurrences.get(index, ()):
            if kind != "package_quantity":
                continue
            line = source_lines[line_number - 1]
            owned_line_id = _line_id(line_number, line)
            if owned_line_id in directly_owned_lines:
                continue
            for match in _COUNT_PREFIXED_INLINE_EQUIPMENT_ALIAS.finditer(line):
                printed_alias = match.group("equipment")
                alias_review = review_source_equipment_surface(
                    printed_alias,
                    temperature_present=target_container.get("temperatureSetpoint") is not None,
                )
                if (
                    alias_review.resolution == "reviewed_source_grammar"
                    and (alias_review.size_category, alias_review.type_category) == source_pair
                ):
                    prefix = match.group(0)[: match.start("equipment") - match.start()]
                    counted_aliases.append(
                        (
                            owned_line_id,
                            match.group(0),
                            prefix
                            + _equipment_surface_with_source_case(
                                printed_alias, target_surface
                            ),
                        )
                    )

        for printed_surface in dict.fromkeys(surface for _, surface in matches):
            line_ids = tuple(line_id for line_id, surface in matches if surface == printed_surface)
            requirements.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=(f"documentPatch.containers[{index}].printedEquipmentSurface",),
                    sourceLineIds=line_ids,
                    sourceSurface=printed_surface,
                    targetSurface=_equipment_surface_with_source_case(
                        printed_surface, target_surface
                    ),
                )
            )
        requirements.extend(
            AnchoredScalarReplacementRequirement(
                targetPaths=(f"documentPatch.containers[{index}].printedEquipmentSurface",),
                sourceLineIds=(line_id,),
                sourceSurface=source_alias,
                targetSurface=target_alias,
            )
            for line_id, source_alias, target_alias in counted_aliases
        )
    return tuple(requirements)


def _package_categories_by_container(
    label: Mapping[str, Any],
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Resolve one unambiguous task-facing package category per container."""

    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return {}
    packages: dict[str, tuple[str, str]] = {}
    for package_index, package in enumerate(patch.get("cargoPackages") or ()):
        if not isinstance(package, Mapping):
            raise ValueError("cargo package is not an object")
        package_id = package.get("packageId")
        category = package.get("typeCategory")
        if not isinstance(package_id, str) or not isinstance(category, str):
            continue
        if package_id in packages:
            raise ValueError(f"duplicate cargo package identifier: {package_id}")
        packages[package_id] = (
            category,
            f"documentPatch.cargoPackages[{package_index}].typeCategory",
        )

    resolved: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for allocation_group in patch.get("cargoAllocationGroups") or ():
        if not isinstance(allocation_group, Mapping):
            raise ValueError("cargo allocation group is not an object")
        group_package_ids = allocation_group.get("packageIds") or ()
        if not isinstance(group_package_ids, Sequence) or isinstance(
            group_package_ids, (str, bytes)
        ):
            raise ValueError("cargo allocation packageIds is not an array")
        for allocation in allocation_group.get("allocations") or ():
            if not isinstance(allocation, Mapping):
                raise ValueError("cargo allocation is not an object")
            container_number = allocation.get("containerNumber")
            if not isinstance(container_number, str):
                continue
            package_id = allocation.get("packageId")
            if not isinstance(package_id, str):
                package_id = group_package_ids[0] if len(group_package_ids) == 1 else None
            if not isinstance(package_id, str):
                continue
            package = packages.get(package_id)
            if package is None:
                raise ValueError(f"cargo allocation names unknown package {package_id}")
            resolved[container_number].append(package)

    output: dict[str, tuple[str, tuple[str, ...]]] = {}
    for container_number, values in resolved.items():
        categories = {category for category, _path in values}
        if len(categories) != 1:
            continue
        output[container_number] = (
            next(iter(categories)),
            tuple(sorted({path for _category, path in values})),
        )
    return output


def bind_cargo_package_surface_guards(
    requirements: Sequence[CargoFlavorRewriteRequirement],
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    package_registry: LoadedPackageRegistry,
) -> tuple[CargoFlavorRewriteRequirement, ...]:
    """Attach an allowlist of package semantics to safely editable cargo spans.

    Models must not use surplus cargo lines to invent a package level that is absent from the
    synthetic target.  A guard is enabled only when package identities are unchanged and every
    source category either remains unchanged or every category changes.  Mixed transitions lack
    enough raw-text ownership to distinguish an unchanged alias from a stale one and therefore
    remain under the existing fail-closed model contract.
    """

    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    if not isinstance(source_patch, Mapping) or not isinstance(target_patch, Mapping):
        return tuple(requirements)
    source_packages = source_patch.get("cargoPackages") or ()
    target_packages = target_patch.get("cargoPackages") or ()
    target_groups = target_patch.get("cargoGroups") or ()
    if any(
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
        for value in (source_packages, target_packages, target_groups)
    ):
        return tuple(requirements)

    def keyed_categories(rows: Sequence[Any]) -> dict[tuple[str, str], str]:
        output: dict[tuple[str, str], str] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            group_id = row.get("groupId")
            package_id = row.get("packageId")
            category = row.get("typeCategory")
            if all(isinstance(value, str) for value in (group_id, package_id, category)):
                output[(cast(str, group_id), cast(str, package_id))] = cast(str, category)
        return output

    def explicit_target_package_surfaces(value: Any) -> set[str]:
        """Return package nouns explicitly authorized by task-facing cargo text."""

        if isinstance(value, str):
            return {
                _semantic_normalize(match.group("package"))
                for match in _RAW_PACKAGE_NOUN_ONLY.finditer(value)
            }
        if isinstance(value, Mapping):
            return set().union(
                *(explicit_target_package_surfaces(item) for item in value.values()),
                set(),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return set().union(
                *(explicit_target_package_surfaces(item) for item in value),
                set(),
            )
        return set()

    source_categories = keyed_categories(cast(Sequence[Any], source_packages))
    target_categories = keyed_categories(cast(Sequence[Any], target_packages))
    if source_categories.keys() != target_categories.keys():
        return tuple(requirements)

    output: list[CargoFlavorRewriteRequirement] = []
    for requirement in requirements:
        match = re.fullmatch(
            r"documentPatch\.cargoGroups\[([0-9]+)\]\.description",
            requirement.targetPath,
        )
        if match is None:
            output.append(requirement)
            continue
        group_index = int(match.group(1))
        if group_index >= len(target_groups) or not isinstance(target_groups[group_index], Mapping):
            output.append(requirement)
            continue
        group_id = target_groups[group_index].get("groupId")
        keys = tuple(key for key in source_categories if key[0] == group_id)
        if not keys:
            output.append(requirement)
            continue
        transition_flags = {source_categories[key] != target_categories[key] for key in keys}
        if len(transition_flags) != 1:
            output.append(requirement)
            continue

        allowed: set[str] = set()
        for key in keys:
            entry = package_registry.entry(target_categories[key])
            display_parts = tuple(
                part.strip() for part in entry.displayName.split(",") if part.strip()
            )
            base = (
                " ".join((*display_parts[1:], display_parts[0]))
                if len(display_parts) > 1
                else display_parts[0]
            )
            allowed.update(
                {
                    _semantic_normalize(base),
                    _semantic_normalize(_pluralize_package_surface(base)),
                }
            )
        if transition_flags == {False}:
            allowed.update(
                _semantic_normalize(match.group("package"))
                for surface in requirement.sourceSurfaces
                for match in _RAW_PACKAGE_NOUN_ONLY.finditer(surface)
            )
        # A package surface can also be literal task truth inside cargo description, marks,
        # additional information, or a dangerous-goods text field.  Such text is explicitly
        # authorized even when it differs from the primary transport-package category; rejecting
        # it would conflate semantic package hierarchy with a word appearing in label-backed
        # prose (for example ``101 PACKAGES PACKED IN STEEL BUNDLES``).
        allowed.update(explicit_target_package_surfaces(target_groups[group_index]))
        output.append(
            requirement.model_copy(
                update={
                    "enforcePackageSurfaceGuard": True,
                    "allowedPackageSurfaces": tuple(sorted(allowed)),
                }
            )
        )
    return tuple(output)


_CARGO_AUXILIARY_END_HEADING = re.compile(
    r"^[ \t]*(?:TOTAL(?:S)?\b|PLACE[ \t]+AND[ \t]+DATE[ \t]+OF[ \t]+ISSUE|"
    r"SHIPPED[ \t]+ON[ \t]+BOARD|FREIGHT(?:[ \t]+AND)?[ \t]+CHARGES|"
    r"CARRIER(?:['\u2019]S)?[ \t]+RECEIPT|SIGNED[ \t]+(?:BY|ON[ \t]+BEHALF))\b",
    re.IGNORECASE,
)


def unexpected_cargo_package_surfaces(
    line: str,
    requirement: CargoFlavorRewriteRequirement,
) -> tuple[str, ...]:
    """Return package nouns not licensed by one target cargo group.

    Package words explicitly present in the target description remain label-backed prose. Every
    other accepted word must be one of the target package-category surfaces frozen on the
    requirement. Keeping this decision in one function prevents discovery, model-slot creation,
    and final validation from drifting apart.
    """

    if not requirement.enforcePackageSurfaceGuard:
        return ()
    allowed = set(requirement.allowedPackageSurfaces)
    normalized_target = _semantic_normalize(requirement.targetDescription)
    unexpected: list[str] = []
    for match in _RAW_PACKAGE_NOUN_ONLY.finditer(line):
        package_surface = match.group("package")
        normalized_package = _semantic_normalize(package_surface)
        described_by_target = (
            re.search(
                rf"(?<![A-Z0-9]){re.escape(normalized_package)}(?![A-Z0-9])",
                normalized_target,
            )
            is not None
        )
        if normalized_package in allowed or described_by_target:
            continue
        # LOT, UNIT, SET, and SHEET are valid package nouns only in package grammar. They are also
        # common non-package words in cargo, rate-table, and page-continuation prose (``EXPORT
        # LOT``, ``Rate Unit``, ``SET AT``, and ``Next Sheet``). Numeric adjacency, an ``... OF``
        # construction, slash-delimited compact packaging, or explicit PACKED/PACKING syntax
        # proves package semantics without relying on a template-specific phrase list.
        if normalized_package in {
            "LOT",
            "LOTS",
            "SET",
            "SETS",
            "SHEET",
            "SHEETS",
            "UNIT",
            "UNITS",
        }:
            prefix = line[: match.start()]
            suffix = line[match.end() :]
            quantity_owned = any(
                package.start("package") == match.start()
                and package.end("package") == match.end()
                for package in _RAW_PACKAGE_TYPE_SURFACE.finditer(line)
            )
            written_quantity_owned = _immediate_written_cardinal(prefix) is not None
            compact_packaging = prefix.rstrip().endswith("/") or suffix.lstrip().startswith("/")
            package_of = re.match(r"[ \t]+OF\b", suffix, re.IGNORECASE) is not None
            packed_context = (
                re.search(
                    r"\bPACK(?:ED|ING)?[ \t]+(?:IN(?:TO)?[ \t]+)?"
                    r"(?:[A-Z0-9-]+[ \t]+){0,3}$",
                    prefix,
                    re.IGNORECASE,
                )
                is not None
            )
            if not (
                quantity_owned
                or written_quantity_owned
                or compact_packaging
                or package_of
                or packed_context
            ):
                continue
        if package_surface not in unexpected:
            unexpected.append(package_surface)
    return tuple(unexpected)


def _immediate_written_cardinal(prefix: str) -> int | None:
    """Parse an English cardinal immediately before a package noun.

    Only a whitespace- or hyphen-delimited suffix of the prefix is considered. Walking suffixes
    from the adjacent word outward lets ``TOTAL ONE LOT`` resolve to ``ONE`` while preventing a
    remote number from licensing ``Business Unit`` or ``temperature ... set at`` prose. The
    existing complete-cardinal parser remains the single authority for accepted number grammar.
    """

    cursor = len(prefix)
    while cursor and prefix[cursor - 1] in " \t-":
        cursor -= 1
    if not cursor or not prefix[cursor - 1].isalpha():
        return None
    last_word_end = cursor
    words: list[tuple[int, int]] = []
    parsed: int | None = None
    # Twenty-one tokens exceeds the longest value emitted by the companion formatter below.
    # Walk that bounded suffix backwards instead of rescanning the whole line for every ambiguous
    # noun; long cargo rows therefore remain linear in their number of package facts.
    for _ in range(21):
        word_end = cursor
        while cursor and prefix[cursor - 1].isalpha():
            cursor -= 1
        if word_end == cursor:
            break
        # No accepted cardinal token is longer than NINETEEN/SEVENTEEN. Reject an unrelated
        # unbounded word immediately rather than walking a pathological free-text token.
        if word_end - cursor > 16:
            break
        words.append((cursor, word_end))
        candidate = prefix[cursor:last_word_end]
        value = _parse_english_cardinal(candidate)
        if value is not None:
            parsed = value
        separator_end = cursor
        while cursor and prefix[cursor - 1] in " \t-":
            cursor -= 1
        if separator_end == cursor or not cursor or not prefix[cursor - 1].isalpha():
            break
    return parsed


def cargo_auxiliary_package_requirements(
    raw_text: str,
    requirements: Sequence[CargoFlavorRewriteRequirement],
    *,
    excluded_line_ids: frozenset[str] = frozenset(),
) -> tuple[CargoFlavorRewriteRequirement, ...]:
    """Add narrowly bounded cargo-detail rows carrying stale package semantics.

    Product and DG detail rows can contain an inner package configuration that is absent from
    the task label (for example ``100BAGS/CARTON``).  When the synthetic target changes the
    package category, those source-only rows must be rewritten too.  Starting at a label-grounded
    cargo row, scan only until the next cargo group or an explicit post-cargo heading, and add
    only short non-legal lines containing a package noun outside that group's target allowlist.
    """

    if not requirements:
        return ()
    lines = raw_text.splitlines()
    ordered = sorted(requirements, key=lambda row: _line_number(row.sourceLineIds[0]))
    preamble_by_requirement: dict[str, int] = {}
    preamble_owner: dict[int, str] = {}
    for requirement in ordered:
        line_number = _preceding_typed_cargo_preamble_line_number(
            lines,
            min((_line_number(value) for value in requirement.sourceLineIds), default=0),
        )
        if line_number is None:
            continue
        previous_owner = preamble_owner.setdefault(line_number, requirement.requirementId)
        if previous_owner != requirement.requirementId:
            raise ValueError(
                "one typed cargo preamble is adjacent to multiple cargo requirements"
            )
        previous_line = preamble_by_requirement.setdefault(
            requirement.requirementId, line_number
        )
        if previous_line != line_number:
            raise ValueError("one cargo requirement has multiple typed preambles")
    auxiliary: list[CargoFlavorRewriteRequirement] = []
    for index, requirement in enumerate(ordered):
        if not requirement.enforcePackageSurfaceGuard:
            continue
        start = max(_line_number(value) for value in requirement.sourceLineIds)
        next_start = (
            min(_line_number(value) for value in ordered[index + 1].sourceLineIds)
            if index + 1 < len(ordered)
            else len(lines) + 1
        )
        selected: list[tuple[str, str]] = []
        own_preamble = preamble_by_requirement.get(requirement.requirementId)
        if own_preamble is not None:
            line = lines[own_preamble - 1]
            current_line_id = _line_id(own_preamble, line)
            if (
                current_line_id not in excluded_line_ids
                and unexpected_cargo_package_surfaces(line, requirement)
            ):
                selected.append((current_line_id, line))
        for line_number in range(start + 1, min(next_start, len(lines) + 1)):
            line = lines[line_number - 1]
            current_line_id = _line_id(line_number, line)
            if _PAGE_MARKER.fullmatch(line) is not None or _CARGO_AUXILIARY_END_HEADING.match(line):
                break
            # A path-owned scalar renderer has stronger semantic evidence than positional
            # proximity to the preceding cargo description.  In particular, container rows
            # between two cargo descriptions must retain the package category allocated to that
            # container; they are not auxiliary packaging prose for the preceding group.
            if current_line_id in excluded_line_ids:
                continue
            if line_number in preamble_owner:
                continue
            if len(line) > 300 or _PRESERVABLE_LEGAL_BOILERPLATE_LINE.fullmatch(line):
                continue
            if unexpected_cargo_package_surfaces(line, requirement):
                selected.append((current_line_id, line))
        if selected:
            auxiliary.append(
                CargoFlavorRewriteRequirement(
                    requirementId=requirement.requirementId + "-auxiliary-packaging",
                    lineRole="source_only_auxiliary_packaging",
                    targetPath=requirement.targetPath,
                    targetDescription=requirement.targetDescription,
                    sourceLineIds=tuple(row[0] for row in selected),
                    sourceSurfaces=tuple(row[1] for row in selected),
                    enforcePackageSurfaceGuard=True,
                    allowedPackageSurfaces=requirement.allowedPackageSurfaces,
                )
            )
    return (*requirements, *auxiliary)


def _typed_cargo_preamble_package_match(line: str) -> re.Match[str] | None:
    """Return the sole noun-bound package fact after a typed containment relation."""

    relation = re.search(
        r"\bCONTAINERS?[ \t]+SAID[ \t]+TO[ \t]+CONTAIN\b", line, re.IGNORECASE
    )
    if relation is None:
        return None
    matches = tuple(_RAW_PACKAGE_TYPE_SURFACE.finditer(line, relation.end()))
    return matches[0] if len(matches) == 1 else None


def _preceding_typed_cargo_preamble_line_number(
    lines: Sequence[str], first_cargo_line_number: int
) -> int | None:
    """Return a strongly typed preamble immediately before one cargo span.

    Carrier forms commonly separate ``Container Said to Contain`` from the description with one
    or two blank OCR rows. The explicit relation and noun-bound numeric quantity establish
    ownership; a page marker, content row, or wider blank gap terminates the lookup.
    """

    if not 1 < first_cargo_line_number <= len(lines):
        return None
    cursor = first_cargo_line_number - 2
    blank_count = 0
    while cursor >= 0 and not lines[cursor].strip():
        blank_count += 1
        if blank_count > 2:
            return None
        cursor -= 1
    if cursor < 0 or _PAGE_MARKER.fullmatch(lines[cursor]) is not None:
        return None
    line = lines[cursor]
    if _typed_cargo_preamble_package_match(line) is None:
        return None
    return cursor + 1


def _render_target_package_surface(
    *,
    package_registry: LoadedPackageRegistry,
    target_category: str,
    plural: bool,
    parenthesized_plural: bool,
    source_surface: str,
) -> str:
    display_name = package_registry.entry(target_category).displayName
    display_parts = tuple(part.strip() for part in display_name.split(",") if part.strip())
    target_base = (
        " ".join((*display_parts[1:], display_parts[0]))
        if len(display_parts) > 1
        else display_parts[0]
    )
    target_surface = (
        target_base
        if parenthesized_plural or not plural
        else _pluralize_package_surface(target_base)
    )
    return _equipment_surface_with_source_case(source_surface, target_surface)


_PACKAGE_SURFACE_TOKEN_ALIASES = {
    "BOXES": "BOX",
    "CARTONS": "CARTON",
    "CTN": "CARTON",
    "CTNS": "CARTON",
    "FIBERBOARD": "FIBREBOARD",
    "LOTS": "LOT",
    "PACKAGES": "PACKAGE",
    "PKG": "PACKAGE",
    "PKGS": "PACKAGE",
    "PALLETS": "PALLET",
    "PLT": "PALLET",
    "PLTS": "PALLET",
    "PCS": "PIECE",
    "PIECES": "PIECE",
    "SETS": "SET",
    "SHEETS": "SHEET",
    "UNITS": "UNIT",
}


def _package_category_surface_key(value: str) -> tuple[str, ...]:
    """Normalize only established package inflections and carrier abbreviations."""

    normalized = _semantic_normalize(value)
    return tuple(
        _PACKAGE_SURFACE_TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[A-Z]+", normalized)
    )


def _package_surface_matches_category(
    surface: str,
    category: str,
    package_registry: LoadedPackageRegistry,
) -> bool:
    """Require a printed package noun to agree with its reviewed source category."""

    display_parts = tuple(
        part.strip()
        for part in package_registry.entry(category).displayName.split(",")
        if part.strip()
    )
    display_surface = (
        " ".join((*display_parts[1:], display_parts[0]))
        if len(display_parts) > 1
        else display_parts[0]
    )
    return _package_category_surface_key(surface) == _package_category_surface_key(display_surface)


def cargo_preamble_package_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    package_registry: LoadedPackageRegistry,
    cargo_requirements: Sequence[CargoFlavorRewriteRequirement],
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Render the complete noun-bound package fact in typed cargo preambles.

    ``1 Container Said to Contain 1 PIECE`` carries two independent counts. Replacing the bare
    package numeral corrupts the container count, so this grammar is compiled as one exact
    ``quantity + noun`` surface owned by the following cargo group. Blank lines are tolerated
    only through the bounded preamble relation proven above.
    """

    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    if not isinstance(source_patch, Mapping) or not isinstance(target_patch, Mapping):
        return ()
    source_packages = source_patch.get("cargoPackages") or ()
    target_packages = target_patch.get("cargoPackages") or ()
    target_groups = target_patch.get("cargoGroups") or ()
    if any(
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
        for value in (source_packages, target_packages, target_groups)
    ):
        return ()

    def keyed_packages(
        rows: Sequence[Any],
    ) -> dict[tuple[str, str], tuple[int, Mapping[str, Any]]]:
        output: dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            group_id = row.get("groupId")
            package_id = row.get("packageId")
            if isinstance(group_id, str) and isinstance(package_id, str):
                output[(group_id, package_id)] = (index, row)
        return output

    source_by_key = keyed_packages(cast(Sequence[Any], source_packages))
    target_by_key = keyed_packages(cast(Sequence[Any], target_packages))
    if source_by_key.keys() != target_by_key.keys():
        raise ValueError("source and target package identities differ")

    source_allocation_groups = source_patch.get("cargoAllocationGroups") or ()
    target_allocation_groups = target_patch.get("cargoAllocationGroups") or ()
    if not isinstance(source_allocation_groups, Sequence) or isinstance(
        source_allocation_groups, (str, bytes)
    ):
        source_allocation_groups = ()
    if not isinstance(target_allocation_groups, Sequence) or isinstance(
        target_allocation_groups, (str, bytes)
    ):
        target_allocation_groups = ()
    source_allocations_by_group = {
        row.get("groupId"): row
        for row in source_allocation_groups
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    target_allocations_by_group = {
        row.get("groupId"): (index, row)
        for index, row in enumerate(target_allocation_groups)
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }

    lines = raw_text.splitlines()
    owned_preambles: dict[int, str] = {}
    output: list[AnchoredScalarReplacementRequirement] = []
    for requirement in cargo_requirements:
        path_match = re.fullmatch(
            r"documentPatch\.cargoGroups\[([0-9]+)\]\.description",
            requirement.targetPath,
        )
        if path_match is None:
            continue
        group_index = int(path_match.group(1))
        if group_index >= len(target_groups) or not isinstance(
            target_groups[group_index], Mapping
        ):
            continue
        group_id = target_groups[group_index].get("groupId")
        if not isinstance(group_id, str):
            continue
        first_line = min(
            (_line_number(line_id) for line_id in requirement.sourceLineIds), default=0
        )
        preamble_line = _preceding_typed_cargo_preamble_line_number(lines, first_line)
        if preamble_line is None:
            continue
        previous_group = owned_preambles.setdefault(preamble_line, group_id)
        if previous_group != group_id:
            raise ValueError("one typed cargo preamble is owned by multiple cargo groups")

        line = lines[preamble_line - 1]
        match = _typed_cargo_preamble_package_match(line)
        if match is None:
            raise RuntimeError("typed cargo preamble predicate changed during compilation")
        printed_quantity = _parse_package_integer_surface(match.group("value"))
        if printed_quantity is None:
            raise RuntimeError("typed cargo preamble quantity grammar is inconsistent")

        group_keys = tuple(key for key in source_by_key if key[0] == group_id)
        package_rows: list[CargoPackageProjectionRow] = []
        for key in group_keys:
            _source_index, source_package = source_by_key[key]
            target_index, target_package = target_by_key[key]
            source_quantity = source_package.get("quantity")
            target_quantity = target_package.get("quantity")
            source_category = source_package.get("typeCategory")
            target_category = target_package.get("typeCategory")
            if (
                not isinstance(source_quantity, int)
                or isinstance(source_quantity, bool)
                or not isinstance(target_quantity, int)
                or isinstance(target_quantity, bool)
                or not isinstance(source_category, str)
                or not isinstance(target_category, str)
            ):
                raise ValueError("typed cargo preamble package semantics are incomplete")
            package_rows.append(
                CargoPackageProjectionRow(
                    key=key,
                    target_index=target_index,
                    source_quantity=source_quantity,
                    target_quantity=target_quantity,
                    source_category=source_category,
                    target_category=target_category,
                )
            )
        source_package_surface = match.group("package")
        exact_rows = tuple(
            row
            for row in package_rows
            if row.source_quantity == printed_quantity
            and _package_surface_matches_category(
                source_package_surface,
                row.source_category,
                package_registry,
            )
        )
        aggregate = (
            bool(package_rows)
            and sum(row.source_quantity for row in package_rows) == printed_quantity
            and len({row.source_category for row in package_rows}) == 1
            and _package_surface_matches_category(
                source_package_surface,
                package_rows[0].source_category,
                package_registry,
            )
        )
        selected_rows: tuple[CargoPackageProjectionRow, ...]
        if len(exact_rows) == 1:
            selected_rows = exact_rows
            target_quantity = exact_rows[0].target_quantity
            target_category = exact_rows[0].target_category
        elif aggregate:
            selected_rows = tuple(package_rows)
            target_quantity = sum(row.target_quantity for row in selected_rows)
            target_categories = {row.target_category for row in selected_rows}
            target_category = (
                next(iter(target_categories))
                if len(target_categories) == 1
                else "PACKAGE_PACKAGE"
            )
        else:
            global_rows = list(package_rows)
            for key in (key for key in source_by_key if key not in group_keys):
                _source_index, source_package = source_by_key[key]
                target_index, target_package = target_by_key[key]
                values = (
                    source_package.get("quantity"),
                    target_package.get("quantity"),
                    source_package.get("typeCategory"),
                    target_package.get("typeCategory"),
                )
                if (
                    not isinstance(values[0], int)
                    or isinstance(values[0], bool)
                    or not isinstance(values[1], int)
                    or isinstance(values[1], bool)
                    or not isinstance(values[2], str)
                    or not isinstance(values[3], str)
                ):
                    raise ValueError("typed cargo preamble package semantics are incomplete")
                global_rows.append(
                    CargoPackageProjectionRow(
                        key=key,
                        target_index=target_index,
                        source_quantity=values[0],
                        target_quantity=values[1],
                        source_category=values[2],
                        target_category=values[3],
                    )
                )
            if (
                global_rows
                and sum(row.source_quantity for row in global_rows) == printed_quantity
                and len({row.source_category for row in global_rows}) == 1
                and _package_surface_matches_category(
                    source_package_surface,
                    global_rows[0].source_category,
                    package_registry,
                )
            ):
                selected_rows = tuple(global_rows)
                target_quantity = sum(row.target_quantity for row in selected_rows)
                target_categories = {row.target_category for row in selected_rows}
                target_category = (
                    next(iter(target_categories))
                    if len(target_categories) == 1
                    else "PACKAGE_PACKAGE"
                )
            else:
                # The line carries a presentation-only outer package level absent from the task
                # package list. Its exact target is linguistic cargo context, so retain it as the
                # correctly associated model-owned preamble rather than inventing a scalar map.
                continue

        source_unchanged = all(
            row.source_quantity == row.target_quantity
            and row.source_category == row.target_category
            for row in selected_rows
        )
        if source_unchanged:
            continue

        parenthesized = match.group("parenthesized_plural") or ""
        target_noun = _render_target_package_surface(
            package_registry=package_registry,
            target_category=target_category,
            plural=target_quantity != 1,
            parenthesized_plural=bool(parenthesized),
            source_surface=match.group("package"),
        )
        source_end = match.end()
        packaging_code = re.match(
            r"[ \t]*-[ \t]*[1-9][A-Z](?:[0-9])?(?![A-Z0-9])",
            line[source_end:],
            re.IGNORECASE,
        )
        if packaging_code is not None:
            source_end += packaging_code.end()
        source_surface = line[match.start("value") : source_end]
        retained_packaging_code = (
            line[match.end() : source_end]
            if packaging_code is not None
            and all(row.source_category == row.target_category for row in selected_rows)
            else ""
        )
        target_surface = (
            _render_integer_surface(target_quantity, source_surface=match.group("value"))
            + line[match.end("value") : match.start("package")]
            + target_noun
            + parenthesized
            + retained_packaging_code
        )
        target_paths: set[str] = set()
        for row in selected_rows:
            if row.source_quantity != row.target_quantity:
                target_paths.add(
                    f"documentPatch.cargoPackages[{row.target_index}].quantity"
                )
            if row.source_category != row.target_category:
                target_paths.add(
                    f"documentPatch.cargoPackages[{row.target_index}].typeCategory"
                )

        if (
            len(selected_rows) == 1
            and selected_rows[0].source_quantity != selected_rows[0].target_quantity
        ):
            selected = selected_rows[0]
            source_allocation = source_allocations_by_group.get(group_id)
            target_binding = target_allocations_by_group.get(group_id)
            if isinstance(source_allocation, Mapping) and target_binding is not None:
                target_group_index, target_allocation = target_binding
                source_rows = source_allocation.get("allocations") or ()
                target_rows = target_allocation.get("allocations") or ()
                if (
                    isinstance(source_rows, Sequence)
                    and not isinstance(source_rows, (str, bytes))
                    and isinstance(target_rows, Sequence)
                    and not isinstance(target_rows, (str, bytes))
                    and len(source_rows) == len(target_rows) == 1
                    and isinstance(source_rows[0], Mapping)
                    and isinstance(target_rows[0], Mapping)
                    and (
                        source_rows[0].get("packageId") in (None, selected.key[1])
                        or selected.key[1] in (source_allocation.get("packageIds") or ())
                    )
                    and source_rows[0].get("packageQuantity") == selected.source_quantity
                    and target_rows[0].get("packageQuantity") == selected.target_quantity
                ):
                    target_paths.add(
                        "documentPatch.cargoAllocationGroups"
                        f"[{target_group_index}].allocations[0].packageQuantity"
                    )
        if not target_paths:
            raise RuntimeError("typed cargo preamble replacement has no target semantics")
        output.append(
            AnchoredScalarReplacementRequirement(
                targetPaths=tuple(sorted(target_paths)),
                sourceLineIds=(_line_id(preamble_line, line),),
                sourceSurface=source_surface,
                targetSurface=target_surface,
            )
        )
    return tuple(output)


def cargo_package_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    package_registry: LoadedPackageRegistry,
    cargo_requirements: Sequence[CargoFlavorRewriteRequirement],
    *,
    excluded_line_ids: frozenset[str] = frozenset(),
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Compile non-overlapping package quantity and noun projections for cargo text."""

    preambles = cargo_preamble_package_replacement_requirements(
        raw_text,
        source_label,
        target_label,
        package_registry,
        cargo_requirements,
    )
    lines = raw_text.splitlines()
    typed_preamble_line_ids = frozenset(
        _line_id(line_number, lines[line_number - 1])
        for requirement in cargo_requirements
        if (
            line_number := _preceding_typed_cargo_preamble_line_number(
                lines,
                min((_line_number(value) for value in requirement.sourceLineIds), default=0),
            )
        )
        is not None
    )
    owned = excluded_line_ids | typed_preamble_line_ids | frozenset(
        line_id for requirement in preambles for line_id in requirement.sourceLineIds
    )
    package_types = cargo_package_type_replacement_requirements(
        raw_text,
        source_label,
        target_label,
        package_registry,
        cargo_requirements,
        excluded_line_ids=owned,
    )
    package_quantities = cargo_package_quantity_replacement_requirements(
        raw_text,
        source_label,
        target_label,
        cargo_requirements,
        excluded_line_ids=owned,
    )
    type_candidates_by_occurrence: dict[
        tuple[str, str], list[tuple[int, AnchoredScalarReplacementRequirement]]
    ] = defaultdict(list)
    for index, requirement in enumerate(package_types):
        for line_id in requirement.sourceLineIds:
            type_candidates_by_occurrence[(line_id, requirement.sourceSurface.casefold())].append(
                (index, requirement)
            )
    quantity_candidates_by_occurrence: dict[
        tuple[str, str], list[tuple[int, AnchoredScalarReplacementRequirement]]
    ] = defaultdict(list)
    for index, requirement in enumerate(package_quantities):
        for line_id in requirement.sourceLineIds:
            quantity_candidates_by_occurrence[
                (line_id, requirement.sourceSurface.casefold())
            ].append((index, requirement))

    package_occurrences: list[
        tuple[int, str, str, re.Match[str], str, str]
    ] = []
    type_occurrence_counts: Counter[tuple[str, str]] = Counter()
    quantity_occurrence_counts: Counter[tuple[str, str]] = Counter()
    for line_number, line in enumerate(lines, start=1):
        line_id = _line_id(line_number, line)
        for match in _RAW_PACKAGE_TYPE_SURFACE.finditer(line):
            type_source_surface = match.group("package")
            packaging_code = re.match(
                r"[ \t]*-[ \t]*[1-9][A-Z](?:[0-9])?(?![A-Z0-9])",
                line[match.end("package") :],
                re.IGNORECASE,
            )
            if packaging_code is not None:
                type_source_surface += packaging_code.group(0)
            quantity_source_surface = match.group("value")
            package_occurrences.append(
                (
                    line_number,
                    line_id,
                    line,
                    match,
                    type_source_surface,
                    quantity_source_surface,
                )
            )
            type_occurrence_counts[(line_id, type_source_surface.casefold())] += 1
            quantity_occurrence_counts[(line_id, quantity_source_surface.casefold())] += 1

    def reject_partially_owned_repeats(
        candidates: Mapping[
            tuple[str, str], Sequence[tuple[int, AnchoredScalarReplacementRequirement]]
        ],
        occurrences: Mapping[tuple[str, str], int],
        *,
        kind: str,
    ) -> None:
        for key, occurrence_count in occurrences.items():
            candidate_count = len(candidates.get(key, ()))
            if 0 < candidate_count != occurrence_count:
                raise ValueError(
                    f"repeated noun-bound package {kind} lacks occurrence-level ownership: "
                    f"{key[0]}, surface={key[1]!r}, occurrences={occurrence_count}, "
                    f"projections={candidate_count}"
                )

    # A line ID plus a literal is insufficient when the same fact occurs twice and only one
    # semantic owner changes. In that case a global anchored substitution would corrupt the
    # unchanged occurrence, so compilation must stop instead of silently choosing an ordinal.
    reject_partially_owned_repeats(
        type_candidates_by_occurrence,
        type_occurrence_counts,
        kind="noun",
    )
    reject_partially_owned_repeats(
        quantity_candidates_by_occurrence,
        quantity_occurrence_counts,
        kind="quantity",
    )

    consumed_types: set[tuple[int, str]] = set()
    consumed_quantities: set[tuple[int, str]] = set()
    compound: list[AnchoredScalarReplacementRequirement] = []
    type_positions: Counter[tuple[str, str]] = Counter()
    quantity_positions: Counter[tuple[str, str]] = Counter()
    for (
        _line_number_value,
        line_id,
        line,
        match,
        type_source_surface,
        quantity_source_surface,
    ) in package_occurrences:
        type_key = (line_id, type_source_surface.casefold())
        quantity_key = (line_id, quantity_source_surface.casefold())
        type_rows = type_candidates_by_occurrence.get(type_key, ())
        quantity_rows = quantity_candidates_by_occurrence.get(quantity_key, ())
        type_position = type_positions[type_key]
        quantity_position = quantity_positions[quantity_key]
        type_candidate = type_rows[type_position] if type_position < len(type_rows) else None
        quantity_candidate = (
            quantity_rows[quantity_position]
            if quantity_position < len(quantity_rows)
            else None
        )
        type_positions[type_key] += 1
        quantity_positions[quantity_key] += 1
        if type_candidate is None and quantity_candidate is None:
            continue
        if type_candidate is not None and quantity_candidate is None:
            # A noun-only projection must not absorb the adjacent quantity. That quantity can be
            # a separately model-owned allocation (for example one repeated 22-PLT row of a
            # 44-pallet total whose target allocations are 4 and 3). Expanding ``PLT -> PALLETS``
            # into ``22 PLT -> 22 PALLETS`` would turn the stale source quantity into a host lock
            # and contradict the allocation edit. The noun requirement already has exact line and
            # surface ownership, so leave it unconsumed unless a deterministic quantity owner is
            # present to justify composing the whole fact.
            continue

        source_end = match.end()
        if type_candidate is not None and len(type_source_surface) > len(match.group("package")):
            source_end += len(type_source_surface) - len(match.group("package"))
        source_start = match.start("value")
        left_context = ""
        if source_start > 0 and line[source_start - 1].isalnum():
            context = re.search(
                r"(?:KGS?|QTY[ \t]*:?|[0-9]+[ \t]*X)$",
                line[:source_start],
                re.IGNORECASE,
            )
            if context is None:
                raise ValueError(
                    "noun-bound package fact is fused to an unknown left-hand grammar"
                )
            source_start = context.start()
            left_context = line[source_start : match.start("value")]
        source_surface = line[source_start:source_end]
        target_value = (
            quantity_candidate[1].targetSurface
            if quantity_candidate is not None
            else match.group("value")
        )
        target_package = (
            type_candidate[1].targetSurface
            if type_candidate is not None
            else match.group("package")
        )
        parenthesized = match.group("parenthesized_plural") or ""
        target_surface = (
            left_context
            + target_value
            + line[match.end("value") : match.start("package")]
            + target_package
            + parenthesized
        )
        owners = tuple(
            row for row in (type_candidate, quantity_candidate) if row is not None
        )
        target_paths = {path for _, requirement in owners for path in requirement.targetPaths}
        compound.append(
            AnchoredScalarReplacementRequirement(
                targetPaths=tuple(sorted(target_paths)),
                sourceLineIds=(line_id,),
                sourceSurface=source_surface,
                targetSurface=target_surface,
            )
        )
        if type_candidate is not None:
            consumed_types.add((type_candidate[0], line_id))
        if quantity_candidate is not None:
            consumed_quantities.add((quantity_candidate[0], line_id))

    def unconsumed(
        requirements: Sequence[AnchoredScalarReplacementRequirement],
        consumed: set[tuple[int, str]],
    ) -> tuple[AnchoredScalarReplacementRequirement, ...]:
        output: list[AnchoredScalarReplacementRequirement] = []
        for index, requirement in enumerate(requirements):
            line_ids = tuple(
                line_id
                for line_id in requirement.sourceLineIds
                if (index, line_id) not in consumed
            )
            if line_ids:
                output.append(requirement.model_copy(update={"sourceLineIds": line_ids}))
        return tuple(output)

    return merge_anchored_scalar_replacement_requirements(
        preambles,
        compound,
        unconsumed(package_types, consumed_types),
        unconsumed(package_quantities, consumed_quantities),
    )


def container_package_type_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    package_registry: LoadedPackageRegistry,
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Render container-row package nouns from allocation-owned target categories.

    Package categories are deterministic task semantics, not prose. This renderer changes only
    the noun captured beside a container-local package quantity, retains its exact line and count,
    and preserves its letter-case style. Alias spelling is retained when the semantic category
    did not change. Ambiguous multi-level allocations receive no write authority here.
    """

    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    source_containers = (
        source_patch.get("containers") if isinstance(source_patch, Mapping) else None
    )
    target_containers = (
        target_patch.get("containers") if isinstance(target_patch, Mapping) else None
    )
    if not isinstance(source_containers, Sequence) or isinstance(source_containers, (str, bytes)):
        return ()
    if not isinstance(target_containers, Sequence) or isinstance(target_containers, (str, bytes)):
        return ()
    if len(source_containers) != len(target_containers):
        raise ValueError("cannot bind package surfaces across different container counts")

    source_categories = _package_categories_by_container(source_label)
    target_categories = _package_categories_by_container(target_label)
    target_quantities: dict[str, int] | None = None
    occurrences = _container_measurement_occurrences(
        raw_text, cast(Sequence[Mapping[str, Any]], source_containers)
    )
    lines = raw_text.splitlines()
    requirements: list[AnchoredScalarReplacementRequirement] = []
    for container_index, observed_rows in occurrences.items():
        source_container = source_containers[container_index]
        target_container = target_containers[container_index]
        if not isinstance(source_container, Mapping) or not isinstance(target_container, Mapping):
            raise ValueError("container package binding received a non-object container")
        source_number = source_container.get("containerNumber")
        target_number = target_container.get("containerNumber")
        if not isinstance(source_number, str) or not isinstance(target_number, str):
            continue
        source_category = source_categories.get(source_number)
        target_category = target_categories.get(target_number)
        if source_category is None or target_category is None:
            continue
        if source_category[0] == target_category[0]:
            continue
        if target_quantities is None:
            # Membership-only relations deliberately omit per-container quantities.  Do not
            # demand that stronger relationship unless a container-owned printed package noun
            # actually needs changing.
            target_quantities = _target_package_allocations(target_label)
        try:
            quantity = target_quantities[target_number]
        except KeyError as error:
            raise ValueError(
                f"target package allocations do not cover printed container {target_number}"
            ) from error
        display_name = package_registry.entry(target_category[0]).displayName.split(",", 1)[0]
        for kind, line_number, source_quantity, _evidence, grammar in observed_rows:
            if kind != "package_quantity" or grammar != "labeled_measurement":
                continue
            line = lines[line_number - 1]
            match = _operational_measurement_match(
                line, "package_quantity", expected_surface=source_quantity
            )
            if match is None:
                raise ValueError("container package noun lost its source quantity grammar")
            source_word = match.group("package")
            # ``PALLET(S)`` is already a source-owned singular/plural envelope.  Pluralizing the
            # captured noun as well would create malformed ``BUNDLES(S)`` surfaces.  Retain the
            # suffix and substitute a singular target noun; ordinary plural source nouns still
            # receive the quantity-aware plural target.
            has_parenthesized_plural_suffix = (
                re.match(r"[ \t]*\(S\)", line[match.end("package") :], re.IGNORECASE) is not None
            )
            target_word = (
                display_name
                if has_parenthesized_plural_suffix or quantity == 1
                else _pluralize_package_surface(display_name)
            )
            rendered_word = _equipment_surface_with_source_case(source_word, target_word)
            if source_word == rendered_word:
                continue
            requirements.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=target_category[1],
                    sourceLineIds=(_line_id(line_number, line),),
                    sourceSurface=source_word,
                    targetSurface=rendered_word,
                    surfaceKind="package_noun",
                )
            )
    return tuple(requirements)


def cargo_package_type_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    package_registry: LoadedPackageRegistry,
    cargo_requirements: Sequence[CargoFlavorRewriteRequirement],
    *,
    excluded_line_ids: frozenset[str] = frozenset(),
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Render every relation-owned package noun whose target category is unambiguous.

    Container-table rendering alone misses shipment totals and repeated cargo rows.  This compiler
    binds a printed count+noun pair to a package identity using only schema relations: the source
    package total, an explicit allocation quantity, a container number on the same row, or the
    cargo span already grounded by the linguistic plan.  It emits an edit only when every possible
    owner resolves to one target category and one singular/plural class.  Ambiguous multi-level
    rows remain model-owned and fail closed.
    """

    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    if not isinstance(source_patch, Mapping) or not isinstance(target_patch, Mapping):
        return ()
    source_packages = source_patch.get("cargoPackages") or ()
    target_packages = target_patch.get("cargoPackages") or ()
    if (
        not isinstance(source_packages, Sequence)
        or isinstance(source_packages, (str, bytes))
        or not isinstance(target_packages, Sequence)
        or isinstance(target_packages, (str, bytes))
    ):
        return ()

    def package_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
        group_id = row.get("groupId")
        package_id = row.get("packageId")
        if not isinstance(group_id, str) or not isinstance(package_id, str):
            return None
        return group_id, package_id

    source_by_key = {
        key: row
        for row in source_packages
        if isinstance(row, Mapping) and (key := package_key(row)) is not None
    }
    target_by_key = {
        key: row
        for row in target_packages
        if isinstance(row, Mapping) and (key := package_key(row)) is not None
    }
    if source_by_key.keys() != target_by_key.keys():
        raise ValueError("source and target package identities differ")

    target_category_by_key: dict[tuple[str, str], str] = {}
    target_category_path_by_key: dict[tuple[str, str], str] = {}
    target_quantity_by_key: dict[tuple[str, str], int] = {}
    source_quantity_keys: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for index, row in enumerate(source_packages):
        if not isinstance(row, Mapping) or (key := package_key(row)) is None:
            continue
        target_row = target_by_key[key]
        source_category = row.get("typeCategory")
        target_category = target_row.get("typeCategory")
        source_quantity = row.get("quantity")
        target_quantity = target_row.get("quantity")
        if (
            not isinstance(source_category, str)
            or not isinstance(target_category, str)
            or not isinstance(source_quantity, int)
            or isinstance(source_quantity, bool)
            or not isinstance(target_quantity, int)
            or isinstance(target_quantity, bool)
        ):
            continue
        # Keep unchanged package identities in the ownership graph.  Omitting them makes a
        # repeated source quantity (for example two cargo groups that each contain ``1
        # PACKAGE``) appear to belong solely to the one package whose category changed.  The
        # compiler can then rewrite both groups with that category.  Rendering below still emits
        # only surfaces that actually differ, so retaining the complete graph adds no no-op edits.
        target_category_by_key[key] = target_category
        target_category_path_by_key[key] = (
            f"documentPatch.cargoPackages[{index}].typeCategory"
        )
        target_quantity_by_key[key] = target_quantity
        source_quantity_keys[source_quantity].add(key)

    if not target_category_by_key:
        return ()

    group_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key in source_by_key:
        if key in target_category_by_key:
            group_keys[key[0]].add(key)

    source_allocations = source_patch.get("cargoAllocationGroups") or ()
    target_allocations = target_patch.get("cargoAllocationGroups") or ()
    if not isinstance(source_allocations, Sequence) or isinstance(
        source_allocations, (str, bytes)
    ):
        source_allocations = ()
    if not isinstance(target_allocations, Sequence) or isinstance(
        target_allocations, (str, bytes)
    ):
        target_allocations = ()
    target_groups = {
        row.get("groupId"): row
        for row in target_allocations
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    container_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    target_allocation_quantities: dict[tuple[str, str], set[int]] = defaultdict(set)
    for source_group in source_allocations:
        if not isinstance(source_group, Mapping):
            continue
        group_id = source_group.get("groupId")
        target_group = target_groups.get(group_id)
        if not isinstance(group_id, str) or not isinstance(target_group, Mapping):
            continue
        source_rows = source_group.get("allocations") or ()
        target_rows = target_group.get("allocations") or ()
        if (
            not isinstance(source_rows, Sequence)
            or isinstance(source_rows, (str, bytes))
            or not isinstance(target_rows, Sequence)
            or isinstance(target_rows, (str, bytes))
            or len(source_rows) != len(target_rows)
        ):
            raise ValueError("source and target package-allocation topology differs")
        default_ids = tuple(
            value
            for value in source_group.get("packageIds") or ()
            if isinstance(value, str)
        )
        for source_row, target_row in zip(source_rows, target_rows, strict=True):
            if not isinstance(source_row, Mapping) or not isinstance(target_row, Mapping):
                continue
            explicit_id = source_row.get("packageId")
            package_ids = (explicit_id,) if isinstance(explicit_id, str) else default_ids
            keys = {
                (group_id, package_id)
                for package_id in package_ids
                if (group_id, package_id) in target_category_by_key
            }
            source_quantity = source_row.get("packageQuantity")
            target_quantity = target_row.get("packageQuantity")
            source_container = source_row.get("containerNumber")
            if isinstance(source_quantity, int) and not isinstance(source_quantity, bool):
                source_quantity_keys[source_quantity].update(keys)
            if isinstance(source_container, str):
                container_keys[source_container].update(keys)
            if isinstance(target_quantity, int) and not isinstance(target_quantity, bool):
                for key in keys:
                    target_allocation_quantities[key].add(target_quantity)

    cargo_line_keys: dict[int, set[tuple[str, str]]] = defaultdict(set)
    source_lines = raw_text.splitlines()
    target_groups_by_index = target_patch.get("cargoGroups") or ()
    for requirement in cargo_requirements:
        match = re.fullmatch(
            r"documentPatch\.cargoGroups\[([0-9]+)\]\.description",
            requirement.targetPath,
        )
        if match is None:
            continue
        index = int(match.group(1))
        if (
            not isinstance(target_groups_by_index, Sequence)
            or isinstance(target_groups_by_index, (str, bytes))
            or index >= len(target_groups_by_index)
            or not isinstance(target_groups_by_index[index], Mapping)
        ):
            continue
        group_id = target_groups_by_index[index].get("groupId")
        if not isinstance(group_id, str):
            continue
        for value in requirement.sourceLineIds:
            cargo_line_keys[_line_number(value)].update(group_keys.get(group_id, ()))

    requirements: list[AnchoredScalarReplacementRequirement] = []
    for line_number, line in enumerate(source_lines, start=1):
        current_line_id = _line_id(line_number, line)
        if current_line_id in excluded_line_ids:
            continue
        same_line_container_keys = set().union(
            *(
                keys
                for container, keys in container_keys.items()
                if re.search(rf"(?<![A-Z0-9]){re.escape(container)}(?![A-Z0-9])", line, re.I)
            ),
            set(),
        )
        for match in _RAW_PACKAGE_TYPE_SURFACE.finditer(line):
            source_quantity = _parse_package_integer_surface(match.group("value"))
            if source_quantity is None:
                continue
            owners = set(source_quantity_keys.get(source_quantity, ()))
            contextual = cargo_line_keys.get(line_number, set())
            if contextual and owners & contextual:
                owners &= contextual
            if same_line_container_keys and owners & same_line_container_keys:
                owners &= same_line_container_keys
            owners = {key for key in owners if key in target_category_by_key}
            target_categories = {target_category_by_key[key] for key in owners}
            if len(target_categories) != 1:
                continue
            target_category = next(iter(target_categories))
            target_quantities = {
                quantity
                for key in owners
                for quantity in (
                    target_allocation_quantities.get(key) or {target_quantity_by_key[key]}
                )
            }
            plurality = {quantity != 1 for quantity in target_quantities}
            if len(plurality) != 1:
                continue
            source_surface = match.group("package")
            target_surface = _render_target_package_surface(
                package_registry=package_registry,
                target_category=target_category,
                plural=next(iter(plurality)),
                parenthesized_plural=bool(match.group("parenthesized_plural")),
                source_surface=source_surface,
            )
            # A UN packaging code can be printed as a suffix of the source package noun
            # (``Fibreboard boxes-4G``).  Once the task-facing package category changes, keeping
            # that code would assert the old packaging material.  The target schema does not
            # select a particular construction variant (for example 4C1 versus 4C2), so remove
            # the stale source code rather than inventing an unsupported subtype.
            packaging_code = re.match(
                r"[ \t]*-[ \t]*[1-9][A-Z](?:[0-9])?(?![A-Z0-9])",
                line[match.end("package") :],
                re.IGNORECASE,
            )
            if packaging_code is not None:
                source_surface += packaging_code.group(0)
            if source_surface == target_surface:
                continue
            requirements.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=tuple(sorted(target_category_path_by_key[key] for key in owners)),
                    sourceLineIds=(current_line_id,),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    surfaceKind="package_noun",
                )
            )

    # Some carriers repeat the package total in a parenthesized English-cardinal clause split
    # over physical OCR lines.  It is the same typed fact as the adjacent numeric total, not free
    # cargo prose.  Bind the noun through the parsed source total and change only its exact line;
    # the existing quantity renderer remains responsible for the cardinal words themselves.
    for match in _RAW_WRITTEN_TOTAL_PACKAGE_TYPE.finditer(raw_text):
        source_quantity = _parse_english_cardinal(match.group("written_count"))
        if source_quantity is None:
            continue
        owners = {
            key
            for key in source_quantity_keys.get(source_quantity, ())
            if key in target_category_by_key
        }
        target_categories = {target_category_by_key[key] for key in owners}
        target_pluralities = {target_quantity_by_key[key] != 1 for key in owners}
        if len(target_categories) != 1 or len(target_pluralities) != 1:
            continue
        package_line_number = raw_text.count("\n", 0, match.start("package")) + 1
        package_line = raw_text.splitlines()[package_line_number - 1]
        package_line_id = _line_id(package_line_number, package_line)
        if package_line_id in excluded_line_ids:
            continue
        source_surface = match.group("package")
        target_surface = _render_target_package_surface(
            package_registry=package_registry,
            target_category=next(iter(target_categories)),
            plural=next(iter(target_pluralities)),
            parenthesized_plural=bool(match.group("parenthesized_plural")),
            source_surface=source_surface,
        )
        if source_surface == target_surface:
            continue
        requirements.append(
            AnchoredScalarReplacementRequirement(
                targetPaths=tuple(sorted(target_category_path_by_key[key] for key in owners)),
                sourceLineIds=(package_line_id,),
                sourceSurface=source_surface,
                targetSurface=target_surface,
                surfaceKind="package_noun",
            )
        )
    return tuple(requirements)


def exact_cargo_line_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    cargo_requirements: Sequence[CargoFlavorRewriteRequirement],
    *,
    excluded_line_ids: frozenset[str] = frozenset(),
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Compile a cargo description when its evidence owns one complete physical line.

    No linguistic decision is needed when the source label value is the entire OCR line and the
    target description is itself one line. A line already owned by any other deterministic
    projection is composite even if the label description happens to equal its complete source
    text; replacing it wholesale would erase or conflict with the independently represented
    fact. Rendering a genuinely plain line locally prevents a model from adding unrelated target
    UN/HS/package facts and removes that line from the paid edit scope. Partial, composite,
    repeated, and multiline descriptions remain under the cargo renderer.
    """

    lines = raw_text.splitlines()
    source_patch = source_label.get("documentPatch")
    source_groups = source_patch.get("cargoGroups") if isinstance(source_patch, Mapping) else None
    if not isinstance(source_groups, Sequence) or isinstance(source_groups, (str, bytes)):
        return ()
    output: list[AnchoredScalarReplacementRequirement] = []
    for requirement in cargo_requirements:
        if (
            len(requirement.sourceLineIds) != 1
            or len(requirement.sourceSurfaces) != 1
            or "\n" in requirement.targetDescription
            or "\r" in requirement.targetDescription
        ):
            continue
        line_id = requirement.sourceLineIds[0]
        if line_id in excluded_line_ids:
            continue
        line_number = _line_number(line_id)
        if not 0 < line_number <= len(lines):
            raise ValueError(f"exact cargo line is outside the OCR: {line_id}")
        source_line = lines[line_number - 1]
        source_surface = requirement.sourceSurfaces[0]
        group_match = re.fullmatch(
            r"documentPatch\.cargoGroups\[([0-9]+)\]\.description",
            requirement.targetPath,
        )
        if group_match is None or int(group_match.group(1)) >= len(source_groups):
            continue
        source_group = source_groups[int(group_match.group(1))]
        source_description = (
            source_group.get("description") if isinstance(source_group, Mapping) else None
        )
        if (
            not isinstance(source_description, str)
            or source_line.strip() != source_surface.strip()
            or source_line.strip() != source_description.strip()
        ):
            continue
        leading = source_line[: len(source_line) - len(source_line.lstrip())]
        trailing = source_line[len(source_line.rstrip()) :]
        target_surface = leading + requirement.targetDescription + trailing
        if target_surface == source_line:
            continue
        output.append(
            AnchoredScalarReplacementRequirement(
                targetPaths=(requirement.targetPath,),
                sourceLineIds=(line_id,),
                sourceSurface=source_line,
                targetSurface=target_surface,
            )
        )
    return tuple(output)


def cargo_package_quantity_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    cargo_requirements: Sequence[CargoFlavorRewriteRequirement],
    *,
    excluded_line_ids: frozenset[str] = frozenset(),
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Project an aggregate package total over its explicit cargo-row breakdown.

    Some templates print one task-facing package total and then repeat a physical row for each
    product or allocation.  Those row values are not separately represented in the task schema.
    When one cargo group has exactly one package level and the complete set of noun-bound rows
    sums exactly to the source total, the rows are a lossless presentation of that total.  Scale
    them with largest-remainder arithmetic so the rendered rows sum exactly to the target total.
    Any multi-package, incomplete, or non-reconciling block remains model-owned.
    """

    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    if not isinstance(source_patch, Mapping) or not isinstance(target_patch, Mapping):
        return ()
    source_packages = source_patch.get("cargoPackages") or ()
    target_packages = target_patch.get("cargoPackages") or ()
    target_groups = target_patch.get("cargoGroups") or ()
    if any(
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
        for value in (source_packages, target_packages, target_groups)
    ):
        return ()

    def keyed_packages(rows: Sequence[Any]) -> dict[tuple[str, str], tuple[int, Mapping[str, Any]]]:
        output: dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            group_id = row.get("groupId")
            package_id = row.get("packageId")
            if not isinstance(group_id, str) or not isinstance(package_id, str):
                continue
            output[(group_id, package_id)] = (index, row)
        return output

    source_by_key = keyed_packages(cast(Sequence[Any], source_packages))
    target_by_key = keyed_packages(cast(Sequence[Any], target_packages))
    if source_by_key.keys() != target_by_key.keys():
        raise ValueError("source and target package identities differ")

    source_quantity_keys: dict[int, set[tuple[str, str]]] = defaultdict(set)
    source_total_by_key: dict[tuple[str, str], int] = {}
    target_quantity_by_key: dict[tuple[str, str], int] = {}
    target_quantity_path_by_key: dict[tuple[str, str], str] = {}
    package_keys_by_group: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, (_source_index, source_row) in source_by_key.items():
        target_index, target_row = target_by_key[key]
        source_quantity = source_row.get("quantity")
        target_quantity = target_row.get("quantity")
        if (
            isinstance(source_quantity, int)
            and not isinstance(source_quantity, bool)
            and isinstance(target_quantity, int)
            and not isinstance(target_quantity, bool)
        ):
            source_quantity_keys[source_quantity].add(key)
            source_total_by_key[key] = source_quantity
            target_quantity_by_key[key] = target_quantity
            target_quantity_path_by_key[key] = (
                f"documentPatch.cargoPackages[{target_index}].quantity"
            )
            package_keys_by_group[key[0]].append(key)

    source_allocation_groups = source_patch.get("cargoAllocationGroups") or ()
    target_allocation_groups = target_patch.get("cargoAllocationGroups") or ()
    if not isinstance(source_allocation_groups, Sequence) or isinstance(
        source_allocation_groups, (str, bytes)
    ):
        source_allocation_groups = ()
    if not isinstance(target_allocation_groups, Sequence) or isinstance(
        target_allocation_groups, (str, bytes)
    ):
        target_allocation_groups = ()
    target_allocations_by_group = {
        row.get("groupId"): (index, row)
        for index, row in enumerate(target_allocation_groups)
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    container_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    source_allocation_by_container_key: dict[tuple[str, tuple[str, str]], int] = {}
    target_allocation_by_container_key: dict[tuple[str, tuple[str, str]], int] = {}
    target_allocation_path_by_container_key: dict[
        tuple[str, tuple[str, str]], str
    ] = {}
    for source_group in source_allocation_groups:
        if not isinstance(source_group, Mapping):
            continue
        group_id = source_group.get("groupId")
        target_group_binding = target_allocations_by_group.get(group_id)
        if not isinstance(group_id, str) or target_group_binding is None:
            continue
        target_group_index, target_group = target_group_binding
        source_rows = source_group.get("allocations") or ()
        target_rows = target_group.get("allocations") or ()
        if (
            not isinstance(source_rows, Sequence)
            or isinstance(source_rows, (str, bytes))
            or not isinstance(target_rows, Sequence)
            or isinstance(target_rows, (str, bytes))
            or len(source_rows) != len(target_rows)
        ):
            raise ValueError("source and target package-allocation topology differs")
        default_ids = tuple(
            value for value in source_group.get("packageIds") or () if isinstance(value, str)
        )
        for allocation_index, (source_row, target_row) in enumerate(
            zip(source_rows, target_rows, strict=True)
        ):
            if not isinstance(source_row, Mapping) or not isinstance(target_row, Mapping):
                continue
            explicit_id = source_row.get("packageId")
            package_ids = (explicit_id,) if isinstance(explicit_id, str) else default_ids
            keys = {
                (group_id, package_id)
                for package_id in package_ids
                if (group_id, package_id) in target_quantity_by_key
            }
            source_quantity = source_row.get("packageQuantity")
            source_container = source_row.get("containerNumber")
            target_quantity = target_row.get("packageQuantity")
            if isinstance(source_quantity, int) and not isinstance(source_quantity, bool):
                source_quantity_keys[source_quantity].update(keys)
            if not isinstance(source_container, str):
                continue
            container_keys[source_container].update(keys)
            if isinstance(target_quantity, int) and not isinstance(target_quantity, bool):
                for key in keys:
                    compound_key = source_container, key
                    if isinstance(source_quantity, int) and not isinstance(
                        source_quantity, bool
                    ):
                        source_allocation_by_container_key[compound_key] = source_quantity
                    target_allocation_by_container_key[compound_key] = target_quantity
                    target_allocation_path_by_container_key[compound_key] = (
                        "documentPatch.cargoAllocationGroups"
                        f"[{target_group_index}].allocations"
                        f"[{allocation_index}].packageQuantity"
                    )

    def equivalent_quantity_paths(key: tuple[str, str]) -> set[str]:
        """Return every task path represented by one printed aggregate quantity.

        The relation projection repeats a package total in ``packageQuantity`` when the whole
        package group belongs to one container.  That is one printed fact with two schema paths,
        not evidence that the OCR must print the same number twice.  Bind both paths only when
        source *and* target prove exact aggregate equivalence; partial allocations remain
        independently anchored or model-owned.
        """

        paths = {target_quantity_path_by_key[key]}
        for compound_key, source_allocation in source_allocation_by_container_key.items():
            _container, allocation_key = compound_key
            if (
                allocation_key == key
                and source_allocation == source_total_by_key[key]
                and target_allocation_by_container_key.get(compound_key)
                == target_quantity_by_key[key]
            ):
                paths.add(target_allocation_path_by_container_key[compound_key])
        return paths

    def anchored_quantity_surfaces(
        match: re.Match[str], target_quantity: int
    ) -> tuple[str, str, Literal["scalar", "measurement"]]:
        """Return the numeric projection consumed by the compound package compiler."""

        source_surface = match.group("value")
        target_surface = _render_integer_surface(
            target_quantity, source_surface=source_surface
        )
        return source_surface, target_surface, "scalar"

    line_ids_by_group: dict[str, set[str]] = defaultdict(set)
    line_keys: dict[int, set[tuple[str, str]]] = defaultdict(set)
    source_lines = raw_text.splitlines()
    for requirement in cargo_requirements:
        match = re.fullmatch(
            r"documentPatch\.cargoGroups\[([0-9]+)\]\.description",
            requirement.targetPath,
        )
        if match is None:
            continue
        group_index = int(match.group(1))
        if group_index >= len(target_groups) or not isinstance(target_groups[group_index], Mapping):
            continue
        group_id = target_groups[group_index].get("groupId")
        if isinstance(group_id, str):
            line_ids_by_group[group_id].update(requirement.sourceLineIds)
            keys = package_keys_by_group.get(group_id, ())
            for line_id in requirement.sourceLineIds:
                line_keys[_line_number(line_id)].update(keys)

    requirements: list[AnchoredScalarReplacementRequirement] = []
    # First cover exact, repeated presentations of a package total or a container allocation.
    # Unlike a row breakdown, each such occurrence carries the complete source quantity and must
    # independently render the corresponding complete target value.  Role context is resolved by
    # the adjacent cargo preamble or the source container on the same row; unresolved collisions
    # remain model-owned and fail closed.
    for line_number, line in enumerate(source_lines, start=1):
        line_id = _line_id(line_number, line)
        if line_id in excluded_line_ids:
            continue
        same_line_containers = tuple(
            container
            for container in container_keys
            if re.search(rf"(?<![A-Z0-9]){re.escape(container)}(?![A-Z0-9])", line, re.I)
        )
        for match in _RAW_PACKAGE_TYPE_SURFACE.finditer(line):
            source_quantity = _parse_package_integer_surface(match.group("value"))
            if source_quantity is None:
                continue
            owners = set(source_quantity_keys.get(source_quantity, ()))
            contextual = line_keys.get(line_number, set())
            if contextual and owners & contextual:
                owners &= contextual
            container_owned = set().union(
                *(container_keys[container] for container in same_line_containers), set()
            )
            if container_owned and owners & container_owned:
                owners &= container_owned
            if not same_line_containers:
                # A quantity in a cargo span can be either a repeated presentation of the
                # package total or one row of an unlabeled breakdown.  Only the former is an
                # exact scalar replacement.  Breakdown rows are reconciled together below;
                # treating each row as the total first creates contradictory edits such as
                # ``472 -> 991`` followed by ``472 -> 249`` on the same source line.
                owners = {
                    key for key in owners if source_total_by_key.get(key) == source_quantity
                }
            if not owners:
                continue
            target_values: set[int] = set()
            target_paths: set[str] = set()
            if same_line_containers:
                for container in same_line_containers:
                    for key in owners:
                        compound_key = container, key
                        if compound_key in target_allocation_by_container_key:
                            target_values.add(target_allocation_by_container_key[compound_key])
                            target_paths.add(
                                target_allocation_path_by_container_key[compound_key]
                            )
                            if (
                                source_allocation_by_container_key.get(compound_key)
                                == source_total_by_key.get(key)
                                and target_allocation_by_container_key[compound_key]
                                == target_quantity_by_key.get(key)
                            ):
                                target_paths.update(equivalent_quantity_paths(key))
            if not target_values:
                target_values.update(target_quantity_by_key[key] for key in owners)
                for key in owners:
                    target_paths.update(equivalent_quantity_paths(key))
            if len(target_values) != 1:
                continue
            source_surface, target_surface, surface_kind = anchored_quantity_surfaces(
                match,
                next(iter(target_values)),
            )
            if target_surface == source_surface:
                continue
            requirements.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=tuple(sorted(target_paths)),
                    sourceLineIds=(line_id,),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    surfaceKind=surface_kind,
                )
            )

    for group_id, keys in sorted(package_keys_by_group.items()):
        if len(keys) != 1:
            continue
        key = keys[0]
        _target_index, target_package = target_by_key[key]
        _source_index, source_package = source_by_key[key]
        source_total = source_package.get("quantity")
        target_total = target_package.get("quantity")
        if (
            not isinstance(source_total, int)
            or isinstance(source_total, bool)
            or not isinstance(target_total, int)
            or isinstance(target_total, bool)
            or source_total <= 0
            or target_total <= 0
            or source_total == target_total
        ):
            continue
        occurrences: list[tuple[str, str, int]] = []
        for line_id in sorted(line_ids_by_group.get(group_id, ()), key=_line_number):
            if line_id in excluded_line_ids:
                continue
            line_number = _line_number(line_id)
            if not 0 < line_number <= len(source_lines):
                raise ValueError(f"cargo package line is outside the OCR: {line_id}")
            line = source_lines[line_number - 1]
            for match in _RAW_PACKAGE_TYPE_SURFACE.finditer(line):
                value = _parse_package_integer_surface(match.group("value"))
                if value is not None and value > 0:
                    occurrences.append((line_id, match.group("value"), value))
        if len(occurrences) < 2 or sum(row[2] for row in occurrences) != source_total:
            continue
        distributed = largest_remainder_allocation(
            target_total,
            tuple(row[2] for row in occurrences),
        )
        if any(value <= 0 for value in distributed):
            raise ValueError(
                "target package total cannot preserve every positive cargo presentation row"
            )
        target_paths = tuple(sorted(equivalent_quantity_paths(key)))
        for (line_id, source_surface, _source_value), projected_quantity in zip(
            occurrences, distributed, strict=True
        ):
            target_surface = _render_integer_surface(
                projected_quantity,
                source_surface=source_surface,
            )
            if source_surface == target_surface:
                continue
            requirements.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=target_paths,
                    sourceLineIds=(line_id,),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    surfaceKind="measurement",
                )
            )
    return tuple(requirements)


def party_country_metadata_replacement_requirements(
    raw_text: str,
    target_label: Mapping[str, Any],
    resources: TargetIntegrityResources,
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Bind explicit shipper/exporter and consignee/importer country metadata.

    These fields are source-only presentation metadata rather than task-label fields of their
    own. Their semantic owner is nevertheless explicit in the printed heading. The target party
    country is resolved against the pinned ISO-3166 snapshot; when the task label omits that
    optional party field, the corresponding export/import route jurisdiction is used. Both
    country names and alpha-2 codes can therefore be rendered locally without asking a model to
    infer a jurisdiction.
    """

    patch = target_label.get("documentPatch")
    parties = patch.get("parties") if isinstance(patch, Mapping) else None
    if not isinstance(parties, Mapping):
        return ()

    target_by_role: dict[str, tuple[str, str, str]] = {}
    for role in ("shipper", "consignee"):
        party = parties.get(role)
        country = party.get("country") if isinstance(party, Mapping) else None
        if isinstance(country, str) and country.strip():
            code = resources.countries.require(
                country, field=f"documentPatch.parties.{role}.country"
            )
            entry = resources.countries.entry(code)
            canonical_surfaces = {
                _semantic_normalize(value)
                for value in (
                    entry.alpha2,
                    entry.alpha3,
                    entry.name,
                    entry.official_name,
                    entry.common_name,
                )
                if value is not None
            }
            if _semantic_normalize(country) not in canonical_surfaces:
                raise ValueError(
                    "synthetic target party country is not a canonical ISO registry surface: "
                    f"{role}={country!r}"
                )
            target_by_role[role] = (
                entry.name,
                code,
                f"documentPatch.parties.{role}.country",
            )
            continue
        direction: Literal["export", "import"] = "export" if role == "shipper" else "import"
        route_code = _target_route_country_code(
            target_label,
            direction=direction,
            resources=resources,
        )
        if route_code is not None:
            entry = resources.countries.entry(route_code)
            target_by_role[role] = (entry.name, route_code, "documentPatch.route")

    requirements: list[AnchoredScalarReplacementRequirement] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        match = _PARTY_COUNTRY_METADATA_LINE.fullmatch(raw_line)
        if match is None:
            continue
        label = _semantic_normalize(match.group("label"))
        role = "consignee" if "CONSIGNEE" in label or "IMPORTER" in label else "shipper"
        target = target_by_role.get(role)
        if target is None:
            raise ValueError(f"country metadata has no synthetic {role} country: {raw_line!r}")
        source_surface = match.group("value")
        source_code = resources.countries.resolve(source_surface)
        source_entry = resources.countries.entry(source_code) if source_code is not None else None
        source_normalized = _semantic_normalize(source_surface)
        source_is_alpha2 = source_entry is not None and source_normalized == _semantic_normalize(
            source_entry.alpha2
        )
        source_is_alpha3 = (
            source_entry is not None
            and source_entry.alpha3 is not None
            and source_normalized == _semantic_normalize(source_entry.alpha3)
        )
        target_entry = resources.countries.entry(target[1])
        if label.endswith("COUNTRY CODE") or source_is_alpha2:
            target_surface = target_entry.alpha2
        elif source_is_alpha3:
            if target_entry.alpha3 is None:
                raise ValueError(f"target country has no alpha-3 code: {target_entry.alpha2}")
            target_surface = target_entry.alpha3
        else:
            target_surface = target[0]
        letters = [character for character in source_surface if character.isalpha()]
        if letters and all(character.isupper() for character in letters):
            target_surface = target_surface.upper()
        elif letters and all(character.islower() for character in letters):
            target_surface = target_surface.lower()
        if source_surface == target_surface:
            continue
        requirements.append(
            AnchoredScalarReplacementRequirement(
                targetPaths=(target[2],),
                sourceLineIds=(_line_id(line_number, raw_line),),
                sourceSurface=source_surface,
                targetSurface=target_surface,
            )
        )
    return tuple(requirements)


def _anchored_scalar_replacements_rendered(
    value: str, requirements: Sequence[AnchoredScalarReplacementRequirement]
) -> bool:
    lines = value.splitlines()
    for requirement in requirements:
        target_pattern = _anchored_requirement_pattern(requirement, requirement.targetSurface)
        source_pattern = _anchored_requirement_pattern(requirement, requirement.sourceSurface)
        for line_id in requirement.sourceLineIds:
            line_number = _line_number(line_id)
            if line_number > len(lines):
                return False
            line = lines[line_number - 1]
            source_may_be_inside_target = (
                requirement.sourceSurface.casefold() in requirement.targetSurface.casefold()
            )
            if target_pattern.search(line) is None or (
                not source_may_be_inside_target and source_pattern.search(line) is not None
            ):
                return False
    return True


def merge_anchored_scalar_replacement_requirements(
    *groups: Sequence[AnchoredScalarReplacementRequirement],
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Merge duplicate exact substitutions and reject contradictory line authority."""

    merged: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    line_targets: dict[tuple[str, str], str] = {}
    for requirement in (row for group in groups for row in group):
        for line_id in requirement.sourceLineIds:
            authority_key = (line_id, requirement.sourceSurface)
            existing_target = line_targets.setdefault(authority_key, requirement.targetSurface)
            if existing_target != requirement.targetSurface:
                raise ValueError(
                    "conflicting deterministic scalar targets for "
                    f"{line_id} and {requirement.sourceSurface!r}: "
                    f"{existing_target!r} != {requirement.targetSurface!r}"
                )
        merged_row = merged.setdefault(
            (requirement.sourceSurface, requirement.targetSurface, requirement.surfaceKind),
            {"paths": set(), "lines": set()},
        )
        merged_row["paths"].update(requirement.targetPaths)
        merged_row["lines"].update(requirement.sourceLineIds)
    return tuple(
        AnchoredScalarReplacementRequirement(
            targetPaths=tuple(sorted(values["paths"])),
            sourceLineIds=tuple(sorted(values["lines"], key=_line_number)),
            sourceSurface=source_surface,
            targetSurface=target_surface,
            surfaceKind=cast(Literal["scalar", "measurement", "package_noun"], surface_kind),
        )
        for (source_surface, target_surface, surface_kind), values in merged.items()
    )


def apply_deterministic_prefills(
    workspace: RewriteWorkspace,
) -> tuple[AppliedDeterministicPrefill, ...]:
    """Apply exact scalar and unambiguous derived-surface substitutions before rewriting.

    Anchored scalars plus target-rendered dates, HS codes, receipt counts, and receipt equipment
    breakdowns have no linguistic degrees of freedom. Doing them locally is cheaper and more
    reliable than asking a model to copy exact values. A derived source surface is prefilled only
    when every requirement for that surface agrees on one target; ambiguous projections stay in
    the model-visible contract and remain fail-closed. The provider sees all resulting lines and
    must preserve them.
    """

    lines = workspace.current_text.splitlines(keepends=True)
    applied: list[AppliedDeterministicPrefill] = []

    # Container-local operational values are arithmetic projections, not language.  Leaving
    # these slots to a model was both expensive and unsafe: a model could repeat the shipment
    # total in every container row or retain a source-only utilization figure.  Requirements are
    # already bound to one physical line and carry a fully rendered, source-grammar-preserving
    # target surface, so apply them before any provider-visible work is compiled.
    operational_by_line: dict[str, list[OperationalFlavorRequirement]] = defaultdict(list)
    for requirement in workspace.operational_flavor_requirements:
        operational_by_line[requirement.sourceLineId].append(requirement)
    for line_id, requirements in sorted(
        operational_by_line.items(), key=lambda row: _line_number(row[0])
    ):
        line_index = _line_number(line_id) - 1
        if not 0 <= line_index < len(lines):
            raise ValueError(f"operational prefill line is outside the OCR: {line_id}")
        # Apply right-to-left so a shorter or longer value cannot shift the source column of a
        # later requirement on the same physical row. Inline tuple requirements share one regex
        # start and are safely re-resolved from their named groups after each substitution.
        for requirement in sorted(
            requirements,
            key=lambda row: row.sourceMeasurementStartColumn,
            reverse=True,
        ):
            before = _line_body(lines[line_index])
            ending = _line_ending(lines[line_index])
            if requirement.sourceGrammar == "inline_container_breakdown":
                match = _RAW_INLINE_CONTAINER_MEASURES.search(before)
                group_name = {
                    "gross_weight_kg": "gross",
                    "volume_m3": "volume",
                    "package_quantity": "packages",
                }[requirement.kind]
            else:
                match = _operational_measurement_match(
                    before,
                    requirement.kind,
                    expected_surface=requirement.targetValueSurface,
                    source_grammar=requirement.sourceGrammar,
                )
                if match is None:
                    match = _operational_measurement_match(
                        before,
                        requirement.kind,
                        expected_surface=requirement.sourceValueSurface,
                        source_grammar=requirement.sourceGrammar,
                    )
                group_name = "value"
            if match is None or match.start() != requirement.sourceMeasurementStartColumn:
                raise ValueError(
                    "operational prefill lost its exact source grammar on "
                    f"{line_id}: {requirement.kind}"
                )
            observed = match.group(group_name)
            if observed == requirement.targetValueSurface:
                continue
            if observed != requirement.sourceValueSurface:
                raise ValueError(
                    "operational prefill source surface differs from its contract on "
                    f"{line_id}: expected={requirement.sourceValueSurface!r}, "
                    f"observed={observed!r}"
                )
            start, end = match.span(group_name)
            after = before[:start] + requirement.targetValueSurface + before[end:]
            lines[line_index] = after + ending
            task_paths: tuple[str, ...] = ()
            if requirement.kind == "package_quantity":
                patch = cast(
                    Mapping[str, Any], workspace.current_target_label.get("documentPatch") or {}
                )
                packages = tuple(
                    row
                    for row in cast(Sequence[Any], patch.get("cargoPackages") or ())
                    if isinstance(row, Mapping)
                )
                package_index = {
                    (row.get("groupId"), row.get("packageId")): index
                    for index, row in enumerate(packages)
                }
                paths: set[str] = set()
                rendered_quantity = _parse_package_integer_surface(requirement.targetValueSurface)
                for group_index, group in enumerate(
                    cast(Sequence[Any], patch.get("cargoAllocationGroups") or ())
                ):
                    if not isinstance(group, Mapping):
                        continue
                    group_id = group.get("groupId")
                    group_package_ids = tuple(
                        value
                        for value in cast(Sequence[Any], group.get("packageIds") or ())
                        if isinstance(value, str)
                    )
                    for allocation_index, allocation in enumerate(
                        cast(Sequence[Any], group.get("allocations") or ())
                    ):
                        if (
                            not isinstance(allocation, Mapping)
                            or allocation.get("containerNumber")
                            != requirement.targetContainerNumber
                        ):
                            continue
                        allocation_quantity = allocation.get("packageQuantity")
                        if allocation_quantity == rendered_quantity:
                            paths.add(
                                "documentPatch.cargoAllocationGroups["
                                f"{group_index}].allocations[{allocation_index}].packageQuantity"
                            )
                        package_id = allocation.get("packageId")
                        owned_ids = (
                            (package_id,) if isinstance(package_id, str) else group_package_ids
                        )
                        for owned_id in owned_ids:
                            index = package_index.get((group_id, owned_id))
                            if index is not None:
                                paths.add(f"documentPatch.cargoPackages[{index}].quantity")
                # A one-container aggregate row can own several package leaves without an
                # explicit relation object.  The operational projection already proved that its
                # target value is the exact sum; bind all leaves so the compiler does not ask a
                # model to rewrite an already-correct deterministic surface.
                if not paths:
                    containers = cast(Sequence[Any], patch.get("containers") or ())
                    quantities = tuple(row.get("quantity") for row in packages)
                    if (
                        len(containers) == 1
                        and quantities
                        and all(isinstance(value, int) and value > 0 for value in quantities)
                        and sum(cast(tuple[int, ...], quantities))
                        == _parse_package_integer_surface(requirement.targetValueSurface)
                    ):
                        paths.update(
                            f"documentPatch.cargoPackages[{index}].quantity"
                            for index in range(len(packages))
                        )
                task_paths = tuple(sorted(paths))
            applied.append(
                AppliedDeterministicPrefill(
                    lineId=line_id,
                    targetPaths=(
                        f"rawOperational.{requirement.requirementId}",
                        f"auxiliary.container[{requirement.targetContainerNumber}]."
                        f"{requirement.kind}",
                        *task_paths,
                    ),
                    sourceSurface=requirement.sourceValueSurface,
                    targetSurface=requirement.targetValueSurface,
                    beforeLine=before,
                    afterLine=after,
                )
            )
    for role_hint in workspace.source_role_hints:
        if role_hint.sourceSurface == role_hint.requiredOutputSurface:
            continue
        line_index = _line_number(role_hint.sourceLineId) - 1
        if not 0 <= line_index < len(lines):
            raise ValueError(
                f"source-role prefill line is outside the OCR: {role_hint.sourceLineId}"
            )
        before = _line_body(lines[line_index])
        ending = _line_ending(lines[line_index])
        if before.strip() != role_hint.sourceSurface:
            raise ValueError(
                "source-role prefill no longer matches its exact OCR line: "
                f"{role_hint.sourceLineId}"
            )
        after = before.replace(role_hint.sourceSurface, role_hint.requiredOutputSurface, 1)
        lines[line_index] = after + ending
        applied.append(
            AppliedDeterministicPrefill(
                lineId=role_hint.sourceLineId,
                targetPaths=("auxiliary.anonymousTransportUnitCount",),
                sourceSurface=role_hint.sourceSurface,
                targetSurface=role_hint.requiredOutputSurface,
                beforeLine=before,
                afterLine=after,
            )
        )

    # Exact DG class surfaces are not represented directly by the task label: the label carries
    # a readable category while the semantic plan carries the printed class (for example 2.2).
    # Render the complete class/UN phrase before scalar UN-number prefills.  Reversing that order
    # creates an intermediate phrase that matches neither the source nor target contract.
    dangerous_groups: dict[str, list[SurfaceRenderingRequirement]] = defaultdict(list)
    for dg_requirement in workspace.surface_requirements:
        if dg_requirement.kind in {
            "dangerous_goods_tuple",
            "dangerous_goods_proper_shipping_name",
        }:
            dangerous_groups[dg_requirement.sourceSurface].append(dg_requirement)
    for source_surface, dg_requirements in dangerous_groups.items():
        target_surfaces = {row.targetSurface for row in dg_requirements}
        expected_occurrences = {row.sourceOccurrences for row in dg_requirements}
        if len(target_surfaces) != 1 or len(expected_occurrences) != 1:
            raise ValueError("one printed dangerous-goods tuple has conflicting target projections")
        target_surface = next(iter(target_surfaces))
        expected = next(iter(expected_occurrences))
        owned_line_ids = tuple(
            sorted(
                {
                    line_id
                    for requirement in dg_requirements
                    for line_id in requirement.sourceLineIds
                },
                key=_line_number,
            )
        )
        candidate_indexes = (
            tuple(_line_number(line_id) - 1 for line_id in owned_line_ids)
            if owned_line_ids
            else tuple(range(len(lines)))
        )
        if any(not 0 <= index < len(lines) for index in candidate_indexes):
            raise ValueError("dangerous-goods source line is outside the OCR")
        found = sum(_line_body(lines[index]).count(source_surface) for index in candidate_indexes)
        if found == 0:
            target_count = sum(_line_body(line).count(target_surface) for line in lines)
            if target_count >= expected:
                continue
            if owned_line_ids:
                # The semantic requirement can span OCR-wrapped text (for example the proper
                # shipping name split across two physical lines).  There is then no exact
                # literal for the host to replace, but every line remains explicitly owned by
                # the bounded cargo editor and is verified by the final surface contract.
                continue
            raise ValueError(
                "deterministic dangerous-goods prefill cannot locate its exact source phrase: "
                f"{source_surface!r}"
            )
        if found > expected:
            raise ValueError(
                "dangerous-goods source phrase count exceeds its contract: "
                f"{source_surface!r}, expected={expected}, found={found}"
            )
        # ``found < expected`` means some owned occurrences are OCR-wrapped rather than exact
        # literals. Replace the exact subset locally; the bounded cargo editor renders the
        # remaining owned lines and the final surface audit proves every requirement.
        target_paths = tuple(sorted({row.targetPath for row in dg_requirements}))
        for line_index in candidate_indexes:
            line = lines[line_index]
            before = _line_body(line)
            if source_surface not in before:
                continue
            ending = _line_ending(line)
            after = before.replace(source_surface, target_surface)
            lines[line_index] = after + ending
            applied.append(
                AppliedDeterministicPrefill(
                    lineId=_line_id(line_index + 1, before),
                    targetPaths=target_paths,
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    beforeLine=before,
                    afterLine=after,
                )
            )
    # Resolve every scalar against one immutable snapshot of its physical line, then apply all
    # non-overlapping spans right-to-left as one transaction. Sequential regex substitutions can
    # otherwise rewrite text generated by an earlier requirement: ``Package -> Unit`` followed
    # by ``UNIT -> UNITS`` used to turn ``Package(s)`` into ``UNITS(s)``. Quantities can suffer
    # the same cascade whenever one target happens to equal another source value.
    scalar_by_line: dict[str, list[AnchoredScalarReplacementRequirement]] = defaultdict(list)
    for requirement in workspace.anchored_scalar_replacement_requirements:
        for line_id in requirement.sourceLineIds:
            scalar_by_line[line_id].append(requirement)
    for line_id, requirements in sorted(
        scalar_by_line.items(), key=lambda row: _line_number(row[0])
    ):
        line_index = _line_number(line_id) - 1
        if not 0 <= line_index < len(lines):
            raise ValueError(f"deterministic prefill line is outside the OCR: {line_id}")
        before = _line_body(lines[line_index])
        ending = _line_ending(lines[line_index])
        planned: dict[tuple[int, int], tuple[str, list[AnchoredScalarReplacementRequirement]]] = {}
        already_rendered: set[int] = set()
        for requirement_index, requirement in enumerate(requirements):
            source_pattern = _anchored_requirement_pattern(requirement, requirement.sourceSurface)
            target_pattern = _anchored_requirement_pattern(requirement, requirement.targetSurface)
            matches = tuple(source_pattern.finditer(before))
            if not matches:
                if target_pattern.search(before) is not None:
                    already_rendered.add(requirement_index)
                    continue
                raise ValueError(
                    "deterministic prefill cannot locate its source scalar on "
                    f"{line_id}: {requirement.sourceSurface!r}"
                )
            for match in matches:
                span = match.span()
                for existing_span in planned:
                    overlaps = span[0] < existing_span[1] and existing_span[0] < span[1]
                    if overlaps and span != existing_span:
                        raise ValueError(
                            "overlapping deterministic scalar spans on "
                            f"{line_id}: {requirement.sourceSurface!r} conflicts with "
                            f"{before[existing_span[0]:existing_span[1]]!r}"
                        )
                existing = planned.get(span)
                if existing is not None:
                    if existing[0] != requirement.targetSurface:
                        raise ValueError(
                            "one deterministic scalar span has conflicting targets on "
                            f"{line_id}: {existing[0]!r} != {requirement.targetSurface!r}"
                        )
                    existing[1].append(requirement)
                else:
                    planned[span] = (requirement.targetSurface, [requirement])
        after = before
        for (start, end), (target_surface, _owners) in sorted(
            planned.items(), reverse=True
        ):
            after = after[:start] + target_surface + after[end:]
        for requirement_index, requirement in enumerate(requirements):
            target_pattern = _anchored_requirement_pattern(requirement, requirement.targetSurface)
            source_pattern = _anchored_requirement_pattern(requirement, requirement.sourceSurface)
            source_may_be_inside_target = (
                requirement.sourceSurface.casefold() in requirement.targetSurface.casefold()
            )
            if target_pattern.search(after) is None or (
                not source_may_be_inside_target and source_pattern.search(after) is not None
            ):
                raise ValueError(f"deterministic prefill did not replace its scalar on {line_id}")
            if requirement_index in already_rendered:
                continue
            applied.append(
                AppliedDeterministicPrefill(
                    lineId=line_id,
                    targetPaths=requirement.targetPaths,
                    sourceSurface=requirement.sourceSurface,
                    targetSurface=requirement.targetSurface,
                    beforeLine=before,
                    afterLine=after,
                )
            )
        lines[line_index] = after + ending

    # Legal carrier principals are exact substrings inside signed/agency phrases.  Their grammar
    # proves role ownership independently of spelling, so render the principal deterministically
    # while leaving any separately named signing agent available for fictionalization.
    for surface_requirement in workspace.surface_requirements:
        if surface_requirement.kind != "carrier_principal_identity":
            continue
        for line_id in surface_requirement.sourceLineIds:
            line_index = _line_number(line_id) - 1
            if not 0 <= line_index < len(lines):
                raise ValueError(f"carrier-principal line is outside the OCR: {line_id}")
            before = _line_body(lines[line_index])
            ending = _line_ending(lines[line_index])
            if surface_requirement.sourceSurface == surface_requirement.targetSurface:
                continue
            source_count = before.count(surface_requirement.sourceSurface)
            if source_count == 0:
                if surface_requirement.targetSurface in before:
                    continue
                raise ValueError(
                    "deterministic carrier-principal prefill cannot locate its exact source "
                    f"on {line_id}: {surface_requirement.sourceSurface!r}"
                )
            if source_count != 1:
                raise ValueError(
                    "carrier-principal source is not unique on its owned line: "
                    f"{line_id}, count={source_count}"
                )
            after = before.replace(
                surface_requirement.sourceSurface,
                surface_requirement.targetSurface,
                1,
            )
            lines[line_index] = after + ending
            applied.append(
                AppliedDeterministicPrefill(
                    lineId=line_id,
                    targetPaths=(surface_requirement.targetPath,),
                    sourceSurface=surface_requirement.sourceSurface,
                    targetSurface=surface_requirement.targetSurface,
                    beforeLine=before,
                    afterLine=after,
                )
            )

    # Carrier logo/header identities are exact whole-line surfaces with independently captured
    # context. Render only those complete lines. A global substring replacement would corrupt
    # longer legal or agency names that happen to contain the same short brand token.
    for surface_requirement in workspace.surface_requirements:
        if (
            surface_requirement.kind != "carrier_header_identity"
            or surface_requirement.sourceSurface == surface_requirement.targetSurface
        ):
            continue
        source_line_indexes = tuple(
            index
            for index, line in enumerate(lines)
            if _line_body(line) == surface_requirement.sourceSurface
        )
        target_line_count = sum(
            _line_body(line) == surface_requirement.targetSurface for line in lines
        )
        if not source_line_indexes:
            if target_line_count >= surface_requirement.sourceOccurrences:
                continue
            raise ValueError(
                "deterministic carrier-header prefill cannot locate its exact source line: "
                f"{surface_requirement.sourceSurface!r}"
            )
        if len(source_line_indexes) != surface_requirement.sourceOccurrences:
            raise ValueError(
                "deterministic carrier-header source line count differs from its contract: "
                f"{surface_requirement.sourceSurface!r}, "
                f"expected={surface_requirement.sourceOccurrences}, "
                f"found={len(source_line_indexes)}"
            )
        for line_index in source_line_indexes:
            before = _line_body(lines[line_index])
            ending = _line_ending(lines[line_index])
            after = surface_requirement.targetSurface
            lines[line_index] = after + ending
            applied.append(
                AppliedDeterministicPrefill(
                    lineId=_line_id(line_index + 1, before),
                    targetPaths=(surface_requirement.targetPath,),
                    sourceSurface=surface_requirement.sourceSurface,
                    targetSurface=surface_requirement.targetSurface,
                    beforeLine=before,
                    afterLine=after,
                )
            )

    deterministic_surface_kinds = {
        "date",
        "date_global",
        "hs_code",
        "hs_code_block",
        "cargo_origin",
        "carrier_receipt_count",
        "carrier_receipt_equipment_breakdown",
        "aggregate_equipment_breakdown",
    }
    surface_groups: dict[str, list[SurfaceRenderingRequirement]] = {}
    for surface_requirement in workspace.surface_requirements:
        if surface_requirement.kind in deterministic_surface_kinds:
            surface_groups.setdefault(surface_requirement.sourceSurface, []).append(
                surface_requirement
            )
    for source_surface, grouped_surface_requirements in surface_groups.items():
        target_surfaces = {row.targetSurface for row in grouped_surface_requirements}
        target_surfaces_by_line: dict[str, set[str]] = defaultdict(set)
        for surface_row in grouped_surface_requirements:
            for line_id in surface_row.sourceLineIds:
                target_surfaces_by_line[line_id].add(surface_row.targetSurface)
        exact_lines_are_unambiguous = all(
            len(line_targets) == 1 for line_targets in target_surfaces_by_line.values()
        )
        if (
            all(row.kind == "hs_code" for row in grouped_surface_requirements)
            and all(row.sourceLineIds for row in grouped_surface_requirements)
            and not exact_lines_are_unambiguous
        ):
            # A dense cargo row may print the same HS code more than once while the aligned
            # target cargo rows intentionally project those occurrences to different codes.
            # ``surface_rendering_requirements`` emits rows in physical-match / cargo-index
            # order. When—and only when—the number of exact occurrences on an owned line equals
            # the number of requirements, that order is a complete positional proof. Render
            # right-to-left so replacements cannot shift later spans. Any count mismatch remains
            # agent-owned and fail-closed.
            rows_by_line: dict[str, list[SurfaceRenderingRequirement]] = defaultdict(list)
            for row in grouped_surface_requirements:
                for line_id in row.sourceLineIds:
                    rows_by_line[line_id].append(row)
            positional_rendered = True
            planned_lines: list[
                tuple[str, int, str, str, list[SurfaceRenderingRequirement]]
            ] = []
            leading_boundary = r"(?<![A-Za-z0-9])" if source_surface[0].isalnum() else ""
            trailing_boundary = r"(?![A-Za-z0-9])" if source_surface[-1].isalnum() else ""
            exact_pattern = re.compile(
                leading_boundary + re.escape(source_surface) + trailing_boundary
            )
            for line_id, line_rows in rows_by_line.items():
                line_index = _line_number(line_id) - 1
                if not 0 <= line_index < len(lines):
                    raise ValueError(f"deterministic HS line is outside the OCR: {line_id}")
                before = _line_body(lines[line_index])
                matches = tuple(exact_pattern.finditer(before))
                if len(matches) != len(line_rows):
                    positional_rendered = False
                    break
                after = before
                for match, row in reversed(tuple(zip(matches, line_rows, strict=True))):
                    after = after[: match.start()] + row.targetSurface + after[match.end() :]
                planned_lines.append((line_id, line_index, before, after, line_rows))
            if positional_rendered:
                for line_id, line_index, before, after, line_rows in planned_lines:
                    ending = _line_ending(lines[line_index])
                    lines[line_index] = after + ending
                    for row in line_rows:
                        applied.append(
                            AppliedDeterministicPrefill(
                                lineId=line_id,
                                targetPaths=tuple(row.targetPath.split(";")),
                                sourceSurface=source_surface,
                                targetSurface=row.targetSurface,
                                beforeLine=before,
                                afterLine=after,
                            )
                        )
                continue
        if (
            all(row.sourceLineIds for row in grouped_surface_requirements)
            and exact_lines_are_unambiguous
        ):
            # Exact line ownership is stronger than a global literal. It also permits one
            # repeated source surface to project to different indexed rows (most commonly an
            # HS code repeated in a cargo table) without asking a model to copy digits. A line
            # shared by conflicting targets is left to the bounded line renderer: the host can
            # prove the line, but not which identical occurrence belongs to which indexed row.
            targets_by_line: dict[str, str] = {}
            paths_by_line: dict[str, set[str]] = defaultdict(set)
            for surface_row in grouped_surface_requirements:
                for line_id in surface_row.sourceLineIds:
                    targets_by_line.setdefault(line_id, surface_row.targetSurface)
                    paths_by_line[line_id].update(surface_row.targetPath.split(";"))
            for line_id in sorted(targets_by_line, key=_line_number):
                line_index = _line_number(line_id) - 1
                if not 0 <= line_index < len(lines):
                    raise ValueError(f"deterministic surface line is outside the OCR: {line_id}")
                before = _line_body(lines[line_index])
                ending = _line_ending(lines[line_index])
                target_surface = targets_by_line[line_id]
                source_count = before.count(source_surface)
                if source_surface == target_surface:
                    continue
                if source_count == 0:
                    if target_surface in before:
                        continue
                    raise ValueError(
                        "exact deterministic prefill cannot locate source or target on "
                        f"{line_id}: {source_surface!r} -> {target_surface!r}"
                    )
                # Multiple identical occurrences on one owned line necessarily share the same
                # target. Replace all; the requirement builder has already rejected any
                # path-conflicting projection on that physical line.
                after = before.replace(source_surface, target_surface)
                if after == before or (
                    source_surface not in target_surface and source_surface in after
                ):
                    raise ValueError(f"exact deterministic surface prefill failed on {line_id}")
                lines[line_index] = after + ending
                applied.append(
                    AppliedDeterministicPrefill(
                        lineId=line_id,
                        targetPaths=tuple(sorted(paths_by_line[line_id])),
                        sourceSurface=source_surface,
                        targetSurface=target_surface,
                        beforeLine=before,
                        afterLine=after,
                    )
                )
            continue
        if all(row.kind == "date" for row in grouped_surface_requirements):
            # The issue and on-board dates can legitimately have the same source literal and
            # different synthetic targets, or one of the two dates may remain unchanged. Their
            # nearest printed headings provide exact field ownership, so every field-bound date
            # is rendered locally. Failure to prove every occurrence is fatal; there is no
            # global replacement fallback for path-conflicting dates.
            assigned_lines: dict[int, str] = {}
            for surface_requirement in grouped_surface_requirements:
                matches = tuple(
                    match
                    for match in re.finditer(re.escape(source_surface), workspace.original_text)
                    if _date_context_path(workspace.original_text, match.start())
                    == surface_requirement.targetPath
                )
                if len(matches) != surface_requirement.sourceOccurrences:
                    raise ValueError(
                        "cannot bind every conflicting date occurrence to its semantic heading: "
                        f"{surface_requirement.targetPath}, "
                        f"expected={surface_requirement.sourceOccurrences}, "
                        f"found={len(matches)}"
                    )
                line_numbers = tuple(
                    sorted(
                        {
                            workspace.original_text.count("\n", 0, match.start()) + 1
                            for match in matches
                        }
                    )
                )
                if len(line_numbers) != len(matches):
                    raise ValueError("multiple conflicting date occurrences share one OCR line")
                for line_number in line_numbers:
                    existing = assigned_lines.setdefault(
                        line_number, surface_requirement.targetSurface
                    )
                    if existing != surface_requirement.targetSurface:
                        raise ValueError(
                            "one OCR date line has conflicting semantic targets: "
                            f"L{line_number:05d}"
                        )
                    line_index = line_number - 1
                    before = _line_body(lines[line_index])
                    ending = _line_ending(lines[line_index])
                    source_pattern = _anchored_scalar_pattern(source_surface)
                    if source_pattern.search(before) is None:
                        raise ValueError(
                            "path-bound date prefill cannot locate source surface on "
                            f"L{line_number:05d}: {source_surface!r}"
                        )
                    literal_replacement = surface_requirement.targetSurface.replace("\\", "\\\\")
                    after = source_pattern.sub(literal_replacement, before)
                    if after == before or source_pattern.search(after) is not None:
                        raise ValueError(f"path-bound date prefill failed on L{line_number:05d}")
                    lines[line_index] = after + ending
                    applied.append(
                        AppliedDeterministicPrefill(
                            lineId=_line_id(line_number, before),
                            targetPaths=(surface_requirement.targetPath,),
                            sourceSurface=source_surface,
                            targetSurface=surface_requirement.targetSurface,
                            beforeLine=before,
                            afterLine=after,
                        )
                    )
            continue
        if len(target_surfaces) != 1:
            if all(row.kind == "date_global" for row in grouped_surface_requirements):
                raise ValueError(
                    "one globally rendered date surface has conflicting target projections: "
                    f"{source_surface!r}"
                )
            # A repeated HS or other derived surface can intentionally project to multiple
            # semantic rows. It is not safe for the deterministic global prefill; retain the
            # unchanged source here so the exact host-owned residual slots render it instead.
            continue
        target_surface = next(iter(target_surfaces))
        if source_surface == target_surface:
            continue
        leading_boundary = r"(?<![A-Za-z0-9])" if source_surface[0].isalnum() else ""
        trailing_boundary = r"(?![A-Za-z0-9])" if source_surface[-1].isalnum() else ""
        source_pattern = re.compile(
            leading_boundary + re.escape(source_surface) + trailing_boundary
        )
        literal_replacement = target_surface.replace("\\", "\\\\")
        target_paths = tuple(sorted({row.targetPath for row in grouped_surface_requirements}))
        matched = False
        for line_index, line in enumerate(lines):
            before = _line_body(line)
            if source_pattern.search(before) is None:
                continue
            matched = True
            ending = _line_ending(line)
            after = source_pattern.sub(literal_replacement, before)
            if after == before:
                raise ValueError(
                    "deterministic surface prefill did not replace "
                    f"{source_surface!r} on line {line_index + 1}"
                )
            lines[line_index] = after + ending
            applied.append(
                AppliedDeterministicPrefill(
                    lineId=_line_id(line_index + 1, before),
                    targetPaths=target_paths,
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    beforeLine=before,
                    afterLine=after,
                )
            )
        if not matched:
            current = "".join(lines)
            if target_surface not in current:
                raise ValueError(
                    "deterministic surface prefill cannot locate source or target surface: "
                    f"{source_surface!r} -> {target_surface!r}"
                )
    updated = "".join(lines)
    if len(updated.splitlines()) != len(workspace.original_text.splitlines()):
        raise ValueError("deterministic prefill changes the source line count")
    if _blank_line_topology(updated) != _blank_line_topology(workspace.original_text):
        raise ValueError("deterministic prefill changes the source blank-line topology")
    if not _anchored_scalar_replacements_rendered(
        updated, workspace.anchored_scalar_replacement_requirements
    ):
        raise ValueError("deterministic prefill did not satisfy every anchored scalar")
    workspace.current_text = updated
    workspace.deterministic_prefills.extend(applied)
    return tuple(applied)


def _introduced_source_line_duplicates(source: str, target: str) -> tuple[str, ...]:
    """Return long source lines whose exact normalized count was increased by editing."""

    def counts(value: str) -> dict[str, int]:
        output: dict[str, int] = {}
        for line in value.splitlines():
            normalized = " ".join(line.split())
            if len(normalized) < 24 or _PAGE_MARKER.fullmatch(normalized):
                continue
            output[normalized] = output.get(normalized, 0) + 1
        return output

    before = counts(source)
    after = counts(target)
    return tuple(sorted(line for line, count in before.items() if after.get(line, 0) > count))


def compact_label_change_contract(
    directives: Sequence[LabelChangeDirective],
) -> tuple[CompactLabelChangeDirective, ...]:
    """Deduplicate schema aliases that require the same printed substitution."""

    grouped: dict[tuple[str, bytes, bytes], list[str]] = {}
    values: dict[tuple[str, bytes, bytes], tuple[JsonValue, JsonValue]] = {}
    for directive in directives:
        source = canonical_json_bytes(directive.sourceValue)
        target = canonical_json_bytes(directive.targetValue)
        key = (directive.action, source, target)
        grouped.setdefault(key, []).append(directive.path)
        values[key] = (directive.sourceValue, directive.targetValue)
    return tuple(
        CompactLabelChangeDirective(
            paths=tuple(paths),
            action=cast(Any, key[0]),
            sourceValue=values[key][0],
            targetValue=values[key][1],
        )
        for key, paths in grouped.items()
    )


def _date_value(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _year(value: str) -> int:
    parsed = int(value)
    if len(value) == 2:
        return 2000 + parsed if parsed <= 79 else 1900 + parsed
    return parsed


def _numeric_date_order(
    match: re.Match[str], expected: date
) -> Literal["ymd", "dmy", "mdy"] | None:
    a, b, c = match.group("a"), match.group("b"), match.group("c")
    candidates: list[tuple[Literal["ymd", "dmy", "mdy"], tuple[int, int, int]]] = []
    if len(a) == 4:
        candidates.append(("ymd", (int(a), int(b), int(c))))
    else:
        candidates.extend(
            (
                ("dmy", (_year(c), int(b), int(a))),
                ("mdy", (_year(c), int(a), int(b))),
            )
        )
    matches: list[Literal["ymd", "dmy", "mdy"]] = []
    for order, parts in candidates:
        try:
            if date(*parts) == expected:
                matches.append(order)
        except ValueError:
            continue
    if "dmy" in matches:
        # The frozen project policy is day-first when a numeric surface is ambiguous.
        return "dmy"
    return matches[0] if matches else None


def _render_numeric_date(match: re.Match[str], target: date, order: str) -> str:
    a, b, c = match.group("a"), match.group("b"), match.group("c")
    separator = match.group("s")

    def padded(value: int, width: int) -> str:
        return str(value).zfill(width) if width > 1 else str(value)

    year = str(target.year) if max(len(a), len(c)) == 4 else str(target.year % 100).zfill(2)
    if order == "ymd":
        values = (year, padded(target.month, len(b)), padded(target.day, len(c)))
    elif order == "dmy":
        values = (padded(target.day, len(a)), padded(target.month, len(b)), year)
    else:
        values = (padded(target.month, len(a)), padded(target.day, len(b)), year)
    return separator.join(values)


def _named_month_number(value: str) -> int | None:
    return _MONTHS.get(value.upper())


def _month_surface(source: str, month: int) -> str:
    full = date(2000, month, 1).strftime("%B")
    rendered = full[: len(source)] if len(source) <= 4 else full
    if source.isupper():
        return rendered.upper()
    if source.islower():
        return rendered.lower()
    return rendered.title()


def _rendered_date_candidates(
    raw_text: str,
    *,
    source: date,
    target: date,
) -> tuple[tuple[int, int, str, str], ...]:
    output: list[tuple[int, int, str, str]] = []
    for match in _NUMERIC_DATE.finditer(raw_text):
        if (order := _numeric_date_order(match, source)) is not None:
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    _render_numeric_date(match, target, order),
                )
            )
    for match in _DAY_NAMED_MONTH_DATE.finditer(raw_text):
        month = _named_month_number(match.group("m"))
        if month is None:
            continue
        try:
            observed = date(_year(match.group("y")), month, int(match.group("d")))
        except ValueError:
            continue
        if observed == source:
            year = (
                str(target.year) if len(match.group("y")) == 4 else str(target.year % 100).zfill(2)
            )
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    match.group("s").join(
                        (
                            str(target.day).zfill(len(match.group("d"))),
                            _month_surface(match.group("m"), target.month),
                            year,
                        )
                    ),
                )
            )
    for match in _NAMED_MONTH_DAY_DELIMITED_DATE.finditer(raw_text):
        month = _named_month_number(match.group("m"))
        if month is None:
            continue
        try:
            observed = date(_year(match.group("y")), month, int(match.group("d")))
        except ValueError:
            continue
        if observed == source:
            year = (
                str(target.year) if len(match.group("y")) == 4 else str(target.year % 100).zfill(2)
            )
            separator = match.group("s")
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    f"{_month_surface(match.group('m'), target.month)}{separator}"
                    f"{str(target.day).zfill(len(match.group('d')))}"
                    f"{match.group('s2')}{match.group('gap')}{year}",
                )
            )
    for match in _NAMED_MONTH_DAY_DATE.finditer(raw_text):
        month = _named_month_number(match.group("m"))
        if month is None:
            continue
        try:
            observed = date(_year(match.group("y")), month, int(match.group("d")))
        except ValueError:
            continue
        if observed == source:
            year = (
                str(target.year) if len(match.group("y")) == 4 else str(target.year % 100).zfill(2)
            )
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    f"{_month_surface(match.group('m'), target.month)}"
                    f"{match.group('month_punct')} "
                    f"{str(target.day).zfill(len(match.group('d')))}{match.group('comma')} {year}",
                )
            )
    for match in _YEAR_NAMED_MONTH_DAY_DATE.finditer(raw_text):
        month = _named_month_number(match.group("m"))
        if month is None:
            continue
        try:
            observed = date(int(match.group("y")), month, int(match.group("d")))
        except ValueError:
            continue
        if observed == source:
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    f"{target.year}{match.group('s1')}"
                    f"{_month_surface(match.group('m'), target.month)}{match.group('s2')}"
                    f"{str(target.day).zfill(len(match.group('d')))}",
                )
            )
    return tuple(output)


def _date_context_path(raw_text: str, start: int) -> str | None:
    window_start = max(0, start - 180)
    window = raw_text[window_start:start]
    markers = tuple(re.finditer(r"(?m)^--- PAGE [1-9][0-9]* ---[ \t]*(?:\r?\n|$)", window))
    if markers:
        window = window[markers[-1].end() :]
    prefix = window
    issue = tuple(_ISSUE_DATE_CONTEXT.finditer(prefix))
    shipped = tuple(_SHIPPED_DATE_CONTEXT.finditer(prefix))
    if not issue and not shipped:
        return None
    if issue and shipped and issue[-1].group(0).upper() == "DATED":
        issue_line_start = prefix.rfind("\n", 0, issue[-1].start()) + 1
        shipped_line_start = prefix.rfind("\n", 0, shipped[-1].start()) + 1
        if issue_line_start == shipped_line_start:
            return "documentPatch.shippedOnBoardDate"
    if issue and (not shipped or issue[-1].start() > shipped[-1].start()):
        return "documentPatch.issueDate"
    return "documentPatch.shippedOnBoardDate"


def _format_numeric_surface(source: str, target_digits: str) -> str:
    groups = tuple(re.finditer(r"[0-9]+", source))
    if not groups:
        raise ValueError("numeric source surface has no digits")
    cursor = 0
    chunks: list[str] = []
    for index, group in enumerate(groups):
        if cursor >= len(target_digits):
            break
        if index:
            previous = groups[index - 1]
            chunks.append(source[previous.end() : group.start()])
        width = len(group.group(0))
        chunks.append(target_digits[cursor : cursor + width])
        cursor += width
    if cursor < len(target_digits):
        chunks.append(target_digits[cursor:])
    return "".join(chunks)


def _hs_code_surface_matches(raw_text: str, source_digits: str) -> tuple[re.Match[str], ...]:
    """Locate one exact HS digit sequence, including inside slash-separated code lists.

    A slash is used both inside formatted codes and between complete codes. Matching the entire
    generic numeric surface therefore turns ``30043929/30043100`` into one 16-digit value. Build
    an exact digit-by-digit pattern for the known source code instead: it preserves any printed
    separators while allowing each list member to be projected independently.
    """

    if re.fullmatch(r"[0-9]{4,18}", source_digits) is None:
        raise ValueError(f"invalid source HS digits: {source_digits!r}")
    separator = r"(?:[./-]?[ \t]*)"
    pattern = re.compile(
        r"(?<![0-9])" + separator.join(re.escape(digit) for digit in source_digits) + r"(?![0-9])"
    )
    return tuple(pattern.finditer(raw_text))


def _carrier_receipt_target_count(
    observed: int,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> int:
    """Project an explicit carrier-receipt count through matching source label quantities."""

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    candidates: list[int] = []
    source_containers = source_patch.get("containers") or []
    target_containers = target_patch.get("containers") or []
    if len(source_containers) == observed:
        candidates.append(len(target_containers))
    source_packages = source_patch.get("cargoPackages") or []
    target_packages = target_patch.get("cargoPackages") or []
    for source_package, target_package in zip(source_packages, target_packages, strict=False):
        if not isinstance(source_package, Mapping) or not isinstance(target_package, Mapping):
            continue
        source_quantity = source_package.get("quantity")
        target_quantity = target_package.get("quantity")
        if source_quantity == observed and isinstance(target_quantity, int):
            candidates.append(target_quantity)
    source_quantities: list[int] = []
    target_quantities: list[int] = []
    for row in source_packages:
        quantity = row.get("quantity") if isinstance(row, Mapping) else None
        if isinstance(quantity, int) and not isinstance(quantity, bool):
            source_quantities.append(quantity)
    for row in target_packages:
        quantity = row.get("quantity") if isinstance(row, Mapping) else None
        if isinstance(quantity, int) and not isinstance(quantity, bool):
            target_quantities.append(quantity)
    if source_quantities and sum(source_quantities) == observed and target_quantities:
        candidates.append(sum(target_quantities))
    resolved = set(candidates)
    if not resolved:
        raise ValueError(
            "carrier-receipt count is not grounded in source container/package quantities: "
            f"{observed}"
        )
    if len(resolved) != 1:
        raise ValueError(
            "carrier-receipt count has conflicting target projections: "
            f"source={observed}, targets={sorted(resolved)}"
        )
    return next(iter(resolved))


def _target_equipment_counts(target_label: Mapping[str, Any]) -> dict[str, int]:
    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    counts: dict[str, int] = {}
    for container in patch.get("containers") or []:
        if not isinstance(container, Mapping):
            raise ValueError("target container is not an object")
        size = container.get("sizeCategory")
        if not isinstance(size, str):
            # Template projection deliberately removes the semantic equipment pair when the
            # source OCR has no slot capable of expressing it.  A document-level size summary
            # can only be rewritten from a *complete* fleet, so retain the source summary rather
            # than publishing a misleading partial count.
            return {}
        if size.startswith("TWENTY_FOOT"):
            token = "20"
        elif size.startswith("FORTY_FIVE_FOOT"):
            token = "45"
        elif size.startswith("FORTY_FOOT"):
            token = "40"
        else:
            raise ValueError(f"unsupported target equipment size category: {size}")
        counts[token] = counts.get(token, 0) + 1
    return counts


def _target_equipment_semantic_counts(
    target_label: Mapping[str, Any],
) -> tuple[tuple[str, str, int], ...]:
    """Count exact semantic equipment pairs while preserving target encounter order."""

    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    ordered: list[tuple[str, str]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for container in patch.get("containers") or []:
        if not isinstance(container, Mapping):
            raise ValueError("target container is not an object")
        size = container.get("sizeCategory")
        category = container.get("typeCategory")
        if not isinstance(size, str) or not isinstance(category, str):
            # See ``_target_equipment_counts``: an aggregate equipment clause represents the
            # whole fleet and must never be rendered from only the containers whose target pair
            # survived template projection.
            return ()
        pair = (size, category)
        if pair not in counts:
            ordered.append(pair)
        counts[pair] += 1
    return tuple((size, category, counts[(size, category)]) for size, category in ordered)


def _parenthetical_equipment_components(
    match: re.Match[str],
) -> tuple[tuple[str, str, int, tuple[str, str]], ...]:
    """Parse every exact equipment/count component from a carrier total clause."""

    body = match.group("body")
    components = tuple(_PARENTHETICAL_EQUIPMENT_COMPONENT.finditer(body))
    if not components:
        return ()
    residue = _PARENTHETICAL_EQUIPMENT_COMPONENT.sub("", body)
    if re.sub(r"[ \t&,;+]", "", residue):
        return ()
    parsed: list[tuple[str, str, int, tuple[str, str]]] = []
    for component in components:
        equipment = component.group("equipment")
        reviewed = review_source_equipment_surface(equipment, temperature_present=False)
        if reviewed.size_category is None or reviewed.type_category is None:
            return ()
        parsed.append(
            (
                equipment,
                component.group("x"),
                int(component.group("count").replace(",", "")),
                (reviewed.size_category, reviewed.type_category),
            )
        )
    return tuple(parsed)


def _aggregate_equipment_breakdown_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[SurfaceRenderingRequirement, ...]:
    """Project document-level equipment summaries without model inference.

    Three reviewed source grammars are supported: a split ``4X 40' CONTAINER SAID TO`` clause, a
    compact ``5 X 40HC`` continuation-sheet summary, and a carrier total such as ``SAY: EIGHT
    (20DRX8) CONTAINERS ONLY.``.  A row acquires equipment semantics only when its count equals the
    complete source-container topology and its local syntax proves the row's role. Mixed targets
    are always rendered as one group per exact semantic size/type pair; they can therefore never
    be flattened into one stale source type.
    """

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    source_containers = source_patch.get("containers") or ()
    if not isinstance(source_containers, Sequence) or isinstance(source_containers, (str, bytes)):
        return ()
    if not source_containers:
        return ()
    lines = raw_text.splitlines()

    def next_nonempty_after(index: int) -> str | None:
        return next(
            (line for line in lines[index + 1 : min(len(lines), index + 4)] if line.strip()),
            None,
        )

    def compact_summary_is_owned(index: int, match: re.Match[str]) -> bool:
        source_count = int(match.group("count").replace(",", ""))
        if source_count != len(source_containers):
            return False
        following = next_nonempty_after(index)
        if following is not None and re.match(r"^[ \t]*CONTINUED\b", following, re.I):
            return True
        if re.search(r"\b(?:CONTAINERS?|CNTRS?)\b", match.group(0), re.I):
            return True
        equipment_surface = _semantic_normalize(match.group("equipment"))
        printed_surfaces = tuple(
            _semantic_normalize(cast(str, container.get("typeDescription")))
            for container in source_containers
            if isinstance(container, Mapping)
            and isinstance(container.get("typeDescription"), str)
        )
        return len(printed_surfaces) == source_count and set(printed_surfaces) == {
            equipment_surface
        }

    has_owned_summary = any(
        (
            _SPLIT_ANONYMOUS_EQUIPMENT_LINE.fullmatch(line) is not None
            and (following := next_nonempty_after(index)) is not None
            and re.match(r"^[ \t]*CONTAIN\b", following, re.I) is not None
        )
        or (
            (counted := _COUNTED_AGGREGATE_EQUIPMENT_LINE.fullmatch(line)) is not None
            and int(counted.group("count").replace(",", "")) == len(source_containers)
        )
        or (
            (compact := _COMPACT_AGGREGATE_EQUIPMENT_LINE.fullmatch(line)) is not None
            and compact_summary_is_owned(index, compact)
        )
        or _PARENTHETICAL_AGGREGATE_EQUIPMENT_LINE.fullmatch(line) is not None
        for index, line in enumerate(lines)
    )
    if not has_owned_summary:
        return ()
    target_pairs = _target_equipment_semantic_counts(target_label)
    if not target_pairs:
        return ()
    target_count = sum(count for _size, _category, count in target_pairs)
    candidate_rows: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for index, source_surface in enumerate(lines):
        parenthetical_match = _PARENTHETICAL_AGGREGATE_EQUIPMENT_LINE.fullmatch(source_surface)
        if parenthetical_match is not None and source_surface not in seen:
            components = _parenthetical_equipment_components(parenthetical_match)
            source_count = sum(component[2] for component in components)
            written_count = _parse_english_cardinal(parenthetical_match.group("written_count"))
            source_pair_counts = Counter(
                pair for _equipment, _x, count, pair in components for _ in range(count)
            )
            source_pairs = tuple(
                (size, category, source_pair_counts[(size, category)])
                for size, category in dict.fromkeys(component[3] for component in components)
            )
            if (
                components
                and source_count == len(source_containers)
                and written_count == source_count
                and target_pairs != source_pairs
            ):
                x_surface = components[0][1]
                rendered_groups = [
                    canonical_equipment_surface(
                        cast(bill_of_lading_v5.ContainerSizeCategory, size),
                        cast(bill_of_lading_v5.ContainerTypeCategory, category),
                    )
                    + x_surface
                    + str(count)
                    for size, category, count in target_pairs
                ]
                target_surface = (
                    parenthetical_match.group("prefix")
                    + _render_english_cardinal_like(
                        target_count,
                        source_surface=parenthetical_match.group("written_count"),
                    )
                    + parenthetical_match.group("open")
                    + " + ".join(rendered_groups)
                    + parenthetical_match.group("close")
                    + ("CONTAINER" if target_count == 1 else "CONTAINERS")
                    + parenthetical_match.group("suffix")
                )
                candidate_rows.append((index, source_surface, target_surface))
                seen.add(source_surface)
            continue

        counted_match = _COUNTED_AGGREGATE_EQUIPMENT_LINE.fullmatch(source_surface)
        if counted_match is not None and source_surface not in seen:
            source_count = int(counted_match.group("count").replace(",", ""))
            if source_count == len(source_containers):
                rendered_groups = [
                    f"{count}{counted_match.group('x')}"
                    + canonical_equipment_surface(
                        cast(bill_of_lading_v5.ContainerSizeCategory, size),
                        cast(bill_of_lading_v5.ContainerTypeCategory, category),
                    )
                    for size, category, count in target_pairs
                ]
                target_surface = (
                    counted_match.group("indent")
                    + " + ".join(rendered_groups)
                    + counted_match.group("space")
                    + counted_match.group("container")
                    + counted_match.group("suffix")
                    + counted_match.group("trailing")
                )
                candidate_rows.append((index, source_surface, target_surface))
                seen.add(source_surface)
            continue

        split_match = _SPLIT_ANONYMOUS_EQUIPMENT_LINE.fullmatch(source_surface)
        if split_match is not None:
            source_count = int(split_match.group("count").replace(",", ""))
            next_nonempty = next_nonempty_after(index)
            if (
                source_surface not in seen
                and source_count == len(source_containers)
                and next_nonempty is not None
                and re.match(r"^[ \t]*CONTAIN\b", next_nonempty, re.I) is not None
            ):
                rendered_groups = [
                    f"{count}{split_match.group('x')}"
                    + canonical_equipment_surface(
                        cast(bill_of_lading_v5.ContainerSizeCategory, size),
                        cast(bill_of_lading_v5.ContainerTypeCategory, category),
                    )
                    for size, category, count in target_pairs
                ]
                container_word = "CONTAINER" if target_count == 1 else "CONTAINERS"
                target_surface = (
                    split_match.group("indent")
                    + " + ".join(rendered_groups)
                    + " "
                    + container_word
                    + split_match.group("suffix")
                    + split_match.group("trailing")
                )
                candidate_rows.append((index, source_surface, target_surface))
                seen.add(source_surface)
            continue

        compact_match = _COMPACT_AGGREGATE_EQUIPMENT_LINE.fullmatch(source_surface)
        if compact_match is None or source_surface in seen:
            continue
        source_count = int(compact_match.group("count").replace(",", ""))
        if not compact_summary_is_owned(index, compact_match):
            continue
        reviewed_summary = review_source_equipment_surface(
            compact_match.group("equipment"), temperature_present=False
        )
        source_pairs: set[tuple[str, str]] = set()
        for container in source_containers:
            if not isinstance(container, Mapping):
                raise ValueError("source container is not an object")
            printed = next(
                (
                    cast(str, container[key])
                    for key in ("typeDescription", "typeCode")
                    if isinstance(container.get(key), str)
                ),
                None,
            )
            reviewed = review_source_equipment_surface(printed, temperature_present=False)
            if reviewed.size_category is None or reviewed.type_category is None:
                source_pairs.clear()
                break
            source_pairs.add((reviewed.size_category, reviewed.type_category))
        summary_pair = (
            reviewed_summary.size_category,
            reviewed_summary.type_category,
        )
        source_surface_equivalent = (
            reviewed_summary.size_category is not None
            and reviewed_summary.type_category is not None
            and source_pairs == {cast(tuple[str, str], summary_pair)}
        )
        source_surface_literal = all(
            isinstance(container, Mapping)
            and isinstance(container.get("typeDescription"), str)
            and _semantic_normalize(cast(str, container["typeDescription"]))
            == _semantic_normalize(compact_match.group("equipment"))
            for container in source_containers
        )
        if not source_surface_equivalent and not source_surface_literal:
            continue
        if source_surface_equivalent and target_pairs == (
            (cast(str, summary_pair[0]), cast(str, summary_pair[1]), source_count),
        ):
            continue
        rendered_groups = [
            f"{count}{compact_match.group('x')}"
            + canonical_equipment_surface(
                cast(bill_of_lading_v5.ContainerSizeCategory, size),
                cast(bill_of_lading_v5.ContainerTypeCategory, category),
            )
            for size, category, count in target_pairs
        ]
        target_surface = (
            compact_match.group("indent")
            + " + ".join(rendered_groups)
            + compact_match.group("trailing")
        )
        candidate_rows.append((index, source_surface, target_surface))
        seen.add(source_surface)
    if not candidate_rows:
        return ()

    requirements: list[SurfaceRenderingRequirement] = []
    for index, source_surface, target_surface in candidate_rows:
        source_occurrences = lines.count(source_surface)
        for container_index in range(target_count):
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="aggregate_equipment_breakdown",
                    targetPath=(
                        f"documentPatch.containers[{container_index}].printedEquipmentSurface"
                    ),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    sourceOccurrences=source_occurrences,
                    contextEvidence=_bounded_line_context(
                        lines,
                        focus_index=index,
                        before=1,
                        after=2,
                    ),
                )
            )
    return tuple(requirements)


def _replace_regex_groups(
    value: str,
    match: re.Match[str],
    replacements: Mapping[str, str],
) -> str:
    """Replace named regex groups without disturbing any surrounding punctuation or spacing."""

    rendered = value
    offset = match.start()
    for group_name, target in sorted(
        replacements.items(), key=lambda row: match.start(row[0]), reverse=True
    ):
        start, end = match.span(group_name)
        start -= offset
        end -= offset
        rendered = rendered[:start] + target + rendered[end:]
    return rendered


def _dangerous_goods_tuple_surfaces(
    raw_text: str,
    *,
    source_un_number: str,
    target_un_number: str,
    target_exact_class: str,
    target_packing_group: str | None,
) -> tuple[tuple[str, str, bool], ...]:
    """Render exact class/UN phrases from source grammar and a pinned semantic plan."""

    if re.fullmatch(_EXACT_DG_CLASS_PATTERN, target_exact_class) is None:
        raise ValueError(f"unsupported exact dangerous-goods class: {target_exact_class!r}")
    packing_surface = {
        "HIGH_DANGER": "I",
        "MEDIUM_DANGER": "II",
        "LOW_DANGER": "III",
        "NOT_ASSIGNED": "NOT ASSIGNED",
    }
    if target_packing_group is not None and target_packing_group not in packing_surface:
        raise ValueError(f"unsupported dangerous-goods packing group: {target_packing_group!r}")
    output: list[tuple[str, str, bool]] = []
    for pattern in (_DG_CLASS_BEFORE_UN, _DG_UN_BEFORE_CLASS):
        for match in pattern.finditer(raw_text):
            if match.group("un") != source_un_number:
                continue
            source_surface = match.group(0)
            replacements = {"class": target_exact_class, "un": target_un_number}
            source_has_packing_group = match.group("pg_clause") is not None
            if source_has_packing_group:
                if target_packing_group is None:
                    replacements["pg_clause"] = ""
                else:
                    replacements["pg"] = packing_surface[target_packing_group]
            target_surface = _replace_regex_groups(
                source_surface,
                match,
                replacements,
            )
            output.append((source_surface, target_surface, source_has_packing_group))
    return tuple(dict.fromkeys(output))


def _dangerous_goods_proper_shipping_name_surfaces(
    raw_text: str,
    *,
    source_un_number: str,
    target_un_number: str,
    target_proper_shipping_name: str,
) -> tuple[tuple[str, str, str], ...]:
    """Rewrite a free-text proper-shipping-name tail attached to an explicit UN number.

    Some templates print ``UN 3077 ENVIRONMENTALLY HAZARDOUS ...`` without separately labeling
    a class or packing group. Replacing only the UN digits leaves the source substance name
    attached to the synthetic UN number. This grammar owns the complete physical line only when
    everything after the UN number is free text; structured ``CLASS``, ``PG`` and flash-point
    clauses remain under the exact tuple renderer instead.
    """

    un_pattern = re.compile(
        r"(?<![A-Z0-9])UN(?:DG)?(?:[ \t]+NUMBER|[ \t]+NO\.?)?[ \t]*:?[ \t]*"
        rf"(?P<un>{re.escape(source_un_number)})(?![0-9])",
        re.IGNORECASE,
    )
    structured_tail = re.compile(
        r"\b(?:(?:IMDG[ \t]+)?CLASS|PACKING[ \t]+GROUP|FLASH[ \t]*POINT)\b|"
        r"(?<![A-Z0-9])P\.?[ \t]*G\.?(?![A-Z0-9])|"
        r"(?<![A-Z0-9])FP(?![A-Z0-9])",
        re.IGNORECASE,
    )
    output: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        match = un_pattern.search(line)
        if match is None:
            continue
        tail = line[match.end() :]
        if (
            not re.search(r"[A-Za-z]", tail)
            or structured_tail.search(line) is not None
            or _INLINE_SLOT_LABEL.search(tail) is not None
        ):
            continue
        leading = len(tail) - len(tail.lstrip())
        trailing = len(tail) - len(tail.rstrip())
        separator = tail[:leading]
        suffix = tail[len(tail) - trailing :] if trailing else ""
        source_name = tail.strip()
        letters = tuple(character for character in source_name if character.isalpha())
        target_name = target_proper_shipping_name
        if letters and all(character.isupper() for character in letters):
            target_name = target_name.upper()
        elif letters and all(character.islower() for character in letters):
            target_name = target_name.lower()
        rendered = line[: match.start("un")] + target_un_number + separator + target_name + suffix
        if rendered != line:
            output.append((_line_id(line_number, line), line, rendered))
    return tuple(output)


def _dangerous_goods_context_surfaces(
    raw_text: str,
    *,
    source_un_number: str,
    target_un_number: str,
    target_proper_shipping_name: str,
    target_exact_class: str,
    target_packing_group: str | None,
) -> tuple[tuple[str, str, str, str], ...]:
    """Render separately printed DG name, class, and dense tuple fields near a source UN row."""

    lines = raw_text.splitlines()
    un_token = re.compile(
        r"(?<![A-Z0-9])UN(?:DG)?(?:[ \t]+NUMBER|[ \t]+NO\.?|[ \t]*#)?[ \t]*:?[ \t]*"
        r"(?:UN[ \t]*)?"
        rf"{re.escape(source_un_number)}(?![0-9])",
        re.IGNORECASE,
    )
    un_lines = tuple(index for index, line in enumerate(lines) if un_token.search(line))
    if not un_lines:
        return ()
    packing_surface = {
        "HIGH_DANGER": "I",
        "MEDIUM_DANGER": "II",
        "LOW_DANGER": "III",
        "NOT_ASSIGNED": "NOT ASSIGNED",
    }
    target_pg = packing_surface.get(target_packing_group) if target_packing_group else None
    proper_name = re.compile(
        r"(?P<prefix>(?:PSN|SUBSTANCE[ \t]+NAME[ \t]*\([ \t]*PROPER[ \t]+"
        r"SHIPPING[ \t]+NAME[ \t]*\))[ \t]*:[ \t]*)"
        r"(?P<value>[^\r\n]*?)(?P<suffix>\*?[ \t]*)$",
        re.IGNORECASE,
    )
    labeled_class = re.compile(
        rf"(?P<prefix>\bCLASS[ \t]*:[ \t]*)(?P<value>{_EXACT_DG_CLASS_PATTERN})",
        re.IGNORECASE,
    )
    dense_tuple = re.compile(
        rf"^(?P<leading>[ \t]*)(?P<class>{_EXACT_DG_CLASS_PATTERN})"
        rf"(?P<s1>[ \t]+){re.escape(source_un_number)}"
        r"(?:(?P<s2>[ \t]+)(?P<pg>I{1,3}|NOT[ \t]+ASSIGNED))?"
        r"(?P<trailing>[ \t]*)$",
        re.IGNORECASE,
    )
    # DG declarations are commonly split across a compact leading class/UN/PG row, a blank,
    # a heading, a proper-shipping-name row, a product row, and a labeled tuple row.  Six lines
    # on either side is the smallest symmetric window covering that complete observed grammar;
    # every mutation inside it must still match an exact source UN or a dedicated DG syntax.
    context_radius = 6
    relevant_lines = {
        index
        for un_index in un_lines
        for index in range(
            max(0, un_index - context_radius),
            min(len(lines), un_index + context_radius + 1),
        )
    }
    output: list[tuple[str, str, str, str]] = []
    for index in sorted(relevant_lines):
        line = lines[index]
        line_id = _line_id(index + 1, line)
        match = proper_name.search(line)
        if match is not None and match.group("value").strip():
            source_surface = match.group("value")
            source_letters = tuple(
                character for character in source_surface if character.isalpha()
            )
            target_surface = target_proper_shipping_name
            if source_letters and all(character.isupper() for character in source_letters):
                target_surface = target_surface.upper()
            elif source_letters and all(character.islower() for character in source_letters):
                target_surface = target_surface.lower()
            if source_surface != target_surface:
                output.append((line_id, source_surface, target_surface, "properShippingName"))

        # ``Label/Subrisk: CLASS ...`` is rendered by the complete tuple grammar.  This scalar
        # class rule covers a separate product/chemical line such as ``... Class: 9``.
        for class_match in labeled_class.finditer(line):
            source_surface = class_match.group(0)
            if (
                _DG_CLASS_BEFORE_UN.search(line) is not None
                or _DG_UN_BEFORE_CLASS.search(line) is not None
            ):
                continue
            target_surface = class_match.group("prefix") + target_exact_class
            if source_surface != target_surface:
                output.append((line_id, source_surface, target_surface, "exactHazardClass"))

        dense_match = dense_tuple.fullmatch(line)
        if dense_match is not None:
            rendered = (
                dense_match.group("leading")
                + target_exact_class
                + dense_match.group("s1")
                + target_un_number
            )
            if target_pg is not None:
                rendered += (dense_match.group("s2") or " ") + target_pg
            rendered += dense_match.group("trailing")
            if rendered != line:
                output.append((line_id, line, rendered, "denseTuple"))
    return tuple(dict.fromkeys(output))


def _dangerous_goods_rendering_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    linguistic_plan: DocumentLinguisticPlan,
) -> tuple[SurfaceRenderingRequirement, ...]:
    """Bind broad task categories to exact class surfaces supplied by the semantic plan."""

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    source_groups = source_patch.get("cargoGroups") or ()
    target_groups = target_patch.get("cargoGroups") or ()
    plan_groups = {group.groupId: group for group in linguistic_plan.cargoSeed.cargoGroups}
    if len(source_groups) != len(target_groups):
        raise ValueError("source and target dangerous-goods cargo topology differs")
    requirements: list[SurfaceRenderingRequirement] = []
    raw_lines = raw_text.splitlines()
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups, strict=True)
    ):
        if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
            raise ValueError("dangerous-goods cargo group is not an object")
        group_id = target_group.get("groupId")
        if not isinstance(group_id, str) or group_id not in plan_groups:
            raise ValueError(f"dangerous-goods cargo group has no semantic plan: {group_id!r}")
        source_rows = source_group.get("dangerousGoods") or ()
        target_rows = target_group.get("dangerousGoods") or ()
        plan_rows = plan_groups[group_id].dangerousGoods
        if len(source_rows) != len(target_rows) or len(target_rows) != len(plan_rows):
            raise ValueError(f"dangerous-goods tuple topology differs for cargo group {group_id!r}")
        for dangerous_index, (source_row, target_row, plan_row) in enumerate(
            zip(source_rows, target_rows, plan_rows, strict=True)
        ):
            if not isinstance(source_row, Mapping) or not isinstance(target_row, Mapping):
                raise ValueError("dangerous-goods tuple is not an object")
            target_un = target_row.get("unNumber")
            target_category = target_row.get("hazardCategory")
            if target_un != plan_row.unNumber or (
                target_category is not None and target_category != plan_row.hazardCategory
            ):
                raise ValueError(
                    "target dangerous-goods tuple differs from its pinned semantic plan: "
                    f"{group_id!r}[{dangerous_index}]"
                )
            source_un = source_row.get("unNumber")
            if not isinstance(source_un, str) or not isinstance(target_un, str):
                continue
            surfaces = _dangerous_goods_tuple_surfaces(
                raw_text,
                source_un_number=source_un,
                target_un_number=target_un,
                target_exact_class=plan_row.exactHazardClass,
                target_packing_group=plan_row.packingGroupCategory,
            )
            for line_id, source_surface, target_surface in (
                _dangerous_goods_proper_shipping_name_surfaces(
                    raw_text,
                    source_un_number=source_un,
                    target_un_number=target_un,
                    target_proper_shipping_name=plan_row.properShippingName,
                )
            ):
                requirements.append(
                    SurfaceRenderingRequirement(
                        kind="dangerous_goods_proper_shipping_name",
                        targetPath=(
                            f"auxiliary.cargoGroups[{group_index}].dangerousGoods["
                            f"{dangerous_index}].properShippingName"
                        ),
                        sourceSurface=source_surface,
                        targetSurface=target_surface,
                        sourceOccurrences=sum(
                            candidate == source_surface for candidate in raw_lines
                        ),
                        contextEvidence=source_surface[:600],
                        sourceLineIds=(line_id,),
                    )
                )
            for line_id, source_surface, target_surface, field_name in (
                _dangerous_goods_context_surfaces(
                    raw_text,
                    source_un_number=source_un,
                    target_un_number=target_un,
                    target_proper_shipping_name=plan_row.properShippingName,
                    target_exact_class=plan_row.exactHazardClass,
                    target_packing_group=plan_row.packingGroupCategory,
                )
            ):
                requirements.append(
                    SurfaceRenderingRequirement(
                        kind="dangerous_goods_tuple",
                        targetPath=(
                            f"auxiliary.cargoGroups[{group_index}].dangerousGoods["
                            f"{dangerous_index}].{field_name}"
                        ),
                        sourceSurface=source_surface,
                        targetSurface=target_surface,
                        sourceOccurrences=sum(
                            source_surface in candidate for candidate in raw_lines
                        ),
                        contextEvidence=raw_lines[_line_number(line_id) - 1][:600],
                        sourceLineIds=(line_id,),
                    )
                )
            if not surfaces:
                # The template does not print an exact class/UN tuple.  Do not add a new field;
                # the task-facing category can remain supported by whatever sparse DG evidence
                # the source template actually contains.
                continue
            target_paths = []
            if source_un != target_un:
                target_paths.append(
                    f"documentPatch.cargoGroups[{group_index}].dangerousGoods["
                    f"{dangerous_index}].unNumber"
                )
            if target_category is not None and (
                source_row.get("hazardCategory") != target_category
                or any(source != target for source, target, _has_pg in surfaces)
            ):
                target_paths.append(
                    f"documentPatch.cargoGroups[{group_index}].dangerousGoods["
                    f"{dangerous_index}].hazardCategory"
                )
            for source_surface, target_surface, source_has_packing_group in surfaces:
                source_occurrences = raw_text.count(source_surface)
                matching_line_ids = tuple(
                    _line_id(index + 1, line)
                    for index, line in enumerate(raw_lines)
                    if source_surface in line
                )
                first_line = raw_lines[_line_number(matching_line_ids[0]) - 1]
                for target_path in target_paths:
                    requirements.append(
                        SurfaceRenderingRequirement(
                            kind="dangerous_goods_tuple",
                            targetPath=target_path,
                            sourceSurface=source_surface,
                            targetSurface=target_surface,
                            sourceOccurrences=source_occurrences,
                            contextEvidence=first_line[:600],
                            sourceLineIds=matching_line_ids,
                        )
                    )
                if source_has_packing_group:
                    requirements.append(
                        SurfaceRenderingRequirement(
                            kind="dangerous_goods_tuple",
                            targetPath=(
                                f"documentPatch.cargoGroups[{group_index}].dangerousGoods["
                                f"{dangerous_index}].packingGroupCategory"
                                if plan_row.packingGroupCategory is not None
                                else (
                                    f"auxiliary.cargoGroups[{group_index}].dangerousGoods["
                                    f"{dangerous_index}].packingGroup"
                                )
                            ),
                            sourceSurface=source_surface,
                            targetSurface=target_surface,
                            sourceOccurrences=source_occurrences,
                            contextEvidence=first_line[:600],
                            sourceLineIds=matching_line_ids,
                        )
                    )
    return tuple(requirements)


def _carrier_receipt_equipment_breakdown_requirements(
    raw_text: str, target_label: Mapping[str, Any]
) -> tuple[SurfaceRenderingRequirement, ...]:
    lines = raw_text.splitlines()
    counts = _target_equipment_counts(target_label)
    if not counts:
        return ()
    output: list[SurfaceRenderingRequirement] = []
    for index, line in enumerate(lines):
        if _CARRIER_RECEIPT_HEADING.search(line) is None:
            continue
        for candidate_index in range(index + 1, min(len(lines), index + 5)):
            source_surface = lines[candidate_index]
            if not source_surface.strip():
                continue
            if _EQUIPMENT_BREAKDOWN_LINE.fullmatch(source_surface) is None:
                break
            source_order = tuple(dict.fromkeys(re.findall(r"(?:20|40|45)(?=')", source_surface)))
            target_order = source_order + tuple(
                token
                for token in ("40", "20", "45")
                if token in counts and token not in source_order
            )
            target_surface = " + ".join(
                f"{counts[token]} X {token}'" for token in target_order if counts.get(token, 0) > 0
            )
            output.append(
                SurfaceRenderingRequirement(
                    kind="carrier_receipt_equipment_breakdown",
                    targetPath="auxiliary.carrierReceiptEquipmentBreakdown",
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    sourceOccurrences=lines.count(source_surface),
                    contextEvidence=_bounded_line_context(
                        lines,
                        focus_index=candidate_index,
                        before=candidate_index - max(0, index - 1),
                        after=min(len(lines) - 1, candidate_index + 1) - candidate_index,
                    ),
                )
            )
            break
    return tuple(output)


def _bounded_line_context(
    lines: Sequence[str],
    *,
    focus_index: int,
    before: int,
    after: int,
    max_characters: int = 600,
) -> str:
    """Build bounded audit context while always retaining the matched line.

    OCR occasionally places an entire bill-of-lading clause on one adjacent line.  Joining a
    fixed number of neighboring lines can therefore exceed the evidence schema even when the
    matched surface itself is short.  The matched line is separately preserved as
    ``sourceSurface``; this excerpt keeps that line in full when possible and admits nearest
    neighbors only while they fit the explicit evidence bound.
    """

    if not 0 <= focus_index < len(lines):
        raise ValueError("line-context focus is outside the document")
    if before < 0 or after < 0 or max_characters <= 0:
        raise ValueError("line-context bounds must be non-negative with a positive size")
    focus = lines[focus_index]
    if len(focus) > max_characters:
        return focus[:max_characters]
    selected: dict[int, str] = {focus_index: focus}
    for distance in range(1, max(before, after) + 1):
        for candidate in (focus_index - distance, focus_index + distance):
            if candidate < focus_index - before or candidate > focus_index + after:
                continue
            if not 0 <= candidate < len(lines) or candidate in selected:
                continue
            proposed = {**selected, candidate: lines[candidate]}
            rendered = "\n".join(proposed[index] for index in sorted(proposed))
            if len(rendered) <= max_characters:
                selected = proposed
    return "\n".join(selected[index] for index in sorted(selected))


def _carrier_header_target_surface(
    source_surface: str,
    *,
    source_carrier: str,
    target_carrier: str,
) -> str | None:
    """Project a one- or two-token carrier logo without a carrier-specific alias table.

    A short logo in the page header is either the carrier's leading brand tokens or the initials
    of its printed name. Preserve that representation class for the synthetic carrier; longer
    identity lines continue to use the complete target name.
    """

    source_words = re.findall(r"[A-Z0-9]+", _semantic_normalize(source_carrier))
    target_words = re.findall(r"[A-Z0-9]+", _semantic_normalize(target_carrier))
    surface_words = re.findall(r"[A-Z0-9]+", _semantic_normalize(source_surface))
    if not 1 <= len(surface_words) <= 2 or not source_words or not target_words:
        return None
    if surface_words == source_words[: len(surface_words)]:
        target = " ".join(target_words[: len(surface_words)])
    else:
        if len(surface_words) != 1:
            return None
        surface = surface_words[0]
        legal_suffixes = {
            "A",
            "AG",
            "AS",
            "CO",
            "CORP",
            "GMBH",
            "INC",
            "LTD",
            "LLC",
            "NV",
            "PLC",
            "SA",
            "SAC",
            "SPA",
        }
        source_initials = "".join(word[0] for word in source_words if word not in legal_suffixes)
        if surface != source_initials or len(surface) < 2:
            return None
        target = "".join(word[0] for word in target_words if word not in legal_suffixes)
        if len(target) < 2:
            return None
    rendered = target
    return rendered if source_surface.isupper() else rendered.title()


def _flexible_literal_line_numbers(raw_text: str, value: str) -> set[int]:
    pieces = value.strip().split()
    if not pieces:
        return set()
    pattern = re.compile(r"\s+".join(re.escape(piece) for piece in pieces), re.IGNORECASE)
    output: set[int] = set()
    for match in pattern.finditer(raw_text):
        first = raw_text.count("\n", 0, match.start()) + 1
        last = raw_text.count("\n", 0, max(match.start(), match.end() - 1)) + 1
        output.update(range(first, last + 1))
    return output


def _same_paragraph_line_distance(
    raw_lines: Sequence[str], owned_lines: frozenset[int], target_line: int
) -> int | None:
    """Return distance to a unique semantic row without crossing a structural boundary."""

    if not owned_lines:
        return None
    if target_line in owned_lines:
        return 0
    nearest = min(owned_lines, key=lambda line: (abs(line - target_line), line))
    lower, upper = sorted((nearest, target_line))
    intervening = raw_lines[lower : upper - 1] if upper - lower > 1 else ()
    if any(
        not line.strip() or _PAGE_MARKER.fullmatch(line) is not None
        for line in intervening
    ):
        return None
    return abs(nearest - target_line)


def _nearby_equipment_line_distance(
    raw_lines: Sequence[str], owned_lines: frozenset[int], target_line: int
) -> int | None:
    """Bind a nearby equipment row without treating decorative blank rows as boundaries."""

    if not owned_lines:
        return None
    nearest = min(owned_lines, key=lambda line: (abs(line - target_line), line))
    lower, upper = sorted((nearest, target_line))
    intervening = raw_lines[lower : upper - 1] if upper - lower > 1 else ()
    if any(_PAGE_MARKER.fullmatch(line) is not None for line in intervening):
        return None
    return abs(nearest - target_line)


def surface_rendering_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    linguistic_plan: DocumentLinguisticPlan | None = None,
) -> tuple[SurfaceRenderingRequirement, ...]:
    """Derive exact host-owned surfaces from source formatting and canonical targets."""

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    requirements: list[SurfaceRenderingRequirement] = []
    source_carrier = _party_name(source_label, "carrier")
    target_carrier = _party_name(target_label, "carrier")
    if source_carrier is not None and target_carrier is not None:
        source_tokens = set(re.findall(r"[A-Z0-9]+", _semantic_normalize(source_carrier)))
        compound_carrier_lines = (
            _flexible_literal_line_numbers(raw_text, source_carrier)
            if _compound_relationships(source_carrier)
            else set()
        )
        seen_carrier_header_surfaces: set[str] = set()
        raw_lines = raw_text.splitlines()
        carrier_principals: dict[int, str] = {}
        principal_patterns = (
            _RAW_AGENT_FOR_CARRIER_LINE,
            _RAW_INLINE_AGENT_FOR_CARRIER_LINE,
            _RAW_SIGNED_ON_BEHALF_LINE,
            _RAW_SIGNED_FOR_CARRIER_LINE,
        )
        for line_number, raw_line in enumerate(raw_lines, start=1):
            if line_number in compound_carrier_lines:
                continue
            for pattern in principal_patterns:
                match = pattern.fullmatch(raw_line)
                if match is None:
                    continue
                principal = match.groupdict().get("principal")
                if principal is None or not principal.strip():
                    continue
                observed = principal.strip()
                previous = carrier_principals.setdefault(line_number, observed)
                if previous != observed:
                    raise ValueError(
                        "one legal carrier-principal line has conflicting parses: "
                        f"L{line_number:05d}"
                    )
        for line_number, source_principal in sorted(carrier_principals.items()):
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="carrier_principal_identity",
                    targetPath="documentPatch.parties.carrier.name",
                    sourceSurface=source_principal,
                    targetSurface=target_carrier,
                    sourceOccurrences=1,
                    contextEvidence=raw_lines[line_number - 1][:600],
                    sourceLineIds=(f"L{line_number:05d}",),
                )
            )
        first_party_heading = next(
            (
                index
                for index, line in enumerate(raw_lines, start=1)
                if _PARTY_HEADING_LINE.fullmatch(line)
            ),
            min(len(raw_lines) + 1, 16),
        )
        for line_number, source_surface in enumerate(raw_lines, start=1):
            if line_number in compound_carrier_lines:
                # A wrapped legal identity is owned by the compound-party requirement. Treating
                # one fragment as an independent logo destroys the relationship before the
                # residual compiler can realize its fictional secondary identity.
                continue
            normalized_surface = _semantic_normalize(source_surface)
            surface_tokens = set(re.findall(r"[A-Z0-9]+", normalized_surface))
            overlap = source_tokens & surface_tokens
            short_header_target = (
                _carrier_header_target_surface(
                    source_surface,
                    source_carrier=source_carrier,
                    target_carrier=target_carrier,
                )
                if line_number < first_party_heading
                else None
            )
            if normalized_surface == _semantic_normalize(source_carrier) or (
                short_header_target is None
                and (
                    len(surface_tokens) < 3
                    or len(overlap) / len(surface_tokens) < 0.8
                    or len(overlap) / len(source_tokens) < 0.6
                )
            ):
                continue
            if normalized_surface in seen_carrier_header_surfaces:
                continue
            seen_carrier_header_surfaces.add(normalized_surface)
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="carrier_header_identity",
                    targetPath="documentPatch.parties.carrier.name",
                    sourceSurface=source_surface,
                    targetSurface=short_header_target or target_carrier,
                    sourceOccurrences=raw_text.splitlines().count(source_surface),
                    contextEvidence=_bounded_line_context(
                        raw_lines,
                        focus_index=line_number - 1,
                        before=2,
                        after=2,
                    ),
                    sourceLineIds=tuple(
                        f"L{index:05d}"
                        for index, line in enumerate(raw_lines, start=1)
                        if line == source_surface
                    ),
                )
            )
    date_changes: dict[str, tuple[date, date]] = {}
    targets_by_source_date: dict[date, set[date]] = {}
    for date_field in ("issueDate", "shippedOnBoardDate"):
        source = _date_value(source_patch.get(date_field))
        target = _date_value(target_patch.get(date_field))
        if source is None or target is None:
            continue
        targets_by_source_date.setdefault(source, set()).add(target)
        if source != target:
            date_changes[date_field] = (source, target)

    emitted_global_dates: set[date] = set()
    for date_field, (source, target) in date_changes.items():
        path = f"documentPatch.{date_field}"
        candidates = _rendered_date_candidates(raw_text, source=source, target=target)
        requirement_kind: Literal["date", "date_global"]
        if len(targets_by_source_date[source]) == 1:
            # A repeated date is globally safe only when every labeled field carrying the source
            # date—including unchanged fields—projects it to the same target date. Emit one
            # document-wide contract so unheaded copies and mixed print formats are all shifted,
            # without falsely demanding the document-wide count under each individual heading.
            if source in emitted_global_dates:
                continue
            emitted_global_dates.add(source)
            selected_rows = candidates
            requirement_kind = "date_global"
            requirement_path = ";".join(
                f"documentPatch.{field}"
                for field, values in date_changes.items()
                if values == (source, target)
            )
        else:
            # Conflicting projections—including one unchanged field—must remain heading-bound.
            # Keep every repeated page copy for this path; a global replacement would overwrite
            # the other field with the wrong date.
            selected_rows = tuple(
                row for row in candidates if _date_context_path(raw_text, row[0]) == path
            )
            requirement_kind = "date"
            requirement_path = path
        if not selected_rows:
            raise ValueError(f"cannot locate an unambiguous source date surface for {path}")
        rows_by_surface: dict[tuple[str, str], list[tuple[int, int, str, str]]] = {}
        for row in selected_rows:
            rows_by_surface.setdefault((row[2], row[3]), []).append(row)
        for (source_surface, target_surface), surface_rows in rows_by_surface.items():
            start, end, _, _ = surface_rows[0]
            requirements.append(
                SurfaceRenderingRequirement(
                    kind=requirement_kind,
                    targetPath=requirement_path,
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    sourceOccurrences=len(surface_rows),
                    contextEvidence=raw_text[max(0, start - 90) : min(len(raw_text), end + 40)],
                    sourceLineIds=tuple(
                        f"L{line_number:05d}"
                        for line_number in sorted(
                            {
                                raw_text.count("\n", 0, row_start) + 1
                                for row_start, _row_end, _source, _target in surface_rows
                            }
                        )
                    ),
                )
            )

    source_groups = source_patch.get("cargoGroups") or []
    target_groups = target_patch.get("cargoGroups") or []
    # Cargo descriptions provide reviewed, group-specific physical row ownership. Reuse that
    # ownership for HS surfaces: the same printed code commonly occurs in several container rows,
    # while each synthetic cargo group deliberately receives a different replacement code. A
    # document-order modulo assignment crosses those row boundaries and produces internally
    # inconsistent cargo. This index is deterministic and derives only from label group order and
    # exact source line IDs established by ``cargo_flavor_rewrite_requirements``.
    cargo_hs_lines_by_group_index: dict[int, frozenset[int]] = {}
    for cargo_requirement in cargo_flavor_rewrite_requirements(
        raw_text, source_label, target_label
    ):
        cargo_match = re.match(
            r"^documentPatch\.cargoGroups\[([0-9]+)\]", cargo_requirement.targetPath
        )
        if cargo_match is None:
            continue
        cargo_hs_lines_by_group_index[int(cargo_match.group(1))] = frozenset(
            _line_number(line_id) for line_id in cargo_requirement.sourceLineIds
        )

    hs_pairs: dict[str, list[tuple[int, str, str]]] = {}
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups, strict=False)
    ):
        if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
            continue
        source_codes = source_group.get("hsCodes") or []
        target_codes = target_group.get("hsCodes") or []
        for code_index, (source_code, target_code) in enumerate(
            zip(source_codes, target_codes, strict=False)
        ):
            if not isinstance(source_code, str) or not isinstance(target_code, str):
                continue
            source_digits = "".join(re.findall(r"[0-9]", source_code))
            target_digits = "".join(re.findall(r"[0-9]", target_code))
            if source_digits == target_digits:
                continue
            path = f"documentPatch.cargoGroups[{group_index}].hsCodes[{code_index}]"
            hs_pairs.setdefault(source_digits, []).append(
                (group_index, path, target_digits)
            )
    raw_hs_matches = {
        source_digits: _hs_code_surface_matches(raw_text, source_digits)
        for source_digits in hs_pairs
    }
    hs_matches: dict[str, tuple[re.Match[str], ...]] = {}
    for source_digits, matches in raw_hs_matches.items():
        longer_spans = tuple(
            (other.start(), other.end())
            for other_digits, other_matches in raw_hs_matches.items()
            if len(other_digits) > len(source_digits)
            for other in other_matches
        )
        hs_matches[source_digits] = tuple(
            match
            for match in matches
            if not any(
                start <= match.start()
                and match.end() <= end
                and (start, end) != (match.start(), match.end())
                for start, end in longer_spans
            )
        )
    for source_digits, hs_pair_rows in hs_pairs.items():
        matches = hs_matches[source_digits]
        distinct_group_targets = {
            (group_index, target_digits)
            for group_index, _path, target_digits in hs_pair_rows
        }
        distinct_targets = {
            target_digits for _group_index, _path, target_digits in hs_pair_rows
        }
        if len(matches) == 1 and len(hs_pair_rows) > 1:
            # One printed aggregate HS slot may intentionally represent several labeled cargo
            # groups.  It has no per-row ownership to infer; render every distinct target code in
            # the same physical slot, preserving first target-path order and source punctuation.
            match = matches[0]
            ordered_target_digits = tuple(
                dict.fromkeys(target_digits for _group_index, _path, target_digits in hs_pair_rows)
            )
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="hs_code_block",
                    targetPath=";".join(
                        path for _group_index, path, _target in hs_pair_rows
                    ),
                    sourceSurface=match.group(0),
                    targetSurface=", ".join(
                        _format_numeric_surface(match.group(0), target_digits)
                        for target_digits in ordered_target_digits
                    ),
                    sourceOccurrences=1,
                    contextEvidence=raw_text[
                        max(0, match.start() - 90) : min(len(raw_text), match.end() + 40)
                    ],
                    sourceLineIds=(f"L{raw_text.count(chr(10), 0, match.start()) + 1:05d}",),
                )
            )
            continue
        relation_ownership_available = all(
            group_index in cargo_hs_lines_by_group_index
            for group_index, _path, _target in hs_pair_rows
        )
        if (
            len(distinct_group_targets) > 1
            and len(distinct_targets) > 1
            and relation_ownership_available
        ):
            # Repeated source codes with different group targets are safe only when every exact
            # occurrence belongs to exactly one reviewed cargo span. Unowned summary/header copies
            # or overlapping spans are genuinely ambiguous and stop publication rather than being
            # assigned by position.
            raw_lines = raw_text.splitlines()
            owned_matches: list[tuple[re.Match[str], tuple[int, str, str]]] = []
            for match in matches:
                match_line = raw_text.count("\n", 0, match.start()) + 1
                distances = {
                    group_index: distance
                    for group_index, _path, _target in hs_pair_rows
                    if (
                        distance := _same_paragraph_line_distance(
                            raw_lines,
                            cargo_hs_lines_by_group_index.get(group_index, frozenset()),
                            match_line,
                        )
                    )
                    is not None
                }
                if not distances:
                    owners: tuple[tuple[int, str, str], ...] = ()
                else:
                    minimum = min(distances.values())
                    nearest_groups = {
                        group_index
                        for group_index, distance in distances.items()
                        if distance == minimum
                    }
                    owners = tuple(
                        pair for pair in hs_pair_rows if pair[0] in nearest_groups
                    )
                if len(owners) != 1:
                    raise ValueError(
                        "shared source HS surface lacks one cargo-group owner: "
                        f"source={source_digits!r}, line=L{match_line:05d}, "
                        f"owners={[path for _index, path, _target in owners]}"
                    )
                owned_matches.append((match, owners[0]))
            owned_paths = {path for _match, (_index, path, _target) in owned_matches}
            missing_paths = {
                path for _index, path, _target in hs_pair_rows
            } - owned_paths
            if missing_paths:
                raise ValueError(
                    "cannot locate a group-owned source HS surface for paths "
                    f"{sorted(missing_paths)}"
                )
            for match, (_group_index, path, target_digits) in owned_matches:
                requirements.append(
                    SurfaceRenderingRequirement(
                        kind="hs_code",
                        targetPath=path,
                        sourceSurface=match.group(0),
                        targetSurface=_format_numeric_surface(match.group(0), target_digits),
                        sourceOccurrences=1,
                        contextEvidence=raw_text[
                            max(0, match.start() - 90) : min(len(raw_text), match.end() + 40)
                        ],
                        sourceLineIds=(
                            f"L{raw_text.count(chr(10), 0, match.start()) + 1:05d}",
                        ),
                    )
                )
            continue
        if len(matches) < len(hs_pair_rows):
            raise ValueError(
                "cannot locate every source HS surface for paths "
                f"{[path for _group_index, path, _target in hs_pair_rows]}"
            )
        for index, match in enumerate(matches):
            _group_index, path, target_digits = hs_pair_rows[index % len(hs_pair_rows)]
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="hs_code",
                    targetPath=path,
                    sourceSurface=match.group(0),
                    targetSurface=_format_numeric_surface(match.group(0), target_digits),
                    sourceOccurrences=1,
                    contextEvidence=raw_text[
                        max(0, match.start() - 90) : min(len(raw_text), match.end() + 40)
                    ],
                    sourceLineIds=(f"L{raw_text.count(chr(10), 0, match.start()) + 1:05d}",),
                )
            )
    origin_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups, strict=False)
    ):
        if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
            continue
        source_origin = source_group.get("origin")
        target_origin = target_group.get("origin")
        source_name = source_origin.get("name") if isinstance(source_origin, Mapping) else None
        target_name = target_origin.get("name") if isinstance(target_origin, Mapping) else None
        if (
            not isinstance(source_name, str)
            or not isinstance(target_name, str)
            or _semantic_normalize(source_name) == _semantic_normalize(target_name)
        ):
            continue
        origin_pairs[source_name].append(
            (f"documentPatch.cargoGroups[{group_index}].origin.name", target_name)
        )
    raw_lines = raw_text.splitlines()
    for source_name, pairs in origin_pairs.items():
        target_names = {target for _path, target in pairs}
        if len(target_names) != 1:
            # One physical origin phrase cannot represent two different cargo-group origins.
            # The cargo-block renderer retains ownership of these path-conflicting cases.
            continue
        target_name = next(iter(target_names))
        context_pattern = re.compile(
            r"(?:\bMADE[ \t]+IN\b|\bCOUNTRY[ \t]+OF[ \t]+ORIGIN\b[.:' -]*|"
            r"\bORIGIN\b[.:' -]*|\bEX\b[ \t]+)"
            r"[^\r\n]{0,40}?" + re.escape(source_name),
            re.IGNORECASE,
        )
        owned_line_numbers = tuple(
            number
            for number, line in enumerate(raw_lines, start=1)
            if context_pattern.search(line) is not None
        )
        if not owned_line_numbers:
            continue
        rendered_name = _preserve_case(source_name, target_name)
        requirements.append(
            SurfaceRenderingRequirement(
                kind="cargo_origin",
                targetPath=";".join(path for path, _target in pairs),
                sourceSurface=source_name,
                targetSurface=rendered_name,
                sourceOccurrences=len(owned_line_numbers),
                contextEvidence=_bounded_line_context(
                    raw_lines,
                    focus_index=owned_line_numbers[0] - 1,
                    before=1,
                    after=1,
                ),
                sourceLineIds=tuple(f"L{number:05d}" for number in owned_line_numbers),
            )
        )
    for match in _CARRIER_RECEIPT_COUNT.finditer(raw_text):
        source_surface = match.group(0)
        observed = int(match.group("count").replace(",", ""))
        target_count = _carrier_receipt_target_count(observed, source_label, target_label)
        target_digits = f"{target_count:,}" if "," in match.group("count") else str(target_count)
        target_surface = match.group("prefix") + target_digits + match.group("suffix")
        requirements.append(
            SurfaceRenderingRequirement(
                kind="carrier_receipt_count",
                targetPath="auxiliary.carrierReceiptCount",
                sourceSurface=source_surface,
                targetSurface=target_surface,
                sourceOccurrences=1,
                contextEvidence=raw_text[
                    max(0, match.start() - 60) : min(len(raw_text), match.end() + 60)
                ],
            )
        )
    requirements.extend(_carrier_receipt_equipment_breakdown_requirements(raw_text, target_label))
    requirements.extend(
        _aggregate_equipment_breakdown_requirements(raw_text, source_label, target_label)
    )
    if linguistic_plan is not None:
        requirements.extend(
            _dangerous_goods_rendering_requirements(
                raw_text,
                source_label,
                target_label,
                linguistic_plan,
            )
        )
    return tuple(requirements)


def _anonymous_transport_unit_target_count(
    target_label: Mapping[str, Any], *, source_count: int
) -> int:
    """Resolve a flattened *equipment* count without confusing it with packages.

    When the task label contains explicit containers, their cardinality owns the printed
    anonymous-equipment count.  When it does not, the OCR assertion is source-only template
    flavor: preserve its count rather than projecting an unrelated cargo-package quantity into
    a container field.
    """

    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    containers = patch.get("containers") or ()
    if isinstance(containers, Sequence) and not isinstance(containers, (str, bytes)) and containers:
        return len(containers)
    return source_count


def source_semantic_role_hints(
    raw_text: str, target_label: Mapping[str, Any]
) -> tuple[SourceSemanticRoleHint, ...]:
    """Recognize explicit flattened grammar without guessing from isolated numbers."""

    lines = raw_text.splitlines()
    output: list[SourceSemanticRoleHint] = []
    for index, line in enumerate(lines):
        if re.fullmatch(r"[ \t]*[1-9][0-9]*[ \t]*", line) is None:
            continue
        end = min(len(lines), index + 8)
        block = "\n".join(lines[index:end])
        if _ANONYMOUS_EQUIPMENT.search(block) is None:
            continue
        source_count = int(line.strip())
        target_count = _anonymous_transport_unit_target_count(
            target_label, source_count=source_count
        )
        stripped = line.strip()
        target_digits = f"{target_count:,}" if "," in stripped else str(target_count)
        target_surface = line.replace(stripped, target_digits, 1)
        output.append(
            SourceSemanticRoleHint(
                role="anonymous_equipment_count",
                sourceLineId=_line_id(index + 1, line),
                sourceSurface=line,
                requiredOutputSurface=target_surface,
                sourceEvidence=block,
                forbiddenTargetPathPrefixes=("documentPatch.cargoGroups[].marksAndNumbers",),
            )
        )
        target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
        target_containers = target_patch.get("containers") or ()
        if not (
            isinstance(target_containers, Sequence)
            and not isinstance(target_containers, (str, bytes))
            and target_containers
        ):
            for equipment_index in range(index + 1, end):
                equipment_line = lines[equipment_index]
                if _ANONYMOUS_EQUIPMENT.search(equipment_line) is None:
                    continue
                output.append(
                    SourceSemanticRoleHint(
                        role="anonymous_equipment_surface",
                        sourceLineId=_line_id(equipment_index + 1, equipment_line),
                        sourceSurface=equipment_line,
                        requiredOutputSurface=equipment_line,
                        sourceEvidence=block,
                        forbiddenTargetPathPrefixes=(
                            "documentPatch.cargoPackages[].typeCategory",
                            "documentPatch.cargoGroups[].marksAndNumbers",
                        ),
                    )
                )
                break
    return tuple(output)


def _source_semantic_role_surfaces_preserved(
    value: str, requirements: Sequence[SourceSemanticRoleHint]
) -> bool:
    lines = value.splitlines()
    return all(
        _line_number(row.sourceLineId) <= len(lines)
        and lines[_line_number(row.sourceLineId) - 1] == row.requiredOutputSurface
        for row in requirements
    )


def _label_has_marks(label: Mapping[str, Any]) -> bool:
    patch = cast(Mapping[str, Any], label.get("documentPatch") or {})
    return any(
        isinstance(group, Mapping) and bool(group.get("marksAndNumbers"))
        for group in patch.get("cargoGroups") or []
    )


def _finding_misclassifies_anonymous_equipment(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    if not any(row.role == "anonymous_equipment_count" for row in workspace.source_role_hints):
        return False
    if _label_has_marks(workspace.source_label) or _label_has_marks(workspace.current_target_label):
        return False
    return any("marksAndNumbers" in path for path in finding.affectedPaths)


def _finding_conflicts_with_equipment_authority(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    """Reject equipment findings whose cited container proves the target semantics."""

    affected = {
        int(match.group(1))
        for path in finding.affectedPaths
        if (match := _CONTAINER_EQUIPMENT_REVIEW_PATH.search(path)) is not None
    }
    if not affected:
        return False
    patch = workspace.current_target_label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if not isinstance(containers, list):
        return False
    for evidence in finding.currentEvidence:
        normalized_evidence = _semantic_normalize(evidence)
        for index, container in enumerate(containers):
            if not isinstance(container, Mapping):
                continue
            number = container.get("containerNumber")
            if (
                not isinstance(number, str)
                or _semantic_normalize(number) not in normalized_evidence
            ):
                continue
            if index not in affected:
                return True
            size = container.get("sizeCategory")
            category = container.get("typeCategory")
            if not isinstance(size, str) or not isinstance(category, str):
                continue
            reviewed = review_source_equipment_surface(
                evidence,
                temperature_present=container.get("temperatureSetpoint") is not None,
            )
            if (
                finding.category in {"missing_or_wrong_target_fact", "stale_source_fact"}
                and reviewed.resolution == "reviewed_source_grammar"
                and reviewed.size_category == size
                and reviewed.type_category == category
            ):
                return True
    return False


def _finding_conflicts_with_surface_authority(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    """Reject a review that asks to change an already-proven derived exact surface."""

    for requirement in workspace.surface_requirements:
        if requirement.kind not in {
            "carrier_receipt_count",
            "carrier_receipt_equipment_breakdown",
        }:
            continue
        if requirement.targetSurface not in workspace.current_text:
            continue
        if any(
            requirement.targetSurface in evidence or evidence in requirement.targetSurface
            for evidence in finding.currentEvidence
        ):
            return True
    return False


def _finding_conflicts_with_anchored_scalar_authority(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    """Reject reclassification of a proven line-bound target scalar as auxiliary flavor."""

    if finding.category not in {"stale_source_fact", "incomplete_auxiliary_anonymization"}:
        return False

    current_lines = workspace.current_text.splitlines()
    affected_paths = set(finding.affectedPaths)
    for requirement in workspace.anchored_scalar_replacement_requirements:
        if affected_paths & set(requirement.targetPaths):
            continue
        target_pattern = _anchored_requirement_pattern(requirement, requirement.targetSurface)
        for line_id in requirement.sourceLineIds:
            line_number = _line_number(line_id)
            if line_number > len(current_lines):
                continue
            current_line = current_lines[line_number - 1]
            if target_pattern.search(current_line) is None:
                continue
            if any(
                _evidence_is_isolated_anchored_target(evidence, requirement.targetSurface)
                for evidence in finding.currentEvidence
            ):
                return True
    return False


def _evidence_is_isolated_anchored_target(evidence: str, target_surface: str) -> bool:
    """Distinguish reclassifying a target scalar from flagging a neighboring stale value."""

    normalized_evidence = _semantic_normalize(evidence)
    normalized_target = _semantic_normalize(target_surface)
    if normalized_evidence == normalized_target:
        return True
    if normalized_target not in normalized_evidence:
        return False
    remainder = normalized_evidence.replace(normalized_target, " ", 1).split()
    return len(remainder) <= 3 and all(token.isalpha() for token in remainder)


def _finding_grounds_format_damage_only_in_unchanged_text(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    """Formatting evidence must quote changed output, not bytes identical to the source."""

    return finding.category in {"format_or_layout_damage", "unnecessary_change"} and all(
        evidence in workspace.original_text for evidence in finding.currentEvidence
    )


def _required_surface_rendered(value: str, requirement: SurfaceRenderingRequirement) -> bool:
    if requirement.sourceLineIds:
        lines = value.splitlines()
        for line_id in requirement.sourceLineIds:
            line_number = _line_number(line_id)
            if not 1 <= line_number <= len(lines):
                return False
            line = lines[line_number - 1]
            if requirement.targetSurface not in line:
                return False
            if (
                requirement.sourceSurface != requirement.targetSurface
                and requirement.sourceSurface not in requirement.targetSurface
                and requirement.sourceSurface in line
            ):
                return False
        return True
    if requirement.kind == "carrier_header_identity":
        lines = value.splitlines()
        target_count = lines.count(requirement.targetSurface)
        source_count = lines.count(requirement.sourceSurface)
        return target_count >= requirement.sourceOccurrences and (
            requirement.sourceSurface == requirement.targetSurface or source_count == 0
        )
    if requirement.kind == "date":
        target_matches = tuple(_literal_phrase_pattern(requirement.targetSurface).finditer(value))
        target_count = sum(
            _date_context_path(value, match.start()) == requirement.targetPath
            for match in target_matches
        )
        if target_count < requirement.sourceOccurrences:
            return False
        if requirement.sourceSurface == requirement.targetSurface:
            return True
        return not any(
            _date_context_path(value, match.start()) == requirement.targetPath
            for match in _literal_phrase_pattern(requirement.sourceSurface).finditer(value)
        )
    if requirement.kind == "date_global":
        target_count = len(
            tuple(_literal_phrase_pattern(requirement.targetSurface).finditer(value))
        )
        if target_count < requirement.sourceOccurrences:
            return False
        return requirement.sourceSurface == requirement.targetSurface or not tuple(
            _literal_phrase_pattern(requirement.sourceSurface).finditer(value)
        )
    target_surfaces = {requirement.targetSurface}
    return requirement.targetSurface in value and (
        requirement.sourceSurface == requirement.targetSurface
        or requirement.sourceSurface in target_surfaces
        or requirement.sourceSurface not in value
    )


def _required_surfaces_rendered(
    value: str, requirements: Sequence[SurfaceRenderingRequirement]
) -> bool:
    target_surfaces = {row.targetSurface for row in requirements}
    for row in requirements:
        if row.sourceLineIds or row.kind in {
            "carrier_header_identity",
            "carrier_principal_identity",
            "date",
            "date_global",
        }:
            if not _required_surface_rendered(value, row):
                return False
            continue
        if row.targetSurface not in value:
            return False
        if (
            row.sourceSurface != row.targetSurface
            and row.sourceSurface not in target_surfaces
            and row.sourceSurface in value
        ):
            return False
    return True


def _compound_relationships(
    value: str,
) -> tuple[Literal["on_behalf_of", "trading_as"], ...]:
    return tuple(name for name, pattern in _COMPOUND_PARTY_RELATIONS if pattern.search(value))


def _party_name(label: Mapping[str, Any], role: str) -> str | None:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return None
    parties = patch.get("parties")
    if not isinstance(parties, Mapping):
        return None
    party = parties.get(role)
    if not isinstance(party, Mapping):
        return None
    value = party.get("name")
    return value if isinstance(value, str) else None


def compound_party_flavor_requirements(
    source_label: Mapping[str, Any], target_label: Mapping[str, Any]
) -> tuple[CompoundPartyFlavorRequirement, ...]:
    """Find source name relationships to preserve as fictional raw-text flavor.

    The synthetic target remains authoritative and unchanged. A compound relationship in the
    reviewed source name describes template topology: the raw realization keeps that relationship
    around the exact synthetic primary identity using a distinct fictional secondary identity.
    Similar wording occurring solely in raw signature or agency text is handled as ordinary
    auxiliary flavor by the editor.
    """

    output: list[CompoundPartyFlavorRequirement] = []
    for role in _COMPOUND_PARTY_ROLES:
        source_name = _party_name(source_label, role)
        target_name = _party_name(target_label, role)
        if source_name is None or target_name is None:
            continue
        relationships = _compound_relationships(source_name)
        if not relationships or set(relationships) <= set(_compound_relationships(target_name)):
            continue
        output.append(
            CompoundPartyFlavorRequirement(
                targetPath=f"documentPatch.parties.{role}.name",
                relationships=relationships,
                sourceLabelName=source_name,
                targetPrimaryName=target_name,
            )
        )
    return tuple(output)


def _active_compound_party_flavor_requirements(
    workspace: RewriteWorkspace,
) -> tuple[CompoundPartyFlavorRequirement, ...]:
    """Return every compound relationship that the current atomic edit must attest."""

    return compound_party_flavor_requirements(
        workspace.source_label, workspace.current_target_label
    )


def _validate_compound_party_flavor_realizations(
    workspace: RewriteWorkspace,
    realizations: Sequence[CompoundPartyFlavorRealization],
) -> tuple[AppliedCompoundPartyFlavorRealization, ...]:
    requirements = _active_compound_party_flavor_requirements(workspace)
    required_by_path = {row.targetPath: row for row in requirements}
    submitted_by_path = {row.targetPath: row for row in realizations}
    if len(submitted_by_path) != len(realizations):
        raise ValueError("compound-party flavor realizations contain duplicate paths")
    if set(submitted_by_path) != set(required_by_path):
        missing = sorted(set(required_by_path) - set(submitted_by_path))
        extra = sorted(set(submitted_by_path) - set(required_by_path))
        raise ValueError(
            "compound-party flavor realizations differ from requirements; "
            f"missing={missing}, extra={extra}"
        )
    if not requirements:
        return ()
    applied: list[AppliedCompoundPartyFlavorRealization] = []
    for path in sorted(required_by_path):
        requirement = required_by_path[path]
        realization = submitted_by_path[path]
        if realization.targetPrimaryName != requirement.targetPrimaryName:
            raise ValueError(f"compound-party realization uses a stale target name: {path}")
        normalized_primary = _semantic_normalize(requirement.targetPrimaryName)
        normalized_rendered = _semantic_normalize(realization.renderedName)
        if not normalized_rendered.startswith(normalized_primary + " "):
            raise ValueError(
                f"compound raw realization must begin with the target primary identity: {path}"
            )
        observed_relationships = _compound_relationships(realization.renderedName)
        if set(observed_relationships) != set(requirement.relationships):
            raise ValueError(
                f"compound raw realization does not preserve the relation topology: {path}"
            )
        for relationship, pattern in _COMPOUND_PARTY_RELATIONS:
            if relationship not in requirement.relationships:
                continue
            match = pattern.search(realization.renderedName)
            assert match is not None
            auxiliary = _semantic_normalize(realization.renderedName[match.end() :])
            if len(auxiliary.split()) < 2 or auxiliary == normalized_primary:
                raise ValueError(
                    "compound raw realization needs a distinct realistic auxiliary identity: "
                    f"{path}"
                )
        applied.append(
            AppliedCompoundPartyFlavorRealization(
                targetPath=path,
                relationships=requirement.relationships,
                targetPrimaryName=requirement.targetPrimaryName,
                renderedName=realization.renderedName,
            )
        )
    return tuple(applied)


def _shipped_on_board_trailing_identity(value: str) -> str | None:
    """Split a trailing signing identity after an explicit on-board date in linear time."""

    if "\n" in value or "\r" in value:
        return None
    heading = _RAW_SHIPPED_ON_BOARD.search(value)
    if heading is None:
        return None
    selected: str | None = None
    for date_match in _RAW_SHIPPED_ON_BOARD_DATE.finditer(value, heading.end()):
        if date_match.end() >= len(value) or value[date_match.end()] not in " \t":
            continue
        identity = value[date_match.end() :].strip()
        # This is the same deliberately narrow identity width enforced by the surrounding
        # carrier-agency grammar. Unicode letters and digits are valid leading characters;
        # punctuation-only suffixes are not identities.
        if 2 <= len(identity) <= 160 and identity[0].isalnum():
            selected = identity
    return selected


def _raw_agent_blocks(value: str) -> tuple[tuple[int, str, int, str, str], ...]:
    """Return explicit raw-only agency identities on either side of their legal relation."""

    lines = value.splitlines()
    output: list[tuple[int, str, int, str, str]] = []
    for relation_index, relation_line in enumerate(lines):
        match = _RAW_AGENT_FOR_CARRIER_LINE.fullmatch(relation_line)
        if match is None:
            continue
        cursor = relation_index - 1
        gap = 0
        while cursor >= 0 and not lines[cursor].strip() and gap < 2:
            gap += 1
            cursor -= 1
        end = cursor + 1
        while (
            cursor >= 0
            and end - cursor <= 3
            and lines[cursor].strip()
            and _PAGE_MARKER.fullmatch(lines[cursor]) is None
        ):
            # ``SIGNED FOR THE CARRIER <principal>`` is the principal relationship line,
            # not another line of the auxiliary ``BY <agent>`` identity immediately below it.
            if _RAW_SIGNED_FOR_CARRIER_LINE.fullmatch(lines[cursor]) is not None:
                break
            if (
                _PRESERVABLE_STATUS_LINE.fullmatch(lines[cursor]) is not None
                or _PRESERVABLE_LEGAL_BOILERPLATE_LINE.fullmatch(lines[cursor]) is not None
                or _PARTY_HEADING_LINE.fullmatch(lines[cursor]) is not None
            ):
                # The relation may follow a blank-separated company identity, but a formal
                # caption or legal clause can never be that identity.  Stop at the grammar-proven
                # boundary instead of anonymizing immutable bill-of-lading wording as a company.
                break
            cursor -= 1
        start = cursor + 1
        identity_lines = lines[start:end]
        if not identity_lines:
            continue
        output.append(
            (
                start + 1,
                "\n".join(identity_lines),
                gap,
                match.group("principal").strip(),
                "\n".join(lines[start : relation_index + 1]),
            )
        )
    for relation_index, relation_line in enumerate(lines):
        match = _RAW_ISSUED_AGENT_BY_LINE.fullmatch(relation_line)
        if match is None:
            continue
        cursor = relation_index + 1
        gap = 0
        while cursor < len(lines) and not lines[cursor].strip() and gap < 2:
            gap += 1
            cursor += 1
        start = cursor
        while (
            cursor < len(lines)
            and cursor - start < 3
            and lines[cursor].strip()
            and _PAGE_MARKER.fullmatch(lines[cursor]) is None
        ):
            cursor += 1
        identity_lines = lines[start:cursor]
        if not identity_lines:
            continue
        output.append(
            (
                start + 1,
                "\n".join(identity_lines),
                gap,
                match.group("principal").strip(),
                "\n".join(lines[relation_index:cursor]),
            )
        )
    for relation_index, relation_line in enumerate(lines):
        match = _RAW_INLINE_AGENT_FOR_CARRIER_LINE.fullmatch(relation_line)
        if match is None:
            continue
        identity = match.group("identity").strip().rstrip(",").strip()
        shipped_inline = _shipped_on_board_trailing_identity(identity)
        if shipped_inline is not None:
            # Some forms keep the on-board heading, vessel, date, signing agent, and generic
            # carrier relationship on one physical line.  The generic inline grammar sees the
            # whole prefix as the identity; split at the independently recognizable on-board date
            # so only the trailing agent is anonymized and vessel/date ownership remains intact.
            identity = shipped_inline.rstrip(",").strip()
        # A ``BY <identity> AS AGENT ...`` line without an explicit principal is owned by the
        # dedicated signed/on-behalf grammar below.  When the principal is explicit on this same
        # line, this parser is the only complete source of both identities and must retain it.
        if re.match(r"(?i)^BY\b", identity) is not None and match.group("principal") is None:
            continue
        principal = match.group("principal")
        gap = 0
        identity_line_number = relation_index + 1
        evidence_start = relation_index
        evidence_end = relation_index + 1
        if principal is None:
            if shipped_inline is not None:
                # The relation itself is complete on this line. A following weight, total, or
                # signature field is not a wrapped carrier principal merely because blank lines
                # separate it from the on-board statement.
                principal = "THE CARRIER"
            else:
                previous = relation_index - 1
                previous_gap = 0
                while previous >= 0 and not lines[previous].strip() and previous_gap < 2:
                    previous_gap += 1
                    previous -= 1
                signed = (
                    _RAW_SIGNED_FOR_CARRIER_LINE.fullmatch(lines[previous])
                    if previous >= 0
                    else None
                )
                shipped = (
                    _shipped_on_board_trailing_identity(lines[previous])
                    if previous >= 0 and previous_gap == 0
                    else None
                )
                if signed is not None:
                    principal = signed.group("principal").strip()
                    evidence_start = previous
                elif shipped is not None:
                    # Some carrier forms append the signing agent after the on-board date and wrap
                    # its legal suffix onto the next line. The date grammar proves the split point;
                    # retain the surrounding vessel/date prefix as a host lock and synthesize the
                    # complete two-line identity as one consistency unit.
                    previous_identity = shipped.rstrip(",").strip()
                    identity = previous_identity + "\n" + identity
                    identity_line_number = previous + 1
                    evidence_start = previous
                    principal = "THE CARRIER"
                else:
                    cursor = relation_index + 1
                    while cursor < len(lines) and not lines[cursor].strip() and gap < 2:
                        gap += 1
                        cursor += 1
                    if (
                        cursor < len(lines)
                        and lines[cursor].strip()
                        and _PAGE_MARKER.fullmatch(lines[cursor]) is None
                        and _PARTY_HEADING_LINE.fullmatch(lines[cursor]) is None
                    ):
                        # A second observed grammar prints the principal on the following line.
                        # Source-label verification in ``raw_auxiliary_identity_requirements``
                        # rejects an unrelated following field before it can grant write authority.
                        principal = lines[cursor].strip()
                        evidence_end = cursor + 1
                    else:
                        # ``as agent for the Carrier`` is also a complete generic relation.
                        principal = "THE CARRIER"
        if not identity:
            continue
        output.append(
            (
                identity_line_number,
                identity,
                gap,
                principal.strip(),
                "\n".join(lines[evidence_start:evidence_end]),
            )
        )
    for relation_index, relation_line in enumerate(lines):
        relation = _RAW_SIGNED_ON_BEHALF_LINE.fullmatch(relation_line)
        if relation is None:
            continue
        cursor = relation_index + 1
        gap = 0
        while cursor < len(lines) and not lines[cursor].strip() and gap < 2:
            gap += 1
            cursor += 1
        if cursor >= len(lines):
            continue
        agent = _RAW_BY_AGENT_LINE.fullmatch(lines[cursor])
        if agent is None:
            continue
        output.append(
            (
                cursor + 1,
                agent.group("identity").strip(),
                gap,
                relation.group("principal").strip(),
                "\n".join(lines[relation_index : cursor + 1]),
            )
        )
    return tuple(sorted(output, key=lambda row: row[0]))


def carrier_principal_occurrence_count(
    text: str,
    surface: str,
    *,
    excluded_line_groups: Sequence[frozenset[int]] = (),
) -> int:
    """Count carrier principals/brands while excluding separately synthesized agent aliases.

    A short carrier name commonly occurs inside both the carrier principal and a local agent
    identity (for example ``CMA CGM`` versus ``CMA CGM XIAMEN``).  Requiring the target carrier
    name at both sites destroys the legal relationship.  This projection keeps the principal
    side of explicit agency grammar and removes only the agent side before applying the same
    punctuation-tolerant party-scalar counter used by the atomic audit.
    """

    raw_agent_lines: set[int] = set()
    for line_number, source_identity, _gap, _principal, _evidence in _raw_agent_blocks(text):
        raw_agent_lines.update(range(line_number, line_number + len(source_identity.splitlines())))

    excluded_lines = {number for group in excluded_line_groups for number in group}
    principal_lines: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line_number in excluded_lines:
            continue
        relation = _RAW_AGENT_FOR_CARRIER_LINE.fullmatch(line)
        if relation is not None:
            principal_lines.append(relation.group("principal"))
            continue
        inline = _RAW_INLINE_AGENT_FOR_CARRIER_LINE.fullmatch(line)
        if inline is not None:
            principal = inline.group("principal")
            if principal is not None:
                principal_lines.append(principal)
            continue
        if line_number in raw_agent_lines:
            continue
        principal_lines.append(line)
    return _party_scalar_occurrence_count("\n".join(principal_lines), surface)


def carrier_principal_template_slot_groups(
    text: str,
    source_surface: str,
    *,
    excluded_line_groups: Sequence[frozenset[int]] = (),
) -> tuple[frozenset[int], ...]:
    """Count role-owned carrier-principal slots even when the template abbreviates the name.

    A task label may contain a carrier's full legal name while signature blocks print an
    abbreviation (for example ``Société Anonyme`` versus ``S.A.``).  Lexical equality therefore
    undercounts the target cardinality and can cause a valid repeated principal to be rewritten
    as unrelated fictional carriers.  Explicit legal grammar establishes the role independently
    of spelling; all other lines still require the exact source scalar.
    """

    explicit_lines: set[int] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        inline = _RAW_INLINE_AGENT_FOR_CARRIER_LINE.fullmatch(line)
        if (
            _RAW_AGENT_FOR_CARRIER_LINE.fullmatch(line) is not None
            or _RAW_SIGNED_ON_BEHALF_LINE.fullmatch(line) is not None
            or _RAW_SIGNED_FOR_CARRIER_LINE.fullmatch(line) is not None
            or (inline is not None and inline.group("principal") is not None)
        ):
            explicit_lines.add(number)
    raw_agent_lines: set[int] = set()
    for line_number, identity, _gap, _principal, _evidence in _raw_agent_blocks(text):
        raw_agent_lines.update(range(line_number, line_number + len(identity.splitlines())))
    scalar_groups = tuple(
        group
        for group in _party_scalar_occurrence_line_sets(text, source_surface)
        if not group & raw_agent_lines
        and not any(group & excluded for excluded in excluded_line_groups)
    )
    scalar_lines = frozenset(number for group in scalar_groups for number in group)
    abbreviated_groups = tuple(
        frozenset((number,))
        for number in sorted(explicit_lines)
        if number not in scalar_lines
        and _party_scalar_occurrence_count(text.splitlines()[number - 1], source_surface) == 0
    )
    return scalar_groups + abbreviated_groups


def carrier_principal_template_slot_count(
    text: str,
    source_surface: str,
    *,
    excluded_line_groups: Sequence[frozenset[int]] = (),
) -> int:
    """Return the number of disjoint literal or legal-syntax carrier slots.

    ``carrier_principal_template_slot_groups`` already combines complete scalar occurrences with
    abbreviation-only legal slots.  Counting the multiline scalar occurrences a second time as
    abbreviations would turn one wrapped legal name into two required target occurrences.
    """

    return len(
        carrier_principal_template_slot_groups(
            text,
            source_surface,
            excluded_line_groups=excluded_line_groups,
        )
    )


def _auxiliary_identity_token_key(value: str) -> str:
    return " ".join(
        sorted(
            token for token in re.findall(r"[A-Z0-9]+", _semantic_normalize(value)) if token != "BY"
        )
    )


_GENERIC_SIGNATORY_TITLES = frozenset(
    {
        "AGENT",
        "AGENTS",
        "BRANCH MANAGER",
        "DIRECTOR",
        "GENERAL MANAGER",
        "MANAGING DIRECTOR",
        "OPERATIONS MANAGER",
        "VICE PRESIDENT",
    }
)
_GENERIC_SIGNATORY_CAPTIONS = frozenset(
    {
        "AS AUTHORISED SIGNATORY",
        "AS AUTHORIZED SIGNATORY",
        "AUTHORISED SIGNATORY",
        "AUTHORIZED SIGNATORY",
        "BY",
        "CARRIER",
        "ON BEHALF OF THE CARRIER",
        "SIGNATORY",
        "SIGNATURE",
        "SIGNED",
        "SIGNED BY",
        "SIGNED FOR THE CARRIER",
        "SIGNED ON BEHALF OF THE CARRIER",
        "THE CARRIER",
    }
)
_GENERIC_SIGNATORY_SURFACES = _GENERIC_SIGNATORY_TITLES | _GENERIC_SIGNATORY_CAPTIONS
_GENERIC_SIGNATORY_EXECUTION_PREFIXES = ("BY ", "SIGNED BY ", "SIGNATURE OF ")


def _is_generic_signatory_role(value: str) -> bool:
    """Return whether a captured agency prefix contains no person or company identity.

    Bill-of-lading signature blocks often place generic captions such as ``By`` and
    ``General Manager`` immediately before an explicit ``as agent for the Carrier`` relation.
    Those captions describe how the signature is executed; they are neither private identity
    data nor a synthetic party slot. Only exact, structurally recognized captions and titles are
    accepted, so a person or organization that happens to contain a role word remains owned.
    """

    normalized = _semantic_normalize(value)
    if normalized in _GENERIC_SIGNATORY_SURFACES:
        return True
    for prefix in _GENERIC_SIGNATORY_EXECUTION_PREFIXES:
        if (
            normalized.startswith(prefix)
            and normalized[len(prefix) :] in _GENERIC_SIGNATORY_SURFACES
        ):
            return True
    return False


def _auxiliary_identity_consistency_group(value: str) -> str:
    return (
        "carrier-agent-" + sha256_bytes(_auxiliary_identity_token_key(value).encode("utf-8"))[:16]
    )


def raw_auxiliary_identity_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[RawAuxiliaryIdentityRequirement, ...]:
    """Locate syntactically explicit raw-only agents adjacent to carrier relationships."""

    source_carrier = _party_name(source_label, "carrier")
    target_carrier = _party_name(target_label, "carrier")
    requirements: list[RawAuxiliaryIdentityRequirement] = []
    for (
        line_number,
        source_identity,
        gap,
        source_principal,
        evidence,
    ) in _raw_agent_blocks(raw_text):
        normalized = _semantic_normalize(source_identity)
        normalized_source_principal = _semantic_normalize(source_principal)
        generic_principal = normalized_source_principal in {"CARRIER", "THE CARRIER"}
        if (
            _is_generic_signatory_role(source_identity)
            or normalized == _semantic_normalize(source_carrier or "")
            or normalized == _semantic_normalize(target_carrier or "THE CARRIER")
            or len(normalized.split()) < 2
        ):
            continue
        # Explicit ``agent(s) for the carrier`` grammar establishes the legal role even when the
        # printed principal abbreviates a longer labeled carrier name (for example ``S.A.`` versus
        # ``Société Anonyme au Capital ...``).  Lexical equality is therefore neither necessary
        # nor sufficient here: leaving a syntactically proven agent out would retain a source
        # identity. Preserve the relationship actually printed by the template. A generic phrase
        # such as
        # ``as agents for the Carrier`` must remain generic; requiring a full synthetic legal
        # name invents a topology that is absent from the source and makes a simple agent edit
        # unnecessarily fragile. Explicit printed principals still project to the target carrier.
        target_principal = "THE CARRIER" if generic_principal else target_carrier or "THE CARRIER"
        requirements.append(
            RawAuxiliaryIdentityRequirement(
                requirementId=f"carrier-agent-L{line_number:05d}",
                relationship="agent_for_carrier",
                sourceIdentity=source_identity,
                sourceIdentityLineCount=len(source_identity.splitlines()),
                gapLineCount=gap,
                consistencyGroupId=_auxiliary_identity_consistency_group(source_identity),
                targetPrincipalName=target_principal,
                sourceEvidence=evidence,
            )
        )
    return tuple(requirements)


def _numeric_span_overlaps_date(line: str, start: int, end: int) -> bool:
    """Return whether a numeric token belongs to a complete printed date surface."""

    return any(
        match.start() < end and start < match.end()
        for pattern in _DATE_SURFACE_PATTERNS
        for match in pattern.finditer(line)
    )


def _cargo_measure_present(label: Mapping[str, Any], field_name: str) -> bool:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return False
    return any(
        isinstance(group, Mapping) and group.get(field_name) is not None
        for group in cast(Sequence[Any], patch.get("cargoGroups") or ())
    )


def _measurement_pattern(kind: Literal["gross_weight_kg", "volume_m3"]) -> re.Pattern[str]:
    return _RAW_GROSS_WEIGHT if kind == "gross_weight_kg" else _RAW_VOLUME


def _operational_measurement_match(
    line: str,
    kind: OperationalKind,
    *,
    expected_surface: str | None = None,
    source_grammar: OperationalGrammar = "labeled_measurement",
) -> re.Match[str] | None:
    if source_grammar == "dense_container_table":
        match = _RAW_DENSE_MEASUREMENT_LINE.fullmatch(line)
        if match is None:
            tuple_matches = _dense_tuple_value_matches(line)
            tuple_index = {
                "gross_weight_kg": 0,
                "tare_weight_kg": 1,
                "volume_m3": 2,
            }.get(kind)
            match = (
                tuple_matches[tuple_index]
                if tuple_matches is not None and tuple_index is not None
                else None
            )
        matches: tuple[re.Match[str], ...] = (match,) if match is not None else ()
    elif kind == "gross_weight_kg":
        # Prefer an explicit gross label.  A generic trailing ``123 KGS`` pattern is needed for
        # compact container rows, but it must never reinterpret an explicitly labeled net or
        # tare row as gross weight.  That previously created two owners for the same number and
        # caused the deterministic prefill to rewrite a tare before trying to rewrite it again.
        explicit = _RAW_GROSS_WEIGHT.search(line)
        if explicit is not None:
            matches = (explicit,)
        elif any(
            marker in _semantic_normalize(line)
            for marker in ("TARE", "NET WEIGHT", "NET WT", "NETT WEIGHT", "N W")
        ):
            matches = ()
        else:
            trailing = _RAW_TRAILING_GROSS_WEIGHT.search(line)
            matches = (trailing,) if trailing is not None else ()
    elif kind == "tare_weight_kg":
        match = _RAW_TARE_WEIGHT.search(line)
        matches = (match,) if match is not None else ()
    elif kind == "volume_m3":
        explicit = _RAW_VOLUME.search(line)
        if explicit is not None:
            matches = (explicit,)
        else:
            trailing = _RAW_TRAILING_VOLUME.search(line)
            if trailing is not None and re.match(
                r"[ \t]*(?:/[ \t]*|PER[ \t]+)(?:H|HR|HOUR|MIN|MINUTE|DAY)S?\b",
                line[trailing.end() :],
                re.IGNORECASE,
            ) is not None:
                # Ventilation/air-exchange rates use the same CBM surface as cargo volume but
                # are different physical quantities.  They remain linguistic handling text.
                trailing = None
            matches = (trailing,) if trailing is not None else ()
    else:
        match = _RAW_PACKAGE_QUANTITY.search(line)
        matches = (match,) if match is not None else ()
    if expected_surface is not None:
        matches = tuple(match for match in matches if match.group("value") == expected_surface)
    return min(matches, key=lambda match: match.start("value")) if matches else None


def _parse_measurement_surface(
    surface: str,
    *,
    maximum: Decimal | None,
) -> ParsedMeasurementSurface | None:
    """Parse a carrier numeric surface without assuming a national locale.

    A three-digit suffix is ambiguous between decimal and grouping notation.  In that case the
    physically valid interpretation with the larger magnitude is selected: ``13.519`` gross kg
    means 13,519 when both 13.519 and 13,519 fit, while ``70.1530`` CBM has an unambiguous
    four-place decimal suffix.  Capacity bounds reject impossible alternatives.
    """

    value = surface.strip()
    if not re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", value):
        return None
    candidates: list[ParsedMeasurementSurface] = []
    separators = {character for character in value if character in ".,"}
    if len(separators) == 2:
        decimal_separator = "." if value.rfind(".") > value.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        integer, fraction = value.rsplit(decimal_separator, 1)
        normalized = integer.replace(grouping_separator, "") + "." + fraction
        candidates.append(
            ParsedMeasurementSurface(
                value=Decimal(normalized),
                decimal_separator=decimal_separator,
                grouping_separator=grouping_separator,
                decimal_places=len(fraction),
            )
        )
    elif len(separators) == 1:
        separator = next(iter(separators))
        pieces = value.split(separator)
        if len(pieces) == 2:
            integer, fraction = pieces
            candidates.append(
                ParsedMeasurementSurface(
                    value=Decimal(integer + "." + fraction),
                    decimal_separator=separator,
                    grouping_separator=None,
                    decimal_places=len(fraction),
                )
            )
            if len(fraction) == 3:
                candidates.append(
                    ParsedMeasurementSurface(
                        value=Decimal(integer + fraction),
                        decimal_separator=None,
                        grouping_separator=separator,
                        decimal_places=0,
                    )
                )
        elif all(len(piece) == 3 for piece in pieces[1:]):
            candidates.append(
                ParsedMeasurementSurface(
                    value=Decimal("".join(pieces)),
                    decimal_separator=None,
                    grouping_separator=separator,
                    decimal_places=0,
                )
            )
    else:
        candidates.append(
            ParsedMeasurementSurface(
                value=Decimal(value),
                decimal_separator=None,
                grouping_separator=None,
                decimal_places=0,
            )
        )
    valid = [
        row
        for row in candidates
        if row.value.is_finite() and row.value > 0 and (maximum is None or row.value <= maximum)
    ]
    return max(valid, key=lambda row: row.value) if valid else None


def _render_measurement_surface(
    value: Decimal,
    *,
    style: ParsedMeasurementSurface,
    maximum: Decimal | None,
) -> str:
    quantum = Decimal(1).scaleb(-style.decimal_places)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    if maximum is not None and rounded > maximum:
        rounded = maximum.quantize(quantum, rounding=ROUND_DOWN)
    if rounded <= 0:
        raise ValueError("resampled operational measurement rounds to zero")
    plain = f"{rounded:.{style.decimal_places}f}"
    integer, dot, fraction = plain.partition(".")
    if style.grouping_separator is not None:
        integer = f"{int(integer):,}".replace(",", style.grouping_separator)
    if style.decimal_separator is None:
        return integer
    return integer + style.decimal_separator + fraction if dot else integer


def _line_has_measurement_role(
    lines: Sequence[str],
    line_index: int,
    kind: Literal["gross_weight_kg", "net_weight_kg", "volume_m3"],
) -> bool:
    current = _semantic_normalize(lines[line_index])
    preceding = _semantic_normalize(" ".join(lines[max(0, line_index - 2) : line_index]))
    numeric_only = re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", lines[line_index].strip()) is not None
    if kind == "gross_weight_kg":
        terms: tuple[str, ...] = ("GROSS WEIGHT", "GROSS WT", "G W")
        return any(surface in current for surface in terms) or (
            numeric_only and any(surface in preceding for surface in terms)
        )
    if kind == "net_weight_kg":
        terms = ("NET WEIGHT", "NET WT", "N W")
        return any(surface in current for surface in terms) or (
            numeric_only and any(surface in preceding for surface in terms)
        )
    terms = ("MEASUREMENT", "VOLUME", "CBM", "CUBIC METER", "CUBIC METRE")
    return any(surface in current for surface in terms) or (
        numeric_only and any(surface in preceding for surface in terms)
    )


def anchored_measurement_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Render changed labeled measures in the exact grouping and precision of their source line."""

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    source_groups = cast(Sequence[Any], source_patch.get("cargoGroups") or ())
    target_groups = cast(Sequence[Any], target_patch.get("cargoGroups") or ())
    kinds: tuple[Literal["gross_weight_kg", "net_weight_kg", "volume_m3"], ...] = (
        "gross_weight_kg",
        "net_weight_kg",
        "volume_m3",
    )
    field_names = {
        "gross_weight_kg": "grossWeight",
        "net_weight_kg": "netWeight",
        "volume_m3": "volume",
    }
    pairs: list[
        tuple[
            int,
            Literal["gross_weight_kg", "net_weight_kg", "volume_m3"],
            Decimal,
            Decimal,
        ]
    ] = []
    source_value_counts: dict[tuple[str, Decimal], int] = {}
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups, strict=False)
    ):
        if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
            continue
        for kind in kinds:
            source_value = _target_measure_value(source_group, kind)
            target_value = _target_measure_value(target_group, kind)
            if source_value is None or target_value is None or source_value == target_value:
                continue
            pairs.append((group_index, kind, source_value, target_value))
            key = (kind, source_value)
            source_value_counts[key] = source_value_counts.get(key, 0) + 1

    lines = raw_text.splitlines()
    claimed_occurrences: set[tuple[int, str]] = set()
    requirements: list[AnchoredScalarReplacementRequirement] = []
    for group_index, kind, source_value, target_value in pairs:
        candidates: list[tuple[int, str, str]] = []
        for line_index, line in enumerate(lines):
            if not _line_has_measurement_role(lines, line_index, kind):
                continue
            for match in _RAW_MEASUREMENT_NUMBER.finditer(line):
                source_surface = match.group("value")
                style = _parse_measurement_surface(source_surface, maximum=source_value)
                if style is None or style.value != source_value:
                    continue
                target_surface = _render_measurement_surface(
                    target_value,
                    style=_measurement_style_with_exact_target_precision(style, target_value),
                    maximum=None,
                )
                candidates.append((line_index, source_surface, target_surface))
        if not candidates:
            continue
        available = [row for row in candidates if (row[0], row[1]) not in claimed_occurrences]
        if not available:
            continue
        selected = available if source_value_counts[(kind, source_value)] == 1 else available[:1]
        grouped: dict[tuple[str, str], list[int]] = {}
        for line_index, source_surface, target_surface in selected:
            claimed_occurrences.add((line_index, source_surface))
            grouped.setdefault((source_surface, target_surface), []).append(line_index)
        for (source_surface, target_surface), line_indices in grouped.items():
            requirements.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=(
                        f"documentPatch.cargoGroups[{group_index}].{field_names[kind]}.value",
                    ),
                    sourceLineIds=tuple(
                        _line_id(line_index + 1, lines[line_index]) for line_index in line_indices
                    ),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    surfaceKind="measurement",
                )
            )
    return tuple(requirements)


def cargo_component_measurement_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    cargo_requirements: Sequence[CargoFlavorRewriteRequirement],
    surface_requirements: Sequence[SurfaceRenderingRequirement],
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Scale source-only cargo components with their labeled aggregate measure.

    A rider can print one component weight inside a DG or product-detail row while the task label
    contains only the shipment-level cargo-group total.  Leaving that component unchanged after
    the total changes creates an internally inconsistent synthetic document.  The component's
    source share is the only evidence-backed projection available, so retain that share and the
    exact printed numeric style.  This renderer is restricted to lines already proven to belong
    to exactly one cargo group by label evidence or another path-owned semantic renderer.
    """

    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    if not isinstance(source_patch, Mapping) or not isinstance(target_patch, Mapping):
        return ()
    source_groups = source_patch.get("cargoGroups") or ()
    target_groups = target_patch.get("cargoGroups") or ()
    if (
        not isinstance(source_groups, Sequence)
        or isinstance(source_groups, (str, bytes))
        or not isinstance(target_groups, Sequence)
        or isinstance(target_groups, (str, bytes))
        or len(source_groups) != len(target_groups)
    ):
        return ()

    line_groups: dict[int, set[int]] = defaultdict(set)
    for requirement in cargo_requirements:
        match = re.fullmatch(
            r"documentPatch\.cargoGroups\[([0-9]+)\]\.description",
            requirement.targetPath,
        )
        if match is not None:
            group_index = int(match.group(1))
            for line_id in requirement.sourceLineIds:
                line_groups[_line_number(line_id)].add(group_index)
    for requirement in surface_requirements:
        match = re.match(
            r"(?:auxiliary|documentPatch)\.cargoGroups\[([0-9]+)\]",
            requirement.targetPath,
        )
        if match is None:
            continue
        group_index = int(match.group(1))
        for line_id in requirement.sourceLineIds:
            line_groups[_line_number(line_id)].add(group_index)

    field_names = {"gross_weight_kg": "grossWeight", "volume_m3": "volume"}
    lines = raw_text.splitlines()
    output: list[AnchoredScalarReplacementRequirement] = []
    claimed: set[tuple[int, str]] = set()
    for line_number, owners in sorted(line_groups.items()):
        if len(owners) != 1 or not 1 <= line_number <= len(lines):
            continue
        group_index = next(iter(owners))
        if group_index >= len(source_groups):
            continue
        source_group = source_groups[group_index]
        target_group = target_groups[group_index]
        if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
            continue
        line = lines[line_number - 1]
        for kind in ("gross_weight_kg", "volume_m3"):
            source_total = _target_measure_value(source_group, kind)
            target_total = _target_measure_value(target_group, kind)
            if (
                source_total is None
                or target_total is None
                or source_total <= 0
                or target_total <= 0
                or source_total == target_total
            ):
                continue
            match = _operational_measurement_match(line, kind)
            if match is None or (line_number, match.group("value")) in claimed:
                continue
            if kind == "gross_weight_kg" and re.match(
                r"[ \t]*(?:(?:NET[ \t]+)?EACH\b|(?:X|\u00d7)[ \t]*[0-9][0-9,]*[ \t]*"
                r"(?:PACKAGES?|PKGS?|PCS?|PIECES?|PALLETS?|CARTONS?|DRUMS?|BAGS?|"
                r"BOX(?:ES)?|BALES?|ROLLS?|CRATES?|CASES?|BUNDLES?|SETS?|LOTS?|UNITS?))",
                line[match.end() :],
                re.IGNORECASE,
            ) is not None:
                # ``1KG X 18 BAGS`` and ``24 KG NET EACH`` describe the size of one retail or
                # inner package, not a component share of the labeled gross cargo weight.  The
                # cargo-language slot owns such source-only packaging prose; scaling it as an
                # aggregate produces physically meaningless sub-kilogram values.
                continue
            source_surface = match.group("value")
            style = _parse_measurement_surface(source_surface, maximum=source_total)
            if style is None or not Decimal(0) < style.value < source_total:
                continue
            target_component = target_total * style.value / source_total
            target_surface = _render_measurement_surface(
                target_component,
                style=style,
                maximum=target_total,
            )
            if source_surface == target_surface:
                continue
            claimed.add((line_number, source_surface))
            output.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=(
                        f"auxiliary.cargoGroups[{group_index}]."
                        f"{field_names[kind]}.component",
                    ),
                    sourceLineIds=(_line_id(line_number, line),),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    surfaceKind="measurement",
                )
            )
    return tuple(output)


def anchored_package_quantity_replacement_requirements(
    raw_text: str,
    leaves: Sequence[ChangedLeaf],
    *,
    excluded_line_ids: frozenset[str] = frozenset(),
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Render all unambiguous numeric package quantities without touching other numbers.

    A numeric match is semantic evidence only when it is bound to an observed package noun.
    Dates, phone numbers, and clause ordinals are therefore outside this compiler.  If multiple
    changed package/allocation paths reuse one source number, every one must resolve to the same
    target number; conflicting indexed projections remain model-owned and fail closed.

    Only the numeric capture is changed, retaining grouping, decimal-zero style, whitespace,
    package spelling, punctuation, and physical line topology.
    """

    quantity_path = re.compile(
        r"^documentPatch\.(?:"
        r"cargoPackages\[[0-9]+\]\.quantity|"
        r"cargoAllocationGroups\[[0-9]+\]\.allocations\[[0-9]+\]\.packageQuantity"
        r")$"
    )
    grouped: dict[int, list[ChangedLeaf]] = defaultdict(list)
    for leaf in leaves:
        if (
            leaf.requiresTextEdit
            and quantity_path.fullmatch(leaf.path) is not None
            and isinstance(leaf.sourceValue, int)
            and not isinstance(leaf.sourceValue, bool)
            and isinstance(leaf.targetValue, int)
            and not isinstance(leaf.targetValue, bool)
            and leaf.sourceValue != leaf.targetValue
        ):
            grouped[leaf.sourceValue].append(leaf)

    lines = raw_text.splitlines()
    requirements: list[AnchoredScalarReplacementRequirement] = []
    for source_value, related in sorted(grouped.items()):
        target_values = {cast(int, leaf.targetValue) for leaf in related}
        if len(target_values) != 1:
            continue
        target_value = next(iter(target_values))
        target_paths = tuple(sorted(leaf.path for leaf in related))
        occurrences: dict[tuple[str, str], list[str]] = defaultdict(list)
        for line_number, line in enumerate(lines, start=1):
            line_id = _line_id(line_number, line)
            if line_id in excluded_line_ids:
                continue
            for match in _RAW_PACKAGE_QUANTITY.finditer(line):
                source_value_surface = match.group("value")
                if _parse_package_integer_surface(source_value_surface) != source_value:
                    continue
                target_value_surface = _render_integer_surface(
                    target_value, source_surface=source_value_surface
                )
                source_surface = match.group(0)
                relative_start = match.start("value") - match.start()
                relative_end = match.end("value") - match.start()
                target_surface = (
                    source_surface[:relative_start]
                    + target_value_surface
                    + source_surface[relative_end:]
                )
                occurrences[(source_surface, target_surface)].append(line_id)
        for (source_surface, target_surface), line_ids in sorted(occurrences.items()):
            requirements.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=target_paths,
                    sourceLineIds=tuple(line_ids),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                )
            )
        for line_number, line in enumerate(lines, start=1):
            line_id = _line_id(line_number, line)
            if line_id in excluded_line_ids:
                continue
            for match in _RAW_WRITTEN_PACKAGE_QUANTITY.finditer(line):
                source_value_surface = match.group("value")
                if _parse_english_cardinal(source_value_surface) != source_value:
                    continue
                target_value_surface = _render_english_cardinal_like(
                    target_value, source_surface=source_value_surface
                )
                source_surface = match.group(0)
                relative_start = match.start("value") - match.start()
                relative_end = match.end("value") - match.start()
                target_surface = (
                    source_surface[:relative_start]
                    + target_value_surface
                    + source_surface[relative_end:]
                )
                requirements.append(
                    AnchoredScalarReplacementRequirement(
                        targetPaths=target_paths,
                        sourceLineIds=(line_id,),
                        sourceSurface=source_surface,
                        targetSurface=target_surface,
                    )
                )
    return tuple(requirements)


def _parse_english_cardinal(surface: str) -> int | None:
    """Parse a complete non-negative English cardinal used in package-total clauses."""

    tokens = tuple(re.findall(r"[A-Za-z]+", surface.upper()))
    if not tokens:
        return None
    total = 0
    current = 0
    used_numeric = False
    for token in tokens:
        if token == "AND":
            if not used_numeric:
                return None
            continue
        if token in _CARDINAL_SMALL:
            current += _CARDINAL_SMALL[token]
            used_numeric = True
            continue
        if token in _CARDINAL_TENS:
            current += _CARDINAL_TENS[token]
            used_numeric = True
            continue
        if token == "HUNDRED":
            if current < 1 or current >= 10:
                return None
            current *= 100
            used_numeric = True
            continue
        scale = _CARDINAL_SCALES.get(token)
        if scale is None or current < 1:
            return None
        total += current * scale
        current = 0
        used_numeric = True
    return total + current if used_numeric else None


def _english_cardinal_tokens(value: int, *, include_and: bool) -> tuple[str, ...]:
    if not 0 <= value < 1_000_000_000:
        raise ValueError("written package quantity is outside the supported cardinal range")
    small_by_value = {number: word for word, number in _CARDINAL_SMALL.items()}
    tens_by_value = {number: word for word, number in _CARDINAL_TENS.items()}

    def below_thousand(number: int) -> list[str]:
        output: list[str] = []
        hundreds, remainder = divmod(number, 100)
        if hundreds:
            output.extend((small_by_value[hundreds], "HUNDRED"))
            if remainder and include_and:
                output.append("AND")
        if remainder < 20:
            if remainder:
                output.append(small_by_value[remainder])
        else:
            tens, ones = divmod(remainder, 10)
            output.append(tens_by_value[tens * 10])
            if ones:
                output.append(small_by_value[ones])
        return output

    if value == 0:
        return ("ZERO",)
    output: list[str] = []
    for divisor, scale_word in ((1_000_000, "MILLION"), (1_000, "THOUSAND")):
        quotient, value = divmod(value, divisor)
        if quotient:
            output.extend(below_thousand(quotient))
            output.append(scale_word)
    output.extend(below_thousand(value))
    return tuple(output)


def _render_english_cardinal_like(value: int, *, source_surface: str) -> str:
    tokens = _english_cardinal_tokens(
        value,
        include_and=bool(re.search(r"\bAND\b", source_surface, re.I)),
    )
    separator = "-" if "-" in source_surface else " "
    rendered = separator.join(tokens)
    letters = "".join(character for character in source_surface if character.isalpha())
    if letters.islower():
        return rendered.lower()
    if source_surface.istitle():
        return rendered.title()
    return rendered


def _source_equipment_capacity(
    container: Mapping[str, Any],
    *,
    local_evidence: str,
    limits: TransportCapacityLimits,
) -> EquipmentCapacity:
    printed = next(
        (
            cast(str, container[key])
            for key in ("typeDescription", "typeCode")
            if isinstance(container.get(key), str)
        ),
        None,
    )
    reviewed = review_source_equipment_surface(
        printed,
        temperature_present=bool(
            re.search(r"[-+]?[0-9]+(?:[.,][0-9]+)?\s*°?\s*C\b", local_evidence, re.I)
        ),
    )
    if reviewed.size_category is not None and reviewed.type_category is not None:
        return equipment_capacity(
            {
                "sizeCategory": reviewed.size_category,
                "typeCategory": reviewed.type_category,
            },
            limits,
        )
    return equipment_capacity(container, limits)


def _dense_measurement_columns_declared(lines: Sequence[str]) -> bool:
    """Recognize the compact carrier table's gross/tare/volume column declaration.

    OCR may carry the detail rows onto a later page without repeating the headings.  Ownership
    therefore comes from one short heading block anywhere in the document, while the callers
    still require either a container-local row or an explicit shipment-total marker.  The
    combination is deliberately stricter than matching an arbitrary three-number line.
    """

    def phrase(value: str, words: str) -> bool:
        return f" {words} " in f" {value} "

    for index, line in enumerate(lines):
        # The cheap lexical gate avoids Unicode normalization for virtually every body line.
        # Exact token ownership is still decided by the normalized phrase check below.
        upper_line = line.upper()
        if "GROSS" not in upper_line or "WEIGHT" not in upper_line:
            continue
        if not phrase(_semantic_normalize(line), "GROSS WEIGHT"):
            continue
        window = lines[max(0, index - 2) : min(len(lines), index + 8)]
        context = _semantic_normalize("\n".join(window))
        volume_declared = any(
            phrase(context, value)
            for value in ("CBM", "MEASUREMENT", "CUBIC METER", "CUBIC METRE")
        )
        # Two adjacent KGS headings can mean GROSS/NET just as readily as GROSS/TARE. The
        # aggregate columns are deterministic only when TARE itself is explicitly declared.
        tare_declared = phrase(context, "TARE")
        if volume_declared and tare_declared:
            return True
    return False


def _dense_tuple_value_matches(
    line: str,
    *,
    allow_marker_prefix: bool = True,
) -> tuple[re.Match[str], re.Match[str], re.Match[str]] | None:
    """Return the three numeric tokens of one exact reviewed dense-tuple grammar."""

    trailing = _DENSE_AGGREGATE_POST_TUPLE_CAPTION.search(line)
    value_line = line[: trailing.start()] if trailing is not None else line
    if _RAW_DENSE_MEASUREMENT_TUPLE.fullmatch(value_line) is not None:
        matches = tuple(_RAW_MEASUREMENT_NUMBER.finditer(value_line))
        if len(matches) != 3:
            raise RuntimeError("dense tuple grammar and numeric tokenizer disagree")
        return matches

    marker = _DENSE_AGGREGATE_MARKER.search(value_line) if allow_marker_prefix else None
    if marker is not None:
        tail_matches = _dense_tuple_value_matches(
            value_line[marker.end() :], allow_marker_prefix=False
        )
        if tail_matches is not None:
            matches = tuple(_RAW_MEASUREMENT_NUMBER.finditer(value_line, marker.end()))[-3:]
            if len(matches) != 3:
                raise RuntimeError(
                    "marker-prefixed dense tuple tokenizer disagrees with its grammar"
                )
            return matches
        if _RAW_DENSE_MEASUREMENT_TUPLE.fullmatch(value_line[marker.end() :]) is not None:
            raise RuntimeError("marker-prefixed dense tuple tokenizer disagrees with its grammar")

    numeric_matches = tuple(_RAW_MEASUREMENT_NUMBER.finditer(value_line))
    if len(numeric_matches) < 3:
        return None
    matches = cast(
        tuple[re.Match[str], re.Match[str], re.Match[str]], numeric_matches[-3:]
    )
    if (
        _RAW_DENSE_MEASUREMENT_TUPLE.fullmatch(value_line[matches[0].start() :]) is None
        or _DENSE_AGGREGATE_CAPTION.fullmatch(value_line[: matches[0].start()]) is None
    ):
        return None
    return matches


def _dense_vertical_value_matches(
    lines: Sequence[str], start_index: int
) -> tuple[tuple[int, re.Match[str]], tuple[int, re.Match[str]], tuple[int, re.Match[str]]] | None:
    """Return exactly three consecutive numeric-only rows, never a prefix of a longer run."""

    if start_index + 3 > len(lines):
        return None
    matches = tuple(
        _RAW_DENSE_MEASUREMENT_LINE.fullmatch(lines[index])
        for index in range(start_index, start_index + 3)
    )
    if any(match is None for match in matches):
        return None
    if (
        start_index + 3 < len(lines)
        and _RAW_DENSE_MEASUREMENT_LINE.fullmatch(lines[start_index + 3]) is not None
    ):
        return None
    return cast(
        tuple[
            tuple[int, re.Match[str]],
            tuple[int, re.Match[str]],
            tuple[int, re.Match[str]],
        ],
        tuple((start_index + offset, match) for offset, match in enumerate(matches)),
    )


def _dense_aggregate_measurement_rows_from_lines(
    lines: Sequence[str], *, columns_declared: bool
) -> tuple[DenseAggregateMeasurementRow, ...]:
    """Locate aggregate rows after the caller has proved the shared dense-table header."""

    if not columns_declared:
        return ()
    output: list[DenseAggregateMeasurementRow] = []
    for marker_index, line in enumerate(lines):
        marker = _DENSE_AGGREGATE_MARKER.search(line)
        if marker is None:
            continue
        count = _parse_package_integer_surface(marker.group("count"))
        candidates: list[
            tuple[
                tuple[int, re.Match[str]],
                tuple[int, re.Match[str]],
                tuple[int, re.Match[str]],
            ]
        ] = []

        inline = _dense_tuple_value_matches(line)
        if inline is not None:
            candidates.append(
                (
                    (marker_index, inline[0]),
                    (marker_index, inline[1]),
                    (marker_index, inline[2]),
                )
            )

        next_index = marker_index + 1
        if next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if next_index < len(lines):
            next_tuple = _dense_tuple_value_matches(
                lines[next_index], allow_marker_prefix=False
            )
            if next_tuple is not None:
                candidates.append(
                    (
                        (next_index, next_tuple[0]),
                        (next_index, next_tuple[1]),
                        (next_index, next_tuple[2]),
                    )
                )
            else:
                data_index = next_index
                if _DENSE_AGGREGATE_CAPTION.fullmatch(lines[next_index]) is not None:
                    data_index += 1
                if data_index < len(lines):
                    data_tuple = _dense_tuple_value_matches(
                        lines[data_index], allow_marker_prefix=False
                    )
                    if data_tuple is not None:
                        candidates.append(
                            (
                                (data_index, data_tuple[0]),
                                (data_index, data_tuple[1]),
                                (data_index, data_tuple[2]),
                            )
                        )
                    else:
                        vertical = _dense_vertical_value_matches(lines, data_index)
                        if vertical is not None:
                            candidates.append(vertical)
        if len(candidates) > 1:
            raise ValueError(
                "shipment-total marker has multiple nearby dense measurement tuples; "
                "aggregate ownership is ambiguous"
            )
        if not candidates:
            continue
        located = candidates[0]
        last_line_index = max(row[0] for row in located)
        output.append(
            DenseAggregateMeasurementRow(
                container_count=count,
                value_line_indexes=cast(
                    tuple[int, int, int], tuple(row[0] for row in located)
                ),
                value_spans=cast(
                    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
                    tuple(match.span("value") for _, match in located),
                ),
                value_surfaces=cast(
                    tuple[str, str, str],
                    tuple(match.group("value") for _, match in located),
                ),
                evidence="\n".join(
                    lines[max(0, marker_index - 1) : last_line_index + 1]
                )[-600:],
            )
        )
    return tuple(output)


def _dense_aggregate_measurement_rows(raw_text: str) -> tuple[DenseAggregateMeasurementRow, ...]:
    """Locate exact three-column rows owned by ``Weight in Kgs Total`` markers.

    Values can share the marker line, form one strict tuple after a reviewed continuation caption,
    or occupy three consecutive numeric rows after that caption. More than one candidate for a
    marker is ambiguous and is rejected rather than guessed.
    """

    lines = raw_text.splitlines()
    return _dense_aggregate_measurement_rows_from_lines(
        lines,
        columns_declared=_dense_measurement_columns_declared(lines),
    )


def _container_measurement_occurrences(
    raw_text: str,
    containers: Sequence[Mapping[str, Any]],
) -> dict[
    int,
    tuple[
        tuple[
            OperationalKind,
            int,
            str,
            str,
            OperationalGrammar,
        ],
        ...,
    ],
]:
    """Find only measurement rows in the local printed block headed by a container number."""

    lines = raw_text.splitlines()
    numbers = [
        cast(str, row["containerNumber"])
        for row in containers
        if isinstance(row.get("containerNumber"), str)
    ]
    number_patterns = tuple(
        re.compile(rf"(?<![A-Z0-9]){re.escape(number)}(?![A-Z0-9])", re.I)
        for number in numbers
    )
    header_patterns = tuple(
        re.compile(rf"^[ \t]*{re.escape(number)}(?:\b|(?=[ \t/,:#-]))", re.I)
        for number in numbers
    )
    number_matches_by_line = tuple(
        tuple(
            (container_index, match)
            for container_index, pattern in enumerate(number_patterns)
            if (match := pattern.search(line)) is not None
        )
        for line in lines
    )
    output: dict[
        int,
        list[
            tuple[
                OperationalKind,
                int,
                str,
                str,
                OperationalGrammar,
            ]
        ],
    ] = {}
    dense_header = _dense_measurement_columns_declared(lines)

    def append_once(
        container_index: int,
        row: tuple[
            OperationalKind,
            int,
            str,
            str,
            OperationalGrammar,
        ],
    ) -> None:
        identity = (row[0], row[1])
        if any(
            (existing[0], existing[1]) == identity for existing in output.get(container_index, ())
        ):
            return
        output.setdefault(container_index, []).append(row)

    for container_index, header in enumerate(header_patterns):
        for line_index, line in enumerate(lines):
            if header.search(line) is None:
                continue
            evidence_lines = [line]
            found: set[str] = set()
            # Carrier container-detail blocks put seal/tare between the container header and the
            # local measurements. Six following occupied lines covers the reviewed grammar while
            # preventing a last container from absorbing document-level totals farther below.
            for offset in range(0, 7):
                cursor = line_index + offset
                if cursor >= len(lines):
                    break
                current = lines[cursor]
                if offset and (
                    _PAGE_MARKER.fullmatch(current)
                    or not current.strip()
                    or any(
                        other_index != container_index
                        for other_index, _match in number_matches_by_line[cursor]
                    )
                ):
                    break
                if offset:
                    evidence_lines.append(current)
                for kind in ("gross_weight_kg", "tare_weight_kg", "volume_m3"):
                    if kind in found:
                        continue
                    match = _operational_measurement_match(current, kind)
                    if match is not None and _parse_measurement_surface(
                        match.group("value"), maximum=None
                    ) is not None:
                        append_once(
                            container_index,
                            (
                                cast(Any, kind),
                                cursor + 1,
                                match.group("value"),
                                "\n".join(evidence_lines)[-600:],
                                "labeled_measurement",
                            ),
                        )
                        found.add(kind)
                if "package_quantity" not in found:
                    package_match = _RAW_PACKAGE_QUANTITY.search(current)
                    if package_match is not None:
                        append_once(
                            container_index,
                            (
                                "package_quantity",
                                cursor + 1,
                                package_match.group("value"),
                                "\n".join(evidence_lines)[-600:],
                                "labeled_measurement",
                            ),
                        )
                        found.add("package_quantity")
                if len(found) == 4:
                    break

            # A compact carrier table may print gross/tare/volume either as three numeric-only
            # rows or as one three-column row after each container/package row. Multi-page tables
            # often repeat the rows but not their page-one column declaration, so the strong
            # document-level header proof is combined with this exact local container topology.
            if dense_header:
                package_cursor: int | None = None
                for cursor in range(line_index + 1, min(len(lines), line_index + 5)):
                    candidate = lines[cursor]
                    if (
                        _PAGE_MARKER.fullmatch(candidate) is not None
                        or not candidate.strip()
                        or any(
                            other_index != container_index
                            for other_index, _match in number_matches_by_line[cursor]
                        )
                    ):
                        break
                    if _RAW_PACKAGE_QUANTITY.search(candidate) is not None:
                        package_cursor = cursor
                        break
                if package_cursor is not None:
                    tuple_index = package_cursor + 1
                    tuple_match = (
                        _RAW_DENSE_MEASUREMENT_TUPLE.fullmatch(lines[tuple_index])
                        if tuple_index < len(lines)
                        else None
                    )
                    if tuple_match is not None:
                        evidence = "\n".join(lines[line_index : tuple_index + 1])[-600:]
                        for kind, group in (
                            ("gross_weight_kg", "gross"),
                            ("tare_weight_kg", "tare"),
                            ("volume_m3", "volume"),
                        ):
                            append_once(
                                container_index,
                                (
                                    cast(OperationalKind, kind),
                                    tuple_index + 1,
                                    tuple_match.group(group),
                                    evidence,
                                    "dense_container_table",
                                ),
                            )
                    else:
                        numeric_rows: list[tuple[int, re.Match[str]]] = []
                        cursor = package_cursor + 1
                        while cursor < len(lines) and len(numeric_rows) < 3:
                            if not lines[cursor].strip():
                                break
                            match = _RAW_DENSE_MEASUREMENT_LINE.fullmatch(lines[cursor])
                            if match is None:
                                break
                            numeric_rows.append((cursor, match))
                            cursor += 1
                        if len(numeric_rows) != 3:
                            continue
                        evidence = "\n".join(lines[line_index:cursor])[-600:]
                        for kind, (row_index, match) in zip(
                            ("gross_weight_kg", "tare_weight_kg", "volume_m3"),
                            numeric_rows,
                            strict=True,
                        ):
                            append_once(
                                container_index,
                                (
                                    cast(OperationalKind, kind),
                                    row_index + 1,
                                    match.group("value"),
                                    evidence,
                                    "dense_container_table",
                                ),
                            )

    # A one-container shipment-total tuple is also that container's only or repeated operational
    # row. Bind a numeric-only tuple directly so it receives the same allocation, empirical
    # resampling, and exact grammar audit as a local detail row. Multi-container totals are
    # reconciled separately because assigning the whole total to any one container is false.
    aggregate_rows = _dense_aggregate_measurement_rows_from_lines(
        lines,
        columns_declared=dense_header,
    )
    if aggregate_rows and len(numbers) == 1 and all(
        row.container_count == 1 for row in aggregate_rows
    ):
        for aggregate in aggregate_rows:
            for kind, line_index, surface in zip(
                ("gross_weight_kg", "tare_weight_kg", "volume_m3"),
                aggregate.value_line_indexes,
                aggregate.value_surfaces,
                strict=True,
            ):
                append_once(
                    0,
                    (
                        cast(OperationalKind, kind),
                        line_index + 1,
                        surface,
                        aggregate.evidence,
                        "dense_container_table",
                    ),
                )

    # The most common carrier grammar puts container, seal/equipment, package quantity, gross
    # weight, and volume on one row.  Bind only values *after* the exact container identifier;
    # this deliberately excludes the shifted ``(...)/NEXT_CONTAINER`` grammar handled below,
    # where the measurements before the identifier belong to the preceding container.
    for line_index, line in enumerate(lines):
        for container_index, number_match in number_matches_by_line[line_index]:
            tail = line[number_match.end() :]
            evidence = line[-600:]
            for kind, pattern in (
                ("gross_weight_kg", _RAW_TRAILING_GROSS_WEIGHT),
                ("volume_m3", _RAW_TRAILING_VOLUME),
                ("package_quantity", _RAW_PACKAGE_QUANTITY),
            ):
                match = pattern.search(tail)
                if match is None:
                    continue
                append_once(
                    container_index,
                    (
                        cast(Any, kind),
                        line_index + 1,
                        match.group("value"),
                        evidence,
                        "labeled_measurement",
                    ),
                )
    # Some carrier riders encode one container's measures at the start of the line that introduces
    # the next container. Preserve that topology by assigning the parenthesized tuple to the most
    # recently printed container before advancing the container cursor.
    last_container_index: int | None = None
    for line_index, line in enumerate(lines):
        inline = _RAW_INLINE_CONTAINER_MEASURES.search(line)
        if inline is not None and last_container_index is not None:
            evidence = "\n".join(lines[max(0, line_index - 1) : line_index + 1])[-600:]
            for row in (
                (
                    "gross_weight_kg",
                    line_index + 1,
                    inline.group("gross"),
                    evidence,
                    "inline_container_breakdown",
                ),
                (
                    "volume_m3",
                    line_index + 1,
                    inline.group("volume"),
                    evidence,
                    "inline_container_breakdown",
                ),
                (
                    "package_quantity",
                    line_index + 1,
                    inline.group("packages"),
                    evidence,
                    "inline_container_breakdown",
                ),
            ):
                append_once(last_container_index, row)
        if number_matches_by_line[line_index]:
            last_container_index = number_matches_by_line[line_index][0][0]
    return {key: tuple(value) for key, value in output.items()}


def build_empirical_operational_profiles(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    target_field: str,
    limits: TransportCapacityLimits,
) -> tuple[EmpiricalOperationalProfile, ...]:
    """Build paired real-document utilization support for raw-only measurement synthesis."""

    profiles: list[EmpiricalOperationalProfile] = []
    seen: set[tuple[str, str, Decimal | None, Decimal | None, Decimal | None]] = set()
    for row in source_rows:
        document_id = row.get("documentId")
        raw_text = row.get("joinedRawText")
        target = row.get(target_field)
        if (
            not isinstance(document_id, str)
            or not isinstance(raw_text, str)
            or not isinstance(target, Mapping)
        ):
            continue
        patch = target.get("documentPatch")
        if not isinstance(patch, Mapping):
            continue
        containers = tuple(
            cast(Mapping[str, Any], value)
            for value in cast(Sequence[Any], patch.get("containers") or ())
            if isinstance(value, Mapping)
        )
        occurrences = _container_measurement_occurrences(raw_text, containers)
        for container_index, rows in occurrences.items():
            if container_index >= len(containers):
                continue
            container = containers[container_index]
            evidence = "\n".join(value[3] for value in rows)
            capacity = _source_equipment_capacity(
                container,
                local_evidence=evidence,
                limits=limits,
            )
            values: dict[str, Decimal] = {}
            for kind, _, surface, _, _ in rows:
                if kind == "package_quantity":
                    continue
                maximum = (
                    capacity.payload_kg
                    if kind == "gross_weight_kg"
                    else (capacity.volume_m3 if kind == "volume_m3" else None)
                )
                parsed = _parse_measurement_surface(surface, maximum=maximum)
                if parsed is not None:
                    values[kind] = parsed.value
            gross = values.get("gross_weight_kg")
            tare = values.get("tare_weight_kg")
            volume = values.get("volume_m3")
            if gross is None and tare is None and volume is None:
                continue
            container_number = container.get("containerNumber")
            if not isinstance(container_number, str):
                continue
            identity = (document_id, container_number, gross, tare, volume)
            if identity in seen:
                continue
            seen.add(identity)
            profiles.append(
                EmpiricalOperationalProfile(
                    document_id=document_id,
                    container_number=container_number,
                    equipment_family=capacity.family,
                    gross_weight_kg=gross,
                    gross_utilization=(gross / capacity.payload_kg if gross is not None else None),
                    tare_weight_kg=tare,
                    volume_m3=volume,
                    volume_utilization=(
                        volume / capacity.volume_m3
                        if volume is not None and capacity.volume_m3 is not None
                        else None
                    ),
                )
            )
    return tuple(sorted(profiles, key=lambda row: (row.document_id, row.container_number)))


_MASS_TO_KG = {
    "kilogram": Decimal(1),
    "metric_tonne": Decimal(1000),
    "pound": Decimal("0.45359237"),
}


def _target_measure_value(
    group: Mapping[str, Any],
    kind: Literal["gross_weight_kg", "net_weight_kg", "volume_m3"],
) -> Decimal | None:
    field_name = {
        "gross_weight_kg": "grossWeight",
        "net_weight_kg": "netWeight",
        "volume_m3": "volume",
    }[kind]
    measure = group.get(field_name)
    if measure is None:
        return None
    if not isinstance(measure, Mapping) or not isinstance(measure.get("unit"), str):
        raise ValueError(f"target {field_name} is not a canonical measure")
    try:
        value = Decimal(str(measure["value"]))
    except (KeyError, ArithmeticError) as error:
        raise ValueError(f"target {field_name} has no finite numeric value") from error
    if kind in {"gross_weight_kg", "net_weight_kg"}:
        factor = _MASS_TO_KG.get(cast(str, measure["unit"]))
        if factor is None:
            raise ValueError(f"unsupported target {field_name} unit: {measure['unit']}")
        value *= factor
    elif measure["unit"] != "cubic_metre":
        raise ValueError(f"unsupported target volume unit: {measure['unit']}")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"target {field_name} must be finite and positive")
    return value


def _decimal_places(value: Decimal) -> int:
    return max(0, -cast(int, value.as_tuple().exponent))


def _measurement_style_with_exact_target_precision(
    style: ParsedMeasurementSurface,
    target: Decimal,
) -> ParsedMeasurementSurface:
    """Retain source separators while ensuring the printed number equals target truth.

    A generated value can carry more precision than its source template (for example a source
    ``16,000`` and target ``24715.9``). Rounding that to ``24,716`` makes the OCR disagree with
    its training label. Keep the grouping convention and introduce only the minimum compatible
    decimal precision needed to represent the target exactly.
    """

    required_places = _decimal_places(target.normalize()) if target != target.to_integral() else 0
    if required_places <= style.decimal_places:
        return style
    decimal_separator = style.decimal_separator
    if decimal_separator is None:
        decimal_separator = "," if style.grouping_separator == "." else "."
    return ParsedMeasurementSurface(
        value=style.value,
        decimal_separator=decimal_separator,
        grouping_separator=style.grouping_separator,
        decimal_places=required_places,
    )


def _bounded_largest_remainder_allocation(
    total: int,
    weights: Sequence[int],
    maxima: Sequence[int],
) -> tuple[int, ...]:
    """Allocate exact integer units by weights without exceeding per-recipient capacity.

    Package counts are useful allocation weights, but they do not imply uniform cargo density.
    When their direct proportional split would overload one equipment unit, cap that unit and
    deterministically redistribute the remainder.  This retains exact aggregate arithmetic and
    uses only target relationships and configured physical capacities.
    """

    if total < 0 or not weights or len(weights) != len(maxima):
        raise ValueError("bounded allocation received an invalid shape")
    if any(weight <= 0 for weight in weights) or any(maximum < 0 for maximum in maxima):
        raise ValueError("bounded allocation weights must be positive and maxima non-negative")
    if total > sum(maxima):
        raise ValueError("bounded allocation total exceeds combined capacity")
    allocated = [0] * len(weights)
    active = list(range(len(weights)))
    remaining = total
    while active:
        proposed = largest_remainder_allocation(
            remaining,
            tuple(weights[index] for index in active),
        )
        overflowing = [
            index for index, value in zip(active, proposed, strict=True) if value > maxima[index]
        ]
        if not overflowing:
            for index, value in zip(active, proposed, strict=True):
                allocated[index] = value
            remaining = 0
            break
        for index in overflowing:
            allocated[index] = maxima[index]
            remaining -= maxima[index]
        saturated = set(overflowing)
        active = [index for index in active if index not in saturated]
    if remaining != 0 or sum(allocated) != total:
        raise RuntimeError("bounded allocation failed exact conservation")
    return tuple(allocated)


def _target_measure_allocations(
    target_label: Mapping[str, Any],
    *,
    kind: Literal["gross_weight_kg", "volume_m3"],
    printed_precision: int,
    limits: TransportCapacityLimits,
) -> dict[str, Decimal]:
    """Allocate each labeled cargo-group measure across its explicit container relations."""

    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    containers = {
        cast(str, row["containerNumber"]): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], patch.get("containers") or ())
        if isinstance(row, Mapping) and isinstance(row.get("containerNumber"), str)
    }
    allocation_groups = {
        cast(str, row["groupId"]): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], patch.get("cargoAllocationGroups") or ())
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    output: dict[str, Decimal] = {}
    for group in cast(Sequence[Any], patch.get("cargoGroups") or ()):
        if not isinstance(group, Mapping) or not isinstance(group.get("groupId"), str):
            continue
        total = _target_measure_value(group, kind)
        if total is None:
            continue
        allocation_group = allocation_groups.get(cast(str, group["groupId"]))
        weights: dict[str, int] = {}
        if allocation_group is not None:
            for allocation in cast(Sequence[Any], allocation_group.get("allocations") or ()):
                if not isinstance(allocation, Mapping):
                    continue
                number = allocation.get("containerNumber")
                quantity = allocation.get("packageQuantity")
                if isinstance(number, str) and isinstance(quantity, int) and quantity >= 0:
                    weights[number] = weights.get(number, 0) + quantity
        if not weights and len(containers) == 1:
            weights = {next(iter(containers)): 1}
        unknown = sorted(set(weights) - set(containers))
        if unknown or not weights or sum(weights.values()) <= 0:
            raise ValueError(
                f"target {kind} has no complete positive container allocation for "
                f"group {group['groupId']}: unknown={unknown}"
            )
        precision = max(printed_precision, _decimal_places(total))
        scale = 10**precision
        total_units = int((total * scale).to_integral_exact())
        numbers = tuple(weights)
        maxima: list[int] = []
        for number in numbers:
            capacity = equipment_capacity(containers[number], limits)
            maximum = capacity.payload_kg if kind == "gross_weight_kg" else capacity.volume_m3
            if maximum is None:
                maxima.append(total_units)
                continue
            available = maximum - output.get(number, Decimal(0))
            maxima.append(max(0, int((available * scale).to_integral_value(rounding=ROUND_DOWN))))
        units = _bounded_largest_remainder_allocation(
            total_units,
            tuple(weights[row] for row in numbers),
            tuple(maxima),
        )
        for number, allocated_units in zip(numbers, units, strict=True):
            output[number] = output.get(number, Decimal(0)) + Decimal(allocated_units) / scale
    for number, value in output.items():
        capacity = equipment_capacity(containers[number], limits)
        maximum = capacity.payload_kg if kind == "gross_weight_kg" else capacity.volume_m3
        if maximum is not None and value > maximum:
            raise ValueError(
                f"allocated target {kind} exceeds {number} capacity: {value} > {maximum}"
            )
    return output


def _target_package_allocations(target_label: Mapping[str, Any]) -> dict[str, int]:
    """Return the explicit task-facing package count allocated to each container."""

    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    containers = {
        cast(str, row["containerNumber"])
        for row in cast(Sequence[Any], patch.get("containers") or ())
        if isinstance(row, Mapping) and isinstance(row.get("containerNumber"), str)
    }
    output: dict[str, int] = {}
    for group in cast(Sequence[Any], patch.get("cargoAllocationGroups") or ()):
        if not isinstance(group, Mapping):
            continue
        coverage = group.get("coverage")
        for allocation in cast(Sequence[Any], group.get("allocations") or ()):
            if not isinstance(allocation, Mapping):
                continue
            number = allocation.get("containerNumber")
            quantity = allocation.get("packageQuantity")
            if coverage == "container_membership_only":
                if not isinstance(number, str) or quantity is not None:
                    raise ValueError("target membership-only allocation has package facts")
                if number not in containers:
                    raise ValueError(f"target cargo allocation names unknown container {number}")
                continue
            if not isinstance(number, str) or not isinstance(quantity, int) or quantity < 0:
                raise ValueError("target cargo allocation has an invalid package quantity")
            if number not in containers:
                raise ValueError(f"target cargo allocation names unknown container {number}")
            output[number] = output.get(number, 0) + quantity
    return output


def _single_container_unallocated_package_projection(
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    source_containers: Sequence[Mapping[str, Any]],
    target_containers: Sequence[Mapping[str, Any]],
    occurrences: Mapping[
        int,
        Sequence[
            tuple[
                OperationalKind,
                int,
                str,
                str,
                OperationalGrammar,
            ]
        ],
    ],
) -> dict[str, int]:
    """Project an aggregate package row when exactly one container owns the shipment.

    Some reviewed labels omit an allocation object because a one-container document has no
    relationship ambiguity.  The printed container row still owns the aggregate package count.
    We may project it only when that count equals the sum of *all* source package leaves and the
    source/target package identities are unchanged.  This excludes hierarchical package labels,
    partial rows, and every multi-container case instead of guessing an allocation.
    """

    if len(source_containers) != 1 or len(target_containers) != 1:
        return {}
    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    if source_patch.get("cargoAllocationGroups") or target_patch.get("cargoAllocationGroups"):
        return {}
    printed = tuple(
        _parse_package_integer_surface(surface)
        for kind, _, surface, _, _ in occurrences.get(0, ())
        if kind == "package_quantity"
    )
    if len(printed) != 1:
        return {}
    source_packages = tuple(
        row
        for row in cast(Sequence[Any], source_patch.get("cargoPackages") or ())
        if isinstance(row, Mapping)
    )
    target_packages = tuple(
        row
        for row in cast(Sequence[Any], target_patch.get("cargoPackages") or ())
        if isinstance(row, Mapping)
    )
    source_keys = tuple((row.get("groupId"), row.get("packageId")) for row in source_packages)
    target_keys = tuple((row.get("groupId"), row.get("packageId")) for row in target_packages)
    source_quantities = tuple(row.get("quantity") for row in source_packages)
    target_quantities = tuple(row.get("quantity") for row in target_packages)
    if (
        not source_packages
        or source_keys != target_keys
        or any(not isinstance(value, int) or value <= 0 for value in source_quantities)
        or any(not isinstance(value, int) or value <= 0 for value in target_quantities)
        or sum(cast(tuple[int, ...], source_quantities)) != printed[0]
    ):
        return {}
    target_number = target_containers[0].get("containerNumber")
    if not isinstance(target_number, str):
        raise ValueError("single-container target has no container number")
    return {target_number: sum(cast(tuple[int, ...], target_quantities))}


def _parse_package_integer_surface(surface: str) -> int:
    """Parse the deliberately narrow integer grammar accepted by package-row discovery."""

    value = surface.strip()
    if re.fullmatch(r"[0-9][0-9,]*[.,]00", value) is not None:
        value = value[:-3]
    normalized = value.replace(",", "")
    if not normalized.isdigit():
        raise ValueError(f"cannot parse printed package quantity {surface!r}")
    return int(normalized)


def _membership_package_allocations(
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    source_containers: Sequence[Mapping[str, Any]],
    target_containers: Sequence[Mapping[str, Any]],
    occurrences: Mapping[
        int,
        Sequence[
            tuple[
                OperationalKind,
                int,
                str,
                str,
                OperationalGrammar,
            ]
        ],
    ],
) -> dict[str, int]:
    """Project membership-only rows through the nearest evidenced package level.

    ``container_membership_only`` deliberately omits quantities because the reviewed label could
    not prove which package level each container row represents.  The raw template can still
    provide that missing ownership: its per-container printed quantities form one exact total.
    We bind that total to the unique source package level with the same quantity, or to the unique
    immediately enclosing task-facing level (the smallest greater quantity).  Package IDs and
    group topology are preserved by synthesis, so the corresponding target quantity can then be
    distributed with the source row proportions.  Ambiguous ties and incomplete rows fail closed.
    """

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    source_groups = {
        cast(str, row["groupId"]): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], source_patch.get("cargoAllocationGroups") or ())
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    source_packages = {
        (cast(str, row["groupId"]), cast(str, row["packageId"])): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], source_patch.get("cargoPackages") or ())
        if isinstance(row, Mapping)
        and isinstance(row.get("groupId"), str)
        and isinstance(row.get("packageId"), str)
    }
    target_packages = {
        (cast(str, row["groupId"]), cast(str, row["packageId"])): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], target_patch.get("cargoPackages") or ())
        if isinstance(row, Mapping)
        and isinstance(row.get("groupId"), str)
        and isinstance(row.get("packageId"), str)
    }
    source_number_by_index = {
        index: cast(str, row["containerNumber"])
        for index, row in enumerate(source_containers)
        if isinstance(row.get("containerNumber"), str)
    }
    target_number_by_source = {
        cast(str, source["containerNumber"]): cast(str, target["containerNumber"])
        for source, target in zip(source_containers, target_containers, strict=True)
        if isinstance(source.get("containerNumber"), str)
        and isinstance(target.get("containerNumber"), str)
    }
    printed_quantities_by_source: dict[str, tuple[int, ...]] = {}
    for container_index, rows in occurrences.items():
        quantities = tuple(
            _parse_package_integer_surface(surface)
            for kind, _, surface, _, _ in rows
            if kind == "package_quantity"
        )
        if not quantities:
            continue
        source_number = source_number_by_index.get(container_index)
        if source_number is None:
            raise ValueError("printed package row has no source container identity")
        printed_quantities_by_source[source_number] = quantities

    output: dict[str, int] = {}
    for target_group in cast(Sequence[Any], target_patch.get("cargoAllocationGroups") or ()):
        if not isinstance(target_group, Mapping) or target_group.get("coverage") != (
            "container_membership_only"
        ):
            continue
        group_id = target_group.get("groupId")
        if not isinstance(group_id, str):
            raise ValueError("target membership-only allocation has no groupId")
        source_group = source_groups.get(group_id)
        if source_group is None or source_group.get("coverage") != "container_membership_only":
            raise ValueError("membership-only allocation topology changed across synthesis")
        source_members = [
            row.get("containerNumber")
            for row in cast(Sequence[Any], source_group.get("allocations") or ())
            if isinstance(row, Mapping)
        ]
        target_members = [
            row.get("containerNumber")
            for row in cast(Sequence[Any], target_group.get("allocations") or ())
            if isinstance(row, Mapping)
        ]
        if (
            not source_members
            or len(source_members) != len(target_members)
            or any(not isinstance(number, str) for number in (*source_members, *target_members))
        ):
            raise ValueError("membership-only allocation changed its container topology")
        expected_targets = [target_number_by_source[cast(str, number)] for number in source_members]
        if expected_targets != target_members:
            raise ValueError("membership-only allocation changed its source-to-target ordering")
        try:
            observed_presence = tuple(
                cast(str, number) in printed_quantities_by_source for number in source_members
            )
            if not any(observed_presence):
                # Membership-only labels intentionally carry no per-container quantity.  When
                # the template likewise prints no container-local package row, there is nothing
                # to project or rewrite.  Requiring evidence in that case blocked valid
                # single-container riders whose package count is printed only in the cargo body.
                continue
            if not all(observed_presence):
                raise ValueError(
                    "membership-only allocation has an incomplete printed breakdown"
                )
            printed_rows = tuple(
                printed_quantities_by_source[cast(str, number)] for number in source_members
            )
        except KeyError as error:
            raise ValueError(
                "membership-only allocation has an incomplete printed breakdown"
            ) from error
        if any(len(rows) != 1 for rows in printed_rows):
            raise ValueError("membership-only container has multiple printed package quantities")
        weights = tuple(rows[0] for rows in printed_rows)
        if any(value <= 0 for value in weights):
            raise ValueError("membership-only printed package quantities must be positive")
        source_total = sum(weights)
        package_candidates = [
            row
            for (candidate_group, _), row in source_packages.items()
            if candidate_group == group_id
            and isinstance(row.get("quantity"), int)
            and cast(int, row["quantity"]) >= source_total
        ]
        exact = [row for row in package_candidates if row.get("quantity") == source_total]
        if len(exact) == 1:
            source_package = exact[0]
        elif not exact and package_candidates:
            nearest_quantity = min(cast(int, row["quantity"]) for row in package_candidates)
            nearest = [row for row in package_candidates if row.get("quantity") == nearest_quantity]
            if len(nearest) != 1:
                raise ValueError("membership-only printed package level has an ambiguous owner")
            source_package = nearest[0]
        else:
            raise ValueError("membership-only printed package level has no task-facing owner")
        package_id = cast(str, source_package["packageId"])
        target_package = target_packages.get((group_id, package_id))
        target_total = target_package.get("quantity") if target_package is not None else None
        if not isinstance(target_total, int) or target_total <= 0:
            raise ValueError("membership-only target package owner has no positive quantity")
        distributed = largest_remainder_allocation(target_total, weights)
        if any(
            source > 0 and target <= 0 for source, target in zip(weights, distributed, strict=True)
        ):
            raise ValueError("membership-only projection erased a positive container row")
        for number, quantity in zip(target_members, distributed, strict=True):
            target_number = cast(str, number)
            if target_number in output:
                raise ValueError("multiple membership-only groups own one printed package row")
            output[target_number] = quantity
    return output


def _render_integer_surface(value: int, *, source_surface: str) -> str:
    if value < 0:
        raise ValueError("cannot render a negative package quantity")
    decimal_match = re.fullmatch(
        r"(?P<whole>[0-9][0-9,]*)(?P<separator>[.,])(?P<zeroes>0+)", source_surface
    )
    if decimal_match is not None:
        whole = f"{value:,}" if "," in decimal_match.group("whole") else str(value)
        return whole + decimal_match.group("separator") + decimal_match.group("zeroes")
    return f"{value:,}" if "," in source_surface else str(value)


def _operational_capacity_maximum(
    kind: Literal["gross_weight_kg", "tare_weight_kg", "volume_m3"],
    capacity: EquipmentCapacity,
) -> Decimal | None:
    """Return a physical ceiling only for measures that equipment capacity constrains."""

    if kind == "gross_weight_kg":
        return capacity.payload_kg
    if kind == "volume_m3":
        return capacity.volume_m3
    return None


def _profile_operational_value(
    profile: EmpiricalOperationalProfile,
    kind: Literal["gross_weight_kg", "tare_weight_kg", "volume_m3"],
) -> tuple[Decimal | None, Decimal | None]:
    """Return ``(absolute, utilization)`` for one empirical operational field."""

    if kind == "gross_weight_kg":
        return profile.gross_weight_kg, profile.gross_utilization
    if kind == "volume_m3":
        return profile.volume_m3, profile.volume_utilization
    return profile.tare_weight_kg, None


def _resampled_empirical_value(
    *,
    absolute: Decimal | None,
    utilization: Decimal | None,
    maximum: Decimal | None,
) -> Decimal | None:
    """Scale a utilization only to the precision actually observed in its source profile."""

    if utilization is None or maximum is None:
        return absolute
    if absolute is None:
        # Every utilization produced by the profile builder has its source absolute beside it.
        # Refusing an incomplete external profile avoids inventing measurement precision.
        return None
    value = utilization * maximum
    quantum = Decimal(1).scaleb(-_decimal_places(absolute))
    value = value.quantize(quantum, rounding=ROUND_HALF_UP)
    if value > maximum:
        value = maximum.quantize(quantum, rounding=ROUND_DOWN)
    return value


def operational_flavor_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    *,
    source_document_id: str,
    scenario_id: str,
    profiles: Sequence[EmpiricalOperationalProfile],
    limits: TransportCapacityLimits,
) -> tuple[OperationalFlavorRequirement, ...]:
    """Create exact capacity-safe edits for container-local operational measurements.

    When the target label contains an aggregate measure, its explicit cargo-to-container
    allocations determine a lossless per-container breakdown. Otherwise, raw-only measurements
    are resampled from paired real-document utilization profiles.
    """

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    source_containers = tuple(
        cast(Mapping[str, Any], value)
        for value in cast(Sequence[Any], source_patch.get("containers") or ())
        if isinstance(value, Mapping)
    )
    target_containers = tuple(
        cast(Mapping[str, Any], value)
        for value in cast(Sequence[Any], target_patch.get("containers") or ())
        if isinstance(value, Mapping)
    )
    if len(source_containers) != len(target_containers):
        raise ValueError("cannot pair raw operational flavor across different container counts")
    occurrences = _container_measurement_occurrences(raw_text, source_containers)
    if not occurrences:
        return ()
    requirements: list[OperationalFlavorRequirement] = []
    kinds: tuple[Literal["gross_weight_kg", "tare_weight_kg", "volume_m3"], ...] = (
        "gross_weight_kg",
        "tare_weight_kg",
        "volume_m3",
    )
    labeled_kinds: tuple[Literal["gross_weight_kg", "volume_m3"], ...] = (
        "gross_weight_kg",
        "volume_m3",
    )
    source_fields = {"gross_weight_kg": "grossWeight", "volume_m3": "volume"}
    target_has_measure: dict[OperationalKind, bool] = {
        **{
            measure_kind: _cargo_measure_present(target_label, source_fields[measure_kind])
            for measure_kind in labeled_kinds
        },
        "tare_weight_kg": False,
    }
    for measure_kind in labeled_kinds:
        if (
            _cargo_measure_present(source_label, source_fields[measure_kind])
            and not target_has_measure[measure_kind]
        ):
            raise ValueError(
                f"source exposes labeled {measure_kind} but synthetic target removes it "
                "from a fixed measurement topology"
            )
    printed_precision: dict[OperationalKind, int] = {kind: 0 for kind in kinds}
    source_capacities: dict[int, EquipmentCapacity] = {}
    source_canonical_values: dict[tuple[int, int], Decimal] = {}
    for container_index, observed_rows in occurrences.items():
        source_capacity = _source_equipment_capacity(
            source_containers[container_index],
            local_evidence="\n".join(row[3] for row in observed_rows),
            limits=limits,
        )
        source_capacities[container_index] = source_capacity
        for row_index, (operational_kind, _, surface, _, _) in enumerate(observed_rows):
            if operational_kind == "package_quantity":
                continue
            maximum = _operational_capacity_maximum(operational_kind, source_capacity)
            parsed = _parse_measurement_surface(surface, maximum=maximum)
            if parsed is None:
                raise ValueError(f"cannot parse source {operational_kind} surface {surface!r}")
            source_canonical_values[(container_index, row_index)] = parsed.value
            printed_precision[operational_kind] = max(
                printed_precision[operational_kind], parsed.decimal_places
            )
    allocated_values = {
        measure_kind: (
            _target_measure_allocations(
                target_label,
                kind=measure_kind,
                printed_precision=printed_precision[measure_kind],
                limits=limits,
            )
            if target_has_measure[measure_kind]
            else {}
        )
        for measure_kind in labeled_kinds
    }
    package_allocations = _target_package_allocations(target_label)
    membership_package_allocations = _membership_package_allocations(
        source_label,
        target_label,
        source_containers,
        target_containers,
        occurrences,
    )
    unallocated_single_container = _single_container_unallocated_package_projection(
        source_label,
        target_label,
        source_containers,
        target_containers,
        occurrences,
    )
    overlap = set(package_allocations) & set(membership_package_allocations)
    if overlap:
        raise ValueError(
            "printed package rows have both explicit and membership-only owners: "
            + ", ".join(sorted(overlap))
        )
    package_allocations.update(membership_package_allocations)
    overlap = set(package_allocations) & set(unallocated_single_container)
    if overlap:
        raise ValueError(
            "single-container package projection overlaps an explicit allocation: "
            + ", ".join(sorted(overlap))
        )
    package_allocations.update(unallocated_single_container)
    raw_lines = raw_text.splitlines()
    source_numbers = [
        cast(str, row["containerNumber"])
        for row in source_containers
        if isinstance(row.get("containerNumber"), str)
    ]
    for container_index, observed_rows in occurrences.items():
        target_container = target_containers[container_index]
        target_number = target_container.get("containerNumber")
        if not isinstance(target_number, str):
            raise ValueError("synthetic container has no containerNumber")
        capacity = equipment_capacity(target_container, limits)
        source_capacity = source_capacities[container_index]
        empirical_kinds = {
            row[0]
            for row in observed_rows
            if row[0] != "package_quantity" and not target_has_measure[row[0]]
        }
        candidates = [
            row
            for row in profiles
            if row.document_id != source_document_id
            and all(
                (row.gross_utilization is not None)
                if kind == "gross_weight_kg"
                else (
                    row.tare_weight_kg is not None
                    if kind == "tare_weight_kg"
                    else (
                        _profile_operational_value(row, kind)[1] is not None
                        or _profile_operational_value(row, kind)[0] is not None
                    )
                )
                for kind in empirical_kinds
            )
        ]
        same_family = [row for row in candidates if row.equipment_family == capacity.family]
        if same_family:
            candidates = same_family
        elif "tare_weight_kg" in empirical_kinds:
            raise ValueError(
                "no same-equipment-family tare profile supports container " + target_number
            )
        if empirical_kinds and not candidates:
            raise ValueError(f"no empirical operational profile supports container {target_number}")
        digest = int.from_bytes(
            bytes.fromhex(
                sha256_bytes(f"{scenario_id}\0{target_number}\0operational-profile-v1".encode())
            ),
            "big",
        )
        selected_profile: EmpiricalOperationalProfile | None = None
        selected_empirical_surfaces: dict[int, str] = {}
        selected_empirical_values: dict[int, Decimal] = {}
        for offset in range(max(1, len(candidates))):
            if not empirical_kinds:
                break
            candidate = candidates[(digest + offset) % len(candidates)]
            rendered: dict[int, str] = {}
            rendered_values: dict[int, Decimal] = {}
            for row_index, (operational_kind, _, source_surface, _, _) in enumerate(observed_rows):
                if operational_kind == "package_quantity":
                    continue
                if operational_kind not in empirical_kinds:
                    continue
                maximum = _operational_capacity_maximum(operational_kind, capacity)
                source_maximum = _operational_capacity_maximum(operational_kind, source_capacity)
                source_style = _parse_measurement_surface(source_surface, maximum=source_maximum)
                absolute, utilization = _profile_operational_value(candidate, operational_kind)
                target_value = _resampled_empirical_value(
                    absolute=absolute,
                    utilization=utilization,
                    maximum=maximum,
                )
                if source_style is None or target_value is None:
                    rendered = {}
                    break
                rendered[row_index] = _render_measurement_surface(
                    target_value,
                    style=_measurement_style_with_exact_target_precision(
                        source_style, target_value
                    ),
                    maximum=maximum,
                )
                rendered_values[row_index] = target_value
            if rendered and all(
                target_surface != observed_rows[row_index][2]
                for row_index, target_surface in rendered.items()
            ):
                selected_profile = candidate
                selected_empirical_surfaces = rendered
                selected_empirical_values = rendered_values
                break
        if empirical_kinds and selected_profile is None:
            raise ValueError(
                f"empirical support cannot replace every stale measurement for {target_number}"
            )
        for row_index, (
            operational_kind,
            line_number,
            source_surface,
            evidence,
            grammar,
        ) in enumerate(observed_rows):
            if operational_kind == "package_quantity":
                source_quantity = _parse_package_integer_surface(source_surface)
                try:
                    target_quantity = package_allocations[target_number]
                except KeyError as error:
                    raise ValueError(
                        f"target package allocations do not cover printed container {target_number}"
                    ) from error
                target_surface = _render_integer_surface(
                    target_quantity, source_surface=source_surface
                )
                source_canonical_value = Decimal(source_quantity)
                target_canonical_value = Decimal(target_quantity)
                maximum = None
                profile_document_id = None
                method: Literal[
                    "empirical_capacity_utilization_resample_v1",
                    "empirical_same_family_tare_resample_v1",
                    "target_measure_allocation_v1",
                    "target_package_allocation_v1",
                    "target_membership_package_projection_v1",
                    "target_single_container_aggregate_projection_v1",
                ] = (
                    "target_membership_package_projection_v1"
                    if target_number in membership_package_allocations
                    else (
                        "target_single_container_aggregate_projection_v1"
                        if target_number in unallocated_single_container
                        else "target_package_allocation_v1"
                    )
                )
            else:
                maximum = _operational_capacity_maximum(operational_kind, capacity)
                source_maximum = _operational_capacity_maximum(operational_kind, source_capacity)
                source_style = _parse_measurement_surface(source_surface, maximum=source_maximum)
                if source_style is None:
                    raise ValueError(f"cannot preserve source {operational_kind} numeric grammar")
                if target_has_measure[operational_kind]:
                    labeled_kind = cast(Literal["gross_weight_kg", "volume_m3"], operational_kind)
                    try:
                        target_value = allocated_values[labeled_kind][target_number]
                    except KeyError as error:
                        raise ValueError(
                            f"target {operational_kind} has no allocation for printed container "
                            f"{target_number}"
                        ) from error
                    target_surface = _render_measurement_surface(
                        target_value,
                        style=_measurement_style_with_exact_target_precision(
                            source_style, target_value
                        ),
                        maximum=maximum,
                    )
                    profile_document_id = None
                    method = "target_measure_allocation_v1"
                else:
                    target_surface = selected_empirical_surfaces[row_index]
                    target_value = selected_empirical_values[row_index]
                    profile_document_id = cast(
                        EmpiricalOperationalProfile, selected_profile
                    ).document_id
                    method = (
                        "empirical_same_family_tare_resample_v1"
                        if operational_kind == "tare_weight_kg"
                        else "empirical_capacity_utilization_resample_v1"
                    )
                source_canonical_value = source_canonical_values[(container_index, row_index)]
                target_parsed = _parse_measurement_surface(target_surface, maximum=maximum)
                if target_parsed is None:
                    raise ValueError(
                        f"cannot recover rendered target {operational_kind} numeric value"
                    )
                target_canonical_value = target_parsed.value
            source_line = raw_lines[line_number - 1]
            if grammar == "inline_container_breakdown":
                inline = _RAW_INLINE_CONTAINER_MEASURES.search(source_line)
                if inline is None:
                    raise ValueError("inline operational measurement lost its source grammar")
                measurement_start = inline.start()
                following_number = next(
                    (
                        target_containers[index].get("containerNumber")
                        for index, source_number in enumerate(source_numbers)
                        if re.search(
                            rf"(?<![A-Z0-9]){re.escape(source_number)}(?![A-Z0-9])",
                            source_line[inline.end() :],
                            re.I,
                        )
                    ),
                    None,
                )
                if following_number is not None and not isinstance(following_number, str):
                    raise ValueError("following synthetic container has no containerNumber")
            else:
                measurement_match = _operational_measurement_match(
                    source_line,
                    operational_kind,
                    expected_surface=source_surface,
                    source_grammar=grammar,
                )
                if measurement_match is None:
                    raise ValueError("labeled operational measurement lost its source grammar")
                measurement_start = measurement_match.start()
                following_number = None
            requirements.append(
                OperationalFlavorRequirement(
                    requirementId=f"operational-L{line_number:05d}-{operational_kind}",
                    kind=operational_kind,
                    sourceGrammar=grammar,
                    sourceLineId=f"L{line_number:05d}",
                    sourceMeasurementStartColumn=measurement_start,
                    sourceValueSurface=source_surface,
                    targetValueSurface=target_surface,
                    sourceCanonicalValue=str(source_canonical_value),
                    targetCanonicalValue=str(target_canonical_value),
                    consistencyGroupId=f"{target_number}-{operational_kind}",
                    targetContainerNumber=target_number,
                    targetEquipmentFamily=capacity.family,
                    maximumValue=str(maximum) if maximum is not None else None,
                    sameLineFollowingContainerNumber=following_number,
                    empiricalProfileDocumentId=profile_document_id,
                    samplingMethod=method,
                    sourceEvidence=evidence,
                )
            )
    return tuple(requirements)


def _label_measure_total(
    label: Mapping[str, Any],
    kind: Literal["gross_weight_kg", "volume_m3"],
) -> Decimal | None:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return None
    values = tuple(
        value
        for group in cast(Sequence[Any], patch.get("cargoGroups") or ())
        if isinstance(group, Mapping)
        and (value := _target_measure_value(group, kind)) is not None
    )
    return sum(values, start=Decimal(0)) if values else None


def _aggregate_empirical_measure_total(
    *,
    kind: Literal["gross_weight_kg", "tare_weight_kg", "volume_m3"],
    target_containers: Sequence[Mapping[str, Any]],
    requirements: Sequence[OperationalFlavorRequirement],
    source_document_id: str,
    scenario_id: str,
    profiles: Sequence[EmpiricalOperationalProfile],
    limits: TransportCapacityLimits,
) -> Decimal:
    """Sum target container values, reusing printed-row projections where available."""

    observed: dict[str, Decimal] = {}
    for requirement in requirements:
        if requirement.kind != kind:
            continue
        value = Decimal(requirement.targetCanonicalValue)
        previous = observed.setdefault(requirement.targetContainerNumber, value)
        if previous != value:
            raise ValueError(
                "one dense container measure has conflicting repeated projections: "
                f"{requirement.consistencyGroupId}"
            )
    target_numbers = {
        cast(str, container["containerNumber"])
        for container in target_containers
        if isinstance(container.get("containerNumber"), str)
    }
    unknown = sorted(set(observed) - target_numbers)
    if unknown:
        raise ValueError(f"dense aggregate requirements name unknown containers: {unknown}")

    values: list[Decimal] = []
    for container in target_containers:
        number = container.get("containerNumber")
        if not isinstance(number, str):
            raise ValueError("dense aggregate target container has no containerNumber")
        if number in observed:
            values.append(observed[number])
            continue
        capacity = equipment_capacity(container, limits)
        candidates = [
            profile
            for profile in profiles
            if profile.document_id != source_document_id
            and (
                profile.gross_utilization is not None
                if kind == "gross_weight_kg"
                else (
                    profile.tare_weight_kg is not None
                    if kind == "tare_weight_kg"
                    else profile.volume_utilization is not None or profile.volume_m3 is not None
                )
            )
        ]
        same_family = [
            profile for profile in candidates if profile.equipment_family == capacity.family
        ]
        if same_family:
            candidates = same_family
        elif kind == "tare_weight_kg":
            raise ValueError(
                "no same-equipment-family tare profile supports aggregate container " + number
            )
        if not candidates:
            raise ValueError(
                f"no empirical {kind} profile supports aggregate container {number}"
            )
        digest = int.from_bytes(
            bytes.fromhex(
                sha256_bytes(f"{scenario_id}\0{number}\0operational-profile-v1".encode())
            ),
            "big",
        )
        selected_value: Decimal | None = None
        for offset in range(len(candidates)):
            profile = candidates[(digest + offset) % len(candidates)]
            absolute, utilization = _profile_operational_value(profile, kind)
            maximum = _operational_capacity_maximum(kind, capacity)
            candidate_value = _resampled_empirical_value(
                absolute=absolute,
                utilization=utilization,
                maximum=maximum,
            )
            if (
                candidate_value is not None
                and candidate_value.is_finite()
                and candidate_value > 0
                and (maximum is None or candidate_value <= maximum)
            ):
                selected_value = candidate_value
                break
        if selected_value is None:
            raise ValueError(
                f"empirical support cannot produce a capacity-safe aggregate {kind} for {number}"
            )
        values.append(selected_value)
    if len(values) != len(target_containers) or not values:
        raise ValueError("dense aggregate has no complete target-container measurement coverage")
    return sum(values, start=Decimal(0))


def aggregate_operational_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    requirements: Sequence[OperationalFlavorRequirement],
    *,
    source_document_id: str,
    scenario_id: str,
    profiles: Sequence[EmpiricalOperationalProfile],
    limits: TransportCapacityLimits,
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Reconcile an explicit shipment-total tuple with target measures and container flavor.

    The carrier grammar fixes the three columns as gross, tare, and volume. Gross/volume values
    represented by the task label come directly from target truth; raw-only columns are rebuilt
    from capacity-safe empirical container profiles. Printed detail-row projections are reused,
    and unprinted containers receive deterministic profile values, so the total remains coherent
    even when a continuation page shows only a subset of its container rows.
    """

    aggregate_rows = _dense_aggregate_measurement_rows(raw_text)
    if not aggregate_rows:
        return ()
    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    if not isinstance(source_patch, Mapping) or not isinstance(target_patch, Mapping):
        raise ValueError("dense aggregate requires source and target documentPatch objects")
    source_containers = tuple(
        cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], source_patch.get("containers") or ())
        if isinstance(row, Mapping)
    )
    target_containers = tuple(
        cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], target_patch.get("containers") or ())
        if isinstance(row, Mapping)
    )
    printed_counts = {row.container_count for row in aggregate_rows}
    if printed_counts != {len(source_containers)} or len(source_containers) != len(
        target_containers
    ):
        raise ValueError(
            "dense aggregate container count disagrees with source/target topology: "
            f"printed={sorted(printed_counts)}, source={len(source_containers)}, "
            f"target={len(target_containers)}"
        )

    source_gross_total = _label_measure_total(source_label, "gross_weight_kg")
    if source_gross_total is None:
        raise ValueError("dense aggregate gross column has no source-label owner")
    gross_styles = tuple(
        _parse_measurement_surface(row.value_surfaces[0], maximum=source_gross_total)
        for row in aggregate_rows
    )
    if any(style is None or style.value != source_gross_total for style in gross_styles):
        raise ValueError(
            "one or more repeated dense aggregate gross columns disagree with the source-label "
            f"total: expected={source_gross_total}, "
            f"surfaces={[row.value_surfaces[0] for row in aggregate_rows]}"
        )
    parsed_gross_styles = cast(tuple[ParsedMeasurementSurface, ...], gross_styles)
    target_gross_total = _label_measure_total(target_label, "gross_weight_kg")
    if target_gross_total is None:
        raise ValueError("synthetic target removes gross weight from a fixed aggregate topology")
    allocated_gross = _target_measure_allocations(
        target_label,
        kind="gross_weight_kg",
        printed_precision=max(style.decimal_places for style in parsed_gross_styles),
        limits=limits,
    )
    if sum(allocated_gross.values(), start=Decimal(0)) != target_gross_total:
        raise ValueError("dense aggregate target gross allocations do not conserve the label total")

    tare_styles = tuple(
        _parse_measurement_surface(row.value_surfaces[1], maximum=None)
        for row in aggregate_rows
    )
    if any(style is None for style in tare_styles):
        raise ValueError("dense aggregate tare column is not a positive measurement")
    parsed_tare_styles = cast(tuple[ParsedMeasurementSurface, ...], tare_styles)
    source_tare_values = {style.value for style in parsed_tare_styles}
    if len(source_tare_values) != 1:
        raise ValueError("repeated dense aggregate tare columns disagree")
    source_tare_total = next(iter(source_tare_values))
    target_tare_total = _aggregate_empirical_measure_total(
        kind="tare_weight_kg",
        target_containers=target_containers,
        requirements=requirements,
        source_document_id=source_document_id,
        scenario_id=scenario_id,
        profiles=profiles,
        limits=limits,
    )
    if target_tare_total == source_tare_total:
        raise ValueError("empirical support cannot replace the stale aggregate tare total")

    source_volume_total = _label_measure_total(source_label, "volume_m3")
    source_volume_maximum = source_volume_total
    if source_volume_maximum is None:
        source_capacities = tuple(
            _source_equipment_capacity(
                container,
                local_evidence="\n".join(row.evidence for row in aggregate_rows),
                limits=limits,
            )
            for container in source_containers
        )
        if any(capacity.volume_m3 is None for capacity in source_capacities):
            raise ValueError(
                "raw-only dense aggregate volume has incomplete source equipment capacity"
            )
        source_volume_maximum = sum(
            (cast(Decimal, capacity.volume_m3) for capacity in source_capacities),
            start=Decimal(0),
        )
    volume_styles = tuple(
        _parse_measurement_surface(
            row.value_surfaces[2], maximum=source_volume_maximum
        )
        for row in aggregate_rows
    )
    if any(style is None for style in volume_styles):
        raise ValueError("dense aggregate volume column is not capacity-safe")
    parsed_volume_styles = cast(tuple[ParsedMeasurementSurface, ...], volume_styles)
    source_volume_values = {style.value for style in parsed_volume_styles}
    if len(source_volume_values) != 1:
        raise ValueError("repeated dense aggregate volume columns disagree")
    source_printed_volume_total = next(iter(source_volume_values))
    if (
        source_volume_total is not None
        and source_printed_volume_total != source_volume_total
    ):
        raise ValueError(
            "dense aggregate volume column disagrees with source-label total: "
            f"expected={source_volume_total}, "
            f"surfaces={[row.value_surfaces[2] for row in aggregate_rows]}"
        )
    target_volume_total = _label_measure_total(target_label, "volume_m3")
    if target_volume_total is not None:
        allocated_volume = _target_measure_allocations(
            target_label,
            kind="volume_m3",
            printed_precision=max(style.decimal_places for style in parsed_volume_styles),
            limits=limits,
        )
        if sum(allocated_volume.values(), start=Decimal(0)) != target_volume_total:
            raise ValueError(
                "dense aggregate target volume allocations do not conserve the label total"
            )
    elif source_volume_total is not None:
        raise ValueError("synthetic target removes volume from a fixed aggregate topology")
    else:
        target_volume_total = _aggregate_empirical_measure_total(
            kind="volume_m3",
            target_containers=target_containers,
            requirements=requirements,
            source_document_id=source_document_id,
            scenario_id=scenario_id,
            profiles=profiles,
            limits=limits,
        )
        if target_volume_total == source_printed_volume_total:
            raise ValueError("empirical support cannot replace the stale aggregate volume total")

    assert target_volume_total is not None
    target_values = (target_gross_total, target_tare_total, target_volume_total)
    kinds = ("gross_weight_kg", "tare_weight_kg", "volume_m3")
    values_by_line: dict[
        int, list[tuple[int, tuple[int, int], Decimal, ParsedMeasurementSurface, str]]
    ] = defaultdict(list)
    for aggregate, styles in zip(
        aggregate_rows,
        zip(parsed_gross_styles, parsed_tare_styles, parsed_volume_styles, strict=True),
        strict=True,
    ):
        for ordinal, (line_index, span, value, style, kind) in enumerate(
            zip(
                aggregate.value_line_indexes,
                aggregate.value_spans,
                target_values,
                styles,
                kinds,
                strict=True,
            )
        ):
            values_by_line[line_index].append((ordinal, span, value, style, kind))

    lines = raw_text.splitlines()
    output: list[AnchoredScalarReplacementRequirement] = []
    for line_index, rows in sorted(values_by_line.items()):
        line = lines[line_index]
        start = min(row[1][0] for row in rows)
        end = max(row[1][1] for row in rows)
        source_surface = line[start:end]
        target_surface = source_surface
        for _ordinal, span, value, style, _kind in sorted(
            rows, key=lambda row: row[1], reverse=True
        ):
            relative_start = span[0] - start
            relative_end = span[1] - start
            rendered = _render_measurement_surface(
                value,
                style=_measurement_style_with_exact_target_precision(style, value),
                maximum=None,
            )
            target_surface = (
                target_surface[:relative_start] + rendered + target_surface[relative_end:]
            )
        if target_surface == source_surface:
            continue
        output.append(
            AnchoredScalarReplacementRequirement(
                targetPaths=tuple(
                    f"auxiliary.aggregate.dense_container_table.{row[4]}" for row in rows
                ),
                sourceLineIds=(_line_id(line_index + 1, line),),
                sourceSurface=source_surface,
                targetSurface=target_surface,
            )
        )
    return tuple(output)


def _operational_flavor_requirements_rendered(
    value: str, requirements: Sequence[OperationalFlavorRequirement]
) -> bool:
    lines = value.splitlines()
    for requirement in requirements:
        line_number = _line_number(requirement.sourceLineId)
        if not 1 <= line_number <= len(lines):
            return False
        line = lines[line_number - 1]
        if requirement.sourceGrammar == "inline_container_breakdown":
            match = _RAW_INLINE_CONTAINER_MEASURES.search(line)
            group = {
                "gross_weight_kg": "gross",
                "volume_m3": "volume",
                "package_quantity": "packages",
            }[requirement.kind]
        else:
            match = _operational_measurement_match(
                line,
                requirement.kind,
                expected_surface=requirement.targetValueSurface,
                source_grammar=requirement.sourceGrammar,
            )
            group = "value"
        # ``sourceMeasurementStartColumn`` proves the source-side compiler anchor. It is not an
        # output-format invariant: a legitimate equipment or seal replacement earlier on the
        # same line can change its width and therefore shift every following column. Semantic
        # ownership is instead preserved by the immutable physical line plus the named grammar
        # group (and, for shifted inline rows, the following container identity).
        if match is None or match.group(group) != requirement.targetValueSurface:
            if requirement.kind != "package_quantity":
                return False
            # The package-category renderer may replace the source noun with a longer semantic
            # surface (for example ``CARTONS`` -> ``INTERMEDIATE BULK CONTAINERS``). The numeric
            # operational prefill is already anchored to this immutable physical line, and every
            # model-writable version of that line carries an exact host lock for its target
            # literal. Validate the standalone target number without coupling arithmetic truth
            # to a finite package-noun vocabulary.
            target_match = re.search(
                rf"(?<![0-9.,]){re.escape(requirement.targetValueSurface)}(?![0-9.,])",
                line,
            )
            if target_match is None:
                return False
            match = target_match
        following = requirement.sameLineFollowingContainerNumber
        if following is not None:
            following_match = re.search(
                rf"(?<![A-Z0-9]){re.escape(following)}(?![A-Z0-9])", line, re.I
            )
            if following_match is None or following_match.start() < match.end():
                return False
    return True


def _derive_raw_auxiliary_identity_realizations(
    workspace: RewriteWorkspace,
    updated: str,
) -> tuple[AppliedRawAuxiliaryIdentityRealization, ...]:
    """Derive raw-agent audit receipts from the rewritten lines, never duplicate model output."""

    blocks_by_line: dict[int, list[tuple[str, int, str]]] = {}
    for line_number, identity, gap, principal, _ in _raw_agent_blocks(updated):
        blocks_by_line.setdefault(line_number, []).append((identity, gap, principal))
    applied: list[AppliedRawAuxiliaryIdentityRealization] = []
    rendered_by_group: dict[str, str] = {}
    for requirement in workspace.raw_auxiliary_identity_requirements:
        requirement_id = requirement.requirementId
        source_line = int(requirement_id.rsplit("-L", 1)[1])
        candidates = blocks_by_line.get(source_line, [])
        if len(candidates) != 1:
            raise ValueError(
                "raw auxiliary identity relationship is missing or ambiguous at its source line: "
                f"{requirement_id}"
            )
        rendered, observed_gap, principal = candidates[0]
        normalized_rendered = _semantic_normalize(rendered)
        rendered_lines = rendered.splitlines()
        if len(rendered_lines) != requirement.sourceIdentityLineCount or any(
            not line.strip() for line in rendered_lines
        ):
            raise ValueError(
                f"raw auxiliary identity must preserve its populated line count: {requirement_id}"
            )
        if observed_gap != requirement.gapLineCount:
            raise ValueError(
                f"raw auxiliary identity must preserve its gap line count: {requirement_id}"
            )
        if not _principal_surface_realizes_target(principal, requirement.targetPrincipalName):
            raise ValueError(
                "raw auxiliary identity must retain its exact target carrier principal: "
                f"{requirement_id}"
            )
        if len(normalized_rendered.split()) < 2:
            raise ValueError(
                f"raw auxiliary identity must contain at least two lexical tokens: {requirement_id}"
            )
        if normalized_rendered in {
            _semantic_normalize(requirement.sourceIdentity),
            _semantic_normalize(requirement.targetPrincipalName),
        }:
            raise ValueError(
                "raw auxiliary identity must be distinct from source and principal: "
                f"{requirement_id}"
            )
        rendered_key = _auxiliary_identity_token_key(rendered)
        previous = rendered_by_group.setdefault(requirement.consistencyGroupId, rendered_key)
        if previous != rendered_key:
            raise ValueError(
                "repeated raw auxiliary identity has inconsistent fictional realizations: "
                f"{requirement.consistencyGroupId}"
            )
        applied.append(
            AppliedRawAuxiliaryIdentityRealization(
                requirementId=requirement_id,
                relationship=requirement.relationship,
                sourceIdentity=requirement.sourceIdentity,
                sourceIdentityLineCount=requirement.sourceIdentityLineCount,
                gapLineCount=requirement.gapLineCount,
                consistencyGroupId=requirement.consistencyGroupId,
                targetPrincipalName=requirement.targetPrincipalName,
                renderedIdentity=rendered,
            )
        )
    return tuple(applied)


def _raw_auxiliary_identity_realization_rendered(
    value: str, realization: AppliedRawAuxiliaryIdentityRealization
) -> bool:
    """Require the fictional agent and target principal in the original agency grammar."""

    return any(
        _semantic_normalize(identity) == _semantic_normalize(realization.renderedIdentity)
        and gap == realization.gapLineCount
        and _principal_surface_realizes_target(principal, realization.targetPrincipalName)
        for _, identity, gap, principal, _ in _raw_agent_blocks(value)
    )


def _principal_surface_realizes_target(surface: str, target_name: str) -> bool:
    normalized_surface = _semantic_normalize(surface)
    normalized_target = _semantic_normalize(target_name)
    if normalized_surface == normalized_target:
        return True
    return any(
        normalized_surface.startswith(f"{normalized_target} {marker}")
        for marker in ("TRADING AS", "DOING BUSINESS AS", "T A", "D B A")
    )


def _raw_auxiliary_identity_realizations_rendered(
    value: str, realizations: Sequence[AppliedRawAuxiliaryIdentityRealization]
) -> bool:
    required: dict[tuple[str, int, str], int] = {}
    for realization in realizations:
        key = (
            _semantic_normalize(realization.renderedIdentity),
            realization.gapLineCount,
            _semantic_normalize(realization.targetPrincipalName),
        )
        required[key] = required.get(key, 0) + 1
    blocks = _raw_agent_blocks(value)
    return all(
        sum(
            _semantic_normalize(identity) == rendered
            and observed_gap == gap
            and _principal_surface_realizes_target(principal, target)
            for _, identity, observed_gap, principal, _ in blocks
        )
        == count
        for (rendered, gap, target), count in required.items()
    )


def apply_line_range_replacements(
    workspace: RewriteWorkspace,
    edits: Sequence[LineRangeReplacement],
    compound_party_flavor_realizations: Sequence[CompoundPartyFlavorRealization] = (),
) -> AtomicRewriteCommit:
    """Validate one complete line-range patch and commit it atomically."""

    if not edits:
        raise ValueError("at least one line-range replacement is required")
    applied_realizations = _validate_compound_party_flavor_realizations(
        workspace, compound_party_flavor_realizations
    )
    snapshot = workspace.current_text
    lines = snapshot.splitlines(keepends=True)
    if not lines:
        raise ValueError("current OCR has no addressable lines")
    default_newline = _default_newline(snapshot)
    ordered = sorted(
        edits,
        key=lambda row: (_line_number(row.startLineId), _line_number(row.endLineId)),
    )
    previous_end = 0
    applied: list[AppliedLineRangeReplacement] = []
    chunks: list[str] = []
    cursor = 0
    for index, edit in enumerate(ordered):
        start_line = _line_number(edit.startLineId)
        end_line = _line_number(edit.endLineId)
        if end_line < start_line:
            raise ValueError(f"edit[{index}] end line precedes its start line")
        if start_line <= previous_end:
            raise ValueError(f"edit[{index}] overlaps or duplicates a preceding line range")
        if end_line > len(lines):
            raise ValueError(
                f"edit[{index}] ends at line {end_line}, but current OCR has {len(lines)} lines"
            )
        start_index = start_line - 1
        end_index = end_line
        old_lines = lines[start_index:end_index]
        if any(_PAGE_MARKER.fullmatch(_line_body(line)) for line in old_lines):
            raise ValueError(f"edit[{index}] includes a protected page marker")
        new_text = edit.newText
        if new_text.endswith("\r\n"):
            new_text = new_text[:-2]
        elif new_text.endswith(("\r", "\n")):
            new_text = new_text[:-1]
        normalized_new = new_text.replace("\r\n", "\n").replace("\r", "\n")
        if any(_PAGE_MARKER.fullmatch(line) for line in normalized_new.split("\n")):
            raise ValueError(f"edit[{index}] introduces a protected page marker")
        old_bodies = tuple(_line_body(line) for line in old_lines)
        new_lines = tuple(normalized_new.split("\n")) if normalized_new else ()
        if len(new_lines) != len(old_bodies):
            raise ValueError(
                f"edit[{index}] changes source line count: selected={len(old_bodies)}, "
                f"output={len(new_lines)}; preserve one output line per source line"
            )
        old_blank = tuple(not line.strip() for line in old_bodies)
        new_blank = tuple(not line.strip() for line in new_lines)
        if new_blank != old_blank:
            raise ValueError(
                f"edit[{index}] changes occupied/blank line topology; preserve every source "
                "slot and synthesize coherent content in populated target-absent slots"
            )
        for relative_line, (old_body, new_line) in enumerate(
            zip(old_bodies, new_lines, strict=True)
        ):
            if any(character.isalnum() for character in old_body) and not any(
                character.isalnum() for character in new_line
            ):
                raise ValueError(
                    f"edit[{index}] line {start_line + relative_line} degenerates lexical "
                    "content to punctuation-only filler"
                )
        ending = _line_ending(old_lines[-1])
        replacement = normalized_new.replace("\n", default_newline)
        if replacement and ending:
            replacement += ending
        old_text = "".join(old_lines)
        if _placeholder_count(normalized_new) > _placeholder_count(old_text):
            raise ValueError(f"edit[{index}] introduces an unavailable/unknown placeholder")
        if replacement == old_text:
            previous_end = end_line
            continue
        chunks.append("".join(lines[cursor:start_index]))
        chunks.append(replacement)
        applied.append(
            AppliedLineRangeReplacement(
                startLine=start_line,
                endLine=end_line,
                startLineId=edit.startLineId,
                endLineId=edit.endLineId,
                oldText=old_text,
                newText=replacement,
                oldTextSha256=sha256_bytes(old_text.encode("utf-8")),
                sourceLinesReplaced=end_line - start_line + 1,
                outputLinesInserted=(len(normalized_new.split("\n")) if normalized_new else 0),
            )
        )
        cursor = end_index
        previous_end = end_line
    chunks.append("".join(lines[cursor:]))
    updated = "".join(chunks)
    if updated == snapshot:
        raise ValueError("the atomic line-range patch did not change the OCR")
    if _page_markers(updated) != _page_markers(workspace.original_text):
        raise ValueError("the line-range patch changes page markers or page order")
    if not _newline_convention(updated) <= _newline_convention(workspace.original_text):
        raise ValueError("the line-range patch introduces a different newline convention")
    if len(updated.splitlines()) != len(workspace.original_text.splitlines()):
        raise ValueError("the line-range patch changes the source line count")
    if _blank_line_topology(updated) != _blank_line_topology(workspace.original_text):
        raise ValueError("the line-range patch changes the source blank-line topology")
    inline_slot_mismatches = _inline_slot_topology_mismatches(workspace.original_text, updated)
    if inline_slot_mismatches:
        raise ValueError(
            "the line-range patch changes an inline labeled slot or its populated/empty state: "
            f"{list(inline_slot_mismatches)}"
        )
    structural_prefix_mismatches = _structural_line_prefix_mismatches(
        workspace.original_text, updated
    )
    if structural_prefix_mismatches:
        raise ValueError(
            "the line-range patch changes or removes a numbered structural line prefix: "
            f"{list(structural_prefix_mismatches)}"
        )
    isolated_formatting = _isolated_formatting_changes(workspace, updated)
    if isolated_formatting:
        raise ValueError(
            "the line-range patch changes punctuation or whitespace outside an actual value "
            f"change: {isolated_formatting[:6]}"
        )
    if not _source_semantic_role_surfaces_preserved(updated, workspace.source_role_hints):
        missing = [
            row.model_dump(mode="json")
            for row in workspace.source_role_hints
            if not _source_semantic_role_surfaces_preserved(updated, (row,))
        ]
        raise ValueError(
            f"the line-range patch changes a deterministic flattened-OCR role surface: {missing}"
        )
    if not _source_status_surfaces_preserved(updated, workspace.source_status_requirements):
        missing = [
            row.model_dump(mode="json")
            for row in workspace.source_status_requirements
            if updated.splitlines().count(row.sourceSurface) != row.sourceOccurrences
        ]
        raise ValueError(
            f"the line-range patch changes an unchanged source status surface: {missing}"
        )
    if _max_consecutive_blank_lines(updated) > (
        _max_consecutive_blank_lines(workspace.original_text) + 1
    ):
        lines = updated.splitlines()
        blank_index = next(
            (
                index
                for index in range(1, len(lines))
                if not lines[index].strip() and not lines[index - 1].strip()
            ),
            0,
        )
        context = "\n".join(lines[max(0, blank_index - 2) : blank_index + 3])
        raise ValueError(
            "the line-range patch introduces more than one additional blank line; "
            f"include the surrounding lines in one replacement; first gap context={context!r}"
        )
    if _placeholder_count(updated) > _placeholder_count(snapshot):
        raise ValueError("the line-range patch introduces an unavailable/unknown placeholder")
    introduced_entities = _introduced_html_entities(workspace.original_text, updated)
    if introduced_entities:
        raise ValueError(
            "the line-range patch introduces HTML/XML entities into plain OCR text: "
            f"{list(introduced_entities[:6])}"
        )
    duplicated_lines = _introduced_source_line_duplicates(workspace.original_text, updated)
    if duplicated_lines:
        raise ValueError(
            f"the line-range patch duplicates unchanged source lines: {list(duplicated_lines[:6])}"
        )
    if not _required_surfaces_rendered(updated, workspace.surface_requirements):
        missing = [
            row.model_dump(mode="json")
            for row in workspace.surface_requirements
            if not _required_surfaces_rendered(updated, (row,))
        ]
        raise ValueError(
            f"the line-range patch violates deterministic date/HS rendering requirements: {missing}"
        )
    if not _jurisdictional_surfaces_rendered(updated, workspace.jurisdictional_requirements):
        missing = [
            row.model_dump(mode="json")
            for row in workspace.jurisdictional_requirements
            if _literal_phrase_pattern(row.sourceSurface).search(updated) is not None
            or _literal_phrase_pattern(row.targetSurface).search(updated) is None
        ]
        raise ValueError(
            "the line-range patch retains a stale named customs program or omits its coherent "
            f"generic role: {missing}"
        )
    missing_literals = _missing_target_literals(updated, workspace.target_literal_requirements)
    if missing_literals:
        raise ValueError(
            "the line-range patch omits changed target text: "
            f"{[row.model_dump(mode='json') for row in missing_literals]}"
        )
    if not _target_value_occurrences_exact(updated, workspace.target_value_occurrence_requirements):
        incorrect = [
            row.model_dump(mode="json")
            for row in workspace.target_value_occurrence_requirements
            if _target_value_occurrence_count(updated, row) != row.requiredOccurrences
        ]
        raise ValueError(
            f"the line-range patch duplicates or omits role-bound party values: {incorrect}"
        )
    if not _anchored_scalar_replacements_rendered(
        updated, workspace.anchored_scalar_replacement_requirements
    ):
        incorrect = [
            row.model_dump(mode="json")
            for row in workspace.anchored_scalar_replacement_requirements
            if not _anchored_scalar_replacements_rendered(updated, (row,))
        ]
        raise ValueError(
            f"the line-range patch moves or omits an exact changed label scalar: {incorrect}"
        )
    normalized_updated = _semantic_normalize(updated)
    applied_raw_identities = _derive_raw_auxiliary_identity_realizations(workspace, updated)
    for compound_realization in applied_realizations:
        if _semantic_normalize(compound_realization.renderedName) not in normalized_updated:
            raise ValueError(
                "compound-party flavor realization is not rendered in rewritten OCR: "
                f"{compound_realization.targetPath}"
            )
    for raw_realization in applied_raw_identities:
        if _semantic_normalize(raw_realization.sourceIdentity) in normalized_updated:
            raise ValueError(
                "raw auxiliary source identity remains after rewrite: "
                f"{raw_realization.requirementId}"
            )
        if _semantic_normalize(raw_realization.renderedIdentity) not in normalized_updated:
            raise ValueError(
                "raw auxiliary identity realization is not rendered in rewritten OCR: "
                f"{raw_realization.requirementId}"
            )
        if not _raw_auxiliary_identity_realization_rendered(updated, raw_realization):
            raise ValueError(
                "raw auxiliary identity does not preserve the agent-for-carrier relationship: "
                f"{raw_realization.requirementId}"
            )
    if not _raw_auxiliary_identity_realizations_rendered(updated, applied_raw_identities):
        raise ValueError(
            "raw auxiliary identity realization counts do not match the source agency topology"
        )
    if not _operational_flavor_requirements_rendered(
        updated, workspace.operational_flavor_requirements
    ):
        incorrect = [
            row.model_dump(mode="json")
            for row in workspace.operational_flavor_requirements
            if not _operational_flavor_requirements_rendered(updated, (row,))
        ]
        raise ValueError(
            "the line-range patch retains or misformats a raw-only operational measurement: "
            f"{incorrect}"
        )
    if not _deactivated_reefer_operation_coherent(workspace, updated):
        raise ValueError(
            "the line-range patch retains a positive temperature or plugging assertion after "
            "the target deactivates every reefer setpoint; rewrite each occupied source slot "
            "as coherent non-operating/ventilation flavor"
        )
    cargo_failures = _cargo_flavor_rewrite_failures(
        updated, workspace.cargo_flavor_rewrite_requirements
    )
    if cargo_failures:
        raise ValueError(
            "the line-range patch leaves a source cargo-description line unchanged or omits the "
            f"target description: {json.dumps(cargo_failures, sort_keys=True)}"
        )
    commit = AtomicRewriteCommit(
        beforeTextSha256=workspace.current_sha256,
        afterTextSha256=sha256_bytes(updated.encode("utf-8")),
        beforeTargetLabelSha256=workspace.current_target_sha256,
        afterTargetLabelSha256=workspace.current_target_sha256,
        compoundPartyFlavorRealizations=applied_realizations,
        rawAuxiliaryIdentityRealizations=applied_raw_identities,
        appliedReplacements=tuple(applied),
    )
    workspace.current_text = updated
    workspace.commits.append(commit)
    return commit


EditBatch = Annotated[list[LineRangeReplacement], Field(min_length=1, max_length=256)]


class AtomicRewriteArguments(BaseModel):
    """Audit mirror of the strict terminal output-function arguments."""

    model_config = _STRICT

    edits: EditBatch
    compoundPartyFlavorRealizations: Annotated[
        list[CompoundPartyFlavorRealization],
        Field(
            max_length=6,
            description=(
                "Exactly the auxiliary raw-text realizations listed in "
                "compoundPartyFlavorRequirements; empty when none are required."
            ),
        ),
    ]


def commit_atomic_rewrite(
    ctx: RunContext[RewriteWorkspace],
    edits: EditBatch,
    compoundPartyFlavorRealizations: Annotated[
        list[CompoundPartyFlavorRealization],
        Field(
            max_length=6,
            description=(
                "Exactly the auxiliary raw-text realizations listed in "
                "compoundPartyFlavorRequirements; empty when none are required."
            ),
        ),
    ],
) -> AtomicRewriteCommit:
    """Atomically apply the complete line-range patch and finish this editor pass.

    Args:
        ctx: Isolated mutable OCR workspace for this document.
        edits: Complete non-overlapping line-range patch against currentRawOcrLines.
    """

    try:
        return apply_line_range_replacements(
            ctx.deps,
            edits,
            compound_party_flavor_realizations=compoundPartyFlavorRealizations,
        )
    except ValueError as error:
        raise ModelRetry(
            "Atomic rewrite rejected and fully rolled back; zero edits were applied. "
            "Resubmit the complete patch against the unchanged current OCR, including every "
            f"previously attempted edit plus the correction. Cause: {error}"
        ) from error


def _terminal_editor_output(
    workspace: RewriteWorkspace,
    *,
    max_retries: int,
) -> ToolOutput[AtomicRewriteCommit]:
    """Build a strict output tool whose line IDs are limited to editable source lines.

    Blank lines and page markers are deliberately omitted from the editor payload. Encoding the
    same finite set as JSON Schema enums makes a model-selected omitted line impossible under
    provider strict decoding, instead of paying for a post-generation retry that can only fail.
    The complete patch still passes through the transactional semantic validators.
    """

    allowed_line_ids = tuple(
        _line_id(index, line)
        for index, line in enumerate(workspace.current_text.splitlines(keepends=True), start=1)
        if (body := _line_body(line)).strip() and _PAGE_MARKER.fullmatch(body) is None
    )
    if not allowed_line_ids:
        raise ValueError("current OCR has no editable populated lines")
    allowed_line_id_type = cast(Any, Literal).__getitem__(allowed_line_ids)
    model_suffix = sha256_bytes("\n".join(allowed_line_ids).encode("utf-8"))[:12]
    allowed_line_replacement = create_model(
        f"AllowedLineRangeReplacement_{model_suffix}",
        __config__=_STRICT,
        startLineId=(
            allowed_line_id_type,
            Field(description="First populated editable line ID in this replacement."),
        ),
        endLineId=(
            allowed_line_id_type,
            Field(description="Last populated editable line ID in this replacement."),
        ),
        newText=(
            NonEmptyText,
            Field(description="Replacement text with exactly one output line per selected line."),
        ),
    )
    dynamic_edit_batch = cast(Any, Annotated).__class_getitem__(
        (
            list.__class_getitem__(allowed_line_replacement),
            Field(min_length=1, max_length=256),
        )
    )
    compound_batch = Annotated[
        list[CompoundPartyFlavorRealization],
        Field(
            max_length=6,
            description=(
                "Exactly the auxiliary raw-text realizations listed in "
                "compoundPartyFlavorRequirements; empty when none are required."
            ),
        ),
    ]

    def commit_for_allowed_lines(
        ctx: RunContext[RewriteWorkspace],
        edits: list[BaseModel],
        compoundPartyFlavorRealizations: list[CompoundPartyFlavorRealization],
    ) -> AtomicRewriteCommit:
        parsed_edits = [
            LineRangeReplacement.model_validate(row.model_dump(mode="python")) for row in edits
        ]
        return commit_atomic_rewrite(
            ctx,
            parsed_edits,
            compoundPartyFlavorRealizations,
        )

    commit_for_allowed_lines.__name__ = "commit_atomic_rewrite"
    commit_for_allowed_lines.__doc__ = commit_atomic_rewrite.__doc__
    commit_for_allowed_lines.__annotations__ = {
        "ctx": RunContext[RewriteWorkspace],
        "edits": dynamic_edit_batch,
        "compoundPartyFlavorRealizations": compound_batch,
        "return": AtomicRewriteCommit,
    }
    return ToolOutput(
        commit_for_allowed_lines,
        name="commit_atomic_rewrite",
        description=(
            "Submit the complete line-addressed patch once. It is validated and applied "
            "atomically; a successful call ends the editor pass."
        ),
        max_retries=max_retries,
        strict=True,
        sequential=True,
    )


def _residual_candidates(
    workspace: RewriteWorkspace, leaves: Sequence[ChangedLeaf]
) -> tuple[ResidualCandidate, ...]:
    candidates: list[ResidualCandidate] = []
    for leaf in leaves:
        if not leaf.requiresTextEdit or leaf.sourceValue == leaf.targetValue:
            continue
        if not isinstance(leaf.sourceValue, (str, int, float)) or isinstance(
            leaf.sourceValue, bool
        ):
            continue
        if _source_equipment_surface_matches_target(workspace, leaf):
            continue
        surface = str(leaf.sourceValue)
        if len(surface.strip()) < 4:
            continue
        offsets = _all_occurrences(workspace.current_text, surface)
        if not offsets or not _all_occurrences(workspace.original_text, surface):
            continue
        offset = offsets[0]
        start = max(0, offset - 120)
        end = min(len(workspace.current_text), offset + len(surface) + 120)
        candidates.append(
            ResidualCandidate(
                targetPath=leaf.path,
                sourceValue=leaf.sourceValue,
                currentOccurrences=len(offsets),
                currentEvidence=workspace.current_text[start:end],
            )
        )
    return tuple(candidates)


def _source_equipment_surface_matches_target(
    workspace: RewriteWorkspace, leaf: ChangedLeaf
) -> bool:
    """Recognize a v3 printed equipment surface that already realizes v5 semantics."""

    match = _CONTAINER_TYPE_DESCRIPTION_PATH.fullmatch(leaf.path)
    if match is None or not isinstance(leaf.sourceValue, str):
        return False
    patch = workspace.current_target_label.get("documentPatch")
    if not isinstance(patch, dict):
        return False
    containers = patch.get("containers")
    index = int(match.group(1))
    if not isinstance(containers, list) or index >= len(containers):
        return False
    target = containers[index]
    if not isinstance(target, dict):
        return False
    size = target.get("sizeCategory")
    type_category = target.get("typeCategory")
    if not isinstance(size, str) or not isinstance(type_category, str):
        return False
    reviewed = review_source_equipment_surface(
        leaf.sourceValue,
        temperature_present=target.get("temperatureSetpoint") is not None,
    )
    return (
        reviewed.resolution == "reviewed_source_grammar"
        and reviewed.size_category == size
        and reviewed.type_category == type_category
    )


def _target_equipment_semantics(
    workspace: RewriteWorkspace, index: int
) -> dict[str, JsonValue] | None:
    patch = workspace.current_target_label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, dict) else None
    if not isinstance(containers, list) or index >= len(containers):
        return None
    target = containers[index]
    if not isinstance(target, dict):
        return None
    size = target.get("sizeCategory")
    type_category = target.get("typeCategory")
    if not isinstance(size, str) or not isinstance(type_category, str):
        return None
    return {
        "sizeCategory": size,
        "typeCategory": type_category,
    }


def label_change_contract(workspace: RewriteWorkspace) -> tuple[LabelChangeDirective, ...]:
    """Project raw schema deltas into concise, printable semantic directives.

    Source labels use a reviewed free-text container ``typeDescription`` while target labels use
    semantic size/type categories.  Treating those three leaves independently caused models to
    print enum names or delete already-correct surfaces.  This projection makes that cross-schema
    relationship explicit while leaving every other reviewed-label change exact and auditable.
    """

    leaves = changed_leaves(workspace.source_label, workspace.current_target_label)
    suppressed_paths: set[str] = set()
    projected_by_path: dict[str, LabelChangeDirective] = {}
    for leaf in leaves:
        match = _CONTAINER_TYPE_DESCRIPTION_PATH.fullmatch(leaf.path)
        if match is None or not isinstance(leaf.sourceValue, str):
            continue
        index = int(match.group(1))
        target_semantics = _target_equipment_semantics(workspace, index)
        if target_semantics is None:
            continue
        projected_path = f"documentPatch.containers[{index}].printedEquipmentSurface"
        projected_by_path[leaf.path] = LabelChangeDirective(
            path=projected_path,
            action=(
                "preserve_equivalent_equipment_surface"
                if _source_equipment_surface_matches_target(workspace, leaf)
                else "replace_equipment_surface"
            ),
            sourceValue=leaf.sourceValue,
            targetValue=target_semantics,
        )
        suppressed_paths.update(
            {
                leaf.path,
                f"documentPatch.containers[{index}].sizeCategory",
                f"documentPatch.containers[{index}].typeCategory",
            }
        )

    projected_indices = {
        int(match.group(1))
        for leaf in leaves
        if (match := _CONTAINER_TYPE_DESCRIPTION_PATH.fullmatch(leaf.path)) is not None
        and leaf.path in suppressed_paths
    }
    target_category_triggers: dict[int, str] = {}
    for leaf in leaves:
        match = _CONTAINER_SIZE_CATEGORY_PATH.fullmatch(leaf.path)
        if match is None:
            match = _CONTAINER_TYPE_CATEGORY_PATH.fullmatch(leaf.path)
        if match is None:
            continue
        index = int(match.group(1))
        if index in projected_indices or index in target_category_triggers:
            continue
        target_semantics = _target_equipment_semantics(workspace, index)
        if target_semantics is None:
            continue
        projected_path = f"documentPatch.containers[{index}].printedEquipmentSurface"
        projected_by_path[leaf.path] = LabelChangeDirective(
            path=projected_path,
            action="add_equipment_surface",
            sourceValue=None,
            targetValue=target_semantics,
        )
        target_category_triggers[index] = leaf.path
        suppressed_paths.update(
            {
                f"documentPatch.containers[{index}].sizeCategory",
                f"documentPatch.containers[{index}].typeCategory",
            }
        )

    directives: list[LabelChangeDirective] = []
    emitted_projections: set[str] = set()
    for leaf in leaves:
        projection = projected_by_path.get(leaf.path)
        if projection is not None and projection.path not in emitted_projections:
            directives.append(projection)
            emitted_projections.add(projection.path)
            continue
        if leaf.path in suppressed_paths or not leaf.requiresTextEdit:
            continue
        if leaf.sourcePresent and leaf.targetPresent:
            action: Literal["replace", "deactivate_extractable_fact_preserve_slot", "add"] = (
                "replace"
            )
        elif leaf.sourcePresent:
            action = "deactivate_extractable_fact_preserve_slot"
        else:
            action = "add"
        directives.append(
            LabelChangeDirective(
                path=leaf.path,
                action=action,
                sourceValue=leaf.sourceValue,
                targetValue=leaf.targetValue,
            )
        )
    return tuple(directives)


def deterministic_rewrite_audit(
    workspace: RewriteWorkspace, leaves: Sequence[ChangedLeaf]
) -> DeterministicRewriteAudit:
    normalized_current = _semantic_normalize(workspace.current_text)
    applied_raw_identities = {
        row.requirementId: row
        for commit in workspace.commits
        for row in commit.rawAuxiliaryIdentityRealizations
    }
    return DeterministicRewriteAudit(
        currentTextSha256=workspace.current_sha256,
        textChanged=workspace.current_text != workspace.original_text,
        pageMarkersPreserved=_page_markers(workspace.current_text)
        == _page_markers(workspace.original_text),
        newlineConventionPreserved=_newline_convention(workspace.current_text)
        <= _newline_convention(workspace.original_text),
        lineCountPreserved=(
            len(workspace.current_text.splitlines()) == len(workspace.original_text.splitlines())
        ),
        blankLineTopologyPreserved=(
            _blank_line_topology(workspace.current_text)
            == _blank_line_topology(workspace.original_text)
        ),
        inlineSlotTopologyPreserved=_inline_slot_topology_preserved(
            workspace.original_text, workspace.current_text
        ),
        structuralLinePrefixesPreserved=_structural_line_prefixes_preserved(
            workspace.original_text, workspace.current_text
        ),
        isolatedFormattingChangesAbsent=not _isolated_formatting_changes(
            workspace, workspace.current_text
        ),
        sourceStatusSurfacesPreserved=_source_status_surfaces_preserved(
            workspace.current_text, workspace.source_status_requirements
        ),
        blankLineDepthBounded=_max_consecutive_blank_lines(workspace.current_text)
        <= _max_consecutive_blank_lines(workspace.original_text) + 1,
        noPlaceholderIntroduced=_placeholder_count(workspace.current_text)
        <= _placeholder_count(workspace.original_text),
        requiredSurfacesRendered=_required_surfaces_rendered(
            workspace.current_text, workspace.surface_requirements
        ),
        jurisdictionalSurfacesRendered=_jurisdictional_surfaces_rendered(
            workspace.current_text, workspace.jurisdictional_requirements
        ),
        requiredTargetLiteralsRendered=not _missing_target_literals(
            workspace.current_text, workspace.target_literal_requirements
        ),
        targetValueOccurrencesExact=_target_value_occurrences_exact(
            workspace.current_text, workspace.target_value_occurrence_requirements
        ),
        anchoredScalarReplacementsRendered=_anchored_scalar_replacements_rendered(
            workspace.current_text,
            workspace.anchored_scalar_replacement_requirements,
        ),
        rawAuxiliaryIdentitiesReplaced=all(
            _semantic_normalize(requirement.sourceIdentity) not in normalized_current
            and requirement.requirementId in applied_raw_identities
            for requirement in workspace.raw_auxiliary_identity_requirements
        )
        and _raw_auxiliary_identity_realizations_rendered(
            workspace.current_text, tuple(applied_raw_identities.values())
        ),
        operationalFlavorRendered=_operational_flavor_requirements_rendered(
            workspace.current_text, workspace.operational_flavor_requirements
        ),
        deactivatedReeferOperationCoherent=_deactivated_reefer_operation_coherent(
            workspace, workspace.current_text
        ),
        cargoFlavorRewritten=_cargo_flavor_rewrite_requirements_rendered(
            workspace.current_text, workspace.cargo_flavor_rewrite_requirements
        ),
        residualCandidates=_residual_candidates(workspace, leaves),
    )


def _review_audit_payload(audit: DeterministicRewriteAudit) -> dict[str, JsonValue]:
    """Keep deterministic signals without echoing OCR evidence already in the prompt."""

    return {
        "textChanged": audit.textChanged,
        "pageMarkersPreserved": audit.pageMarkersPreserved,
        "newlineConventionPreserved": audit.newlineConventionPreserved,
        "lineCountPreserved": audit.lineCountPreserved,
        "blankLineTopologyPreserved": audit.blankLineTopologyPreserved,
        "inlineSlotTopologyPreserved": audit.inlineSlotTopologyPreserved,
        "structuralLinePrefixesPreserved": audit.structuralLinePrefixesPreserved,
        "sourceStatusSurfacesPreserved": audit.sourceStatusSurfacesPreserved,
        "blankLineDepthBounded": audit.blankLineDepthBounded,
        "noPlaceholderIntroduced": audit.noPlaceholderIntroduced,
        "requiredSurfacesRendered": audit.requiredSurfacesRendered,
        "jurisdictionalSurfacesRendered": audit.jurisdictionalSurfacesRendered,
        "requiredTargetLiteralsRendered": audit.requiredTargetLiteralsRendered,
        "targetValueOccurrencesExact": audit.targetValueOccurrencesExact,
        "anchoredScalarReplacementsRendered": audit.anchoredScalarReplacementsRendered,
        "rawAuxiliaryIdentitiesReplaced": audit.rawAuxiliaryIdentitiesReplaced,
        "operationalFlavorRendered": audit.operationalFlavorRendered,
        "deactivatedReeferOperationCoherent": audit.deactivatedReeferOperationCoherent,
        "cargoFlavorRewritten": audit.cargoFlavorRewritten,
        "residualCandidates": cast(
            JsonValue,
            [
                {
                    "targetPath": row.targetPath,
                    "sourceValue": row.sourceValue,
                    "currentOccurrences": row.currentOccurrences,
                }
                for row in audit.residualCandidates
            ],
        ),
    }


def _combine_usage(rows: Sequence[LinguisticUsageReceipt]) -> LinguisticUsageReceipt:
    requestful_rows = tuple(row for row in rows if row.requests)
    provider_costs = tuple(
        row.providerReportedCostUsd
        for row in requestful_rows
        if row.providerReportedCostUsd is not None
    )
    complete_provider_cost_coverage = len(provider_costs) == len(requestful_rows)
    return LinguisticUsageReceipt(
        requests=sum(row.requests for row in rows),
        providerResponseIds=tuple(value for row in rows for value in row.providerResponseIds),
        finishReasons=tuple(value for row in rows for value in row.finishReasons),
        inputTokens=sum(row.inputTokens for row in rows),
        cacheReadTokens=sum(row.cacheReadTokens for row in rows),
        cacheWriteTokens=sum(row.cacheWriteTokens for row in rows),
        outputTokens=sum(row.outputTokens for row in rows),
        reasoningTokens=sum(row.reasoningTokens for row in rows),
        visibleOutputTokens=sum(row.visibleOutputTokens for row in rows),
        providerTokenAccountingAnomaly=any(row.providerTokenAccountingAnomaly for row in rows),
        estimatedCostUsd=sum((row.estimatedCostUsd for row in rows), Decimal(0)),
        # A partial sum would look authoritative while silently under-reporting billed cost.
        # Publish no aggregate provider total unless every requestful receipt contains it; the
        # pinned token-price estimate remains available independently.
        providerReportedCostUsd=(
            sum(provider_costs, Decimal(0))
            if provider_costs and complete_provider_cost_coverage
            else None
        ),
        downstreamProviders=tuple(value for row in rows for value in row.downstreamProviders),
    )


def _usage_by_stage(
    bundles: Sequence[RewriteCycleBundle],
) -> dict[str, LinguisticUsageReceipt]:
    grouped: dict[str, list[LinguisticUsageReceipt]] = {
        "draft_editor": [],
        "correction_editor": [],
        "semantic_reviewer": [],
    }
    for bundle in bundles:
        for record in bundle.stageRecords:
            grouped[record.stage].append(record.usage)
    return {
        stage: _combine_usage(rows) if rows else _empty_usage() for stage, rows in grouped.items()
    }


def _settings(
    provider: Any,
    *,
    stage: Literal["editor", "reviewer"],
    prompt_sha256: str,
    route_provider: str | None = None,
) -> ModelSettings:
    if provider.kind == "openrouter":
        routing: dict[str, Any] = {
            "require_parameters": provider.require_parameters,
            "data_collection": provider.data_collection,
            "allow_fallbacks": provider.allow_fallbacks if route_provider is None else False,
        }
        if route_provider is not None:
            # OpenRouter can surface an upstream endpoint's capacity error without trying the
            # remaining provider order. The hybrid runner therefore pins one route at a time and
            # performs observable application-level failover. A pinned attempt must never escape
            # to an unrecorded provider inside the gateway.
            routing["only"] = [route_provider]
            routing["order"] = [route_provider]
        elif provider.provider_only is not None:
            routing["only"] = list(provider.provider_only)
        if route_provider is None and provider.provider_order is not None:
            routing["order"] = list(provider.provider_order)
        if provider.provider_sort is not None:
            routing["sort"] = provider.provider_sort
        if provider.max_price is not None:
            routing["max_price"] = {
                "prompt": provider.max_price.prompt,
                "completion": provider.max_price.completion,
            }
        openrouter_settings: OpenRouterModelSettings = {
            "max_tokens": provider.max_output_tokens,
            "timeout": provider.request_timeout_seconds,
            "openrouter_reasoning": {"effort": provider.reasoning_effort},
            "openrouter_provider": cast(Any, routing),
        }
        generation = provider.generation_settings
        if generation is not None:
            if generation.temperature is not None:
                openrouter_settings["temperature"] = generation.temperature
            if generation.top_p is not None:
                openrouter_settings["top_p"] = generation.top_p
        return cast(ModelSettings, openrouter_settings)

    settings: OpenAIResponsesModelSettings = openai_responses_settings(provider)
    settings["parallel_tool_calls"] = False
    # Retries need the validation feedback, not a replay of earlier hidden reasoning. Keeping
    # reasoning scoped to the current turn avoids paying input tokens for stale thought traces.
    settings["openai_reasoning_context"] = "current_turn"
    # Only the stable system prompt, strict schema, and marker preceding CachePoint are cached.
    # The document-specific suffix is deliberately outside the explicit breakpoint.
    settings["openai_prompt_cache_key"] = (
        f"dococr:rewrite16:{provider.model}:{stage[0]}:{prompt_sha256[:12]}"
    )
    settings["openai_prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
    # Provider responses are retained for audit, so let PydanticAI continue validation-repair
    # requests from the stored response instead of replaying the same prompt and tool call. This
    # applies only inside one Agent.run; independent editor/reviewer passes remain isolated.
    if provider.store_responses:
        settings["openai_previous_response_id"] = "auto"
    return cast(ModelSettings, settings)


def _prompt_content(
    *,
    stage: Literal["editor", "reviewer"],
    immutable: Mapping[str, JsonValue],
    active: Mapping[str, JsonValue],
    use_cache_point: bool,
) -> tuple[UserContent, ...]:
    """Place every case-specific byte after the explicit provider cache boundary."""

    dynamic = (
        "IMMUTABLE CASE CONTEXT\n"
        + json.dumps(immutable, ensure_ascii=False, separators=(",", ":"))
        + "\nACTIVE PASS\n"
        + json.dumps(active, ensure_ascii=False, separators=(",", ":"))
    )
    stable = f"Stable synthetic B/L atomic {stage} contract version 16."
    if use_cache_point:
        return stable, CachePoint(), dynamic
    return stable, dynamic


def _prompt_payload_sha256(content: Sequence[UserContent]) -> str:
    serializable: list[JsonValue] = []
    for value in content:
        if isinstance(value, str):
            serializable.append(value)
        elif isinstance(value, CachePoint):
            serializable.append({"kind": value.kind, "ttl": value.ttl})
        else:
            raise TypeError(f"unsupported prompt payload content: {type(value).__name__}")
    return sha256_bytes(canonical_json_bytes(serializable))


class RewriteRuntime:
    def __init__(
        self,
        *,
        config: SynthesisRawTextRewriteCycleProbeConfig,
        editor_prompt: str,
        reviewer_prompt: str,
        editor_model: Model,
        reviewer_model: Model,
        target_integrity_resources: TargetIntegrityResources,
        limiter: ConcurrencyLimiter,
        staged: StagedArtifactRun,
    ) -> None:
        self.config = config
        self.editor_prompt = editor_prompt
        self.reviewer_prompt = reviewer_prompt
        self.editor_model = editor_model
        self.reviewer_model = reviewer_model
        self.target_integrity_resources = target_integrity_resources
        self.limiter = limiter
        self.staged = staged

    @staticmethod
    def _case_prefix(state: RewriteState) -> str:
        return f"cases/{state.case_number:02d}-{state.document_id}"

    def _persist_stage(self, state: RewriteState, record: RewriteStageRecord) -> None:
        if record.sequence != len(state.stages) + 1:
            raise RuntimeError("rewrite stage sequence is not contiguous")
        relative = f"{self._case_prefix(state)}/stages/{record.sequence:02d}-{record.stage}.json"
        self.staged.publish_json(relative, record.model_dump(mode="json"))
        state.stages.append(record)

    def cost(self, state: RewriteState) -> Decimal:
        return sum((row.usage.estimatedCostUsd for row in state.stages), Decimal(0))

    def cost_limited(self, state: RewriteState) -> bool:
        return self.cost(state) >= Decimal(
            str(self.config.workflow.max_total_estimated_cost_usd_per_case)
        )

    async def run_editor(
        self,
        state: RewriteState,
        *,
        phase: Literal["draft", "correction"],
        feedback: SemanticReviewReceipt | None,
    ) -> AtomicRewriteCommit | None:
        provider = self.config.providers.editor
        change_contract = label_change_contract(state.workspace)
        compact_contract = compact_label_change_contract(change_contract)
        immutable: dict[str, JsonValue] = {
            "documentId": state.document_id,
            "syntheticTargetLabel": cast(JsonValue, state.target_label),
            "targetRouteJurisdictions": cast(
                JsonValue,
                _target_route_jurisdictions(state.target_label, self.target_integrity_resources),
            ),
            "operationalCapacityLimits": cast(
                JsonValue,
                self.config.target_integrity.transport_capacity.model_dump(mode="json"),
            ),
            "labelChangeContract": cast(
                JsonValue, [row.model_dump(mode="json") for row in compact_contract]
            ),
            "surfaceRenderingRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.surface_requirements],
            ),
            "targetValueOccurrenceRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.target_value_occurrence_requirements
                ],
            ),
            "anchoredScalarReplacementRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.anchored_scalar_replacement_requirements
                ],
            ),
            "deterministicPrefills": cast(
                JsonValue,
                [
                    {
                        "lineId": row.lineId,
                        "targetPaths": list(row.targetPaths),
                        "targetSurface": row.targetSurface,
                    }
                    for row in state.workspace.deterministic_prefills
                ],
            ),
            "sourceSemanticRoleHints": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.source_role_hints],
            ),
            "inlineSlotTopologyRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.inline_slot_requirements],
            ),
            "sourceStatusPreservationRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.source_status_requirements],
            ),
            "compoundPartyFlavorRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in _active_compound_party_flavor_requirements(state.workspace)
                ],
            ),
            "rawAuxiliaryIdentityRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.raw_auxiliary_identity_requirements
                ],
            ),
            "operationalFlavorRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.operational_flavor_requirements
                ],
            ),
            "cargoFlavorRewriteRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json", exclude={"sourceSurfaces"})
                    for row in state.workspace.cargo_flavor_rewrite_requirements
                ],
            ),
            "jurisdictionalSurfaceRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.jurisdictional_requirements
                ],
            ),
            # The correction pass edits the complete current OCR and already receives the
            # authoritative label delta plus grounded reviewer findings. Re-sending the nearly
            # identical original OCR would only increase input cost and reasoning noise.
            "editableSourceRawOcrLines": (
                editable_indexed_ocr_lines(state.workspace.current_text)
                if phase == "draft"
                else None
            ),
        }
        active: dict[str, JsonValue] = {
            "task": "Rewrite current OCR to faithfully render the synthetic target label.",
            "phase": phase,
            "editableCurrentRawOcrLines": (
                editable_indexed_ocr_lines(state.workspace.current_text)
                if phase == "correction"
                else None
            ),
            "reviewFeedback": cast(
                JsonValue, feedback.model_dump(mode="json") if feedback is not None else None
            ),
        }
        payload = _prompt_content(
            stage="editor",
            immutable=immutable,
            active=active,
            use_cache_point=self.config.workflow.use_provider_explicit_prompt_cache,
        )
        settings = _settings(
            provider,
            stage="editor",
            prompt_sha256=self.config.prompts.editor.sha256,
        )
        agent = Agent[RewriteWorkspace, AtomicRewriteCommit](
            self.editor_model,
            deps_type=RewriteWorkspace,
            output_type=_terminal_editor_output(
                state.workspace,
                max_retries=self.config.workflow.editor_output_retries,
            ),
            system_prompt=self.editor_prompt,
            model_settings=settings,
            retries={"output": self.config.workflow.editor_output_retries},
            max_concurrency=self.limiter,
            name="synthetic-bl-atomic-raw-text-editor",
        )
        before_sha = state.workspace.current_sha256
        started_at = datetime.now(UTC)
        commit: AtomicRewriteCommit | None = None
        error_type: str | None = None
        error_message: str | None = None
        messages: JsonValue = []
        with capture_run_messages() as captured:
            try:
                result = await agent.run(
                    payload,
                    deps=state.workspace,
                    usage_limits=UsageLimits(
                        request_limit=self.config.workflow.max_editor_requests_per_pass,
                        output_tokens_limit=(
                            provider.max_output_tokens
                            * self.config.workflow.max_editor_requests_per_pass
                        ),
                    ),
                )
                commit = result.output
            except Exception as error:
                error_type = type(error).__name__
                error_message = str(error)
            finally:
                messages = model_messages(captured)
        responses = tuple(row for row in captured if isinstance(row, ModelResponse))
        record = RewriteStageRecord(
            schemaVersion=17,
            sequence=len(state.stages) + 1,
            stage="draft_editor" if phase == "draft" else "correction_editor",
            correctionCycle=sum(row.stage == "correction_editor" for row in state.stages),
            providerModel=provider.model,
            reasoningEffort=provider.reasoning_effort,
            startedAt=started_at,
            completedAt=datetime.now(UTC),
            inputPayloadSha256=_prompt_payload_sha256(payload),
            currentTextSha256Before=before_sha,
            currentTextSha256After=state.workspace.current_sha256,
            editorCommit=commit,
            semanticReview=None,
            deterministicAudit=None,
            usage=usage_receipt(
                responses,
                provider.pricing,
                require_provider_cost=provider.kind == "openrouter",
            ),
            modelMessages=messages,
            errorType=error_type,
            errorMessage=error_message,
        )
        self._persist_stage(state, record)
        return commit

    async def run_reviewer(
        self, state: RewriteState, audit: DeterministicRewriteAudit
    ) -> SemanticReviewReceipt | None:
        provider = self.config.providers.reviewer
        change_contract = label_change_contract(state.workspace)
        compact_contract = compact_label_change_contract(change_contract)
        immutable: dict[str, JsonValue] = {
            "documentId": state.document_id,
            "reviewedSourceLabel": cast(JsonValue, state.source_label),
            "syntheticTargetLabel": cast(JsonValue, state.target_label),
            "targetRouteJurisdictions": cast(
                JsonValue,
                _target_route_jurisdictions(state.target_label, self.target_integrity_resources),
            ),
            "operationalCapacityLimits": cast(
                JsonValue,
                self.config.target_integrity.transport_capacity.model_dump(mode="json"),
            ),
            "labelChangeContract": cast(
                JsonValue, [row.model_dump(mode="json") for row in compact_contract]
            ),
            "surfaceRenderingRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.surface_requirements],
            ),
            "targetValueOccurrenceRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.target_value_occurrence_requirements
                ],
            ),
            "anchoredScalarReplacementRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.anchored_scalar_replacement_requirements
                ],
            ),
            "sourceSemanticRoleHints": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.source_role_hints],
            ),
            "inlineSlotTopologyRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.inline_slot_requirements],
            ),
            "sourceStatusPreservationRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.source_status_requirements],
            ),
            "compoundPartyFlavorRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in compound_party_flavor_requirements(
                        state.source_label, state.target_label
                    )
                ],
            ),
            "rawAuxiliaryIdentityRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.raw_auxiliary_identity_requirements
                ],
            ),
            "operationalFlavorRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.operational_flavor_requirements
                ],
            ),
            "cargoFlavorRewriteRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json", exclude={"sourceSurfaces"})
                    for row in state.workspace.cargo_flavor_rewrite_requirements
                ],
            ),
            "jurisdictionalSurfaceRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.jurisdictional_requirements
                ],
            ),
            "sourceRawOcr": state.workspace.original_text,
        }
        active: dict[str, JsonValue] = {
            "task": "Independently accept or request exact corrections to the rewritten OCR.",
            "rewrittenRawOcr": state.workspace.current_text,
            "deterministicAudit": cast(JsonValue, _review_audit_payload(audit)),
        }
        payload = _prompt_content(
            stage="reviewer",
            immutable=immutable,
            active=active,
            use_cache_point=self.config.workflow.use_provider_explicit_prompt_cache,
        )
        settings = _settings(
            provider,
            stage="reviewer",
            prompt_sha256=self.config.prompts.reviewer.sha256,
        )
        reviewer_output = (
            NativeOutput(
                SemanticReviewReceipt,
                name="synthetic_bl_atomic_rewrite_review",
                description="Compact evidence-grounded semantic acceptance or correction list.",
                strict=True,
            )
            if provider.kind == "openai_responses"
            else ToolOutput(
                SemanticReviewReceipt,
                name="synthetic_bl_atomic_rewrite_review",
                description="Compact evidence-grounded semantic acceptance or correction list.",
                strict=True,
                max_retries=self.config.workflow.reviewer_output_retries,
            )
        )
        agent = Agent[None, SemanticReviewReceipt](
            self.reviewer_model,
            output_type=reviewer_output,
            system_prompt=self.reviewer_prompt,
            model_settings=settings,
            retries={"output": self.config.workflow.reviewer_output_retries},
            max_concurrency=self.limiter,
            name="synthetic-bl-atomic-raw-text-reviewer",
        )

        @agent.output_validator
        def validate_output(output: SemanticReviewReceipt) -> SemanticReviewReceipt:
            try:
                for finding in output.findings:
                    for evidence in finding.currentEvidence:
                        if not _evidence_occurs(evidence, state.workspace.current_text):
                            raise ValueError("current review evidence is not grounded")
                    if _finding_misclassifies_anonymous_equipment(finding, state.workspace):
                        raise ValueError(
                            "reviewer classified a deterministic anonymous equipment count as marks"
                        )
                    if _finding_conflicts_with_equipment_authority(finding, state.workspace):
                        raise ValueError(
                            "reviewer equipment finding conflicts with the reviewed surface grammar"
                        )
                    if _finding_conflicts_with_surface_authority(finding, state.workspace):
                        raise ValueError(
                            "reviewer finding conflicts with a deterministic derived surface"
                        )
                    if _finding_conflicts_with_anchored_scalar_authority(finding, state.workspace):
                        raise ValueError(
                            "reviewer reclassified a line-bound target scalar as auxiliary flavor"
                        )
                    if _finding_grounds_format_damage_only_in_unchanged_text(
                        finding, state.workspace
                    ):
                        raise ValueError(
                            "reviewer formatting finding quotes only source-identical text"
                        )
                if output.verdict == "pass" and not audit.core_passed:
                    raise ValueError("review cannot pass a failing deterministic core audit")
                return output
            except ValueError as error:
                raise ModelRetry(f"Semantic review rejected: {error}") from error

        before_sha = state.workspace.current_sha256
        started_at = datetime.now(UTC)
        review: SemanticReviewReceipt | None = None
        error_type: str | None = None
        error_message: str | None = None
        messages: JsonValue = []
        with capture_run_messages() as captured:
            try:
                result = await agent.run(
                    payload,
                    usage_limits=UsageLimits(
                        request_limit=self.config.workflow.max_reviewer_requests_per_cycle,
                        output_tokens_limit=(
                            provider.max_output_tokens
                            * self.config.workflow.max_reviewer_requests_per_cycle
                        ),
                    ),
                )
                review = result.output
            except Exception as error:
                error_type = type(error).__name__
                error_message = str(error)
            finally:
                messages = model_messages(captured)
        responses = tuple(row for row in captured if isinstance(row, ModelResponse))
        record = RewriteStageRecord(
            schemaVersion=17,
            sequence=len(state.stages) + 1,
            stage="semantic_reviewer",
            correctionCycle=sum(row.stage == "correction_editor" for row in state.stages),
            providerModel=provider.model,
            reasoningEffort=provider.reasoning_effort,
            startedAt=started_at,
            completedAt=datetime.now(UTC),
            inputPayloadSha256=_prompt_payload_sha256(payload),
            currentTextSha256Before=before_sha,
            currentTextSha256After=state.workspace.current_sha256,
            editorCommit=None,
            semanticReview=review,
            deterministicAudit=audit,
            usage=usage_receipt(
                responses,
                provider.pricing,
                require_provider_cost=provider.kind == "openrouter",
            ),
            modelMessages=messages,
            errorType=error_type,
            errorMessage=error_message,
        )
        self._persist_stage(state, record)
        return review


async def _run_state(
    state: RewriteState, runtime: RewriteRuntime
) -> tuple[Literal["quality_validated", "needs_review", "call_failed", "cost_limited"], str]:
    feedback: SemanticReviewReceipt | None = None
    for cycle in range(runtime.config.workflow.max_correction_cycles + 1):
        phase: Literal["draft", "correction"] = "draft" if cycle == 0 else "correction"
        commit = await runtime.run_editor(state, phase=phase, feedback=feedback)
        if commit is None:
            return "call_failed", f"{phase} editor call failed"
        audit = deterministic_rewrite_audit(state.workspace, state.changed_leaves)
        state.audits.append(audit)
        if not audit.core_passed:
            return "needs_review", "atomic output failed a deterministic fidelity check"
        if runtime.cost_limited(state):
            return "cost_limited", "configured per-case estimated-cost limit reached"
        review = await runtime.run_reviewer(state, audit)
        if review is None:
            return "call_failed", "semantic reviewer call failed"
        state.reviews.append(review)
        if review.verdict == "pass":
            return "quality_validated", "atomic rewrite passed deterministic and semantic review"
        if runtime.cost_limited(state):
            return "cost_limited", "configured per-case estimated-cost limit reached"
        feedback = review
    return "needs_review", "semantic findings remain after the bounded correction cycles"


def _empty_usage() -> LinguisticUsageReceipt:
    return LinguisticUsageReceipt(
        requests=0,
        providerResponseIds=(),
        finishReasons=(),
        inputTokens=0,
        cacheReadTokens=0,
        cacheWriteTokens=0,
        outputTokens=0,
        reasoningTokens=0,
        visibleOutputTokens=0,
        estimatedCostUsd=Decimal(0),
    )


def _target_before_semantic_equipment(target: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the capacity envelope used before relation-v5 equipment assignment."""

    source = copy.deepcopy(dict(target))
    patch = source.get("documentPatch")
    if not isinstance(patch, dict):
        raise ValueError("synthetic target has no documentPatch")
    for container in patch.get("containers") or []:
        if not isinstance(container, dict):
            raise ValueError("synthetic target container is not an object")
        container.pop("sizeCategory", None)
        container.pop("typeCategory", None)
    return source


def _topology_signature(label: Mapping[str, Any]) -> dict[str, JsonValue]:
    patch = cast(Mapping[str, Any], label.get("documentPatch") or {})
    parties = cast(Mapping[str, Any], patch.get("parties") or {})
    allocation_groups = patch.get("cargoAllocationGroups") or []
    return {
        "partyRoles": cast(
            list[JsonValue],
            sorted(key for key, value in parties.items() if value not in (None, [], {})),
        ),
        "notifyPartyCount": len(parties.get("notifyParties") or []),
        "containerCount": len(patch.get("containers") or []),
        "cargoGroupCount": len(patch.get("cargoGroups") or []),
        "cargoPackageCount": len(patch.get("cargoPackages") or []),
        "allocationGroupCount": len(allocation_groups),
        "allocationCount": sum(
            len(row.get("allocations") or [])
            for row in allocation_groups
            if isinstance(row, Mapping)
        ),
    }


def _party_slug(party: Mapping[str, Any]) -> str:
    name = party.get("name")
    if not isinstance(name, str):
        raise ValueError("cannot synthesize a domain for a party without a name")
    slug = _DOMAIN_TOKEN.sub(
        "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    )
    if len(slug) < 3:
        raise ValueError("party name is too short to derive a synthetic domain")
    return slug[:48]


def _domain_reason(hostname: str, country: str | None, countries: CountryRegistry) -> str | None:
    lowered = hostname.rstrip(".").casefold()
    tld = lowered.rsplit(".", 1)[-1]
    if lowered in _SPECIAL_USE_DOMAINS or tld in _SPECIAL_USE_TLDS:
        return "special_use_domain"
    if country is None or len(tld) != 2 or not tld.isalpha():
        return None
    country_code = countries.resolve(country)
    if country_code is None:
        raise ValueError(
            f"target party country is absent from the pinned ISO registry: {country!r}"
        )
    return "country_incoherent_cctld" if tld != country_code.casefold() else None


def _repair_party_domains(
    target: dict[str, Any], resources: TargetIntegrityResources
) -> tuple[TargetIntegrityChange, ...]:
    patch = cast(dict[str, Any], target["documentPatch"])
    parties = cast(dict[str, Any], patch.get("parties") or {})
    changes: list[TargetIntegrityChange] = []
    for role, value in parties.items():
        rows = value if role == "notifyParties" and isinstance(value, list) else [value]
        for occurrence, party in enumerate(rows):
            if not isinstance(party, dict):
                continue
            contacts = party.get("contactDetails")
            if not isinstance(contacts, dict):
                continue
            country = party.get("country") if isinstance(party.get("country"), str) else None
            prefix = f"documentPatch.parties.{role}"
            if role == "notifyParties":
                prefix += f"[{occurrence}]"
            for contact_field in ("emailAddresses", "websiteUrls"):
                values = contacts.get(contact_field)
                if not isinstance(values, list):
                    continue
                for index, before in enumerate(values):
                    if not isinstance(before, str):
                        continue
                    if contact_field == "emailAddresses":
                        if "@" not in before:
                            continue
                        local, old_domain = before.rsplit("@", 1)
                        reason = _domain_reason(old_domain, country, resources.countries)
                    else:
                        parsed = urlsplit(before if "://" in before else f"https://{before}")
                        reason = _domain_reason(parsed.hostname or "", country, resources.countries)
                    if reason is None:
                        continue
                    domain = f"{_party_slug(party)}.com"
                    if contact_field == "emailAddresses":
                        after = f"{local}@{domain}"
                    else:
                        after = urlunsplit(
                            (
                                parsed.scheme or "https",
                                domain,
                                parsed.path,
                                parsed.query,
                                parsed.fragment,
                            )
                        )
                    values[index] = after
                    changes.append(
                        TargetIntegrityChange(
                            path=f"{prefix}.contactDetails.{contact_field}[{index}]",
                            reason=cast(Any, reason),
                            before=before,
                            after=after,
                        )
                    )
    return tuple(changes)


def _pluralize_package_surface(value: str) -> str:
    lowered = value.casefold()
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return value + "es"
    if lowered.endswith("y") and len(value) > 1 and lowered[-2] not in "aeiou":
        return value[:-1] + "ies"
    return value + "s"


def _preserve_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.islower():
        return target.lower()
    if source.istitle() and target.islower():
        return target.title()
    return target


def _repair_package_information(
    target: dict[str, Any], resources: TargetIntegrityResources
) -> tuple[TargetIntegrityChange, ...]:
    patch = cast(dict[str, Any], target["documentPatch"])
    packages_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for package in patch.get("cargoPackages") or []:
        if isinstance(package, Mapping) and isinstance(package.get("groupId"), str):
            packages_by_group.setdefault(cast(str, package["groupId"]), []).append(package)
    base_surfaces = {row.displayName.split(",", 1)[0] for row in resources.packages.payload.entries}
    known_surfaces = sorted(
        {surface for base in base_surfaces for surface in (base, _pluralize_package_surface(base))},
        key=len,
        reverse=True,
    )
    noun_pattern = re.compile(
        r"(?P<noun>" + "|".join(re.escape(value) for value in known_surfaces) + r")\b",
        re.IGNORECASE,
    )
    changes: list[TargetIntegrityChange] = []
    for group_index, group in enumerate(patch.get("cargoGroups") or []):
        if not isinstance(group, dict) or not isinstance(group.get("groupId"), str):
            continue
        packages = packages_by_group.get(cast(str, group["groupId"]), [])
        if len(packages) != 1:
            continue
        package = packages[0]
        quantity = package.get("quantity")
        category = package.get("typeCategory")
        if not isinstance(quantity, int) or not isinstance(category, str):
            continue
        desired_base = resources.packages.entry(category).displayName.split(",", 1)[0]
        desired = _pluralize_package_surface(desired_base) if quantity != 1 else desired_base
        quantity_pattern = re.compile(
            rf"(?<![0-9])(?:{re.escape(str(quantity))}|{re.escape(f'{quantity:,}')})\s+",
            re.IGNORECASE,
        )
        values = group.get("additionalInformation")
        if not isinstance(values, list):
            continue
        for value_index, before in enumerate(values):
            if not isinstance(before, str):
                continue
            quantity_match = quantity_pattern.search(before)
            if quantity_match is None:
                continue
            noun_match = noun_pattern.match(before, quantity_match.end())
            if noun_match is None:
                continue
            observed = noun_match.group("noun")
            if observed.casefold() in {desired_base.casefold(), desired.casefold()}:
                continue
            replacement = _preserve_case(observed, desired)
            after = before[: noun_match.start()] + replacement + before[noun_match.end() :]
            values[value_index] = after
            changes.append(
                TargetIntegrityChange(
                    path=(
                        f"documentPatch.cargoGroups[{group_index}].additionalInformation"
                        f"[{value_index}]"
                    ),
                    reason="package_surface_category_mismatch",
                    before=before,
                    after=after,
                )
            )
    return tuple(changes)


def _surface_chunks(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", value))


def _embedded_chunk_range(short: str, long: str) -> tuple[int, int] | None:
    short_chunks = _surface_chunks(short)
    long_chunks = _surface_chunks(long)
    normalized_short = tuple(chunk.casefold() for chunk in short_chunks)
    normalized_long = tuple(chunk.casefold() for chunk in long_chunks)
    matches = tuple(
        (start, start + len(short_chunks))
        for start in range(len(long_chunks) - len(short_chunks) + 1)
        if normalized_long[start : start + len(short_chunks)] == normalized_short
    )
    return matches[0] if len(matches) == 1 else None


def _same_chunk_topology(source: str, target: str) -> bool:
    source_chunks = _surface_chunks(source)
    target_chunks = _surface_chunks(target)
    if len(source_chunks) != len(target_chunks):
        return False
    return all(
        source_chunk == target_chunk
        for source_chunk, target_chunk in zip(source_chunks, target_chunks, strict=True)
        if not source_chunk.isalnum()
    )


def repair_overlapping_source_scalar_targets(
    raw_text: str,
    source_label: Mapping[str, JsonValue],
    target: dict[str, JsonValue],
) -> tuple[TargetIntegrityChange, ...]:
    """Preserve one-to-many label semantics represented by an overlapping OCR scalar.

    Some reviewed labels intentionally expose both a complete printed mark and an identifier
    embedded inside that mark. Independent synthesis can otherwise assign mutually inconsistent
    target strings even though the template has only one physical occurrence. This repair is
    permitted only when every short-surface occurrence is contained by the same longer source
    surface and both generated long surfaces preserve the source punctuation/run topology.
    """

    leaves = tuple(
        leaf
        for leaf in rewrite_changed_leaves(source_label, target)
        if leaf.requiresTextEdit
        and leaf.sourcePresent
        and leaf.targetPresent
        and isinstance(leaf.sourceValue, str)
        and isinstance(leaf.targetValue, str)
        and _ANCHORABLE_IDENTIFIER_PATH.fullmatch(leaf.path) is not None
    )
    changes: list[TargetIntegrityChange] = []
    candidates = sorted(leaves, key=lambda row: (-len(cast(str, row.sourceValue)), row.path))
    for long_leaf in candidates:
        source_long = cast(str, long_leaf.sourceValue)
        long_matches = tuple(_anchored_scalar_pattern(source_long).finditer(raw_text))
        if not long_matches:
            continue
        for short_leaf in candidates:
            source_short = cast(str, short_leaf.sourceValue)
            if short_leaf.path == long_leaf.path or len(source_short) >= len(source_long):
                continue
            chunk_range = _embedded_chunk_range(source_short, source_long)
            if chunk_range is None:
                continue
            short_matches = tuple(_anchored_scalar_pattern(source_short).finditer(raw_text))
            if not short_matches or any(
                not any(
                    long_match.start() <= short_match.start()
                    and short_match.end() <= long_match.end()
                    for long_match in long_matches
                )
                for short_match in short_matches
            ):
                continue
            current_long = target_value(target, long_leaf.path)
            current_short = target_value(target, short_leaf.path)
            if not isinstance(current_long, str) or not isinstance(current_short, str):
                continue
            if _anchored_scalar_pattern(current_short).search(current_long) is not None:
                continue
            if not _same_chunk_topology(source_long, current_long):
                raise ValueError(
                    "cannot align overlapping source scalars because the synthetic long value "
                    f"changed punctuation topology: {long_leaf.path}"
                )
            start, end = chunk_range
            target_chunks = _surface_chunks(current_long)
            repaired = "".join((*target_chunks[:start], current_short, *target_chunks[end:]))
            if not repaired.strip() or repaired == current_long:
                raise ValueError(f"overlapping scalar repair made no change: {long_leaf.path}")
            set_target_value(cast(dict[str, Any], target), long_leaf.path, repaired)
            changes.append(
                TargetIntegrityChange(
                    path=long_leaf.path,
                    reason="overlapping_source_scalar_topology",
                    before=current_long,
                    after=repaired,
                )
            )
    # A carrier brand can be the exact prefix of another labeled party's legal name (for
    # example ``TRANSGLORY`` and forwarding agent ``TRANSGLORY S.A.``). Independent party
    # synthesis otherwise creates two incompatible targets for the same physical substring.
    # Preserve the longer role's legal suffix while making its generated identity share the
    # carrier target prefix. This is a semantic topology repair, not fuzzy name matching: the
    # only accepted remainder is a conventional legal-entity suffix.
    legal_suffix_tokens = {
        "A",
        "AB",
        "AG",
        "AS",
        "BV",
        "CO",
        "CORP",
        "GMBH",
        "INC",
        "KG",
        "KK",
        "LIMITED",
        "LLC",
        "LTD",
        "NV",
        "PLC",
        "PTE",
        "PTY",
        "S",
        "SA",
        "SAC",
        "SARL",
        "SAS",
        "SPA",
        "SRL",
    }
    party_names = tuple(
        leaf
        for leaf in rewrite_changed_leaves(source_label, target)
        if leaf.requiresTextEdit
        and leaf.sourcePresent
        and leaf.targetPresent
        and isinstance(leaf.sourceValue, str)
        and isinstance(leaf.targetValue, str)
        and re.fullmatch(
            r"documentPatch\.parties\.(?:notifyParties\[[0-9]+\]|[A-Za-z]+)\.name",
            leaf.path,
        )
        is not None
    )
    for long_leaf in sorted(
        party_names, key=lambda row: (-len(cast(str, row.sourceValue)), row.path)
    ):
        source_long = cast(str, long_leaf.sourceValue)
        long_chunks = _surface_chunks(source_long)
        for short_leaf in party_names:
            source_short = cast(str, short_leaf.sourceValue)
            if short_leaf.path == long_leaf.path or len(source_short) >= len(source_long):
                continue
            chunk_range = _embedded_chunk_range(source_short, source_long)
            if chunk_range is None or chunk_range[0] != 0:
                continue
            suffix = "".join(long_chunks[chunk_range[1] :])
            suffix_tokens = set(re.findall(r"[A-Z0-9]+", _semantic_normalize(suffix)))
            if not suffix_tokens or not suffix_tokens <= legal_suffix_tokens:
                continue
            current_long = target_value(target, long_leaf.path)
            current_short = target_value(target, short_leaf.path)
            if not isinstance(current_long, str) or not isinstance(current_short, str):
                continue
            if _anchored_scalar_pattern(current_short).search(current_long) is not None:
                continue
            repaired = current_short.rstrip() + suffix
            if not repaired.strip() or repaired == current_long:
                continue
            set_target_value(cast(dict[str, Any], target), long_leaf.path, repaired)
            changes.append(
                TargetIntegrityChange(
                    path=long_leaf.path,
                    reason="shared_source_party_identity_topology",
                    before=current_long,
                    after=repaired,
                )
            )
            break
    return tuple(changes)


def _label_hs_codes(label: Mapping[str, Any]) -> tuple[str, ...]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return ()
    output: list[str] = []
    for group in cast(Sequence[Any], patch.get("cargoGroups") or ()):
        if not isinstance(group, Mapping):
            continue
        for value in cast(Sequence[Any], group.get("hsCodes") or ()):
            if isinstance(value, str):
                output.append("".join(re.findall(r"[0-9]", value)))
    return tuple(output)


def _explicit_hs_surfaces(raw_text: str, code: str) -> tuple[tuple[str, int, int], ...]:
    matches = tuple(
        (match.group(0), match.start(), match.end())
        for match in _HS_NUMERIC_SURFACE.finditer(raw_text)
        if "".join(re.findall(r"[0-9]", match.group(0))) == code
    )
    if not matches:
        raise ValueError(f"explicit HS evidence parser returned an unlocatable code: {code}")
    return matches


def recover_explicit_hs_target_facts(
    *,
    raw_text: str,
    source_label: Mapping[str, Any],
    target: Mapping[str, Any],
    plan: DocumentLinguisticPlan,
) -> tuple[
    dict[str, Any],
    tuple[RecoveredExplicitHsTargetFact, ...],
    tuple[SurfaceRenderingRequirement, ...],
]:
    """Restore a source-visible HS slot omitted by the reviewed source label.

    The semantic completion plan is the sole source of replacement identities.  The model never
    invents a task-extractable code.  A single printed aggregate source code may expand to one
    code per target cargo identity; multiple printed codes must have exactly the same cardinality
    as the target identities.  Any partial or ambiguous situation fails before a provider call.
    """

    assigned = copy.deepcopy(dict(target))
    explicit_codes = explicit_hs_codes_from_text(raw_text)
    if not explicit_codes:
        return assigned, (), ()
    source_codes = _label_hs_codes(source_label)
    target_codes = _label_hs_codes(assigned)
    missing_from_source_label = tuple(code for code in explicit_codes if code not in source_codes)
    if not missing_from_source_label:
        return assigned, (), ()
    if source_codes or target_codes:
        raise ValueError(
            "explicit raw HS evidence is only partially represented by source/target labels; "
            f"explicit={explicit_codes}, source={source_codes}, target={target_codes}"
        )
    if plan.baseDocumentId != plan.cargoSeed.sourceDocumentId:
        raise ValueError("linguistic plan document identity differs from its cargo seed")
    if plan.scenarioId != plan.cargoSeed.scenarioId:
        raise ValueError("linguistic plan scenario identity differs from its cargo seed")

    patch = assigned.get("documentPatch")
    if not isinstance(patch, dict):
        raise ValueError("synthetic target has no documentPatch")
    target_groups = patch.get("cargoGroups")
    if not isinstance(target_groups, list):
        raise ValueError("synthetic target has no cargo groups")
    plan_groups = {group.groupId: group for group in plan.cargoSeed.cargoGroups}
    target_group_ids = tuple(
        group.get("groupId") if isinstance(group, Mapping) else None for group in target_groups
    )
    if any(not isinstance(group_id, str) for group_id in target_group_ids):
        raise ValueError("synthetic target cargo group lacks a string groupId")
    if set(cast(tuple[str, ...], target_group_ids)) != set(plan_groups):
        raise ValueError(
            "linguistic plan and synthetic target cargo-group identities differ: "
            f"target={target_group_ids}, plan={tuple(plan_groups)}"
        )

    planned: list[tuple[int, int, str, str]] = []
    for group_index, group in enumerate(target_groups):
        group_id = cast(str, group["groupId"])
        identities = plan_groups[group_id].goodsIdentities
        if not identities or any(identity.hsCode is None for identity in identities):
            raise ValueError(
                "explicit raw HS slot cannot be populated because its pinned semantic plan lacks "
                f"an HS identity for cargo group {group_id!r}"
            )
        for identity_index, identity in enumerate(identities):
            hs_code = cast(str, identity.hsCode)
            planned.append((group_index, identity_index, group_id, hs_code[:6]))
    if len(explicit_codes) not in {1, len(planned)}:
        raise ValueError(
            "explicit raw HS slot cardinality cannot be projected without guessing: "
            f"printed={len(explicit_codes)}, target_identities={len(planned)}"
        )

    source_widths = (
        (len(explicit_codes[0]),) * len(planned)
        if len(explicit_codes) == 1
        else tuple(len(code) for code in explicit_codes)
    )
    stream = DeterministicStream(
        seed=0,
        namespace="raw-text-explicit-hs-target-recovery-v1",
        identity=f"{plan.baseDocumentId}:{plan.scenarioId}",
    )
    rendered: list[str] = []
    recoveries: list[RecoveredExplicitHsTargetFact] = []
    codes_by_group: dict[int, list[str]] = {}
    target_paths: list[str] = []
    for (group_index, identity_index, group_id, hs6), width in zip(
        planned, source_widths, strict=True
    ):
        code = render_synthetic_hs_code(
            hs6=hs6,
            output_digits=width,
            stream=stream.derive(f"{group_id}:{identity_index}"),
        )
        path = f"documentPatch.cargoGroups[{group_index}].hsCodes[{identity_index}]"
        rendered.append(code)
        target_paths.append(path)
        codes_by_group.setdefault(group_index, []).append(code)
        recoveries.append(
            RecoveredExplicitHsTargetFact(
                targetPath=path,
                sourceEvidenceCodes=explicit_codes,
                semanticPlanPath=(
                    f"cargoSeed.cargoGroups[{group_id}].goodsIdentities[{identity_index}].hsCode"
                ),
                hs6=hs6,
                renderedCode=code,
                outputDigits=width,
                method="explicit_raw_hs_slot_from_pinned_semantic_plan_v1",
            )
        )
    for group_index, codes in codes_by_group.items():
        cast(dict[str, Any], target_groups[group_index])["hsCodes"] = codes

    requirements: list[SurfaceRenderingRequirement] = []
    if len(explicit_codes) == 1:
        source_code = explicit_codes[0]
        surfaces = _explicit_hs_surfaces(raw_text, source_code)
        for source_surface in dict.fromkeys(surface for surface, _, _ in surfaces):
            target_surface = ", ".join(
                _format_numeric_surface(source_surface, code) for code in rendered
            )
            first = next(row for row in surfaces if row[0] == source_surface)
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="hs_code_block",
                    targetPath=";".join(target_paths),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    sourceOccurrences=sum(row[0] == source_surface for row in surfaces),
                    contextEvidence=raw_text[
                        max(0, first[1] - 120) : min(len(raw_text), first[2] + 80)
                    ],
                    sourceLineIds=tuple(
                        f"L{raw_text.count(chr(10), 0, start) + 1:05d}"
                        for surface, start, _end in surfaces
                        if surface == source_surface
                    ),
                )
            )
    else:
        for source_code, target_code, target_path in zip(
            explicit_codes, rendered, target_paths, strict=True
        ):
            surfaces = _explicit_hs_surfaces(raw_text, source_code)
            for source_surface in dict.fromkeys(surface for surface, _, _ in surfaces):
                first = next(row for row in surfaces if row[0] == source_surface)
                requirements.append(
                    SurfaceRenderingRequirement(
                        kind="hs_code",
                        targetPath=target_path,
                        sourceSurface=source_surface,
                        targetSurface=_format_numeric_surface(source_surface, target_code),
                        sourceOccurrences=sum(row[0] == source_surface for row in surfaces),
                        contextEvidence=raw_text[
                            max(0, first[1] - 120) : min(len(raw_text), first[2] + 80)
                        ],
                        sourceLineIds=tuple(
                            f"L{raw_text.count(chr(10), 0, start) + 1:05d}"
                            for surface, start, _end in surfaces
                            if surface == source_surface
                        ),
                    )
                )
    return assigned, tuple(recoveries), tuple(requirements)


def prepare_target_integrity(
    source_label: Mapping[str, Any],
    target: Mapping[str, Any],
    config: RewritePreparationConfig,
    resources: TargetIntegrityResources,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Validate and, if necessary, reproject a synthetic target before any API call."""

    assigned = copy.deepcopy(dict(target))
    source_topology = _topology_signature(source_label)
    target_topology = _topology_signature(assigned)
    if source_topology != target_topology:
        raise ValueError(
            "source/target structural topology differs; select a compatible template before "
            f"rewriting: source={source_topology}, target={target_topology}"
        )
    semantic_changes = [
        *_repair_party_domains(assigned, resources),
        *_repair_package_information(assigned, resources),
    ]
    source = _target_before_semantic_equipment(assigned)
    receipt = reproject_measures_for_semantic_equipment(
        source_target=source,
        assigned_target=assigned,
        limits=capacity_limits(config.target_integrity.transport_capacity),
    )
    canonical = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
        document_id="synthetic-target-integrity",
        target=assigned,
    )
    payload = cast(dict[str, JsonValue], receipt.to_dict())
    payload["capacity_changed"] = cast(bool, payload["changed"])
    payload["changed"] = bool(payload["changed"] or semantic_changes)
    payload["semantic_changes"] = cast(
        JsonValue, [row.model_dump(mode="json") for row in semantic_changes]
    )
    payload["source_topology"] = cast(JsonValue, source_topology)
    payload["target_topology"] = cast(JsonValue, target_topology)
    payload["topology_matched"] = True
    return canonical, payload


def _project_unrenderable_equipment_to_template(
    raw_text: str,
    source_label: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[TargetIntegrityChange, ...]]:
    """Remove target equipment semantics only when the OCR has no slot that can express them."""

    projected = copy.deepcopy(dict(target))
    source_patch = source_label.get("documentPatch")
    target_patch = projected.get("documentPatch")
    source_containers = (
        source_patch.get("containers") if isinstance(source_patch, Mapping) else None
    )
    target_containers = target_patch.get("containers") if isinstance(target_patch, dict) else None
    if not isinstance(source_containers, Sequence) or isinstance(source_containers, (str, bytes)):
        return projected, ()
    if not isinstance(target_containers, list) or len(target_containers) != len(source_containers):
        return projected, ()

    anchored = container_equipment_replacement_requirements(
        raw_text, source_label, projected
    )
    aggregate = (
        *_aggregate_equipment_breakdown_requirements(raw_text, source_label, projected),
        *_carrier_receipt_equipment_breakdown_requirements(raw_text, projected),
    )
    renderable_paths = {
        path for requirement in anchored for path in requirement.targetPaths
    } | {requirement.targetPath for requirement in aggregate}
    changes: list[TargetIntegrityChange] = []
    for index, (source_container, target_container) in enumerate(
        zip(source_containers, target_containers, strict=True)
    ):
        if not isinstance(source_container, Mapping) or not isinstance(target_container, dict):
            raise ValueError("equipment topology projection received a non-object container")
        size = target_container.get("sizeCategory")
        category = target_container.get("typeCategory")
        if not isinstance(size, str) or not isinstance(category, str):
            continue
        path = f"documentPatch.containers[{index}].printedEquipmentSurface"
        source_surface = next(
            (
                cast(str, source_container[key])
                for key in ("typeDescription", "typeCode")
                if isinstance(source_container.get(key), str)
            ),
            None,
        )
        source_already_expresses_target = False
        if source_surface is not None and _anchored_scalar_pattern(source_surface).search(raw_text):
            reviewed = review_source_equipment_surface(
                source_surface,
                temperature_present=target_container.get("temperatureSetpoint") is not None,
            )
            source_already_expresses_target = (
                reviewed.size_category,
                reviewed.type_category,
            ) == (size, category)
        if path in renderable_paths or source_already_expresses_target:
            continue
        if target_container.get("temperatureSetpoint") is not None:
            raise ValueError(
                "temperature-bearing target equipment has no printable source topology: " + path
            )
        target_container.pop("sizeCategory", None)
        target_container.pop("typeCategory", None)
        changes.append(
            TargetIntegrityChange(
                path=path,
                reason="unprinted_equipment_topology",
                before=f"{size}/{category}",
                after="null",
            )
        )
    return projected, tuple(changes)


def prepare_rewrite_state(
    *,
    case_number: int,
    source_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    linguistic_plan: DocumentLinguisticPlan,
    feature: Mapping[str, JsonValue],
    config: RewritePreparationConfig,
    target_integrity_resources: TargetIntegrityResources,
    defer_ambiguous_party_scalar_cardinality: bool = False,
) -> RewriteState:
    """Build the complete host-owned rewrite contract without constructing or calling a model."""

    document_id = cast(str, source_row["documentId"])
    source_text = cast(str, source_row["joinedRawText"])
    source_label = cast(
        dict[str, JsonValue],
        BILL_OF_LADING_TASK_ADAPTER.validate_target(
            document_id=document_id,
            target=cast(Mapping[str, Any], source_row[config.inputs.source_target_field]),
        ),
    )
    scenario_id = cast(str, target_row["scenarioId"])
    upstream_target_label = cast(
        dict[str, JsonValue],
        BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
            document_id=scenario_id,
            target=cast(Mapping[str, Any], target_row["target"]),
        ),
    )
    recovered_target_label, recovered_hs_facts, recovered_hs_surfaces = (
        recover_explicit_hs_target_facts(
            raw_text=source_text,
            source_label=source_label,
            target=upstream_target_label,
            plan=linguistic_plan,
        )
    )
    topology_projected_target, equipment_topology_changes = (
        _project_unrenderable_equipment_to_template(
            source_text,
            source_label,
            recovered_target_label,
        )
    )
    effective_target_label, target_integrity = prepare_target_integrity(
        source_label,
        topology_projected_target,
        config,
        target_integrity_resources,
    )
    if equipment_topology_changes:
        existing_changes = cast(list[JsonValue], target_integrity["semantic_changes"])
        target_integrity["semantic_changes"] = cast(
            JsonValue,
            [
                *(row.model_dump(mode="json") for row in equipment_topology_changes),
                *existing_changes,
            ],
        )
        target_integrity["changed"] = True
    overlap_changes = repair_overlapping_source_scalar_targets(
        source_text, source_label, effective_target_label
    )
    if overlap_changes:
        effective_target_label = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
            document_id=scenario_id,
            target=effective_target_label,
        )
        existing_changes = cast(list[JsonValue], target_integrity["semantic_changes"])
        target_integrity["semantic_changes"] = cast(
            JsonValue,
            [
                *existing_changes,
                *(row.model_dump(mode="json") for row in overlap_changes),
            ],
        )
        target_integrity["changed"] = True
    target_integrity["recovered_task_facts"] = cast(
        JsonValue, [row.model_dump(mode="json") for row in recovered_hs_facts]
    )
    target_integrity["recovered_task_fact_count"] = len(recovered_hs_facts)
    rendering_requirements = (
        *surface_rendering_requirements(
            source_text,
            source_label,
            effective_target_label,
            linguistic_plan,
        ),
        *recovered_hs_surfaces,
    )
    jurisdiction_requirements = jurisdictional_surface_requirements(
        source_text, effective_target_label, target_integrity_resources
    )
    role_hints = source_semantic_role_hints(source_text, effective_target_label)
    leaves = rewrite_changed_leaves(source_label, effective_target_label)
    inline_slot_requirements = inline_slot_topology_requirements(source_text)
    status_requirements = source_status_preservation_requirements(source_text, leaves)
    auxiliary_identity_requirements = raw_auxiliary_identity_requirements(
        source_text, source_label, effective_target_label
    )
    operational_requirements = operational_flavor_requirements(
        source_text,
        source_label,
        effective_target_label,
        source_document_id=document_id,
        scenario_id=scenario_id,
        profiles=target_integrity_resources.operational_profiles,
        limits=capacity_limits(config.target_integrity.transport_capacity),
    )
    cargo_rewrite_requirements = cargo_flavor_rewrite_requirements(
        source_text, source_label, effective_target_label
    )
    cargo_rewrite_requirements = bind_cargo_package_surface_guards(
        cargo_rewrite_requirements,
        source_label,
        effective_target_label,
        target_integrity_resources.packages,
    )
    container_package_requirements = container_package_type_replacement_requirements(
        source_text,
        source_label,
        effective_target_label,
        target_integrity_resources.packages,
    )
    cargo_package_requirements = cargo_package_replacement_requirements(
        source_text,
        source_label,
        effective_target_label,
        target_integrity_resources.packages,
        cargo_rewrite_requirements,
        excluded_line_ids=frozenset(
            line_id
            for requirement in container_package_requirements
            for line_id in requirement.sourceLineIds
        )
        | frozenset(
            requirement.sourceLineId
            for requirement in operational_requirements
            if requirement.kind == "package_quantity"
        ),
    )
    package_quantity_owned_line_ids = frozenset(
        line_id
        for requirement in (*container_package_requirements, *cargo_package_requirements)
        for line_id in requirement.sourceLineIds
    ) | frozenset(
        requirement.sourceLineId
        for requirement in operational_requirements
        if requirement.kind == "package_quantity"
    )
    scalar_requirements = anchored_scalar_replacement_requirements(source_text, leaves)
    container_equipment_requirements = container_equipment_replacement_requirements(
        source_text, source_label, effective_target_label
    )
    measurement_requirements = anchored_measurement_replacement_requirements(
        source_text, source_label, effective_target_label
    )
    cargo_measurement_requirements = cargo_component_measurement_replacement_requirements(
        source_text,
        source_label,
        effective_target_label,
        cargo_rewrite_requirements,
        rendering_requirements,
    )
    package_quantity_requirements = anchored_package_quantity_replacement_requirements(
        source_text,
        leaves,
        excluded_line_ids=package_quantity_owned_line_ids,
    )
    aggregate_requirements = aggregate_operational_replacement_requirements(
        source_text,
        source_label,
        effective_target_label,
        operational_requirements,
        source_document_id=document_id,
        scenario_id=scenario_id,
        profiles=target_integrity_resources.operational_profiles,
        limits=capacity_limits(config.target_integrity.transport_capacity),
    )
    country_metadata_requirements = party_country_metadata_replacement_requirements(
        source_text, effective_target_label, target_integrity_resources
    )
    non_cargo_line_requirements = merge_anchored_scalar_replacement_requirements(
        scalar_requirements,
        container_equipment_requirements,
        container_package_requirements,
        cargo_package_requirements,
        measurement_requirements,
        cargo_measurement_requirements,
        package_quantity_requirements,
        aggregate_requirements,
        country_metadata_requirements,
    )
    deterministic_owned_line_ids = frozenset(
        line_id
        for requirement in non_cargo_line_requirements
        for line_id in requirement.sourceLineIds
    ) | frozenset(
        line_id for requirement in rendering_requirements for line_id in requirement.sourceLineIds
    ) | frozenset(
        line_id
        for requirement in jurisdiction_requirements
        for line_id in requirement.sourceLineIds
    ) | frozenset(requirement.sourceLineId for requirement in role_hints) | frozenset(
        requirement.sourceLineId for requirement in operational_requirements
    )
    exact_cargo_requirements = exact_cargo_line_replacement_requirements(
        source_text,
        source_label,
        cargo_rewrite_requirements,
        excluded_line_ids=deterministic_owned_line_ids,
    )
    anchored_replacement_requirements = merge_anchored_scalar_replacement_requirements(
        non_cargo_line_requirements,
        exact_cargo_requirements,
    )
    literal_requirements = target_literal_requirements(leaves, anchored_replacement_requirements)
    occurrence_requirements = target_value_occurrence_requirements(
        source_text,
        leaves,
        auxiliary_identity_requirements,
        rendering_requirements,
        anchored_replacement_requirements,
        defer_ambiguous_party_scalar_cardinality=(defer_ambiguous_party_scalar_cardinality),
    )
    cargo_rewrite_requirements = cargo_auxiliary_package_requirements(
        source_text,
        cargo_rewrite_requirements,
        excluded_line_ids=frozenset(
            line_id
            for requirement in anchored_replacement_requirements
            for line_id in requirement.sourceLineIds
        ),
    )
    if not any(row.requiresTextEdit for row in leaves):
        raise ValueError(f"rewrite case has no printable target changes: {document_id}")
    workspace = RewriteWorkspace(
        scenario_id=scenario_id,
        source_label=source_label,
        upstream_target_label=upstream_target_label,
        current_target_label=copy.deepcopy(effective_target_label),
        surface_requirements=rendering_requirements,
        target_literal_requirements=literal_requirements,
        target_value_occurrence_requirements=occurrence_requirements,
        anchored_scalar_replacement_requirements=anchored_replacement_requirements,
        source_role_hints=role_hints,
        inline_slot_requirements=inline_slot_requirements,
        source_status_requirements=status_requirements,
        jurisdictional_requirements=jurisdiction_requirements,
        raw_auxiliary_identity_requirements=auxiliary_identity_requirements,
        operational_flavor_requirements=operational_requirements,
        cargo_flavor_rewrite_requirements=cargo_rewrite_requirements,
        original_text=source_text,
        current_text=source_text,
    )
    apply_deterministic_prefills(workspace)
    return RewriteState(
        case_number=case_number,
        document_id=document_id,
        scenario_id=scenario_id,
        feature=dict(feature),
        source_label=source_label,
        target_integrity=target_integrity,
        workspace=workspace,
        started_at=datetime.now(UTC),
    )


def build_rewrite_contract_bundle(state: RewriteState) -> RewriteCycleBundle:
    """Freeze a prepared rewrite contract without pretending that a model pass occurred."""

    source_text = state.workspace.original_text
    output_text = state.workspace.current_text
    source_label = state.source_label
    upstream_target_label = state.workspace.upstream_target_label
    target_label = state.target_label
    changed = state.changed_leaves
    change_contract = label_change_contract(state.workspace)
    compact_contract = compact_label_change_contract(change_contract)
    result = RewriteCycleResult(
        schemaVersion=17,
        caseNumber=state.case_number,
        documentId=state.document_id,
        scenarioId=state.scenario_id,
        status="needs_review",
        reason="Host rewrite contract prepared; no editor or reviewer model has run.",
        sourceTextSha256=sha256_bytes(source_text.encode("utf-8")),
        outputTextSha256=sha256_bytes(output_text.encode("utf-8")),
        sourceLabelSha256=sha256_bytes(canonical_json_bytes(source_label)),
        upstreamTargetLabelSha256=sha256_bytes(canonical_json_bytes(upstream_target_label)),
        targetLabelSha256=sha256_bytes(canonical_json_bytes(target_label)),
        targetIntegritySha256=sha256_bytes(canonical_json_bytes(state.target_integrity)),
        targetCapacityReprojected=bool(state.target_integrity["capacity_changed"]),
        changedLeavesSha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in changed])
        ),
        labelChangeContractSha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in change_contract])
        ),
        editorOutputSchemaSha256=sha256_bytes(
            canonical_json_bytes(AtomicRewriteArguments.model_json_schema(mode="validation"))
        ),
        reviewerOutputSchemaSha256=sha256_bytes(
            canonical_json_bytes(SemanticReviewReceipt.model_json_schema(mode="validation"))
        ),
        correctionCycles=0,
        editorPasses=0,
        reviewPasses=0,
        compoundPartyFlavorRealizations=0,
        rawAuxiliaryIdentityRealizations=0,
        deterministicPrefills=len(state.workspace.deterministic_prefills),
        replacements=0,
        sourceLinesReplaced=0,
        outputLinesInserted=0,
        # A prepared contract is an immutable declaration, not an executed stage.  Persisting
        # wall-clock preparation time here made the contract hash change on every exact resume.
        durationMs=0.0,
        totalUsage=_empty_usage(),
    )
    return RewriteCycleBundle(
        schemaVersion=17,
        result=result,
        feature=state.feature,
        sourceLabel=source_label,
        upstreamTargetLabel=upstream_target_label,
        targetLabel=target_label,
        targetIntegrity=state.target_integrity,
        changedLeaves=changed,
        labelChangeContract=change_contract,
        compactLabelChangeContract=compact_contract,
        surfaceRenderingRequirements=state.workspace.surface_requirements,
        targetLiteralRequirements=state.workspace.target_literal_requirements,
        targetValueOccurrenceRequirements=state.workspace.target_value_occurrence_requirements,
        anchoredScalarReplacementRequirements=(
            state.workspace.anchored_scalar_replacement_requirements
        ),
        sourceSemanticRoleHints=state.workspace.source_role_hints,
        inlineSlotTopologyRequirements=state.workspace.inline_slot_requirements,
        sourceStatusPreservationRequirements=state.workspace.source_status_requirements,
        compoundPartyFlavorRequirements=compound_party_flavor_requirements(
            source_label, target_label
        ),
        rawAuxiliaryIdentityRequirements=state.workspace.raw_auxiliary_identity_requirements,
        operationalFlavorRequirements=state.workspace.operational_flavor_requirements,
        cargoFlavorRewriteRequirements=state.workspace.cargo_flavor_rewrite_requirements,
        jurisdictionalSurfaceRequirements=state.workspace.jurisdictional_requirements,
        sourceText=source_text,
        outputText=output_text,
        deterministicPrefills=tuple(state.workspace.deterministic_prefills),
        appliedReplacements=(),
        deterministicAudits=(),
        semanticReviews=(),
        stageRecords=(),
        diff=unified_text_diff(source_text, output_text),
    )


async def _run_case(
    *,
    case_number: int,
    source_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    linguistic_plan: DocumentLinguisticPlan,
    feature: Mapping[str, JsonValue],
    runtime: RewriteRuntime,
) -> RewriteCycleBundle:
    state = prepare_rewrite_state(
        case_number=case_number,
        source_row=source_row,
        target_row=target_row,
        linguistic_plan=linguistic_plan,
        feature=feature,
        config=runtime.config,
        target_integrity_resources=runtime.target_integrity_resources,
    )
    document_id = state.document_id
    scenario_id = state.scenario_id
    source_label = state.source_label
    source_text = state.workspace.original_text
    upstream_target_label = state.workspace.upstream_target_label
    status, reason = await _run_state(state, runtime)
    usage = _combine_usage([row.usage for row in state.stages]) if state.stages else _empty_usage()
    commits = [row.editorCommit for row in state.stages if row.editorCommit is not None]
    replacements = tuple(
        replacement for commit in commits for replacement in commit.appliedReplacements
    )
    effective_leaves = state.changed_leaves
    effective_change_contract = label_change_contract(state.workspace)
    compact_change_contract = compact_label_change_contract(effective_change_contract)
    flavor_requirements = compound_party_flavor_requirements(source_label, state.target_label)
    result = RewriteCycleResult(
        schemaVersion=17,
        caseNumber=case_number,
        documentId=document_id,
        scenarioId=scenario_id,
        status=status,
        reason=reason,
        sourceTextSha256=sha256_bytes(source_text.encode("utf-8")),
        outputTextSha256=state.workspace.current_sha256,
        sourceLabelSha256=sha256_bytes(canonical_json_bytes(source_label)),
        upstreamTargetLabelSha256=sha256_bytes(canonical_json_bytes(upstream_target_label)),
        targetLabelSha256=state.workspace.current_target_sha256,
        targetIntegritySha256=sha256_bytes(canonical_json_bytes(state.target_integrity)),
        targetCapacityReprojected=bool(state.target_integrity["capacity_changed"]),
        changedLeavesSha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in effective_leaves])
        ),
        labelChangeContractSha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in effective_change_contract])
        ),
        editorOutputSchemaSha256=sha256_bytes(
            canonical_json_bytes(AtomicRewriteArguments.model_json_schema(mode="validation"))
        ),
        reviewerOutputSchemaSha256=sha256_bytes(
            canonical_json_bytes(SemanticReviewReceipt.model_json_schema(mode="validation"))
        ),
        correctionCycles=sum(row.stage == "correction_editor" for row in state.stages),
        editorPasses=sum(row.stage != "semantic_reviewer" for row in state.stages),
        reviewPasses=sum(row.stage == "semantic_reviewer" for row in state.stages),
        compoundPartyFlavorRealizations=sum(
            len(commit.compoundPartyFlavorRealizations) for commit in commits
        ),
        rawAuxiliaryIdentityRealizations=sum(
            len(commit.rawAuxiliaryIdentityRealizations) for commit in commits
        ),
        deterministicPrefills=len(state.workspace.deterministic_prefills),
        replacements=len(replacements),
        sourceLinesReplaced=sum(row.sourceLinesReplaced for row in replacements),
        outputLinesInserted=sum(row.outputLinesInserted for row in replacements),
        durationMs=(datetime.now(UTC) - state.started_at).total_seconds() * 1000.0,
        totalUsage=usage,
    )
    bundle = RewriteCycleBundle(
        schemaVersion=17,
        result=result,
        feature=state.feature,
        sourceLabel=source_label,
        upstreamTargetLabel=upstream_target_label,
        targetLabel=state.target_label,
        targetIntegrity=state.target_integrity,
        changedLeaves=effective_leaves,
        labelChangeContract=effective_change_contract,
        compactLabelChangeContract=compact_change_contract,
        surfaceRenderingRequirements=state.workspace.surface_requirements,
        targetLiteralRequirements=state.workspace.target_literal_requirements,
        targetValueOccurrenceRequirements=(state.workspace.target_value_occurrence_requirements),
        anchoredScalarReplacementRequirements=(
            state.workspace.anchored_scalar_replacement_requirements
        ),
        sourceSemanticRoleHints=state.workspace.source_role_hints,
        inlineSlotTopologyRequirements=state.workspace.inline_slot_requirements,
        sourceStatusPreservationRequirements=state.workspace.source_status_requirements,
        compoundPartyFlavorRequirements=flavor_requirements,
        rawAuxiliaryIdentityRequirements=(state.workspace.raw_auxiliary_identity_requirements),
        operationalFlavorRequirements=(state.workspace.operational_flavor_requirements),
        cargoFlavorRewriteRequirements=(state.workspace.cargo_flavor_rewrite_requirements),
        jurisdictionalSurfaceRequirements=state.workspace.jurisdictional_requirements,
        sourceText=source_text,
        outputText=state.workspace.current_text,
        deterministicPrefills=tuple(state.workspace.deterministic_prefills),
        appliedReplacements=replacements,
        deterministicAudits=tuple(state.audits),
        semanticReviews=tuple(state.reviews),
        stageRecords=tuple(state.stages),
        diff=unified_text_diff(source_text, state.workspace.current_text),
    )
    runtime.staged.publish_json(
        f"cases/{case_number:02d}-{document_id}/bundle.json", bundle.model_dump(mode="json")
    )
    return bundle


def _load_bundle(path: Path) -> RewriteCycleBundle | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"rewrite bundle resume path is not a regular file: {path}")
    return RewriteCycleBundle.model_validate_json(read_regular_file_bytes(path), strict=True)


def build_target_integrity_resources(
    *,
    project_root: Path,
    config: RewritePreparationConfig,
    source_rows: Sequence[Mapping[str, Any]],
    staged: StagedArtifactRun | None = None,
) -> TargetIntegrityResources:
    """Load pinned registries and corpus profiles shared by all rewrite execution strategies."""

    iso_path = _resolve_pinned_file(
        project_root,
        config.target_integrity.iso3166_snapshot.path,
        config.target_integrity.iso3166_snapshot.sha256,
        label="target-integrity ISO-3166 snapshot",
    )
    package_path = _resolve_pinned_file(
        project_root,
        config.target_integrity.package_registry.path,
        config.target_integrity.package_registry.sha256,
        label="target-integrity package registry",
    )
    route_locations_path = _resolve_pinned_file(
        project_root,
        config.target_integrity.route_locations.path,
        config.target_integrity.route_locations.sha256,
        label="target-integrity UN/LOCODE locations",
    )
    customs_program_path = _resolve_pinned_file(
        project_root,
        config.target_integrity.customs_program_registry.path,
        config.target_integrity.customs_program_registry.sha256,
        label="target-integrity customs-program registry",
    )
    route_locations = load_pinned_unlocode_locations(
        route_locations_path,
        expected_sha256=config.target_integrity.route_locations.sha256,
        expected_records=config.target_integrity.route_location_records,
    )
    customs_programs = _load_customs_program_registry(
        customs_program_path,
        expected_entries=config.target_integrity.customs_program_registry_entries,
    )
    transport_limits = capacity_limits(config.target_integrity.transport_capacity)
    operational_profiles = build_empirical_operational_profiles(
        source_rows,
        target_field=config.inputs.source_target_field,
        limits=transport_limits,
    )
    if not operational_profiles:
        raise ValueError("pinned source corpus produced no empirical operational profiles")
    if staged is not None:
        staged.publish_json(
            "generation/operational-profile-support.json",
            {
                "schemaVersion": 1,
                "method": "empirical_capacity_utilization_and_same_family_tare_resample_v2",
                "profiles": len(operational_profiles),
                "documents": len({row.document_id for row in operational_profiles}),
                "grossWeightProfiles": sum(
                    row.gross_utilization is not None for row in operational_profiles
                ),
                "volumeProfiles": sum(
                    row.volume_utilization is not None or row.volume_m3 is not None
                    for row in operational_profiles
                ),
                "tareWeightProfiles": sum(
                    row.tare_weight_kg is not None for row in operational_profiles
                ),
                "equipmentFamilies": {
                    family: sum(row.equipment_family == family for row in operational_profiles)
                    for family in sorted({row.equipment_family for row in operational_profiles})
                },
            },
        )
    return TargetIntegrityResources(
        countries=load_iso_country_registry(
            iso_path=iso_path,
            iso_sha256=config.target_integrity.iso3166_snapshot.sha256,
        ),
        packages=load_package_registry(
            package_path,
            expected_sha256=config.target_integrity.package_registry.sha256,
            expected_entries=config.target_integrity.package_registry_entries,
        ),
        route_countries_by_name=_route_country_index(route_locations),
        route_port_countries_by_name=_route_country_index(route_locations, required_function="1"),
        customs_programs=customs_programs.entries,
        operational_profiles=operational_profiles,
    )


async def _run_cases(
    *,
    selected: Sequence[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            DocumentLinguisticPlan,
            Mapping[str, JsonValue],
        ]
    ],
    source_rows: Sequence[Mapping[str, Any]],
    editor_prompt: str,
    reviewer_prompt: str,
    staged: StagedArtifactRun,
    config: SynthesisRawTextRewriteCycleProbeConfig,
    project_root: Path,
) -> tuple[RewriteCycleBundle, ...]:
    target_integrity_resources = build_target_integrity_resources(
        project_root=project_root,
        config=config,
        source_rows=source_rows,
        staged=staged,
    )
    editor_provider = config.providers.editor
    reviewer_provider = config.providers.reviewer
    key = load_provider_key(
        project_root,
        config.environment_file,
        editor_provider.api_key_env,
    )
    client = AsyncOpenAI(
        api_key=key,
        base_url=(_OPENROUTER_BASE_URL if editor_provider.kind == "openrouter" else None),
        max_retries=max(
            editor_provider.transport_max_retries,
            reviewer_provider.transport_max_retries,
        ),
        timeout=max(
            editor_provider.request_timeout_seconds,
            reviewer_provider.request_timeout_seconds,
        ),
        default_headers=(
            {"X-Title": "DocumentParsing"} if editor_provider.kind == "openrouter" else None
        ),
    )
    if editor_provider.kind == "openrouter":
        openrouter_provider = OpenRouterProvider(openai_client=client)
        editor_model: Model = OpenRouterModel(
            editor_provider.model,
            provider=openrouter_provider,
        )
        reviewer_model: Model = OpenRouterModel(
            reviewer_provider.model,
            provider=openrouter_provider,
        )
    else:
        openai_provider = OpenAIProvider(openai_client=client)
        editor_model = OpenAIResponsesModel(
            editor_provider.model,
            provider=openai_provider,
        )
        reviewer_model = OpenAIResponsesModel(
            reviewer_provider.model,
            provider=openai_provider,
        )
    if not editor_model.profile.get("supports_tools", False):
        raise ValueError(f"rewrite editor model does not support tools: {editor_provider.model}")
    if not reviewer_model.profile.get("supports_tools", False):
        raise ValueError(
            f"rewrite reviewer model does not support tools: {reviewer_provider.model}"
        )
    limiter = ConcurrencyLimiter(
        config.workflow.max_concurrent_requests,
        name="synthetic-raw-text-atomic-rewrite-provider-requests",
    )
    runtime = RewriteRuntime(
        config=config,
        editor_prompt=editor_prompt,
        reviewer_prompt=reviewer_prompt,
        editor_model=editor_model,
        reviewer_model=reviewer_model,
        target_integrity_resources=target_integrity_resources,
        limiter=limiter,
        staged=staged,
    )
    case_limiter = asyncio.Semaphore(config.workflow.max_concurrent_cases)
    progress_lock = asyncio.Lock()
    results: list[RewriteCycleBundle | None] = [None] * len(selected)
    processed = 0
    started = time.perf_counter()

    async def run_one(index: int) -> None:
        nonlocal processed
        source, target, linguistic_plan, feature = selected[index]
        document_id = cast(str, source["documentId"])
        existing = _load_bundle(
            staged.stage_root / f"cases/{index + 1:02d}-{document_id}/bundle.json"
        )
        if existing is not None:
            results[index] = existing
        else:
            async with case_limiter:
                results[index] = await _run_case(
                    case_number=index + 1,
                    source_row=source,
                    target_row=target,
                    linguistic_plan=linguistic_plan,
                    feature=feature,
                    runtime=runtime,
                )
        async with progress_lock:
            processed += 1
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            print(
                json.dumps(
                    {
                        "command": "run-raw-text-rewrite-cycle-probe",
                        "phase": "atomic_edit_review_correct",
                        "processed_cases": processed,
                        "remaining_cases": len(selected) - processed,
                        "quality_validated_cases": sum(
                            row is not None and row.result.status == "quality_validated"
                            for row in results
                        ),
                        "elapsed_seconds": round(elapsed, 3),
                        "throughput_cases_per_hour": round(rate * 3600.0, 3),
                        "eta_seconds": round((len(selected) - processed) / rate, 3)
                        if rate
                        else None,
                        "status": "progress",
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    try:
        await asyncio.gather(*(run_one(index) for index in range(len(selected))))
    finally:
        await client.close()
    if any(row is None for row in results):
        raise RuntimeError("atomic rewrite worker pool returned an incomplete result set")
    return tuple(cast(RewriteCycleBundle, row) for row in results)


def _publish_case_files(staged: StagedArtifactRun, bundle: RewriteCycleBundle) -> None:
    prefix = f"cases/{bundle.result.caseNumber:02d}-{bundle.result.documentId}"
    payloads = {
        "source-label.json": json_artifact_bytes(bundle.sourceLabel),
        "upstream-target-label.json": json_artifact_bytes(bundle.upstreamTargetLabel),
        "target-label.json": json_artifact_bytes(bundle.targetLabel),
        "target-integrity.json": json_artifact_bytes(bundle.targetIntegrity),
        "changed-leaves.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.changedLeaves]
        ),
        "label-change-contract.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.labelChangeContract]
        ),
        "model-change-contract.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.compactLabelChangeContract]
        ),
        "surface-rendering-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.surfaceRenderingRequirements]
        ),
        "target-literal-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.targetLiteralRequirements]
        ),
        "target-value-occurrence-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.targetValueOccurrenceRequirements]
        ),
        "anchored-scalar-replacement-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.anchoredScalarReplacementRequirements]
        ),
        "deterministic-prefills.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.deterministicPrefills]
        ),
        "source-semantic-role-hints.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.sourceSemanticRoleHints]
        ),
        "inline-slot-topology-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.inlineSlotTopologyRequirements]
        ),
        "source-status-preservation-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.sourceStatusPreservationRequirements]
        ),
        "compound-party-flavor-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.compoundPartyFlavorRequirements]
        ),
        "raw-auxiliary-identity-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.rawAuxiliaryIdentityRequirements]
        ),
        "operational-flavor-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.operationalFlavorRequirements]
        ),
        "cargo-flavor-rewrite-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.cargoFlavorRewriteRequirements]
        ),
        "jurisdictional-surface-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.jurisdictionalSurfaceRequirements]
        ),
        "raw-before.txt": bundle.sourceText.encode("utf-8"),
        "raw-after.txt": bundle.outputText.encode("utf-8"),
        "diff.patch": bundle.diff.encode("utf-8"),
        "replacements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.appliedReplacements]
        ),
        "deterministic-audits.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.deterministicAudits]
        ),
        "semantic-reviews.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.semanticReviews]
        ),
        "stages.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.stageRecords]
        ),
        "result.json": json_artifact_bytes(bundle.result.model_dump(mode="json")),
    }
    for name, payload in payloads.items():
        staged.publish_bytes(f"{prefix}/{name}", payload)


def _fence(value: str, language: str) -> str:
    fence = "~~~~"
    while fence in value:
        fence += "~"
    return f"{fence}{language}\n{value}\n{fence}"


def _format_provider_cost(value: Decimal | str | None) -> str:
    return "unavailable" if value is None else f"${value}"


def _report(summary: Mapping[str, Any], bundles: Sequence[RewriteCycleBundle]) -> str:
    provider_cost = summary["providerReportedCostUsd"]
    provider_cost_per_thousand = summary["providerReportedCostUsdPerThousand"]
    lines = [
        "# Atomic synthetic B/L raw-text rewrite probe",
        "",
        "## Outcome",
        "",
        f"- Quality validated: **{summary['qualityValidatedCases']}/{summary['cases']}**",
        f"- Needs review / call failures: **{summary['needsReviewCases']} / "
        f"{summary['callFailedCases']}**",
        f"- Requests: **{summary['requests']}**",
        f"- Estimated list-price cost: **${summary['estimatedCostUsd']}**",
        f"- Provider-reported billed cost: **{_format_provider_cost(provider_cost)}**",
        f"- Estimated list-price cost per 1,000 attempts: "
        f"**${summary['estimatedCostUsdPerThousand']}**",
        f"- Provider-reported cost per 1,000 attempts: "
        f"**{_format_provider_cost(provider_cost_per_thousand)}**",
        f"- Provider-reported cost per 1,000 validated outputs: "
        f"**{_format_provider_cost(summary['providerReportedCostUsdPerThousandValidated'])}**",
        f"- Input / cache-read / cache-write tokens: **{summary['inputTokens']} / "
        f"{summary['cacheReadTokens']} / {summary['cacheWriteTokens']}**",
        f"- Visible / reasoning output tokens: **{summary['visibleOutputTokens']} / "
        f"{summary['reasoningTokens']}**",
        f"- Wall time / throughput: **{summary['wallSeconds']:.3f}s / "
        f"{summary['throughputCasesPerHour']:.3f} cases/hour**",
        "",
        "Estimated cost applies the pinned rate card to provider token receipts. When available, "
        "provider-reported billed cost is the gateway's per-response charge; the account billing "
        "dashboard remains authoritative and may post asynchronously.",
        "",
        "The editor uses one strict terminal output function per pass. Its compact line-range ",
        "patch is applied atomically in-process and is never replayed to the model. A separate ",
        "provider-constrained reviewer can request a bounded correction. Training records remain ",
        "unpublished.",
        "",
        "## Usage by stage",
        "",
        "| Stage | Requests | Input | Cache read | Cache write | Reasoning | Visible | "
        "Estimated list cost | Provider cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, usage in summary["usageByStage"].items():
        lines.append(
            f"| `{stage}` | {usage['requests']} | {usage['inputTokens']} | "
            f"{usage['cacheReadTokens']} | {usage['cacheWriteTokens']} | "
            f"{usage['reasoningTokens']} | {usage['visibleOutputTokens']} | "
            f"${usage['estimatedCostUsd']} | "
            f"{_format_provider_cost(usage['providerReportedCostUsd'])} |"
        )
    lines.extend(
        [
            "",
            "## Case index",
            "",
            "| Case | Document | Status | Editor/review/correction | Requests | Input | "
            "Reasoning | "
            "Visible | Estimated cost | Provider cost |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bundle in bundles:
        row = bundle.result
        lines.append(
            f"| {row.caseNumber} | `{row.documentId}` | {row.status} | "
            f"{row.editorPasses}/{row.reviewPasses}/{row.correctionCycles} | "
            f"{row.totalUsage.requests} | {row.totalUsage.inputTokens} | "
            f"{row.totalUsage.reasoningTokens} | {row.totalUsage.visibleOutputTokens} | "
            f"${row.totalUsage.estimatedCostUsd} | "
            f"{_format_provider_cost(row.totalUsage.providerReportedCostUsd)} |"
        )
    for bundle in bundles:
        row = bundle.result
        lines.extend(
            [
                "",
                f"## Case {row.caseNumber}: `{row.documentId}`",
                "",
                f"- Status: **{row.status}** — {row.reason}",
                f"- Line ranges / source lines / output lines: **{row.replacements} / "
                f"{row.sourceLinesReplaced} / {row.outputLinesInserted}**",
                f"- Deterministic scalar prefills: **{row.deterministicPrefills}**",
                f"- Requests / estimated cost: **{row.totalUsage.requests} / "
                f"${row.totalUsage.estimatedCostUsd}**",
                f"- Provider-reported cost: "
                f"**{_format_provider_cost(row.totalUsage.providerReportedCostUsd)}**",
                f"- Retained provider call trace: "
                f"`cases/{row.caseNumber:02d}-{row.documentId}/stages.json`",
                "",
                "### Stage usage",
                "",
                "| Sequence | Stage | Requests | Input | Reasoning | Visible | "
                "Estimated | Provider |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {stage.sequence} | `{stage.stage}` | {stage.usage.requests} | "
                    f"{stage.usage.inputTokens} | {stage.usage.reasoningTokens} | "
                    f"{stage.usage.visibleOutputTokens} | ${stage.usage.estimatedCostUsd} |"
                    f" {_format_provider_cost(stage.usage.providerReportedCostUsd)} |"
                    for stage in bundle.stageRecords
                ],
                "",
                "### Reviews",
                "",
                _fence(
                    json.dumps(
                        [review.model_dump(mode="json") for review in bundle.semanticReviews],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "json",
                ),
                "",
                "### Authoritative label-change contract",
                "",
                _fence(
                    json.dumps(
                        [
                            directive.model_dump(mode="json")
                            for directive in bundle.labelChangeContract
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "json",
                ),
                "",
                "### Compact model-facing change contract",
                "",
                _fence(
                    json.dumps(
                        [
                            directive.model_dump(mode="json")
                            for directive in bundle.compactLabelChangeContract
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "json",
                ),
                "",
                "### Exact surface, semantic-role, compound-flavor, and jurisdiction requirements",
                "",
                _fence(
                    json.dumps(
                        {
                            "surfaceRenderingRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.surfaceRenderingRequirements
                            ],
                            "targetLiteralRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.targetLiteralRequirements
                            ],
                            "targetValueOccurrenceRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.targetValueOccurrenceRequirements
                            ],
                            "anchoredScalarReplacementRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.anchoredScalarReplacementRequirements
                            ],
                            "sourceSemanticRoleHints": [
                                hint.model_dump(mode="json")
                                for hint in bundle.sourceSemanticRoleHints
                            ],
                            "inlineSlotTopologyRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.inlineSlotTopologyRequirements
                            ],
                            "sourceStatusPreservationRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.sourceStatusPreservationRequirements
                            ],
                            "compoundPartyFlavorRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.compoundPartyFlavorRequirements
                            ],
                            "rawAuxiliaryIdentityRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.rawAuxiliaryIdentityRequirements
                            ],
                            "operationalFlavorRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.operationalFlavorRequirements
                            ],
                            "cargoFlavorRewriteRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.cargoFlavorRewriteRequirements
                            ],
                            "jurisdictionalSurfaceRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.jurisdictionalSurfaceRequirements
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "json",
                ),
                "",
                "### Target-integrity receipt",
                "",
                _fence(
                    json.dumps(bundle.targetIntegrity, ensure_ascii=False, indent=2),
                    "json",
                ),
                "",
                "### Source label",
                "",
                _fence(json.dumps(bundle.sourceLabel, ensure_ascii=False, indent=2), "json"),
                "",
                "### Synthetic target label",
                "",
                "The upstream target is shown first. Target-integrity checks may reproject "
                "physically incompatible measures, but auxiliary compound identities remain "
                "raw-text flavor and never alter the effective training target.",
                "",
                "#### Upstream target",
                "",
                _fence(
                    json.dumps(bundle.upstreamTargetLabel, ensure_ascii=False, indent=2),
                    "json",
                ),
                "",
                "#### Effective training target",
                "",
                _fence(json.dumps(bundle.targetLabel, ensure_ascii=False, indent=2), "json"),
                "",
                "### Source raw OCR",
                "",
                _fence(bundle.sourceText, "text"),
                "",
                "### Final raw OCR",
                "",
                _fence(bundle.outputText, "text"),
                "",
                "### Exact diff",
                "",
                _fence(bundle.diff or "(no diff)", "diff"),
            ]
        )
    return "\n".join(lines) + "\n"


def _plot_bytes(bundles: Sequence[RewriteCycleBundle]) -> dict[str, bytes]:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd  # type: ignore[import-untyped]
    import seaborn as sns  # type: ignore[import-untyped]

    rows = [
        {
            "case": row.result.caseNumber,
            "status": row.result.status,
            "cost_usd": float(
                row.result.totalUsage.providerReportedCostUsd
                if row.result.totalUsage.providerReportedCostUsd is not None
                else row.result.totalUsage.estimatedCostUsd
            ),
            "requests": row.result.totalUsage.requests,
            "visible_output": row.result.totalUsage.visibleOutputTokens,
            "reasoning": row.result.totalUsage.reasoningTokens,
            "input": row.result.totalUsage.inputTokens,
            "seconds": row.result.durationMs / 1000.0,
        }
        for row in bundles
    ]
    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="notebook")
    plots: dict[str, bytes] = {}

    def render(name: str, figure: Any) -> None:
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=170, bbox_inches="tight")
        plots[name] = stream.getvalue()
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(data=frame, x="case", y="cost_usd", hue="status", dodge=False, ax=axes[0])
    sns.barplot(data=frame, x="case", y="seconds", hue="status", dodge=False, ax=axes[1])
    axes[0].set(title="Estimated cost by case", xlabel="case", ylabel="USD")
    axes[1].set(title="Wall time by case", xlabel="case", ylabel="seconds")
    render("01_cost_and_duration.png", figure)

    token_frame = frame.melt(
        id_vars=("case",),
        value_vars=("input", "visible_output", "reasoning"),
        var_name="token_kind",
        value_name="tokens",
    )
    figure, axis = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=token_frame, x="case", y="tokens", hue="token_kind", ax=axis)
    axis.set(title="Provider token usage", xlabel="case", ylabel="tokens")
    render("02_tokens.png", figure)

    stage_rows = [
        {
            "case": bundle.result.caseNumber,
            "stage": record.stage,
            "reasoning": record.usage.reasoningTokens,
            "visible_output": record.usage.visibleOutputTokens,
            "cost_usd": float(
                record.usage.providerReportedCostUsd
                if record.usage.providerReportedCostUsd is not None
                else record.usage.estimatedCostUsd
            ),
        }
        for bundle in bundles
        for record in bundle.stageRecords
    ]
    stage_frame = pd.DataFrame(stage_rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    stage_tokens = stage_frame.melt(
        id_vars=("stage",),
        value_vars=("reasoning", "visible_output"),
        var_name="token_kind",
        value_name="tokens",
    )
    sns.barplot(
        data=stage_tokens, x="stage", y="tokens", hue="token_kind", estimator="sum", ax=axes[0]
    )
    sns.barplot(data=stage_frame, x="stage", y="cost_usd", estimator="sum", ax=axes[1])
    axes[0].set(title="Output tokens by stage", xlabel="stage", ylabel="tokens")
    axes[1].set(title="Estimated list cost by stage", xlabel="stage", ylabel="USD")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].tick_params(axis="x", rotation=15)
    render("03_stage_usage.png", figure)
    return plots


def _artifact_inventory(root: Path) -> list[str]:
    rows: list[str] = []
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directory_names):
            raise ValueError("rewrite artifact tree contains a symlink directory")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError("rewrite artifact tree contains a symlink file")
            relative = path.relative_to(root).as_posix()
            if relative not in {"_TRANSACTION.json", "_COMMIT.json"}:
                rows.append(relative)
    return sorted(rows)


def run_raw_text_rewrite_cycle_probe(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextRewriteCycleProbeConfig,
) -> dict[str, JsonValue]:
    linguistic_root = _validate_linguistic_run(project_root, config)
    results_path = _resolve_pinned_file(
        project_root,
        config.inputs.linguistic_results.path,
        config.inputs.linguistic_results.sha256,
        label="linguistic results",
    )
    targets_path = _resolve_pinned_file(
        project_root,
        config.inputs.synthetic_targets.path,
        config.inputs.synthetic_targets.sha256,
        label="linguistic targets",
    )
    summary_path = _resolve_pinned_file(
        project_root,
        config.inputs.linguistic_summary.path,
        config.inputs.linguistic_summary.sha256,
        label="linguistic summary",
    )
    plan_path = linguistic_root / "planning/document-plans.jsonl"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError(f"linguistic completion run has no regular document plan: {plan_path}")
    plan_path = plan_path.resolve(strict=True)
    for path in (results_path, targets_path, summary_path, plan_path):
        if not path.is_relative_to(linguistic_root):
            raise ValueError("linguistic artifact is outside the pinned committed run")
    source_path = _resolve_pinned_file(
        project_root,
        config.inputs.source_corpus.path,
        config.inputs.source_corpus.sha256,
        label="source corpus",
    )
    feature_path = _resolve_pinned_file(
        project_root,
        config.inputs.document_features.path,
        config.inputs.document_features.sha256,
        label="document features",
    )
    editor_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.editor.path,
        config.prompts.editor.sha256,
        label="rewrite editor prompt",
    )
    reviewer_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.reviewer.path,
        config.prompts.reviewer.sha256,
        label="rewrite reviewer prompt",
    )
    editor_prompt_bytes = read_regular_file_bytes(editor_prompt_path)
    reviewer_prompt_bytes = read_regular_file_bytes(reviewer_prompt_path)
    editor_prompt = editor_prompt_bytes.decode("utf-8")
    reviewer_prompt = reviewer_prompt_bytes.decode("utf-8")

    results = _load_jsonl(
        results_path,
        records=config.inputs.linguistic_results.records,
        key="baseDocumentId",
        label="linguistic results",
    )
    targets = _load_jsonl(
        targets_path,
        records=config.inputs.synthetic_targets.records,
        key="baseDocumentId",
        label="linguistic targets",
    )
    raw_plans = _load_jsonl(
        plan_path,
        records=config.inputs.linguistic_results.records,
        key="baseDocumentId",
        label="linguistic document plans",
    )
    plans = tuple(
        DocumentLinguisticPlan.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in raw_plans
    )
    sources = _load_jsonl(
        source_path,
        records=config.inputs.source_corpus.records,
        key="documentId",
        label="source corpus",
    )
    features = _load_jsonl(
        feature_path,
        records=config.inputs.document_features.records,
        key="document_id",
        label="document features",
    )
    result_by_id = {row["baseDocumentId"]: row for row in results}
    target_by_id = {row["baseDocumentId"]: row for row in targets}
    plan_by_id = {row.baseDocumentId: row for row in plans}
    source_by_id = {row["documentId"]: row for row in sources}
    feature_by_id = {row["document_id"]: row for row in features}
    if tuple(result_by_id) != tuple(target_by_id) or tuple(result_by_id) != tuple(plan_by_id):
        raise ValueError("linguistic result, target, and document-plan order differs")
    for document_id, result in result_by_id.items():
        target = target_by_id[document_id]
        plan = plan_by_id[document_id]
        if result.get("status") != "success" or result.get("target") != target.get("target"):
            raise ValueError(f"linguistic result/target mismatch for {document_id}")
        if target.get("scenarioId") != plan.scenarioId:
            raise ValueError(f"linguistic plan/target scenario mismatch for {document_id}")
        target_sha = sha256_bytes(canonical_json_bytes(target["target"]))
        if target.get("targetSha256") != target_sha or result.get("targetSha256") != target_sha:
            raise ValueError(f"linguistic target SHA-256 mismatch for {document_id}")

    selected: list[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            DocumentLinguisticPlan,
            Mapping[str, JsonValue],
        ]
    ] = []
    for case in config.cases:
        try:
            source = source_by_id[case.document_id]
            target = target_by_id[case.document_id]
            plan = plan_by_id[case.document_id]
            feature = feature_by_id[case.document_id]
        except KeyError as error:
            raise ValueError(
                f"rewrite case is absent from pinned inputs: {case.document_id}"
            ) from error
        raw_text = source.get("joinedRawText")
        if not isinstance(raw_text, str) or source.get("joinedRawTextSha256") != sha256_bytes(
            raw_text.encode("utf-8")
        ):
            raise ValueError(f"source raw text fails its SHA-256: {case.document_id}")
        selected.append((source, target, plan, cast(Mapping[str, JsonValue], feature)))

    editor_schema = AtomicRewriteArguments.model_json_schema(mode="validation")
    reviewer_schema = SemanticReviewReceipt.model_json_schema(mode="validation")
    transaction = {
        "schemaVersion": 17,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "linguisticRunTransactionSha256": (
            config.inputs.linguistic_completion_run.transaction_sha256
        ),
        "linguisticResultsSha256": config.inputs.linguistic_results.sha256,
        "linguisticDocumentPlansSha256": sha256_file(plan_path),
        "syntheticTargetsSha256": config.inputs.synthetic_targets.sha256,
        "sourceCorpusSha256": config.inputs.source_corpus.sha256,
        "documentFeaturesSha256": config.inputs.document_features.sha256,
        "iso3166SnapshotSha256": config.target_integrity.iso3166_snapshot.sha256,
        "routeLocationsSha256": config.target_integrity.route_locations.sha256,
        "customsProgramRegistrySha256": (config.target_integrity.customs_program_registry.sha256),
        "packageRegistrySha256": config.target_integrity.package_registry.sha256,
        "editorPromptSha256": config.prompts.editor.sha256,
        "reviewerPromptSha256": config.prompts.reviewer.sha256,
        "editorSchemaSha256": sha256_bytes(canonical_json_bytes(editor_schema)),
        "reviewerSchemaSha256": sha256_bytes(canonical_json_bytes(reviewer_schema)),
        "caseDocumentIds": [row.document_id for row in config.cases],
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "sourceSchemaImplementationSha256": sha256_file(
            Path(bill_of_lading_v3.__file__).resolve(strict=True)
        ),
        "targetSchemaImplementationSha256": sha256_file(
            Path(bill_of_lading_v5.__file__).resolve(strict=True)
        ),
        "runtime": {
            "editorProviderKind": config.providers.editor.kind,
            "editorModel": config.providers.editor.model,
            "editorReasoningEffort": config.providers.editor.reasoning_effort,
            "reviewerProviderKind": config.providers.reviewer.kind,
            "reviewerModel": config.providers.reviewer.model,
            "reviewerReasoningEffort": config.providers.reviewer.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
        },
    }
    staged = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if staged.completed:
        return cast(
            dict[str, JsonValue],
            json.loads(read_regular_file_bytes(staged.final_root / "generation/summary.json")),
        )
    staged.recover_interrupted_temporary_files()
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    staged.publish_bytes("prompts/editor.md", editor_prompt_bytes)
    staged.publish_bytes("prompts/reviewer.md", reviewer_prompt_bytes)
    staged.publish_bytes(
        "schema/line-range-replacement.schema.json", json_artifact_bytes(editor_schema)
    )
    staged.publish_bytes("schema/semantic-review.schema.json", json_artifact_bytes(reviewer_schema))
    started = time.perf_counter()
    bundles = asyncio.run(
        _run_cases(
            selected=selected,
            source_rows=tuple(sources),
            editor_prompt=editor_prompt,
            reviewer_prompt=reviewer_prompt,
            staged=staged,
            config=config,
            project_root=project_root,
        )
    )
    wall_seconds = time.perf_counter() - started
    for bundle in bundles:
        _publish_case_files(staged, bundle)
    total_usage = _combine_usage([row.result.totalUsage for row in bundles])
    usage_by_stage = _usage_by_stage(bundles)
    estimated_cost_per_thousand = total_usage.estimatedCostUsd / Decimal(len(bundles)) * 1000
    provider_cost_per_thousand = (
        total_usage.providerReportedCostUsd / Decimal(len(bundles)) * 1000
        if total_usage.providerReportedCostUsd is not None
        else None
    )
    quality_validated_cases = sum(row.result.status == "quality_validated" for row in bundles)
    provider_cost_per_thousand_validated = (
        total_usage.providerReportedCostUsd / Decimal(quality_validated_cases) * 1000
        if total_usage.providerReportedCostUsd is not None and quality_validated_cases
        else None
    )
    downstream_provider_counts = Counter(total_usage.downstreamProviders)
    summary: dict[str, JsonValue] = {
        "schemaVersion": 17,
        "runId": config.run.run_id,
        "status": "complete",
        "cases": len(bundles),
        "qualityValidatedCases": quality_validated_cases,
        "needsReviewCases": sum(row.result.status == "needs_review" for row in bundles),
        "callFailedCases": sum(row.result.status == "call_failed" for row in bundles),
        "costLimitedCases": sum(row.result.status == "cost_limited" for row in bundles),
        "editorPasses": sum(row.result.editorPasses for row in bundles),
        "reviewPasses": sum(row.result.reviewPasses for row in bundles),
        "correctionCycles": sum(row.result.correctionCycles for row in bundles),
        "requests": total_usage.requests,
        "inputTokens": total_usage.inputTokens,
        "cacheReadTokens": total_usage.cacheReadTokens,
        "cacheWriteTokens": total_usage.cacheWriteTokens,
        "outputTokens": total_usage.outputTokens,
        "reasoningTokens": total_usage.reasoningTokens,
        "visibleOutputTokens": total_usage.visibleOutputTokens,
        "estimatedCostUsd": str(total_usage.estimatedCostUsd),
        "estimatedCostUsdPerThousand": str(estimated_cost_per_thousand),
        "providerReportedCostUsd": (
            str(total_usage.providerReportedCostUsd)
            if total_usage.providerReportedCostUsd is not None
            else None
        ),
        "providerReportedCostUsdPerThousand": (
            str(provider_cost_per_thousand) if provider_cost_per_thousand is not None else None
        ),
        "providerReportedCostUsdPerThousandValidated": (
            str(provider_cost_per_thousand_validated)
            if provider_cost_per_thousand_validated is not None
            else None
        ),
        "downstreamProviderRequestCounts": cast(
            JsonValue,
            dict(sorted(downstream_provider_counts.items())),
        ),
        "usageByStage": cast(
            JsonValue,
            {stage: usage.model_dump(mode="json") for stage, usage in usage_by_stage.items()},
        ),
        "wallSeconds": wall_seconds,
        "throughputCasesPerHour": len(bundles) / wall_seconds * 3600.0,
        "peakResidentMemoryMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "trainingRecordsPublished": False,
        "terminalOutputFunction": True,
        "lineAddressedAtomicPatches": True,
    }
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(
            canonical_json_bytes(row.result.model_dump(mode="json")) + b"\n" for row in bundles
        ),
    )
    staged.publish_json("generation/summary.json", summary)
    for name, payload in _plot_bytes(bundles).items():
        staged.publish_bytes(f"plots/{name}", payload)
    staged.publish_bytes("REPORT.md", _report(summary, bundles).encode("utf-8"))
    commit = staged.commit(
        expected_artifacts=_artifact_inventory(staged.stage_root),
        metadata={"cases": len(bundles), "schema_version": 15, "training_records": 0},
    )
    summary["commitContentSha256"] = commit.receipt.content_sha256
    return summary
