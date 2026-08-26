"""Deterministic policy checks for model-authored semantic-review findings."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import TypeAdapter, ValidationError

from document_ocr.label_schemas.mpci_bill_of_lading import ContainerIdentifier, ImoNumber
from document_ocr.label_schemas.semantic_review import SemanticReviewFinding
from document_ocr.labeling_agents.corrections import (
    TargetScopedCorrectionError,
    normalize_correction_paths,
)
from document_ocr.labeling_agents.work_items import WorkItemError


class ReviewPolicyError(WorkItemError):
    """A review finding demands a value outside the frozen target semantics."""


_CONTAINER_CANDIDATE = re.compile(r"(?i)(?<![A-Z0-9])[A-Z]{4}[ /-]?[0-9]{6}[ /-]?[0-9](?![0-9])")
_CONTAINER_ADAPTER: TypeAdapter[ContainerIdentifier] = TypeAdapter(ContainerIdentifier)
_IMO_CANDIDATE = re.compile(r"(?<![0-9])[0-9]{7}(?![0-9])")
_IMO_ADAPTER: TypeAdapter[ImoNumber] = TypeAdapter(ImoNumber)
_SUPPORTED_FREIGHT = re.compile(
    r"(?ix)\b(?:freight\s+)?(?:prepaid|collect|third[- ]party|payable\s+elsewhere)\b"
)
_UNSUPPORTED_FREIGHT_TERM = re.compile(
    r"(?ix)\b(?:free\s+in\s+free\s+out|f\.?i\.?o\.?|liner\s+terms?|as\s+arranged)\b"
)
_GOODS_ORIGIN = re.compile(
    r"(?ix)\b(?:country\s+of\s+origin|origin\s+of\s+(?:the\s+)?goods|goods\s+origin|"
    r"product\s+of|made\s+in)\b|(?m:^\s*origin\s*[:\-])"
)
_PARTY_COUNTRY = re.compile(
    r"(?ix)\b(?:exporter|shipper|consignee|importer|agent)\s+(?:country|contry|origin)\b"
)
_SUPPORTED_REFERENCE = re.compile(
    r"(?ix)\b(?:forward(?:ing)?|export\s+(?:ref(?:erence)?s?|no|number)|"
    r"ref\s*\.\s*exp\s*\.?|"
    r"aes|caed(?:\s*(?:no|number))?|shipping\s+bill|s/?bill|"
    r"du-e|f\s*/\s*agent(?:\s+name)?\s*&\s*ref|"
    r"s\.?b\.?(?:\s*(?:no|number))?|"
    r"ed\s*(?:no|number)\.?\s*(?=[:#-]|\s|$)|"
    r"itn\s*(?=[:#-]|\s)|imp\s*/+\s*exp\s*[#]?|"
    r"exp\s*(?=[:#-]?\s*[0-9])|"
    r"p\s*/?\s*i\s*(?:no|number|ref)|"
    r"inv(?:oice)?\.?\s*(?:(?:no|number|ref)\.?\s*(?=[:#-]|\s|$)|"
    r"(?=[:#-])|(?=[A-Z0-9._/-]*[0-9])))"
)
_EXCLUDED_ID_REFERENCE = re.compile(
    r"(?ix)\b(?:(?:exp(?:orter)?|imp(?:orter)?)\s*(?:id|identification|registration)|"
    r"acid|ern(?:\s*(?:no|number))?|tax(?:ation)?(?:\s*(?:id|no|number))?|vat|cnpj|"
    r"customs?(?:\s*(?:id|no|number))?|gpc|registration(?:\s*(?:id|no|number))?)\b"
)
_NON_VALUE_REFERENCE_HEADING = re.compile(
    r"(?ix)\A\s*(?:(?:export\s+references?|svc\s+contract|service\s+contract|"
    r"booking(?:\s*(?:no|number|ref))?|shipper(?:'s)?\s+ref(?:erence)?)"
    r"\s*[:#-]?\s*)+\Z"
)
_INVALID_CONTAINER_ASSERTION = re.compile(
    r"(?ix)\b(?:invalid|not\s+valid|fails?|incorrect|bad)\b.{0,40}\b(?:iso|check\s*digit|"
    r"container\s+(?:number|identifier))\b|\b(?:iso|check\s*digit)\b.{0,40}\b(?:invalid|"
    r"fails?|incorrect|bad)\b"
)
_CONTAINER_CORRECTION_TARGET = re.compile(
    r"^documentPatch\.containers(?:\[[0-9]+\])?(?:\.containerNumber)?$"
)
_PHONE_DIGITS_ONLY_ASSERTION = re.compile(
    r"(?ix)\b(?:non[- ]digit\s+character|digits?\s+only|must\s+contain\s+(?:only\s+)?"
    r"(?:printed\s+)?digits?)\b"
)
_PER_PACKAGE_NET_MASS = re.compile(
    r"(?ix)\b(?:1|one|per)\b.{0,36}\b(?:bag|drum|package|carton|case|crate|"
    r"pallet|bundle)s?\b.{0,72}\b(?:net\s*(?:wt|weight)|kgs?\s+net)\b"
)
_GENERIC_PAYMENT_PLACE = re.compile(
    r"(?ix)\b(?:ocean\s+)?freight\s+(?:is\s+)?(?:payable\s+)?(?:at\s+)?"
    r"(?:origin|destination)\b"
)
_GENERIC_FREIGHT_LOCATION_VALUE = re.compile(
    r"(?ix)^\s*(?:origin|destination)(?:\s+[0-9]+\s*/\s*[0-9]+)?\s*$"
)
_HS_TARGET = re.compile(r"^documentPatch\.cargoGroups\[[0-9]+\]\.hsCodes(?:\[[0-9]+\])?$")
_NET_WEIGHT_TARGET = re.compile(
    r"^documentPatch\.cargoGroups\[[0-9]+\]\.netWeight(?:\.value)?$"
)
_GROSS_WEIGHT_TARGET = re.compile(
    r"^documentPatch\.cargoGroups\[[0-9]+\]\.grossWeight(?:\.value)?$"
)
_VOLUME_TARGET = re.compile(
    r"^documentPatch\.cargoGroups\[[0-9]+\]\.volume(?:\.value)?$"
)
_NET_WEIGHT_CONTEXT = re.compile(r"(?ix)\bnet\s*(?:wt\.?|weight)?\b|\bkgs?\s+net\b")
_GROSS_WEIGHT_CONTEXT = re.compile(r"(?ix)\bgross\s*(?:wt\.?|weight)?\b|\bkgs?\s+gross\b")
_TARE_WEIGHT_CONTEXT = re.compile(r"(?ix)\btare\s*(?:wt\.?|weight)?\b|\bkgs?\s+tare\b")
_VOLUME_UNIT_CONTEXT = re.compile(
    r"(?ix)(?:\b(?:CBM|MTQ|M3)\b|\bCU\.?\s*M\.?\b|"
    r"\bCUBIC\s+MET(?:ER|RE)S?\b)"
)
_EXPLICIT_FLASH_POINT_CONTEXT = re.compile(
    r"(?ix)\b(?:flash\s*[- ]?(?:point|pt)|flp)\b\.?:?"
)
_CLOSED_CUP_CONTEXT = re.compile(
    r"(?ix)(?:\bclosed\s+cup\b|(?<![A-Z0-9])(?:C\.?\s*C\.?(?:\s*C\.)?|C\s*-\s*CC)"
    r"(?![A-Z0-9]))"
)
_DANGEROUS_GOODS_CONTEXT = re.compile(
    r"(?ix)(?:\bUN(?:DG)?(?:\s*(?:NO|NUMBER)\.?)?\s*[:#.-]?\s*[0-9]{4}\b|"
    r"\b(?:IMO\s+)?(?:HAZARD\s+)?CLASS\s*[:#.-]?\s*[1-9](?:\.[0-9])?\b|"
    r"\b(?:PACKING\s+GROUP|P\.?\s*G\.?)\s*[:#.-]?\s*(?:I|II|III|1|2|3)\b)"
)
_SHIPPER_LOAD_COUNT_BOILERPLATE = re.compile(
    r"(?ix)\bshipper(?:'s|s)?\s+load(?:ed)?\s*[,]?\s*stow(?:ed)?\s+and\s+count(?:ed)?\b"
)
_FAX_LABEL = re.compile(r"(?ix)\bfax(?:\s*(?:no|number))?\s*[:.]?\s*")
_JOINT_PHONE_FAX_LABEL = re.compile(
    r"(?ix)\b(?:tel(?:ephone)?|phone)\s*(?:&|/|-|and)\s*fax\s*[:.]?\s*"
)
_VIA_PARTY = re.compile(r"(?ix)\A\s*via\s+\S")
_FORWARDING_AGENT_CONTEXT = re.compile(r"(?ix)\bforward(?:ing)?\s+agent\b")
_EXPLICIT_SAME_AS = re.compile(
    r"(?ix)\b(?:the\s+)?same\s+as\s+(?:the\s+)?(?P<role>shipper|consignee|shpr|cnee)\b"
)
_ZERO_ORIGINALS = re.compile(
    r"(?isx)(?:\b(?:no\.?|number)\s+of\s+original(?:\s+bills?(?:\s+of\s+lading)?)?\b"
    r".{0,48}\b(?:0|zero)\b|\b0\s*\(\s*zero\s*\).{0,48}\boriginal)"
)
_NON_NEGOTIABLE_COPY = re.compile(r"(?ix)\bnon[- ]negotiable\s+copy\b")
_BILL_OF_LADING_FORM = re.compile(r"(?ix)\bbill\s+of\s+lading\b")
_POSITIVE_ORIGINAL_COUNT = re.compile(
    r"(?isx)(?:\b(?:no\.?|number)\s+of\s+original.{0,64}"
    r"\b(?:[1-9][0-9]*|one|two|three|four|five|six|seven|eight|nine)\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine)\s*(?:\([1-9]\))?"
    r"\s+originals?\b)"
)
_ORIGINAL_SURRENDER_OR_VOID = re.compile(
    r"(?isx)(?:\bsurrender(?:ed|ing)?\b.{0,96}\boriginal\b|"
    r"\boriginal\b.{0,96}\bsurrender(?:ed|ing)?\b|"
    r"\bone\b.{0,64}\b(?:accomplished|others?\s+(?:to\s+)?stand\s+void)\b)"
)
_EXPRESS_RELEASE = re.compile(r"(?ix)\b(?:express|telex)\s+release\b")
_CONSIGNEE_HEADING = re.compile(
    r"(?ix)^\s*(?:\(?[0-9]+\)?\s*[.)-]?\s*)?"
    r"(?:CONSIGNEE(?:\s*/\s*ORDER\s+OF)?|CONSIGNED\s+TO)\b"
)
_PARTY_OR_FORM_HEADING = re.compile(
    r"(?ix)^(?:shipper|exporter|consignee|notify(?:\s+party)?|carrier|forwarding\s+agent|"
    r"delivery\s+agent|vessel|voyage|port\s+of|place\s+of|marks(?:\s+and\s+numbers)?|"
    r"description\s+of\s+goods|container|freight|bill\s+of\s+lading)\b"
)
_ORDER_CONSIGNEE = re.compile(r"(?ix)\b(?:to\s+(?:the\s+)?order(?:\s+of)?|order\s+of)\b")
_CONSIGNEE_FORM_INSTRUCTION = re.compile(
    r"(?ix)^(?:"
    r"\(?\s*(?:complete|insert)\s+name(?:\s+and\s+address)?|"
    r"\(?\s*negotiable\s+only\s+if|"
    r"as\s+principal\s*,?\s+where|"
    r"this\s+B/?L\s+is\s+not\s+negotiable\s+unless|"
    r"\(?\s*if\s+['\u2018\u2019\"]?to\s+order['\u2018\u2019\"]?\s+so\s+indicate|"
    r"name\s+and\s+address"
    r")"
)
_VGM_TARGET = re.compile(
    r"^documentPatch\.containers\[[0-9]+\]\.verifiedGrossMass(?:\.value)?$"
)
_EXPLICIT_VGM_CONTEXT = re.compile(r"(?ix)\bVGM\b|\bVERIFIED\s+GROSS(?:\s+MASS)?\b")
_SHIPPER_PARTY_TARGET = re.compile(r"^documentPatch\.parties\.shipper(?:\.|$)")
_SHIPPER_REFERENCE_VALUE = re.compile(r"(?ix)^\s*shipper(?:'s)?\s+ref(?:erence)?\s*[#:]?")
_DESCRIPTION_TARGET = re.compile(r"^documentPatch\.cargoGroups\[[0-9]+\]\.description$")
_DESCRIPTION_MEASURE_TOKEN = re.compile(
    r"(?ix)\b(?:KGM|KGS?|LBS?|CBM|MTQ|M3|TON(?:NE)?S?)\b"
)
_REMOVE_OR_ABSENCE_ASSERTION = re.compile(
    r"(?ix)\b(?:remove|delete|unsupported|already\s+absent|must\s+(?:remain|be)\s+"
    r"(?:absent|omitted|null)|should\s+(?:remain|be)\s+(?:absent|omitted|null))\b"
)
_PRODUCT_MODEL_SPECIFICATION = re.compile(
    r"(?is)\b(?:model|pump)\b.{0,100}\bsize\s+[0-9]+(?:x[0-9]+)+(?:-[0-9]+)?\b"
)
_ADDRESS_TARGET = re.compile(
    r"^documentPatch\.parties\.(?P<party>shipper|consignee|carrier|forwardingAgent|"
    r"deliveryAgent|consolidator|notifyParties\[[0-9]+\])\.address$"
)
_ADDRESS_REDUNDANCY_ASSERTION = re.compile(
    r"(?ix)\b(?:redundan|duplicat|separately\s+model(?:ed|led))"
)
_FOREIGN_EXPORTER_COUNTRY = re.compile(r"(?ix)\bforeign\s+exporter\s+country\b")
_EXPLICIT_SHIPPER_EXPORTER_IDENTITY = re.compile(
    r"(?ix)\b(?:shipper\s*(?:/|and|is)\s*(?:the\s+)?exporter|"
    r"exporter\s*(?:/|and|is)\s*(?:the\s+)?shipper)\b"
)
_DELIVERY_AGENT_TARGET = re.compile(r"^documentPatch\.parties\.deliveryAgent(?:\.|$)")
_SHIPPING_AGENT_HEADING = re.compile(
    r"(?ix)\bshipping\s+agen(?:t|cy)(?:\s+details)?\b"
)
_EXPLICIT_DELIVERY_AGENT_HEADING = re.compile(
    r"(?ix)\b(?:delivery\s+agent|cargo\s+release\s+agent|agent\s+at\s+destination|"
    r"shipping\s+agen(?:t|cy)\s+at\s+port\s+of\s+discharge)\b"
)
_ALLOCATION_ROW_PACKAGE_ID_TARGET = re.compile(
    r"^documentPatch\.cargoAllocationGroups\[(?P<group>[0-9]+)\]"
    r"\.allocations\[(?P<allocation>[0-9]+)\]\.packageId$"
)
_HAZARD_CATEGORY_TARGET = re.compile(
    r"^documentPatch\.cargoGroups\[(?P<group>[0-9]+)\]\.dangerousGoods"
    r"\[(?P<dangerous_goods>[0-9]+)\](?:\.hazardCategory)?$"
)
_CLASS_TWO_VALUE = re.compile(
    r"(?ix)(?:\b(?:imo\s+)?class|\bcl\.?)\s*2(?:\.1)?\b|"
    r"^\s*2(?:\.1)?\s*$"
)
_PROPER_SHIPPING_NAME_VALUE = re.compile(
    r"(?ix)^\s*(?:PSN|PROPER\s+SHIPPING\s+NAME)\s*[:#-]"
)
_PACKING_GROUP_VALUE = re.compile(
    r"(?ix)^\s*(?:P\.?\s*G\.?|PACKING\s+GROUP)\s*[:#-]?\s*(?:I|II|III|1|2|3)\s*$"
)
_CARGO_DESCRIPTION_OR_ADDITIONAL_TARGET = re.compile(
    r"^documentPatch\.cargoGroups\[[0-9]+\]"
    r"\.(?:description|additionalInformation(?:\[[0-9]+\])?)$"
)


def named_non_order_consignee_spans(raw_ocr_text: str) -> tuple[tuple[int, int], ...]:
    """Return exact OCR spans that complete a straight named-consignee field.

    The supported forms are the ones observed in the corpus: a standalone or
    numbered ``Consignee`` heading, ``Consigned to``, an inline name, and the
    common multi-line form instructions between the heading and the value.  A
    literal ``to order`` value remains excluded.  Returning the source span,
    rather than only a Boolean, lets evidence generation and semantic review
    share one policy without fabricating a synthetic phrase.
    """

    lines = raw_ocr_text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    spans: list[tuple[int, int]] = []
    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r\n")
        heading = _CONSIGNEE_HEADING.match(line)
        if heading is None:
            continue

        candidates: list[tuple[int, str]] = []
        inline = line[heading.end() :].strip(" \t:.-")
        if inline:
            inline_start = line.find(inline, heading.end())
            candidates.append((offsets[index] + inline_start, inline))
        for following_index in range(index + 1, min(index + 13, len(lines))):
            following_line = lines[following_index].rstrip("\r\n")
            value = following_line.strip()
            if not value:
                continue
            value_start = following_line.find(value)
            candidates.append((offsets[following_index] + value_start, value))

        for start, value in candidates:
            if _CONSIGNEE_FORM_INSTRUCTION.search(value):
                continue
            if value.startswith(("(", ")")) and not re.search(r"(?iu)[^\W\d_]", value):
                continue
            if _PARTY_OR_FORM_HEADING.match(value):
                break
            if _ORDER_CONSIGNEE.search(value):
                break
            if re.search(r"(?iu)[^\W\d_]", value):
                spans.append((start, start + len(value)))
                break
    return tuple(dict.fromkeys(spans))


def _has_named_non_order_consignee(raw_ocr_text: str) -> bool:
    """Return whether OCR completes a consignee box with a straight named party."""

    return bool(named_non_order_consignee_spans(raw_ocr_text))


def has_flash_point_context(text: str) -> bool:
    """Accept explicit flash-point labels or DG-local printed closed-cup notation."""

    return bool(
        _EXPLICIT_FLASH_POINT_CONTEXT.search(text)
        or (
            _CLOSED_CUP_CONTEXT.search(text)
            and _DANGEROUS_GOODS_CONTEXT.search(text)
        )
    )


def non_negotiable_copy_is_only_copy_status(raw_ocr_text: str) -> bool:
    """Return whether a copy stamp lacks a completed straight-consignee basis."""

    return bool(
        _NON_NEGOTIABLE_COPY.search(raw_ocr_text)
        and _BILL_OF_LADING_FORM.search(raw_ocr_text)
        and _POSITIVE_ORIGINAL_COUNT.search(raw_ocr_text)
        and _ORIGINAL_SURRENDER_OR_VOID.search(raw_ocr_text)
        and not _EXPRESS_RELEASE.search(raw_ocr_text)
        and not _has_named_non_order_consignee(raw_ocr_text)
    )


def _finding_text(finding: SemanticReviewFinding) -> str:
    return "\n".join(
        value for row in finding.rawOcrEvidence for value in (row.ocrExcerpt, row.rawValue)
    )


def _candidate_document_patch(candidate: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if candidate is None:
        return None
    relation = candidate.get("relationExplicitLabel")
    if not isinstance(relation, Mapping):
        return None
    patch = relation.get("documentPatch")
    return patch if isinstance(patch, Mapping) else None


def _candidate_party(
    candidate: Mapping[str, Any] | None, party_path: str
) -> Mapping[str, Any] | None:
    patch = _candidate_document_patch(candidate)
    if patch is None:
        return None
    parties = patch.get("parties")
    if not isinstance(parties, Mapping):
        return None
    notify = re.fullmatch(r"notifyParties\[(?P<index>[0-9]+)\]", party_path)
    if notify is not None:
        rows = parties.get("notifyParties")
        index = int(notify.group("index"))
        if not isinstance(rows, list) or index >= len(rows):
            return None
        row = rows[index]
        return row if isinstance(row, Mapping) else None
    row = parties.get(party_path)
    return row if isinstance(row, Mapping) else None


def _fold_semantic_text(value: str) -> str:
    return " ".join(re.findall(r"[0-9a-z]+", value.casefold()))


def _address_redundancy_claim_is_true(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool | None:
    """Check a reviewer's claim against the exact candidate it reviewed.

    ``None`` means the narrow rule does not apply or candidate data is unavailable.
    """

    if not _ADDRESS_REDUNDANCY_ASSERTION.search(finding.message):
        return None
    target = next(
        (match for path in finding.targetPaths if (match := _ADDRESS_TARGET.fullmatch(path))),
        None,
    )
    if target is None:
        return None
    party = _candidate_party(candidate, target.group("party"))
    if party is None or not isinstance(address := party.get("address"), str):
        return None
    folded_address = _fold_semantic_text(address)
    modeled_components = tuple(
        _fold_semantic_text(value)
        for field in ("city", "country")
        if isinstance((value := party.get(field)), str) and value.strip()
    )
    return any(component and component in folded_address for component in modeled_components)


def _candidate_description_contains_all_evidence(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool | None:
    if not re.search(
        r"(?ix)\b(?:omit(?:s|ted|ting)?|missing|absent|incomplete)\b", finding.message
    ):
        return None
    target = next(
        (path for path in finding.targetPaths if _DESCRIPTION_TARGET.fullmatch(path)),
        None,
    )
    if target is None:
        return None
    patch = _candidate_document_patch(candidate)
    if patch is None:
        return None
    groups = patch.get("cargoGroups")
    matched = re.fullmatch(
        r"documentPatch\.cargoGroups\[(?P<index>[0-9]+)\]\.description", target
    )
    if not isinstance(groups, list) or matched is None:
        return None
    index = int(matched.group("index"))
    if index >= len(groups) or not isinstance(groups[index], Mapping):
        return None
    description = groups[index].get("description")
    if not isinstance(description, str):
        return False
    folded_description = _fold_semantic_text(description)
    cited_values = tuple(
        folded
        for row in finding.rawOcrEvidence
        if (folded := _fold_semantic_text(row.rawValue))
    )
    return bool(cited_values) and all(value in folded_description for value in cited_values)


_MISSING_TARGET = object()


def _candidate_target_value(candidate: Mapping[str, Any] | None, path: str) -> Any:
    """Read one public correction path from the exact candidate under review."""

    if candidate is None:
        return _MISSING_TARGET
    if path == "documentType":
        return candidate.get("documentType", _MISSING_TARGET)
    if not path.startswith("documentPatch"):
        return _MISSING_TARGET
    relation = candidate.get("relationExplicitLabel")
    if not isinstance(relation, Mapping):
        return _MISSING_TARGET
    current: Any = relation
    tokens: list[str | int] = []
    for field, index in re.findall(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]", path):
        tokens.append(field if field else int(index))
    for token in tokens:
        if isinstance(token, str) and isinstance(current, Mapping):
            current = current.get(token, _MISSING_TARGET)
        elif isinstance(token, int) and isinstance(current, list) and token < len(current):
            current = current[token]
        else:
            return _MISSING_TARGET
        if current is _MISSING_TARGET:
            return current
    return current


def _finding_demands_absence_already_satisfied(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool:
    """Reject a blocking removal finding when every target is already null/absent."""

    if finding.category not in {"truth_boundary", "contamination", "redundancy", "incorrect_field"}:
        return False
    if not _REMOVE_OR_ABSENCE_ASSERTION.search(finding.message) or not finding.targetPaths:
        return False
    values = tuple(_candidate_target_value(candidate, path) for path in finding.targetPaths)
    return all(value is _MISSING_TARGET or value is None for value in values)


def _finding_claims_absent_description_measure(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool:
    """Reject removal of a measurement token absent from the reviewed description."""

    if finding.category != "contamination" or not re.search(
        r"(?ix)\b(?:remove|delete|strip)\b", finding.message
    ):
        return False
    claimed = {
        _fold_semantic_text(row.group())
        for row in _DESCRIPTION_MEASURE_TOKEN.finditer(finding.message)
    }
    if not claimed:
        return False
    matched_target = next(
        (path for path in finding.targetPaths if _DESCRIPTION_TARGET.fullmatch(path)),
        None,
    )
    if matched_target is None:
        return False
    description = _candidate_target_value(candidate, matched_target)
    if not isinstance(description, str):
        return False
    present = {
        _fold_semantic_text(row.group()) for row in _DESCRIPTION_MEASURE_TOKEN.finditer(description)
    }
    return claimed.isdisjoint(present)


def _finding_demands_allocation_without_container(
    finding: SemanticReviewFinding,
    raw_ocr_text: str | None,
) -> bool:
    """Package.groupId already links package facts when no container can participate."""

    return bool(
        finding.category == "relationship"
        and any("cargoAllocationGroups" in path for path in finding.targetPaths)
        and raw_ocr_text is not None
        and not _contains_valid_container(raw_ocr_text)
    )


def _finding_targets_invalid_schema_path(finding: SemanticReviewFinding) -> tuple[str, ...]:
    invalid: list[str] = []
    for path in finding.targetPaths:
        try:
            normalize_correction_paths((path,))
        except TargetScopedCorrectionError:
            invalid.append(path)
    return tuple(invalid)


def _finding_demands_row_package_id_for_single_package_level(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool:
    """Reject a field forbidden by the selected allocation union branch."""

    patch = _candidate_document_patch(candidate)
    if patch is None:
        return False
    groups = patch.get("cargoAllocationGroups")
    if not isinstance(groups, list):
        return False
    broad_group_demand = bool(
        finding.category == "relationship"
        and re.search(
            r"(?ix)\ballocation\b.{0,100}\b(?:does\s+not|missing|omit(?:s|ted)?)\b"
            r".{0,60}\b(?:package|package\s+id|packageId)\b",
            finding.message,
        )
    )
    for path in finding.targetPaths:
        match = _ALLOCATION_ROW_PACKAGE_ID_TARGET.fullmatch(path)
        if match is not None:
            group_index = int(match.group("group"))
        elif broad_group_demand:
            broad_match = re.fullmatch(
                r"documentPatch\.cargoAllocationGroups\[(?P<group>[0-9]+)\]",
                path,
            )
            if broad_match is None:
                continue
            group_index = int(broad_match.group("group"))
        else:
            continue
        if group_index >= len(groups) or not isinstance(groups[group_index], Mapping):
            continue
        if groups[group_index].get("coverage") == "single_package_level":
            return True
    return False


def _finding_claims_duplicate_reference_not_present(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool:
    """Reject a duplicate-reference finding when the cited value occurs only once."""

    if finding.category != "redundancy":
        return False
    claimed = re.search(
        r"(?ix)\breference\s+([^\s,;:]+)\s+is\s+(?:represented|present)\s+twice\b",
        finding.message,
    )
    if claimed is None:
        return False
    patch = _candidate_document_patch(candidate)
    if patch is None:
        return False
    references = patch.get("forwardingAndExportReferences")
    if not isinstance(references, list):
        return True
    token = _fold_semantic_text(claimed.group(1))
    occurrences = sum(
        token in _fold_semantic_text(value)
        for value in references
        if isinstance(value, str)
    )
    return occurrences < 2


def _finding_rejects_canonical_class_two_category(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool:
    """Class 2 and 2.1 canonically project to the readable GASES category."""

    patch = _candidate_document_patch(candidate)
    if patch is None:
        return False
    cargo_groups = patch.get("cargoGroups")
    if not isinstance(cargo_groups, list):
        return False
    class_two_is_cited = any(
        _CLASS_TWO_VALUE.search(row.rawValue) is not None for row in finding.rawOcrEvidence
    )
    if not class_two_is_cited:
        return False
    for path in finding.targetPaths:
        match = _HAZARD_CATEGORY_TARGET.fullmatch(path)
        if match is None:
            continue
        group_index = int(match.group("group"))
        dangerous_goods_index = int(match.group("dangerous_goods"))
        if group_index >= len(cargo_groups):
            continue
        group = cargo_groups[group_index]
        if not isinstance(group, Mapping):
            continue
        dangerous_goods = group.get("dangerousGoods")
        if not isinstance(dangerous_goods, list) or dangerous_goods_index >= len(
            dangerous_goods
        ):
            continue
        row = dangerous_goods[dangerous_goods_index]
        if isinstance(row, Mapping) and row.get("hazardCategory") == "GASES":
            return True
    return False


def _finding_forces_out_of_scope_dangerous_goods_text(
    finding: SemanticReviewFinding,
) -> tuple[bool, bool]:
    """Detect attempts to force omitted DG-only fields into neighboring cargo text.

    The frozen target intentionally has no standalone proper-shipping-name or
    packing-group field.  Those values may be recorded in a warning, but they
    must not be injected into product description or generic cargo notes.
    """

    if not any(
        _CARGO_DESCRIPTION_OR_ADDITIONAL_TARGET.fullmatch(path)
        for path in finding.targetPaths
    ):
        return False, False
    has_proper_shipping_name = any(
        _PROPER_SHIPPING_NAME_VALUE.match(row.rawValue)
        for row in finding.rawOcrEvidence
    )
    has_standalone_packing_group = any(
        _PACKING_GROUP_VALUE.fullmatch(row.rawValue)
        for row in finding.rawOcrEvidence
    )
    return has_proper_shipping_name, has_standalone_packing_group


def _finding_demands_impossible_partial_package_link(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool:
    """Detect a link that would falsely reconcile a partial row to a shipment total."""

    if candidate is None or not any(
        "cargoAllocationGroups" in path
        and (
            "packageId" in path
            or "packageIds" in path
            or re.fullmatch(
                r"documentPatch\.cargoAllocationGroups(?:\[[0-9]+\])?",
                path,
            )
            is not None
        )
        for path in finding.targetPaths
    ):
        return False
    patch = _candidate_document_patch(candidate)
    if patch is None:
        return False
    packages = patch.get("cargoPackages")
    allocations = patch.get("cargoAllocationGroups")
    if not isinstance(packages, list) or not isinstance(allocations, list):
        return False
    for allocation in allocations:
        if not isinstance(allocation, Mapping) or allocation.get("coverage") != (
            "unlinked_package_quantities"
        ):
            continue
        group_id = allocation.get("groupId")
        package_total = sum(
            quantity
            for package in packages
            if isinstance(package, Mapping)
            and package.get("groupId") == group_id
            and isinstance((quantity := package.get("quantity")), int | float)
        )
        rows = allocation.get("allocations")
        if not isinstance(rows, list):
            continue
        allocation_total = sum(
            quantity
            for row in rows
            if isinstance(row, Mapping)
            and isinstance((quantity := row.get("packageQuantity")), int | float)
        )
        if 0 < allocation_total < package_total:
            return True
    return False


def _finding_collapses_distinct_row_linked_packages(
    finding: SemanticReviewFinding,
    candidate: Mapping[str, Any] | None,
) -> bool:
    """Reject collapsing identical package facts that belong to distinct OCR rows."""

    if candidate is None or not re.search(
        r"(?ix)\b(?:repeated|duplicate)\b.{0,80}\b(?:package|pallet)s?\b|"
        r"\b(?:single|one)\s+(?:shipment[- ]level\s+)?package\s+(?:fact|level)\b",
        finding.message,
    ):
        return False
    patch = _candidate_document_patch(candidate)
    if patch is None:
        return False
    packages = patch.get("cargoPackages")
    allocation_groups = patch.get("cargoAllocationGroups")
    if not isinstance(packages, list) or not isinstance(allocation_groups, list):
        return False
    packages_by_id = {
        row.get("packageId"): row
        for row in packages
        if isinstance(row, Mapping) and isinstance(row.get("packageId"), str)
    }
    cited_containers = {
        re.sub(r"[^A-Za-z0-9]", "", match.group()).upper()
        for row in finding.rawOcrEvidence
        for match in _CONTAINER_CANDIDATE.finditer(f"{row.rawValue}\n{row.ocrExcerpt}")
    }
    for group in allocation_groups:
        if not isinstance(group, Mapping) or group.get("coverage") != (
            "one_to_one_package_allocations"
        ):
            continue
        allocations = group.get("allocations")
        if not isinstance(allocations, list) or len(allocations) < 2:
            continue
        rows = [row for row in allocations if isinstance(row, Mapping)]
        package_rows = [
            packages_by_id.get(row.get("packageId"))
            for row in rows
            if isinstance(row.get("packageId"), str)
        ]
        if len(package_rows) != len(rows) or any(row is None for row in package_rows):
            continue
        semantic_keys = {
            (
                row.get("groupId"),
                row.get("quantity"),
                row.get("typeDescription"),
                row.get("typeCategory"),
            )
            for row in package_rows
            if isinstance(row, Mapping)
        }
        allocation_containers = {
            row.get("containerNumber")
            for row in rows
            if isinstance(row.get("containerNumber"), str)
        }
        quantities_match = all(
            isinstance(package_row, Mapping)
            and allocation.get("packageQuantity") == package_row.get("quantity")
            for allocation, package_row in zip(rows, package_rows, strict=True)
        )
        if (
            len(semantic_keys) == 1
            and len(allocation_containers) == len(rows)
            and len(cited_containers & allocation_containers) >= 2
            and quantities_match
        ):
            return True
    return False


def _contains_valid_container(value: str) -> bool:
    for match in _CONTAINER_CANDIDATE.finditer(value):
        candidate = re.sub(r"[^A-Za-z0-9]", "", match.group()).upper()
        try:
            _CONTAINER_ADAPTER.validate_python(candidate, strict=True)
        except ValidationError:
            continue
        return True
    return False


def _contains_valid_imo(value: str) -> bool:
    for match in _IMO_CANDIDATE.finditer(value):
        try:
            _IMO_ADAPTER.validate_python(match.group(), strict=True)
        except ValidationError:
            continue
        return True
    return False


def _has_standalone_fax_evidence(finding: SemanticReviewFinding) -> bool:
    """Return whether a demanded phone value is explicitly fax-only in OCR.

    Joint ``TEL & FAX`` values remain valid phone targets because the printed
    value serves both roles.  Only a separately labeled fax is excluded.
    """

    def label_precedes_value(label: re.Pattern[str], excerpt: str, raw_value: str) -> bool:
        return any(
            excerpt[match.end() :].lstrip().startswith(raw_value)
            for match in label.finditer(excerpt)
        )

    for row in finding.rawOcrEvidence:
        raw_value = row.rawValue.strip()
        if not raw_value:
            continue
        if _JOINT_PHONE_FAX_LABEL.match(raw_value) or label_precedes_value(
            _JOINT_PHONE_FAX_LABEL, row.ocrExcerpt, raw_value
        ):
            continue
        if _FAX_LABEL.match(raw_value) or label_precedes_value(
            _FAX_LABEL, row.ocrExcerpt, raw_value
        ):
            return True
    return False


def _out_of_range_hs_values(finding: SemanticReviewFinding) -> tuple[str, ...]:
    values: list[str] = []
    for row in finding.rawOcrEvidence:
        digits = re.sub(r"\D", "", row.rawValue)
        if digits and not 6 <= len(digits) <= 18:
            values.append(digits)
    return tuple(values)


def _truncated_hs_values(finding: SemanticReviewFinding) -> tuple[str, ...]:
    """Reject reviewer-invented prefixes/suffixes of longer printed HS tokens."""

    values: list[str] = []
    for row in finding.rawOcrEvidence:
        raw_value = row.rawValue.strip()
        if not raw_value.isdigit():
            continue
        if re.search(rf"(?<![0-9]){re.escape(raw_value)}(?![0-9])", row.ocrExcerpt):
            continue
        if any(
            raw_value in printed and raw_value != printed
            for printed in re.findall(r"(?<![0-9])[0-9]{6,}(?![0-9])", row.ocrExcerpt)
        ):
            values.append(raw_value)
    return tuple(values)


def _contains_excluded_reference_value(finding: SemanticReviewFinding) -> bool:
    for row in finding.rawOcrEvidence:
        if _SUPPORTED_REFERENCE.search(row.rawValue):
            continue
        if _EXCLUDED_ID_REFERENCE.search(row.rawValue):
            return True
        escaped = re.escape(row.rawValue.strip())
        if not escaped:
            continue
        for label in _EXCLUDED_ID_REFERENCE.finditer(row.ocrExcerpt):
            suffix = row.ocrExcerpt[label.end() :]
            if re.match(
                rf"(?ix)\s*(?:(?:id|no|number)\.?\s*)?[:#-]?\s*{escaped}",
                suffix,
            ):
                return True
    return False


def _contains_heading_without_value(finding: SemanticReviewFinding) -> bool:
    return any(
        _NON_VALUE_REFERENCE_HEADING.fullmatch(row.rawValue)
        for row in finding.rawOcrEvidence
    )


def _measure_context_is_scoped(
    finding: SemanticReviewFinding,
    *,
    qualifier: re.Pattern[str],
) -> bool:
    """Require a gross/net qualifier associated with the cited scalar.

    A flattened excerpt that merely contains several table headings does not
    prove which column owns a scalar.  Same-line/adjacent context is accepted;
    a PDF-assisted finding may also use an explicit qualifier elsewhere in the
    cited excerpt because ``imageUse`` records that layout association.
    """

    text = _finding_text(finding)
    if not qualifier.search(text):
        return False
    for row in finding.rawOcrEvidence:
        raw_value = row.rawValue.strip()
        if qualifier.search(raw_value):
            return True
        lines = row.ocrExcerpt.splitlines()
        for line_index, line in enumerate(lines):
            if raw_value and raw_value not in line:
                continue
            local_lines = lines[max(0, line_index - 1) : line_index + 1]
            local_context = "\n".join(local_lines)
            if qualifier.search(local_context):
                return True
            if _TARE_WEIGHT_CONTEXT.search(local_context):
                return False
    if finding.imageUse != "not_used":
        return True
    return not _TARE_WEIGHT_CONTEXT.search(text)


def _same_as_reference_is_grounded(finding: SemanticReviewFinding) -> bool:
    text = _finding_text(finding)
    relation = _EXPLICIT_SAME_AS.search(text)
    if relation is None:
        return False
    role = relation.group("role").lower()
    role_name = "shipper" if role in {"shipper", "shpr"} else "consignee"
    role_heading = re.compile(rf"(?i)\b{role_name}\b")
    return any(
        not _EXPLICIT_SAME_AS.search(row.rawValue)
        and role_heading.search(row.ocrExcerpt)
        and row.rawValue.strip()
        for row in finding.rawOcrEvidence
    )


def _via_is_unscoped_to_forwarding_agent(finding: SemanticReviewFinding) -> bool:
    return any(_VIA_PARTY.match(row.rawValue) for row in finding.rawOcrEvidence) and not any(
        _FORWARDING_AGENT_CONTEXT.search(row.ocrExcerpt)
        for row in finding.rawOcrEvidence
    )


def review_policy_violations(
    findings: Sequence[SemanticReviewFinding],
    *,
    candidate: Mapping[str, Any] | None = None,
    raw_ocr_text: str | None = None,
) -> tuple[str, ...]:
    """Return exact reasons that findings contradict frozen target semantics.

    These checks are deliberately narrow. They cover only failure modes
    observed in retained reviewer transcripts and only when the cited evidence
    proves that the demanded value cannot populate the named target.
    """

    violations: list[str] = []
    for index, finding in enumerate(findings):
        text = _finding_text(finding)
        targets = finding.targetPaths
        if invalid_paths := _finding_targets_invalid_schema_path(finding):
            violations.append(
                f"finding {index}: correction target path(s) do not exist in the frozen "
                f"extraction schema: {', '.join(invalid_paths)}"
            )
        if _finding_demands_absence_already_satisfied(finding, candidate):
            violations.append(
                f"finding {index}: every requested removal/omission target is already null or "
                "absent in the reviewed candidate; an already-satisfied truth-boundary check "
                "cannot be blocking"
            )
        if _finding_claims_absent_description_measure(finding, candidate):
            violations.append(
                f"finding {index}: the measurement token claimed as description contamination "
                "is absent from the reviewed candidate description"
            )
        if _finding_demands_allocation_without_container(finding, raw_ocr_text):
            violations.append(
                f"finding {index}: cargoAllocationGroups represent container relationships; "
                "without a valid printed container, cargoPackages.groupId already links each "
                "package level to its cargo group"
            )
        if (
            candidate is not None
            and candidate.get("documentType") == "bill_of_lading"
            and raw_ocr_text is not None
            and _NON_NEGOTIABLE_COPY.search(text)
            and non_negotiable_copy_is_only_copy_status(raw_ocr_text)
            and (
                finding.category == "document_unit"
                or "documentType" in targets
                or "documentPatch.negotiability" in targets
            )
        ):
            violations.append(
                f"finding {index}: NON-NEGOTIABLE COPY is a copy-status stamp and cannot "
                "override the completed positive-original and original-surrender terms of the "
                "underlying bill of lading"
            )
        if any(_VGM_TARGET.fullmatch(path) for path in targets) and not (
            _EXPLICIT_VGM_CONTEXT.search(text)
        ):
            violations.append(
                f"finding {index}: verifiedGrossMass requires explicit local VGM or verified "
                "gross-mass wording; an ordinary container-row KGS/LBS value is not VGM"
            )
        if (
            finding.category == "missing_field"
            and any(_SHIPPER_PARTY_TARGET.match(path) for path in targets)
            and any(_SHIPPER_REFERENCE_VALUE.match(row.rawValue) for row in finding.rawOcrEvidence)
        ):
            violations.append(
                f"finding {index}: a Shipper Ref value is a reference, not a shipper-role party"
            )
        if (
            finding.category == "missing_field"
            and any(path == "documentPatch.parties.shipper.country" for path in targets)
            and _FOREIGN_EXPORTER_COUNTRY.search(text)
            and not _EXPLICIT_SHIPPER_EXPORTER_IDENTITY.search(text)
        ):
            violations.append(
                f"finding {index}: FOREIGN EXPORTER COUNTRY alone does not prove the shipper's "
                "country; exporter-to-shipper identity must be explicit"
            )
        if (
            finding.category in {"contamination", "incorrect_field"}
            and any(_DESCRIPTION_TARGET.fullmatch(path) for path in targets)
            and _PRODUCT_MODEL_SPECIFICATION.search(text)
            and re.search(r"(?ix)\b(?:measure|dimension|size)\b", finding.message)
        ):
            violations.append(
                f"finding {index}: integral product model/specification dimensions remain in "
                "cargo description and are not shipment-measure contamination"
            )
        if (
            finding.category in {"contamination", "incorrect_field", "redundancy"}
            and _address_redundancy_claim_is_true(finding, candidate) is False
        ):
            violations.append(
                f"finding {index}: the candidate address does not contain its separately modeled "
                "city or country, so the claimed address redundancy is absent"
            )
        if (
            finding.category in {"missing_field", "incorrect_field"}
            and _candidate_description_contains_all_evidence(finding, candidate) is True
        ):
            violations.append(
                f"finding {index}: every cited product value is already present in the reviewed "
                "candidate description, so the claimed description omission is absent"
            )
        if _finding_demands_impossible_partial_package_link(finding, candidate):
            violations.append(
                f"finding {index}: a partial explicit container-row quantity cannot be linked "
                "to a larger shipment-level package fact without inventing the unprinted "
                "remainder; retain unlinked_package_quantities"
            )
        if _finding_collapses_distinct_row_linked_packages(finding, candidate):
            violations.append(
                f"finding {index}: identical package facts tied one-to-one to distinct printed "
                "container rows are row facts, not redundant shipment-level duplicates"
            )
        if _finding_demands_row_package_id_for_single_package_level(finding, candidate):
            violations.append(
                f"finding {index}: single_package_level stores its packageId once on the "
                "allocation group; its allocation rows intentionally contain only container "
                "and quantity"
            )
        if _finding_claims_duplicate_reference_not_present(finding, candidate):
            violations.append(
                f"finding {index}: the cited forwarding/export reference occurs only once in "
                "the reviewed candidate, so the claimed duplication is absent"
            )
        if _finding_rejects_canonical_class_two_category(finding, candidate):
            violations.append(
                f"finding {index}: printed dangerous-goods class 2 or 2.1 canonically maps to "
                "the readable GASES category; the target does not preserve numeric subclass "
                "2.1 separately"
            )
        forces_proper_shipping_name, forces_packing_group = (
            _finding_forces_out_of_scope_dangerous_goods_text(finding)
        )
        if forces_proper_shipping_name:
            violations.append(
                f"finding {index}: standalone proper shipping name is outside the frozen "
                "target and must not be forced into cargo description or additionalInformation"
            )
        if forces_packing_group:
            violations.append(
                f"finding {index}: standalone dangerous-goods packing group is outside the "
                "frozen target and must not be forced into cargo description or "
                "additionalInformation"
            )
        if any(".flashPoint" in path for path in targets) and not has_flash_point_context(text):
            violations.append(
                f"finding {index}: flashPoint requires an explicit FLASH POINT/FLASH PT/FLP "
                "label or a temperature with printed closed-cup notation in local dangerous-"
                "goods context"
            )
        if (
            any(path == "documentPatch.transport.vesselImoNumber" for path in targets)
            and not _contains_valid_imo(text)
        ):
            violations.append(
                f"finding {index}: vesselImoNumber accepts only a printed seven-digit value "
                "with a valid IMO checksum; an invalid Lloyds/MO scalar must remain omitted"
            )
        if any(".handlingInstructions" in path for path in targets) and (
            finding.rawOcrEvidence
            and all(
                _SHIPPER_LOAD_COUNT_BOILERPLATE.search(row.rawValue) is not None
                for row in finding.rawOcrEvidence
            )
        ):
            violations.append(
                f"finding {index}: SHIPPERS LOAD, STOW AND COUNT is responsibility boilerplate, "
                "not an operational cargo handling instruction"
            )
        if finding.category == "missing_field":
            if (
                any(_DELIVERY_AGENT_TARGET.match(path) for path in targets)
                and _SHIPPING_AGENT_HEADING.search(text)
                and not _EXPLICIT_DELIVERY_AGENT_HEADING.search(text)
            ):
                violations.append(
                    f"finding {index}: a generic SHIPPING AGENT heading does not establish the "
                    "delivery-agent role; explicit delivery, cargo-release, or destination-agent "
                    "scope is required"
                )
            if any(".phoneNumbers" in path for path in targets) and _has_standalone_fax_evidence(
                finding
            ):
                violations.append(
                    f"finding {index}: a separately labeled fax value cannot populate "
                    "phoneNumbers; only a joint TEL/FAX value is also a phone target"
                )
            if any(_HS_TARGET.fullmatch(path) for path in targets) and (
                unsupported_hs := _out_of_range_hs_values(finding)
            ):
                violations.append(
                    f"finding {index}: HS output supports 6-18 printed digits; "
                    f"out-of-range value(s) cannot be demanded: {', '.join(unsupported_hs)}"
                )
            if any(_HS_TARGET.fullmatch(path) for path in targets) and (
                truncated_hs := _truncated_hs_values(finding)
            ):
                violations.append(
                    f"finding {index}: an HS value must be a complete printed token after "
                    "separator removal; a prefix or suffix of a longer code cannot be "
                    f"demanded: {', '.join(truncated_hs)}"
                )
            if any(path == "documentPatch.containers" for path in targets) and not (
                _contains_valid_container(text)
            ):
                violations.append(
                    f"finding {index}: an anonymous container count/type cannot populate "
                    "documentPatch.containers without a valid printed ISO 6346 identifier"
                )
            if any(path.startswith("documentPatch.freight") for path in targets) and (
                _UNSUPPORTED_FREIGHT_TERM.search(text) and not _SUPPORTED_FREIGHT.search(text)
            ):
                violations.append(
                    f"finding {index}: the cited operating/handling freight term is not one of "
                    "the supported freight payment values"
                )
            if any(path.startswith("documentPatch.freight") for path in targets) and any(
                _GENERIC_FREIGHT_LOCATION_VALUE.fullmatch(row.rawValue)
                for row in finding.rawOcrEvidence
            ) and not _SUPPORTED_FREIGHT.search(text):
                violations.append(
                    f"finding {index}: generic ORIGIN/DESTINATION is a payment-location selector, "
                    "not evidence for prepaid/collect payment arrangement or a named place"
                )
            if any(
                path.startswith("documentPatch.freight.paymentPlace") for path in targets
            ) and (
                _GENERIC_PAYMENT_PLACE.search(text)
            ):
                violations.append(
                    f"finding {index}: generic Origin/Destination wording is not a named freight "
                    "payment location"
                )
            if any(_NET_WEIGHT_TARGET.fullmatch(path) for path in targets):
                if _PER_PACKAGE_NET_MASS.search(text):
                    violations.append(
                        f"finding {index}: a per-package net mass cannot populate the cargo-group "
                        "aggregate netWeight"
                    )
                elif not _measure_context_is_scoped(
                    finding,
                    qualifier=_NET_WEIGHT_CONTEXT,
                ):
                    violations.append(
                        f"finding {index}: cargo-group netWeight requires a column-scoped "
                        "net-weight qualifier; a tare or unqualified table scalar is not net "
                        "weight"
                    )
            if any(_GROSS_WEIGHT_TARGET.fullmatch(path) for path in targets) and not (
                _measure_context_is_scoped(
                    finding,
                    qualifier=_GROSS_WEIGHT_CONTEXT,
                )
            ):
                violations.append(
                    f"finding {index}: cargo-group grossWeight requires a column-scoped "
                    "gross-weight qualifier; a tare or unqualified table scalar is not gross "
                    "weight"
                )
            if any(_VOLUME_TARGET.fullmatch(path) for path in targets) and not (
                _VOLUME_UNIT_CONTEXT.search(text)
            ):
                violations.append(
                    f"finding {index}: cargo-group volume requires an explicit printed cubic "
                    "volume unit; a unitless MEASUREMENT scalar cannot populate volume"
                )
            if any(
                re.match(r"^documentPatch\.cargoGroups\[[0-9]+\]\.origin$", path)
                for path in targets
            ):
                if _PARTY_COUNTRY.search(text) and not _GOODS_ORIGIN.search(text):
                    violations.append(
                        f"finding {index}: exporter/shipper/importer country is a party fact, "
                        "not cargo origin"
                    )
                elif not _GOODS_ORIGIN.search(text):
                    violations.append(
                        f"finding {index}: cargo origin requires explicit goods-origin context"
                    )
            if (
                any(
                    path.startswith("documentPatch.forwardingAndExportReferences")
                    for path in targets
                )
                and _contains_excluded_reference_value(finding)
            ):
                violations.append(
                    f"finding {index}: regulatory/customs data or exporter/importer identity or "
                    "registration data is excluded from forwardingAndExportReferences"
                )
            if any(
                path.startswith("documentPatch.forwardingAndExportReferences")
                for path in targets
            ) and _contains_heading_without_value(finding):
                violations.append(
                    f"finding {index}: an adjacent blank heading is not a forwarding/export "
                    "reference value"
                )
            if any(
                path.startswith("documentPatch.parties.forwardingAgent") for path in targets
            ) and _via_is_unscoped_to_forwarding_agent(finding):
                violations.append(
                    f"finding {index}: unscoped VIA text inside another party block does not "
                    "populate forwardingAgent; an explicit forwarding-agent role heading is "
                    "required"
                )
        if finding.category in {"missing_field", "relationship"}:
            same_as_is_demanded = any(".sameAs" in path for path in targets) or (
                any(
                    path == "documentPatch.parties.notifyParties"
                    or re.fullmatch(r"documentPatch\.parties\.notifyParties\[[0-9]+\]", path)
                    for path in targets
                )
                and _EXPLICIT_SAME_AS.search(text) is not None
            )
            if same_as_is_demanded and not _same_as_reference_is_grounded(finding):
                violations.append(
                    f"finding {index}: sameAs requires explicit SAME AS SHIPPER/CONSIGNEE "
                    "wording and a cited OCR-grounded value for the named referenced party"
                )
        if (
            finding.category == "document_unit"
            and _ZERO_ORIGINALS.search(text)
            and re.search(
                r"(?ix)\b(?:sea[ -]?waybill|non[- ]negotiable|document\s+type)\b",
                finding.message,
            )
        ):
            violations.append(
                f"finding {index}: a completed zero-original count supports sea_waybill "
                "classification and must not be rejected for lacking generic form wording"
            )
        if (
            finding.category == "incorrect_field"
            and any(_CONTAINER_CORRECTION_TARGET.fullmatch(path) for path in targets)
            and _INVALID_CONTAINER_ASSERTION.search(finding.message)
            and _contains_valid_container(text)
        ):
            violations.append(
                f"finding {index}: cited container identifier has a valid ISO 6346 check "
                "digit; reviewers must not override deterministic identifier validation"
            )
        if (
            finding.category == "incorrect_field"
            and any(".phoneNumbers" in path for path in targets)
            and _PHONE_DIGITS_ONLY_ASSERTION.search(finding.message)
            and any(character.isdigit() for character in text)
        ):
            violations.append(
                f"finding {index}: OCR-conditioned phone values may retain printed OCR "
                "characters when the cited value contains digits; reviewers must not demand "
                "an image-derived correction or digits-only rewrite"
            )
    return tuple(violations)


def validate_review_policy(
    findings: Sequence[SemanticReviewFinding],
    *,
    candidate: Mapping[str, Any] | None = None,
    raw_ocr_text: str | None = None,
) -> None:
    violations = review_policy_violations(
        findings,
        candidate=candidate,
        raw_ocr_text=raw_ocr_text,
    )
    if violations:
        raise ReviewPolicyError("; ".join(violations))
