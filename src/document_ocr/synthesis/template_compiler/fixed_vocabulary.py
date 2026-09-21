"""Closed, non-identifying template vocabulary, never a source-copy fallback.

Only complete, enumerated surfaces qualify. In particular a value-kind label such
as operational_text is not evidence that arbitrary text is invariant: it can
contain quantities, organizations or shipment-dependent instructions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .models import SemanticBinding

_EQUIPMENT_KIND = (
    r"(?:HC|HQ|GP|DC|DV|RF|RH|RE|HR|IRE|OT|FR|TK|ST|SP|H|DR|FCL|"
    r"HIGHCUBE(?:STANDARD)?|STANDARD|ISOTANK)"
)
_CONTAINER_UNIT = r"(?:CONTAINER|CTNR|CNTR)(?:\(S\)|S)?"
_EQUIPMENT_CORE = (
    rf"(?:20|40|45)(?:FT)?(?:{_EQUIPMENT_KIND}(?:/{_EQUIPMENT_KIND})?)?"
    rf"(?:FCL)?(?:{_CONTAINER_UNIT})?"
)
_EQUIPMENT_CONTEXT = re.compile(
    rf"(?:{_EQUIPMENT_CORE}|0*[1-9][0-9]*[X*]{_EQUIPMENT_CORE}"
    rf"|{_EQUIPMENT_CORE}[X*]0*[1-9][0-9]*"
    rf"|0*[1-9][0-9]*{_CONTAINER_UNIT}(?:\(0*[1-9][0-9]*X(?:20|40|45)\))?"
    rf"|(?:ONE|TWO|THREE|FOUR|FIVE){_CONTAINER_UNIT}|0*[1-9][0-9]*X"
    r"|COC|SOC|FI/FO|GEN|SEAL|CTNR|CNTRS|FCL|HC|HI-CUBE|REEFERCNTR"
    r"|FC40H|45G0?1|HC(?:40|45)|(?:20SP20FT|40SP40FT))"
)

# These are document-state labels and freight/service names, not generated facts.
# Numbers, locations, organizations, cargo descriptions and DG declarations are
# intentionally absent. Keep the grammar closed rather than accepting a suffix
# like "charge", which could conceal a newly generated party or quantity.
_LABELS = frozenset(
    {
        "FCL",
        "LCL",
        "CY",
        "CFS",
        "CY/CY",
        "CY/CFS",
        "CFS/CY",
        "CFS/CFS",
        "FCL/FCL",
        "LCL/LCL",
        "FCL/FREE OUT",
        "PORT TO PORT",
        "DOOR TO DOOR",
        "DOOR/DOOR",
        "DOOR/CY",
        "CY/DOOR",
        "DOOR/FREE OUT",
        "SD/CY",
        "CY/SD",
        "SHIPPED ON BOARD",
        "ON BOARD",
        "ON BOARD VESSEL",
        "DEPARTURE",
        "PREPAID",
        "COLLECT",
        "FREIGHT PREPAID",
        "FREIGHT COLLECT",
        "FREIGHT AS PER AGREEMENT",
        "FREIGHT AS ARRANGED",
        "FREIGHT ALL AS ARRANGED",
        "FREE OUT TERM",
        "AS PER AGREEMENT",
        "AS ARRANGED",
        "FREE IN",
        "FREE OUT",
        "FREE IN/FREE OUT",
        "NVD",
        "NO VALUE DECLARED",
        "SHIPPER'S LOAD & COUNT",
        "SHIPPER'S LOAD AND COUNT",
        "SAID TO CONTAIN",
        "SHIPPER'S LOAD, STOW AND COUNT",
        "BASIC OCEAN FREIGHT",
        "OCEAN FREIGHT",
        "BUNKER ADJUSTMENT FACTOR",
        "EMERGENCY CONTINGENCY SURCHARGE",
        "DISCHARGE FEE - DESTINATION",
        "FREE OUT SERVICE",
        "DOCUMENTATION FEE - ORIGIN",
        "TERMINAL HANDLING SERVICE - ORIGIN",
        "TERMINAL HANDLING SERVICE - DESTINATION",
        "BOOKING SERVICES",
        "EXPORT SERVICE",
        "CUSTOMS CLEARANCE",
        "CUSTOMS ADDITIONAL ITEM CHARGE - IMP",
        "CUSTOMS ADDITIONAL ITEM CHARGE - IM",
        "CUSTOMS IMPORT",
        "CUSTOMS ADDITIONAL ITEM",
        "ENVIRONMENTAL FUEL FEE",
        "PREMIUM PACKAGE",
        "LADEN ON BOARD",
        "RECEIVED FOR SHIPMENT",
        '"RECEIVED FOR SHIPMENT"',
        "DECLARED CLEAN BY SHIPPER",
        "SHIPMENT TO BE EFFECTED IN CONTAINER",
        "FREIGHT FORWARDER - ORIGIN",
        "ORIGIN",
        "EXW",
        "FCA",
        "CPT",
        "CIP",
        "DAP",
        "DPU",
        "DDP",
        "FAS",
        "FOB",
        "CFR",
        "CIF",
        "EX WORKS",
        "DESTINATION",
        "FREIGHT",
        "CUSTOMS",
        "USD",
        "EGP",
        "EUR",
        "PER CONTAINER",
        "VALUE PROTECT STARTER",
        "VALUE PROTECT COOL STANDARD",
        "CAPTAIN PETER - PREMIUM PACKAGE",
        "FREETIME EXTENSION CONTRACTS",
        "FREE IN FREE OUT",
        "LINER IN/FREE OUT",
        "ADDITIONAL ITEM CHARGE - IMP",
        "EMISSION SURCHARGE SPOT AND ST CONT",
        "SPOT BOOKING AMENDMENT FEE",
        "FREE IN SERVICE",
        "GOVERNMENT AND PORT TAX EXPORTS",
        "PICK-UP CHARGE (EXPORTS)",
        "AS AGREED",
        "DTHC COLLECT",
        "FREETIME AS PER AGREEMENT",
        "SVC CONTRACT",
        "NOMINATION CARGO",
        "DESTINATION CHARGES COLLECT",
        "LOCAL CHARGES AT DESTINATION ARE FOR RECEIVERS ACCOUNT",
        "ALL DESTINATION CHARGES ON ACCOUNT OF CONSIGNEE",
        "ALL DESTINATION CHARGES ARE FOR THE ACCOUNT OF CONSIGNEE",
        "STRIPPING CHARGES AT C.F.S. ARE ON RECEIVER'S ACCOUNT",
        "FREE OUT - ALL COSTS FROM FREE OUT VESSEL ARE FOR RECEIVER'S ACCOUNT",
    }
)
_DOCUMENT_LABELS = frozenset(
    {
        "EBL ELECTRONIC BILL OF LADING",
        "PDF",
        "DRAFT",
        "EXPRESS",
        "NO REF",
        "ISSUE AND TRANSFER OF DOCUMENT POSSESSION",
        "ISSUE AND TRANSFER OF DOCUMENT POSSESION",
        "TRANSFER OF DOCUMENT POSSESSION",
        "TRANSFER OF DOCUMENT POSSESION",
        "ISSUE AND TRANSFER EBL DOCUMENT",
        "REQUEST SURRENDER FOR AMENDMENT",
        "ACCEPT SURRENDER FOR AMENDMENT (VOID)",
        "REQUEST DOCUMENT SURRENDER FOR DELIVERY",
        "REJECT DOCUMENT SURRENDER FOR DELIVERY",
        "PLACE AND DATE OF ISSUE",
        "NUMBER OF ORIGINAL B/L",
    }
)
_MOVEMENT_LABELS = frozenset(
    {
        "FCLFCL",
        "FCL CARGO",
        "FCL CONTR",
        "LINER",
        "UNLOADING",
        "CLEAN ON BOARD",
        "TO BE NOMINATED",
        "ECONOMY",
        "ON CY-FO TERM",
        "FCL CY-CY",
        "FCL/FCL CY/CY",
        "L.C.L./L.C.L.",
        "O/O",
        "0/O",
        "SD",
    }
)
_LEGAL_CLAUSES = frozenset(
    {
        "ALL COSTS, CHARGES, LIABILITIES AND DELAYS RESULTING FROM EMERGENCY QUARANTINE "
        "FROM WOOD AND WOODEN PACKAGING OR FROM INSUFFICENT OR IMPROPER LABELING OF "
        "NON-WOODEN PACKAGING IS FOR THE ACCOUNT OF THE CUSTOMER",
        "THE CARRIER IS NOT RESPONSIBLE FOR INCORRECT ACID-NUMBER AND THE RESPONSIBILITY "
        "REMAINS WITH THE MERCHANT!",
        "ANY EXPENSES FOR TRANSPORTATION FROM TERMINAL TO OUTSIDE TERMINAL AND VICE VERSA, "
        "TO BE COLLECTED FROM CARGO RECEIVERS",
        "ANY EXPENSES FOR TRANSPORTATION FROM TERMINAL TO OUTSIDE TERMINAL AND VICE VERSA, "
        "TO BE COLLECTED FROM CARO RECEIVERS",
    }
)
_CARGO_LABELS = frozenset(
    {
        "SLAC",
        "SLAC*",
        "STC",
        "FCL CNTRS",
        "SAID TO CONTAIN",
        "SHIPPER'S LOAD & COUNT",
        "SHIPPER'S LOAD AND COUNT",
        "SHIPPER'S LOAD, STOW AND COUNT",
    }
)


def _fixed_label(surface: str) -> bool:
    return bool(
        surface in _LABELS | _CARGO_LABELS
        # Edition and free-time are explicit commercial context, never cargo
        # quantities. No free-form trailing text is admitted by these grammars.
        or re.fullmatch(r"INCOTERMS (?:19|20)\d{2}", surface)
        or re.fullmatch(
            r"\d{1,3} (?:CALENDAR )?DAYS (?:DETENTION|DEMURRAGE) FREE AT (?:ORIGIN|DESTINATION)",
            surface,
        )
    )


def fixed_vocabulary(binding: SemanticBinding) -> bool:
    """A source-only binding is fixed only if its *entire* surface is known."""
    if (
        binding.target_paths
        or binding.dependency_paths
        or binding.dependency_bindings
        or binding.source_relationships
        or binding.derivation is not None
        or binding.value_kind
        not in {"operational_text", "commercial_text", "legal_text", "other_text"}
        or binding.group_kind in {"party", "customs", "dangerous_goods"}
    ):
        return False
    surfaces = {
        re.sub(r"\s*/\s*", "/", " ".join(slot.source_text.upper().split())).rstrip(".")
        for slot in binding.occurrences
    }
    if binding.group_kind in {"cargo", "package"}:
        return bool(surfaces) and surfaces <= _CARGO_LABELS
    return bool(surfaces) and all(
        _fixed_label(s)
        or s in _DOCUMENT_LABELS | _MOVEMENT_LABELS | _LEGAL_CLAUSES
        or re.fullmatch(
            r"\(?(?:CY|CFS|FCL|LCL|DOOR)[/-](?:CY|CFS|FCL|LCL|FO|FREE OUT|DOOR)\)?\*?", s
        )
        or (s.endswith(" COLLECT") and s.removesuffix(" COLLECT") in _LABELS)
        for s in surfaces
    )


def fixed_context(
    binding: SemanticBinding, source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    """Closed contextual atoms, with explicit unchanged semantic dimensions.

    No free-form text is accepted based only on its compiler category. Package
    words, unit words and packing groups remain conditioned on their source facts.
    Shipment quantities, addresses, identities and arbitrary clauses never qualify.
    Source-only route locations stay in the complete unchanged route scenario.
    """
    if fixed_vocabulary(binding):
        return True
    if (
        binding.target_paths
        or binding.dependency_paths
        or binding.dependency_bindings
        or binding.source_relationships
        or binding.derivation
    ):
        return False
    texts = {" ".join(s.source_text.upper().split()).rstrip(".") for s in binding.occurrences}
    if not texts:
        return False
    key = set(re.findall(r"[a-z]+", binding.logical_key.lower()))
    old, new = source["documentPatch"], target["documentPatch"]
    if binding.group_kind == "route" and binding.value_kind == "location":
        return bool(old.get("route")) and old.get("route") == new.get("route")
    if binding.value_kind == "equipment":

        def shape(patch: Mapping[str, Any]) -> list[tuple[Any, Any]]:
            return [
                (c.get("sizeCategory"), c.get("typeCategory")) for c in patch.get("containers", [])
            ]

        if shape(old) == shape(new) and all(
            _EQUIPMENT_CONTEXT.fullmatch(re.sub(r"[\s'\"`\u2019]", "", s)) for s in texts
        ):
            return True
        # Independent feeder/intended vessels remain part of the source route;
        # a main-vessel alias must instead follow its generated target.
        role = re.sub(r"[^a-z]", "", binding.logical_key.lower())
        main = re.sub(r"[^A-Z0-9]", "", old.get("transport", {}).get("vesselName", "").upper())
        if (
            binding.group_kind in {"transport", "route"}
            and any(
                k in role for k in ("precarriage", "firstleg", "intendedvessel", "vesselauxiliary")
            )
            and old.get("route")
            and old.get("route") == new.get("route")
            and all(not main or re.sub(r"[^A-Z0-9]", "", s) not in main for s in texts)
        ):
            return True
    if binding.group_kind == "document" and binding.value_kind in {
        "other_text",
        "operational_text",
        "commercial_text",
    }:
        return all(
            s in _DOCUMENT_LABELS
            or re.fullmatch(
                r"(?:PDF \()?\d+ KB\)?|(?:AT )?\d{2}:\d{2}(?: \(UTC\))?|SEE ATTACHED PAGE NO\. \d+",
                s,
            )
            for s in texts
        )
    if binding.group_kind == "customs" and texts <= {"VAT NUMBER", "COMPANY"}:
        return True
    if binding.value_kind in {"legal_text", "operational_text"} and texts <= _LEGAL_CLAUSES:
        return True
    if binding.value_kind in {"commercial_text", "operational_text"} and binding.group_kind in {
        "commercial",
        "transport",
        "other",
    }:
        if (key & {"freetime", "demurrage", "detention"} or {"free", "time"} <= key) and all(
            re.fullmatch(
                r"\d{1,3} (?:CALENDAR )?DAYS(?: (?:DETENTION|DEMURRAGE|DEMMURAGE|FREE|TIME|"
                r"FREETIME|AT|DESTINATION|ORIGIN|MERGED|COMBINED|AND|OF|PORT))*",
                s,
            )
            for s in texts
        ):
            return True
        # A rate per day/container is a fixed commercial parameter, not a total.
        if (
            binding.value_kind == "commercial_text"
            and key & {"rate", "demurrage", "detention"}
            and all(re.fullmatch(r"\d+(?:\.\d+)?/(?:DAY|20RF|40RF)", s) for s in texts)
        ):
            return True
    if binding.group_kind in {
        "cargo",
        "package",
        "equipment",
        "customs",
    } and binding.value_kind in {"package", "other_text"}:
        old_packages = [
            (p.get("typeCategory"), p.get("typeDescription")) for p in old.get("cargoPackages", [])
        ]
        new_packages = [
            (p.get("typeCategory"), p.get("typeDescription")) for p in new.get("cargoPackages", [])
        ]
        from .descendant import _PACKAGE_SURFACES

        labels = {word.upper() for words in _PACKAGE_SURFACES.values() for word in words}
        labels.update({"PALLETS SLAC", "PIECE(S)", "BG"})
        if old_packages == new_packages and texts <= labels:
            return True
        units = {"KGM", "KGS", "KG", "MTQ", "CBM", "LBR", "M.TON", "ADMT", "EA"}
        from .descendant import _flatten_leaves

        old_units = {p: v for p, v in _flatten_leaves(source).items() if p.endswith(".unit")}
        new_units = {p: v for p, v in _flatten_leaves(target).items() if p.endswith(".unit")}
        if old_units == new_units and texts <= units:
            return True
    if binding.value_kind == "dangerous_goods" and binding.group_kind == "dangerous_goods":
        if not ({"packing", "group"} <= key or "emb" in key):
            return False
        from .descendant import _flatten_leaves

        def identities(patch: Mapping[str, Any]) -> dict[str, Any]:
            return {
                p: v
                for p, v in _flatten_leaves(patch).items()
                if "dangerousGoods" in p or ".hsCodes[" in p
            }

        return (
            bool(identities(source))
            and identities(source) == identities(target)
            and all(
                re.fullmatch(r"(?:(?:PG|EMB|PACKING GROUP) )?(?:I|II|III|1|2|3)", s) for s in texts
            )
        )
    return False
