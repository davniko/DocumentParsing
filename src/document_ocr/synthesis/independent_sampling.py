"""Registry-backed shipment facts independent of source or donor cargo.

The source document may constrain printed cardinality and field presence, but
cannot supply a commodity, quantity, package, tare, or load ratio to this module.
The package category is an independently certified choice supplied by the
caller. These small physical envelopes are synthesis bounds, not a commercial
packing approval.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from document_ocr.hashing import canonical_json_sha256, sha256_file
from document_ocr.synthesis.curated_dg_flash import load_flash_points
from document_ocr.synthesis.dangerous_goods_registry import load_dangerous_goods_registry
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import (
    compile_uk_global_tariff_registry,
    load_ukgt_source_pin,
)
from document_ocr.synthesis.independent_cargo_presentation import (
    CargoPresentation,
    validate_cargo_presentation,
)
from document_ocr.synthesis.independent_envelopes import (
    DEFAULT_ROW_PAYLOAD_KG,
    DEFAULT_ROW_VOLUME_M3,
    maximum_units_per_row,
    package_envelope,
)
from document_ocr.synthesis.independent_packages import package_material
from document_ocr.synthesis.independent_unit_profiles import (
    UnitProfileRequest,
    sample_unit_profile,
    validate_profile,
)
from document_ocr.synthesis.tank_equipment import TANK_SIZES, TANK_TYPES
from document_ocr.synthesis.template_compiler.dangerous_goods_packaging import profile
from document_ocr.synthesis.thermal_goods import classify_thermal_descriptions

_REGIMES = frozenset({"ambient", "frozen", "chilled", "dg"})
_GP_SIZES = (
    "TWENTY_FOOT_STANDARD_HEIGHT",
    "TWENTY_FOOT_HIGH_CUBE",
    "FORTY_FOOT_STANDARD_HEIGHT",
    "FORTY_FOOT_HIGH_CUBE",
    "FORTY_FIVE_FOOT_HIGH_CUBE",
)
_REEFER_SIZES = ("FORTY_FOOT_STANDARD_HEIGHT", "FORTY_FOOT_HIGH_CUBE")
_OPEN_TOP_SIZES = (
    "TWENTY_FOOT_STANDARD_HEIGHT",
    "FORTY_FOOT_STANDARD_HEIGHT",
    "FORTY_FOOT_HIGH_CUBE",
)
_PLATFORM_SIZES = (
    "TWENTY_FOOT_STANDARD_HEIGHT",
    "FORTY_FOOT_STANDARD_HEIGHT",
    "FORTY_FOOT_HIGH_CUBE",
)
_PIN_KEYS = frozenset(
    {"hsManifest", "hsMetadata", "hsReport", "phrases", "dgHmt", "dgEcics", "flash"}
)


@dataclass(frozen=True, slots=True)
class PinnedFile:
    path: Path
    sha256: str

    @classmethod
    def from_config(cls, root: Path, value: Mapping[str, Any]) -> PinnedFile:
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ValueError("registry file pin needs exactly path and sha256")
        relative = Path(value["path"])
        path = relative if relative.is_absolute() else root / relative
        digest = value["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("registry file pin needs a SHA-256 hex digest")
        return cls(path.resolve(), digest)

    def verify(self) -> None:
        if sha256_file(self.path) != self.sha256:
            raise ValueError(f"registry file differs from configured pin: {self.path}")


@dataclass(frozen=True, slots=True)
class RegistryInputs:
    hs_manifest: PinnedFile
    hs_metadata: PinnedFile
    hs_report: PinnedFile
    phrases: PinnedFile
    dg_hmt: PinnedFile
    dg_ecics: PinnedFile
    flash: PinnedFile

    @classmethod
    def from_config(cls, root: Path, value: Mapping[str, Any]) -> RegistryInputs:
        if not isinstance(value, Mapping) or set(value) != _PIN_KEYS:
            raise ValueError("independent registry configuration needs seven exact file pins")
        pins = {key: PinnedFile.from_config(root, value[key]) for key in _PIN_KEYS}
        return cls(
            hs_manifest=pins["hsManifest"],
            hs_metadata=pins["hsMetadata"],
            hs_report=pins["hsReport"],
            phrases=pins["phrases"],
            dg_hmt=pins["dgHmt"],
            dg_ecics=pins["dgEcics"],
            flash=pins["flash"],
        )

    def pins(self) -> dict[str, str]:
        return {
            "hsManifest": self.hs_manifest.sha256,
            "hsMetadata": self.hs_metadata.sha256,
            "hsReport": self.hs_report.sha256,
            "phrases": self.phrases.sha256,
            "dgHmt": self.dg_hmt.sha256,
            "dgEcics": self.dg_ecics.sha256,
            "flash": self.flash.sha256,
        }


def _pick(values: Sequence[Any], stream: DeterministicStream) -> Any:
    if not values:
        raise ValueError("requested independent sampling domain is empty")
    return values[stream.randbelow(len(values))]


def _integer(low: int, high: int, stream: DeterministicStream) -> int:
    if low > high:
        raise ValueError("requested independent integer domain is empty")
    return low + stream.randbelow(high - low + 1)


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sample_equipment(
    *,
    regime: str,
    equipment_mode: str,
    seed: int,
    draw_key: str,
    container_count: int,
    equipment_domain: Sequence[Mapping[str, str]] | None = None,
    equipment_slots: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Draw carrying hardware independently of commodity and package choices."""
    if regime not in _REGIMES or equipment_mode not in {"uniform", "mixed"}:
        raise ValueError("unknown independent regime or equipment mode")
    if not draw_key or type(container_count) is not int or container_count < 0:
        raise ValueError("equipment draw needs a key and nonnegative container count")
    thermal = regime in {"frozen", "chilled"}
    if equipment_domain is None:
        equipment_domain = tuple(
            {"sizeCategory": size, "typeCategory": "REFRIGERATED" if thermal else "GENERAL_PURPOSE"}
            for size in (_REEFER_SIZES if thermal else _GP_SIZES)
        )
    domain = tuple(dict(item) for item in equipment_domain)
    if not domain or len({(d.get("sizeCategory"), d.get("typeCategory")) for d in domain}) != len(
        domain
    ):
        raise ValueError("equipment domain is empty or duplicated")
    if any(
        set(item) != {"sizeCategory", "typeCategory"}
        or not (
            (
                item["typeCategory"] == ("REFRIGERATED" if thermal else "GENERAL_PURPOSE")
                and item["sizeCategory"] in (_REEFER_SIZES if thermal else _GP_SIZES)
            )
            or (
                regime == "ambient"
                and item["typeCategory"] == "OPEN_TOP"
                and item["sizeCategory"] in _OPEN_TOP_SIZES
            )
            or (
                regime == "ambient"
                and item["typeCategory"] == "PLATFORM"
                and item["sizeCategory"] in _PLATFORM_SIZES
            )
            or (
                regime == "ambient"
                and item["typeCategory"] == "REFRIGERATED"
                and item["sizeCategory"] in _REEFER_SIZES
            )
            or (
                regime in {"ambient", "dg"}
                and item["typeCategory"] in TANK_TYPES
                and item["sizeCategory"] in TANK_SIZES
            )
        )
        for item in domain
    ):
        raise ValueError("equipment domain is incompatible with independent cargo regime")
    tank_types = {item["typeCategory"] for item in domain if item["typeCategory"] in TANK_TYPES}
    if tank_types and (
        len(tank_types) != 1 or any(item["typeCategory"] not in tank_types for item in domain)
    ):
        raise ValueError(
            "tank domain requires one explicit tank family, not a dry-box substitution"
        )
    if any(item["typeCategory"] == "OPEN_TOP" for item in domain) and any(
        item["typeCategory"] != "OPEN_TOP" for item in domain
    ):
        raise ValueError("open-top domain cannot mix unsupported carrying-hardware families")
    if any(item["typeCategory"] == "PLATFORM" for item in domain) and any(
        item["typeCategory"] != "PLATFORM" for item in domain
    ):
        raise ValueError("platform domain cannot mix unsupported carrying-hardware families")
    if equipment_slots is not None:
        if len(equipment_slots) != container_count:
            raise ValueError("fixed equipment slots differ from printed container count")
        equipment = [dict(item) for item in equipment_slots]
        if any(item not in domain for item in equipment):
            raise ValueError("fixed equipment slot is outside the certified equipment domain")
    else:
        stream = DeterministicStream(seed, "independent-facts-v2", draw_key)
        equipment = []
        first = _pick(domain, stream.derive("equipment-first"))
        for row in range(container_count):
            if equipment_mode == "uniform" or row == 0:
                item = first
            else:
                candidates = tuple(item for item in domain if row != 1 or item != first)
                item = _pick(candidates, stream.derive(f"equipment-{row}"))
            equipment.append(dict(item))
    distinct = {(item["sizeCategory"], item["typeCategory"]) for item in equipment}
    if equipment_mode == "mixed" and len(distinct) < 2:
        raise ValueError("mixed equipment requires two distinct printed rows")
    if equipment_mode == "uniform" and len(distinct) > 1:
        raise ValueError("uniform equipment request conflicts with fixed printed mix")
    return equipment


class IndependentSampler:
    """Load exact public registries and draw facts without observed shipments."""

    def __init__(self, registry: RegistryInputs):
        self.registry = registry
        for pin in (
            registry.hs_manifest,
            registry.hs_metadata,
            registry.hs_report,
            registry.phrases,
            registry.dg_hmt,
            registry.dg_ecics,
            registry.flash,
        ):
            pin.verify()
        source_pin = load_ukgt_source_pin(registry.hs_manifest.path)
        self.hs = compile_uk_global_tariff_registry(
            metadata_path=registry.hs_metadata.path,
            report_path=registry.hs_report.path,
            source=source_pin,
        )
        self.phrases: dict[str, str] = {}
        with registry.phrases.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                if set(row) < {"hs6", "phrase"} or not isinstance(row["phrase"], str):
                    raise ValueError(f"malformed HS phrase row {number}")
                code = row["hs6"]
                if code in self.phrases or not row["phrase"].strip():
                    raise ValueError(f"duplicate or empty HS phrase at row {number}")
                self.phrases[code] = row["phrase"]
        if set(self.phrases) != set(self.hs.global_codes):
            raise ValueError("HS phrases do not cover the complete pinned global HS6 registry")
        self.dg = load_dangerous_goods_registry(
            hmt_path=registry.dg_hmt.path,
            hmt_sha256=registry.dg_hmt.sha256,
            ecics_path=registry.dg_ecics.path,
            ecics_sha256=registry.dg_ecics.sha256,
        )
        self.flash = load_flash_points(
            registry.flash.path, registry.flash.sha256, None
        ).identity_index(self.dg)
        self.hmt_states: dict[str, str] = {}
        for record in self.dg.hmt_records:
            try:
                self.hmt_states[record.record_id] = profile(record).state
            except ValueError:
                continue
        self.hmt = {
            record.record_id: record
            for record in self.dg.hmt_records
            if record.record_id in self.hmt_states
        }
        links: dict[str, set[str]] = defaultdict(set)
        matched_link_rows = 0
        for link in self.dg.ecics_links:
            if (
                link.disposition == "eligible_unique_maritime_hmt"
                and link.hmt_record_id in self.hmt
                and link.hs6 in self.phrases
            ):
                matched_link_rows += 1
                links[link.hmt_record_id].add(link.hs6)
        self.matched_link_rows = matched_link_rows
        self.hs_links = {record_id: frozenset(codes) for record_id, codes in links.items()}
        pools: dict[str, list[str]] = {regime: [] for regime in ("ambient", "frozen", "chilled")}
        for code in self.hs.global_codes:
            thermal = classify_thermal_descriptions(
                tuple(reversed(self.hs.global_description_path(code)))
            )
            pools[(thermal or "ambient").lower()].append(code)
        if set().union(*(set(codes) for codes in pools.values())) != set(self.phrases):
            raise ValueError("thermal identity pools omit a pinned HS6 code")
        self.pools = {regime: tuple(sorted(codes)) for regime, codes in pools.items()}

    @classmethod
    def from_config(cls, root: Path, value: Mapping[str, Any]) -> IndependentSampler:
        return cls(RegistryInputs.from_config(root, value))

    def support(self) -> dict[str, Any]:
        """Report eligible registry reachability, not realized accepted coverage."""
        return {
            "registryHs6": len(self.phrases),
            "hs6ByRegime": {key: len(value) for key, value in self.pools.items()},
            "joinedDgRecords": len(self.hs_links),
            "joinedDgLinkRows": self.matched_link_rows,
            "joinedDgPairs": sum(map(len, self.hs_links.values())),
            "joinedDgHs6": len(set().union(*map(set, self.hs_links.values()))),
            "hmtRegistryRecords": len(self.dg.hmt_records),
            "hmtUsableRecords": len(self.hmt),
            "flashPointRecords": len(self.flash),
            "sourceCargoAccess": False,
            "registryPins": self.registry.pins(),
        }

    def _dg_pool(self, requirements: Mapping[str, Any]) -> tuple[str, ...]:
        if set(requirements) - {"flashPoint", "subsidiaryCount", "packingGroup"}:
            raise ValueError("unknown DG presentation requirement")
        flash = requirements.get("flashPoint", False)
        packing = requirements.get("packingGroup", False)
        subsidiary = requirements.get("subsidiaryCount")
        if (
            type(flash) is not bool
            or type(packing) is not bool
            or (subsidiary is not None and (type(subsidiary) is not int or subsidiary < 0))
        ):
            raise ValueError("invalid DG presentation requirement")
        return tuple(
            sorted(
                record_id
                for record_id in self.hs_links
                if (not flash or record_id in self.flash)
                and (not packing or self.hmt[record_id].packing_group_category is not None)
                and (
                    subsidiary is None
                    or len(self.hmt[record_id].subsidiary_hazard_categories) == subsidiary
                )
            )
        )

    def prepare_request(
        self,
        *,
        source_document_id: str,
        variant: int,
        regime: str,
        identity_count: int,
        seed: int,
        equipment_mode: str = "uniform",
        dg_requirements: Mapping[str, Any] | None = None,
        excluded_identifiers: Sequence[str] = (),
        cargo_presentation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Freeze goods-first identities using only presentation constraints.

        ``source_document_id`` is a deterministic identity key, not a source
        data access path. The caller may exclude earlier variant identities to
        encourage diversity; when that empties the eligible domain, sampling
        explicitly returns to the complete domain.
        """
        if (
            not isinstance(source_document_id, str)
            or not source_document_id
            or type(variant) is not int
            or variant < 1
            or type(seed) is not int
        ):
            raise ValueError("request needs a source identity, positive variant and integer seed")
        if regime not in _REGIMES or equipment_mode not in {"uniform", "mixed"}:
            raise ValueError("unknown independent regime or equipment mode")
        count = _require_positive_int(identity_count, "identity_count")
        if regime == "dg" and count != 1:
            raise ValueError("one DG declaration needs one independently joined HMT record")
        if regime != "dg" and dg_requirements:
            raise ValueError("non-DG request cannot require DG fields")
        stream = DeterministicStream(
            seed, "independent-registry-request-v1", f"{source_document_id}:{variant}"
        )
        if isinstance(excluded_identifiers, str) or any(
            not isinstance(item, str) for item in excluded_identifiers
        ):
            raise ValueError("excluded registry identities must be a string sequence")
        excluded = set(excluded_identifiers)
        if len(excluded) != len(excluded_identifiers):
            raise ValueError("duplicate excluded registry identities")
        declaration = None
        presentation = (
            None
            if cargo_presentation is None
            else CargoPresentation.model_validate(cargo_presentation)
        )
        if regime == "dg":
            requirements = dict(dg_requirements or {})
            eligible = self._dg_pool(requirements)
            if presentation is not None:
                eligible = tuple(
                    record_id
                    for record_id in eligible
                    if any(presentation.permits(code) for code in self.hs_links[record_id])
                )
            if not excluded <= set(eligible):
                raise ValueError("excluded DG identities are outside the eligible registry pool")
            domain = tuple(record_id for record_id in eligible if record_id not in excluded)
            reused_after_exhaustion = not domain
            if reused_after_exhaustion:
                domain = eligible
            record_id = _pick(domain, stream.derive("hmt-record"))
            record = self.hmt[record_id]
            codes = tuple(
                sorted(
                    code
                    for code in self.hs_links[record_id]
                    if presentation is None or presentation.permits(code)
                )
            )
            code = _pick(codes, stream.derive("exact-hs-join"))
            goods = [{"hs6": code, "description": record.proper_shipping_name}]
            declaration = {
                "recordId": record_id,
                "unNumber": record.un_number,
                "properShippingName": record.proper_shipping_name,
                "class": record.exact_hazard_class,
                "packingGroup": record.packing_group_code,
                "subsidiaryHazards": list(record.exact_subsidiary_hazards),
            }
        else:
            requirements = {}
            eligible = self.pools[regime]
            if presentation is not None:
                eligible = tuple(code for code in eligible if presentation.permits(code))
            if not excluded <= set(eligible):
                raise ValueError("excluded HS identities are outside the eligible registry pool")
            domain = tuple(code for code in eligible if code not in excluded)
            reused_after_exhaustion = len(domain) < count
            if reused_after_exhaustion:
                domain = eligible
            if len(domain) < count:
                raise ValueError("registry regime has fewer codes than printed identity slots")
            chosen: list[str] = []
            for index in range(count):
                code = _pick(
                    tuple(code for code in domain if code not in chosen),
                    stream.derive(f"hs6-{index}"),
                )
                chosen.append(code)
            goods = [{"hs6": code, "description": self.phrases[code]} for code in chosen]
        result = {
            "sourceDocumentId": source_document_id,
            "variant": variant,
            "regime": regime,
            "goods": goods,
            "goodsForm": "; ".join(item["description"] for item in goods),
            "dangerousGoods": declaration,
            "equipmentMode": equipment_mode,
            "dgRequirements": requirements,
            "registrySelection": {
                "eligibleIdentities": len(eligible),
                "fullHs6": len(self.phrases),
                "joinedDgRecords": len(self.hs_links),
                "excludedIdentities": len(excluded),
                "reusedAfterExhaustion": reused_after_exhaustion,
            },
            "sourceCargoUsedForSelection": False,
        }
        if presentation is not None:
            result["cargoPresentation"] = presentation.model_dump(mode="json")
        return result

    def validate_request(self, request: Mapping[str, Any]) -> None:
        """Reject any frozen goods request outside its pinned registry ancestry."""
        regime = request.get("regime")
        goods = request.get("goods")
        if regime not in _REGIMES or not isinstance(goods, list) or not goods:
            raise ValueError("frozen request has no valid regime/goods")
        if request.get("sourceCargoUsedForSelection") is not False:
            raise ValueError("frozen request lacks donor-independent provenance")
        if "cargoPresentation" in request:
            validate_cargo_presentation(request["cargoPresentation"], request)
        if (
            not isinstance(request.get("sourceDocumentId"), str)
            or not request["sourceDocumentId"]
            or type(request.get("variant")) is not int
            or request["variant"] < 1
        ):
            raise ValueError("frozen request lacks a valid presentation identity")
        if request.get("equipmentMode") not in {"uniform", "mixed"}:
            raise ValueError("frozen request has unsupported equipment mode")
        if request.get("goodsForm") != "; ".join(g["description"] for g in goods):
            raise ValueError("frozen goods form differs from registry descriptions")
        if regime == "dg":
            if len(goods) != 1 or not isinstance(request.get("dangerousGoods"), Mapping):
                raise ValueError("DG request needs one joined identity and declaration")
            declaration = request["dangerousGoods"]
            record_id = declaration.get("recordId")
            if record_id not in self.hs_links or goods[0]["hs6"] not in self.hs_links[record_id]:
                raise ValueError("DG identity lacks its exact eligible HMT/HS6 join")
            record = self.hmt[record_id]
            expected = {
                "recordId": record_id,
                "unNumber": record.un_number,
                "properShippingName": record.proper_shipping_name,
                "class": record.exact_hazard_class,
                "packingGroup": record.packing_group_code,
                "subsidiaryHazards": list(record.exact_subsidiary_hazards),
            }
            if (
                dict(declaration) != expected
                or goods[0]["description"] != record.proper_shipping_name
            ):
                raise ValueError("DG request changed an authoritative registry tuple")
            if record_id not in self._dg_pool(request.get("dgRequirements", {})):
                raise ValueError("DG request fails its printed-field requirements")
        else:
            if request.get("dangerousGoods") is not None or request.get("dgRequirements"):
                raise ValueError("non-DG request carries DG facts")
            if len({item["hs6"] for item in goods}) != len(goods):
                raise ValueError("frozen request repeats an HS6 identity")
            for item in goods:
                code = item["hs6"]
                if code not in self.pools[regime] or item["description"] != self.phrases[code]:
                    raise ValueError("frozen request differs from its full HS6 registry ancestry")

    def draw(
        self,
        *,
        request: Mapping[str, Any],
        seed: int,
        draw_key: str,
        container_count: int,
        selected_category: str,
        counted_unit_context: str | None = None,
        quantity_multiple: int = 1,
        fixed_quantity: int | None = None,
        equipment_domain: Sequence[Mapping[str, str]] | None = None,
        equipment_slots: Sequence[Mapping[str, str]] | None = None,
        row_capacities: Sequence[Mapping[str, Any]] | None = None,
        unit_profile_precision: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Draw quantities and equipment against an independently chosen package."""
        self.validate_request(request)
        if not draw_key or type(container_count) is not int or container_count < 0:
            raise ValueError("draw needs a key and nonnegative printed container count")
        multiple = _require_positive_int(quantity_multiple, "quantity_multiple")
        if fixed_quantity is not None and (
            type(fixed_quantity) is not int or fixed_quantity < 1 or fixed_quantity % multiple
        ):
            raise ValueError("fixed quantity violates the independent count lattice")
        regime = request["regime"]
        thermal = regime in {"frozen", "chilled"}
        envelope = package_envelope(selected_category, counted_unit_context=counted_unit_context)
        if request.get("selectedCategory") != selected_category:
            raise ValueError("chosen package category differs from frozen request")
        if (
            "countedUnitContext" not in request
            or request["countedUnitContext"] != counted_unit_context
        ):
            raise ValueError("counted-unit context differs from frozen request")
        stream = DeterministicStream(seed, "independent-facts-v2", draw_key)
        equipment = sample_equipment(
            regime=regime,
            equipment_mode=request["equipmentMode"],
            seed=seed,
            draw_key=draw_key,
            container_count=container_count,
            equipment_domain=equipment_domain,
            equipment_slots=equipment_slots,
        )
        if any(item["typeCategory"] in TANK_TYPES for item in equipment):
            if counted_unit_context != "tank_loading_lot":
                raise ValueError(
                    "installed tank requires explicit tank_loading_lot counting context"
                )
        elif counted_unit_context == "tank_loading_lot":
            raise ValueError("tank_loading_lot requires installed tank carrying equipment")
        rows = max(container_count, 1)
        if row_capacities is not None and len(row_capacities) != rows:
            raise ValueError("row capacities do not match independent load rows")
        capacities = tuple(
            (
                Decimal(item["payloadKg"]),
                Decimal(item["volumeM3"]),
            )
            for item in (
                row_capacities
                if row_capacities is not None
                else (
                    {
                        "payloadKg": DEFAULT_ROW_PAYLOAD_KG,
                        "volumeM3": DEFAULT_ROW_VOLUME_M3,
                    },
                )
                * rows
            )
        )
        if any(
            payload <= 0 or cube <= 0 or not payload.is_finite() or not cube.is_finite()
            for payload, cube in capacities
        ):
            raise ValueError("row capacities must be positive finite decimals")
        variable_unit = request.get("nativeUnitProfileEvidence")
        composed_unit = request.get("nativeComposedProfileEvidence")
        if variable_unit is not None and composed_unit is not None:
            raise ValueError("one physical planning authority is required")
        native_unit = variable_unit is not None or composed_unit is not None
        if native_unit and row_capacities is None:
            raise ValueError("native unit profiles require explicit current planning capacities")
        # Explicit row limits can tighten the synthetic envelope, never loosen
        # its common bound or turn a carrier-specific rating into a new default.
        if not native_unit:
            capacities = tuple(
                (min(payload, DEFAULT_ROW_PAYLOAD_KG), min(cube, DEFAULT_ROW_VOLUME_M3))
                for payload, cube in capacities
            )
            unit_net = _integer(*envelope.net_kg, stream.derive("unit-net-kg"))
            unit_gross = unit_net + envelope.tare_kg
            unit_cube = envelope.cube_m3
            numeric_authority = envelope.receipt()
        else:
            if unit_profile_precision is None or set(unit_profile_precision) != {
                "massKg",
                "volumeM3",
            }:
                raise ValueError("native unit profile requires reviewed source numeric precision")
            precision = dict(
                mass_quantum_kg=Decimal(unit_profile_precision["massKg"]),
                volume_quantum_m3=Decimal(unit_profile_precision["volumeM3"]),
            )
            if composed_unit is not None:
                from document_ocr.synthesis.independent_composed_profiles import (
                    ComposedRequest,
                    sample_composed_profile,
                )

                unit_net, unit_gross, unit_cube, numeric_authority = sample_composed_profile(
                    ComposedRequest.model_validate(composed_unit["request"]),
                    composed_unit["output"],
                    capacities,
                    stream,
                    **precision,
                )
            else:
                unit_request = UnitProfileRequest.model_validate(variable_unit["request"])
                unit_net, unit_gross, unit_cube, numeric_authority = sample_unit_profile(
                    unit_request,
                    validate_profile(unit_request, variable_unit["output"]),
                    capacities,
                    stream,
                    **precision,
                )
        if not native_unit and envelope.cube_range_m3 is not None:
            low_cube, high_cube = envelope.cube_range_m3
            unit_cube = (
                Decimal(
                    _integer(
                        int(low_cube * 100),
                        int(high_cube * 100),
                        stream.derive("unit-cube-centim3"),
                    )
                )
                / 100
            )
        maxima = tuple(
            min(int(payload // unit_gross), int(cube // unit_cube))
            if native_unit
            else maximum_units_per_row(
                envelope,
                unit_net_kg=unit_net,
                unit_cube_m3=unit_cube,
                payload_kg=payload,
                volume_m3=cube,
            )
            for payload, cube in capacities
        )
        count_step = numeric_authority.get("quantityMultiple", 1)
        if fixed_quantity is not None and fixed_quantity % (multiple * count_step):
            raise ValueError("fixed quantity violates the source precision count lattice")
        maxima = tuple(value // count_step for value in maxima)
        if any(value < 1 for value in maxima):
            raise ValueError("independent unit cannot fit every requested load row")
        low = (rows + multiple - 1) // multiple
        high = sum(maxima) // multiple
        quantity_groups = (
            _integer(low, high, stream.derive("declared-quantity")) * multiple
            if fixed_quantity is None
            else fixed_quantity // count_step
        )
        if quantity_groups < rows or quantity_groups > sum(maxima):
            raise ValueError("fixed quantity exceeds independent load envelope")
        quantity = quantity_groups * count_step
        allocations = []
        remaining = quantity_groups
        for row, maximum in enumerate(maxima):
            later_maximum = sum(maxima[row + 1 :])
            later_rows = rows - row - 1
            count = _integer(
                max(1, remaining - later_maximum),
                min(maximum, remaining - later_rows),
                stream.derive(f"row-quantity-{row}"),
            )
            remaining -= count
            count *= count_step
            allocations.append(
                {
                    "row": row,
                    "packageQuantity": count,
                    "netKg": str(unit_net * count),
                    "grossKg": str(unit_gross * count),
                    "volumeM3": str(unit_cube * count),
                }
            )
        if remaining:
            raise AssertionError("independent allocation failed conservation")
        declarations = None
        if regime == "dg":
            record_id = request["dangerousGoods"]["recordId"]
            record = self.hmt[record_id]
            declarations = {
                "recordId": record_id,
                "properShippingName": record.proper_shipping_name,
                "unNumber": record.un_number,
                "exactHazardClass": record.exact_hazard_class,
                "hazardCategory": record.hazard_category,
                "packingGroupCode": record.packing_group_code,
                "packingGroupCategory": record.packing_group_category,
                "exactSubsidiaryHazards": list(record.exact_subsidiary_hazards),
                "subsidiaryHazardCategories": list(record.subsidiary_hazard_categories),
            }
            if request["dgRequirements"].get("flashPoint"):
                fact = self.flash[record_id]
                declarations["flashPoint"] = {
                    "valueCelsius": str(fact.value("celsius")),
                    "sourceUrl": fact.source_url,
                    "casNumber": fact.cas_number,
                }
        thermal_context = None
        if thermal:
            thermal_context = {
                "temperatureCelsius": str(
                    Decimal(-24) + Decimal("0.5") * stream.derive("temperature").randbelow(13)
                    if regime == "frozen"
                    else Decimal(0)
                ),
                "ventilationCbmPerHour": 0,
                "basis": (
                    "unambiguous full-registry frozen/chilled ancestry; "
                    "no respiring produce inferred"
                ),
            }
        return {
            "identities": [
                {
                    "hs6": item["hs6"],
                    "phrase": item["description"],
                    "phraseRole": "commercial" if regime == "dg" else "classification_context",
                }
                for item in request["goods"]
            ],
            "profile": {"id": "registry-" + regime, "form": request["goodsForm"]},
            "package": {
                "packageQuantity": quantity,
                "packageTypeFamily": envelope.family,
                "packageMaterial": package_material(selected_category),
            },
            "internalCategory": selected_category,
            "physical": {
                "unitNetKg": str(unit_net),
                "unitGrossKg": str(unit_gross),
                "unitVolumeM3": str(unit_cube),
                "netWeight": {"value": str(unit_net * quantity), "unit": "kilogram"},
                "grossWeight": {"value": str(unit_gross * quantity), "unit": "kilogram"},
                "volume": {"value": str(unit_cube * quantity), "unit": "cubic_metre"},
            },
            "allocations": allocations if container_count else [],
            "equipment": equipment,
            "thermal": thermal_context,
            "dangerousGoods": declarations,
            "provenance": {
                "method": "independent_full_registry_flat_v1",
                "registryPins": self.registry.pins(),
                "requestSha256": canonical_json_sha256(dict(request)),
                "seed": seed,
                "drawKey": draw_key,
                "regime": regime,
                "sourceShipmentDependence": False,
                "packingApproval": False,
                "selectedFlatCategory": selected_category,
                "countedUnitContext": counted_unit_context,
                "equipmentSlotsFixed": equipment_slots is not None,
                "quantityMultiple": multiple * count_step,
                "fixedQuantity": fixed_quantity,
                "rowCapacities": [
                    {"payloadKg": str(payload), "volumeM3": str(cube)}
                    for payload, cube in capacities
                ],
                "numericEnvelope": numeric_authority,
            },
        }


def summarize_requests(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate eligible registry scope from actually selected finite requests."""
    return {
        "requests": len(requests),
        "uniqueHs6": len({goods["hs6"] for request in requests for goods in request["goods"]}),
        "uniqueDgRecords": len(
            {
                request["dangerousGoods"]["recordId"]
                for request in requests
                if request["dangerousGoods"] is not None
            }
        ),
        "regimes": dict(Counter(request["regime"] for request in requests)),
    }
