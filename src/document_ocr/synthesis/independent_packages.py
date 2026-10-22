"""Goods-conditioned flat package choices and explicit coverage accounting.

The caller owns registry sampling, native calls, budgets, receipts and final
review. This module neither reads shipment donors nor admits generated samples.
Plausible alternatives are not regulatory packing approval or numeric capacity.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Container, Mapping, Sequence, Set
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    create_model,
    model_validator,
)

from document_ocr.synthesis.independent_envelopes import PACKAGE_COUNTING_CONTEXTS
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES

PACKAGE_TYPES = tuple(sorted(PACKAGE_ALIASES))
_PACKAGE_TYPES = frozenset(PACKAGE_TYPES)
if len(PACKAGE_TYPES) != 48 or set(PACKAGE_COUNTING_CONTEXTS) != _PACKAGE_TYPES:
    raise ValueError("independent package choice requires the complete flat48 vocabulary")

if TYPE_CHECKING:
    PackageType = str
else:
    PackageType = Literal.__getitem__(PACKAGE_TYPES)
_PACKAGE_MATERIALS = MappingProxyType(
    {
        "PACKAGE_BAG_PAPER": "PAPER",
        "PACKAGE_BAG_PAPER_MULTI_WALL": "PAPER",
        "PACKAGE_BAG_PLASTIC": "PLASTIC",
        "PACKAGE_BOX_FIBREBOARD": "FIBREBOARD",
        "PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY": "WOOD",
        "PACKAGE_CASE_WOODEN": "WOOD",
        "PACKAGE_CRATE_WOODEN": "WOOD",
        "PACKAGE_DRUM_FIBRE": "FIBRE",
        "PACKAGE_DRUM_PLASTIC": "PLASTIC",
        "PACKAGE_DRUM_STEEL": "STEEL",
        "PACKAGE_PALLET_WOODEN": "WOOD",
    }
)
_NonEmptyText = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Hs6 = Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", strict=True, frozen=True)


def package_material(category: str) -> str | None:
    """Private wording authority from a specific flat type, never cargo inference."""
    if category not in _PACKAGE_TYPES:
        raise ValueError("package material requires a permitted flat package category")
    return _PACKAGE_MATERIALS.get(category)


class PackageChoiceGoods(BaseModel):
    """A sampled registry identity, not a source document's commodity donor."""

    model_config = _STRICT

    hs6: _Hs6
    description: _NonEmptyText


class PackageGoodsContext(BaseModel):
    """Only directly relevant goods/transport facts enter native packing decisions."""

    model_config = _STRICT

    goods: list[PackageChoiceGoods] = Field(min_length=1)
    regime: Literal["ambient", "chilled", "frozen", "dg"]
    equipment: _NonEmptyText
    form: _NonEmptyText | None = None
    dangerousGoods: dict[str, JsonValue] | None = None
    packingContext: _NonEmptyText | None = None

    @model_validator(mode="after")
    def coherent_dg_context(self) -> PackageGoodsContext:
        if self.regime == "dg" and not self.dangerousGoods:
            raise ValueError("DG package choice requires the supplied genuine DG identity")
        if self.dangerousGoods is not None and self.regime != "dg":
            raise ValueError("DG identity contradicts a non-DG regime")
        return self


class PackageChoiceRequest(PackageGoodsContext):
    """The optional candidate is independently judged, never a forced choice."""

    requiredCategory: PackageType | None = None
    allowedPackageTypes: list[PackageType] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def valid_presentation_scope(self) -> PackageChoiceRequest:
        if self.allowedPackageTypes is not None:
            if len(self.allowedPackageTypes) != len(set(self.allowedPackageTypes)):
                raise ValueError("duplicate presentation package type")
            if (
                self.requiredCategory is not None
                and self.requiredCategory not in self.allowedPackageTypes
            ):
                raise ValueError("candidate is outside presentation package scope")
        return self


def equipment_choice_brief(slots: Sequence[Mapping[str, str]]) -> str:
    """Describe the same host-owned hardware used by the subsequent fact draw."""
    if isinstance(slots, (str, bytes)) or not isinstance(slots, Sequence):
        raise ValueError("carrying hardware must be an explicit sequence of equipment rows")
    if any(
        not isinstance(slot, Mapping)
        or set(slot) != {"sizeCategory", "typeCategory"}
        or not all(isinstance(value, str) and value for value in slot.values())
        for slot in slots
    ):
        raise ValueError("carrying hardware rows require explicit size and type categories")
    if not slots:
        return "Maritime shipment with no printed carrying containers; do not invent equipment."
    if {slot["typeCategory"] for slot in slots} == {"PLATFORM"}:
        return (
            "Fixed sampled carrying platforms: "
            + json.dumps(list(slots), sort_keys=True)
            + ". These are platform carrying units with no enclosed-box doors or walls, "
            "not ordinary GP containers or ro-ro service. Goods must be supportable and "
            "securable on the sampled platform; do not invent an enclosed carrying box, "
            "dismantling or an out-of-gauge declaration. No tank, liner, bladder or liquid-"
            "containment system is supplied; liquids need self-contained discrete "
            "receptacles rather than uncontained bulk loading."
        )
    if {slot["typeCategory"] for slot in slots} == {"OPEN_TOP"}:
        return (
            "Fixed sampled carrying containers: "
            + json.dumps(list(slots), sort_keys=True)
            + ". These are open-top containers with side walls, not flat racks or ro-ro "
            "service. Goods may be top-loaded but must remain within the equipment's "
            "normal length, width and height. Do not invent dismantling, over-height "
            "or other out-of-gauge declarations. No tank, liner, bladder or liquid-"
            "containment system is supplied; liquids need self-contained discrete "
            "receptacles rather than uncontained bulk loading."
        )
    return (
        "Fixed sampled carrying containers: "
        + json.dumps(list(slots), sort_keys=True)
        + ". These are closed-box containers, not flat racks, open tops, vehicle decks "
        "or ro-ro service. Intact goods must fit through their normal doors and inside "
        "them; do not invent dismantling or out-of-gauge transport."
    )


PACKAGE_CHOICE_SYSTEM = """Choose plausible maritime extraction-training packages.
Use only the supplied flat categories, once each. Consider ALL goods together,
their actual form and available equipment. Unusual but workable choices are fine;
legal DG packing approval is not requested. Preserve intact goods and the supplied
form; do not dismantle a machine or invent an inner liquid container to rescue an
incompatible package. A shared outer crate or pallet can hold normal inner packs.
The accounting package is not the ocean freight container, and need not be the
outermost receptacle: coordinated product sets can be SET inside cartons or pallets.
Every alternative must apply to the WHOLE shared goods group, not a union of
individual items' options. Material-specific categories are prospective packing
choices, not cargo-material inference. If complete goods cannot fit the equipment
or essential form is unknown, return alternatives=[]; this explicitly means UNKNOWN.
Never change frozen goods to ambient. Cargo as large as its carrying container's
exterior cannot fit its smaller interior; do not recast cargo as transport equipment.
Each alternatives entry owns packageType and declaredUnitContext together. For
ordinary categories declaredUnitContext is null; for sensitive categories choose
one permitted literal counting context from the goods, never the source document.
Do not invent counts, masses, dimensions or capacities. Keep goodsForm concise.
Independently judge a supplied candidate; otherwise candidateCompatible is null.
If candidateCompatible is true, include that candidate exactly once as a complete
alternatives entry with its permitted declaredUnitContext. If false, exclude it.
Preserve UNKNOWN when no alternative is viable. PACKAGE_SET requires a coordinated
goods set explicitly supplied in the goods description or form; do not invent a
kit, assortment or retail bundle to turn ordinary fluid or bulk goods into a SET.
Genuine supplied sets can still have carton/pallet alternatives. No quotas are supplied.
"""


def _alternative_types(categories: Sequence[str] = PACKAGE_TYPES) -> tuple[type[BaseModel], ...]:
    groups: dict[tuple[str, ...], list[str]] = {}
    for category in categories:
        contexts = tuple(sorted(PACKAGE_COUNTING_CONTEXTS.get(category, ())))
        groups.setdefault(contexts, []).append(category)
    return tuple(
        create_model(
            f"IndependentPackageAlternative{index}",
            __config__=_STRICT,
            packageType=(Literal.__getitem__(tuple(categories)), ...),
            declaredUnitContext=(Literal.__getitem__(contexts) if contexts else type(None), ...),
        )
        for index, (contexts, categories) in enumerate(sorted(groups.items()))
    )


_ALTERNATIVE_TYPES = _alternative_types()
if TYPE_CHECKING:

    class PackageAlternative(BaseModel):
        packageType: str
        declaredUnitContext: str | None

else:
    PackageAlternative = Union.__getitem__(_ALTERNATIVE_TYPES)


class PackageChoice(BaseModel):
    """Native alternatives; empty explicitly means UNKNOWN, never a default."""

    model_config = _STRICT

    alternatives: list[PackageAlternative] = Field(min_length=0, max_length=8)
    goodsForm: _NonEmptyText = Field(description="Brief physical form; audit text, not host facts.")
    candidateCompatible: bool | None

    @property
    def package_types(self) -> tuple[str, ...]:
        return tuple(row.packageType for row in self.alternatives)

    @property
    def unknown(self) -> bool:
        return not self.alternatives

    @property
    def counting_contexts(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                row.packageType: row.declaredUnitContext
                for row in self.alternatives
                if row.declaredUnitContext is not None
            }
        )

    @model_validator(mode="after")
    def distinct_alternatives(self) -> PackageChoice:
        if len(self.alternatives) != len(set(self.package_types)):
            raise ValueError("duplicate package alternative")
        return self


class _ChoiceWithCandidate(PackageChoice):
    candidateCompatible: bool


class _ChoiceWithoutCandidate(PackageChoice):
    candidateCompatible: None


def _request(value: PackageChoiceRequest | Mapping[str, Any]) -> PackageChoiceRequest:
    return (
        value
        if isinstance(value, PackageChoiceRequest)
        else PackageChoiceRequest.model_validate(value)
    )


def package_choice_output_type(
    request: PackageChoiceRequest | Mapping[str, Any],
) -> type[PackageChoice]:
    """Expose known candidate-presence constraints to native decoding itself."""
    value = _request(request)
    if value.allowedPackageTypes is not None:
        return _scoped_choice_type(
            tuple(value.allowedPackageTypes), value.requiredCategory is not None
        )
    return _ChoiceWithCandidate if value.requiredCategory is not None else _ChoiceWithoutCandidate


@lru_cache(maxsize=48)
def _scoped_choice_type(categories: tuple[str, ...], candidate: bool) -> type[PackageChoice]:
    """Expose the reviewed domain and its cardinality to constrained decoding."""
    alternative = Union.__getitem__(_alternative_types(categories))
    identity = hashlib.sha256(json.dumps([categories, candidate]).encode()).hexdigest()[:12]
    return create_model(
        "ScopedIndependentPackageChoice" + identity,
        __base__=_ChoiceWithCandidate if candidate else _ChoiceWithoutCandidate,
        alternatives=(list[alternative], Field(min_length=0, max_length=min(8, len(categories)))),
    )


@lru_cache(maxsize=1)
def _category_prompt() -> str:
    return "\n".join(
        ["Available flat package categories:"]
        + [category + ": " + PACKAGE_ALIASES[category].singular for category in PACKAGE_TYPES]
        + [
            "Each alternative is {packageType, declaredUnitContext}. Ordinary context is null.",
            "Sensitive categories and their permitted counting contexts:",
            json.dumps(
                {
                    category: sorted(values)
                    for category, values in PACKAGE_COUNTING_CONTEXTS.items()
                    if values
                },
                sort_keys=True,
            ),
            "Select contexts from supplied goods, not the source document. No counts requested.",
            "If no viable alternative exists, return alternatives=[] (UNKNOWN).",
        ]
    )


def package_choice_prompt(
    request: PackageChoiceRequest | Mapping[str, Any], *, hs_codes: Container[str]
) -> str:
    """Require caller-owned registry membership; never load an implicit artifact."""
    value = _request(request)
    if any(goods.hs6 not in hs_codes for goods in value.goods):
        raise ValueError("package-choice goods must belong to the supplied HS registry")
    lines = ["Complete shared goods group:"]
    lines.extend(f"- HS {goods.hs6}: {goods.description}" for goods in value.goods)
    lines.extend(
        ["Transport: maritime", "Regime: " + value.regime, "Equipment: " + value.equipment]
    )
    for name in ("form", "dangerousGoods", "packingContext"):
        context = getattr(value, name)
        if context is not None:
            lines.append(name + ": " + json.dumps(context, ensure_ascii=False, sort_keys=True))
    if value.requiredCategory is not None:
        lines.append("Candidate package to judge independently: " + value.requiredCategory)
    lines.append(_category_prompt())
    if value.allowedPackageTypes is not None:
        lines.append(
            "Reviewed source presentation permits only these declared package types: "
            + json.dumps(value.allowedPackageTypes)
            + ". Restrict alternatives to these types or return UNKNOWN. "
            "This constrains presentation, not whether the goods are plausible."
        )
    return "\n".join(lines)


def validate_package_choice(
    request: PackageChoiceRequest | Mapping[str, Any],
    value: Mapping[str, Any] | PackageChoice,
) -> PackageChoice:
    """Hold malformed or contradictory answers; never insert an omitted choice."""
    request = _request(request)
    data = value.model_dump(mode="python") if isinstance(value, PackageChoice) else value
    answer = package_choice_output_type(request).model_validate(data)
    candidate = request.requiredCategory
    if request.allowedPackageTypes is not None and not set(answer.package_types) <= set(
        request.allowedPackageTypes
    ):
        raise ValueError("native alternatives violate reviewed presentation package scope")
    if candidate is not None and (candidate in answer.package_types) != answer.candidateCompatible:
        raise ValueError("candidate compatibility and complete alternative disagree")
    if request.regime == "dg" and "non_dg_liquid_system" in answer.counting_contexts.values():
        raise ValueError("a non-DG liquid-system context contradicts the supplied DG regime")
    return answer


def package_choice_facts(answer: PackageChoice) -> dict[str, Any]:
    """Only typed choices are operative; generated goodsForm remains audit-only."""
    return answer.model_dump(mode="json", exclude={"goodsForm"})


def verify_native_choice_evidence(
    root: Path,
    frozen_request: Mapping[str, Any],
    *,
    provider: Mapping[str, Any],
    hs_codes: Container[str],
) -> PackageChoice:
    """Validate selection against received native evidence, not a forgeable stage label."""
    from document_ocr.synthesis.independent_native_evidence import verify_native_evidence

    evidence = frozen_request.get("nativeChoiceEvidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("frozen package choice lacks native evidence")
    request = PackageChoiceRequest.model_validate(evidence["request"])
    expected = {
        "goods": frozen_request["goods"],
        "regime": frozen_request["regime"],
        "form": frozen_request["goodsForm"],
        "dangerousGoods": frozen_request["dangerousGoods"],
    }
    data = request.model_dump(mode="json")
    presentation = frozen_request.get("cargoPresentation")
    expected_scope = None if presentation is None else presentation["packageTypes"]
    if request.allowedPackageTypes != expected_scope:
        raise ValueError("native choice package scope differs from reviewed presentation")
    if any(data[key] != value for key, value in expected.items()):
        raise ValueError("native choice goods/operational context differs from frozen request")
    if "equipmentSlots" in frozen_request:
        slots = frozen_request["equipmentSlots"]
        if not isinstance(slots, list):
            raise ValueError("frozen carrying hardware must be an explicit list")
        if data["equipment"] != equipment_choice_brief(slots):
            raise ValueError("native choice equipment differs from frozen carrying hardware")
    output = verify_native_evidence(
        root,
        evidence,
        stage="independent-package-choice",
        identity=f"{frozen_request['sourceDocumentId']}:{frozen_request['variant']}",
        request=data,
        model=package_choice_output_type(request),
        system=PACKAGE_CHOICE_SYSTEM,
        prompt=package_choice_prompt(request, hs_codes=hs_codes),
        provider=provider,
    )
    answer = validate_package_choice(request, output)
    selected = frozen_request.get("selectedCategory")
    if (
        selected not in answer.package_types
        or "countedUnitContext" not in frozen_request
        or frozen_request["countedUnitContext"] != answer.counting_contexts.get(selected)
    ):
        raise ValueError("frozen category/counting context is not an approved native alternative")
    return answer


def _counts(value: Mapping[str, int]) -> None:
    if any(
        category not in _PACKAGE_TYPES or type(count) is not int or count < 0
        for category, count in value.items()
    ):
        raise ValueError("package counts require permitted categories and nonnegative integers")


def select_package_type(
    alternatives: Sequence[str],
    *,
    targets: Mapping[str, int],
    accepted: Mapping[str, int],
    reserved: Mapping[str, int],
    used: Set[str],
    seed: int,
    key: str,
) -> str:
    """Prefer soft deficits ONLY among approved alternatives; never count a reservation as final."""
    if not alternatives or len(set(alternatives)) != len(alternatives):
        raise ValueError("selection requires distinct nonempty approved alternatives")
    if (
        any(category not in _PACKAGE_TYPES for category in alternatives)
        or not used <= _PACKAGE_TYPES
    ):
        raise ValueError("selection contains a category outside the flat vocabulary")
    if type(seed) is not int or not isinstance(key, str) or not key:
        raise ValueError("selection requires an integer seed and nonempty request key")
    _counts(targets)
    _counts(accepted)
    _counts(reserved)
    best = None
    candidates = []
    for category in alternatives:
        usage = accepted.get(category, 0) + reserved.get(category, 0)
        rank = (-max(0, targets.get(category, 0) - usage), category in used, usage)
        if best is None or rank < best:
            best, candidates = rank, [category]
        elif rank == best:
            candidates.append(category)
    if len(candidates) == 1:
        return candidates[0]
    return min(
        candidates,
        key=lambda category: hashlib.sha256(
            json.dumps([seed, key, category], sort_keys=True).encode()
        ).digest(),
    )


class _AcceptedPackageOutcome(BaseModel):
    model_config = _STRICT

    status: Literal["accepted"]
    finalCategory: PackageType
    candidateSha256: _Sha256
    reviewEvidence: _NonEmptyText


class _HeldPackageOutcome(BaseModel):
    model_config = _STRICT

    status: Literal["hold"]
    reason: _NonEmptyText


def realized_package_counts(
    outcomes: Mapping[str, Mapping[str, Any]],
) -> tuple[Counter[str], int]:
    """Count reviewed final categories, not proposals, reservations or held documents."""
    accepted: Counter[str] = Counter()
    held = 0
    witnesses = set()
    for key, outcome in outcomes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("final outcomes require nonempty request keys")
        if outcome.get("status") == "accepted":
            value = _AcceptedPackageOutcome.model_validate(outcome)
            if value.candidateSha256 in witnesses:
                raise ValueError("the same final candidate cannot count twice")
            witnesses.add(value.candidateSha256)
            accepted[value.finalCategory] += 1
        else:
            _HeldPackageOutcome.model_validate(outcome)
            held += 1
    return accepted, held


def package_coverage(
    targets: Mapping[str, int], accepted: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    """Expose actual shortfalls instead of silently substituting specific categories."""
    _counts(targets)
    _counts(accepted)
    return {
        category: {
            "target": target,
            "acceptedFinal": accepted.get(category, 0),
            "shortfall": max(0, target - accepted.get(category, 0)),
            "surplus": max(0, accepted.get(category, 0) - target),
        }
        for category, target in targets.items()
    }
