"""Source-safe structural modeling and realization for transport identities.

Raw source identities are used only by :class:`SourceTransportIdentityGuard`.
They are never columns in the SDV modeling view and are never copied into a
realization.  Voyage generation retains only a character-class shape and draws
every variable character independently.  Vessel renderers are optional,
explicit dependencies because no generic fallback has yet met the project's
lexical-quality gate.  IMO numbers remain absent because this package has no
authoritative assigned-number registry against which to validate inventions.
"""

from __future__ import annotations

import math
import string
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import Any, Literal

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.sdv_harness import BenchmarkView
from document_ocr.synthesis.vessel_lexical import (
    VesselNameRenderer,
    VesselRenderStrategy,
)

TRANSPORT_IDENTITY_VIEW_NAME = "transport_identity_structure"
TRANSPORT_IDENTITY_COLUMNS = (
    "vessel_structure",
    "voyage_shape",
    "source_imo_present",
)
TRANSPORT_IDENTITY_DRIVER_COLUMNS = (
    "vessel_name_present",
    "vessel_word_count",
    "vessel_character_count",
    "vessel_case_style",
    "vessel_digit_count",
    "voyage_number_present",
    "voyage_shape",
    "voyage_letter_count",
    "voyage_digit_count",
    "voyage_separator_count",
    "source_imo_present",
)

# This is a generic word grammar, not a registry of real vessels.  It is paired
# with an exact-length phonotactic fallback.  The full-source guard rejects
# exact and near complete-name collisions before any identity is accepted.
FICTIONAL_VESSEL_RENDERER_ID = "generic_maritime_exact_structure_v5"
_GENERIC_VESSEL_WORDS: Mapping[int, tuple[str, ...]] = MappingProxyType(
    {
        1: ("a", "m", "v"),
        2: ("mv", "ms", "ss"),
        3: ("sea", "sky", "sun", "bay", "ice"),
        4: ("blue", "star", "wave", "dawn", "wind", "tide", "gulf"),
        5: ("ocean", "north", "south", "azure", "coral", "pearl", "light"),
        6: ("marine", "silver", "golden", "island", "breeze", "spirit"),
        7: ("horizon", "voyager", "pacific", "neptune", "mariner", "venture"),
        8: ("atlantic", "majestic", "northern", "southern", "eastward", "westward"),
        9: ("discovery", "navigator", "bluewater", "starlight", "seafarers"),
        10: ("enterprise", "oceanbound", "northlight", "southpoint"),
        11: ("oceanspirit", "silvercrest", "marineglory"),
        12: ("oceanvoyager", "pacificlight", "horizoncrest"),
    }
)
_FICTIONAL_CONSONANTS = "bcdfghjklmnprstvwz"
_FICTIONAL_VOWELS = "aeiou"

_VESSEL_CASE_STYLES = frozenset({"missing", "upper", "lower", "title", "mixed", "caseless"})
_VOYAGE_SHAPE_SYMBOLS = frozenset({"U", "L", "D", "H", "S", "P", "W", "O"})
_IMO_POLICY = "absent_without_authoritative_registry_v1"

CollisionKind = Literal["none", "exact", "near", "absolute_near", "substring_overlap"]
IdentityKind = Literal["vessel_name", "voyage_number"]


@dataclass(frozen=True, slots=True)
class TransportPrivacyPolicy:
    """Explicit privacy boundary for source-distance rejection."""

    minimum_normalized_edit_distance: float
    maximum_attempts: int
    minimum_absolute_edit_distance: int = 2
    maximum_source_substring_fraction: float = 0.70
    minimum_source_substring_characters: int = 5

    def __post_init__(self) -> None:
        distance = self.minimum_normalized_edit_distance
        if not math.isfinite(distance) or distance < 0 or distance >= 1:
            raise ValueError("minimum_normalized_edit_distance must be finite and in [0, 1)")
        if self.maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")
        if self.minimum_absolute_edit_distance < 1:
            raise ValueError("minimum_absolute_edit_distance must be positive")
        if (
            not math.isfinite(self.maximum_source_substring_fraction)
            or not 0 < self.maximum_source_substring_fraction <= 1
        ):
            raise ValueError("maximum_source_substring_fraction must be in (0, 1]")
        if self.minimum_source_substring_characters < 2:
            raise ValueError("minimum_source_substring_characters must be at least two")


@dataclass(frozen=True, slots=True)
class VoyageNumberPolicy:
    """Novelty and bounded-attempt policy for shape-preserving voyages."""

    minimum_normalized_edit_distance: float
    maximum_attempts: int

    def __post_init__(self) -> None:
        distance = self.minimum_normalized_edit_distance
        if not math.isfinite(distance) or distance < 0 or distance >= 1:
            raise ValueError("minimum_normalized_edit_distance must be finite and in [0, 1)")
        if self.maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")


@dataclass(frozen=True, slots=True)
class TransportRealizationFailure:
    """Immutable, identity-safe receipt for bounded realization exhaustion."""

    row_id: str
    identity_kind: IdentityKind
    attempts: int
    reason: Literal["source_or_batch_collision_space_exhausted"]
    minimum_normalized_edit_distance: float


class TransportRealizationError(RuntimeError):
    """A safe realization could not be produced under the declared policy."""

    def __init__(self, receipt: TransportRealizationFailure) -> None:
        self.receipt = receipt
        super().__init__(
            f"{receipt.identity_kind} realization failed for {receipt.row_id!r} "
            f"after {receipt.attempts} attempts"
        )


@dataclass(frozen=True, slots=True)
class TransportIdentityRealization:
    """Generated transport identities and their non-secret provenance."""

    row_id: str
    vessel_name: str | None
    voyage_number: str | None
    vessel_imo_number: None
    vessel_attempts: int
    voyage_attempts: int
    vessel_renderer_id: str
    vessel_render_strategy: VesselRenderStrategy
    imo_policy: str


@dataclass(frozen=True, slots=True)
class VoyageNumberRealization:
    """One independently redrawn, source-safe voyage identity."""

    row_id: str
    voyage_number: str | None
    attempts: int
    method: Literal["observed_character_class_shape_v1"] = "observed_character_class_shape_v1"


@dataclass(frozen=True, slots=True)
class SourceTransportIdentityGuard:
    """Immutable in-memory exact and near-match index over the full source."""

    source_document_count: int
    source_distinct_vessel_count: int
    source_distinct_voyage_count: int
    _vessel_keys: frozenset[str] = field(repr=False)
    _voyage_keys: frozenset[str] = field(repr=False)

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Mapping[str, Any]],
    ) -> SourceTransportIdentityGuard:
        _validate_generic_vessel_grammar()
        document_ids: set[str] = set()
        vessel_keys: set[str] = set()
        voyage_keys: set[str] = set()
        for index, document in enumerate(documents):
            document_id = _required_text(
                document.get("document_id"), f"documents[{index}].document_id"
            )
            if document_id in document_ids:
                raise ValueError(f"duplicate source document_id: {document_id}")
            document_ids.add(document_id)
            vessel = _optional_text(
                document.get("transport_vessel_name"),
                f"documents[{index}].transport_vessel_name",
            )
            voyage = _optional_text(
                document.get("transport_voyage_number"),
                f"documents[{index}].transport_voyage_number",
            )
            if vessel is not None:
                vessel_keys.add(_identity_key(vessel))
            if voyage is not None:
                voyage_keys.add(_identity_key(voyage))

        return cls(
            source_document_count=len(document_ids),
            source_distinct_vessel_count=len(vessel_keys),
            source_distinct_voyage_count=len(voyage_keys),
            _vessel_keys=frozenset(vessel_keys),
            _voyage_keys=frozenset(voyage_keys),
        )

    def classify_vessel_name(
        self, candidate: str, *, policy: TransportPrivacyPolicy
    ) -> CollisionKind:
        return _classify_vessel_collision(_identity_key(candidate), self._vessel_keys, policy)

    def classify_voyage_number(
        self,
        candidate: str,
        *,
        policy: TransportPrivacyPolicy | VoyageNumberPolicy,
    ) -> CollisionKind:
        return _classify_collision(_identity_key(candidate), self._voyage_keys, policy)

    def nearest_vessel_distance(self, candidate: str) -> tuple[float, int]:
        """Return aggregate-safe normalized and absolute distance to the source."""

        diagnostics = self.vessel_distance_diagnostics(candidate)
        return float(diagnostics["normalized"]), int(diagnostics["absolute"])

    def vessel_distance_diagnostics(self, candidate: str) -> dict[str, float | int]:
        """Return aggregate-safe distance and source-substring diagnostics."""

        candidate_key = _identity_key(candidate)
        if not self._vessel_keys:
            raise ValueError("source guard contains no vessel identities")
        normalized = 1.0
        absolute = max(len(candidate_key), 1)
        substring_fraction = 0.0
        substring_characters = 0
        for source_key in self._vessel_keys:
            distance = _levenshtein_distance_keys(candidate_key, source_key)
            normalized = min(distance / max(len(candidate_key), len(source_key)), normalized)
            absolute = min(distance, absolute)
            common = _longest_common_substring_length(candidate_key, source_key)
            substring_characters = max(substring_characters, common)
            substring_fraction = max(substring_fraction, common / max(len(candidate_key), 1))
        return {
            "normalized": normalized,
            "absolute": absolute,
            "substringFraction": substring_fraction,
            "substringCharacters": substring_characters,
        }


@dataclass(frozen=True, slots=True)
class TransportIdentityBundle:
    """A PII-free statistical view paired with its private full-source guard."""

    view: BenchmarkView
    guard: SourceTransportIdentityGuard = field(repr=False)
    source_document_count: int
    fit_document_count: int
    fit_vessel_names: tuple[str | None, ...] = field(repr=False)
    vessel_renderer_id: str = FICTIONAL_VESSEL_RENDERER_ID
    imo_policy: str = _IMO_POLICY


def build_transport_identity_bundle(
    *,
    source_documents: Sequence[Mapping[str, Any]],
    fit_document_ids: Sequence[str],
    template_by_document: Mapping[str, str],
    partition_by_document: Mapping[str, str],
    allowed_partition: str = "train",
) -> TransportIdentityBundle:
    """Build a full-source guard and a fit-only, template-grouped SDV view."""

    if not allowed_partition:
        raise ValueError("allowed_partition must not be empty")
    guard = SourceTransportIdentityGuard.from_documents(source_documents)
    documents_by_id: dict[str, Mapping[str, Any]] = {}
    for document in source_documents:
        document_id = _required_text(document.get("document_id"), "document_id")
        documents_by_id[document_id] = document

    fit_ids = tuple(fit_document_ids)
    if not fit_ids or len(fit_ids) != len(set(fit_ids)) or any(not value for value in fit_ids):
        raise ValueError("fit_document_ids must be non-empty, unique strings")
    missing = sorted(set(fit_ids) - set(documents_by_id))
    if missing:
        raise ValueError("fit_document_ids are absent from source: " + ", ".join(missing))

    rows: list[dict[str, object]] = []
    groups: list[str] = []
    partitions: list[str] = []
    fit_vessel_names: list[str | None] = []
    for document_id in fit_ids:
        try:
            group_id = template_by_document[document_id]
            partition = partition_by_document[document_id]
        except KeyError as error:
            raise ValueError(f"missing lineage for fit document {document_id}") from error
        if not group_id or not partition:
            raise ValueError(f"empty lineage for fit document {document_id}")
        rows.append(_structural_row(documents_by_id[document_id]))
        fit_vessel_names.append(
            _optional_text(
                documents_by_id[document_id].get("transport_vessel_name"),
                "transport_vessel_name",
            )
        )
        groups.append(group_id)
        partitions.append(partition)

    pandas = __import__("pandas")
    data = pandas.DataFrame(rows, columns=TRANSPORT_IDENTITY_COLUMNS)
    view = BenchmarkView(
        name=TRANSPORT_IDENTITY_VIEW_NAME,
        data=data,
        metadata=_metadata(),
        row_ids=fit_ids,
        group_ids=tuple(groups),
        partition_labels=tuple(partitions),
        allowed_partition=allowed_partition,
    )
    return TransportIdentityBundle(
        view=view,
        guard=guard,
        source_document_count=guard.source_document_count,
        fit_document_count=len(fit_ids),
        fit_vessel_names=tuple(fit_vessel_names),
    )


@dataclass(frozen=True, slots=True)
class GenericVesselNameRenderer:
    """Non-lexical baseline renderer retained only for measured comparison."""

    renderer_id: str = FICTIONAL_VESSEL_RENDERER_ID

    def render(
        self,
        *,
        word_count: int,
        character_count: int,
        digit_count: int,
        case_style: str,
        stream: DeterministicStream,
    ) -> tuple[str, VesselRenderStrategy]:
        return _render_fictional_vessel_name(
            word_count=word_count,
            character_count=character_count,
            digit_count=digit_count,
            case_style=case_style,
            stream=stream,
        )

    def accepts(self, candidate: str) -> bool:
        return bool(candidate)


GENERIC_VESSEL_RENDERER = GenericVesselNameRenderer()


def realize_transport_identities(
    *,
    rows: Sequence[Mapping[str, Any]],
    row_ids: Sequence[str],
    stream: DeterministicStream,
    guard: SourceTransportIdentityGuard,
    policy: TransportPrivacyPolicy,
    vessel_renderer: VesselNameRenderer = GENERIC_VESSEL_RENDERER,
) -> tuple[TransportIdentityRealization, ...]:
    """Realize model rows in stable identity order with batch uniqueness."""

    if (
        len(rows) != len(row_ids)
        or len(set(row_ids)) != len(row_ids)
        or any(not value for value in row_ids)
    ):
        raise ValueError("rows require equally sized, unique, non-empty row_ids")
    indexed = sorted(zip(row_ids, rows, strict=True), key=lambda item: item[0])
    used_vessels: set[str] = set()
    used_voyages: set[str] = set()
    realized: list[TransportIdentityRealization] = []
    for row_id, row in indexed:
        vessel_name, vessel_attempts, vessel_strategy = _realize_vessel_name(
            row=row,
            row_id=row_id,
            stream=stream.derive(row_id).derive("vessel-name"),
            guard=guard,
            policy=policy,
            used_keys=used_vessels,
            renderer=vessel_renderer,
        )
        voyage_number, voyage_attempts = _realize_voyage_number(
            row=row,
            row_id=row_id,
            stream=stream.derive(row_id).derive("voyage-number"),
            guard=guard,
            policy=policy,
            used_keys=used_voyages,
        )
        if vessel_name is not None:
            used_vessels.add(_identity_key(vessel_name))
        if voyage_number is not None:
            used_voyages.add(_identity_key(voyage_number))
        realized.append(
            TransportIdentityRealization(
                row_id=row_id,
                vessel_name=vessel_name,
                voyage_number=voyage_number,
                vessel_imo_number=None,
                vessel_attempts=vessel_attempts,
                voyage_attempts=voyage_attempts,
                vessel_renderer_id=vessel_renderer.renderer_id,
                vessel_render_strategy=vessel_strategy,
                imo_policy=_IMO_POLICY,
            )
        )
    return tuple(realized)


def realize_voyage_numbers(
    *,
    rows: Sequence[Mapping[str, Any]],
    row_ids: Sequence[str],
    stream: DeterministicStream,
    guard: SourceTransportIdentityGuard,
    policy: VoyageNumberPolicy,
) -> tuple[VoyageNumberRealization, ...]:
    """Generate voyages without invoking or depending on a vessel renderer.

    Every source letter/digit is reduced to its class before generation.  The
    renderer then independently redraws each variable position while preserving
    source case and literal separators, rejecting full-corpus and batch
    collisions under a bounded deterministic retry policy.
    """

    if (
        len(rows) != len(row_ids)
        or len(set(row_ids)) != len(row_ids)
        or any(not value for value in row_ids)
    ):
        raise ValueError("rows require equally sized, unique, non-empty row_ids")
    indexed = sorted(zip(row_ids, rows, strict=True), key=lambda item: item[0])
    used_voyages: set[str] = set()
    realized: list[VoyageNumberRealization] = []
    for row_id, row in indexed:
        voyage_number, attempts = _realize_voyage_number(
            row=row,
            row_id=row_id,
            stream=stream.derive(row_id).derive("voyage-number"),
            guard=guard,
            policy=policy,
            used_keys=used_voyages,
        )
        if voyage_number is not None:
            used_voyages.add(_identity_key(voyage_number))
        realized.append(
            VoyageNumberRealization(
                row_id=row_id,
                voyage_number=voyage_number,
                attempts=attempts,
            )
        )
    return tuple(realized)


def normalized_edit_distance(left: str, right: str) -> float:
    """Return Levenshtein distance divided by the longer identity-key length."""

    left_key = _identity_key(left)
    right_key = _identity_key(right)
    return _levenshtein_distance_keys(left_key, right_key) / max(len(left_key), len(right_key))


def _levenshtein_distance_keys(left_key: str, right_key: str) -> int:
    if left_key == right_key:
        return 0
    if len(left_key) > len(right_key):
        left_key, right_key = right_key, left_key
    previous = list(range(len(left_key) + 1))
    for right_index, right_character in enumerate(right_key, start=1):
        current = [right_index]
        for left_index, left_character in enumerate(left_key, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def transport_identity_key(value: str) -> str:
    """Return the exact normalization used by source and batch collision guards."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("transport identity key requires non-empty text")
    return _identity_key(value)


def transport_identity_driver_violations(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic reasons a modeled structure cannot be realized.

    SDV proposes the compact structural fields independently.  This validator
    is therefore the explicit task-owned boundary between statistical proposals
    and identity realization; no inconsistent row is silently repaired.
    """

    try:
        vessel_present = _required_bool(row, "vessel_name_present")
        vessel_word_count = _required_int(row, "vessel_word_count", minimum=0)
        vessel_character_count = _required_int(row, "vessel_character_count", minimum=0)
        vessel_digit_count = _required_int(row, "vessel_digit_count", minimum=0)
        vessel_case_style = _required_category(row, "vessel_case_style", _VESSEL_CASE_STYLES)
        voyage_present = _required_bool(row, "voyage_number_present")
        voyage_shape = _required_text(_scalar_value(row.get("voyage_shape")), "voyage_shape")
        voyage_letter_count = _required_int(row, "voyage_letter_count", minimum=0)
        voyage_digit_count = _required_int(row, "voyage_digit_count", minimum=0)
        voyage_separator_count = _required_int(row, "voyage_separator_count", minimum=0)
        _required_bool(row, "source_imo_present")
    except ValueError:
        return ("invalid_scalar_or_category",)

    violations = list(
        _vessel_structure_violations(
            present=vessel_present,
            word_count=vessel_word_count,
            character_count=vessel_character_count,
            digit_count=vessel_digit_count,
            case_style=vessel_case_style,
        )
    )
    if not voyage_present:
        if voyage_shape != "missing":
            violations.append("missing_voyage_shape_nonmissing")
        if voyage_letter_count or voyage_digit_count or voyage_separator_count:
            violations.append("missing_voyage_counts_nonzero")
    else:
        if (
            voyage_shape == "missing"
            or not voyage_shape
            or set(voyage_shape) - _VOYAGE_SHAPE_SYMBOLS
        ):
            violations.append("present_voyage_shape_invalid")
        else:
            letters = sum(symbol in {"U", "L"} for symbol in voyage_shape)
            digits = voyage_shape.count("D")
            separators = sum(symbol in {"H", "S", "P", "W", "O"} for symbol in voyage_shape)
            if voyage_letter_count != letters:
                violations.append("voyage_letter_count_mismatch")
            if voyage_digit_count != digits:
                violations.append("voyage_digit_count_mismatch")
            if voyage_separator_count != separators:
                violations.append("voyage_separator_count_mismatch")
            if letters + digits == 0:
                violations.append("voyage_has_no_variable_symbols")
    return tuple(violations)


def expand_transport_identity_structure(row: Mapping[str, Any]) -> dict[str, object]:
    """Losslessly expand one compact SDV row into deterministic renderer drivers."""

    structure = _required_text(_scalar_value(row.get("vessel_structure")), "vessel_structure")
    voyage_shape = _required_text(_scalar_value(row.get("voyage_shape")), "voyage_shape")
    source_imo_present = _required_bool(row, "source_imo_present")
    if structure == "missing":
        vessel_present = False
        word_count = 0
        character_count = 0
        case_style = "missing"
        digit_count = 0
    else:
        parts = structure.split(":")
        if len(parts) != 5 or parts[0] != "present":
            raise ValueError("vessel_structure has an invalid encoded form")
        try:
            word_count = int(parts[1])
            character_count = int(parts[2])
            digit_count = int(parts[4])
        except ValueError as error:
            raise ValueError("vessel_structure counts are not integers") from error
        if any(
            str(value) != surface
            for value, surface in (
                (word_count, parts[1]),
                (character_count, parts[2]),
                (digit_count, parts[4]),
            )
        ):
            raise ValueError("vessel_structure counts are not canonically encoded")
        vessel_present = True
        case_style = parts[3]

    voyage_present = voyage_shape != "missing"
    expanded: dict[str, object] = {
        "vessel_name_present": vessel_present,
        "vessel_word_count": word_count,
        "vessel_character_count": character_count,
        "vessel_case_style": case_style,
        "vessel_digit_count": digit_count,
        "voyage_number_present": voyage_present,
        "voyage_shape": voyage_shape,
        "voyage_letter_count": (
            sum(symbol in {"U", "L"} for symbol in voyage_shape) if voyage_present else 0
        ),
        "voyage_digit_count": voyage_shape.count("D") if voyage_present else 0,
        "voyage_separator_count": (
            sum(symbol in {"H", "S", "P", "W", "O"} for symbol in voyage_shape)
            if voyage_present
            else 0
        ),
        "source_imo_present": source_imo_present,
    }
    violations = transport_identity_driver_violations(expanded)
    if violations:
        raise ValueError("expanded transport structure is invalid: " + ", ".join(violations))
    return expanded


def _vessel_structure_violations(
    *,
    present: bool,
    word_count: int,
    character_count: int,
    digit_count: int,
    case_style: str,
) -> tuple[str, ...]:
    violations: list[str] = []
    if not present:
        if word_count or character_count or digit_count:
            violations.append("missing_vessel_counts_nonzero")
        if case_style != "missing":
            violations.append("missing_vessel_case_nonmissing")
        return tuple(violations)
    if word_count < 1:
        violations.append("present_vessel_word_count_nonpositive")
    if character_count < 1:
        violations.append("present_vessel_character_count_nonpositive")
    if case_style == "missing":
        violations.append("present_vessel_case_missing")
    separator_count = max(word_count - 1, 0)
    content_count = character_count - separator_count
    if content_count < word_count:
        violations.append("vessel_character_count_too_small_for_words")
    if digit_count > max(content_count, 0):
        violations.append("vessel_digit_count_exceeds_content")
    letter_count = content_count - digit_count
    if case_style == "caseless" and letter_count != 0:
        violations.append("caseless_vessel_contains_letters")
    if case_style not in {"missing", "caseless"} and letter_count < 1:
        violations.append("cased_vessel_has_no_letters")
    if case_style == "mixed" and letter_count < 2:
        violations.append("mixed_vessel_requires_two_letters")
    return tuple(violations)


def _render_fictional_vessel_name(
    *,
    word_count: int,
    character_count: int,
    digit_count: int,
    case_style: str,
    stream: DeterministicStream,
) -> tuple[
    str,
    VesselRenderStrategy,
]:
    separator_count = word_count - 1
    content_count = character_count - separator_count
    letter_count = content_count - digit_count
    if digit_count and word_count > 1 and letter_count >= word_count - 1:
        letter_lengths = _grammar_aware_composition(
            total=letter_count,
            parts=word_count - 1,
            stream=stream.derive("letter-word-lengths"),
        )
        digit_lengths = (digit_count,)
        word_specs = tuple((length, 0) for length in letter_lengths) + tuple(
            (0, length) for length in digit_lengths
        )
    else:
        word_lengths = (
            _grammar_aware_composition(
                total=content_count,
                parts=word_count,
                stream=stream.derive("word-lengths"),
            )
            if digit_count == 0
            else _positive_composition(
                total=content_count,
                parts=word_count,
                stream=stream.derive("word-lengths"),
            )
        )
        remaining_digits = digit_count
        specs: list[tuple[int, int]] = []
        for length in reversed(word_lengths):
            digits_here = min(length, remaining_digits)
            specs.append((length - digits_here, digits_here))
            remaining_digits -= digits_here
        if remaining_digits:
            raise ValueError("vessel digit allocation exceeds generated word capacity")
        word_specs = tuple(reversed(specs))

    words: list[str] = []
    strategies: list[VesselRenderStrategy] = []
    used_words: set[str] = set()
    for word_index, (letters, digits) in enumerate(word_specs):
        word_stream = stream.derive(f"word-{word_index}")
        token = ""
        strategy: VesselRenderStrategy = "generic_vocabulary"
        for token_attempt in range(32):
            token, strategy = _generic_vessel_word(
                letters,
                stream=word_stream.derive("letters").derive(f"attempt-{token_attempt}"),
            )
            if not token or token not in used_words:
                break
        else:
            raise RuntimeError("fictional vessel grammar exhausted unique words")
        if token:
            used_words.add(token)
        strategies.append(strategy)
        if digits:
            digit_stream = word_stream.derive("digits")
            token += "".join(
                string.digits[digit_stream.randbelow(10, counter=index)] for index in range(digits)
            )
        words.append(token)
    candidate = _apply_case_style(" ".join(words), case_style)
    if (
        len(candidate) != character_count
        or len(candidate.split()) != word_count
        or sum(character.isdigit() for character in candidate) != digit_count
        or _case_style(candidate) != case_style
    ):
        raise RuntimeError("fictional vessel renderer violated its exact structure contract")
    if "phonotactic_generated" in strategies:
        overall_strategy: VesselRenderStrategy = "phonotactic_generated"
    elif "generic_compound" in strategies:
        overall_strategy = "generic_compound"
    else:
        overall_strategy = "generic_vocabulary"
    return candidate, overall_strategy


def _grammar_aware_composition(
    *, total: int, parts: int, stream: DeterministicStream
) -> tuple[int, ...]:
    candidates = _positive_compositions(total, parts)
    if not candidates:
        raise ValueError("no positive word-length composition exists")

    def score(values: tuple[int, ...]) -> tuple[int, int, int, int]:
        unsupported = sum(length not in _GENERIC_VESSEL_WORDS for length in values)
        singletons = sum(length == 1 for length in values)
        # One- and two-character maritime prefixes are plausible only as the
        # first token (for example, ``MV``).  Prefer an equally supported
        # composition that does not render them as a trailing pseudo-name.
        misplaced_prefixes = sum(index > 0 and length <= 2 for index, length in enumerate(values))
        spread = max(values) - min(values)
        return unsupported, singletons, misplaced_prefixes, spread

    best_score = min(score(candidate) for candidate in candidates)
    best = tuple(candidate for candidate in candidates if score(candidate) == best_score)
    return best[stream.randbelow(len(best), counter=0)]


@cache
def _positive_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 0:
        return ((),) if total == 0 else ()
    if total < parts or parts < 0:
        return ()
    values: list[tuple[int, ...]] = []
    for first in range(1, total - parts + 2):
        for suffix in _positive_compositions(total - first, parts - 1):
            values.append((first, *suffix))
    return tuple(values)


def _generic_vessel_word(
    length: int, *, stream: DeterministicStream
) -> tuple[
    str,
    VesselRenderStrategy,
]:
    if length == 0:
        return "", "generic_vocabulary"
    if 3 <= length <= 9 and stream.randbelow(5, counter=4) == 0:
        return _phonotactic_word(length, stream=stream), "phonotactic_generated"
    direct = _GENERIC_VESSEL_WORDS.get(length)
    compound_pairs = tuple(
        (left, right)
        for left in sorted(_GENERIC_VESSEL_WORDS)
        for right in sorted(_GENERIC_VESSEL_WORDS)
        if left >= 3 and right >= 3 and left + right == length
    )
    if compound_pairs and (not direct or stream.randbelow(2, counter=5)):
        left_length, right_length = compound_pairs[stream.randbelow(len(compound_pairs), counter=1)]
        left_values = _GENERIC_VESSEL_WORDS[left_length]
        right_values = _GENERIC_VESSEL_WORDS[right_length]
        left = left_values[stream.randbelow(len(left_values), counter=2)]
        right = right_values[stream.randbelow(len(right_values), counter=3)]
        return left + right, "generic_compound"
    if direct:
        return direct[stream.randbelow(len(direct), counter=0)], "generic_vocabulary"
    return _phonotactic_word(length, stream=stream), "phonotactic_generated"


def _positive_composition(
    *, total: int, parts: int, stream: DeterministicStream
) -> tuple[int, ...]:
    if parts == 0:
        if total:
            raise ValueError("zero-part composition requires a zero total")
        return ()
    if parts < 0 or total < parts:
        raise ValueError("positive composition requires total >= positive parts")
    values = [1] * parts
    for counter in range(total - parts):
        values[stream.randbelow(parts, counter=counter)] += 1
    return tuple(values)


def _phonotactic_word(length: int, *, stream: DeterministicStream) -> str:
    if length < 0:
        raise ValueError("fictional word length cannot be negative")
    if length == 0:
        return ""
    starts_with_vowel = bool(stream.randbelow(5, counter=0) == 0)
    output: list[str] = []
    for index in range(length):
        use_vowel = (index % 2 == 0) == starts_with_vowel
        alphabet = _FICTIONAL_VOWELS if use_vowel else _FICTIONAL_CONSONANTS
        output.append(alphabet[stream.randbelow(len(alphabet), counter=index + 1)])
    return "".join(output)


def _realize_vessel_name(
    *,
    row: Mapping[str, Any],
    row_id: str,
    stream: DeterministicStream,
    guard: SourceTransportIdentityGuard,
    policy: TransportPrivacyPolicy,
    used_keys: Collection[str],
    renderer: VesselNameRenderer,
) -> tuple[
    str | None,
    int,
    VesselRenderStrategy,
]:
    present = _required_bool(row, "vessel_name_present")
    word_count = _required_int(row, "vessel_word_count", minimum=0)
    character_count = _required_int(row, "vessel_character_count", minimum=0)
    digit_count = _required_int(row, "vessel_digit_count", minimum=0)
    case_style = _required_category(row, "vessel_case_style", _VESSEL_CASE_STYLES)
    if not present:
        if any((word_count, character_count, digit_count)) or case_style != "missing":
            raise ValueError("missing vessel names require zero counts and missing case style")
        return None, 0, "generic_vocabulary"
    if word_count <= 0 or character_count <= 0 or case_style == "missing":
        raise ValueError("present vessel names require positive size and a non-missing case style")
    vessel_violations = _vessel_structure_violations(
        present=present,
        word_count=word_count,
        character_count=character_count,
        digit_count=digit_count,
        case_style=case_style,
    )
    if vessel_violations:
        raise ValueError("invalid vessel structure: " + ", ".join(vessel_violations))

    for attempt in range(policy.maximum_attempts):
        attempt_stream = stream.derive(f"attempt-{attempt}")
        candidate, strategy = renderer.render(
            word_count=word_count,
            character_count=character_count,
            digit_count=digit_count,
            case_style=case_style,
            stream=attempt_stream,
        )
        candidate_key = _identity_key(candidate)
        if not renderer.accepts(candidate):
            continue
        if candidate_key in used_keys:
            continue
        if guard.classify_vessel_name(candidate, policy=policy) == "none":
            return candidate, attempt + 1, strategy
    raise TransportRealizationError(
        TransportRealizationFailure(
            row_id=row_id,
            identity_kind="vessel_name",
            attempts=policy.maximum_attempts,
            reason="source_or_batch_collision_space_exhausted",
            minimum_normalized_edit_distance=policy.minimum_normalized_edit_distance,
        )
    )


def _realize_voyage_number(
    *,
    row: Mapping[str, Any],
    row_id: str,
    stream: DeterministicStream,
    guard: SourceTransportIdentityGuard,
    policy: TransportPrivacyPolicy | VoyageNumberPolicy,
    used_keys: Collection[str],
) -> tuple[str | None, int]:
    present = _required_bool(row, "voyage_number_present")
    shape = _required_text(_scalar_value(row.get("voyage_shape")), "voyage_shape")
    letter_count = _required_int(row, "voyage_letter_count", minimum=0)
    digit_count = _required_int(row, "voyage_digit_count", minimum=0)
    separator_count = _required_int(row, "voyage_separator_count", minimum=0)
    if not present:
        if shape != "missing" or any((letter_count, digit_count, separator_count)):
            raise ValueError("missing voyages require missing shape and zero counts")
        return None, 0
    if shape == "missing" or not shape or set(shape) - _VOYAGE_SHAPE_SYMBOLS:
        raise ValueError("present voyage_shape contains unsupported symbols")
    if sum(symbol in {"U", "L"} for symbol in shape) != letter_count:
        raise ValueError("voyage_letter_count differs from voyage_shape")
    if shape.count("D") != digit_count:
        raise ValueError("voyage_digit_count differs from voyage_shape")
    if sum(symbol in {"H", "S", "P", "W", "O"} for symbol in shape) != separator_count:
        raise ValueError("voyage_separator_count differs from voyage_shape")

    alphabets = {"U": string.ascii_uppercase, "L": string.ascii_lowercase, "D": string.digits}
    literals = {"H": "-", "S": "/", "P": ".", "W": " ", "O": "_"}
    for attempt in range(policy.maximum_attempts):
        attempt_stream = stream.derive(f"attempt-{attempt}")
        output: list[str] = []
        variable_index = 0
        for symbol in shape:
            if symbol in literals:
                output.append(literals[symbol])
            else:
                alphabet = alphabets[symbol]
                output.append(
                    alphabet[attempt_stream.randbelow(len(alphabet), counter=variable_index)]
                )
                variable_index += 1
        candidate = "".join(output)
        candidate_key = _identity_key(candidate)
        if candidate_key in used_keys:
            continue
        if guard.classify_voyage_number(candidate, policy=policy) == "none":
            return candidate, attempt + 1
    raise TransportRealizationError(
        TransportRealizationFailure(
            row_id=row_id,
            identity_kind="voyage_number",
            attempts=policy.maximum_attempts,
            reason="source_or_batch_collision_space_exhausted",
            minimum_normalized_edit_distance=policy.minimum_normalized_edit_distance,
        )
    )


def _structural_row(document: Mapping[str, Any]) -> dict[str, object]:
    vessel = _optional_text(document.get("transport_vessel_name"), "transport_vessel_name")
    voyage = _optional_text(document.get("transport_voyage_number"), "transport_voyage_number")
    imo = _optional_text(document.get("transport_vessel_imo_number"), "transport_vessel_imo_number")
    voyage_shape = _voyage_shape(voyage) if voyage is not None else "missing"
    if vessel is None:
        vessel_structure = "missing"
    else:
        vessel_structure = ":".join(
            (
                "present",
                str(len(vessel.split())),
                str(len(vessel)),
                _case_style(vessel),
                str(sum(character.isdigit() for character in vessel)),
            )
        )
    compact = {
        "vessel_structure": vessel_structure,
        "voyage_shape": voyage_shape,
        "source_imo_present": imo is not None,
    }
    expand_transport_identity_structure(compact)
    return compact


def _metadata() -> Mapping[str, Any]:
    categorical = {"vessel_structure", "voyage_shape"}
    boolean = {"source_imo_present"}
    columns: dict[str, dict[str, str]] = {}
    for column in TRANSPORT_IDENTITY_COLUMNS:
        if column in categorical:
            columns[column] = {"sdtype": "categorical"}
        elif column in boolean:
            columns[column] = {"sdtype": "boolean"}
        else:
            raise RuntimeError(f"transport identity metadata omitted column {column}")
    return MappingProxyType(
        {
            "tables": {TRANSPORT_IDENTITY_VIEW_NAME: {"columns": columns}},
            "relationships": [],
        }
    )


def _voyage_shape(value: str) -> str:
    symbols: list[str] = []
    separators = {"-": "H", "/": "S", ".": "P"}
    for character in value:
        if character.isalpha():
            symbols.append("L" if character.islower() else "U")
        elif character.isdigit():
            symbols.append("D")
        elif character.isspace():
            symbols.append("W")
        else:
            symbols.append(separators.get(character, "O"))
    return "".join(symbols)


def _case_style(value: str | None) -> str:
    if value is None:
        return "missing"
    if not any(character.isalpha() for character in value):
        return "caseless"
    if value.isupper():
        return "upper"
    if value.islower():
        return "lower"
    if value.istitle():
        return "title"
    return "mixed"


def _apply_case_style(value: str, case_style: str) -> str:
    if case_style == "upper" or case_style == "caseless":
        return value.upper()
    if case_style == "lower":
        return value.lower()
    if case_style == "title":
        return value.title()
    if case_style == "mixed":
        letter_index = 0
        output: list[str] = []
        for character in value.lower():
            if character.isalpha():
                output.append(character.upper() if letter_index % 2 else character)
                letter_index += 1
            else:
                output.append(character)
        return "".join(output)
    raise ValueError(f"unsupported present vessel case style: {case_style}")


def _classify_collision(
    candidate_key: str,
    source_keys: Collection[str],
    policy: TransportPrivacyPolicy | VoyageNumberPolicy,
) -> CollisionKind:
    if candidate_key in source_keys:
        return "exact"
    threshold = policy.minimum_normalized_edit_distance
    if threshold == 0:
        return "none"
    for source_key in source_keys:
        maximum_length = max(len(candidate_key), len(source_key))
        if abs(len(candidate_key) - len(source_key)) / maximum_length > threshold:
            continue
        if normalized_edit_distance(candidate_key, source_key) <= threshold:
            return "near"
    return "none"


def _classify_vessel_collision(
    candidate_key: str,
    source_keys: Collection[str],
    policy: TransportPrivacyPolicy,
) -> CollisionKind:
    if candidate_key in source_keys:
        return "exact"
    for source_key in source_keys:
        distance = _levenshtein_distance_keys(candidate_key, source_key)
        if distance < policy.minimum_absolute_edit_distance:
            return "absolute_near"
        normalized = distance / max(len(candidate_key), len(source_key))
        if normalized <= policy.minimum_normalized_edit_distance:
            return "near"
        common = _longest_common_substring_length(candidate_key, source_key)
        if (
            common >= policy.minimum_source_substring_characters
            and common / max(len(candidate_key), 1) >= policy.maximum_source_substring_fraction
        ):
            return "substring_overlap"
    return "none"


def _longest_common_substring_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    maximum = 0
    for left_character in left:
        current = [0]
        for index, right_character in enumerate(right, start=1):
            value = previous[index - 1] + 1 if left_character == right_character else 0
            current.append(value)
            maximum = max(maximum, value)
        previous = current
    return maximum


def _identity_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    key = "".join(character for character in normalized if character.isalnum())
    if not key:
        key = "".join(character for character in normalized if not character.isspace())
    if not key:
        raise ValueError("transport identity must contain a non-whitespace character")
    return key


def _validate_generic_vessel_grammar() -> None:
    words: list[str] = []
    for expected_length, values in _GENERIC_VESSEL_WORDS.items():
        if expected_length < 1 or not values:
            raise RuntimeError("generic vessel grammar contains an empty length bucket")
        for value in values:
            if (
                len(value) != expected_length
                or not value.isascii()
                or not value.isalpha()
                or not value.islower()
            ):
                raise RuntimeError("generic vessel grammar contains an invalid word")
            words.append(value)
    if len(words) != len(set(words)):
        raise RuntimeError("generic vessel grammar contains duplicate words")


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null")
    stripped = value.strip()
    return stripped or None


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_bool(row: Mapping[str, Any], column: str) -> bool:
    value = _scalar_value(row.get(column))
    if type(value) is not bool:
        raise ValueError(f"{column} must be boolean")
    return value


def _required_int(row: Mapping[str, Any], column: str, *, minimum: int) -> int:
    value = _scalar_value(row.get(column))
    if type(value) is not int or value < minimum:
        raise ValueError(f"{column} must be an integer >= {minimum}")
    return value


def _required_category(row: Mapping[str, Any], column: str, allowed: Collection[str]) -> str:
    value = _scalar_value(row.get(column))
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{column} must be one of: {', '.join(sorted(allowed))}")
    return value


def _scalar_value(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return value
