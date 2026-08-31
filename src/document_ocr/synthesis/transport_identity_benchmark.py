"""Grouped SDV comparison for source-safe vessel/voyage structures.

The statistical candidates see structural features only.  Complete source
identities remain inside :class:`SourceTransportIdentityGuard`; every accepted
proposal is realized by the same deterministic grammar and checked against the
full source before scoring.  This separates SDV method quality from the shared
identity renderer and makes the SDV comparison paired and auditable.  A
separate grouped comparison fits character n-gram renderers on each fold's
training names and evaluates them against held-out lexical distributions before
any structural candidate is materialized.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import canonical_json_bytes, sha256_file
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.sdv_evaluation import (
    CandidateRunScore,
    EvaluationBundle,
    ModelSelection,
    evaluate_single_table,
    select_candidate,
)
from document_ocr.synthesis.sdv_harness import (
    DEFAULT_COMPLEXITY_RANKS,
    BenchmarkView,
    CandidateName,
    CandidateSpec,
    GroupedFold,
    PhaseResources,
    ProposalReceipt,
    dataframe_sha256,
    fit_candidate_model,
    grouped_folds,
    preflight_sdv_accelerator,
)
from document_ocr.synthesis.transport_identity import (
    FICTIONAL_VESSEL_RENDERER_ID,
    GENERIC_VESSEL_RENDERER,
    TRANSPORT_IDENTITY_COLUMNS,
    TRANSPORT_IDENTITY_DRIVER_COLUMNS,
    TRANSPORT_IDENTITY_VIEW_NAME,
    SourceTransportIdentityGuard,
    TransportIdentityBundle,
    TransportPrivacyPolicy,
    expand_transport_identity_structure,
    realize_transport_identities,
    transport_identity_driver_violations,
    transport_identity_key,
)
from document_ocr.synthesis.vessel_lexical import (
    CharacterNGramVesselRenderer,
    VesselNameRenderer,
    deterministic_real_split,
    fit_character_ngram_vessel_renderer,
    lexical_discriminator_auc,
    lexical_realism_metrics,
    vessel_name_fit_exclusion_reason,
)

_EXPECTED_CANDIDATES: tuple[CandidateName, ...] = (
    "empirical",
    "gaussian_copula",
    "ctgan",
    "tvae",
)
_SELECTABLE_CANDIDATES: tuple[CandidateName, ...] = (
    "gaussian_copula",
    "ctgan",
    "tvae",
)


class TransportIdentityBenchmarkError(RuntimeError):
    """The transport comparison violated its immutable experiment contract."""


def transport_identity_candidate_specs(
    *, neural_epochs: int, neural_batch_size: int
) -> tuple[CandidateSpec, ...]:
    """Return four fully explicit candidates; neural arms require CUDA."""

    if type(neural_epochs) is not int or neural_epochs < 1:
        raise ValueError("neural_epochs must be a positive integer")
    if type(neural_batch_size) is not int or neural_batch_size < 10 or neural_batch_size % 10:
        raise ValueError("neural_batch_size must be positive and divisible by CTGAN pac=10")
    bounded = {"enforce_min_max_values": True, "enforce_rounding": True}
    return (
        CandidateSpec("empirical", {}),
        CandidateSpec(
            "gaussian_copula",
            {
                **bounded,
                "locales": ["en_US"],
                "numerical_distributions": {},
                "default_distribution": "beta",
            },
        ),
        CandidateSpec(
            "ctgan",
            {
                **bounded,
                "locales": ["en_US"],
                "embedding_dim": 128,
                "generator_dim": [256, 256],
                "discriminator_dim": [256, 256],
                "generator_lr": 0.0002,
                "generator_decay": 0.000001,
                "discriminator_lr": 0.0002,
                "discriminator_decay": 0.000001,
                "batch_size": neural_batch_size,
                "discriminator_steps": 1,
                "log_frequency": True,
                "verbose": False,
                "epochs": neural_epochs,
                "pac": 10,
                "enable_gpu": True,
            },
        ),
        CandidateSpec(
            "tvae",
            {
                **bounded,
                "embedding_dim": 128,
                "compress_dims": [128, 128],
                "decompress_dims": [128, 128],
                "l2scale": 0.00001,
                "batch_size": neural_batch_size,
                "verbose": False,
                "epochs": neural_epochs,
                "loss_factor": 2,
                "enable_gpu": True,
            },
        ),
    )


@dataclass(frozen=True, slots=True)
class TransportBenchmarkSettings:
    candidates: tuple[CandidateSpec, ...]
    fold_count: int
    fold_seed: int
    seeds: tuple[int, ...]
    proposal_multiplier: int
    proposal_batch_rows: int | None
    quality_margin: float
    stability_penalty: float
    privacy_policy: TransportPrivacyPolicy
    example_rows_per_candidate: int = 8
    production_pilot_rows: int = 50
    lexical_ngram_orders: tuple[int, ...] = (2, 3, 4)
    lexical_quality_margin: float = 0.01
    lexical_stability_penalty: float = 0.25

    def __post_init__(self) -> None:
        if tuple(candidate.name for candidate in self.candidates) != _EXPECTED_CANDIDATES:
            raise ValueError(f"transport benchmark requires candidates {_EXPECTED_CANDIDATES}")
        if type(self.fold_count) is not int or self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if type(self.fold_seed) is not int or not 0 <= self.fold_seed < 2**32:
            raise ValueError("fold_seed must be uint32")
        if (
            not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)
        ):
            raise ValueError("seeds must be unique strict uint32 values")
        if type(self.proposal_multiplier) is not int or self.proposal_multiplier < 1:
            raise ValueError("proposal_multiplier must be positive")
        if self.proposal_batch_rows is not None and (
            type(self.proposal_batch_rows) is not int or self.proposal_batch_rows < 1
        ):
            raise ValueError("proposal_batch_rows must be positive when supplied")
        for label, value in (
            ("quality_margin", self.quality_margin),
            ("stability_penalty", self.stability_penalty),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if type(self.example_rows_per_candidate) is not int or self.example_rows_per_candidate < 1:
            raise ValueError("example_rows_per_candidate must be positive")
        if type(self.production_pilot_rows) is not int or self.production_pilot_rows < 1:
            raise ValueError("production_pilot_rows must be positive")
        if (
            not self.lexical_ngram_orders
            or len(self.lexical_ngram_orders) != len(set(self.lexical_ngram_orders))
            or any(
                type(order) is not int or not 2 <= order <= 5 for order in self.lexical_ngram_orders
            )
        ):
            raise ValueError("lexical_ngram_orders must be unique integers in [2, 5]")
        for label, value in (
            ("lexical_quality_margin", self.lexical_quality_margin),
            ("lexical_stability_penalty", self.lexical_stability_penalty),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ValidityReceipt:
    raw_proposals: int
    valid_proposals: int
    output_rows: int
    violation_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_proposals": self.raw_proposals,
            "valid_proposals": self.valid_proposals,
            "invalid_proposals": self.raw_proposals - self.valid_proposals,
            "raw_valid_fraction": self.valid_proposals / self.raw_proposals,
            "output_rows": self.output_rows,
            "output_valid_fraction": 1.0,
            "violation_counts": dict(sorted(self.violation_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class NoveltyReceipt:
    rows: int
    novel_rows: int
    unique_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "novel_rows": self.novel_rows,
            "exact_training_matches": self.rows - self.novel_rows,
            "novel_fraction": self.novel_rows / self.rows,
            "unique_rows": self.unique_rows,
            "unique_fraction": self.unique_rows / self.rows,
        }


@dataclass(frozen=True, slots=True)
class TransportCandidateRun:
    candidate: CandidateName
    parameters: Mapping[str, Any]
    fold: GroupedFold
    seed: int
    train_data_sha256: str
    validation_data_sha256: str
    synthetic_data_sha256: str
    fit: PhaseResources
    sample: PhaseResources
    evaluation_elapsed_seconds: float
    proposal: ProposalReceipt
    validity: ValidityReceipt
    novelty: NoveltyReceipt
    evaluation: EvaluationBundle
    realization: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "parameters": dict(self.parameters),
            "fold": self.fold.to_dict(),
            "seed": self.seed,
            "train_data_sha256": self.train_data_sha256,
            "validation_data_sha256": self.validation_data_sha256,
            "synthetic_data_sha256": self.synthetic_data_sha256,
            "fit": self.fit.to_dict(),
            "sample": self.sample.to_dict(),
            "evaluation_elapsed_seconds": self.evaluation_elapsed_seconds,
            "total_elapsed_seconds": (
                self.fit.elapsed_seconds
                + self.sample.elapsed_seconds
                + self.evaluation_elapsed_seconds
            ),
            "proposal": self.proposal.to_dict(),
            "validity": self.validity.to_dict(),
            "novelty": self.novelty.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "realization": dict(self.realization),
        }


@dataclass(slots=True)
class _ValidityAccumulator:
    raw: int = 0
    valid: int = 0
    violations: Counter[str] | None = None

    def __post_init__(self) -> None:
        self.violations = Counter()

    def accept(self, data: Any) -> Sequence[bool]:
        mask: list[bool] = []
        for values in data.itertuples(index=False, name=None):
            compact = dict(zip(TRANSPORT_IDENTITY_COLUMNS, values, strict=True))
            self.raw += 1
            try:
                expanded = expand_transport_identity_structure(compact)
                violations = transport_identity_driver_violations(expanded)
            except ValueError:
                violations = ("invalid_compact_structure",)
            if violations:
                assert self.violations is not None
                self.violations.update(violations)
                mask.append(False)
            else:
                self.valid += 1
                mask.append(True)
        return mask

    def receipt(self, proposal: ProposalReceipt) -> ValidityReceipt:
        if (
            self.raw != proposal.raw_proposals
            or self.valid != proposal.accepted_proposals_before_truncation
        ):
            raise TransportIdentityBenchmarkError("proposal validity accounting differs")
        return ValidityReceipt(
            raw_proposals=self.raw,
            valid_proposals=self.valid,
            output_rows=proposal.output_rows,
            violation_counts=MappingProxyType(dict(self.violations or {})),
        )


def _strict_scalar(value: Any) -> str | int | bool:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, (str, int, bool)):
        return value
    raise TransportIdentityBenchmarkError(
        f"transport structure contains unsupported scalar {type(value).__name__}"
    )


def _row_key(row: Sequence[Any]) -> bytes:
    return canonical_json_bytes([_strict_scalar(value) for value in row])


def _expanded_int(row: Mapping[str, object], column: str) -> int:
    value = row[column]
    if type(value) is not int:
        raise TransportIdentityBenchmarkError(f"expanded {column} is not an integer")
    return value


def _novelty(synthetic: Any, train: Any) -> NoveltyReceipt:
    training = {_row_key(row) for row in train.itertuples(index=False, name=None)}
    synthetic_rows = [_row_key(row) for row in synthetic.itertuples(index=False, name=None)]
    source_matches = sum(row in training for row in synthetic_rows)
    return NoveltyReceipt(
        rows=len(synthetic_rows),
        novel_rows=len(synthetic_rows) - source_matches,
        unique_rows=len(set(synthetic_rows)),
    )


def _realization_audit(
    *,
    synthetic: Any,
    candidate: CandidateName,
    fold: GroupedFold,
    seed: int,
    guard: SourceTransportIdentityGuard,
    policy: TransportPrivacyPolicy,
    example_rows: int,
    vessel_renderer: VesselNameRenderer,
) -> dict[str, Any]:
    return audit_and_realize_transport_identity_frame(
        synthetic=synthetic,
        row_id_prefix=f"{candidate}-fold{fold.index:02d}-seed{seed}",
        seed=seed,
        guard=guard,
        policy=policy,
        example_rows=example_rows,
        vessel_renderer=vessel_renderer,
    )


def audit_and_realize_transport_identity_frame(
    *,
    synthetic: Any,
    row_id_prefix: str,
    seed: int,
    guard: SourceTransportIdentityGuard,
    policy: TransportPrivacyPolicy,
    example_rows: int,
    vessel_renderer: VesselNameRenderer,
) -> dict[str, Any]:
    """Realize a sampled structure frame and prove its identity contracts."""

    if not row_id_prefix or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("row_id_prefix must be non-empty and seed must be uint32")
    if type(example_rows) is not int or example_rows < 1:
        raise ValueError("example_rows must be positive")
    compact_rows = synthetic.to_dict(orient="records")
    rows = [expand_transport_identity_structure(row) for row in compact_rows]
    row_ids = tuple(f"{row_id_prefix}-row{index:04d}" for index in range(len(rows)))
    realized = realize_transport_identities(
        rows=rows,
        row_ids=row_ids,
        stream=DeterministicStream(
            seed=seed,
            namespace="transport-identity-benchmark-realization-v1",
            identity=row_id_prefix,
        ),
        guard=guard,
        policy=policy,
        vessel_renderer=vessel_renderer,
    )
    ordered_rows = {row_id: row for row_id, row in zip(row_ids, rows, strict=True)}
    vessel_attempts: list[int] = []
    voyage_attempts: list[int] = []
    strategies: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    nearest_normalized_distances: list[float] = []
    nearest_absolute_distances: list[int] = []
    maximum_source_substring_fractions: list[float] = []
    maximum_source_substring_characters: list[int] = []
    for item in realized:
        row = ordered_rows[item.row_id]
        if item.vessel_name is not None:
            vessel_attempts.append(item.vessel_attempts)
            strategies[item.vessel_render_strategy] += 1
            distance = guard.vessel_distance_diagnostics(item.vessel_name)
            nearest_normalized_distances.append(float(distance["normalized"]))
            nearest_absolute_distances.append(int(distance["absolute"]))
            maximum_source_substring_fractions.append(float(distance["substringFraction"]))
            maximum_source_substring_characters.append(int(distance["substringCharacters"]))
            if (
                len(item.vessel_name) != _expanded_int(row, "vessel_character_count")
                or len(item.vessel_name.split()) != _expanded_int(row, "vessel_word_count")
                or sum(character.isdigit() for character in item.vessel_name)
                != _expanded_int(row, "vessel_digit_count")
                or guard.classify_vessel_name(item.vessel_name, policy=policy) != "none"
            ):
                raise TransportIdentityBenchmarkError(
                    "realized vessel differs from its structure or source-safety contract"
                )
        if item.voyage_number is not None:
            voyage_attempts.append(item.voyage_attempts)
            if guard.classify_voyage_number(item.voyage_number, policy=policy) != "none":
                raise TransportIdentityBenchmarkError("realized voyage collides with source")
        if len(examples) < example_rows:
            examples.append(
                {
                    "syntheticRowId": item.row_id,
                    "vesselName": item.vessel_name,
                    "voyageNumber": item.voyage_number,
                    "vesselRendererStrategy": item.vessel_render_strategy,
                    "vesselWordCount": _expanded_int(row, "vessel_word_count"),
                    "vesselCharacterCount": _expanded_int(row, "vessel_character_count"),
                    "vesselDigitCount": _expanded_int(row, "vessel_digit_count"),
                    "voyageShape": str(row["voyage_shape"]),
                }
            )
    vessel_values = [item.vessel_name for item in realized if item.vessel_name is not None]
    voyage_values = [item.voyage_number for item in realized if item.voyage_number is not None]
    if len(vessel_values) != len(set(vessel_values)) or len(voyage_values) != len(
        set(voyage_values)
    ):
        raise TransportIdentityBenchmarkError("realized transport identities are not batch-unique")
    return {
        "rows": len(realized),
        "exactStructureFraction": 1.0,
        "sourceSafeFraction": 1.0,
        "batchUniqueFraction": 1.0,
        "vesselRendererId": vessel_renderer.renderer_id,
        "vesselStrategyCounts": dict(sorted(strategies.items())),
        "meanVesselAttempts": statistics.fmean(vessel_attempts) if vessel_attempts else 0.0,
        "maximumVesselAttempts": max(vessel_attempts, default=0),
        "meanVoyageAttempts": statistics.fmean(voyage_attempts) if voyage_attempts else 0.0,
        "maximumVoyageAttempts": max(voyage_attempts, default=0),
        "nearestSourceNormalizedEditDistance": {
            "minimum": min(nearest_normalized_distances, default=None),
            "p05": _nearest_rank(nearest_normalized_distances, 0.05),
            "median": _nearest_rank(nearest_normalized_distances, 0.50),
        },
        "nearestSourceAbsoluteEditDistance": {
            "minimum": min(nearest_absolute_distances, default=None),
            "p05": _nearest_rank(nearest_absolute_distances, 0.05),
            "median": _nearest_rank(nearest_absolute_distances, 0.50),
        },
        "maximumSourceSubstringFraction": {
            "maximum": max(maximum_source_substring_fractions, default=None),
            "p95": _nearest_rank(maximum_source_substring_fractions, 0.95),
            "median": _nearest_rank(maximum_source_substring_fractions, 0.50),
        },
        "maximumSourceSubstringCharacters": {
            "maximum": max(maximum_source_substring_characters, default=None),
            "p95": _nearest_rank(maximum_source_substring_characters, 0.95),
            "median": _nearest_rank(maximum_source_substring_characters, 0.50),
        },
        "examples": examples,
    }


def _nearest_rank(values: Sequence[float | int], fraction: float) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def generate_transport_identity_pilot(
    *,
    bundle: TransportIdentityBundle,
    candidate: CandidateSpec,
    requested_rows: int,
    seed: int,
    policy: TransportPrivacyPolicy,
    proposal_multiplier: int,
    proposal_batch_rows: int | None,
    request_id: str,
    vessel_renderer: VesselNameRenderer,
) -> dict[str, Any]:
    """Fit one declared candidate and generate an audited identity pilot.

    This is the supported controlled-generation API after model selection.  It
    fits only the train-isolated structural view, never exposes source identity
    values to SDV, and returns actual fictional vessel/voyage values plus fit,
    sample, validity, collision, and uniqueness receipts.
    """

    if not request_id:
        raise ValueError("request_id must be non-empty")
    if type(requested_rows) is not int or requested_rows < 1:
        raise ValueError("requested_rows must be positive")
    fitted = fit_candidate_model(
        view=bundle.view,
        spec=candidate,
        train_data=bundle.view.data.copy(),
        seed=seed,
    )
    validity = _ValidityAccumulator()
    synthetic, proposal, sample_resources = fitted.sample(
        requested_rows=requested_rows,
        seed=seed,
        columns=TRANSPORT_IDENTITY_COLUMNS,
        acceptance=validity.accept,
        proposal_multiplier=proposal_multiplier,
        proposal_batch_rows=proposal_batch_rows,
    )
    validity_receipt = validity.receipt(proposal)
    realization = audit_and_realize_transport_identity_frame(
        synthetic=synthetic,
        row_id_prefix=request_id,
        seed=seed,
        guard=bundle.guard,
        policy=policy,
        example_rows=requested_rows,
        vessel_renderer=vessel_renderer,
    )
    if len(realization["examples"]) != requested_rows:
        raise TransportIdentityBenchmarkError(
            "production pilot did not publish every requested generated identity"
        )
    return {
        "requestId": request_id,
        "candidate": candidate.name,
        "vesselRendererId": vessel_renderer.renderer_id,
        "parameters": dict(candidate.parameters),
        "fitRows": len(bundle.view.data),
        "fitDataSha256": dataframe_sha256(bundle.view.data),
        "syntheticDataSha256": dataframe_sha256(synthetic),
        "fit": fitted.fit.to_dict(),
        "sample": sample_resources.to_dict(),
        "proposal": proposal.to_dict(),
        "validity": validity_receipt.to_dict(),
        "realization": realization,
    }


def _unique_source_names(
    *, bundle: TransportIdentityBundle, indices: Sequence[int]
) -> tuple[str, ...]:
    names_by_key: dict[str, str] = {}
    for index in indices:
        name = bundle.fit_vessel_names[index]
        if name is not None:
            names_by_key.setdefault(transport_identity_key(name), name)
    return tuple(names_by_key[key] for key in sorted(names_by_key))


def _faker_company_names(*, count: int, seed: int, identity: str) -> tuple[str, ...]:
    if count < 20 or not identity:
        raise ValueError("Faker candidate requires at least 20 names and an identity")
    faker_module = __import__("faker", fromlist=["Faker"])
    generator = faker_module.Faker("en_US")
    identity_seed = int.from_bytes(
        hashlib.sha256(canonical_json_bytes([seed, identity])).digest()[:4], "big"
    )
    generator.seed_instance(identity_seed)
    output: list[str] = []
    seen: set[str] = set()
    for _ in range(count * 20):
        value = str(generator.company()).strip()
        key = transport_identity_key(value)
        if key not in seen:
            seen.add(key)
            output.append(value)
        if len(output) == count:
            return tuple(output)
    raise TransportIdentityBenchmarkError("Faker company candidate did not produce enough names")


def _fit_lexical_renderer(
    *,
    bundle: TransportIdentityBundle,
    indices: Sequence[int],
    order: int,
    source: Literal["source", "faker"],
    seed: int,
    identity: str,
) -> CharacterNGramVesselRenderer:
    source_names = _unique_source_names(bundle=bundle, indices=indices)
    names = (
        source_names
        if source == "source"
        else _faker_company_names(
            count=max(40, len(source_names) * 2), seed=seed, identity=identity
        )
    )
    return fit_character_ngram_vessel_renderer(names=names, order=order)


def fit_selected_vessel_renderer(
    *,
    bundle: TransportIdentityBundle,
    selection: Mapping[str, object],
    seed: int,
    identity: str = "production-lexical-renderer",
) -> CharacterNGramVesselRenderer:
    """Refit an audited lexical selection on the bundle's complete fit scope.

    The benchmark result is deliberately a compact public contract.  This
    helper validates every coupled selection field before fitting so a caller
    cannot accidentally combine a source-trained candidate, a Faker corpus,
    or a renderer identifier from another n-gram order.
    """

    candidate = selection.get("candidate")
    order = selection.get("order")
    source_value = selection.get("source")
    renderer_id = selection.get("rendererId")
    if not isinstance(candidate, str) or not candidate:
        raise TransportIdentityBenchmarkError("lexical selection candidate is required")
    if type(order) is not int or not 2 <= order <= 5:
        raise TransportIdentityBenchmarkError("lexical selection order must be in [2, 5]")
    if source_value == "source":
        source: Literal["source", "faker"] = "source"
    elif source_value == "faker":
        source = "faker"
    else:
        raise TransportIdentityBenchmarkError("lexical selection source must be source or faker")
    if not isinstance(renderer_id, str) or not renderer_id:
        raise TransportIdentityBenchmarkError("lexical selection rendererId is required")
    expected_candidate = (
        f"faker_company_{order}gram" if source == "faker" else f"source_character_{order}gram"
    )
    if candidate != expected_candidate:
        raise TransportIdentityBenchmarkError(
            "lexical selection candidate/order/source are internally inconsistent"
        )

    renderer = _fit_lexical_renderer(
        bundle=bundle,
        indices=tuple(range(len(bundle.view.data))),
        order=order,
        source=source,
        seed=seed,
        identity=identity,
    )
    if renderer.renderer_id != renderer_id:
        raise TransportIdentityBenchmarkError(
            "lexical selection rendererId does not match the refitted renderer"
        )
    return renderer


def _identity_connected_group_ids(bundle: TransportIdentityBundle) -> tuple[str, ...]:
    """Connect template and repeated-identity components before grouped CV."""

    row_count = len(bundle.view.data)
    parents = list(range(row_count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    first_by_template: dict[str, int] = {}
    first_by_identity: dict[str, int] = {}
    for index, (template, name) in enumerate(
        zip(bundle.view.group_ids, bundle.fit_vessel_names, strict=True)
    ):
        previous_template = first_by_template.setdefault(template, index)
        union(index, previous_template)
        if name is not None:
            key = transport_identity_key(name)
            previous_identity = first_by_identity.setdefault(key, index)
            union(index, previous_identity)
    components: dict[int, list[str]] = defaultdict(list)
    for index, row_id in enumerate(bundle.view.row_ids):
        components[find(index)].append(row_id)
    component_ids = {
        root: hashlib.sha256(canonical_json_bytes(sorted(members))).hexdigest()
        for root, members in components.items()
    }
    output = tuple(component_ids[find(index)] for index in range(row_count))
    if len(output) != row_count:
        raise TransportIdentityBenchmarkError("identity-connected grouping lost rows")
    return output


def _categorical_sdv_replay_diagnostic(
    *,
    bundle: TransportIdentityBundle,
    folds: Sequence[GroupedFold],
    settings: TransportBenchmarkSettings,
) -> dict[str, Any]:
    """Measure raw high-cardinality categorical replay without publishing values."""

    pandas = __import__("pandas")
    specs = tuple(
        candidate
        for candidate in settings.candidates
        if candidate.name in {"empirical", "gaussian_copula"}
    )
    receipts: list[dict[str, Any]] = []
    for fold in folds:
        train_names = _unique_source_names(bundle=bundle, indices=fold.train_indices)
        heldout_names = _unique_source_names(bundle=bundle, indices=fold.validation_indices)
        train = pandas.DataFrame({"vessel_name": train_names})
        view = BenchmarkView(
            name="vessel_name_categorical_replay_diagnostic",
            data=train,
            metadata=MappingProxyType(
                {
                    "tables": {
                        "vessel_name_categorical_replay_diagnostic": {
                            "columns": {"vessel_name": {"sdtype": "categorical"}}
                        }
                    },
                    "relationships": [],
                }
            ),
            row_ids=tuple(
                hashlib.sha256(canonical_json_bytes(["row", index])).hexdigest()
                for index in range(len(train))
            ),
            group_ids=tuple(
                hashlib.sha256(canonical_json_bytes(["group", index])).hexdigest()
                for index in range(len(train))
            ),
            partition_labels=("train",) * len(train),
            allowed_partition="train",
        )
        train_keys = {transport_identity_key(value) for value in train_names}
        for spec in specs:
            fitted = fit_candidate_model(
                view=view,
                spec=spec,
                train_data=train,
                seed=settings.seeds[0],
            )
            synthetic, proposal, sample = fitted.sample(
                requested_rows=max(50, len(heldout_names)),
                seed=settings.seeds[0],
                columns=("vessel_name",),
                acceptance=None,
                proposal_multiplier=1,
                proposal_batch_rows=settings.proposal_batch_rows,
            )
            values = tuple(str(value) for value in synthetic["vessel_name"].tolist())
            exact_replays = sum(transport_identity_key(value) in train_keys for value in values)
            source_rejections = sum(
                bundle.guard.classify_vessel_name(value, policy=settings.privacy_policy) != "none"
                for value in values
            )
            receipts.append(
                {
                    "candidate": spec.name,
                    "foldIndex": fold.index,
                    "fitRows": len(train),
                    "sampleRows": len(values),
                    "exactFitReplayFraction": exact_replays / len(values),
                    "fullSourcePrivacyRejectionFraction": source_rejections / len(values),
                    "fit": fitted.fit.to_dict(),
                    "sample": sample.to_dict(),
                    "proposal": proposal.to_dict(),
                }
            )
    summaries = []
    for spec in specs:
        selected = [row for row in receipts if row["candidate"] == spec.name]
        summaries.append(
            {
                "candidate": spec.name,
                "runCount": len(selected),
                "meanExactFitReplayFraction": statistics.fmean(
                    float(row["exactFitReplayFraction"]) for row in selected
                ),
                "meanFullSourcePrivacyRejectionFraction": statistics.fmean(
                    float(row["fullSourcePrivacyRejectionFraction"]) for row in selected
                ),
                "productionEligible": False,
            }
        )
    return {
        "purpose": "measure_high_cardinality_categorical_replay_not_production",
        "rawValuesPublished": False,
        "summaries": summaries,
        "runReceipts": receipts,
    }


def _lexical_renderer_comparison(
    *,
    bundle: TransportIdentityBundle,
    folds: Sequence[GroupedFold],
    settings: TransportBenchmarkSettings,
) -> tuple[dict[str, Any], dict[int, CharacterNGramVesselRenderer]]:
    receipts: list[dict[str, Any]] = []
    fitted_by_fold_candidate: dict[tuple[int, str], CharacterNGramVesselRenderer] = {}
    candidates = (
        "generic_baseline",
        *(f"source_character_{order}gram" for order in settings.lexical_ngram_orders),
        *(f"faker_company_{order}gram" for order in settings.lexical_ngram_orders),
    )
    for fold in folds:
        seen_reference_keys: set[str] = set()
        validation_positions: list[int] = []
        reference_names_list: list[str] = []
        heldout_exclusions: Counter[str] = Counter()
        heldout_input_name_count = 0
        for local_position, index in enumerate(fold.validation_indices):
            name = bundle.fit_vessel_names[index]
            if name is None:
                continue
            heldout_input_name_count += 1
            exclusion_reason = vessel_name_fit_exclusion_reason(name)
            if exclusion_reason is not None:
                heldout_exclusions[exclusion_reason] += 1
                continue
            key = transport_identity_key(name)
            if key in seen_reference_keys:
                continue
            seen_reference_keys.add(key)
            validation_positions.append(local_position)
            reference_names_list.append(name)
        compact_validation = (
            bundle.view.data.iloc[list(fold.validation_indices)]
            .iloc[validation_positions]
            .reset_index(drop=True)
        )
        reference_names = tuple(reference_names_list)
        evaluator_order = min(settings.lexical_ngram_orders, key=lambda value: abs(value - 3))
        evaluator = _fit_lexical_renderer(
            bundle=bundle,
            indices=fold.train_indices,
            order=evaluator_order,
            source="source",
            seed=settings.seeds[0],
            identity=f"evaluator-fold-{fold.index}",
        )
        real_first, real_second = deterministic_real_split(
            reference_names, seed=settings.seeds[0] + fold.index
        )
        real_baseline: dict[str, Any] = {
            "metrics": lexical_realism_metrics(
                reference_names=real_first, generated_names=real_second
            ),
            "discriminator": lexical_discriminator_auc(
                reference_names=real_first,
                generated_names=real_second,
                seed=settings.seeds[0] + fold.index,
            ),
            "bitsPerCharacterGap": abs(
                evaluator.bits_per_character(real_first) - evaluator.bits_per_character(real_second)
            ),
        }
        for candidate in candidates:
            fit_started = time.perf_counter()
            if candidate == "generic_baseline":
                renderer: VesselNameRenderer = GENERIC_VESSEL_RENDERER
                fit_name_count = 0
                fit_token_count = 0
                fit_input_count = 0
                fit_exclusions: Mapping[str, int] = {}
            else:
                source: Literal["source", "faker"] = (
                    "faker" if candidate.startswith("faker_company_") else "source"
                )
                order = int(candidate.rsplit("_", 1)[-1].removesuffix("gram"))
                fitted = _fit_lexical_renderer(
                    bundle=bundle,
                    indices=fold.train_indices,
                    order=order,
                    source=source,
                    seed=settings.seeds[0],
                    identity=f"{candidate}-fold-{fold.index}",
                )
                fitted_by_fold_candidate[(fold.index, candidate)] = fitted
                renderer = fitted
                fit_input_count = fitted.input_name_count
                fit_name_count = fitted.fit_name_count
                fit_token_count = fitted.fit_token_count
                fit_exclusions = fitted.fit_exclusion_counts
            fit_seconds = time.perf_counter() - fit_started
            realization_started = time.perf_counter()
            realization = audit_and_realize_transport_identity_frame(
                synthetic=compact_validation,
                row_id_prefix=f"lexical-{candidate}-fold{fold.index:02d}",
                seed=settings.seeds[0],
                guard=bundle.guard,
                policy=settings.privacy_policy,
                example_rows=len(compact_validation),
                vessel_renderer=renderer,
            )
            realization_seconds = time.perf_counter() - realization_started
            generated_names = tuple(
                row["vesselName"]
                for row in realization["examples"]
                if row["vesselName"] is not None
            )
            metrics = lexical_realism_metrics(
                reference_names=reference_names,
                generated_names=generated_names,
            )
            discriminator = lexical_discriminator_auc(
                reference_names=reference_names,
                generated_names=generated_names,
                seed=settings.seeds[0] + fold.index,
            )
            reference_bits = evaluator.bits_per_character(reference_names)
            generated_bits = evaluator.bits_per_character(generated_names)
            discriminator_excess = max(
                0.0,
                float(discriminator["separability"])
                - float(real_baseline["discriminator"]["separability"]),
            )
            receipts.append(
                {
                    "candidate": candidate,
                    "rendererId": renderer.renderer_id,
                    "foldIndex": fold.index,
                    "fitInputNameCount": fit_input_count,
                    "fitNameCount": fit_name_count,
                    "fitTokenCount": fit_token_count,
                    "fitExclusionCounts": dict(fit_exclusions),
                    "heldOutInputNameCount": heldout_input_name_count,
                    "heldOutExclusionCounts": dict(sorted(heldout_exclusions.items())),
                    "heldOutNameCount": len(reference_names),
                    "fitSeconds": fit_seconds,
                    "realizationSeconds": realization_seconds,
                    "metrics": metrics,
                    "referenceBitsPerCharacter": reference_bits,
                    "generatedBitsPerCharacter": generated_bits,
                    "bitsPerCharacterGap": abs(reference_bits - generated_bits),
                    "discriminator": discriminator,
                    "realVersusRealBaseline": real_baseline,
                    "discriminatorExcessOverRealBaseline": discriminator_excess,
                    "exactStructureFraction": realization["exactStructureFraction"],
                    "sourceSafeFraction": realization["sourceSafeFraction"],
                    "batchUniqueFraction": realization["batchUniqueFraction"],
                }
            )

    summaries: list[dict[str, Any]] = []
    for complexity, candidate in enumerate(candidates):
        selected = [row for row in receipts if row["candidate"] == candidate]
        scores = [float(row["metrics"]["lexicalRealismScore"]) for row in selected]
        trigram = [float(row["metrics"]["characterNgramSimilarity"]["3"]) for row in selected]
        discriminator_excesses = [
            float(row["discriminatorExcessOverRealBaseline"]) for row in selected
        ]
        bits_gaps = [float(row["bitsPerCharacterGap"]) for row in selected]
        summaries.append(
            {
                "candidate": candidate,
                "rendererId": selected[0]["rendererId"],
                "runCount": len(selected),
                "meanLexicalRealism": statistics.fmean(scores),
                "lexicalRealismStddev": statistics.pstdev(scores),
                "stabilityAdjustedLexicalRealism": (
                    statistics.fmean(scores)
                    - settings.lexical_stability_penalty * statistics.pstdev(scores)
                ),
                "meanTrigramSimilarity": statistics.fmean(trigram),
                "meanDiscriminatorExcessOverRealBaseline": statistics.fmean(discriminator_excesses),
                "meanBitsPerCharacterGap": statistics.fmean(bits_gaps),
                "totalFitSeconds": sum(float(row["fitSeconds"]) for row in selected),
                "totalRealizationSeconds": sum(
                    float(row["realizationSeconds"]) for row in selected
                ),
                "complexityRank": complexity,
            }
        )
    selectable = [row for row in summaries if row["candidate"] != "generic_baseline"]
    best_discriminator_excess = min(
        float(row["meanDiscriminatorExcessOverRealBaseline"]) for row in selectable
    )
    eligible = [
        row
        for row in selectable
        if float(row["meanDiscriminatorExcessOverRealBaseline"])
        <= best_discriminator_excess + settings.lexical_quality_margin
    ]
    selected_summary = min(
        eligible,
        key=lambda row: (
            -float(row["meanTrigramSimilarity"]),
            float(row["meanBitsPerCharacterGap"]),
            int(row["complexityRank"]),
            str(row["candidate"]),
        ),
    )
    selected_candidate = str(selected_summary["candidate"])
    selected_order = int(selected_candidate.rsplit("_", 1)[-1].removesuffix("gram"))
    renderer_by_fold = {
        fold.index: fitted_by_fold_candidate[(fold.index, selected_candidate)] for fold in folds
    }
    return (
        {
            "selectionRule": (
                "hard_gates_then_min_discriminator_excess_within_margin_then_max_trigram_"
                "similarity_then_min_bits_gap_then_complexity"
            ),
            "qualityMargin": settings.lexical_quality_margin,
            "stabilityPenalty": settings.lexical_stability_penalty,
            "selectedCandidate": selected_candidate,
            "selectedOrder": selected_order,
            "selectedRendererId": selected_summary["rendererId"],
            "summaries": summaries,
            "runReceipts": receipts,
        },
        renderer_by_fold,
    )


def _membership_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _candidate_summary(
    candidate: CandidateName, runs: Sequence[TransportCandidateRun]
) -> dict[str, Any]:
    selected = [run for run in runs if run.candidate == candidate]
    qualities = [run.evaluation.quality.score for run in selected]
    raw = sum(run.validity.raw_proposals for run in selected)
    valid = sum(run.validity.valid_proposals for run in selected)
    output = sum(run.novelty.rows for run in selected)
    novel = sum(run.novelty.novel_rows for run in selected)
    unique = sum(run.novelty.unique_rows for run in selected)
    cuda_allocated = [
        value
        for run in selected
        for value in (run.fit.cuda_peak_allocated_bytes, run.sample.cuda_peak_allocated_bytes)
        if value is not None
    ]
    cuda_reserved = [
        value
        for run in selected
        for value in (run.fit.cuda_peak_reserved_bytes, run.sample.cuda_peak_reserved_bytes)
        if value is not None
    ]
    return {
        "candidate": candidate,
        "runCount": len(selected),
        "meanQuality": statistics.fmean(qualities),
        "qualityStddev": statistics.pstdev(qualities),
        "minimumDiagnostic": min(run.evaluation.diagnostic.score for run in selected),
        "rawValidFraction": valid / raw,
        "novelFraction": novel / output,
        "uniqueFraction": unique / output,
        "totalFitSeconds": sum(run.fit.elapsed_seconds for run in selected),
        "totalSampleSeconds": sum(run.sample.elapsed_seconds for run in selected),
        "totalEvaluationSeconds": sum(run.evaluation_elapsed_seconds for run in selected),
        "devices": sorted({run.fit.accelerator.device for run in selected}),
        "maximumCudaPeakAllocatedBytes": max(cuda_allocated) if cuda_allocated else None,
        "maximumCudaPeakReservedBytes": max(cuda_reserved) if cuda_reserved else None,
    }


def _aggregate_realization_audit(runs: Sequence[TransportCandidateRun]) -> dict[str, Any]:
    rows = sum(int(run.realization["rows"]) for run in runs)
    if not rows or any(
        float(run.realization[field]) != 1.0
        for run in runs
        for field in (
            "exactStructureFraction",
            "sourceSafeFraction",
            "batchUniqueFraction",
        )
    ):
        raise TransportIdentityBenchmarkError("realization audit is incomplete")
    strategies: Counter[str] = Counter()
    for run in runs:
        raw = run.realization["vesselStrategyCounts"]
        if not isinstance(raw, Mapping):
            raise TransportIdentityBenchmarkError("invalid realization strategy receipt")
        strategies.update({str(key): int(value) for key, value in raw.items()})
    return {
        "rows": rows,
        "runCount": len(runs),
        "exactStructureFraction": 1.0,
        "sourceSafeFraction": 1.0,
        "batchUniqueFraction": 1.0,
        "vesselStrategyCounts": dict(sorted(strategies.items())),
    }


def _report_markdown(
    *,
    summaries: Sequence[Mapping[str, Any]],
    selection: ModelSelection,
    rows: int,
    groups: int,
    realization_rows: int,
    production_pilot_rows: int,
    lexical_comparison: Mapping[str, Any],
    categorical_replay: Mapping[str, Any],
    production_pilot: Mapping[str, Any],
) -> bytes:
    lines = [
        "# Vessel/voyage structural SDV benchmark",
        "",
        (
            f"Paired comparison on {rows:,} train-isolated documents across "
            f"{groups:,} template groups; folds additionally connect repeated vessel identities."
        ),
        (
            "SDV saw structural atoms only. Fit-side vessel names were used only by the "
            "separately grouped lexical renderer; no source identity or lexical model was "
            "published."
        ),
        "",
        (
            "| Candidate | Quality | Stddev | Raw valid | Novel | Unique | Fit s | "
            "Device | Peak CUDA MiB |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summaries:
        peak = row["maximumCudaPeakAllocatedBytes"]
        lines.append(
            "| {candidate} | {quality:.4f} | {std:.4f} | {valid:.4f} | "
            "{novel:.4f} | {unique:.4f} | {fit:.2f} | {device} | {peak} |".format(
                candidate=row["candidate"],
                quality=row["meanQuality"],
                std=row["qualityStddev"],
                valid=row["rawValidFraction"],
                novel=row["novelFraction"],
                unique=row["uniqueFraction"],
                fit=row["totalFitSeconds"],
                device=", ".join(row["devices"]),
                peak=(f"{peak / 2**20:.2f}" if peak is not None else "-"),
            )
        )
    lines.extend(
        [
            "",
            (
                "| Lexical renderer | Realism | Stddev | Adjusted | Trigram | "
                "Discriminator excess | BPC gap | Fit s | Realize s |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in lexical_comparison["summaries"]:
        lines.append(
            "| {candidate} | {mean:.4f} | {std:.4f} | {adjusted:.4f} | "
            "{trigram:.4f} | {discriminator:.4f} | {bpc:.4f} | {fit:.3f} | "
            "{realize:.2f} |".format(
                candidate=row["candidate"],
                mean=row["meanLexicalRealism"],
                std=row["lexicalRealismStddev"],
                adjusted=row["stabilityAdjustedLexicalRealism"],
                trigram=row["meanTrigramSimilarity"],
                discriminator=row["meanDiscriminatorExcessOverRealBaseline"],
                bpc=row["meanBitsPerCharacterGap"],
                fit=row["totalFitSeconds"],
                realize=row["totalRealizationSeconds"],
            )
        )
    lines.extend(
        [
            "",
            "## Raw categorical SDV replay diagnostic",
            "",
            "| Candidate | Exact fit replay | Full-source privacy rejection | Eligible |",
            "|---|---:|---:|---|",
        ]
    )
    for row in categorical_replay["summaries"]:
        lines.append(
            "| {candidate} | {replay:.4f} | {privacy:.4f} | {eligible} |".format(
                candidate=row["candidate"],
                replay=row["meanExactFitReplayFraction"],
                privacy=row["meanFullSourcePrivacyRejectionFraction"],
                eligible="yes" if row["productionEligible"] else "no",
            )
        )
    pilot_realization = production_pilot.get("realization")
    if not isinstance(pilot_realization, Mapping):
        raise TransportIdentityBenchmarkError("production pilot realization is missing")
    pilot_examples = pilot_realization.get("examples")
    if not isinstance(pilot_examples, Sequence):
        raise TransportIdentityBenchmarkError("production pilot examples are missing")
    lines.extend(
        [
            "",
            "## Selected-default examples",
            "",
            "| Vessel name | Voyage number |",
            "|---|---|",
        ]
    )
    for example in pilot_examples[:10]:
        if not isinstance(example, Mapping):
            raise TransportIdentityBenchmarkError("production pilot example is invalid")
        vessel = str(example.get("vesselName") or "-").replace("|", "\\|")
        voyage = str(example.get("voyageNumber") or "-").replace("|", "\\|")
        lines.append(f"| {vessel} | {voyage} |")
    lines.extend(
        [
            "",
            f"Recommended structural candidate: **{selection.selected_candidate}**.",
            (
                "Selected lexical renderer: **{}** (order {}).".format(
                    lexical_comparison["selectedCandidate"],
                    lexical_comparison["selectedOrder"],
                )
            ),
            (
                "Empirical is retained as a fidelity baseline but is not selectable because "
                "it cannot create novel structures."
            ),
            (
                "All SDV candidates use the same grouped-selected character n-gram renderer "
                "after structural sampling."
            ),
            (
                f"Across {realization_rows:,} held-out realizations, exact structural "
                "fidelity, full-source safety, and within-batch uniqueness were each 100%."
            ),
            (
                f"The selected default was refit on all eligible fit rows and produced "
                f"{production_pilot_rows:,} fully materialized identities in "
                "`selected-default-pilot.json`."
            ),
            "IMO numbers remain null without an authoritative assigned-number exclusion registry.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def run_transport_identity_benchmark(
    *,
    bundle: TransportIdentityBundle,
    settings: TransportBenchmarkSettings,
    artifact_dir: Path,
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Run, select, realize, and immutably publish the paired comparison."""

    view: BenchmarkView = bundle.view
    try:
        canonical_json_bytes(dict(input_provenance))
    except (TypeError, ValueError) as error:
        raise ValueError("input_provenance must be finite JSON") from error
    if (
        view.name != TRANSPORT_IDENTITY_VIEW_NAME
        or tuple(view.data.columns) != TRANSPORT_IDENTITY_COLUMNS
    ):
        raise ValueError("benchmark accepts only the exact transport identity view")
    if settings.fold_count > len(set(view.group_ids)):
        raise ValueError("fold_count exceeds transport template groups")
    for values in view.data.itertuples(index=False, name=None):
        expanded = expand_transport_identity_structure(
            dict(zip(TRANSPORT_IDENTITY_COLUMNS, values, strict=True))
        )
        if tuple(expanded) != TRANSPORT_IDENTITY_DRIVER_COLUMNS:
            raise ValueError("expanded driver columns differ from their contract")
        violations = transport_identity_driver_violations(expanded)
        if violations:
            raise ValueError(f"source transport structure violates contract: {violations}")
    if artifact_dir.is_symlink():
        raise ValueError("artifact directory cannot be a symbolic link")
    if artifact_dir.exists():
        if not artifact_dir.is_dir() or any(artifact_dir.iterdir()):
            raise ValueError("artifact directory must be new or empty")
    else:
        artifact_dir.mkdir(parents=True)

    connected_group_ids = _identity_connected_group_ids(bundle)
    folds = grouped_folds(
        group_ids=connected_group_ids,
        fold_count=settings.fold_count,
        seed=settings.fold_seed,
    )
    accelerator = preflight_sdv_accelerator(enable_gpu=True)
    atomic_publish_json(artifact_dir / "gpu-preflight.json", accelerator.to_dict())
    lexical_comparison, renderer_by_fold = _lexical_renderer_comparison(
        bundle=bundle,
        folds=folds,
        settings=settings,
    )
    atomic_publish_json(artifact_dir / "lexical-renderer-comparison.json", lexical_comparison)
    categorical_replay = _categorical_sdv_replay_diagnostic(
        bundle=bundle,
        folds=folds,
        settings=settings,
    )
    atomic_publish_json(artifact_dir / "categorical-sdv-replay-diagnostic.json", categorical_replay)
    atomic_publish_json(
        artifact_dir / "benchmark-contract.json",
        {
            "experiment": "transport_identity_structural_sdv_comparison_v1",
            "view": view.name,
            "rows": len(view.data),
            "templateGroups": len(set(view.group_ids)),
            "identityConnectedGroups": len(set(connected_group_ids)),
            "allowedPartition": view.allowed_partition,
            "columns": list(TRANSPORT_IDENTITY_COLUMNS),
            "viewDataSha256": dataframe_sha256(view.data),
            "rowMembershipSha256": _membership_sha256(view.row_ids),
            "templateGroupMembershipSha256": _membership_sha256(view.group_ids),
            "identityConnectedGroupMembershipSha256": _membership_sha256(connected_group_ids),
            "candidates": [
                {"name": candidate.name, "parameters": dict(candidate.parameters)}
                for candidate in settings.candidates
            ],
            "folds": [fold.to_dict() for fold in folds],
            "settings": {
                "foldCount": settings.fold_count,
                "foldSeed": settings.fold_seed,
                "seeds": list(settings.seeds),
                "proposalMultiplier": settings.proposal_multiplier,
                "proposalBatchRows": settings.proposal_batch_rows,
                "qualityMargin": settings.quality_margin,
                "stabilityPenalty": settings.stability_penalty,
                "minimumNormalizedEditDistance": (
                    settings.privacy_policy.minimum_normalized_edit_distance
                ),
                "minimumAbsoluteEditDistance": (
                    settings.privacy_policy.minimum_absolute_edit_distance
                ),
                "maximumSourceSubstringFraction": (
                    settings.privacy_policy.maximum_source_substring_fraction
                ),
                "minimumSourceSubstringCharacters": (
                    settings.privacy_policy.minimum_source_substring_characters
                ),
                "maximumRealizationAttempts": settings.privacy_policy.maximum_attempts,
                "exampleRowsPerCandidate": settings.example_rows_per_candidate,
                "productionPilotRows": settings.production_pilot_rows,
                "lexicalNgramOrders": list(settings.lexical_ngram_orders),
                "lexicalQualityMargin": settings.lexical_quality_margin,
                "lexicalStabilityPenalty": settings.lexical_stability_penalty,
            },
            "implementation": {
                "transportIdentity": sha256_file(Path(__file__).with_name("transport_identity.py")),
                "transportBenchmark": sha256_file(Path(__file__)),
                "sdvHarness": sha256_file(Path(__file__).with_name("sdv_harness.py")),
                "sdvEvaluation": sha256_file(Path(__file__).with_name("sdv_evaluation.py")),
                "vesselLexical": sha256_file(Path(__file__).with_name("vessel_lexical.py")),
            },
            "environment": {
                "python": sys.version,
                "sdv": version("sdv"),
                "sdmetrics": version("sdmetrics"),
            },
            "rawSourceIdentitiesModeledByStructuralSdv": False,
            "fitSourceVesselNamesModeledByDiagnosticSdv": True,
            "fitSourceVesselNamesModeledByLexicalRenderer": True,
            "sourceVoyageValuesModeled": False,
            "rawSourceIdentitiesPublished": False,
            "genericBaselineRendererId": FICTIONAL_VESSEL_RENDERER_ID,
            "inputProvenance": dict(input_provenance),
        },
    )

    runs: list[TransportCandidateRun] = []
    examples: dict[str, list[dict[str, Any]]] = {}
    try:
        for candidate in settings.candidates:
            for fold in folds:
                for seed in settings.seeds:
                    train = view.data.iloc[list(fold.train_indices)].reset_index(drop=True).copy()
                    validation = (
                        view.data.iloc[list(fold.validation_indices)].reset_index(drop=True).copy()
                    )
                    fitted = fit_candidate_model(
                        view=view, spec=candidate, train_data=train, seed=seed
                    )
                    validity = _ValidityAccumulator()
                    synthetic, proposal, sample_resources = fitted.sample(
                        requested_rows=len(validation),
                        seed=seed,
                        columns=TRANSPORT_IDENTITY_COLUMNS,
                        acceptance=validity.accept,
                        proposal_multiplier=settings.proposal_multiplier,
                        proposal_batch_rows=settings.proposal_batch_rows,
                    )
                    validity_receipt = validity.receipt(proposal)
                    novelty = _novelty(synthetic, train)
                    evaluation_started = time.perf_counter()
                    evaluation = evaluate_single_table(
                        real_data=validation,
                        synthetic_data=synthetic,
                        metadata=view.metadata,
                        table_name=view.name,
                        diagnostic_reference_data=train,
                    )
                    evaluation_elapsed = time.perf_counter() - evaluation_started
                    realization = _realization_audit(
                        synthetic=synthetic,
                        candidate=candidate.name,
                        fold=fold,
                        seed=seed,
                        guard=bundle.guard,
                        policy=settings.privacy_policy,
                        example_rows=settings.example_rows_per_candidate,
                        vessel_renderer=renderer_by_fold[fold.index],
                    )
                    receipt = TransportCandidateRun(
                        candidate=candidate.name,
                        parameters=candidate.parameters,
                        fold=fold,
                        seed=seed,
                        train_data_sha256=dataframe_sha256(train),
                        validation_data_sha256=dataframe_sha256(validation),
                        synthetic_data_sha256=dataframe_sha256(synthetic),
                        fit=fitted.fit,
                        sample=sample_resources,
                        evaluation_elapsed_seconds=evaluation_elapsed,
                        proposal=proposal,
                        validity=validity_receipt,
                        novelty=novelty,
                        evaluation=evaluation,
                        realization=MappingProxyType(realization),
                    )
                    runs.append(receipt)
                    if candidate.name not in examples:
                        examples[candidate.name] = realization["examples"]
                    atomic_publish_json(
                        artifact_dir
                        / "run-receipts"
                        / candidate.name
                        / f"fold-{fold.index:02d}-seed-{seed}.json",
                        receipt.to_dict(),
                    )
    except Exception as error:
        atomic_publish_json(
            artifact_dir / "failure.json",
            {
                "status": "failed",
                "completedRuns": len(runs),
                "errorType": type(error).__name__,
                "message": str(error),
            },
        )
        raise TransportIdentityBenchmarkError(
            f"transport benchmark failed after {len(runs)} runs: {type(error).__name__}: {error}"
        ) from error

    selection = select_candidate(
        runs=[
            CandidateRunScore(
                candidate=run.candidate,
                fold_index=run.fold.index,
                seed=run.seed,
                diagnostic_score=run.evaluation.diagnostic.score,
                quality_score=run.evaluation.quality.score,
            )
            for run in runs
        ],
        complexity_ranks=DEFAULT_COMPLEXITY_RANKS,
        quality_margin=settings.quality_margin,
        stability_penalty=settings.stability_penalty,
        selectable_candidates=_SELECTABLE_CANDIDATES,
    )
    summaries = [_candidate_summary(candidate, runs) for candidate in _EXPECTED_CANDIDATES]
    realization_audit = _aggregate_realization_audit(runs)
    selected_spec = next(
        candidate
        for candidate in settings.candidates
        if candidate.name == selection.selected_candidate
    )
    selected_lexical_order = int(lexical_comparison["selectedOrder"])
    selected_lexical_source: Literal["source", "faker"] = (
        "faker"
        if str(lexical_comparison["selectedCandidate"]).startswith("faker_company_")
        else "source"
    )
    lexical_selection = {
        "candidate": lexical_comparison["selectedCandidate"],
        "order": selected_lexical_order,
        "source": selected_lexical_source,
        "rendererId": str(lexical_comparison["selectedRendererId"]),
    }
    production_renderer = fit_selected_vessel_renderer(
        bundle=bundle,
        selection=lexical_selection,
        seed=settings.seeds[0],
    )
    production_pilot = generate_transport_identity_pilot(
        bundle=bundle,
        candidate=selected_spec,
        requested_rows=settings.production_pilot_rows,
        seed=settings.seeds[0],
        policy=settings.privacy_policy,
        proposal_multiplier=settings.proposal_multiplier,
        proposal_batch_rows=settings.proposal_batch_rows,
        request_id=f"selected-{selection.selected_candidate}-pilot",
        vessel_renderer=production_renderer,
    )
    result = {
        "status": "complete",
        "experiment": "transport_identity_structural_sdv_comparison_v1",
        "rows": len(view.data),
        "templateGroups": len(set(view.group_ids)),
        "identityConnectedGroups": len(set(connected_group_ids)),
        "runCount": len(runs),
        "candidateSummaries": summaries,
        "selection": selection.to_dict(),
        "recommendedDefault": selection.selected_candidate,
        "selectionScope": "structural_model_only_shared_renderer_v1",
        "productionModelFitted": True,
        "productionModelPersisted": False,
        "productionPilotRows": settings.production_pilot_rows,
        "lexicalRendererSelection": lexical_selection,
        "categoricalSdvReplayDiagnostic": {
            "summaries": categorical_replay["summaries"],
            "productionEligible": False,
        },
        "realizationAudit": realization_audit,
        "rawSourceIdentitiesModeledByStructuralSdv": False,
        "fitSourceVesselNamesModeledByDiagnosticSdv": True,
        "fitSourceVesselNamesModeledByLexicalRenderer": True,
        "sourceVoyageValuesModeled": False,
        "rawSourceIdentitiesPublished": False,
    }
    atomic_publish_json(artifact_dir / "examples.json", examples)
    atomic_publish_json(artifact_dir / "selected-default-pilot.json", production_pilot)
    atomic_publish_json(artifact_dir / "result.json", result)
    atomic_publish_bytes(
        artifact_dir / "REPORT.md",
        _report_markdown(
            summaries=summaries,
            selection=selection,
            rows=len(view.data),
            groups=len(set(view.group_ids)),
            realization_rows=int(realization_audit["rows"]),
            production_pilot_rows=settings.production_pilot_rows,
            lexical_comparison=lexical_comparison,
            categorical_replay=categorical_replay,
            production_pilot=production_pilot,
        ),
    )
    files = []
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_symlink():
            raise TransportIdentityBenchmarkError(f"artifact tree contains symlink: {path}")
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(artifact_dir)),
                    "sha256": sha256_file(path),
                    "sizeBytes": path.stat().st_size,
                }
            )
    atomic_publish_json(
        artifact_dir / "manifest.json",
        {
            "status": "complete",
            "experiment": result["experiment"],
            "recommendedDefault": selection.selected_candidate,
            "files": files,
        },
    )
    return result
