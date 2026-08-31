"""Pinned GPU experiment for lexical and mixed-type synthesis methods.

This module is deliberately outside the production generation path.  It
compares train/template-isolated candidates, publishes only source-safe
examples, and records enough provenance to reject a weak method without
silently promoting it into semantic generation.
"""

from __future__ import annotations

import json
import math
import resource
import shutil
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.generator_method_probe_support import (
    evaluate_character_fields,
    evaluate_hs_suffixes,
    evaluate_vessel_names,
    plot_generator_probe,
)
from document_ocr.synthesis.lexical_generator_probe import (
    AcceptedNames,
    RawNameGenerator,
    accept_generated_names,
    eligible_distinct_names,
    fit_namemaker_generator,
    fit_recurrent_name_generator,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.sdv_harness import grouped_folds
from document_ocr.synthesis.transport_identity import (
    TransportIdentityBundle,
    TransportPrivacyPolicy,
    build_transport_identity_bundle,
)
from document_ocr.synthesis.transport_identity_benchmark import (
    _identity_connected_group_ids,
)
from document_ocr.synthesis.vessel_lexical import (
    deterministic_real_split,
    fit_character_ngram_vessel_renderer,
    lexical_discriminator_auc,
    lexical_realism_metrics,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class GeneratorProbeError(RuntimeError):
    """The experiment violated its declared data or method contract."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class PinnedFile(_StrictModel):
    path: NonEmptyString
    sha256: Sha256


class PinnedDataset(PinnedFile):
    records: Annotated[int, Field(gt=0)]


class ProbeInputs(_StrictModel):
    preparation_root: NonEmptyString
    preparation_manifest: PinnedFile
    template_groups: PinnedDataset
    partition_report: PinnedFile
    vessel_sdv_result: PinnedFile
    vessel_lexical_result: PinnedFile
    party_sdv_result: PinnedFile


class ProbeRun(_StrictModel):
    run_id: NonEmptyString
    output_dir: NonEmptyString


class ProbeSelection(_StrictModel):
    split: NonEmptyString
    require_template_wholly_in_split: Literal[True]
    fold_count: Annotated[int, Field(ge=2)]
    fold_seed: Annotated[int, Field(ge=0, lt=2**32)]
    names_per_fold: Annotated[int, Field(ge=20)]


class RecurrentProbeConfig(_StrictModel):
    architectures: list[Literal["gru", "lstm"]] = Field(min_length=2, max_length=2)
    embedding_dim: Annotated[int, Field(gt=0)]
    hidden_dim: Annotated[int, Field(gt=0)]
    layers: Annotated[int, Field(gt=0)]
    dropout: Annotated[float, Field(ge=0, lt=1)]
    batch_size: Annotated[int, Field(gt=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)]
    maximum_epochs: Annotated[int, Field(gt=0)]
    patience: Annotated[int, Field(gt=0)]
    validation_fraction: Annotated[float, Field(gt=0, lt=0.5)]
    temperatures: list[Annotated[float, Field(gt=0)]] = Field(min_length=1)
    top_p: Annotated[float, Field(gt=0, le=1)]
    minimum_characters: Annotated[int, Field(gt=0)]
    maximum_characters: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def complete_architecture_pair(self) -> RecurrentProbeConfig:
        if tuple(self.architectures) != ("gru", "lstm"):
            raise ValueError("recurrent probe must compare GRU then LSTM")
        if self.minimum_characters >= self.maximum_characters:
            raise ValueError("recurrent maximum characters must exceed minimum")
        return self


class MostlyLanguageConfig(_StrictModel):
    model: Literal["MOSTLY_AI/LSTMFromScratch-3m"]
    maximum_epochs: Annotated[int, Field(gt=0)]
    maximum_training_minutes: Annotated[float, Field(gt=0)]
    batch_size: Annotated[int, Field(gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(gt=0)]
    temperature: Annotated[float, Field(gt=0)]
    top_p: Annotated[float, Field(gt=0, le=1)]


class VesselProbeConfig(_StrictModel):
    candidates: list[
        Literal[
            "namemaker_order2",
            "namemaker_order3",
            "namemaker_order4",
            "char_gru",
            "char_lstm",
            "mostlyai_lstm_from_scratch",
        ],
    ] = Field(min_length=6, max_length=6)
    seed: Annotated[int, Field(ge=0, lt=2**32)]
    proposal_multiplier: Annotated[int, Field(gt=0)]
    namemaker_maximum_attempts: Annotated[int, Field(gt=0)]
    recurrent: RecurrentProbeConfig
    mostlyai: MostlyLanguageConfig

    @model_validator(mode="after")
    def complete_candidate_set(self) -> VesselProbeConfig:
        expected = (
            "namemaker_order2",
            "namemaker_order3",
            "namemaker_order4",
            "char_gru",
            "char_lstm",
            "mostlyai_lstm_from_scratch",
        )
        if tuple(self.candidates) != expected:
            raise ValueError(f"vessel candidates must be exactly {expected}")
        return self


class MostlyTabularConfig(_StrictModel):
    model: Literal["MOSTLY_AI/Small"]
    maximum_epochs: Annotated[int, Field(gt=0)]
    maximum_training_minutes: Annotated[float, Field(gt=0)]
    batch_size: Annotated[int, Field(gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(gt=0)]
    sample_rows: Annotated[int, Field(ge=50)]
    temperature: Annotated[float, Field(gt=0)]
    top_p: Annotated[float, Field(gt=0, le=1)]


class ProbePrivacyConfig(_StrictModel):
    minimum_normalized_edit_distance: Annotated[float, Field(ge=0, lt=1)]
    minimum_absolute_edit_distance: Annotated[int, Field(gt=0)]
    maximum_source_substring_fraction: Annotated[float, Field(gt=0, le=1)]
    minimum_source_substring_characters: Annotated[int, Field(ge=2)]
    maximum_attempts: Annotated[int, Field(gt=0)]

    def to_policy(self) -> TransportPrivacyPolicy:
        return TransportPrivacyPolicy(
            minimum_normalized_edit_distance=self.minimum_normalized_edit_distance,
            minimum_absolute_edit_distance=self.minimum_absolute_edit_distance,
            maximum_source_substring_fraction=self.maximum_source_substring_fraction,
            minimum_source_substring_characters=self.minimum_source_substring_characters,
            maximum_attempts=self.maximum_attempts,
        )


class GeneratorMethodProbeConfig(_StrictModel):
    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: ProbeRun
    inputs: ProbeInputs
    selection: ProbeSelection
    vessel: VesselProbeConfig
    party: MostlyTabularConfig
    hs_suffix: MostlyTabularConfig
    privacy: ProbePrivacyConfig
    publish_model_weights: Literal[False]
    publish_unsafe_generated_values: Literal[False]


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    output: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise ValueError(f"duplicate YAML key: {key!r}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclass(frozen=True, slots=True)
class ProbeCorpus:
    documents: tuple[dict[str, Any], ...]
    parties: tuple[dict[str, Any], ...]
    hs_codes: tuple[dict[str, Any], ...]
    fit_document_ids: tuple[str, ...]
    template_by_document: Mapping[str, str]
    partition_by_document: Mapping[str, str]
    bundle: TransportIdentityBundle
    input_receipts: Mapping[str, Any]


class _MostlyLanguageGenerator:
    generator_id = "mostlyai_lstm_from_scratch_v1"

    def __init__(
        self,
        *,
        model: Any,
        batch_size: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, count: int, *, seed: int) -> tuple[str, ...]:
        del seed  # The fitted engine owns the explicitly seeded random stream.
        values = self.model.sample(
            count,
            batch_size=self.batch_size,
            sampling_temperature=self.temperature,
            sampling_top_p=self.top_p,
            device="cuda",
        )
        if "vessel_name" not in values:
            raise GeneratorProbeError("MOSTLY language output has no vessel_name column")
        return tuple(str(value) for value in values["vessel_name"].fillna("").tolist())


def load_generator_method_probe_config(path: Path) -> GeneratorMethodProbeConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("generator probe config is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("generator probe config root must be a mapping")
    return GeneratorMethodProbeConfig.model_validate(value, strict=True)


def _resolve_file(project_root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else project_root / candidate
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _resolve_directory(project_root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else project_root / candidate
    if path.is_symlink() or not path.resolve(strict=True).is_dir():
        raise ValueError(f"{label} must be a plain directory")
    return path.resolve(strict=True)


def _read_json(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _read_jsonl(
    path: Path, *, expected_sha256: str, expected_records: int, label: str
) -> tuple[dict[str, Any], ...]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank rows are forbidden")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            rows.append(value)
    if len(rows) != expected_records:
        raise ValueError(f"{label} expected {expected_records} rows, found {len(rows)}")
    return tuple(rows)


def _load_prepared_table(
    *, root: Path, manifest: Mapping[str, Any], table: str
) -> tuple[dict[str, Any], ...]:
    relative = f"tables/{table}.jsonl"
    receipt = (manifest.get("files") or {}).get(relative)
    count = ((manifest.get("domainProjection") or {}).get("tableRows") or {}).get(table)
    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("sha256"), str):
        raise ValueError(f"preparation manifest has no receipt for {relative}")
    if type(count) is not int:
        raise ValueError(f"preparation manifest has no row count for {table}")
    return _read_jsonl(
        root / relative,
        expected_sha256=cast(str, receipt["sha256"]),
        expected_records=count,
        label=table,
    )


def _template_maps(
    rows: Sequence[Mapping[str, Any]], corpus_ids: frozenset[str]
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    by_document: dict[str, str] = {}
    members_by_template: dict[str, tuple[str, ...]] = {}
    for row in rows:
        template = row.get("template_id")
        members = row.get("member_document_ids")
        if not isinstance(template, str) or not isinstance(members, list):
            raise ValueError("template row has invalid fields")
        frozen = tuple(cast(list[str], members))
        if template in members_by_template or len(frozen) != len(set(frozen)):
            raise ValueError(f"invalid template group: {template}")
        members_by_template[template] = frozen
        for document_id in frozen:
            if document_id in by_document:
                raise ValueError(f"document occurs in multiple templates: {document_id}")
            by_document[document_id] = template
    if set(by_document) != corpus_ids:
        raise ValueError("template groups do not cover the corpus")
    return by_document, members_by_template


def _partition_map(report: Mapping[str, Any], corpus_ids: frozenset[str]) -> dict[str, str]:
    outputs = ((report.get("inspection") or {}).get("partition") or {}).get("outputs") or {}
    if not isinstance(outputs, Mapping):
        raise ValueError("partition report outputs are malformed")
    result: dict[str, str] = {}
    for split, receipt in outputs.items():
        if not isinstance(split, str) or not isinstance(receipt, Mapping):
            raise ValueError("partition split is malformed")
        values = receipt.get("document_ids")
        if not isinstance(values, list) or receipt.get("records") != len(values):
            raise ValueError(f"partition split count differs: {split}")
        for document_id in values:
            if not isinstance(document_id, str) or document_id in result:
                raise ValueError(f"invalid partition document: {document_id!r}")
            result[document_id] = split
    if set(result) != corpus_ids:
        raise ValueError("partition report does not cover the corpus")
    return result


def load_probe_corpus(*, project_root: Path, config: GeneratorMethodProbeConfig) -> ProbeCorpus:
    root = _resolve_directory(
        project_root, config.inputs.preparation_root, label="preparation root"
    )
    manifest_path = _resolve_file(
        project_root,
        config.inputs.preparation_manifest.path,
        label="preparation manifest",
    )
    if manifest_path.parent != root:
        raise ValueError("preparation manifest is outside its pinned root")
    manifest = _read_json(
        manifest_path,
        expected_sha256=config.inputs.preparation_manifest.sha256,
        label="preparation manifest",
    )
    documents = _load_prepared_table(root=root, manifest=manifest, table="documents")
    parties = _load_prepared_table(root=root, manifest=manifest, table="parties")
    hs_codes = _load_prepared_table(root=root, manifest=manifest, table="cargo_hs_codes")
    corpus_ids = frozenset(cast(str, row["document_id"]) for row in documents)
    template_rows = _read_jsonl(
        _resolve_file(project_root, config.inputs.template_groups.path, label="template groups"),
        expected_sha256=config.inputs.template_groups.sha256,
        expected_records=config.inputs.template_groups.records,
        label="template groups",
    )
    template_by_document, members_by_template = _template_maps(template_rows, corpus_ids)
    partition = _partition_map(
        _read_json(
            _resolve_file(
                project_root, config.inputs.partition_report.path, label="partition report"
            ),
            expected_sha256=config.inputs.partition_report.sha256,
            label="partition report",
        ),
        corpus_ids,
    )
    raw_fit = tuple(
        sorted(
            document_id
            for document_id, split in partition.items()
            if split == config.selection.split
        )
    )
    raw_fit_set = frozenset(raw_fit)
    fit_ids = tuple(
        sorted(
            document_id
            for document_id in raw_fit
            if set(members_by_template[template_by_document[document_id]]) <= raw_fit_set
        )
    )
    bundle = build_transport_identity_bundle(
        source_documents=documents,
        fit_document_ids=fit_ids,
        template_by_document=template_by_document,
        partition_by_document=partition,
        allowed_partition=config.selection.split,
    )
    receipts = {
        "preparationManifestSha256": config.inputs.preparation_manifest.sha256,
        "templateGroupsSha256": config.inputs.template_groups.sha256,
        "partitionReportSha256": config.inputs.partition_report.sha256,
        "fitDocuments": len(fit_ids),
        "fitTemplates": len({template_by_document[value] for value in fit_ids}),
        "fitDocumentIdsSha256": sha256_bytes(canonical_json_bytes(fit_ids)),
    }
    return ProbeCorpus(
        documents=documents,
        parties=parties,
        hs_codes=hs_codes,
        fit_document_ids=fit_ids,
        template_by_document=template_by_document,
        partition_by_document=partition,
        bundle=bundle,
        input_receipts=receipts,
    )


def _ranked_subset(values: Sequence[str], *, count: int, seed: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                sha256_bytes(canonical_json_bytes([seed, value])),
                value,
            ),
        )[:count]
    )


def _mostly_progress(workspace: Path) -> dict[str, Any]:
    progress = workspace / "ModelStore/model-data/progress-messages.csv"
    if not progress.is_file():
        raise GeneratorProbeError("MOSTLY AI training produced no progress receipt")
    pandas = __import__("pandas")
    frame = pandas.read_csv(progress)
    if frame.empty:
        raise GeneratorProbeError("MOSTLY AI progress receipt is empty")
    final = frame.iloc[-1]
    validation = frame["val_loss"].dropna()
    return {
        "epochsCompleted": float(final["epoch"]),
        "stepsCompleted": int(final["steps"]),
        "bestValidationLoss": float(validation.min()) if len(validation) else None,
        "engineTrainingSeconds": float(final["total_time"]),
    }


def _fit_mostly_language(
    *, names: Sequence[str], config: MostlyLanguageConfig, seed: int
) -> tuple[RawNameGenerator, dict[str, Any], Path]:
    pandas = __import__("pandas")
    torch = __import__("torch")
    mostly = __import__("mostlyai.engine", fromlist=["LanguageModel"])
    workspace = Path(tempfile.mkdtemp(prefix="documentparsing-mostly-language-"))
    torch.cuda.reset_peak_memory_stats("cuda")
    model = mostly.LanguageModel(
        tgt_encoding_types={"vessel_name": "LANGUAGE_TEXT"},
        model=config.model,
        max_epochs=config.maximum_epochs,
        max_training_time=config.maximum_training_minutes,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        enable_flexible_generation=True,
        value_protection=False,
        device="cuda",
        workspace_dir=str(workspace),
        random_state=seed,
        verbose=0,
    )
    started = time.perf_counter()
    model.fit(pandas.DataFrame({"vessel_name": tuple(names)}))
    fit_seconds = time.perf_counter() - started
    receipt = {
        "model": config.model,
        "fitNames": len(names),
        "fitSeconds": fit_seconds,
        "peakCudaAllocatedBytes": int(torch.cuda.max_memory_allocated("cuda")),
        **_mostly_progress(workspace),
    }
    return (
        _MostlyLanguageGenerator(
            model=model,
            batch_size=config.batch_size,
            temperature=config.temperature,
            top_p=config.top_p,
        ),
        receipt,
        workspace,
    )


def _accepted_vessel_result(
    *,
    candidate: str,
    fold_index: int,
    fit_seconds: float,
    sample_seconds: float,
    training_receipt: Mapping[str, Any],
    accepted: AcceptedNames,
    reference_names: Sequence[str],
    evaluator: Any,
    bundle: TransportIdentityBundle,
    seed: int,
) -> dict[str, Any]:
    metrics = evaluate_vessel_names(
        reference_names=reference_names,
        generated_names=accepted.names,
        evaluator=evaluator,
        guard=bundle.guard,
        seed=seed,
    )
    return {
        "status": "complete",
        "candidate": candidate,
        "foldIndex": fold_index,
        "fitSeconds": fit_seconds,
        "sampleSeconds": sample_seconds,
        "totalSeconds": fit_seconds + sample_seconds,
        "acceptanceFraction": accepted.acceptance_fraction,
        "rawCandidatesConsumed": accepted.raw_candidates,
        "rejectionCounts": dict(accepted.rejections),
        "training": dict(training_receipt),
        "metrics": metrics,
        "safeExamples": list(accepted.names[:12]),
    }


def _run_vessel_probe(*, corpus: ProbeCorpus, config: GeneratorMethodProbeConfig) -> dict[str, Any]:
    connected_groups = _identity_connected_group_ids(corpus.bundle)
    folds = grouped_folds(
        group_ids=connected_groups,
        fold_count=config.selection.fold_count,
        seed=config.selection.fold_seed,
    )
    runs: list[dict[str, Any]] = []
    real_baselines: list[dict[str, Any]] = []
    for fold in folds:
        fit_names = eligible_distinct_names(
            tuple(corpus.bundle.fit_vessel_names[index] for index in fold.train_indices)
        )
        heldout_names = eligible_distinct_names(
            tuple(corpus.bundle.fit_vessel_names[index] for index in fold.validation_indices)
        )
        reference = _ranked_subset(
            heldout_names,
            count=min(config.selection.names_per_fold, len(heldout_names)),
            seed=config.vessel.seed + fold.index,
        )
        if len(reference) < 20:
            raise GeneratorProbeError(f"fold {fold.index} has fewer than 20 held-out names")
        evaluator = fit_character_ngram_vessel_renderer(names=fit_names, order=3)
        first, second = deterministic_real_split(
            reference,
            seed=config.vessel.seed + fold.index,
        )
        real_baselines.append(
            {
                "foldIndex": fold.index,
                "lexicalRealism": lexical_realism_metrics(
                    reference_names=first,
                    generated_names=second,
                ),
                "discriminator": lexical_discriminator_auc(
                    reference_names=first,
                    generated_names=second,
                    seed=config.vessel.seed + fold.index,
                ),
                "bitsPerCharacterGap": abs(
                    evaluator.bits_per_character(first) - evaluator.bits_per_character(second)
                ),
            }
        )
        for candidate in config.vessel.candidates:
            workspace: Path | None = None
            try:
                started = time.perf_counter()
                training: dict[str, Any]
                if candidate.startswith("namemaker_order"):
                    order = int(candidate.removeprefix("namemaker_order"))
                    generator = fit_namemaker_generator(
                        names=fit_names,
                        order=order,
                        maximum_attempts_per_name=config.vessel.namemaker_maximum_attempts,
                    )
                    training = {"order": order, "fitNames": len(fit_names), "device": "cpu"}
                    fit_seconds = time.perf_counter() - started
                elif candidate in {"char_gru", "char_lstm"}:
                    architecture = cast(
                        Literal["gru", "lstm"],
                        candidate.removeprefix("char_"),
                    )
                    recurrent = config.vessel.recurrent
                    generator, receipt = fit_recurrent_name_generator(
                        names=fit_names,
                        architecture=architecture,
                        seed=config.vessel.seed + fold.index,
                        embedding_dim=recurrent.embedding_dim,
                        hidden_dim=recurrent.hidden_dim,
                        layers=recurrent.layers,
                        dropout=recurrent.dropout,
                        batch_size=recurrent.batch_size,
                        learning_rate=recurrent.learning_rate,
                        weight_decay=recurrent.weight_decay,
                        maximum_epochs=recurrent.maximum_epochs,
                        patience=recurrent.patience,
                        validation_fraction=recurrent.validation_fraction,
                        temperatures=recurrent.temperatures,
                        top_p=recurrent.top_p,
                        minimum_characters=recurrent.minimum_characters,
                        maximum_characters=recurrent.maximum_characters,
                        device="cuda",
                    )
                    training = asdict(receipt)
                    fit_seconds = receipt.fit_seconds
                else:
                    generator, training, workspace = _fit_mostly_language(
                        names=fit_names,
                        config=config.vessel.mostlyai,
                        seed=config.vessel.seed + fold.index,
                    )
                    fit_seconds = float(training["fitSeconds"])
                sample_started = time.perf_counter()
                accepted = accept_generated_names(
                    generator=generator,
                    requested=len(reference),
                    seed=config.vessel.seed + fold.index,
                    proposal_multiplier=config.vessel.proposal_multiplier,
                    guard=corpus.bundle.guard,
                    policy=config.privacy.to_policy(),
                )
                sample_seconds = time.perf_counter() - sample_started
                runs.append(
                    _accepted_vessel_result(
                        candidate=candidate,
                        fold_index=fold.index,
                        fit_seconds=fit_seconds,
                        sample_seconds=sample_seconds,
                        training_receipt=training,
                        accepted=accepted,
                        reference_names=reference,
                        evaluator=evaluator,
                        bundle=corpus.bundle,
                        seed=config.vessel.seed + fold.index,
                    )
                )
            except Exception as error:
                runs.append(
                    {
                        "status": "failed",
                        "candidate": candidate,
                        "foldIndex": fold.index,
                        "errorType": type(error).__name__,
                        "message": str(error),
                    }
                )
            finally:
                if workspace is not None:
                    shutil.rmtree(workspace)
    summaries: list[dict[str, Any]] = []
    for candidate in config.vessel.candidates:
        selected = [
            row for row in runs if row["candidate"] == candidate and row["status"] == "complete"
        ]
        summaries.append(
            {
                "candidate": candidate,
                "completedFolds": len(selected),
                "failedFolds": config.selection.fold_count - len(selected),
                "meanLexicalRealism": (
                    statistics.fmean(
                        float(row["metrics"]["lexicalRealism"]["lexicalRealismScore"])
                        for row in selected
                    )
                    if selected
                    else None
                ),
                "meanDiscriminatorExcess": (
                    statistics.fmean(
                        float(row["metrics"]["discriminator"]["excessOverChance"])
                        for row in selected
                    )
                    if selected
                    else None
                ),
                "meanBitsPerCharacterGap": (
                    statistics.fmean(
                        float(row["metrics"]["bitsPerCharacter"]["absoluteGap"]) for row in selected
                    )
                    if selected
                    else None
                ),
                "meanAcceptanceFraction": (
                    statistics.fmean(float(row["acceptanceFraction"]) for row in selected)
                    if selected
                    else None
                ),
                "totalFitSeconds": sum(float(row["fitSeconds"]) for row in selected),
                "totalSampleSeconds": sum(float(row["sampleSeconds"]) for row in selected),
                "totalSeconds": sum(float(row["totalSeconds"]) for row in selected),
            }
        )
    real_data_summary = {
        "completedFolds": len(real_baselines),
        "meanLexicalRealism": statistics.fmean(
            float(row["lexicalRealism"]["lexicalRealismScore"]) for row in real_baselines
        ),
        "meanDiscriminatorExcess": statistics.fmean(
            float(row["discriminator"]["excessOverChance"]) for row in real_baselines
        ),
        "meanBitsPerCharacterGap": statistics.fmean(
            float(row["bitsPerCharacterGap"]) for row in real_baselines
        ),
    }
    return {
        "fitDocuments": corpus.bundle.fit_document_count,
        "identityConnectedGroups": len(set(connected_groups)),
        "folds": config.selection.fold_count,
        "realDataBaselines": real_baselines,
        "realDataSummary": real_data_summary,
        "runs": runs,
        "summaries": summaries,
        "productionSelectionPerformed": False,
    }


def _fit_and_sample_mostly_tabular(
    *,
    training_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    encodings: Mapping[str, str],
    config: MostlyTabularConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pandas = __import__("pandas")
    torch = __import__("torch")
    mostly = __import__("mostlyai.engine", fromlist=["TabularARGN"])
    if not training_rows or not seed_rows:
        raise ValueError("MOSTLY tabular probe requires training and seed rows")
    workspace = Path(tempfile.mkdtemp(prefix="documentparsing-mostly-tabular-"))
    try:
        torch.cuda.reset_peak_memory_stats("cuda")
        model = mostly.TabularARGN(
            model=config.model,
            max_epochs=config.maximum_epochs,
            max_training_time=config.maximum_training_minutes,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            enable_flexible_generation=True,
            value_protection=False,
            tgt_encoding_types=dict(encodings),
            device="cuda",
            workspace_dir=str(workspace),
            random_state=seed,
            verbose=0,
        )
        started = time.perf_counter()
        model.fit(pandas.DataFrame.from_records(training_rows))
        fit_seconds = time.perf_counter() - started
        sample_started = time.perf_counter()
        synthetic = model.sample(
            seed_data=pandas.DataFrame.from_records(seed_rows),
            batch_size=config.batch_size,
            sampling_temperature=config.temperature,
            sampling_top_p=config.top_p,
            device="cuda",
        )
        sample_seconds = time.perf_counter() - sample_started
        rows = cast(list[dict[str, Any]], synthetic.to_dict(orient="records"))
        return rows, {
            "model": config.model,
            "fitRows": len(training_rows),
            "sampleRows": len(rows),
            "fitSeconds": fit_seconds,
            "sampleSeconds": sample_seconds,
            "peakCudaAllocatedBytes": int(torch.cuda.max_memory_allocated("cuda")),
            **_mostly_progress(workspace),
        }
    finally:
        shutil.rmtree(workspace)


def _probe_split_ids(
    corpus: ProbeCorpus, config: GeneratorMethodProbeConfig
) -> tuple[frozenset[str], frozenset[str]]:
    connected = _identity_connected_group_ids(corpus.bundle)
    fold = grouped_folds(
        group_ids=connected,
        fold_count=config.selection.fold_count,
        seed=config.selection.fold_seed,
    )[0]
    return (
        frozenset(corpus.bundle.view.row_ids[index] for index in fold.train_indices),
        frozenset(corpus.bundle.view.row_ids[index] for index in fold.validation_indices),
    )


def _rank_mapping_rows(
    rows: Sequence[Mapping[str, Any]], *, count: int, seed: int
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: sha256_bytes(canonical_json_bytes([seed, dict(row)])),
        )[:count]
    )


def _run_party_probe(*, corpus: ProbeCorpus, config: GeneratorMethodProbeConfig) -> dict[str, Any]:
    train_ids, validation_ids = _probe_split_ids(corpus, config)

    def usable(row: Mapping[str, Any]) -> bool:
        return all(isinstance(row.get(field), str) and row.get(field) for field in ("role", "name"))

    training = [
        {field: row.get(field) for field in ("role", "country", "city", "name", "address")}
        for row in corpus.parties
        if row.get("document_id") in train_ids and usable(row)
    ]
    validation = [
        {field: row.get(field) for field in ("role", "country", "city", "name", "address")}
        for row in corpus.parties
        if row.get("document_id") in validation_ids and usable(row)
    ]
    selected = _rank_mapping_rows(
        validation,
        count=min(config.party.sample_rows, len(validation)),
        seed=config.vessel.seed,
    )
    seeds = [{field: row.get(field) for field in ("role", "country", "city")} for row in selected]
    generated, runtime = _fit_and_sample_mostly_tabular(
        training_rows=training,
        seed_rows=seeds,
        encodings={
            "role": "TABULAR_CATEGORICAL",
            "country": "TABULAR_CATEGORICAL",
            "city": "TABULAR_CATEGORICAL",
            "name": "TABULAR_CHARACTER",
            "address": "TABULAR_CHARACTER",
        },
        config=config.party,
        seed=config.vessel.seed,
    )
    conditioned_mismatches = Counter()
    for expected, actual in zip(seeds, generated, strict=True):
        for field in expected:
            expected_value = expected[field]
            actual_value = actual.get(field)
            if expected_value is None:
                if actual_value is not None and not (
                    isinstance(actual_value, float) and math.isnan(actual_value)
                ):
                    conditioned_mismatches[field] += 1
            elif actual_value != expected_value:
                conditioned_mismatches[field] += 1
    field_metrics = evaluate_character_fields(
        reference_rows=selected,
        generated_rows=generated,
        source_rows=corpus.parties,
        fields=("name", "address"),
    )
    return {
        "status": "complete",
        "purpose": "route_conditioned_party_identity_character_probe_not_production",
        "runtime": runtime,
        "conditionedSeedRows": len(seeds),
        "conditionedMismatchCounts": dict(sorted(conditioned_mismatches.items())),
        "fieldMetrics": field_metrics,
        "rawGeneratedValuesPublished": False,
        "productionEligible": False,
    }


def _hs_extension_rows(
    rows: Sequence[Mapping[str, Any]], allowed_ids: frozenset[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("value")
        if row.get("document_id") not in allowed_ids or not isinstance(value, str):
            continue
        digits = "".join(character for character in value if character.isdigit())
        if digits == value and 7 <= len(digits) <= 18:
            output.append({"hs6": digits[:6], "output_digits": len(digits), "suffix": digits[6:]})
    return output


def _deterministic_suffix_baseline(
    *, seeds: Sequence[Mapping[str, Any]], source_codes: frozenset[str], seed: int
) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in seeds:
        digits = int(row["output_digits"])
        hs6 = str(row["hs6"])
        for _attempt in range(10_000):
            suffix = "".join(str(rng.randrange(10)) for _ in range(digits - 6))
            code = hs6 + suffix
            if code not in source_codes and code not in used:
                used.add(code)
                output.append({"hs6": hs6, "output_digits": digits, "suffix": suffix})
                break
        else:
            raise GeneratorProbeError("deterministic HS suffix space exhausted")
    return output


def _run_hs_suffix_probe(
    *, corpus: ProbeCorpus, config: GeneratorMethodProbeConfig
) -> dict[str, Any]:
    train_ids, validation_ids = _probe_split_ids(corpus, config)
    training = _hs_extension_rows(corpus.hs_codes, train_ids)
    validation = _hs_extension_rows(corpus.hs_codes, validation_ids)
    selected = _rank_mapping_rows(
        validation,
        count=min(config.hs_suffix.sample_rows, len(validation)),
        seed=config.vessel.seed,
    )
    seeds = [
        {"hs6": str(row["hs6"]), "output_digits": int(row["output_digits"])} for row in selected
    ]
    generated, runtime = _fit_and_sample_mostly_tabular(
        training_rows=training,
        seed_rows=seeds,
        encodings={
            "hs6": "TABULAR_CATEGORICAL",
            "output_digits": "TABULAR_NUMERIC_DISCRETE",
            "suffix": "TABULAR_CHARACTER",
        },
        config=config.hs_suffix,
        seed=config.vessel.seed + 1,
    )
    source_codes = frozenset(
        str(row["hs6"]) + str(row["suffix"])
        for row in _hs_extension_rows(
            corpus.hs_codes,
            frozenset(cast(str, row["document_id"]) for row in corpus.documents),
        )
    )
    mostly_metrics = evaluate_hs_suffixes(
        generated_rows=generated,
        source_codes=source_codes,
    )
    baseline_rows = _deterministic_suffix_baseline(
        seeds=seeds,
        source_codes=source_codes,
        seed=config.vessel.seed,
    )
    baseline_metrics = evaluate_hs_suffixes(
        generated_rows=baseline_rows,
        source_codes=source_codes,
    )
    return {
        "status": "complete",
        "purpose": "national_suffix_method_probe_not_registry_authority",
        "runtime": runtime,
        "trainingRows": len(training),
        "validationRows": len(validation),
        "mostlyai": mostly_metrics,
        "deterministicCollisionCheckedBaseline": baseline_metrics,
        "rawGeneratedValuesPublished": False,
        "productionEligible": False,
    }


def _package_versions() -> dict[str, str]:
    metadata = __import__("importlib.metadata", fromlist=["version"])
    result = {}
    for package in (
        "mostlyai-engine",
        "namemaker",
        "torch",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
    ):
        result[package] = metadata.version(package)
    return result


def _report(
    *,
    vessel: Mapping[str, Any],
    party: Mapping[str, Any],
    hs: Mapping[str, Any],
    baselines: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> str:
    lines = [
        "# Generator method GPU probe",
        "",
        "This is a grouped, train-isolated experiment. It does not publish model weights, unsafe "
        "party strings, or training-ready samples.",
        "",
        "## Vessel lexical comparison",
        "",
        "| Candidate | Completed folds | Lexical realism | Discriminator excess | BPC gap | "
        "Acceptance | Fit seconds | Sample seconds | Total seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(row: Mapping[str, Any], field: str) -> str:
        value = row[field]
        return "-" if value is None else f"{float(value):.4f}"

    real = vessel["realDataSummary"]
    lines.append(
        f"| real_vs_real | {real['completedFolds']} | "
        f"{number(real, 'meanLexicalRealism')} | "
        f"{number(real, 'meanDiscriminatorExcess')} | "
        f"{number(real, 'meanBitsPerCharacterGap')} | - | - | - | - |"
    )
    for row in vessel["summaries"]:
        lines.append(
            f"| {row['candidate']} | {row['completedFolds']} | "
            f"{number(row, 'meanLexicalRealism')} | "
            f"{number(row, 'meanDiscriminatorExcess')} | "
            f"{number(row, 'meanBitsPerCharacterGap')} | "
            f"{number(row, 'meanAcceptanceFraction')} | "
            f"{float(row['totalFitSeconds']):.2f} | "
            f"{float(row['totalSampleSeconds']):.2f} | "
            f"{float(row['totalSeconds']):.2f} |"
        )
    lines.extend(
        [
            "",
            "No production method is selected by this probe. Candidate outputs must also support "
            "the exact word/character structure requested by the semantic plan before promotion.",
            "",
            "## Route-conditioned party identity probe",
            "",
            f"- Seed rows: {party['conditionedSeedRows']}",
            f"- Seed mismatches: {party['conditionedMismatchCounts']}",
            f"- Name metrics: {party['fieldMetrics']['name']}",
            f"- Address metrics: {party['fieldMetrics']['address']}",
            "- Raw party outputs are intentionally not published.",
            "",
            "## HS national-suffix probe",
            "",
            f"- MOSTLY AI: {hs['mostlyai']}",
            f"- Deterministic collision-checked baseline: "
            f"{hs['deterministicCollisionCheckedBaseline']}",
            "- These are explicitly synthetic national suffixes, not authoritative tariff leaves.",
            "",
            "## Existing measured baselines",
            "",
            f"```json\n{json.dumps(baselines, indent=2, sort_keys=True)}\n```",
            "",
            "## Environment",
            "",
            f"```json\n{json.dumps(environment, indent=2, sort_keys=True)}\n```",
            "",
        ]
    )
    return "\n".join(lines)


def _maximum_reported_cuda_peak(
    *, vessel: Mapping[str, Any], party: Mapping[str, Any], hs: Mapping[str, Any]
) -> int:
    """Return the largest independently reset component peak, not the final counter."""

    peaks: list[int] = []
    for run in vessel["runs"]:
        training = run.get("training")
        if not isinstance(training, Mapping):
            continue
        value = training.get(
            "peakCudaAllocatedBytes",
            training.get("peak_cuda_allocated_bytes"),
        )
        if type(value) is int and value >= 0:
            peaks.append(value)
    for component in (party, hs):
        runtime = component.get("runtime")
        if not isinstance(runtime, Mapping):
            continue
        value = runtime.get("peakCudaAllocatedBytes")
        if type(value) is int and value >= 0:
            peaks.append(value)
    if not peaks:
        raise GeneratorProbeError("completed probe has no component CUDA peak receipts")
    return max(peaks)


def run_generator_method_probe(
    *, project_root: Path, config_path: Path, config: GeneratorMethodProbeConfig
) -> dict[str, Any]:
    started = time.perf_counter()
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("generator method probe requires CUDA")
    corpus = load_probe_corpus(project_root=project_root, config=config)
    baselines = {
        "vesselSdv": _read_json(
            _resolve_file(
                project_root, config.inputs.vessel_sdv_result.path, label="vessel SDV result"
            ),
            expected_sha256=config.inputs.vessel_sdv_result.sha256,
            label="vessel SDV result",
        ),
        "vesselLexical": _read_json(
            _resolve_file(
                project_root,
                config.inputs.vessel_lexical_result.path,
                label="vessel lexical result",
            ),
            expected_sha256=config.inputs.vessel_lexical_result.sha256,
            label="vessel lexical result",
        ),
        "partySdv": _read_json(
            _resolve_file(
                project_root, config.inputs.party_sdv_result.path, label="party SDV result"
            ),
            expected_sha256=config.inputs.party_sdv_result.sha256,
            label="party SDV result",
        ),
    }
    vessel = _run_vessel_probe(corpus=corpus, config=config)
    party = _run_party_probe(corpus=corpus, config=config)
    hs = _run_hs_suffix_probe(corpus=corpus, config=config)
    environment = {
        "packages": _package_versions(),
        "cudaDevice": torch.cuda.get_device_name(0),
        "cudaCapability": list(torch.cuda.get_device_capability(0)),
        "torchCudaVersion": torch.version.cuda,
        "maximumReportedComponentPeakCudaAllocatedBytes": _maximum_reported_cuda_peak(
            vessel=vessel,
            party=party,
            hs=hs,
        ),
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    compact_baselines = {
        "vesselStructuralRecommended": baselines["vesselSdv"].get("recommendedDefault"),
        "vesselLexicalSelected": baselines["vesselLexical"].get("selectedCandidate"),
        "partySdvCandidateSummaries": baselines["partySdv"].get("candidateSummaries"),
    }
    result = {
        "schemaVersion": 1,
        "status": "complete_experiment_no_production_selection",
        "runId": config.run.run_id,
        "inputReceipts": dict(corpus.input_receipts),
        "vessel": vessel,
        "party": party,
        "hsSuffix": hs,
        "existingBaselines": compact_baselines,
        "environment": environment,
        "elapsedSeconds": time.perf_counter() - started,
        "modelWeightsPublished": False,
        "unsafeGeneratedValuesPublished": False,
    }
    implementation = {
        Path(__file__).name: sha256_file(Path(__file__)),
        "lexical_generator_probe.py": sha256_file(
            Path(__file__).with_name("lexical_generator_probe.py")
        ),
        "generator_method_probe_support.py": sha256_file(
            Path(__file__).with_name("generator_method_probe_support.py")
        ),
    }
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "contract": "generator-method-gpu-probe-v1",
                "configSha256": sha256_file(config_path),
                "config": config.model_dump(mode="json"),
                "inputs": corpus.input_receipts,
                "implementation": implementation,
                "packages": environment["packages"],
            }
        )
    )
    output_parent = Path(config.run.output_dir)
    if not output_parent.is_absolute():
        output_parent = project_root / output_parent
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run.run_id,
        transaction_sha256=transaction,
    )
    if stage.completed:
        stage.validate_committed_run()
        return cast(dict[str, Any], json.loads((stage.final_root / "manifest.json").read_bytes()))
    stage.recover_interrupted_temporary_files()
    stage.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    stage.publish_json("result.json", result)
    stage.publish_bytes(
        "REPORT.md",
        _report(
            vessel=vessel,
            party=party,
            hs=hs,
            baselines=compact_baselines,
            environment=environment,
        ).encode("utf-8"),
    )
    with tempfile.TemporaryDirectory(prefix="documentparsing-generator-plots-") as plot_dir:
        plot_root = Path(plot_dir)
        plot_generator_probe(
            vessel_runs=cast(Sequence[Mapping[str, Any]], vessel["runs"]),
            real_baselines=cast(Sequence[Mapping[str, Any]], vessel["realDataBaselines"]),
            method_results={"party": party, "hsSuffix": hs["mostlyai"]},
            output=plot_root,
        )
        plot_paths = sorted(plot_root.glob("*.png"))
        for path in plot_paths:
            stage.publish_bytes(f"plots/{path.name}", path.read_bytes())
    manifest = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "status": result["status"],
        "trainingEligible": False,
        "productionSelectionPerformed": False,
        "modelWeightsPublished": False,
        "unsafeGeneratedValuesPublished": False,
        "implementationSha256": implementation,
        "transactionSha256": transaction,
        "resultSha256": sha256_bytes(canonical_json_bytes(result)),
        "elapsedSeconds": result["elapsedSeconds"],
    }
    stage.publish_json("manifest.json", manifest)
    expected = (
        "REPORT.md",
        "config.yaml",
        "manifest.json",
        "result.json",
        *(f"plots/{path.name}" for path in plot_paths),
    )
    commit = stage.commit(
        expected_artifacts=expected,
        metadata={
            "runId": config.run.run_id,
            "status": result["status"],
            "productionSelectionPerformed": False,
        },
    )
    return {
        **manifest,
        "commitCreated": commit.created,
        "commitContentSha256": commit.receipt.content_sha256,
    }
