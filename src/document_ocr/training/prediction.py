"""Identity-safe prediction ordering for structured training artifacts."""

from __future__ import annotations

import operator
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


class PredictionPublicationError(RuntimeError):
    """Raised when predictions cannot be mapped to source documents exactly."""


@dataclass(frozen=True, slots=True)
class IdentityOrderedPrediction:
    """One prediction output plus the document identities in its exact row order."""

    output: Any
    document_ids: tuple[str, ...]


class _RecordingSampler:
    """Record the exact indices yielded by one existing Trainer sampler."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._indices: list[int] = []
        self._started = False

    def __iter__(self) -> Iterator[int]:
        if self._started:
            raise PredictionPublicationError(
                "publication prediction sampler was consumed more than once"
            )
        self._started = True
        for raw_index in self._delegate:
            if isinstance(raw_index, bool):
                raise PredictionPublicationError("prediction sampler yielded a boolean index")
            try:
                index = operator.index(raw_index)
            except TypeError as error:
                raise PredictionPublicationError(
                    "prediction sampler yielded a non-integer index"
                ) from error
            self._indices.append(index)
            yield index

    def __len__(self) -> int:
        return len(self._delegate)

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(self._indices)


@dataclass(frozen=True, slots=True)
class _OrderedEvaluationSampler:
    """An explicit, repeatable evaluation order, independent of training RNG."""

    indices: tuple[int, ...]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def length_sorted_evaluation_indices(lengths: Any) -> tuple[int, ...]:
    """Sort longest first, keeping source order for ties; never consume random state."""

    values: list[int] = []
    for raw_length in lengths:
        if isinstance(raw_length, bool):
            raise PredictionPublicationError("evaluation input length must not be boolean")
        try:
            length = operator.index(raw_length)
        except TypeError as error:
            raise PredictionPublicationError("evaluation input lengths must be integers") from error
        if length <= 0:
            raise PredictionPublicationError("evaluation input lengths must be positive")
        values.append(length)
    if not values:
        raise PredictionPublicationError("evaluation dataset must not be empty")
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))


class SamplerAwarePredictionMixin:
    """Use fixed evaluation batches and record prediction identities from actual order.

    Transformers' training length-grouping strategy also randomizes evaluation.
    BF16 generation is sensitive to the resulting batch/padding shapes. Replace
    evaluation-only grouping with a stable length sort; leave training untouched.
    """

    _capture_publication_prediction: bool = False
    _publication_sampler: _RecordingSampler | None = None
    evaluation_recorder: Any = None
    # Checkpoint evaluation may supply its certified historical permutation.
    # Training leaves this unset and uses the deterministic policy below.
    evaluation_indices: tuple[int, ...] | None = None

    def _get_eval_sampler(self, eval_dataset: Any) -> Any:
        if self.evaluation_indices is not None:
            if self.args.world_size != 1:  # type: ignore[attr-defined]
                raise PredictionPublicationError("fixed evaluation requires one process")
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in self.evaluation_indices
            ) or sorted(self.evaluation_indices) != list(range(len(eval_dataset))):
                raise PredictionPublicationError(
                    "recorded evaluation order is not a full permutation"
                )
            delegate = _OrderedEvaluationSampler(self.evaluation_indices)
        else:
            delegate = super()._get_eval_sampler(eval_dataset)  # type: ignore[misc]
        if (
            self.evaluation_indices is None
            and self.args.train_sampling_strategy == "group_by_length"  # type: ignore[attr-defined]
        ):
            if self.args.world_size != 1:  # type: ignore[attr-defined]
                raise PredictionPublicationError("fixed evaluation requires one process")
            # The pinned Trainer has already validated/extracted the configured
            # lengths. Missing lengths are an execution-contract error, not a
            # reason to silently switch sampling algorithms.
            if delegate is None or not hasattr(delegate, "lengths"):
                raise PredictionPublicationError("length-grouped evaluation lacks input lengths")
            delegate = _OrderedEvaluationSampler(length_sorted_evaluation_indices(delegate.lengths))
        if not self._capture_publication_prediction:
            return delegate
        if self._publication_sampler is not None:
            raise PredictionPublicationError(
                "publication prediction requested more than one evaluation sampler"
            )
        if delegate is None:
            raise PredictionPublicationError(
                "publication prediction requires an indexable evaluation sampler"
            )
        recorder = _RecordingSampler(delegate)
        self._publication_sampler = recorder
        return recorder

    def evaluation_loop(self, dataloader: Any, *args: Any, **kwargs: Any) -> Any:
        output = super().evaluation_loop(dataloader, *args, **kwargs)  # type: ignore[misc]
        # Final best-model publication is deliberately separate from terminal-step
        # evaluation. Never attach best weights' predictions to the last step.
        if self.evaluation_recorder is not None and self.is_in_train:  # type: ignore[attr-defined]
            from torch.utils.data import SequentialSampler

            sampler = dataloader.batch_sampler.sampler
            if isinstance(sampler, _OrderedEvaluationSampler):
                indices = sampler.indices
            elif isinstance(sampler, SequentialSampler):
                indices = tuple(range(len(dataloader.dataset)))
            else:
                raise PredictionPublicationError("unrecognized scheduled evaluation sampler")
            self.evaluation_recorder.record(
                step=self.state.global_step,  # type: ignore[attr-defined]
                dataset=dataloader.dataset,
                indices=indices,
                output=output,
            )
        return output

    def _save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        super()._save_checkpoint(*args, **kwargs)  # type: ignore[misc]
        if self.evaluation_recorder is not None:
            self.evaluation_recorder.checkpoint_saved(
                step=self.state.global_step,  # type: ignore[attr-defined]
                output_dir=self.args.output_dir,  # type: ignore[attr-defined]
            )

    def predict_with_identities(
        self,
        dataset: Any,
        *,
        metric_key_prefix: str,
    ) -> IdentityOrderedPrediction:
        """Predict once and return the exact source identity of every output row."""

        if self._capture_publication_prediction:
            raise PredictionPublicationError("nested publication prediction is forbidden")
        args = getattr(self, "args", None)
        if args is None or getattr(args, "world_size", None) != 1:
            raise PredictionPublicationError(
                "identity-safe prediction currently requires exactly one distributed process"
            )
        if getattr(args, "dataloader_drop_last", None) is not False:
            raise PredictionPublicationError(
                "identity-safe prediction requires dataloader_drop_last=false"
            )

        try:
            source_document_ids = list(dataset["document_id"])
        except (KeyError, TypeError) as error:
            raise PredictionPublicationError(
                "prediction dataset must expose a document_id column"
            ) from error
        if not source_document_ids:
            raise PredictionPublicationError("prediction dataset is empty")
        if any(
            not isinstance(document_id, str) or not document_id
            for document_id in source_document_ids
        ):
            raise PredictionPublicationError("prediction document IDs must be non-empty strings")
        if len(source_document_ids) != len(set(source_document_ids)):
            raise PredictionPublicationError("prediction document IDs must be unique")

        self._capture_publication_prediction = True
        self._publication_sampler = None
        try:
            output = super().predict(  # type: ignore[misc]
                dataset,
                metric_key_prefix=metric_key_prefix,
            )
            sampler = self._publication_sampler
            if sampler is None:
                raise PredictionPublicationError(
                    "publication prediction did not consume its identity sampler"
                )
            indices = sampler.indices
            expected_indices = list(range(len(source_document_ids)))
            if len(indices) != len(source_document_ids) or sorted(indices) != expected_indices:
                raise PredictionPublicationError(
                    "publication prediction sampler did not yield each source row exactly once"
                )
            document_ids = tuple(source_document_ids[index] for index in indices)
            return IdentityOrderedPrediction(output=output, document_ids=document_ids)
        finally:
            self._capture_publication_prediction = False
            self._publication_sampler = None
