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


class SamplerAwarePredictionMixin:
    """Run publication predictions with a known efficient sampler permutation.

    Transformers applies the training length-grouping strategy to evaluation and
    prediction as well. Its sampler is randomized, so source-order metadata cannot
    be zipped onto the resulting prediction rows. This mixin records that sampler's
    actual permutation without changing its batching behavior or random state.
    """

    _capture_publication_prediction: bool = False
    _publication_sampler: _RecordingSampler | None = None

    def _get_eval_sampler(self, eval_dataset: Any) -> Any:
        delegate = super()._get_eval_sampler(eval_dataset)  # type: ignore[misc]
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
            raise PredictionPublicationError(
                "prediction document IDs must be non-empty strings"
            )
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
