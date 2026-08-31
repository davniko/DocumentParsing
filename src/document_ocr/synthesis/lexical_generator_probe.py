"""Train-isolated lexical generators used only by the synthesis method probe.

The production renderer remains unchanged.  These adapters deliberately share
one acceptance boundary so NameMaker, recurrent networks, and MOSTLY AI are
compared on identical source-collision, syntax, and uniqueness requirements.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
import statistics
import time
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from document_ocr.hashing import canonical_json_bytes
from document_ocr.synthesis.transport_identity import (
    SourceTransportIdentityGuard,
    TransportPrivacyPolicy,
    transport_identity_key,
)
from document_ocr.synthesis.vessel_lexical import vessel_name_fit_exclusion_reason

_SPACE = re.compile(r"\s+")
_SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")


class LexicalProbeError(RuntimeError):
    """A lexical candidate could not satisfy the declared probe contract."""


class RawNameGenerator(Protocol):
    """Minimal interface shared by every experimental lexical generator."""

    generator_id: str

    def generate(self, count: int, *, seed: int) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class AcceptedNames:
    names: tuple[str, ...]
    raw_candidates: int
    rejections: Mapping[str, int]

    @property
    def acceptance_fraction(self) -> float:
        return len(self.names) / self.raw_candidates


@dataclass(frozen=True, slots=True)
class RecurrentTrainingReceipt:
    architecture: Literal["gru", "lstm"]
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    fit_seconds: float
    peak_cuda_allocated_bytes: int
    parameter_count: int
    selected_temperature: float
    selected_top_p: float
    training_names: int
    validation_names: int


@dataclass(slots=True)
class RecurrentNameGenerator:
    generator_id: str
    model: Any
    character_to_id: Mapping[str, int]
    id_to_character: Mapping[int, str]
    device: str
    temperature: float
    top_p: float
    minimum_characters: int
    maximum_characters: int

    def generate(self, count: int, *, seed: int) -> tuple[str, ...]:
        if count < 1:
            raise ValueError("generated name count must be positive")
        torch = __import__("torch")
        self.model.eval()
        bos = self.character_to_id["<bos>"]
        eos = self.character_to_id["<eos>"]
        output: list[str] = []
        rng = random.Random(seed)
        with torch.inference_mode():
            for _ in range(count):
                hidden = None
                token = torch.tensor([[bos]], device=self.device)
                characters: list[str] = []
                for _position in range(self.maximum_characters):
                    logits, hidden = self.model(token, hidden)
                    probabilities = torch.softmax(
                        logits[0, -1].float().cpu() / self.temperature,
                        dim=-1,
                    )
                    probabilities[: len(_SPECIAL_TOKENS) - 1] = 0
                    if len(characters) < self.minimum_characters:
                        probabilities[eos] = 0
                    probabilities = _top_p(probabilities, self.top_p)
                    threshold = rng.random()
                    cumulative = 0.0
                    selected = eos
                    for index, probability in enumerate(probabilities.tolist()):
                        cumulative += probability
                        if threshold < cumulative:
                            selected = index
                            break
                    if selected == eos:
                        break
                    characters.append(self.id_to_character[selected])
                    token = torch.tensor([[selected]], device=self.device)
                output.append("".join(characters).strip())
        return tuple(output)


@dataclass(slots=True)
class NameMakerGenerator:
    generator_id: str
    model: Any
    maximum_attempts_per_name: int

    def generate(self, count: int, *, seed: int) -> tuple[str, ...]:
        if count < 1:
            raise ValueError("generated name count must be positive")
        namemaker = __import__("namemaker")
        namemaker.set_rng(random.Random(seed))
        output: list[str] = []
        for _ in range(count):
            value = self.model.make_name(
                exclude_real_names=True,
                exclude_history=True,
                add_to_history=True,
                n_candidates=8,
                pref_candidate=namemaker.AVG,
                max_attempts=self.maximum_attempts_per_name,
            )
            output.append(str(value or ""))
        return tuple(output)


def normalize_lexical_name(value: str) -> str:
    """Return the exact ASCII/case surface used by every probe candidate."""

    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _SPACE.sub(" ", normalized.strip()).upper()


def eligible_distinct_names(values: Sequence[str | None]) -> tuple[str, ...]:
    output: dict[str, str] = {}
    for value in values:
        if value is None or vessel_name_fit_exclusion_reason(value) is not None:
            continue
        normalized = normalize_lexical_name(value)
        if normalized:
            output.setdefault(transport_identity_key(normalized), normalized)
    return tuple(output[key] for key in sorted(output))


def fit_namemaker_generator(
    *, names: Sequence[str], order: int, maximum_attempts_per_name: int
) -> NameMakerGenerator:
    if type(order) is not int or not 2 <= order <= 5:
        raise ValueError("NameMaker order must be in [2, 5]")
    if maximum_attempts_per_name < 1:
        raise ValueError("NameMaker maximum attempts must be positive")
    normalized = eligible_distinct_names(tuple(names))
    if len(normalized) < 20:
        raise ValueError("NameMaker requires at least 20 eligible names")
    namemaker = __import__("namemaker")
    return NameMakerGenerator(
        generator_id=f"namemaker_markov_order_{order}_v1",
        model=namemaker.NameSet(normalized, order=order),
        maximum_attempts_per_name=maximum_attempts_per_name,
    )


def accept_generated_names(
    *,
    generator: RawNameGenerator,
    requested: int,
    seed: int,
    proposal_multiplier: int,
    guard: SourceTransportIdentityGuard,
    policy: TransportPrivacyPolicy,
) -> AcceptedNames:
    """Apply one bounded, auditable acceptance boundary to raw candidates."""

    if requested < 1 or proposal_multiplier < 1:
        raise ValueError("requested names and proposal multiplier must be positive")
    raw = generator.generate(requested * proposal_multiplier, seed=seed)
    accepted: list[str] = []
    accepted_keys: set[str] = set()
    rejections: Counter[str] = Counter()
    for value in raw:
        candidate = normalize_lexical_name(value)
        reason = vessel_name_fit_exclusion_reason(candidate) if candidate else "empty"
        if reason is not None:
            rejections[str(reason)] += 1
            continue
        key = transport_identity_key(candidate)
        if key in accepted_keys:
            rejections["batch_duplicate"] += 1
            continue
        source_reason = guard.classify_vessel_name(candidate, policy=policy)
        if source_reason != "none":
            rejections[f"source_{source_reason}"] += 1
            continue
        accepted_keys.add(key)
        accepted.append(candidate)
        if len(accepted) == requested:
            break
    consumed = len(raw) if len(accepted) < requested else sum(rejections.values()) + len(accepted)
    if len(accepted) < requested:
        raise LexicalProbeError(
            f"{generator.generator_id} accepted {len(accepted)}/{requested} names "
            f"from {len(raw)} bounded proposals"
        )
    return AcceptedNames(
        names=tuple(accepted),
        raw_candidates=consumed,
        rejections=dict(sorted(rejections.items())),
    )


def fit_recurrent_name_generator(
    *,
    names: Sequence[str],
    architecture: Literal["gru", "lstm"],
    seed: int,
    embedding_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    maximum_epochs: int,
    patience: int,
    validation_fraction: float,
    temperatures: Sequence[float],
    top_p: float,
    minimum_characters: int,
    maximum_characters: int,
    device: str,
) -> tuple[RecurrentNameGenerator, RecurrentTrainingReceipt]:
    """Fit a compact makemore-style character GRU or LSTM on one outer fold."""

    if architecture not in {"gru", "lstm"}:
        raise ValueError("architecture must be gru or lstm")
    normalized = eligible_distinct_names(tuple(names))
    if len(normalized) < 40:
        raise ValueError("recurrent lexical fit requires at least 40 eligible names")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation fraction must be in (0, 0.5)")
    if not temperatures or any(value <= 0 or not math.isfinite(value) for value in temperatures):
        raise ValueError("temperatures must be finite positive values")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if min(embedding_dim, hidden_dim, layers, batch_size, maximum_epochs, patience) < 1:
        raise ValueError("recurrent dimensions and bounds must be positive")
    if not 0 <= dropout < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("recurrent optimization parameters are invalid")
    if not 1 <= minimum_characters < maximum_characters:
        raise ValueError("recurrent character limits are invalid")

    torch = __import__("torch")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for recurrent lexical fitting")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    characters = tuple(sorted(set("".join(normalized))))
    vocabulary = (*_SPECIAL_TOKENS, *characters)
    character_to_id = {value: index for index, value in enumerate(vocabulary)}
    id_to_character = {index: value for value, index in character_to_id.items()}
    pad, bos, eos = (character_to_id[value] for value in _SPECIAL_TOKENS)

    inner_train: list[str] = []
    inner_validation: list[str] = []
    threshold = int(validation_fraction * 10_000)
    for name in normalized:
        bucket = (
            int.from_bytes(hashlib.sha256(canonical_json_bytes([seed, name])).digest()[:4], "big")
            % 10_000
        )
        (inner_validation if bucket < threshold else inner_train).append(name)
    if min(len(inner_train), len(inner_validation)) < 8:
        ordered = sorted(normalized)
        boundary = max(8, round(len(ordered) * validation_fraction))
        inner_validation, inner_train = ordered[:boundary], ordered[boundary:]

    class _CharacterRNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(len(vocabulary), embedding_dim, padding_idx=pad)
            recurrent = torch.nn.GRU if architecture == "gru" else torch.nn.LSTM
            self.recurrent = recurrent(
                embedding_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.output = torch.nn.Linear(hidden_dim, len(vocabulary))

        def forward(self, tokens: Any, hidden: Any = None) -> tuple[Any, Any]:
            values, hidden = self.recurrent(self.embedding(tokens), hidden)
            return self.output(values), hidden

    model = _CharacterRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=pad)

    def batches(values: Sequence[str], epoch: int) -> Sequence[tuple[Any, Any]]:
        order = list(range(len(values)))
        random.Random(seed + epoch).shuffle(order)
        output = []
        for start in range(0, len(order), batch_size):
            rows = [values[index] for index in order[start : start + batch_size]]
            encoded = [[bos, *(character_to_id[c] for c in row), eos] for row in rows]
            width = max(len(row) for row in encoded)
            tokens = torch.tensor(
                [row + [pad] * (width - len(row)) for row in encoded],
                dtype=torch.long,
                device=device,
            )
            output.append((tokens[:, :-1], tokens[:, 1:]))
        return output

    started = time.perf_counter()
    best_loss = math.inf
    best_epoch = 0
    best_state: Mapping[str, Any] | None = None
    stale_epochs = 0
    completed = 0
    for epoch in range(1, maximum_epochs + 1):
        completed = epoch
        model.train()
        for inputs, targets in batches(inner_train, epoch):
            optimizer.zero_grad(set_to_none=True)
            logits, _hidden = model(inputs)
            loss = loss_function(logits.reshape(-1, len(vocabulary)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for inputs, targets in batches(inner_validation, 0):
                logits, _hidden = model(inputs)
                losses.append(
                    float(
                        loss_function(
                            logits.reshape(-1, len(vocabulary)), targets.reshape(-1)
                        ).item()
                    )
                )
        validation_loss = statistics.fmean(losses)
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise LexicalProbeError("recurrent lexical fit produced no checkpoint")
    model.load_state_dict(best_state)
    generator = RecurrentNameGenerator(
        generator_id=f"makemore_style_char_{architecture}_v1",
        model=model,
        character_to_id=character_to_id,
        id_to_character=id_to_character,
        device=device,
        temperature=float(temperatures[0]),
        top_p=top_p,
        minimum_characters=minimum_characters,
        maximum_characters=maximum_characters,
    )
    # Temperature selection is inner-validation-only; the outer fold remains untouched.
    from document_ocr.synthesis.vessel_lexical import lexical_realism_metrics

    selected_temperature = float(temperatures[0])
    selected_score = -math.inf
    selection_count = min(64, len(inner_validation))
    for index, temperature in enumerate(temperatures):
        generator.temperature = float(temperature)
        candidates = tuple(
            value
            for value in generator.generate(selection_count * 3, seed=seed + 10_000 + index)
            if value and vessel_name_fit_exclusion_reason(value) is None
        )[:selection_count]
        if len(candidates) < 6:
            continue
        score = float(
            lexical_realism_metrics(
                reference_names=inner_validation,
                generated_names=candidates,
            )["lexicalRealismScore"]
        )
        if score > selected_score:
            selected_score = score
            selected_temperature = float(temperature)
    generator.temperature = selected_temperature
    peak = int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0
    receipt = RecurrentTrainingReceipt(
        architecture=architecture,
        epochs_completed=completed,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        fit_seconds=time.perf_counter() - started,
        peak_cuda_allocated_bytes=peak,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        selected_temperature=selected_temperature,
        selected_top_p=top_p,
        training_names=len(inner_train),
        validation_names=len(inner_validation),
    )
    return generator, receipt


def _top_p(probabilities: Any, top_p: float) -> Any:
    torch = __import__("torch")
    if top_p >= 1:
        total = probabilities.sum()
        return probabilities / total
    ordered, indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(ordered, dim=-1)
    keep = cumulative <= top_p
    keep[0] = True
    filtered = torch.zeros_like(probabilities)
    filtered[indices[keep]] = probabilities[indices[keep]]
    return filtered / filtered.sum()
