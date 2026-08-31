"""Fit-isolated, source-safe lexical realization for synthetic vessel names.

The renderer is a boundary-aware interpolated character language model. It is
fit only on the declared fit side, after deterministic contamination filtering
and complete-name deduplication. Exact output length is conditioned by a
backward dynamic program that includes EOS likelihood; word/digit layout is
selected before lexical sampling, so character, word, digit, and case structure
are guaranteed rather than repaired. Source names and model counts remain
private. The caller enforces complete-corpus distance and batch uniqueness.
"""

from __future__ import annotations

import math
import statistics
import string
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Literal, Protocol

from document_ocr.synthesis.generators import DeterministicStream

VesselRenderStrategy = Literal[
    "generic_vocabulary",
    "generic_compound",
    "phonotactic_generated",
    "character_ngram",
]

_EOS = "$"
_LETTERS = string.ascii_lowercase
_ACTIONS = _LETTERS + _EOS
_BASE_PSEUDOCOUNT = 0.1
_ALLOWED_FIT_PUNCTUATION = frozenset(" -'")


class VesselNameRenderer(Protocol):
    """Minimal renderer contract consumed by transport identity realization."""

    @property
    def renderer_id(self) -> str: ...

    def render(
        self,
        *,
        word_count: int,
        character_count: int,
        digit_count: int,
        case_style: str,
        stream: DeterministicStream,
    ) -> tuple[str, VesselRenderStrategy]: ...

    def accepts(self, candidate: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class CharacterNGramVesselRenderer:
    """Immutable Witten-Bell-interpolated exact-structure renderer."""

    order: int
    input_name_count: int
    fit_name_count: int
    fit_token_count: int
    fit_exclusion_counts: Mapping[str, int]
    maximum_fit_consonant_run: int
    maximum_fit_repeat_run: int
    renderer_id: str
    _transitions: Mapping[str, tuple[tuple[str, int], ...]]
    _word_layouts: Mapping[tuple[int, int, int], tuple[tuple[tuple[int, int], ...], ...]]

    def render(
        self,
        *,
        word_count: int,
        character_count: int,
        digit_count: int,
        case_style: str,
        stream: DeterministicStream,
    ) -> tuple[str, VesselRenderStrategy]:
        content_count = character_count - max(word_count - 1, 0)
        if word_count < 1 or content_count < word_count or not 0 <= digit_count <= content_count:
            raise ValueError("vessel structure cannot be rendered")
        layout_key = (word_count, content_count, digit_count)
        layouts = self._word_layouts.get(layout_key)
        if layouts:
            word_specs = layouts[stream.randbelow(len(layouts), counter=0)]
        else:
            word_specs = _fallback_word_layout(
                word_count=word_count,
                content_count=content_count,
                digit_count=digit_count,
                stream=stream.derive("fallback-layout"),
            )

        words: list[str] = []
        used_words: set[str] = set()
        for word_index, (letter_count, word_digit_count) in enumerate(word_specs):
            word_stream = stream.derive(f"word-{word_index}")
            letters = ""
            for attempt in range(64):
                letters = self._render_word(
                    length=letter_count,
                    stream=word_stream.derive(f"letters-{attempt}"),
                )
                if not letters or letters not in used_words:
                    break
            else:
                raise RuntimeError("character language model exhausted unique word attempts")
            if letters:
                used_words.add(letters)
            digits = "".join(
                string.digits[word_stream.derive("digits").randbelow(10, counter=position)]
                for position in range(word_digit_count)
            )
            words.append(letters + digits)

        candidate = _apply_case_style(" ".join(words), case_style)
        if (
            len(candidate) != character_count
            or len(candidate.split()) != word_count
            or sum(character.isdigit() for character in candidate) != digit_count
            or _case_style(candidate) != case_style
        ):
            raise RuntimeError("character language model violated exact structure")
        return candidate, "character_ngram"

    def accepts(self, candidate: str) -> bool:
        """Apply only fit-derived lexical plausibility limits."""

        surface = _lexical_surface(candidate)
        if not surface:
            return not any(character.isalpha() for character in candidate)
        return (
            _maximum_consonant_run(surface) <= self.maximum_fit_consonant_run
            and _maximum_repeat_run(surface) <= self.maximum_fit_repeat_run
        )

    def bits_per_character(self, names: Sequence[str]) -> float:
        total_log_probability = 0.0
        total_characters = 0
        for name in names:
            for word in _fit_tokens(name):
                context = "^" * (self.order - 1)
                for character in word + _EOS:
                    probability = self._probability(context=context, action=character)
                    total_log_probability += math.log2(probability)
                    total_characters += 1
                    if character != _EOS:
                        context = (context + character)[-(self.order - 1) :]
        if not total_characters:
            raise ValueError("bits-per-character requires lexical input")
        return -total_log_probability / total_characters

    def _render_word(self, *, length: int, stream: DeterministicStream) -> str:
        if length < 0:
            raise ValueError("word length cannot be negative")
        if length == 0:
            return ""

        @cache
        def log_partition(context: str, remaining: int) -> float:
            if remaining == 0:
                return math.log(self._probability(context=context, action=_EOS))
            return _logsumexp(
                tuple(
                    math.log(self._probability(context=context, action=character))
                    + log_partition((context + character)[-(self.order - 1) :], remaining - 1)
                    for character in _LETTERS
                )
            )

        context = "^" * (self.order - 1)
        output: list[str] = []
        for position in range(length):
            remaining = length - position - 1
            log_weights = tuple(
                (
                    character,
                    math.log(self._probability(context=context, action=character))
                    + log_partition((context + character)[-(self.order - 1) :], remaining),
                )
                for character in _LETTERS
            )
            character = _log_weighted_choice(
                log_weights,
                stream=stream.derive(f"character-{position}"),
            )
            output.append(character)
            context = (context + character)[-(self.order - 1) :]
        return "".join(output)

    def _probability(self, *, context: str, action: str) -> float:
        if action not in _ACTIONS:
            raise ValueError("language-model action is unsupported")
        if not context:
            counts = dict(self._transitions[""])
            denominator = sum(counts.values()) + _BASE_PSEUDOCOUNT * len(_ACTIONS)
            return (counts.get(action, 0) + _BASE_PSEUDOCOUNT) / denominator
        counts = dict(self._transitions.get(context, ()))
        lower = self._probability(context=context[1:], action=action)
        if not counts:
            return lower
        total = sum(counts.values())
        observed_types = len(counts)
        return (
            counts.get(action, 0) / (total + observed_types)
            + (observed_types / (total + observed_types)) * lower
        )


def fit_character_ngram_vessel_renderer(
    *, names: Sequence[str], order: int
) -> CharacterNGramVesselRenderer:
    """Fit a private renderer after explicit contamination filtering."""

    if type(order) is not int or not 2 <= order <= 5:
        raise ValueError("character n-gram order must be in [2, 5]")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("fit vessel names must be non-empty strings")

    eligible: list[str] = []
    exclusion_counts: Counter[str] = Counter()
    for name in names:
        reason = vessel_name_fit_exclusion_reason(name)
        if reason is None:
            eligible.append(name)
        else:
            exclusion_counts[reason] += 1
    if len(eligible) < 20:
        raise ValueError("fewer than 20 vessel names remain after fit filtering")

    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    layouts: dict[tuple[int, int, int], Counter[tuple[tuple[int, int], ...]]] = defaultdict(Counter)
    token_count = 0
    consonant_runs: list[int] = []
    repeat_runs: list[int] = []
    for name in eligible:
        surface = _lexical_surface(name)
        consonant_runs.append(_maximum_consonant_run(surface))
        repeat_runs.append(_maximum_repeat_run(surface))
        tokens = _fit_tokens(name)
        for token in tokens:
            token_count += 1
            context = "^" * (order - 1)
            for character in token + _EOS:
                for context_length in range(len(context), -1, -1):
                    suffix = context[-context_length:] if context_length else ""
                    transitions[suffix][character] += 1
                if character != _EOS:
                    context = (context + character)[-(order - 1) :]
        layout = _observed_word_layout(name)
        if layout is not None:
            key = (
                len(layout),
                sum(letters + digits for letters, digits in layout),
                sum(digits for _, digits in layout),
            )
            layouts[key][layout] += 1
    if token_count < 20 or not transitions.get(""):
        raise ValueError("character language model requires at least 20 lexical tokens")

    return CharacterNGramVesselRenderer(
        order=order,
        input_name_count=len(names),
        fit_name_count=len(eligible),
        fit_token_count=token_count,
        fit_exclusion_counts=MappingProxyType(dict(sorted(exclusion_counts.items()))),
        maximum_fit_consonant_run=max(2, _nearest_rank(consonant_runs, 0.99)),
        maximum_fit_repeat_run=max(2, _nearest_rank(repeat_runs, 0.99)),
        renderer_id=f"fit_isolated_interpolated_character_{order}gram_exact_structure_v2",
        _transitions=MappingProxyType(
            {key: tuple(sorted(value.items())) for key, value in transitions.items()}
        ),
        _word_layouts=MappingProxyType(
            {
                key: tuple(layout for layout, count in sorted(value.items()) for _ in range(count))
                for key, value in layouts.items()
            }
        ),
    )


def vessel_name_fit_exclusion_reason(value: str) -> str | None:
    """Classify values that can contaminate a vessel-only lexical model."""

    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    if any(character in "/\\|" for character in normalized):
        return "compound_or_field_delimiter"
    if any(character.isdigit() for character in normalized):
        return "contains_numeric_or_voyage_suffix"
    if any(
        character not in string.ascii_letters and character not in _ALLOWED_FIT_PUNCTUATION
        for character in normalized
    ):
        return "unsupported_punctuation"
    surface = _lexical_surface(normalized)
    if sum(character.isalpha() for character in surface) < 3:
        return "fewer_than_three_letters"
    return None


def lexical_realism_metrics(
    *, reference_names: Sequence[str], generated_names: Sequence[str]
) -> dict[str, Any]:
    """Compare boundary-aware n-grams and independent lexical shape features."""

    reference = tuple(filter(None, (_lexical_surface(value) for value in reference_names)))
    generated = tuple(filter(None, (_lexical_surface(value) for value in generated_names)))
    if not reference or not generated:
        raise ValueError("lexical realism requires non-empty reference and generated names")
    similarities = {
        str(order): _distribution_similarity(
            _character_ngrams(reference, order),
            _character_ngrams(generated, order),
        )
        for order in (1, 2, 3, 4)
    }
    reference_shape = _shape_summary(reference)
    generated_shape = _shape_summary(generated)
    shape_similarities = {
        "vowelFraction": _unit_distance_similarity(
            reference_shape["vowelFraction"], generated_shape["vowelFraction"]
        ),
        "meanMaximumConsonantRun": _relative_distance_similarity(
            reference_shape["meanMaximumConsonantRun"],
            generated_shape["meanMaximumConsonantRun"],
        ),
        "repeatFraction": _unit_distance_similarity(
            reference_shape["repeatFraction"], generated_shape["repeatFraction"]
        ),
        "meanDistinctLetterRatio": _unit_distance_similarity(
            reference_shape["meanDistinctLetterRatio"],
            generated_shape["meanDistinctLetterRatio"],
        ),
    }
    score = (sum(similarities.values()) + sum(shape_similarities.values())) / (
        len(similarities) + len(shape_similarities)
    )
    return {
        "referenceNames": len(reference),
        "generatedNames": len(generated),
        "characterNgramSimilarity": similarities,
        "referenceShape": reference_shape,
        "generatedShape": generated_shape,
        "shapeSimilarity": shape_similarities,
        "lexicalRealismScore": score,
    }


def lexical_discriminator_auc(
    *, reference_names: Sequence[str], generated_names: Sequence[str], seed: int
) -> dict[str, float | int]:
    """Return cross-validated character 2-4 gram two-sample separability."""

    if not 0 <= seed < 2**32:
        raise ValueError("discriminator seed must be uint32")
    reference = tuple(filter(None, (_lexical_surface(value) for value in reference_names)))
    generated = tuple(filter(None, (_lexical_surface(value) for value in generated_names)))
    if min(len(reference), len(generated)) < 6:
        raise ValueError("lexical discriminator requires at least six names per class")
    sklearn_linear = __import__("sklearn.linear_model", fromlist=["LogisticRegression"])
    sklearn_metrics = __import__("sklearn.metrics", fromlist=["roc_auc_score"])
    sklearn_model_selection = __import__(
        "sklearn.model_selection", fromlist=["StratifiedKFold", "cross_val_predict"]
    )
    sklearn_pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"])
    sklearn_text = __import__("sklearn.feature_extraction.text", fromlist=["TfidfVectorizer"])
    values = (*reference, *generated)
    labels = (0,) * len(reference) + (1,) * len(generated)
    model = sklearn_pipeline.Pipeline(
        [
            (
                "features",
                sklearn_text.TfidfVectorizer(
                    analyzer="char", ngram_range=(2, 4), lowercase=False, sublinear_tf=True
                ),
            ),
            (
                "classifier",
                sklearn_linear.LogisticRegression(
                    max_iter=2_000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    folds = sklearn_model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    probabilities = sklearn_model_selection.cross_val_predict(
        model, values, labels, cv=folds, method="predict_proba", n_jobs=1
    )[:, 1]
    auc = float(sklearn_metrics.roc_auc_score(labels, probabilities))
    separability = max(auc, 1.0 - auc)
    return {
        "referenceNames": len(reference),
        "generatedNames": len(generated),
        "rocAuc": auc,
        "separability": separability,
        "excessOverChance": separability - 0.5,
    }


def deterministic_real_split(
    names: Sequence[str], *, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split distinct real names for a same-distribution discriminator baseline."""

    first: list[str] = []
    second: list[str] = []
    stream = DeterministicStream(seed=seed, namespace="vessel-real-baseline-v1", identity="split")
    for index, name in enumerate(sorted(set(names))):
        target = first if stream.derive(name).randbelow(2, counter=index) == 0 else second
        target.append(name)
    if min(len(first), len(second)) < 6:
        raise ValueError("real baseline split is too small")
    return tuple(first), tuple(second)


def _fit_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return tuple(
        token
        for part in normalized.casefold().split()
        if (
            token := "".join(character for character in part if character in string.ascii_lowercase)
        )
    )


def _observed_word_layout(value: str) -> tuple[tuple[int, int], ...] | None:
    if not value.isascii() or value != value.strip() or "  " in value:
        return None
    parts = value.split(" ")
    if not parts or any(not part or not part.isalnum() for part in parts):
        return None
    layout = tuple(
        (
            sum(character.isalpha() for character in part),
            sum(character.isdigit() for character in part),
        )
        for part in parts
    )
    return layout if all(sum(spec) > 0 for spec in layout) else None


def _fallback_word_layout(
    *,
    word_count: int,
    content_count: int,
    digit_count: int,
    stream: DeterministicStream,
) -> tuple[tuple[int, int], ...]:
    letter_count = content_count - digit_count
    if digit_count and word_count > 1 and letter_count >= word_count - 1:
        letter_lengths = _balanced_composition(letter_count, word_count - 1, stream)
        return (*((length, 0) for length in letter_lengths), (0, digit_count))
    lengths = _balanced_composition(content_count, word_count, stream)
    remaining_digits = digit_count
    reversed_specs: list[tuple[int, int]] = []
    for length in reversed(lengths):
        digits = min(length, remaining_digits)
        reversed_specs.append((length - digits, digits))
        remaining_digits -= digits
    if remaining_digits:
        raise ValueError("vessel digit allocation exceeds word capacity")
    return tuple(reversed(reversed_specs))


def _balanced_composition(total: int, parts: int, stream: DeterministicStream) -> tuple[int, ...]:
    if parts < 1 or total < parts:
        raise ValueError("positive composition requires total >= parts")
    quotient, remainder = divmod(total, parts)
    values = [quotient] * parts
    available = list(range(parts))
    for counter in range(remainder):
        selected = stream.randbelow(len(available), counter=counter)
        values[available.pop(selected)] += 1
    return tuple(values)


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _log_weighted_choice(
    values: Sequence[tuple[str, float]], *, stream: DeterministicStream
) -> str:
    maximum = max(weight for _, weight in values)
    scaled = tuple((value, math.exp(weight - maximum)) for value, weight in values)
    total = sum(weight for _, weight in scaled)
    random_value = int.from_bytes(stream.bytes(counter=0, length=8), "big") / 2**64
    threshold = random_value * total
    cumulative = 0.0
    for value, weight in scaled:
        cumulative += weight
        if threshold < cumulative:
            return value
    return scaled[-1][0]


def _lexical_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(_fit_tokens(normalized))


def _character_ngrams(values: Sequence[str], order: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in values:
        for word in value.split():
            padded = "^" + word + "$"
            counts.update(
                padded[index : index + order] for index in range(max(0, len(padded) - order + 1))
            )
    return counts


def _distribution_similarity(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not keys or not left_total or not right_total:
        raise ValueError("distribution similarity requires non-empty counters")
    divergence = 0.0
    for key in keys:
        left_probability = left[key] / left_total
        right_probability = right[key] / right_total
        midpoint = (left_probability + right_probability) / 2
        if left_probability:
            divergence += 0.5 * left_probability * math.log2(left_probability / midpoint)
        if right_probability:
            divergence += 0.5 * right_probability * math.log2(right_probability / midpoint)
    return max(0.0, 1.0 - divergence)


def _shape_summary(values: Sequence[str]) -> dict[str, float]:
    letters_by_value = [
        tuple(character for character in value if character.isalpha()) for value in values
    ]
    all_letters = [character for letters in letters_by_value for character in letters]
    transitions = sum(max(0, len(letters) - 1) for letters in letters_by_value)
    repeats = sum(
        left == right for letters in letters_by_value for left, right in pairwise(letters)
    )
    return {
        "vowelFraction": sum(character in "aeiou" for character in all_letters) / len(all_letters),
        "meanMaximumConsonantRun": statistics.fmean(
            _maximum_consonant_run(value) for value in values
        ),
        "repeatFraction": repeats / max(1, transitions),
        "meanDistinctLetterRatio": statistics.fmean(
            len(set(letters)) / len(letters) for letters in letters_by_value if letters
        ),
    }


def _maximum_consonant_run(value: str) -> int:
    maximum = 0
    current = 0
    for character in value.casefold():
        if character in _LETTERS and character not in "aeiou":
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _maximum_repeat_run(value: str) -> int:
    maximum = 0
    previous = ""
    current = 0
    for character in value.casefold():
        if character not in _LETTERS:
            previous = ""
            current = 0
            continue
        current = current + 1 if character == previous else 1
        previous = character
        maximum = max(maximum, current)
    return maximum


def _unit_distance_similarity(left: float, right: float) -> float:
    return max(0.0, 1.0 - abs(left - right))


def _relative_distance_similarity(left: float, right: float) -> float:
    return max(0.0, 1.0 - abs(left - right) / max(left, right, 1.0))


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    if not values:
        raise ValueError("nearest rank requires values")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def _case_style(value: str) -> str:
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
    if case_style in {"upper", "caseless"}:
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
    raise ValueError(f"unsupported vessel case style: {case_style}")
