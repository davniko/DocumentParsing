from __future__ import annotations

import sys
import types
from pathlib import Path

from document_ocr.synthesis.generator_method_probe import (
    _deterministic_suffix_baseline,
    _maximum_reported_cuda_peak,
    load_generator_method_probe_config,
    load_probe_corpus,
)
from document_ocr.synthesis.generator_method_probe_support import evaluate_hs_suffixes
from document_ocr.synthesis.lexical_generator_probe import (
    accept_generated_names,
    eligible_distinct_names,
    fit_namemaker_generator,
    normalize_lexical_name,
)
from document_ocr.synthesis.transport_identity import (
    SourceTransportIdentityGuard,
    TransportPrivacyPolicy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/synthesis/mpci_bl_combined1157_generator_methods_gpu_probe.yaml"


class _FakeGenerator:
    generator_id = "fake"

    def generate(self, count: int, *, seed: int) -> tuple[str, ...]:
        del seed
        values = ("AB", "SAFE HORIZON", "SAFE HORIZON", "MARINE VECTOR")
        return tuple(values[index % len(values)] for index in range(count))


def test_probe_config_and_scope_are_pinned_to_the_train_isolated_corpus() -> None:
    config = load_generator_method_probe_config(CONFIG)
    corpus = load_probe_corpus(project_root=PROJECT_ROOT, config=config)

    assert len(corpus.fit_document_ids) == 778
    assert len(corpus.bundle.view.data) == 778
    assert len(set(corpus.bundle.view.group_ids)) == 600


def test_lexical_normalization_and_filtering_are_deterministic() -> None:
    assert normalize_lexical_name("  Mærsk   Horizon ") == "MRSK HORIZON"
    assert eligible_distinct_names(("MSC ALPHA", "msc alpha", "VESSEL/123", None)) == ("MSC ALPHA",)


def test_generated_vessel_acceptance_rejects_syntax_and_duplicates() -> None:
    guard = SourceTransportIdentityGuard.from_documents(
        (
            {
                "document_id": "doc-1",
                "transport_vessel_name": "SOURCE VESSEL",
                "transport_voyage_number": "123A",
            },
        )
    )
    accepted = accept_generated_names(
        generator=_FakeGenerator(),
        requested=2,
        seed=1,
        proposal_multiplier=4,
        guard=guard,
        policy=TransportPrivacyPolicy(
            minimum_normalized_edit_distance=0.2,
            minimum_absolute_edit_distance=2,
            maximum_source_substring_fraction=0.7,
            minimum_source_substring_characters=5,
            maximum_attempts=32,
        ),
    )

    assert accepted.names == ("SAFE HORIZON", "MARINE VECTOR")
    assert accepted.rejections == {
        "batch_duplicate": 1,
        "fewer_than_three_letters": 1,
    }


def test_namemaker_adapter_uses_a_supported_candidate_preference(monkeypatch) -> None:
    module = types.ModuleType("namemaker")
    module.AVG = 2
    module.set_rng = lambda _rng: None

    class _FakeNameSet:
        def __init__(self, _names, *, order: int) -> None:
            assert order == 3

        def make_name(self, *, pref_candidate: int, **_kwargs) -> str:
            assert pref_candidate == module.AVG
            return "SAFE HORIZON"

    module.NameSet = _FakeNameSet
    monkeypatch.setitem(sys.modules, "namemaker", module)
    generator = fit_namemaker_generator(
        names=tuple(
            f"VESSEL {chr(65 + index // 26)}{chr(65 + index % 26)} ALPHA" for index in range(40)
        ),
        order=3,
        maximum_attempts_per_name=100,
    )

    generated = generator.generate(3, seed=7)

    assert len(generated) == 3
    assert all(generated)


def test_hs_suffix_baseline_is_full_code_unique_and_source_safe() -> None:
    source = frozenset({"1234567"})
    seeds = (
        {"hs6": "123456", "output_digits": 7},
        {"hs6": "654321", "output_digits": 7},
    )
    rows = _deterministic_suffix_baseline(seeds=seeds, source_codes=source, seed=7)
    metrics = evaluate_hs_suffixes(generated_rows=rows, source_codes=source)

    assert metrics["validFraction"] == 1
    assert metrics["uniqueFraction"] == 1
    assert metrics["exactSourceReplayFraction"] == 0


def test_aggregate_cuda_peak_uses_maximum_independent_component_receipt() -> None:
    assert (
        _maximum_reported_cuda_peak(
            vessel={
                "runs": [
                    {"training": {"peak_cuda_allocated_bytes": 12}},
                    {"training": {"peakCudaAllocatedBytes": 40}},
                    {"status": "failed"},
                ]
            },
            party={"runtime": {"peakCudaAllocatedBytes": 30}},
            hs={"runtime": {"peakCudaAllocatedBytes": 20}},
        )
        == 40
    )
