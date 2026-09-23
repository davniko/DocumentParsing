from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tools.analyze_lora_spectrum import (  # noqa: E402
    spectrum_metrics,
    update_singular_values,
)


def test_core_spectrum_matches_dense_update_and_not_separate_factors():
    generator = torch.Generator().manual_seed(7)
    a = torch.randn(5, 19, generator=generator, dtype=torch.float64)
    b = torch.randn(23, 5, generator=generator, dtype=torch.float64)
    actual = update_singular_values(a, b)
    torch.testing.assert_close(actual, torch.linalg.svdvals(b @ a)[:5])
    transform = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64))
    torch.testing.assert_close(
        actual, update_singular_values(transform @ a, b @ transform.inverse())
    )


def test_metrics_distinguish_concentrated_and_flat_updates():
    flat = spectrum_metrics(torch.ones(32))
    concentrated = spectrum_metrics(torch.tensor([1.0] + [0.0] * 31))
    assert flat["rank_99"] == 32
    assert flat["stable_rank"] == 32
    assert concentrated["rank_99"] == 1
    assert concentrated["stable_rank"] == 1


def test_invalid_or_zero_updates_fail_explicitly():
    with pytest.raises(ValueError, match="shapes"):
        update_singular_values(torch.ones(2, 5), torch.ones(5, 3))
    with pytest.raises(ValueError, match="zero update"):
        spectrum_metrics(torch.zeros(4))
