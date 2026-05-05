"""Tests for the noise schedule."""
import pytest
import torch

from src.diffusion.schedule import (
    cosine_betas,
    linear_betas,
    make_schedule,
)

T = 1000


def test_linear_betas_shape_and_range():
    betas = linear_betas(T)
    assert betas.shape == (T,)
    assert betas[0] == pytest.approx(1e-4)
    assert betas[-1] == pytest.approx(0.02)
    assert (betas.diff() > 0).all(), "linear betas should be strictly increasing"


def test_cosine_betas_shape_and_clamped():
    betas = cosine_betas(T)
    assert betas.shape == (T,)
    assert (betas > 0).all()
    assert (betas <= 0.999).all(), "cosine betas should be clamped at 0.999"


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_alphas_cumprod_monotonic_decrease(kind):
    s = make_schedule(T=T, kind=kind)
    assert (s.alphas_cumprod.diff() < 0).all()


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_alphas_cumprod_endpoints(kind):
    s = make_schedule(T=T, kind=kind)
    assert s.alphas_cumprod[0] == pytest.approx(1.0, abs=1e-3)
    assert s.alphas_cumprod[-1] < 1e-2, "alphas_cumprod[-1] should be near zero"


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_sqrt_helpers_consistent(kind):
    s = make_schedule(T=T, kind=kind)
    torch.testing.assert_close(s.sqrt_alphas_cumprod ** 2, s.alphas_cumprod)
    torch.testing.assert_close(s.sqrt_one_minus_alphas_cumprod ** 2, 1.0 - s.alphas_cumprod)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown kind"):
        make_schedule(T=T, kind="bogus")
