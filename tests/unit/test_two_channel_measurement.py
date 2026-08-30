"""Contracts for the executable Foundations running case."""

import jax.numpy as jnp
import pytest
from examples.onboarding.two_channel_measurement import two_channel_measurement


def test_shared_calibration_creates_positive_channel_correlation() -> None:
    independent = two_channel_measurement(calibration_sigma=0.0)
    shared = two_channel_measurement(calibration_sigma=0.2)

    assert jnp.isclose(independent["correlation"], 0.0)
    assert shared["correlation"] > 0.0
    assert jnp.all(shared["covariance"] == shared["covariance"].T)


def test_zero_separation_creates_an_exact_local_null_direction() -> None:
    case = two_channel_measurement(separation=0.0)

    assert jnp.isclose(case["singular_values"][-1], 0.0, atol=1.0e-6)


@pytest.mark.parametrize("keyword", ("calibration_sigma", "separation"))
def test_case_rejects_negative_controls(keyword: str) -> None:
    kwargs = {keyword: -0.1}
    with pytest.raises(ValueError, match=keyword):
        two_channel_measurement(**kwargs)
