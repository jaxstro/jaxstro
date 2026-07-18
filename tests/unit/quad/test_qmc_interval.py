import jax
import jax.numpy as jnp
import pytest

from jaxstro.quad._qmc_interval import (
    empirical_bernstein_half_width,
    fixed_look_interval,
    spent_alpha,
    student_t_quantile,
)


def test_fixed_look_interval_uses_unbiased_replicate_variance():
    estimates = jnp.array(
        [0.9, 1.0, 1.1, 1.2, 0.8, 1.05, 0.95, 1.0],
        dtype=jnp.float64,
    )
    interval = fixed_look_interval(estimates, confidence_level=0.95)
    sample_variance = jnp.sum((estimates - jnp.mean(estimates)) ** 2) / 7
    expected_se = jnp.sqrt(sample_variance / 8)
    assert jnp.allclose(interval.standard_error, expected_se)
    assert interval.half_width > 0.0
    assert interval.valid


@pytest.mark.parametrize(
    ("degrees_of_freedom", "confidence_level", "expected"),
    (
        (7, 0.90, 1.894578605),
        (7, 0.95, 2.364624252),
        (7, 0.99, 3.499483297),
        (15, 0.90, 1.753050356),
        (15, 0.95, 2.131449546),
        (15, 0.99, 2.946712883),
        (31, 0.90, 1.695518783),
        (31, 0.95, 2.039513446),
        (31, 0.99, 2.744041919),
        (63, 0.90, 1.669402222),
        (63, 0.95, 1.998340543),
        (63, 0.99, 2.656145030),
    ),
)
def test_student_t_quantiles_match_scipy_1_16_0(
    degrees_of_freedom,
    confidence_level,
    expected,
):
    probability = jnp.asarray(
        0.5 * (1.0 + confidence_level),
        dtype=jnp.float64,
    )
    actual = student_t_quantile(probability, degrees_of_freedom)
    assert jnp.allclose(actual, expected, rtol=2.0e-9, atol=2.0e-9)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_supported_tail_probability_is_finite_under_jit(dtype):
    quantile = jax.jit(student_t_quantile)(
        jnp.asarray(0.999999, dtype=dtype),
        7,
    )
    assert jnp.isfinite(quantile)
    assert quantile > 0.0


@pytest.mark.parametrize(
    ("probability", "expected", "rtol", "atol"),
    (
        (0.500001, 2.631951849724282e-6, 2.0e-3, 2.0e-9),
        (0.999999, 14.213835920455033, 2.0e-3, 2.0e-3),
    ),
)
def test_float32_quantile_avoids_cdf_cancellation(
    probability,
    expected,
    rtol,
    atol,
):
    # Anchors use SciPy 1.16.0 at the exact represented float32 probability.
    actual = student_t_quantile(jnp.asarray(probability, dtype=jnp.float32), 7)
    assert jnp.allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("probability", (0.5, 1.0, jnp.nan))
def test_unsupported_probability_fails_closed(probability):
    quantile = student_t_quantile(jnp.asarray(probability), 7)
    assert jnp.isnan(quantile)


def test_interval_rejects_too_few_or_nonscalar_replicates():
    with pytest.raises(ValueError, match="at least two"):
        fixed_look_interval(jnp.asarray([1.0]), confidence_level=0.95)
    with pytest.raises(ValueError, match="one-dimensional"):
        fixed_look_interval(jnp.ones((8, 2)), confidence_level=0.95)


def test_alpha_spending_sums_to_requested_alpha_from_below():
    alpha = jnp.asarray(0.05, dtype=jnp.float64)
    spent = sum(spent_alpha(alpha, inspection) for inspection in range(10000))
    assert spent < alpha
    assert jnp.allclose(spent, alpha, rtol=2.0e-4)


def test_empirical_bernstein_range_term_shrinks_only_with_replicates():
    same_replicates = empirical_bernstein_half_width(
        jnp.zeros(8),
        lower=0.0,
        upper=1.0,
        alpha=0.05,
    )
    more_replicates = empirical_bernstein_half_width(
        jnp.zeros(32),
        lower=0.0,
        upper=1.0,
        alpha=0.05,
    )
    assert more_replicates < same_replicates


def test_empirical_bernstein_uses_unbiased_variance_and_exact_range_term():
    estimates = jnp.asarray(
        (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9),
        dtype=jnp.float64,
    )
    alpha = jnp.asarray(0.01, dtype=jnp.float64)
    actual = empirical_bernstein_half_width(
        estimates,
        lower=0.0,
        upper=1.0,
        alpha=alpha,
    )
    mean = jnp.mean(estimates)
    variance = jnp.sum((estimates - mean) ** 2) / 7
    log_term = jnp.log(2.0 / alpha)
    expected = jnp.sqrt(2.0 * variance * log_term / 8) + (7.0 * log_term / (3.0 * 7))
    assert jnp.allclose(actual, expected)
