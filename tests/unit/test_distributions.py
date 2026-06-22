"""Tests for generic distribution kernels."""

import jax
import jax.numpy as jnp

from jaxstro.numerics import distributions
from jaxstro.numerics.integration import trapz


class TestNormal:
    """Tests for normal distribution kernels."""

    def test_normal_logpdf_standard_value(self):
        value = distributions.normal_logpdf(jnp.array(0.0))
        expected = -0.5 * jnp.log(2.0 * jnp.pi)
        assert jnp.allclose(value, expected)

    def test_normal_cdf_ppf_round_trip(self):
        u = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])
        x = distributions.normal_ppf(u, loc=1.0, scale=2.0)
        assert jnp.allclose(distributions.normal_cdf(x, loc=1.0, scale=2.0), u)

    def test_normal_logpdf_integrates_to_one_on_wide_grid(self):
        x = jnp.linspace(-8.0, 8.0, 4097)
        density = jnp.exp(distributions.normal_logpdf(x))
        assert jnp.allclose(trapz(density, x=x), 1.0, rtol=1e-8)


class TestLogNormal:
    """Tests for lognormal distribution kernels."""

    def test_lognormal_support_is_explicit(self):
        x = jnp.array([-1.0, 0.0, 1.0])
        logpdf = distributions.lognormal_logpdf(x)
        cdf = distributions.lognormal_cdf(x)
        assert jnp.isneginf(logpdf[0])
        assert jnp.isneginf(logpdf[1])
        assert jnp.allclose(cdf[:2], jnp.zeros(2))
        assert jnp.isfinite(logpdf[2])

    def test_lognormal_cdf_ppf_round_trip(self):
        u = jnp.array([0.1, 0.5, 0.9])
        x = distributions.lognormal_ppf(u, loc=0.2, scale=0.7)
        assert jnp.all(x > 0.0)
        assert jnp.allclose(distributions.lognormal_cdf(x, loc=0.2, scale=0.7), u)


class TestPowerLaw:
    """Tests for finite-support power-law kernels."""

    def test_powerlaw_logpdf_integrates_to_one(self):
        x = jnp.linspace(1.0, 3.0, 2049)
        density = jnp.exp(
            distributions.powerlaw_logpdf(x, alpha=1.5, xmin=1.0, xmax=3.0)
        )
        assert jnp.allclose(trapz(density, x=x), 1.0, rtol=1e-6)

    def test_powerlaw_cdf_is_monotone_and_round_trips(self):
        u = jnp.array([0.0, 0.2, 0.7, 1.0])
        x = distributions.powerlaw_ppf(u, alpha=-0.5, xmin=2.0, xmax=5.0)
        cdf = distributions.powerlaw_cdf(x, alpha=-0.5, xmin=2.0, xmax=5.0)
        assert jnp.all(jnp.diff(cdf) >= 0.0)
        assert jnp.allclose(cdf, u)

    def test_powerlaw_support_outside_interval(self):
        x = jnp.array([0.5, 1.5, 4.0])
        logpdf = distributions.powerlaw_logpdf(x, alpha=0.0, xmin=1.0, xmax=3.0)
        assert jnp.isneginf(logpdf[0])
        assert jnp.isfinite(logpdf[1])
        assert jnp.isneginf(logpdf[2])


class TestTruncatedNormal:
    """Tests for truncated-normal kernels."""

    def test_truncated_normal_cdf_ppf_round_trip(self):
        u = jnp.array([0.05, 0.5, 0.95])
        x = distributions.truncated_normal_ppf(
            u,
            loc=0.0,
            scale=1.0,
            low=-1.0,
            high=2.0,
        )
        cdf = distributions.truncated_normal_cdf(
            x,
            loc=0.0,
            scale=1.0,
            low=-1.0,
            high=2.0,
        )
        assert jnp.all((x >= -1.0) & (x <= 2.0))
        assert jnp.allclose(cdf, u)

    def test_truncated_normal_logpdf_integrates_to_one(self):
        x = jnp.linspace(-1.0, 2.0, 2049)
        density = jnp.exp(
            distributions.truncated_normal_logpdf(
                x,
                loc=0.0,
                scale=1.0,
                low=-1.0,
                high=2.0,
            )
        )
        assert jnp.allclose(trapz(density, x=x), 1.0, rtol=1e-6)


class TestDistributionTransforms:
    """Tests for transform compatibility."""

    def test_jit_vmap_and_grad(self):
        x = jnp.array([0.2, 0.5, 1.0])
        vmapped = jax.jit(jax.vmap(distributions.normal_logpdf))(x)
        grad = jax.grad(lambda value: distributions.normal_logpdf(value))(
            jnp.array(0.3)
        )
        assert vmapped.shape == x.shape
        assert jnp.isfinite(grad)
