# tests/test_sampling.py
"""
Tests for jaxstro.numerics.sampling.

Covers:
- byte-for-byte parity with progenax's local inverse_cdf_draw (the origin),
  including the zero-total-weight reg-guard case;
- behavioral properties (monotonicity, bounds, known analytic CDF);
- FD-vs-AD grad-checks w.r.t. weight and unif.
"""

import jax
import jax.numpy as jnp

from jaxstro.numerics import inverse_cdf_draw, sampling
from jaxstro.numerics.integration import cumulative_trapz


def _progenax_inverse_cdf_draw(weight, grid, unif, reg=1e-30):
    """Local copy of progenax's reference implementation (the origin).

    Verbatim semantics from
    ``progenax/src/progenax/numerics.py::inverse_cdf_draw`` so the parity test
    compares against the exact origin without importing progenax.
    """
    dx = grid[1] - grid[0]
    cdf = cumulative_trapz(weight, dx=dx)
    cdf = cdf / (cdf[-1] + reg)
    return jnp.interp(unif, cdf, grid)


# A handful of representative inputs: smooth, uniform, peaked-looking, and the
# degenerate zero-total-weight case that exercises the +reg guard.
GRID = jnp.linspace(0.0, 1.0, 65)
WEIGHTS = {
    "uniform": jnp.ones_like(GRID),
    "linear_ramp": GRID,
    "gaussian_bump": jnp.exp(-0.5 * ((GRID - 0.5) / 0.1) ** 2),
    "nonuniform_spiky": jnp.array(
        [3.0 if i % 7 == 0 else 0.2 for i in range(GRID.shape[0])]
    ),
    "zero_total": jnp.zeros_like(GRID),
}
UNIFS = [0.0, 0.1, 0.37, 0.5, 0.83, 1.0]


class TestParity:
    """jaxstro must reproduce the progenax origin byte-for-byte."""

    def test_byte_for_byte_all_cases(self):
        for name, w in WEIGHTS.items():
            for u in UNIFS:
                u = jnp.asarray(u)
                got = inverse_cdf_draw(w, GRID, u)
                ref = _progenax_inverse_cdf_draw(w, GRID, u)
                assert jnp.array_equal(got, ref), f"parity mismatch: {name}, u={u}"

    def test_zero_weight_reg_guard_finite(self):
        """Zero total weight must yield a finite draw (not NaN) via +reg."""
        for u in UNIFS:
            draw = inverse_cdf_draw(WEIGHTS["zero_total"], GRID, jnp.asarray(u))
            # Core reg-guard contract: finite (not NaN) instead of 0/0.
            assert jnp.isfinite(draw), f"zero-weight draw not finite at u={u}"
            # Draw stays within grid bounds (jnp.interp clamps against the
            # all-zero CDF). Exact value is byte-checked against the origin in
            # test_byte_for_byte_all_cases.
            assert GRID.min() <= draw <= GRID.max()


class TestBehavior:
    """Behavioral / analytic checks."""

    def test_monotonic_in_unif(self):
        w = WEIGHTS["gaussian_bump"]
        us = jnp.linspace(0.0, 1.0, 50)
        draws = jax.vmap(lambda u: inverse_cdf_draw(w, GRID, u))(us)
        diffs = jnp.diff(draws)
        assert jnp.all(diffs >= -1e-12), "draws not monotonic non-decreasing in unif"

    def test_draws_within_grid_bounds(self):
        for name, w in WEIGHTS.items():
            for u in UNIFS:
                draw = inverse_cdf_draw(w, GRID, jnp.asarray(u))
                assert draw >= GRID.min() - 1e-12
                assert draw <= GRID.max() + 1e-12

    def test_uniform_weight_linear_cdf(self):
        """Uniform weight -> linear CDF -> draw ~= grid[0] + unif*range."""
        w = WEIGHTS["uniform"]
        lo, hi = GRID[0], GRID[-1]
        for u in [0.1, 0.25, 0.5, 0.75, 0.9]:
            draw = inverse_cdf_draw(w, GRID, jnp.asarray(u))
            expected = lo + u * (hi - lo)
            # CDF from trapezoid of a constant is exactly linear in index; on a
            # uniform grid this matches lo + u*range to interpolation precision.
            assert jnp.abs(draw - expected) < 1e-3, f"u={u}: {draw} vs {expected}"


class TestGradients:
    """FD-vs-AD grad-checks at normal (nonzero-weight) inputs."""

    def test_grad_wrt_unif(self):
        w = WEIGHTS["gaussian_bump"]
        u0 = jnp.asarray(0.42)
        ad = jax.grad(lambda u: inverse_cdf_draw(w, GRID, u))(u0)
        eps = 1e-6
        fd = (
            inverse_cdf_draw(w, GRID, u0 + eps) - inverse_cdf_draw(w, GRID, u0 - eps)
        ) / (2 * eps)
        assert jnp.isfinite(ad)
        assert jnp.abs(ad - fd) <= 1e-6 * (jnp.abs(fd) + 1.0)

    def test_grad_wrt_weight(self):
        w0 = WEIGHTS["gaussian_bump"]
        u0 = jnp.asarray(0.42)

        def f(w):
            return inverse_cdf_draw(w, GRID, u0)

        ad = jax.grad(f)(w0)
        assert jnp.all(jnp.isfinite(ad)), "weight gradient not finite"

        # Finite-difference a few representative components.
        eps = 1e-6
        for i in [10, 25, 32, 40, 55]:
            pert = jnp.zeros_like(w0).at[i].set(eps)
            fd = (f(w0 + pert) - f(w0 - pert)) / (2 * eps)
            rel = jnp.abs(ad[i] - fd) / (jnp.abs(fd) + 1.0)
            assert rel <= 1e-5, f"component {i}: AD={ad[i]} FD={fd} rel={rel}"

    def test_reg_guard_does_not_poison_normal_gradients(self):
        """At nonzero total weight the +reg guard is negligible (1e-30)."""
        w0 = WEIGHTS["linear_ramp"] + 0.5  # strictly positive, large total
        u0 = jnp.asarray(0.6)
        ad = jax.grad(lambda w: inverse_cdf_draw(w, GRID, u0))(w0)
        assert jnp.all(jnp.isfinite(ad))
        assert jnp.any(jnp.abs(ad) > 0)


class TestStratifiedUniform:
    """Tests for deterministic-shape stratified uniform samples."""

    def test_one_sample_per_stratum(self):
        key = jax.random.key(0)
        samples = sampling.stratified_uniform(key, 8)
        assert samples.shape == (8,)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 1.0)
        strata = jnp.floor(samples * 8).astype(int)
        assert jnp.array_equal(jnp.sort(strata), jnp.arange(8))

    def test_custom_bounds(self):
        key = jax.random.key(1)
        samples = sampling.stratified_uniform(key, 4, minval=-2.0, maxval=2.0)
        assert samples.shape == (4,)
        assert jnp.all(samples >= -2.0)
        assert jnp.all(samples < 2.0)

    def test_jit_compatible_with_static_count(self):
        @jax.jit
        def draw(key):
            return sampling.stratified_uniform(key, 5)

        samples = draw(jax.random.key(2))
        assert samples.shape == (5,)

    def test_rejects_invalid_count(self):
        key = jax.random.key(3)
        try:
            sampling.stratified_uniform(key, 0)
        except ValueError as exc:
            assert "n" in str(exc)
        else:
            raise AssertionError("expected ValueError")
