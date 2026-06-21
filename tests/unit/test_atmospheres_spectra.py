"""Tests for the JAX-side atmosphere spectra interface."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.atmospheres import (
    STATUS_MISSING_ABUNDANCE,
    STATUS_OK,
    STATUS_OUT_OF_GRID,
    AtmosphereParams,
    PreparedSpectralGrid,
)


def _prepared_grid() -> PreparedSpectralGrid:
    return PreparedSpectralGrid(
        teff=jnp.array([5000.0, 6000.0]),
        logg=jnp.array([4.0, 5.0]),
        wavelength=jnp.array([100.0, 101.0, 102.0]),
        flux=jnp.array(
            [
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
            ]
        ),
        m_h=0.0,
        alpha_m=0.0,
    )


def test_prepared_grid_interpolates_midpoint_bilinearly():
    grid = _prepared_grid()

    result = grid.spectrum(AtmosphereParams(teff=5500.0, logg=4.5))

    np.testing.assert_allclose(result.spectrum.flux_lambda, [2.5, 3.5, 4.5])
    np.testing.assert_allclose(result.spectrum.wavelength, [100.0, 101.0, 102.0])
    assert int(result.status.code) == STATUS_OK
    assert bool(result.status.in_grid)
    assert not bool(result.status.clamped)
    assert not bool(result.status.unmodeled)


def test_prepared_grid_marks_out_of_grid_without_extrapolating():
    grid = _prepared_grid()

    result = grid.spectrum(AtmosphereParams(teff=4500.0, logg=4.5))

    np.testing.assert_allclose(result.spectrum.flux_lambda, [1.5, 2.5, 3.5])
    assert int(result.status.code) == STATUS_OUT_OF_GRID
    assert not bool(result.status.in_grid)
    assert bool(result.status.clamped)
    assert not bool(result.status.unmodeled)


def test_prepared_grid_marks_wrong_abundance_plane_unmodeled():
    grid = _prepared_grid()

    result = grid.spectrum(AtmosphereParams(teff=5500.0, logg=4.5, m_h=0.5))

    assert int(result.status.code) == STATUS_MISSING_ABUNDANCE
    assert not bool(result.status.in_grid)
    assert bool(result.status.unmodeled)


def test_prepared_grid_is_jittable_and_differentiable():
    grid = _prepared_grid()

    @jax.jit
    def first_flux(teff):
        result = grid.spectrum(AtmosphereParams(teff=teff, logg=4.0))
        return result.spectrum.flux_lambda[0]

    np.testing.assert_allclose(first_flux(5500.0), 2.0)
    np.testing.assert_allclose(jax.grad(first_flux)(5500.0), 0.002)


def test_prepared_grid_supports_vmap_over_params():
    grid = _prepared_grid()

    def evaluate(teff):
        return grid.spectrum(AtmosphereParams(teff=teff, logg=4.5)).spectrum.flux_lambda

    values = jax.vmap(evaluate)(jnp.array([5000.0, 5500.0, 6000.0]))

    assert values.shape == (3, 3)
    np.testing.assert_allclose(values[1], [2.5, 3.5, 4.5])
