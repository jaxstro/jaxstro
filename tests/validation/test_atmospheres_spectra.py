"""Validation and performance-smoke tests for prepared spectra."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxstro.atmospheres import AtmosphereParams, PreparedSpectralGrid


def test_prepared_spectrum_jit_vmap_performance_smoke():
    n_wave = 4096
    wavelength = jnp.linspace(250.0, 2500.0, n_wave)
    base = jnp.sin(wavelength / 250.0)
    flux = jnp.stack(
        [
            jnp.stack([base + 1.0, base + 2.0]),
            jnp.stack([base + 3.0, base + 4.0]),
        ]
    )
    grid = PreparedSpectralGrid(
        teff=jnp.array([5000.0, 6000.0]),
        logg=jnp.array([4.0, 5.0]),
        wavelength=wavelength,
        flux=flux,
    )

    @jax.jit
    def evaluate(teff_values):
        def one(teff):
            params = AtmosphereParams(teff=teff, logg=4.5)
            return grid.spectrum(params).spectrum.flux_lambda

        return jax.vmap(one)(teff_values)

    spectra = evaluate(jnp.linspace(5000.0, 6000.0, 8))

    assert spectra.shape == (8, n_wave)
    assert bool(jnp.all(jnp.isfinite(spectra)))
