"""AD-vs-FD audits for prepared spectral stencil evaluation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

from jaxstro.spectra import (  # noqa: E402
    FluxInterpolation,
    PreparedRectilinearStencil,
    PreparedSimplexStencil,
    SpectralAxis,
    SpectralCoordinate,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
)
from jaxstro.testing import Case, audit_entry_point  # noqa: E402

PROVENANCE = SpectrumProvenance(
    source_id="synthetic",
    product_id="stencil-gradient-audit",
    native_coordinate="wavelength_nm",
    native_density="F_lambda",
    native_unit="erg s^-1 cm^-2 nm^-1",
    canonical_conversion="identity",
    citations=("synthetic:test",),
)
TEMPLATE = Spectrum(
    axis=SpectralAxis.points(
        jnp.array([100.0, 200.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    ),
    values=jnp.ones(2),
    semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
    provenance=PROVENANCE,
)
RECTILINEAR = PreparedRectilinearStencil(
    parameter_axes=(jnp.array([1.0, 3.0]), jnp.array([10.0, 20.0, 50.0])),
    vertex_values=jnp.array(
        [
            [[7.0, 14.0], [12.0, 24.0], [27.0, 54.0]],
            [[11.0, 22.0], [16.0, 32.0], [31.0, 62.0]],
        ]
    ),
    template=TEMPLATE,
)
RECTILINEAR_LOG = PreparedRectilinearStencil(
    parameter_axes=(jnp.array([1.0, 3.0]),),
    vertex_values=jnp.array([[1.0, 4.0], [9.0, 36.0]]),
    template=TEMPLATE,
    interpolation=FluxInterpolation.POSITIVE_LOG,
)
SIMPLEX = PreparedSimplexStencil(
    vertices=jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]]),
    vertex_values=jnp.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]),
    template=TEMPLATE,
)

CASES = (
    Case(
        id="spectral-rectilinear-point",
        direction="parameters->spectrum",
        fn=lambda shift: (
            RECTILINEAR.evaluate(jnp.array([2.0 + shift, 35.0])).spectrum.values
        ),
        param="first_parameter_shift",
        theta0=0.0,
        tol=2.0e-5,
        h_rel=1.0e-4,
        allowed_claim="multilinear sensitivity inside a fixed complete cell",
        forbidden_claims=("differentiability through cell selection",),
    ),
    Case(
        id="spectral-rectilinear-positive-log-point",
        direction="parameters->spectrum",
        fn=lambda shift: (
            RECTILINEAR_LOG.evaluate(jnp.array([2.0 + shift])).spectrum.values
        ),
        param="parameter_shift",
        theta0=0.0,
        tol=2.0e-5,
        h_rel=1.0e-4,
        allowed_claim="positive-log sensitivity inside a fixed complete cell",
        forbidden_claims=("zero or negative vertex flux",),
    ),
    Case(
        id="spectral-simplex-point",
        direction="parameters->spectrum",
        fn=lambda shift: (
            SIMPLEX.evaluate(jnp.array([0.5 + shift, 1.0])).spectrum.values
        ),
        param="first_parameter_shift",
        theta0=0.0,
        tol=2.0e-5,
        h_rel=1.0e-4,
        allowed_claim="barycentric sensitivity inside a fixed simplex",
        forbidden_claims=("differentiability through simplex selection",),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_spectral_stencil_ad_matches_finite_difference(case: Case) -> None:
    result = audit_entry_point(case)

    assert result.finite
    assert result.status == "clean"
    assert abs(result.ratio - 1.0) < result.tol
    assert result.grad_contract == "smooth_pathwise"
