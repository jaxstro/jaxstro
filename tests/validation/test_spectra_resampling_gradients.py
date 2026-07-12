"""AD-vs-FD audits for smooth spectral point-resampling paths."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

from jaxstro.spectra import (  # noqa: E402
    PointResamplingMethod,
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    resample_spectrum,
)
from jaxstro.testing import Case, audit_entry_point  # noqa: E402

PROVENANCE = SpectrumProvenance(
    source_id="synthetic",
    product_id="resampling-gradient-audit",
    native_coordinate="wavelength_nm",
    native_density="F_lambda",
    native_unit="erg s^-1 cm^-2 nm^-1",
    canonical_conversion="identity",
    citations=("synthetic:test",),
)
BASE_AXIS = jnp.array([100.0, 180.0, 280.0, 400.0])
BASE_VALUES = jnp.array([1.0, 2.0, 4.0, 7.0])
BASE_TARGET = jnp.array([130.0, 230.0, 340.0])


def _resample_total(
    method: PointResamplingMethod,
    *,
    values: object,
    target: object,
) -> object:
    source = Spectrum(
        axis=SpectralAxis.points(
            BASE_AXIS,
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="nm",
        ),
        values=values,
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=PROVENANCE,
    )
    target_axis = SpectralAxis.points(
        target,
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
    result = resample_spectrum(
        source,
        SpectralPlan(target_axis, point_method=method),
    )
    return jnp.sum(result.spectrum.values)


CASES = (
    Case(
        id="spectral-linear-values",
        direction="values->resampled-values",
        fn=lambda scale: _resample_total(
            PointResamplingMethod.LINEAR,
            values=scale * BASE_VALUES,
            target=BASE_TARGET,
        ),
        param="value_scale",
        theta0=1.0,
        tol=2.0e-5,
        h_rel=1.0e-5,
        allowed_claim="linear point resampling inside fixed intervals",
        forbidden_claims=("differentiability through knot selection",),
    ),
    Case(
        id="spectral-linear-query",
        direction="target-axis->resampled-values",
        fn=lambda shift: _resample_total(
            PointResamplingMethod.LINEAR,
            values=BASE_VALUES,
            target=BASE_TARGET + shift,
        ),
        param="target_shift_nm",
        theta0=0.0,
        tol=2.0e-5,
        h_rel=1.0e-3,
        allowed_claim="linear query sensitivity away from knots",
        forbidden_claims=("differentiability at knots",),
    ),
    Case(
        id="spectral-pchip-values",
        direction="values->resampled-values",
        fn=lambda scale: _resample_total(
            PointResamplingMethod.MONOTONE_CUBIC,
            values=scale * BASE_VALUES,
            target=BASE_TARGET,
        ),
        param="value_scale",
        theta0=1.0,
        tol=2.0e-5,
        h_rel=1.0e-5,
        allowed_claim="PCHIP value sensitivity inside a fixed limiter branch",
        forbidden_claims=("differentiability through limiter transitions",),
    ),
    Case(
        id="spectral-pchip-query",
        direction="target-axis->resampled-values",
        fn=lambda shift: _resample_total(
            PointResamplingMethod.MONOTONE_CUBIC,
            values=BASE_VALUES,
            target=BASE_TARGET + shift,
        ),
        param="target_shift_nm",
        theta0=0.0,
        tol=2.0e-5,
        h_rel=1.0e-3,
        allowed_claim="PCHIP query sensitivity inside fixed intervals",
        forbidden_claims=("differentiability at knots",),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_spectral_resampling_ad_matches_finite_difference(case: Case) -> None:
    result = audit_entry_point(case)

    assert result.finite
    assert result.status == "clean"
    assert abs(result.ratio - 1.0) < result.tol
    assert result.grad_contract == "smooth_pathwise"
