"""Real-artifact acceptance gates for the canonical atmosphere spectrum path."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.atmospheres import (
    AtmosphereLibrary,
    AtmosphereParams,
    AtmosphereQuery,
)
from jaxstro.spectra import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectrumStatusCode,
)

pytest.importorskip("polars")
pytest.importorskip("zarr")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


def _query(product_id: str, params: AtmosphereParams) -> AtmosphereQuery:
    family = product_id.split("-", maxsplit=1)[0]
    return AtmosphereQuery(
        params=params,
        product_id=product_id,
        family=family,
        spectral_plan=SpectralPlan(
            SpectralAxis.points(
                jnp.linspace(500.0, 2500.0, 64),
                coordinate=SpectralCoordinate.WAVELENGTH,
                unit="nm",
            )
        ),
        requested_parameter_names=("teff", "logg"),
    )


@pytest.fixture(scope="module")
def library() -> AtmosphereLibrary:
    return AtmosphereLibrary.from_local(DATA)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("product_id", "params"),
    [
        ("newera-v3-lowres", AtmosphereParams(teff=2400.0, logg=3.0)),
        (
            "bosz-2025-recomputed:ap:r10000:resam",
            AtmosphereParams(teff=7750.0, logg=3.5),
        ),
        ("tlusty-ostar2002:z1", AtmosphereParams(teff=30000.0, logg=4.0)),
    ],
)
def test_supported_real_products_return_finite_canonical_spectra(
    library: AtmosphereLibrary,
    product_id: str,
    params: AtmosphereParams,
) -> None:
    result = library.spectrum(_query(product_id, params))

    assert int(result.status.code) == SpectrumStatusCode.OK
    assert result.spectrum.values.shape == (64,)
    assert bool(jnp.all(jnp.isfinite(result.spectrum.values)))
    assert result.spectrum.axis.unit == "nm"
    assert result.spectrum.value_unit == "erg s^-1 cm^-2 nm^-1"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("product_id", "params"),
    [
        (
            "sonora-diamondback-2024:f1:m-0.5:co1",
            AtmosphereParams(teff=1000.0, logg=4.5, m_h=-0.5, c_o=1.0),
        ),
        (
            "tlusty-bstar2006:vturb2:z1",
            AtmosphereParams(teff=16000.0, logg=3.25),
        ),
        (
            "tlusty-bstar2006:vturb10:z1:standard",
            AtmosphereParams(teff=16000.0, logg=2.5, vturb_km_s=10.0),
        ),
    ],
)
def test_unratified_real_products_fail_closed(
    library: AtmosphereLibrary,
    product_id: str,
    params: AtmosphereParams,
) -> None:
    preparation = library.prepare(_query(product_id, params))

    assert preparation.status is SpectrumStatusCode.POLICY_NOT_VALIDATED
    assert preparation.prepared is None


@pytest.mark.slow
def test_prepared_real_spectrum_is_io_free_jittable_vmappable_and_ad_consistent(
    library: AtmosphereLibrary,
) -> None:
    query = _query("newera-v3-lowres", AtmosphereParams(teff=2425.0, logg=3.0))
    preparation = library.prepare(query)
    assert preparation.prepared is not None
    prepared = preparation.prepared

    adapter = library.registry.get(query.product_id)
    object.__setattr__(
        adapter,
        "_store",
        _NoReadStore(),
    )

    def values(point):
        return prepared.stencil.evaluate(point).spectrum.values

    point = jnp.array([2425.0, 3.0])
    eager = values(point)
    compiled = jax.jit(values)(point)
    batched = jax.jit(jax.vmap(values))(jnp.array([[2420.0, 3.0], [2450.0, 3.0]]))

    def objective(teff):
        return jnp.sum(jnp.log(values(jnp.array([teff, 3.0]))))

    ad = jax.grad(objective)(2425.0)
    step = 0.1
    fd = (objective(2425.0 + step) - objective(2425.0 - step)) / (2.0 * step)

    assert eager.shape == compiled.shape == (64,)
    assert batched.shape == (2, 64)
    np.testing.assert_allclose(ad, fd, rtol=2e-4, atol=2e-6)


class _NoReadStore:
    def __getitem__(self, key):
        raise AssertionError(f"filesystem-backed store read after preparation: {key}")
